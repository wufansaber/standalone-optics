"""Requirements for fixed-image-plane material search.

spot: worst field's polychromatic RMS spot radius about the common color
centroid, in micrometers. wavefront: worst field/wavelength's RMS wavefront
error, in waves at that wavelength. Inputs come from the same frozen design
conditions and appropriate validation level. Both metrics are minimized.
"""
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


UNITS = {'spot': 'µm', 'wavefront': 'waves'}


def _metric(name):
    if isinstance(name, str):
        name = name.strip().lower()
        if name in UNITS:
            return name
    raise ValueError(f'Expected spot or wavefront, got {name!r}.')


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real number.')
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f'{name} exceeds floating-point range.') from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        sign = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {sign}.')
    return value


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'Duplicate JSON key {name!r}.')
        result[name] = value
    return result


class DesignRequirements:
    def __init__(self, priority=None, limits=None):
        if limits is None:
            limits = {}
        if not isinstance(limits, Mapping):
            raise ValueError('limits must be an object keyed by metric.')
        self._limits = {}
        for name, maximum in limits.items():
            key = _metric(name)
            if key in self._limits:
                raise ValueError(f'Duplicate upper limit for {key}.')
            self._limits[key] = _number(maximum, f'{key} upper limit', positive=True)
        if priority is None:
            if len(self._limits) != 1:
                raise ValueError('Specify priority when there are zero or multiple upper limits.')
            priority = next(iter(self._limits))
        self._priority = _metric(priority)

    @property
    def priority(self):
        return self._priority

    @property
    def limits(self):
        return self._limits.copy()

    @property
    def metrics(self):
        return (self.priority, *(name for name in self._limits if name != self.priority))

    def as_dict(self):
        return {'priority': self.priority, 'limits': self.limits}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise ValueError('Design requirements must be an object.')
        unknown = [name for name in data if name not in ('priority', 'limits')]
        if unknown:
            raise ValueError(f'Unknown requirement fields {unknown!r}.')
        if 'limits' in data and not isinstance(data['limits'], Mapping):
            raise ValueError('limits must be an object keyed by metric.')
        return cls(data.get('priority'), data.get('limits'))

    @classmethod
    def from_json(cls, path):
        contents = Path(path).read_text(encoding='utf-8-sig')
        return cls.from_dict(json.loads(contents, object_pairs_hook=_unique_object))

    def _measurements(self, values):
        if not isinstance(values, Mapping):
            raise ValueError('Expected a mapping of metric names to measured values.')
        measured = {}
        for name in self.metrics:
            if name not in values:
                raise ValueError(f'Missing required measurement {name}.')
            measured[name] = _number(values[name], name)
        return measured

    def _violations(self, measured):
        violations = {name: max(0., measured[name] - maximum) / maximum for name, maximum in self._limits.items()}
        if not all(math.isfinite(ratio) for ratio in violations.values()):
            raise ValueError('Relative upper-limit violation exceeds floating-point range.')
        return violations

    def rank(self, values):
        """Minimize this tuple. A pure ranking or any feasible result has first key 0."""
        measured = self._measurements(values)
        greatest = max(self._violations(measured).values(), default=0.)
        return (int(greatest > 0.), greatest, measured[self.priority])

    def assess(self, values):
        measured = self._measurements(values)
        violations = self._violations(measured)
        greatest = max(violations.values(), default=0.)
        met = greatest == 0. if self._limits else None
        details = {}
        for name, actual in measured.items():
            maximum = self._limits.get(name)
            details[name] = {'actual': actual, 'unit': UNITS[name], 'limit': maximum,
                             'actualMinusLimit': actual - maximum if maximum is not None else None,
                             'relativeViolation': violations.get(name),
                             'withinLimit': actual <= maximum if maximum is not None else None}
        status = 'ranked' if met is None else 'target_met' if met else 'out_of_spec'
        return {'status': status, 'targetMet': met, 'priority': self.priority, 'metrics': details,
                'maxRelativeViolation': greatest if self._limits else None}

    def screening_rank(self, row):
        """Return only an ordering; screening evidence never confirms a target."""
        if not isinstance(row, Mapping):
            raise ValueError('A requirements-aware callback must return a metrics mapping.')
        return self.rank(row.get('metrics'))

    def validated(self, row):
        """Keep independent physical gates as an absolute veto before optical ranking."""
        if not row['passed']:
            return {**row, 'physicalAcceptance': False, 'assessment': {'status': 'rejected', 'targetMet': None}}
        assessment = self.assess(row.get('metrics'))
        return {**row, 'physicalAcceptance': True, 'assessment': assessment,
                'passed': assessment['targetMet'] is not False}

    def select(self, candidates):
        """Candidates are (identity, independently validated assessment row) pairs."""
        valid = [(key, row) for key, row in candidates if row.get('physicalAcceptance')]
        if not valid:
            return {'candidate': None, 'status': 'unverified', 'targetMet': None}
        key, row = min(valid, key=lambda item: (self.rank(item[1]['metrics']), item[0]))
        return {'candidate': key, **row['assessment']}
