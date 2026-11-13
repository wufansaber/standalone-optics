"""Context-aware MF dependency compaction, explicit working sets and budgets.

The whitelist describes the row-reference semantics of supported operands.
Unknown state-changing operands fail closed. No algebraic reassociation, limit
change, weighted-row merging, or deletion of referenced zero-weight rows occurs.
"""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path


CONTROL = {'CONF', 'PRIM'}
BINARY = {'SUMM', 'DIFF', 'PROD', 'DIVI'}
UNARY = {'SQRT', 'PROB', 'OPGT', 'OPLT'}
MEASURE = {'CVVA', 'TTHI', 'INDX', 'CARD', 'PARY', 'REAX', 'REAY',
           'REAA', 'REAB', 'REAC', 'RWCE', 'RWRE', 'MTFT', 'MTFS'}
SUPPORTED = CONTROL | BINARY | UNARY | MEASURE | {'CONS'}
LIMITS = {'OPGT', 'OPLT'}


def normalize(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('MF graph must be a nonempty ordered row list')
    result = []
    for index, source in enumerate(rows, 1):
        row = deepcopy(source)
        op = str(row['operand']).upper()
        if op not in SUPPORTED:
            raise ValueError(f'Unsupported operand at row {index}: {op}; declare its dependencies/state before compiling')
        params = list(map(float, row.get('params', [])))
        target, weight = float(row.get('target', 0.)), float(row.get('weight', 0.))
        if len(params) > 8 or not all(map(math.isfinite, params+[target, weight])) or weight < 0.:
            raise ValueError('Use finite parameters/targets and nonnegative weights; Lagrangian weights require a separate model')
        params += [0.]*(8-len(params))
        positions = (0, 1) if op in BINARY else (0,) if op in UNARY else ()
        for position in positions:
            number = params[position]
            if number != int(number) or not 1 <= number < index:
                raise ValueError(f'Row {index} requires a strictly earlier positive integer row reference')
        if op == 'PROB' and params[1] != 1.:
            raise ValueError('This PROB schema requires Int2=1 and a literal factor in Data1')
        if op in CONTROL and (weight != 0. or params[0] < 1. or params[0] != int(params[0])):
            raise ValueError('Control rows require a positive configuration/wavelength and zero weight')
        row.update(operand=op, params=params, target=target, weight=weight)
        result.append(row)
    return result


def dependencies(row):
    op = row['operand']
    count = 2 if op in BINARY else 1 if op in UNARY else 0
    return [int(v) for v in row['params'][:count]]


def audit(rows, budget=None, *, evaluation_seconds=None, jacobian_seconds=None):
    rows = normalize(rows)
    counts = Counter(r['operand'] for r in rows)
    values = dict(rows=len(rows), weighted_rows=sum(r['weight'] > 0. for r in rows),
        zero_weight_rows=sum(r['weight'] == 0. for r in rows),
        ray_queries=sum(counts[op] for op in ('PARY', 'REAX', 'REAY', 'REAA', 'REAB', 'REAC')),
        wavefront_queries=sum(counts[op] for op in ('RWCE', 'RWRE')),
        mtf_queries=sum(counts[op] for op in ('MTFT', 'MTFS')),
        control_rows=sum(counts[op] for op in CONTROL), arithmetic_rows=sum(counts[op] for op in BINARY | {'SQRT', 'PROB', 'CONS'}))
    budget = dict(budget or {})
    unknown = set(budget)-set(values)
    if unknown or any(type(v) is not int or v < 0 for v in budget.values()):
        raise ValueError('Budget keys must be reported count names with nonnegative integer limits')
    exceeded = {key: dict(actual=values[key], maximum=limit) for key, limit in budget.items() if values[key] > limit}
    for label, seconds in [('evaluation_seconds', evaluation_seconds), ('jacobian_seconds', jacobian_seconds)]:
        if seconds is not None:
            if not math.isfinite(seconds) or seconds < 0.:
                raise ValueError('Measured timings must be finite and nonnegative')
            values[label] = float(seconds)
    return dict(counts=values, operands=dict(counts), budget=budget, budget_passed=not exceeded,
        exceeded=exceeded, timing_scope='Counts describe MF structure, not execution-time estimates; timings are caller measurements')


def compact(rows, *, active_rows=None, protected_rows=(), audit_identity=None,
            keep_rows=(), budget=None):
    """Compile an equivalent graph, or an explicitly selected working graph.

Default mode maps EVERY source row (including diagnostics) to the output and
preserves every weighted row separately. Working mode keeps the transitive
closure of active/protected/explicit rows plus all controls; other weighted
rows are deactivated, never silently called accepted. It requires an identity
for the external full-domain audit. Keep protected physical gates explicitly.
    """
    source = normalize(rows)
    n = len(source)

    def indices(values):
        values = list(values)
        if any(type(i) is not int or not 1 <= i <= n for i in values):
            raise ValueError('Row selections must contain valid one-based integer row numbers')
        return set(values)

    protected, explicit = indices(protected_rows), indices(keep_rows)
    weighted = {i for i, row in enumerate(source, 1) if row['weight'] > 0.}
    working = active_rows is not None
    if working:
        if not isinstance(audit_identity, str) or not audit_identity.strip():
            raise ValueError('A working MF requires a nonempty external full-audit identity')
        active = indices(active_rows)
        if not (active | protected) <= weighted:
            raise ValueError('Active and protected roots must be weighted source constraints')
        retained_weighted = active | protected
        if not retained_weighted:
            raise ValueError('A working MF needs at least one active/protected weighted constraint')
        live = retained_weighted | explicit | {i for i, row in enumerate(source, 1) if row['operand'] in CONTROL}
        pending = list(live)
        while pending:
            for dep in dependencies(source[pending.pop()-1]):
                if dep not in live:
                    live.add(dep); pending.append(dep)
    else:
        live, retained_weighted = set(range(1, n+1)), weighted
    output, mapping, cache = [], [None]*(n+1), {}
    context = (None, None)
    state_epoch = 0
    for old, original in enumerate(source, 1):
        if old not in live:
            continue
        row = deepcopy(original)
        op = row['operand']
        if op == 'CONF':
            state_epoch += 1
            context = (tuple(row['params']), None)
        elif op == 'PRIM':
            state_epoch += 1
            context = (context[0], tuple(row['params']))
        for position, dep in enumerate(dependencies(row)):
            assert mapping[dep] is not None
            row['params'][position] = float(mapping[dep])
        if old not in retained_weighted:
            row['weight'] = 0.
        # Pure arithmetic refers to already stored operand values, independent
        # of the currently selected optical configuration. Measurements do not.
        # Unknown primary/configuration state is NOT a shared state identity.
        # Different CONF entries without an explicit PRIM may inherit different
        # native primary wavelengths, so do not merge across those entries.
        state = (context, state_epoch if None in context else None) if op in MEASURE else None
        key = (op, tuple(row['params']), row['target'], state)
        reusable = op not in CONTROL | LIMITS
        if reusable and row['weight'] == 0. and key in cache:
            mapping[old] = cache[key]
            continue
        output.append(row)
        mapping[old] = len(output)
        if reusable:
            cache.setdefault(key, len(output))
    normalize(output)
    report = dict(mode='working_subset' if working else 'equivalent_all_rows',
        input=audit(source), output=audit(output, budget),
        merged_rows=len(live)-len(output), omitted_rows=n-len(live),
        source_weighted_rows=len(weighted), retained_weighted_rows=len(retained_weighted),
        deactivated_weighted_rows=sorted(weighted-retained_weighted),
        protected_rows=sorted(protected), external_audit_identity=audit_identity,
        full_objective_preserved=not working or retained_weighted == weighted,
        final_acceptance=False, requires_external_full_audit=working,
        all_source_rows_mapped=not working, row_map_1based=mapping)
    # Verify weighted roots remain distinct even when expressions are identical.
    assert len({mapping[i] for i in retained_weighted}) == len(retained_weighted)
    return output, report


def write_native(path, rows, *, header='VERS 221026'):
    rows = normalize(rows)
    if not header.startswith('VERS ') or '\n' in header or '\r' in header:
        raise ValueError('Native MF header must be one VERS line')
    with Path(path).open('w', encoding='utf-16', newline='\n') as stream:
        stream.write(header+'\n')
        for row in rows:
            values = [*row['params'], row['target'], row['weight'], 0.]
            stream.write(row['operand']+' '+' '.join(format(v, '.17g') for v in values)+'\n')


def evaluate(rows, measurement):
    """Reference interpreter for the supported graph, for adapter/native controls.

measurement(row, (configuration_params, primary_params)) supplies physical values;
this function does not trace optics. Domain errors invalidate the whole graph.
    """
    source = normalize(rows)
    values, context = [None], (None, None)
    for row in source:
        op, p = row['operand'], row['params']
        if op == 'CONF':
            context = (tuple(p), None); value = 0.
        elif op == 'PRIM':
            context = (context[0], tuple(p)); value = 0.
        elif op == 'CONS':
            value = row['target']
        elif op in BINARY:
            a, b = [values[int(i)] for i in p[:2]]
            if op == 'SUMM': value = a+b
            elif op == 'DIFF': value = a-b
            elif op == 'PROD': value = a*b
            else: value = a/b
        elif op in UNARY:
            a = values[int(p[0])]
            if op == 'SQRT': value = math.sqrt(a)
            elif op == 'PROB': value = a*p[2]
            elif op == 'OPGT': value = min(a, row['target'])
            else: value = max(a, row['target'])
        else:
            value = float(measurement(row, context))
        if not math.isfinite(value):
            raise ValueError('Nonfinite operand value invalidates the complete MF')
        values.append(value)
    return values


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True, help='JSON ordered row graph')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--working-selection', type=Path, help='Explicit active/protected roots and audit_identity JSON')
    parser.add_argument('--max-rows', type=int)
    parser.add_argument('--max-ray-queries', type=int)
    parser.add_argument('--max-wavefront-queries', type=int)
    parser.add_argument('--max-mtf-queries', type=int)
    args = parser.parse_args(argv)
    selection = {} if args.working_selection is None else json.loads(args.working_selection.read_text(encoding='utf8'))
    budget = {name: value for name, value in [('rows', args.max_rows), ('ray_queries', args.max_ray_queries),
              ('wavefront_queries', args.max_wavefront_queries), ('mtf_queries', args.max_mtf_queries)] if value is not None}
    rows, report = compact(json.loads(args.graph.read_text(encoding='utf8')), budget=budget, **selection)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'rows.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n', encoding='utf8')
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf8')
    if not report['output']['budget_passed']:
        raise SystemExit('MF exceeds the requested budget; report written, native MF not emitted; no constraints silently removed')
    write_native(args.out/'COMPACT.MF', rows)
    print(json.dumps({key: report[key] for key in ('mode', 'merged_rows', 'omitted_rows', 'full_objective_preserved')}))


if __name__ == '__main__':
    main()
