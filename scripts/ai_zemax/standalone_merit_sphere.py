"""Prove redundancy of sampled spherical MF bounds, without a nominal-point fit.

This adapter accepts only the explicit sag DAG emitted by the green8 exporter.
It reduces sampled inequalities, not their off-feasible weighted penalty. Face
radii must stay fixed. Independent full-body/ray audits remain mandatory.
"""
from collections import defaultdict
import math

from .standalone_merit_graph import normalize


def spherical_support(rows, geometry, domain, profiles):
    """Return protected roots implying the source sampled geometry constraints.

On each fixed shoulder interval, s_b'(r)-s_a'(r) has constant sign:
c/sqrt(1-c*c*r*r) increases with c. Either sphere can become a plane at
its fixed face edge. Retaining both sampled endpoints of each activity
interval is therefore sufficient for every original sampled inequality.
Unmatched-radius/rim constraints and curvature bounds are kept verbatim.
"""
    rows = normalize(rows)
    faces = {}
    for profile in profiles:
        surface = profile['surface']
        radius = float(profile['sphere_face_radius_mm'])
        if surface in faces or not math.isfinite(radius) or radius <= 0.:
            raise ValueError('Require unique surfaces and finite positive fixed face radii')
        if profile['outer_annulus'] != 'plane at sphere-face-edge sag':
            raise ValueError('Only spherical faces with fixed plane shoulders are supported')
        faces[surface] = radius

    used_domains = defaultdict(float)

    def get(index, op):
        if type(index) is not int or not 1 <= index <= len(rows) or rows[index-1]['operand'] != op:
            raise ValueError('Spherical proof does not match the supplied MF dependency graph')
        row = rows[index-1]
        used = 3 if op == 'PROB' else 2 if op in {'SUMM', 'DIFF', 'PROD', 'DIVI'} else 1 if op == 'SQRT' else 0
        if op != 'CVVA' and any(row['params'][used:]):
            raise ValueError('Unexpected additional parameters in spherical arithmetic')
        return row

    def ref(index, op):
        return [int(x) for x in get(index, op)['params'][:2]]

    def one(index):
        if get(index, 'CONS')['target'] != 1.:
            raise ValueError('Spherical sag requires literal one')

    def curvature(index, surface):
        if get(index, 'CVVA')['params'] != [float(surface), 1., 0., 0., 0., 0., 0., 0.]:
            raise ValueError('Unexpected curvature query')

    def factor(index):
        p = get(index, 'PROB')['params']
        return int(p[0]), p[2]

    def root(index, surface, radius):
        const, square = ref(index, 'DIFF'); one(const)
        product, r2 = factor(square)
        ca, cb = ref(product, 'PROD')
        curvature(ca, surface)
        if ca != cb or r2 != radius*radius:
            raise ValueError('Unexpected spherical root domain')
        return ca

    def sag(index, surface, radius):
        numerator, denominator = ref(index, 'DIVI')
        ca, r2 = factor(numerator)
        const, sq = ref(denominator, 'SUMM'); one(const)
        domain_row = int(get(sq, 'SQRT')['params'][0])
        cb = root(domain_row, surface, radius)
        if ca != cb or r2 != radius*radius:
            raise ValueError('Unexpected spherical sag numerator')
        used_domains[ca] = max(used_domains[ca], radius)
        return ca

    def bounds(record):
        expected = [('OPGT', record['lower'])] if record.get('lower') is not None else []
        if record.get('upper') is not None:
            expected.append(('OPLT', record['upper']))
        actual = []
        for index in record['bound_rows']:
            r = rows[index-1]
            if r['operand'] not in {'OPGT', 'OPLT'} or int(r['params'][0]) != record['value_row'] or r['weight'] <= 0.:
                raise ValueError('Unexpected geometry bound reference/weight')
            actual.append((r['operand'], r['target']))
        if actual != expected or not actual:
            raise ValueError('Geometry metadata does not match native bound targets')
        return tuple(actual)

    families, domain_families = defaultdict(list), defaultdict(list)
    retained, all_roots, certificates = set(), set(), []
    for rec in geometry:
        signature = bounds(rec)
        all_roots.update(rec['bound_rows'])
        a, b = rec['first_surface'], rec['last_surface']
        ra, rb = float(rec['radius_first_mm']), float(rec['radius_last_mm'])
        sa, sb = float(rec['sag_radius_first_mm']), float(rec['sag_radius_last_mm'])
        if not all(math.isfinite(r) and r >= 0. for r in (ra, rb, sa, sb)):
            raise ValueError('Invalid geometry radii')
        if sa != min(ra, faces[a]) or sb != min(rb, faces[b]):
            raise ValueError('Fixed face clipping does not match the geometry metadata')
        distance, diff = ref(rec['value_row'], 'SUMM')
        rear, front = ref(diff, 'DIFF')
        cb, ca = sag(rear, b, sb), sag(front, a, sa)
        # Distinct radial rim combinations are kept: no corner availability
        # assumption and no inference from a minimum at the current design.
        if ra != rb:
            retained.update(rec['bound_rows']); continue
        key = (distance, ca, cb, a, b, rec['kind'], signature,
               ra <= faces[a], rb <= faces[b])
        families[key].append(rec)
    for records in families.values():
        records.sort(key=lambda r: r['radius_first_mm'])
        endpoints = [records[0], records[-1]]
        roots = sorted({i for r in endpoints for i in r['bound_rows']})
        retained.update(roots)
        certificates.append(dict(kind='monotone_sphere_interval',
            source_roots=[i for r in records for i in r['bound_rows']], retained_roots=roots))

    for rec in domain:
        signature = bounds(rec); all_roots.update(rec['bound_rows'])
        if 'evaluated_radius_mm' not in rec:
            # Explicit curvature limits are part of the protected domain.
            curvature(rec['value_row'], rec['surface'])
            retained.update(rec['bound_rows']); continue
        radius = float(rec['evaluated_radius_mm'])
        if not math.isfinite(radius) or radius < 0. or rec.get('upper') is not None or not 0. < rec['lower'] < 1.:
            raise ValueError('Require a positive spherical root-domain lower bound')
        ca = root(rec['value_row'], rec['surface'], radius)
        domain_families[(ca, signature)].append(rec)
    for records in domain_families.values():
        strongest = max(records, key=lambda r: r['evaluated_radius_mm'])
        retained.update(strongest['bound_rows'])
        certificates.append(dict(kind='largest_radius_root_domain',
            source_roots=[i for r in records for i in r['bound_rows']],
            retained_roots=strongest['bound_rows']))
    for ca, radius in used_domains.items():
        if not any(key[0] == ca and max(r['evaluated_radius_mm'] for r in records) >= radius
                   for key, records in domain_families.items()):
            raise ValueError('Every spherical sag needs a protected positive root-domain bound')
    return dict(protected_rows=sorted(retained), source_rows=sorted(all_roots),
        removed_rows=sorted(all_roots-retained), certificates=certificates,
        sampled_geometry_feasible_set_preserved=True, weighted_penalty_preserved=False,
        assumptions=['explicit matched spherical sag DAG', 'fixed face radii and plane shoulders',
                     'all retained root-domain and geometry bounds enforced',
                     'independent complete-body and ray-validity audit unchanged'],
        continuous_body_certified=False)
