"""Search finite-conjugate starting structures and screen candidates on a CPU pool."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from time import perf_counter

from .standalone_candidate_generation import curvature_candidates
from .standalone_cli import write_json
from .standalone_model import load_zmx
from .standalone_parallel_screen import CandidateScreen, MAX_CANDIDATES_PER_BATCH


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_model(path, glass_dir):
    if path.suffix.lower() == '.zmx':
        if glass_dir is None:
            raise ValueError('Provide --glass-dir for a text ZMX')
        return load_zmx(path, glass_dir)
    if path.suffix.lower() == '.json':
        return read_json(path)
    raise ValueError('Provide a text ZMX or normalized model JSON')


def next_round(variables, center, size, number, random_seed):
    origin = dict(center['curvature_deltas'])
    narrowed = []
    for variable in variables:
        surface = variable['surface']
        lower, upper = variable['min_delta_mm_inv'], variable['max_delta_mm_inv']
        value = origin.get(surface, 0.)
        radius = (upper-lower)/(2**number)
        narrowed.append({'surface': surface, 'min_delta_mm_inv': max(-radius, lower-value),
                         'max_delta_mm_inv': min(radius, upper-value)})
    steps = curvature_candidates(narrowed, size, random_seed=random_seed+number)
    candidates = []
    for step in steps[1:]:
        relative = dict(step['curvature_deltas'])
        delta = [[v['surface'], origin.get(v['surface'], 0.)+relative.get(v['surface'], 0.)]
                 for v in variables]
        for pair, variable in zip(delta, variables):
            pair[1] = min(variable['max_delta_mm_inv'], max(variable['min_delta_mm_inv'], pair[1]))
        candidates.append({'id': f"round_{number:02d}_{step['id']}",
                           'curvature_deltas': [pair for pair in delta if pair[1]]})
    return candidates


def rank_rows(rows):
    return sorted((row for row in rows if row['status'] == 'calculated'),
                  key=lambda row: (row['worst_wfe_waves'], row['mean_wfe_waves'],
                                   row['worst_spot_um'], row['input_index']))


def screen_rounds(screen, initial, variables, rounds, random_seed):
    candidates = list(initial)
    first = screen.screen(initial)
    if rounds == 1:
        return first, candidates
    by_id = {item['id']: item for item in candidates}
    seen = {tuple(map(tuple, item['curvature_deltas'])) for item in candidates}
    rows = list(first['results'])
    best = first['ranked'][0] if first['ranked'] else None
    history = [{'round': 1, 'new_candidates': len(initial), 'best_id': best}]
    for number in range(2, rounds+1):
        if best is None:
            break
        fresh = []
        for item in next_round(variables, by_id[best], len(initial), number, random_seed):
            key = tuple(map(tuple, item['curvature_deltas']))
            if key not in seen:
                seen.add(key)
                fresh.append(item)
        if not fresh:
            break
        current = screen.screen(fresh)
        for row in current['results']:
            row['input_index'] += len(candidates)
        candidates.extend(fresh)
        by_id.update((item['id'], item) for item in fresh)
        rows.extend(current['results'])
        ranked = rank_rows(rows)
        best = ranked[0]['id']
        history.append({'round': number, 'new_candidates': len(fresh), 'best_id': best})
    ranked = rank_rows(rows)
    rejected = [row for row in rows if row['status'] != 'calculated']
    for rank, row in enumerate(ranked, 1):
        row['rank'] = rank
    return dict(first, candidate_count=len(rows), ranked_count=len(ranked), rejected_count=len(rejected),
                ranked=[row['id'] for row in ranked], results=ranked+rejected, rounds=history), candidates


def verify_cuda_scores(reference, actual):
    expected = {row['id']: row for row in reference['results']}
    measured = {row['id']: row for row in actual['results']}
    if expected.keys() != measured.keys() or reference['ranked'] != actual['ranked']:
        raise RuntimeError('CPU and CUDA candidate identities or ranking disagree')
    max_wfe_delta = max_spot_delta = 0.
    for identity, left in expected.items():
        right = measured[identity]
        if (left['status'] != right['status']
                or [reason['code'] for reason in left['reasons']] != [reason['code'] for reason in right['reasons']]):
            raise RuntimeError(f'CPU and CUDA screening status disagrees for {identity}')
        if left.get('manufacturing') != right.get('manufacturing'):
            raise RuntimeError(f'CPU and CUDA nominal-geometry audits disagree for {identity}')
        if len(left['groups']) != len(right['groups']):
            raise RuntimeError(f'CPU and CUDA field/wavelength coverage disagrees for {identity}')
        for a, b in zip(left['groups'], right['groups']):
            if (a['field_index'], a['wave_index']) != (b['field_index'], b['wave_index']):
                raise RuntimeError(f'CPU and CUDA group order disagrees for {identity}')
            max_wfe_delta = max(max_wfe_delta, abs(a['wfe_waves']-b['wfe_waves']))
            max_spot_delta = max(max_spot_delta, abs(a['spot_um']-b['spot_um']))
    if max_wfe_delta > 2e-8 or max_spot_delta > 1e-6:
        raise RuntimeError(f'CUDA score disagreement exceeds tolerance: {max_wfe_delta} waves, {max_spot_delta} um')
    return {'serial_cuda_rank_identical': True, 'max_wfe_delta_waves': max_wfe_delta,
            'max_spot_delta_um': max_spot_delta, 'wfe_tolerance_waves': 2e-8,
            'spot_tolerance_um': 1e-6}


def evaluate_batch(model, candidates, args, screening, *, variables=None, random_seed=0):
    supported = {'field_indices', 'wave_indices', 'wfe_limit_waves', 'spot_limit_um',
                 'manufacturing_policy'}
    if unsupported := screening.keys()-supported:
        raise ValueError(f'Unknown screening settings: {", ".join(sorted(unsupported))}')
    coarse_settings = dict(screening, radial=args.radial, azimuthal=args.azimuthal)
    rounds = getattr(args, 'rounds', 1)
    timing = {'coarse_backend': args.backend}
    if args.verify_serial:
        start = perf_counter()
        with CandidateScreen(model, workers=1, **coarse_settings) as serial:
            serial_result, _ = screen_rounds(serial, candidates, variables, rounds, random_seed)
        timing['serial_seconds'] = perf_counter()-start
    backend_start = perf_counter()
    if args.backend == 'cuda':
        from .standalone_gpu_screen import CudaCandidateScreen

        engine = CudaCandidateScreen(model, device=args.cuda_device or 0, **coarse_settings)
        warmup_size = 8
        warmup_label = 'cuda_start_and_warmup_seconds'
    else:
        engine = CandidateScreen(model, workers=args.workers, **coarse_settings)
        warmup_size = args.workers if args.workers > 1 else 0
        warmup_label = 'pool_start_and_warmup_seconds'
    with engine as screen:
        if args.verify_serial and warmup_size:
            screen.screen(candidates[:warmup_size])
            timing[warmup_label] = perf_counter()-backend_start
        start = perf_counter()
        coarse, evaluated = screen_rounds(screen, candidates, variables, rounds, random_seed)
        timing['coarse_seconds'] = perf_counter()-start
    if args.verify_serial:
        if args.backend == 'cuda':
            timing['cuda_verification'] = verify_cuda_scores(serial_result, coarse)
        else:
            same = serial_result['results'] == coarse['results'] and serial_result['ranked'] == coarse['ranked']
            if not same:
                raise RuntimeError('Serial and parallel screening results disagree')
            timing['serial_parallel_scores_identical'] = True
        timing['warmed_speedup'] = timing['serial_seconds']/timing['coarse_seconds']
    dense = None
    if args.dense_top and coarse['ranked']:
        by_id = {candidate['id']: candidate for candidate in evaluated}
        finalists = coarse['ranked'][:args.dense_top]
        if 'baseline' in by_id and 'baseline' not in finalists:
            finalists.append('baseline')
        dense_settings = {key: screening[key] for key in
                          ('wfe_limit_waves', 'spot_limit_um', 'manufacturing_policy') if key in screening}
        start = perf_counter()
        with CandidateScreen(model, workers=min(args.workers, len(finalists)), radial=20, azimuthal=64,
                             **dense_settings) as screen:
            dense = screen.screen([by_id[identity] for identity in finalists])
        timing['dense_seconds'] = perf_counter()-start
    return coarse, dense, timing, evaluated


def seed_report(args):
    from .standalone_seed_search import search_seeds, seed_search_example
    from .standalone_seed_realize import realize_seed

    if args.demo:
        config = {'search': dict(seed_search_example(), object_na=.035, object_field_radius_mm=2.),
                  'realization': {}}
    else:
        config = read_json(args.config)
    start = perf_counter()
    search = search_seeds(**config['search'])
    report = {'mode': 'seed', 'search': search, 'search_seconds': perf_counter()-start}
    if not search['seeds']:
        return dict(report, status='no_paraxial_seeds'), None
    if 'realization' not in config:
        return dict(report, status='paraxial_seeds_only'), None

    realization = dict(config['realization'])
    if search['object_na'] is not None:
        if realization.setdefault('object_na', search['object_na']) != search['object_na']:
            raise ValueError('Search and realization must use the same object NA')
    if realization.get('stop_group') not in (None, search['stop_group']):
        raise ValueError('Search and realization must use the same stop group')
    radius_limits = search['max_group_radius_mm']
    if radius_limits is not None and 'max_group_radius_mm' not in realization:
        realization['max_group_radius_mm'] = radius_limits
    radius = search['object_field_radius_mm']
    default_fields = [[0., 0.], [0., radius/2.], [0., radius]] if radius else [[0., 0.]]
    fields = np.asarray(realization.setdefault('fields', default_fields), dtype=float)
    if (fields.ndim != 2 or fields.shape[1] != 2 or not len(fields)
            or not np.isfinite(fields).all() or np.max(np.linalg.norm(fields, axis=1)) > radius+1e-10):
        raise ValueError('Realization fields must lie inside the searched object field radius')
    candidates = []
    realizations = []
    for index, seed in enumerate(search['seeds'], 1):
        identity = f'seed_{index:04d}'
        try:
            model = realize_seed(seed, **realization)
            details = model['seed_realization']
            limits = search['max_group_radius_mm']
            if limits is not None and np.any(np.asarray(details['clear_radii_mm']) > limits):
                raise ValueError('Thick seed exceeds a group clear-radius limit')
            power = np.asarray(details['realized_powers_mm_inv'])
            limits = np.asarray(search['power_bounds_mm_inv'])
            if np.any(power < limits[:, 0]-1e-12) or np.any(power > limits[:, 1]+1e-12):
                raise ValueError('Thick seed exceeds a group power bound')
            if np.any(abs(power) < search['minimum_power_mm_inv']-1e-12):
                raise ValueError('Thick seed is below a minimum group power')
        except (ValueError, RuntimeError) as error:
            realizations.append({'id': identity, 'status': 'rejected', 'reason': str(error)})
        else:
            realizations.append({'id': identity, 'status': 'realized',
                                 'details': model.get('seed_realization', {})})
            candidates.append({'id': identity, 'model': model})
    report['realizations'] = realizations
    if not candidates:
        return dict(report, status='no_realized_seeds'), None
    coarse, dense, timing, _ = evaluate_batch(None, candidates, args, config.get('screening', {}))
    ranking = (dense if dense is not None else coarse)['ranked']
    best_id = ranking[0] if ranking else None
    best_model = next((candidate['model'] for candidate in candidates if candidate['id'] == best_id), None)
    return dict(report, status='calculated' if ranking else 'no_viable_candidates',
                best_id=best_id, coarse=coarse, dense=dense, timing=timing), best_model


def screen_report(args):
    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    model = load_model(args.model, args.glass_dir)
    config = read_json(args.config)
    variables = config['curvature_variables']
    random_seed = config.get('random_seed', 0)
    candidates = curvature_candidates(variables, config.get('candidate_count', 48), random_seed=random_seed)
    if any(variable['surface'] >= len(model['surface_rows']) for variable in variables):
        raise ValueError('Screening variables must refer to surfaces before the image')
    coarse, dense, timing, candidates = evaluate_batch(model, candidates, args, config.get('screening', {}),
                                                       variables=variables, random_seed=random_seed)
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != before:
        raise RuntimeError('Input model changed during screening')
    ranking = (dense if dense is not None else coarse)['ranked']
    return dict(mode='screen', status='calculated' if ranking else 'no_viable_candidates',
                input_sha256=before, best_id=ranking[0] if ranking else None,
                coarse=coarse, dense=dense, timing=timing, candidate_definitions=candidates)


def group_report(args):
    from .standalone_group_search import search_group_gaps

    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    model = load_model(args.model, args.glass_dir)
    config = read_json(args.config)
    start = perf_counter()
    controls = config['search']
    if 'curvature_modes' in controls:
        from .standalone_coupled_group_search import search_coupled_group_gaps

        search = search_coupled_group_gaps(model, **controls)
        mode = 'coupled_groups'
    else:
        search = search_group_gaps(model, **controls)
        mode = 'groups'
    candidates = search.pop('candidates')
    report = {'mode': mode, 'search': search, 'search_seconds': perf_counter()-start,
              'input_sha256': before,
              'candidate_geometry': [{key: value for key, value in candidate.items() if key != 'model'}
                                     for candidate in candidates]}
    if not candidates:
        return dict(report, status='no_group_candidates'), None
    tasks = [{'id': candidate['id'], 'model': candidate['model']} for candidate in candidates]
    coarse, dense, timing, _ = evaluate_batch(None, tasks, args, config.get('screening', {}))
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != before:
        raise RuntimeError('Input model changed during group screening')
    ranking = (dense if dense is not None else coarse)['ranked']
    best_id = ranking[0] if ranking else None
    best_model = next((candidate['model'] for candidate in candidates if candidate['id'] == best_id), None)
    return dict(report, status='calculated' if ranking else 'no_viable_candidates',
                best_id=best_id, coarse=coarse, dense=dense, timing=timing), best_model


def add_scoring_arguments(parser):
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1),
                        help='Worker count for CPU screening and dense finalist review')
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu',
                        help='Coarse screening backend; dense finalist review always uses CPU')
    parser.add_argument('--cuda-device', type=int, help='CUDA device index (default 0)')
    parser.add_argument('--radial', type=int, default=8, help='Coarse radial pupil quadrature order')
    parser.add_argument('--azimuthal', type=int, default=32, help='Coarse pupil azimuthal samples')
    parser.add_argument('--dense-top', type=int, default=0, help='Rescore top candidates on all fields and waves')
    parser.add_argument('--verify-serial', action='store_true', help='Also verify against serial CPU scores and report timing')
    parser.add_argument('--out', type=Path, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    seed = commands.add_parser('seed', help='Search and realize three-group finite-conjugate seeds')
    source = seed.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true')
    source.add_argument('--config', type=Path)
    seed.add_argument('--best-model-out', type=Path, help='Write the selected thick seed as a normalized model')
    add_scoring_arguments(seed)

    screen = commands.add_parser('screen', help='Screen bounded curvature candidates for an existing model')
    screen.add_argument('--model', type=Path, required=True)
    screen.add_argument('--glass-dir', type=Path)
    screen.add_argument('--config', type=Path, required=True)
    screen.add_argument('--rounds', type=int, choices=range(1, 17), default=1,
                        metavar='1-16', help='Refine around the leading candidate using the same backend')
    add_scoring_arguments(screen)

    groups = commands.add_parser('groups', help='Search multi-element group gaps with optional bounded curvature modes')
    groups.add_argument('--model', type=Path, required=True)
    groups.add_argument('--glass-dir', type=Path)
    groups.add_argument('--config', type=Path, required=True)
    groups.add_argument('--best-model-out', type=Path, help='Write the leading group candidate as a normalized model')
    add_scoring_arguments(groups)

    args = parser.parse_args(argv)
    if args.cuda_device is not None and (args.backend != 'cuda' or args.cuda_device < 0):
        parser.error('--cuda-device requires --backend cuda and a nonnegative index')
    if not 0 <= args.dense_top < MAX_CANDIDATES_PER_BATCH:
        parser.error(f'--dense-top must be between 0 and {MAX_CANDIDATES_PER_BATCH-1}')
    paths = [path.resolve() for path in (getattr(args, 'model', None), args.config, args.out,
                                        getattr(args, 'best_model_out', None)) if path is not None]
    if len(set(paths)) != len(paths):
        parser.error('Inputs and outputs must use different files')
    try:
        if args.command == 'seed':
            report, best = seed_report(args)
        elif args.command == 'groups':
            report, best = group_report(args)
        else:
            report, best = screen_report(args), None
        write_json(args.out, report)
        if getattr(args, 'best_model_out', None):
            if best is None:
                print('No traced candidate available for --best-model-out', file=sys.stderr)
                return 2
            write_json(args.best_model_out, best)
        print(f"{report['status']} | {args.out}")
        return 0 if report['status'] in ('calculated', 'paraxial_seeds_only') else 2
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f'Optimization workflow error: {error}', file=sys.stderr)
        return 2
