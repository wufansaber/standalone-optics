"""Generate finite-conjugate paraxial seeds for a three-group refractive system."""

import numpy as np


def _transfer(distance):
    return np.array([[1., distance], [0., 1.]])


def _lens(power):
    return np.array([[1., 0.], [-power, 1.]])


def system_matrix(gaps_mm, powers_mm_inv):
    """Return the object-to-image ABCD matrix in air for rays [height, slope]."""
    gaps = np.asarray(gaps_mm, dtype=float)
    powers = np.asarray(powers_mm_inv, dtype=float)
    if (gaps.shape != (4,) or powers.shape != (3,) or not np.isfinite(gaps).all()
            or np.any(gaps <= 0.) or not np.isfinite(powers).all()):
        raise ValueError("Use four positive gaps and three finite signed powers")
    result = _transfer(gaps[0])
    for power, gap in zip(powers, gaps[1:]):
        result = _transfer(gap) @ _lens(power) @ result
    return result


def _bounds(values, count, label):
    values = np.asarray(values, dtype=float)
    if (values.shape != (count, 2) or not np.isfinite(values).all()
            or np.any(values[:, 0] > values[:, 1])):
        raise ValueError(f"{label} requires {count} finite ordered lower/upper pairs")
    return values


def _gap_samples(unit, bounds, track):
    """Map samples to the fixed-track slice without rejecting legal thin gaps."""
    count = len(unit)
    gaps = np.empty((count, 4))
    remaining = np.full(count, track)
    for index in range(3):
        downstream = bounds[index+1:].sum(axis=0)
        low = np.maximum(bounds[index, 0], remaining-downstream[1])
        high = np.minimum(bounds[index, 1], remaining-downstream[0])
        gaps[:, index] = low + unit[:, index]*np.maximum(0., high-low)
        remaining -= gaps[:, index]
    gaps[:, 3] = remaining
    return gaps


def _solve_outer_powers(gaps, middle, magnification, track):
    """Solve the first and last powers from top row [magnification, 0]."""
    a, b, c, d = gaps.T
    # K propagates from just after group 1 to just before group 3.
    k00 = 1.-c*middle
    k01 = b+c-b*c*middle
    k10 = -middle
    k11 = 1.-b*middle
    # The top right entry of T(d) L(p3) K is B - a*A = -a*m.
    target = -a*magnification
    regular = (np.abs(k01) > 1e-10*track) & (np.abs(target) > 1e-10*track)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        last = (k01+d*k11-target)/(d*k01)
        r00 = (1.-d*last)*k00+d*k10
        first = (r00-magnification)/target
    powers = np.column_stack((first, middle, last))
    regular &= np.isfinite(powers).all(axis=1)
    return powers, regular


def _radius_estimates(gaps, powers, stop_group, field, numerical_aperture, track):
    """Envelope of chief and axial marginal rays at the thin group planes."""
    a, b, c = gaps[:, :3].T
    p1, p2 = powers[:, :2].T
    a2 = 1.-b*p1
    b2 = a+b*(1.-a*p1)
    before2 = 1.-a*p1
    height = np.column_stack((np.ones(len(a)), a2, a2-c*(p1+p2*a2)))
    slope = np.column_stack((a, b2, b2+c*(before2-p2*b2)))
    stop_height = height[:, stop_group-1]
    stop_slope = slope[:, stop_group-1]
    usable = np.abs(stop_slope) > 1e-10*track
    chief = np.divide(-stop_height, stop_slope, out=np.zeros_like(stop_height), where=usable)
    radius = np.abs(height+slope*chief[:, None])*field + np.abs(slope)*numerical_aperture
    return radius, usable


def _family(powers):
    return "".join("+" if value > 0. else "-" if value < 0. else "0" for value in powers)


def _select_diverse(ordered, gaps, powers, limit, track):
    # Reserve one representative per signed-power family, then vary spacing.
    families = {}
    layouts = {}
    for index in ordered:
        family = _family(powers[index])
        families.setdefault(family, index)
        layout = tuple(np.floor(8.*gaps[index]/track).astype(int))
        layouts.setdefault((family, layout), index)
    selected = set(list(families.values())[:limit])
    for candidates in (layouts.values(), ordered):
        for index in candidates:
            if len(selected) == limit:
                break
            selected.add(index)
    return [int(index) for index in ordered if index in selected]


def search_seeds(*, track_mm, magnification, gap_bounds_mm, power_bounds_mm_inv,
                 samples=4096, count=24, random_seed=0, stop_group=2,
                 object_na=None, max_group_radius_mm=None, object_field_radius_mm=0.0,
                 minimum_power_mm_inv=0.0):
    """Search three thin groups in air at a finite object and a fixed image plane.

    The four gaps are object to group 1, group 1 to 2, group 2 to 3, and
    group 3 to image. The sampled variables are the gaps (fixed sum) and
    middle-group power. First and last powers are solved algebraically.
    A group-radius estimate uses the centered chief ray at the chosen
    stop and a first-order object-slope radius of NA in air.
    """
    track = float(track_mm)
    magnification = float(magnification)
    gap_bounds = _bounds(gap_bounds_mm, 4, "Gap bounds")
    power_bounds = _bounds(power_bounds_mm_inv, 3, "Power bounds")
    if not np.isfinite(track) or track <= 0. or not np.isfinite(magnification) or magnification == 0.:
        raise ValueError("Use a positive finite track and a finite nonzero signed magnification")
    if np.any(gap_bounds <= 0.) or not gap_bounds[:, 0].sum() <= track <= gap_bounds[:, 1].sum():
        raise ValueError("Positive gap bounds must admit the requested fixed track")
    if (type(samples) is not int or type(count) is not int or samples < 1 or count < 1
            or type(random_seed) is not int or random_seed < 0
            or type(stop_group) is not int or stop_group not in (1, 2, 3)):
        raise ValueError("Use positive integer samples and count, a nonnegative seed and stop group 1, 2 or 3")
    field = float(object_field_radius_mm)
    minimum_power = float(minimum_power_mm_inv)
    na = float(object_na) if object_na is not None else 0.
    if (not np.isfinite(field) or field < 0. or not np.isfinite(na) or not 0. <= na < 1.
            or not np.isfinite(minimum_power) or minimum_power < 0.):
        raise ValueError("Use a nonnegative field radius and minimum power, and an object NA in [0, 1)")
    radius_bound = None
    if max_group_radius_mm is not None:
        if object_na is None:
            raise ValueError("An object NA is required with a maximum group radius")
        radius_bound = np.broadcast_to(np.asarray(max_group_radius_mm, dtype=float), (3,))
        if not np.isfinite(radius_bound).all() or np.any(radius_bound <= 0.):
            raise ValueError("Use a positive maximum radius or three positive radii")

    rng = np.random.default_rng(random_seed)
    # One sample per stratum in each variable keeps small deterministic runs useful.
    unit = np.column_stack([(rng.permutation(samples)+rng.random(samples))/samples for _ in range(4)])
    gaps = _gap_samples(unit, gap_bounds, track)
    middle_low, middle_high = power_bounds[1]
    middle = middle_low + unit[:, 3]*(middle_high-middle_low)
    powers, regular = _solve_outer_powers(gaps, middle, magnification, track)
    tolerance = 64.*np.finfo(float).eps*np.maximum(np.max(np.abs(power_bounds), axis=1), 1./track)
    in_bounds = np.all((powers >= power_bounds[:, 0]-tolerance) & (powers <= power_bounds[:, 1]+tolerance)
                       & (np.abs(powers) >= minimum_power-tolerance), axis=1)
    viable = regular & in_bounds
    rejections = {
        "singular_outer_power_solve": int(np.count_nonzero(~regular)),
        "power_bounds": int(np.count_nonzero(regular & ~in_bounds)),
        "stop_conjugate": 0,
        "group_radius": 0,
        "matrix_residual": 0,
    }

    radii = None
    if object_na is not None or field > 0.:
        radii = np.zeros((samples, 3))
        ids = np.flatnonzero(viable)
        radii[ids], usable = _radius_estimates(gaps[ids], powers[ids], stop_group, field, na, track)
        rejections["stop_conjugate"] = int(np.count_nonzero(~usable))
        viable[ids[~usable]] = False
        if radius_bound is not None:
            too_large = viable & np.any(radii > radius_bound, axis=1)
            rejections["group_radius"] = int(np.count_nonzero(too_large))
            viable[too_large] = False

    # Re-evaluate the product to guard against loss of significance in the solve.
    matrices = {}
    for index in np.flatnonzero(viable):
        matrix = system_matrix(gaps[index], powers[index])
        if (not np.isfinite(matrix).all() or abs(matrix[0, 0]-magnification) > 1e-9*max(1., abs(magnification))
                or abs(matrix[0, 1]) > 1e-9*track):
            viable[index] = False
            rejections["matrix_residual"] += 1
            continue
        matrices[index] = matrix

    accepted = np.flatnonzero(viable)
    normalized = np.abs(powers[accepted])*track
    scores = np.max(normalized, axis=1) + np.sqrt(np.mean(normalized**2, axis=1))
    order = np.argsort(scores, kind="stable")
    ordered = accepted[order]
    score_by_index = dict(zip(accepted, scores))
    # Fixed bounds can map many samples to exactly the same physical seed.
    keys = np.column_stack((gaps[ordered]/track, powers[ordered]*track))
    _, unique_order = np.unique(np.round(keys, decimals=12), axis=0, return_index=True)
    ordered = ordered[np.sort(unique_order)]
    selected = _select_diverse(ordered, gaps, powers, count, track)
    seeds = []
    for rank, index in enumerate(selected, start=1):
        power = powers[index]
        radius = radii[index].tolist() if radii is not None else None
        seeds.append({
            "rank": rank,
            "sample_index": index,
            "score": float(score_by_index[index]),
            "power_sign_family": _family(power),
            "gaps_mm": gaps[index].tolist(),
            "group_positions_mm": np.cumsum(gaps[index, :3]).tolist(),
            "powers_mm_inv": power.tolist(),
            "group_efl_mm": [1./float(value) if value != 0. else None for value in power],
            "stop_group": stop_group,
            "estimated_group_radius_mm": radius,
            "matrix": matrices[index].tolist(),
        })
    return {
        "method": "finite_conjugate_three_group_thin_lens",
        "status": "seeds_found" if seeds else "no_feasible_seeds",
        "track_mm": track,
        "magnification": magnification,
        "object_na": float(na) if object_na is not None else None,
        "object_field_radius_mm": field,
        "stop_group": stop_group,
        "random_seed": random_seed,
        "gap_bounds_mm": gap_bounds.tolist(),
        "power_bounds_mm_inv": power_bounds.tolist(),
        "minimum_power_mm_inv": minimum_power,
        "max_group_radius_mm": radius_bound.tolist() if radius_bound is not None else None,
        "sample_count": samples,
        "accepted_count": int(len(accepted)),
        "unique_accepted_count": int(len(ordered)),
        "requested_seed_count": count,
        "rejections": rejections,
        "score_definition": "peak_abs_track_times_power + rms_track_times_power",
        "selection": "best of each available signed-power family, then spacing diversity and lowest score",
        "gap_convention": "object_to_first, first_to_second, second_to_third, third_to_image",
        "radius_convention": "first_order_centered_stop_envelope_in_air",
        "seeds": seeds,
    }


def seed_search_example():
    """Return inputs for a 240 mm, −2× synthetic finite-conjugate seed search."""
    return {
        "track_mm": 240.,
        "magnification": -2.,
        "gap_bounds_mm": [[30., 70.], [25., 100.], [25., 100.], [30., 80.]],
        "power_bounds_mm_inv": [[-.07, .07], [-.07, .07], [-.07, .07]],
        "object_na": .12,
        "object_field_radius_mm": 4.,
        "max_group_radius_mm": 24.,
    }
