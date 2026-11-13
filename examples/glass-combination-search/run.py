"""Real crown/flint/crown branches: fixed 240 mm, -2x, spherical finite relay."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import numpy as np
from ai_zemax.standalone import disk_quadrature, reference_opd_mm, wavefront_rms
from ai_zemax.standalone_system import OpticalSystem, rectangular_pupil
from ai_zemax.standalone_seed_search import search_seeds
from ai_zemax.standalone_seed_realize import realize_seed
from ai_zemax.standalone_glass_catalog import load_index
from ai_zemax.standalone_glass_search import run_combination_search
from ai_zemax.standalone_local_solver import EfficientSolveMixin
from ai_zemax.standalone_manufacturing import audit_manufacturing, audit_ray_envelope

# All material combinations and all initial structures use this same target.
TARGET = dict(track_mm=240., magnification=-2., object_na=.035, field_radius_mm=2.,
    wavelengths_um=[.486, .588, .656], wavelength_weights=[1., 1., 1.], primary_wave=1,
    wfe_limit_waves=.10, spot_over_airy_limit=1., image_height_error_mm=.02,
    magnification_error=.01, distortion_percent=.5, clear_radius_mm=6.5, mechanical_radius_mm=7.,
    center_thickness_mm=4., minimum_radial_thickness_mm=1.2, minimum_air_gap_mm=2.,
    maximum_surface_slope=.6, all_spherical=True)
COMBINATIONS = [dict(id=vendor.lower(), choices=[dict(vendor=vendor, name=name) for name in names])
    for vendor, names in (("CDGM", ("H-K9L", "H-ZF2", "H-K9L")),
                          ("SCHOTT", ("N-BK7", "N-SF5", "N-BK7")),
                          ("OHARA", ("S-BSL7", "S-TIM2", "S-BSL7")))]
COMBINATIONS += [
    dict(id="schott_barium", choices=[dict(vendor="SCHOTT", name=name)
        for name in ("N-BAK4", "N-SF10", "N-BAK4")]),
    dict(id="mixed_asymmetric", choices=[dict(vendor=vendor, name=name) for vendor, name in (
        ("SCHOTT", "N-BAK4"), ("CDGM", "H-ZF52"), ("OHARA", "S-BSL7"))]),
]


def policy(target, samples=257):
    return dict(schema_version=1, policy_id="visible-three-singlet-demonstration-v1", all_spherical=True,
        radial_samples=samples, defaults=dict(minimum_center_thickness_mm=2.,
            minimum_radial_thickness_mm=target["minimum_radial_thickness_mm"],
            minimum_air_gap_mm=target["minimum_air_gap_mm"], minimum_rim_gap_mm=target["minimum_air_gap_mm"],
            maximum_diameter_mm=14., maximum_surface_slope=target["maximum_surface_slope"],
            minimum_clear_to_mechanical_radial_margin_mm=.5),
        lenses=[dict(front_surface=i, rear_surface=i + 1) for i in (1, 3, 5)],
        surfaces={str(i):dict(clear_radius_mm=6.5, mechanical_radius_mm=7.) for i in range(1, 7)},
        sources=["Demonstration project allocations; not a supplier process specification"])


def realize(seed, frozen, target):
    m = realize_seed(seed, refractive_indices=frozen["indices_wave_by_group"],
        wavelengths_um=target["wavelengths_um"], primary_wave=target["primary_wave"],
        center_thicknesses_mm=[target["center_thickness_mm"]] * 3,
        object_na=target["object_na"], fields=[[0., 0.], [0., 1.], [0., 2.]],
        aperture_margin=1.1, minimum_air_gap_mm=2., minimum_edge_thickness_mm=1.2,
        max_group_radius_mm=6.5)
    m["surface_types"] = ["STANDARD"] * 7
    m["material_names"] = [None] * 8
    m["material_catalogs"] = [None] * 8
    for front, material in zip((1, 3, 5), frozen["materials"]):
        m["material_names"][front] = material["name"]
        m["material_catalogs"][front] = material["catalog_id"]
    for row in m["surface_rows"]:
        row["aperture_radius_mm"] = None
        row["conic"] = 0.
        row["even_coefficients"] = []
    m["glass_combination"] = frozen
    return m


class Fit(EfficientSolveMixin):
    def __init__(self, model, target, fields=None, sampling=(5, 16)):
        self.initial = deepcopy(model)
        self.target = target
        self.fields = np.array([0., 1., 2.] if fields is None else fields)
        self.nodes, self.weights = disk_quadrature(*sampling)
        self.wfe_target, self.spot_target = .09, .95
        self.bounds = np.tile([-3., 3.], (9, 1))
        self.counts = dict(traced_candidates=0, optical_bundles=0, rays=0)

    def model(self, q):
        m = deepcopy(self.initial)
        for i in range(6):
            m["surface_rows"][i]["curvature_mm_inv"] += .02 * float(q[i])
        for j in range(3):
            for i in (2 * j, 2 * j + 1):
                m["vertices_mm"][i] += 10. * float(q[6 + j])
        m["fields"] = [[0., float(h)] for h in self.fields]
        return m

    def batch(self, qs):
        out = []
        for q in qs:
            self.counts["traced_candidates"] += 1
            try:
                m = self.model(q)
                system = OpticalSystem(m)
                wfe, spots, wave_vectors, spot_vectors, heights, mags = [], [], [], [], [], []
                phase = np.linspace(0., 2.*np.pi, 24, endpoint=False)
                rim = np.column_stack((np.cos(phase), np.sin(phase)))
                peaks = np.zeros(7)
                for wave in range(3):
                    axial = system._shoot(wave, rim, *system.launch_field([0., 0.]))
                    self.counts["rays"] += len(rim)
                    if not axial.valid.all():
                        raise ValueError("Axial rim failed")
                    image_na = float(np.max(np.linalg.norm(axial.directions[:, :2], axis=1)))
                    airy = .61 * m["wavelengths_um"][wave] / image_na
                    chief_small = system._shoot(wave, np.zeros((1, 2)), *system.launch_field([0., 1e-4]))
                    self.counts["rays"] += 1
                    mags.append(float(chief_small.points[0, 1] / 1e-4))
                    ww, ss, wv, sv, hh = [], [], [], [], []
                    for h in self.fields:
                        data, rays, chief = system.evaluate([0., h], wave, self.nodes, self.weights)
                        self.counts["optical_bundles"] += 1
                        self.counts["rays"] += len(self.nodes) + 1
                        radius = system.reference_distance_mm / chief.directions[0, 2]
                        opd = reference_opd_mm(rays, chief, radius, system.indices[wave, -1])
                        rms, residual = wavefront_rms(opd, self.weights, m["wavelengths_um"][wave], rays.directions[:, :2])
                        centered = rays.points[:, :2] - np.sum(self.weights[:, None] * rays.points[:, :2], axis=0)
                        ww.append(rms); ss.append(data["spot_rms_um"] / airy)
                        wv.append(np.sqrt(self.weights) * residual)
                        sv.append((np.sqrt(self.weights[:, None]) * centered * 1000. / airy).ravel())
                        hh.append(float(chief.points[0, 1] + 2. * h))
                        rr = system._shoot(wave, rim, *system.launch_field([0., h]), record_path=True)
                        self.counts["rays"] += len(rim)
                        if not rr.valid.all():
                            raise ValueError("Full-field rim failed")
                        peaks = np.maximum(peaks, np.max(np.linalg.norm(rr.surface_points[:, :, :2], axis=2), axis=1))
                    wfe.append(ww); spots.append(ss); wave_vectors.append(wv); spot_vectors.append(sv); heights.append(hh)
                surfaces = system.surfaces[1]
                radius = self.target["mechanical_radius_mm"]
                sag = np.array([sf.sag_and_gradient(radius**2)[0] for sf in surfaces[:6]])
                slopes = np.array([abs(2. * radius * sf.sag_and_gradient(radius**2)[1]) for sf in surfaces[:6]])
                if not np.isfinite(sag).all():
                    raise ValueError("Spherical mechanical domain failed")
                z = np.asarray(m["vertices_mm"])
                edge = np.array([z[i + 1] - z[i] + sag[i + 1] - sag[i] for i in (0, 2, 4)])
                air = np.array([min(z[i + 1] - z[i], z[i + 1] - z[i] + sag[i + 1] - sag[i]) for i in (1, 3)])
                geo = np.r_[edge - self.target["minimum_radial_thickness_mm"], air - 2.05,
                    self.target["maximum_surface_slope"] - slopes, 6.5 - peaks[:6],
                    z[0] - 20., 220. - z[5], np.diff(z[:6])[::2] - 2.]
                out.append(dict(wfe=np.asarray(wfe), spot=np.asarray(spots), wave_vector=np.asarray(wave_vectors),
                    spot_vector=np.asarray(spot_vectors), height=np.asarray(heights), mags=np.asarray(mags), geo=geo))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                out.append(None)
        return out

    def slacks(self, row):
        return dict(wfe=1. - row["wfe"] / self.wfe_target,
            spot=1. - row["spot"] / self.spot_target,
            height=1. - abs(row["height"]) / .018,
            magnification=1. - abs(row["mags"] + 2.) / .009,
            geometry=row["geo"])


def screen(model, target):
    fit = Fit(model, target)
    row = fit.batch([np.zeros(9)])[0]
    if row is None:
        raise ValueError("Physical seed real-ray screen failed")
    if row["geo"].min() < 0.:
        raise ValueError("Physical seed geometry screen failed")
    return dict(score=float(row["wfe"].max() + .1 * row["spot"].max()), trace_counts=fit.counts)


def validate(model, target):
    # Fixed denser holdout fields, two independent quadratures, and all colors.
    dense = deepcopy(model)
    heights = np.linspace(0., 2., 25)
    dense["fields"] = [[float(h * np.sin(a)), float(h * np.cos(a))]
                       for h in heights for a in (0., np.pi / 4.)]
    system = OpticalSystem(dense)
    maximum_wfe, maximum_spot, maximum_height, maximum_mag, maximum_distortion = (0.,) * 5
    optical_rays = 0
    for nodes, weights in (disk_quadrature(16, 48), rectangular_pupil(20)):
        for wave in range(3):
            phase = np.linspace(0., 2.*np.pi, 128, endpoint=False)
            rim = np.column_stack((np.cos(phase), np.sin(phase)))
            axial = system._shoot(wave, rim, *system.launch_field([0., 0.]))
            optical_rays += len(rim)
            if not axial.valid.all():
                return dict(passed=False, reason="dense axial rim failure")
            image_na = float(np.max(np.linalg.norm(axial.directions[:, :2], axis=1)))
            airy = .61 * dense["wavelengths_um"][wave] / image_na
            para = system._shoot(wave, np.zeros((1, 2)), *system.launch_field([0., 1e-4]))
            optical_rays += 1
            maximum_mag = max(maximum_mag, abs(float(para.points[0, 1] / 1e-4) + 2.))
            for field in dense["fields"]:
                data, _, chief = system.evaluate(field, wave, nodes, weights)
                optical_rays += len(nodes) + 1
                h = float(np.linalg.norm(field))
                height = float(chief.points[0, :2] @ (np.asarray(field) / h)) if h else 0.
                maximum_wfe = max(maximum_wfe, data["wavefront_rms_waves"])
                maximum_spot = max(maximum_spot, data["spot_rms_um"] / airy)
                maximum_height = max(maximum_height, abs(height + 2. * h))
                if h >= .25:
                    maximum_distortion = max(maximum_distortion, abs(height / (-2. * h) - 1.) * 100.)
    geometry = audit_manufacturing(dense, policy(target, 2049))
    joined = audit_ray_envelope(system, geometry, disk_quadrature(12, 32)[0], rim_samples=128)
    envelope_rays = len(dense["fields"]) * 3 * (12 * 32 + 1 + 128)
    checks = dict(wavefront=maximum_wfe <= target["wfe_limit_waves"],
        spot=maximum_spot <= target["spot_over_airy_limit"], height=maximum_height <= target["image_height_error_mm"],
        magnification=maximum_mag <= target["magnification_error"], distortion=maximum_distortion <= target["distortion_percent"],
        manufacturing=joined["passed"], all_spherical=all(row.get("conic", 0.) == 0. and
            not any(row.get("even_coefficients", ())) for row in model["surface_rows"]),
        track=abs(model["vertices_mm"][-1] - target["track_mm"]) < 1e-12)
    return dict(passed=all(checks.values()), checks=checks, wfe_worst_waves=maximum_wfe,
        spot_over_airy_worst=maximum_spot, image_height_error_mm=maximum_height,
        magnification_error=maximum_mag, distortion_percent=maximum_distortion,
        field_points=len(dense["fields"]), methods=["GQ16/48", "RECT20"],
        trace_counts=dict(optical_ray_transports=optical_rays, manufacturing_ray_transports=envelope_rays,
                          total_ray_transports=optical_rays + envelope_rays),
        manufacturing=joined, scope="finite-sampled nominal validation; not tolerance/yield qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=65)
    parser.add_argument("--seeds-per-branch", type=int, default=2)
    parser.add_argument("--seed-count", type=int, default=14)
    args = parser.parse_args()
    seeds = search_seeds(track_mm=240., magnification=-2., gap_bounds_mm=[[35., 85.], [8., 50.], [8., 55.], [70., 180.]],
        power_bounds_mm_inv=[[.005, .10], [-.08, -.015], [.005, .10]], samples=4096,
        count=args.seed_count, random_seed=47, stop_group=2, object_na=.035,
        object_field_radius_mm=2., max_group_radius_mm=6.0)["seeds"]
    def optimize(model, target, directory):
        fit = Fit(model, target)
        initial = fit.batch([np.zeros(9)])[0]
        tick = perf_counter()
        report = fit.solve(directory, args.iterations, .25, .25, .0001,
            use_cones=True, second_order=False, verbose=False, tolerances=1e-8,
            difference_step=1e-4, min_trust=1e-7, max_trust=1., stagnation_retries=8)
        result = fit.model(fit.last_solution_q)
        return result, dict(status=report["status"], counters=report["counters"], actual_counts=fit.counts,
            seconds=perf_counter()-tick, initial_wfe_waves=float(initial["wfe"].max()),
            initial_merit=report["baseline"]["merit"], final_merit=report["merit"],
            final_wfe_waves=float(fit.last_solution_row["wfe"].max()), local_slacks=report["gates"])
    result = run_combination_search(index=load_index(args.index), combinations=COMBINATIONS, seeds=seeds,
        target=TARGET, output_directory=args.out, realize=realize, screen=screen, optimize=optimize,
        validate=validate, seeds_per_branch=args.seeds_per_branch)
    (args.out / "INITIAL_STRUCTURES.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("branch_count", "passed_branches", "all_branches_passed", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
