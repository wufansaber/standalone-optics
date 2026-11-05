"""A complete local trace, aberration analysis, and optional singlet refinement."""

import argparse
import json
from pathlib import Path

import numpy as np

from .standalone import (Surface, directions_from_slopes, disk_quadrature,
                         paraxial_transfer, reference_opd_mm, trace, wavefront_rms)
from .standalone_analysis import best_focus, ray_fan, spot_statistics


STOP_Z_MM = 64.0
WAVELENGTH_UM = 0.55
FIELDS_MM = (0.0, 0.5, 1.0)


def lens_surfaces(parameters):
    bend, asphere, _ = parameters
    return (
        Surface(STOP_Z_MM, aperture_radius_mm=4.0),
        Surface(65.0, 1.5168, 1.0 / 30.0 + 0.005 * bend,
                even_coefficients=(0.0, 1e-5 * asphere), aperture_radius_mm=6.0),
        Surface(69.0, 1.0, -1.0 / 45.0 + 0.005 * bend, aperture_radius_mm=6.0),
    )


def trace_field(surfaces, field_mm, pupil):
    origin = np.array([0.0, field_mm, 0.0])
    slopes = (4.0 * pupil - origin[:2]) / STOP_Z_MM
    return trace(surfaces, origin, directions_from_slopes(slopes))


def evaluate(parameters, paraxial_image_mm, pupil, weights, *, include_details=False):
    image_mm = paraxial_image_mm + parameters[2]
    surfaces = (*lens_surfaces(parameters), Surface(image_mm))
    exit_transfer = paraxial_transfer(surfaces[1:], STOP_Z_MM)
    reference_mm = exit_transfer[0, 1] / exit_transfer[1, 1]
    residuals, groups = [], []
    for height in FIELDS_MM:
        rays = trace_field(surfaces, height, pupil)
        chief = trace_field(surfaces, height, np.zeros((1, 2)))
        opd = reference_opd_mm(rays, chief, reference_mm)
        rms, phase = wavefront_rms(opd, weights, WAVELENGTH_UM, rays.directions[:, :2])
        residuals.append(phase * np.sqrt(weights / len(FIELDS_MM)))
        if include_details:
            spot = spot_statistics(rays.points[:, :2], weights)
            focus = best_focus(rays.points[:, :2], rays.directions, weights)
            fan_pupil = np.linspace(-1.0, 1.0, 41)
            fans = {}
            for axis, name in enumerate(("sagittal", "tangential")):
                coordinates = np.zeros((len(fan_pupil), 2))
                coordinates[:, axis] = fan_pupil
                fan_rays = trace_field(surfaces, height, coordinates)
                if not fan_rays.valid.all():
                    raise RuntimeError("Demo fan ray failed")
                transverse_axis = (1.0, 0.0) if axis == 0 else (0.0, 1.0)
                fan = ray_fan(fan_pupil, fan_rays.points[:, :2], chief.points[0, :2], transverse_axis)
                fans[name] = (1000.0 * fan).tolist()
            groups.append({"object_height_mm": height, "wavefront_rms_waves": rms,
                           "spot_rms_um": 1000.0 * spot["rms_radius"],
                           "focus_shifts_mm": focus,
                           "fan_pupil_coordinate": fan_pupil.tolist(), "ray_fans_um": fans})
    if groups:
        axis_focus = groups[0]["focus_shifts_mm"]["mean_shift"]
        for group in groups:
            field_focus = group["focus_shifts_mm"]["mean_shift"]
            group["field_curvature_image_mm"] = field_focus - axis_focus
    return np.concatenate(residuals), groups


def run(optimize=False):
    pupil, weights = disk_quadrature(12, 32)
    start = np.zeros(3)
    transfer = paraxial_transfer(lens_surfaces(start), 0.0)
    image_mm = 69.0 - transfer[0, 1] / transfer[1, 1]
    initial_residual, initial_groups = evaluate(start, image_mm, pupil, weights, include_details=True)
    report = {"model": "synthetic_singlet", "uses_zemax": False,
              "wavelength_um": WAVELENGTH_UM, "pupil_ray_count_per_field": len(pupil),
              "initial": {"image_position_mm": image_mm, "combined_wfe_waves": float(np.linalg.norm(initial_residual)),
                          "parameters": start.tolist(), "fields": initial_groups},
              "parameter_order": ["bend_knob", "a4_knob", "focus_shift_mm"]}
    if optimize:
        from scipy.optimize import least_squares

        def residual(parameters):
            return evaluate(parameters, image_mm, pupil, weights)[0]

        solution = least_squares(residual, start, bounds=([-0.8, -0.4, -5.0], [0.8, 0.4, 5.0]),
                                 diff_step=1e-4, max_nfev=45)
        refined_residual, refined_groups = evaluate(solution.x, image_mm, pupil, weights, include_details=True)
        report["refined"] = {"image_position_mm": image_mm + solution.x[2],
                             "combined_wfe_waves": float(np.linalg.norm(refined_residual)),
                             "parameters": solution.x.tolist(),
                             "optimizer_success": bool(solution.success), "optimizer_evaluations": int(solution.nfev),
                             "fields": refined_groups}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimize", action="store_true", help="Run a small SciPy least-squares refinement")
    parser.add_argument("--out", type=Path, help="Save the full report to a JSON file")
    args = parser.parse_args()
    report = run(args.optimize)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(content, encoding="utf-8")
        print(f"Saved {args.out}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
