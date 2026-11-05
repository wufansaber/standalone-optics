"""Finite-conjugate three-group seed search tests."""
import json
import unittest

import numpy as np

from scripts.ai_zemax.standalone_seed_search import search_seeds, seed_search_example, system_matrix


def propagate(gaps, powers, height, slope):
    """Independent scalar thin-lens propagation before and after each group."""
    group_heights = []
    for gap, power in zip(gaps[:3], powers):
        height += gap*slope
        group_heights.append(height)
        slope -= power*height
    return height + gaps[3]*slope, group_heights


class SeedSearchTests(unittest.TestCase):
    def test_fixed_seed_solves_conjugate_and_deduplicates(self):
        result = search_seeds(
            track_mm=200., magnification=-2., samples=32, count=6,
            gap_bounds_mm=[[100., 100.], [40., 40.], [40., 40.], [20., 20.]],
            power_bounds_mm_inv=[[.02125, .02125], [0., 0.], [-.0625, -.0625]],
        )
        self.assertEqual(result["accepted_count"], 32)
        self.assertEqual(result["unique_accepted_count"], 1)
        self.assertEqual(len(result["seeds"]), 1)
        seed = result["seeds"][0]
        self.assertEqual(seed["power_sign_family"], "+0-")
        self.assertIsNone(seed["group_efl_mm"][1])
        np.testing.assert_allclose(seed["matrix"], system_matrix(seed["gaps_mm"], seed["powers_mm_inv"]))
        self.assertAlmostEqual(propagate(seed["gaps_mm"], seed["powers_mm_inv"], 3., -.07)[0], -6.)

    def test_sampled_seeds_respect_geometry_and_have_a_centered_stop(self):
        inputs = seed_search_example()
        inputs.update(samples=1024, count=20, random_seed=31)
        result = search_seeds(**inputs)
        self.assertEqual(result, search_seeds(**inputs))
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(result["seeds"]), 20)
        self.assertGreater(len({s["power_sign_family"] for s in result["seeds"]}), 1)
        self.assertEqual(result["accepted_count"] + sum(result["rejections"].values()), 1024)
        for seed in result["seeds"]:
            gaps, powers = seed["gaps_mm"], seed["powers_mm_inv"]
            self.assertAlmostEqual(sum(gaps), inputs["track_mm"])
            for gap, (lower, upper) in zip(gaps, inputs["gap_bounds_mm"]):
                self.assertLessEqual(lower, gap)
                self.assertLessEqual(gap, upper)
            on_axis = propagate(gaps, powers, 0., 1.)
            field_ray = propagate(gaps, powers, 1., 0.)
            self.assertAlmostEqual(on_axis[0], 0.)
            self.assertAlmostEqual(field_ray[0], inputs["magnification"])
            chief_slope = -field_ray[1][1] / on_axis[1][1]
            field = inputs["object_field_radius_mm"]
            rays = [propagate(gaps, powers, field*side, field*side*chief_slope+marginal*inputs["object_na"])[1]
                    for side in (-1., 1.) for marginal in (-1., 1.)]
            envelope = np.max(np.abs(rays), axis=0)
            np.testing.assert_allclose(envelope, seed["estimated_group_radius_mm"])
            self.assertLessEqual(max(envelope), inputs["max_group_radius_mm"])

    def test_impossible_search_and_invalid_configuration(self):
        inputs = dict(track_mm=4., magnification=-1., samples=8, gap_bounds_mm=[[1., 1.]]*4,
                      power_bounds_mm_inv=[[0., 0.]]*3)
        result = search_seeds(**inputs)
        self.assertEqual(result["status"], "no_feasible_seeds")
        self.assertEqual(result["seeds"], [])
        self.assertEqual(result["rejections"]["power_bounds"], 8)
        inputs["track_mm"] = 5.
        with self.assertRaisesRegex(ValueError, "fixed track"):
            search_seeds(**inputs)


if __name__ == "__main__":
    unittest.main()
