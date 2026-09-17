import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from materials.day011.data_io_lab import (
    POSE_FIELDS,
    binary_nll_and_grad,
    chain_value_and_grad,
    demo_config,
    demo_points,
    precision_sweep,
    read_bundle,
    run,
    stable_logsumexp,
    validate_config,
    write_bundle,
)


class Day011Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_config_is_valid(self):
        self.assertEqual(validate_config(demo_config())["schema_version"], 1)

    def test_missing_config_key_is_rejected(self):
        config = demo_config()
        del config["pose"]
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_config(config)

    def test_nonfinite_intrinsic_is_rejected(self):
        config = demo_config()
        config["camera"]["K_px"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_config(config)

    def test_bundle_roundtrip_shapes(self):
        write_bundle(self.root)
        loaded = read_bundle(self.root)
        self.assertEqual(loaded["poses"].shape, (3, 5))
        self.assertEqual(loaded["points"].shape, (8, 3))
        self.assertEqual(loaded["image"].shape, (6, 8, 3))

    def test_npy_roundtrip_is_exact(self):
        write_bundle(self.root)
        restored = read_bundle(self.root)["points"]
        np.testing.assert_array_equal(restored, demo_points())

    def test_bad_csv_header_is_rejected(self):
        write_bundle(self.root)
        with (self.root / "poses.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows([["bad"], ["1"]])
        with self.assertRaisesRegex(ValueError, "header"):
            read_bundle(self.root)

    def test_nonincreasing_timestamps_are_rejected(self):
        write_bundle(self.root)
        rows = [[0, 0.1, 0, 0, 0, 0], [1, 0.1, 0, 0, 0, 0]]
        with (self.root / "poses.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(POSE_FIELDS)
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            read_bundle(self.root)

    def test_wrong_image_size_is_rejected(self):
        write_bundle(self.root)
        plt.imsave(self.root / "calibration.png", np.zeros((2, 2, 3)))
        with self.assertRaisesRegex(ValueError, "image must"):
            read_bundle(self.root)

    def test_precision_sweep_error_decreases(self):
        self.root.mkdir(exist_ok=True)
        results = precision_sweep(demo_points(), self.root)
        errors = [row["max_abs_error_m"] for row in results]
        self.assertGreater(errors[0], errors[1])
        self.assertGreaterEqual(errors[1], errors[2])

    def test_precision_sweep_changes_one_factor(self):
        results = precision_sweep(demo_points(), self.root, decimals=(3, 6, 9))
        self.assertEqual([row["decimal_places"] for row in results], [3, 6, 9])
        self.assertTrue(all(row["file_bytes"] > 0 for row in results))

    def test_stable_logsumexp_handles_large_values(self):
        result = stable_logsumexp(np.array([1000.0, 999.0, 998.0]))
        self.assertTrue(math.isfinite(result))
        self.assertAlmostEqual(result, 1000.4076059644444, places=10)

    def test_stable_logsumexp_shift_identity(self):
        values = np.array([-2.0, 0.0, 3.0])
        self.assertAlmostEqual(stable_logsumexp(values + 7.0), stable_logsumexp(values) + 7.0)

    def test_chain_rule_matches_finite_difference(self):
        x, eps = 2.0, 1e-6
        _, analytic = chain_value_and_grad(x)
        plus, _ = chain_value_and_grad(x + eps)
        minus, _ = chain_value_and_grad(x - eps)
        self.assertAlmostEqual(analytic, (plus - minus) / (2 * eps), places=6)

    def test_binary_nll_is_stable_at_large_logits(self):
        good_loss, good_grad = binary_nll_and_grad(1000.0, 1)
        bad_loss, bad_grad = binary_nll_and_grad(-1000.0, 1)
        self.assertEqual(good_loss, 0.0)
        self.assertEqual(good_grad, 0.0)
        self.assertAlmostEqual(bad_loss, 1000.0)
        self.assertAlmostEqual(bad_grad, -1.0)

    def test_full_run_writes_strict_json_report(self):
        report = run(self.root)
        self.assertEqual(report["npy_roundtrip_max_abs_error_m"], 0.0)
        with (self.root / "report.json").open(encoding="utf-8") as handle:
            parsed = json.load(handle, parse_constant=lambda value: self.fail(value))
        self.assertEqual(parsed["shapes"]["points_world_m"], [8, 3])


if __name__ == "__main__":
    unittest.main()
