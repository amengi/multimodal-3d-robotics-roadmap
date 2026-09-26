from __future__ import annotations

import csv
import contextlib
import io
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from materials.day018.verify_cross_entropy import (
    loss_scan,
    manual_cross_entropy_nats,
    multiclass_brier,
    self_test,
    write_scan,
)


ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ROOT / "materials" / "day018"


class CompileLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("c++ compiler not found")
        cls.compiler = compiler
        cls.temp = tempfile.TemporaryDirectory(prefix="day018-test-")
        cls.build = Path(cls.temp.name)
        cls.camera_object = cls.build / "camera.o"
        cls.main_object = cls.build / "main.o"
        cls.app = cls.build / "camera_app"
        common = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
        subprocess.run(common + ["-c", str(MATERIALS / "camera.cpp"), "-o", str(cls.camera_object)], check=True)
        subprocess.run(common + ["-c", str(MATERIALS / "main.cpp"), "-o", str(cls.main_object)], check=True)
        subprocess.run([compiler, str(cls.main_object), str(cls.camera_object), "-o", str(cls.app)], check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def test_01_two_object_files_exist(self) -> None:
        self.assertGreater(self.camera_object.stat().st_size, 0)
        self.assertGreater(self.main_object.stat().st_size, 0)

    def test_02_linked_executable_exists(self) -> None:
        self.assertGreater(self.app.stat().st_size, 0)

    def test_03_known_projection(self) -> None:
        run = subprocess.run([self.app], check=True, text=True, capture_output=True)
        self.assertIn("projection_px=(270.000000,190.000000)", run.stdout)

    def test_04_geometry_error_is_zero(self) -> None:
        run = subprocess.run([self.app], check=True, text=True, capture_output=True)
        self.assertIn("geometry_error_px=0.000000", run.stdout)

    def test_05_program_success_marker(self) -> None:
        run = subprocess.run([self.app], check=True, text=True, capture_output=True)
        self.assertTrue(run.stdout.rstrip().endswith("SUCCESS"))

    def test_06_cpp_self_test(self) -> None:
        run = subprocess.run([self.app, "--self-test"], check=True, text=True, capture_output=True)
        self.assertIn("SELF_TEST_OK: 6 checks", run.stdout)

    def test_07_missing_definition_causes_link_failure(self) -> None:
        broken = self.build / "broken_app"
        run = subprocess.run(
            [self.compiler, str(self.main_object), "-o", str(broken)],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertRegex((run.stdout + run.stderr).lower(), r"undefined|unresolved")


class CrossEntropyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor([2.0, 1.0, 0.1], dtype=torch.float64)
        self.target = 0

    def test_08_probabilities_sum_to_one(self) -> None:
        probabilities = torch.softmax(self.logits, dim=0)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)

    def test_09_manual_matches_negative_log_true_probability(self) -> None:
        probabilities = torch.softmax(self.logits, dim=0)
        self.assertAlmostEqual(
            manual_cross_entropy_nats(self.logits, self.target),
            -math.log(float(probabilities[self.target])),
            places=12,
        )

    def test_10_manual_matches_torch(self) -> None:
        expected = float(F.cross_entropy(self.logits.unsqueeze(0), torch.tensor([self.target])))
        self.assertAlmostEqual(manual_cross_entropy_nats(self.logits, self.target), expected, places=12)

    def test_11_brier_is_bounded(self) -> None:
        value = multiclass_brier(torch.softmax(self.logits, dim=0), self.target)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 2.0)

    def test_12_invalid_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            manual_cross_entropy_nats(torch.ones((1, 3), dtype=torch.float64), 0)

    def test_13_invalid_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target"):
            manual_cross_entropy_nats(self.logits, 3)

    def test_14_loss_decreases_with_true_logit(self) -> None:
        rows = loss_scan([-2.0, 0.0, 2.0, 4.0])
        losses = [row["nll_nats"] for row in rows]
        self.assertTrue(all(left > right for left, right in zip(losses, losses[1:])))

    def test_15_cli_writes_scan_csv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="day018-csv-") as directory:
            output = Path(directory) / "scan.csv"
            write_scan(output, loss_scan([-2.0, 0.0, 2.0, 4.0]))
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)

    def test_16_python_self_test(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self_test()
        self.assertIn("CROSS_ENTROPY_SELF_TEST_OK: 7 checks", output.getvalue())


if __name__ == "__main__":
    unittest.main()
