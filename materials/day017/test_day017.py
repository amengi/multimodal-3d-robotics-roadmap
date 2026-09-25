"""Black-box regression tests for Day 017 materials."""

from __future__ import annotations

import csv
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "materials" / "day017" / "camera_raii_entropy.cpp"


class Day017Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="day017-")
        cls.temp_path = pathlib.Path(cls.temp_dir.name)
        cls.binary = cls.temp_path / "camera_raii_entropy"
        cls.report = cls.temp_path / "projection.csv"
        compiler = os.environ.get("CXX") or shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("No C++17 compiler found; set CXX")
        compile_result = subprocess.run(
            [compiler, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
             str(SOURCE), "-o", str(cls.binary)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if compile_result.returncode != 0:
            raise AssertionError(f"compile failed:\n{compile_result.stdout}\n{compile_result.stderr}")
        cls.normal = subprocess.run([str(cls.binary)], text=True, capture_output=True, check=False)
        cls.lines = cls.normal.stdout.splitlines()
        cls.report_run = subprocess.run(
            [str(cls.binary), "--write-report", str(cls.report)],
            text=True, capture_output=True, check=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def value(self, prefix: str) -> str:
        matches = [line.removeprefix(prefix) for line in self.lines if line.startswith(prefix)]
        self.assertEqual(len(matches), 1, f"expected one line beginning {prefix!r}")
        return matches[0]

    def test_01_binary_built(self) -> None:
        self.assertTrue(self.binary.is_file())

    def test_02_normal_run_succeeds(self) -> None:
        self.assertEqual(self.normal.returncode, 0, self.normal.stderr)

    def test_03_contract_has_shapes_and_frames(self) -> None:
        self.assertEqual(
            self.value("contract="),
            "points_world_m:(N,3),R_cw:(3,3),t_cw_m:(3,),pixels:(N,2)",
        )

    def test_04_unique_ownership_reported(self) -> None:
        self.assertIn("exclusive=1,manual_delete=0", self.value("ownership="))

    def test_05_known_projection(self) -> None:
        self.assertIn("pixel_px=(270.000000,190.000000)", self.value("projection:"))

    def test_06_joint_counts(self) -> None:
        self.assertEqual(self.value("joint_counts:"), "x0y0=3,x0y1=1,x1y0=1,x1y1=3")

    def test_07_marginal_entropies(self) -> None:
        entropy = self.value("entropy:")
        self.assertIn("h_x_bits=1.000000", entropy)
        self.assertIn("h_y_bits=1.000000", entropy)

    def test_08_joint_entropy(self) -> None:
        self.assertIn("h_xy_bits=1.811278", self.value("entropy:"))

    def test_09_conditional_entropy(self) -> None:
        self.assertIn("h_y_given_x_bits=0.811278", self.value("entropy:"))

    def test_10_chain_rule_zero_error(self) -> None:
        self.assertEqual(self.value("chain_rule:"), "error_bits=0.000000")

    def test_11_success_marker(self) -> None:
        self.assertIn(
            "SUCCESS: Camera invariants, RAII ownership, projection, and entropy validated",
            self.lines,
        )

    def test_12_internal_self_tests(self) -> None:
        result = subprocess.run([str(self.binary), "--self-test"], text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "SELF_TEST_OK: 16 checks")

    def test_13_report_run_succeeds(self) -> None:
        self.assertEqual(self.report_run.returncode, 0, self.report_run.stderr)
        self.assertIn("REPORT_OK:", self.report_run.stdout)

    def test_14_raii_scope_closes(self) -> None:
        self.assertIn("raii_scope:active_inside=1", self.report_run.stdout)
        self.assertIn("raii_scope:active_after=0", self.report_run.stdout)

    def test_15_report_has_eight_data_rows(self) -> None:
        with self.report.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
        self.assertEqual(len(rows), 8)

    def test_16_report_has_raii_footer(self) -> None:
        self.assertTrue(self.report.read_text(encoding="utf-8").endswith("# closed_by_raii\n"))

    def test_17_report_units_and_labels(self) -> None:
        header = self.report.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(
            header,
            "x_w_m,y_w_m,z_w_m,x_c_m,y_c_m,z_c_m,u_px,v_px,depth_bin,side",
        )

    def test_18_unknown_argument_rejected(self) -> None:
        result = subprocess.run([str(self.binary), "--unknown"], text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown or incomplete argument", result.stderr)

    def test_19_missing_report_path_rejected(self) -> None:
        result = subprocess.run([str(self.binary), "--write-report"], text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
