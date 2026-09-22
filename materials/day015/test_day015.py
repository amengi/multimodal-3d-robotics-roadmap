"""Black-box regression tests for the Day 015 C++ program."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "materials" / "day015" / "point_stats.cpp"


class Day015Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="day015-")
        cls.binary = pathlib.Path(cls.temp_dir.name) / "point_stats"
        compiler = os.environ.get("CXX") or shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("No C++ compiler found; set CXX to a C++17 compiler")
        completed = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                str(SOURCE),
                "-o",
                str(cls.binary),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(f"compile failed:\n{completed.stdout}\n{completed.stderr}")
        cls.normal = subprocess.run(
            [str(cls.binary)], text=True, capture_output=True, check=False
        )
        cls.lines = cls.normal.stdout.splitlines()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def value(self, prefix: str) -> str:
        matches = [line.removeprefix(prefix) for line in self.lines if line.startswith(prefix)]
        self.assertEqual(len(matches), 1, f"expected one line beginning {prefix!r}")
        return matches[0]

    def information_row(self, probability: str) -> dict[str, float]:
        prefix = f"self_info:p={probability},"
        tail = self.value(prefix)
        return {key: float(value) for key, value in (field.split("=") for field in tail.split(","))}

    def test_01_binary_was_built(self) -> None:
        self.assertTrue(self.binary.is_file())

    def test_02_normal_run_succeeds(self) -> None:
        self.assertEqual(self.normal.returncode, 0, self.normal.stderr)

    def test_03_contract_names_shape_frame_and_unit(self) -> None:
        self.assertEqual(self.value("contract="), "points_m:(N,2),frame=world,unit=m")

    def test_04_count(self) -> None:
        self.assertEqual(int(self.value("count=")), 4)

    def test_05_centroid(self) -> None:
        self.assertEqual(self.value("centroid_m="), "1.000000,1.000000")

    def test_06_aabb_minimum(self) -> None:
        self.assertEqual(self.value("aabb_min_m="), "-1.000000,-2.000000")

    def test_07_aabb_maximum(self) -> None:
        self.assertEqual(self.value("aabb_max_m="), "3.000000,4.000000")

    def test_08_probability_one_has_zero_bits(self) -> None:
        self.assertAlmostEqual(self.information_row("1.000000")["bits"], 0.0)

    def test_09_half_has_one_bit(self) -> None:
        self.assertAlmostEqual(self.information_row("0.500000")["bits"], 1.0)

    def test_10_quarter_has_two_bits(self) -> None:
        self.assertAlmostEqual(self.information_row("0.250000")["bits"], 2.0)

    def test_11_eighth_has_three_bits(self) -> None:
        self.assertAlmostEqual(self.information_row("0.125000")["bits"], 3.0)

    def test_12_half_has_ln_two_nats(self) -> None:
        self.assertAlmostEqual(self.information_row("0.500000")["nats"], 0.693147, places=6)

    def test_13_success_marker(self) -> None:
        self.assertIn("SUCCESS: C++ point statistics and self-information validated", self.lines)

    def test_14_internal_self_tests(self) -> None:
        completed = subprocess.run(
            [str(self.binary), "--self-test"], text=True, capture_output=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "SELF_TEST_OK: 10 checks")

    def test_15_unknown_argument_is_rejected(self) -> None:
        completed = subprocess.run(
            [str(self.binary), "--unknown"], text=True, capture_output=True, check=False
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unknown argument", completed.stderr)


if __name__ == "__main__":
    unittest.main()
