"""Black-box regression tests for Day 016 materials."""

from __future__ import annotations

import csv
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "materials" / "day016" / "lifetime_entropy.cpp"
PLOTTER = ROOT / "materials" / "day016" / "plot_entropy.py"


class Day016Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="day016-")
        cls.temp_path = pathlib.Path(cls.temp_dir.name)
        cls.binary = cls.temp_path / "lifetime_entropy"
        cls.csv_path = cls.temp_path / "entropy.csv"
        cls.svg_path = cls.temp_path / "entropy.svg"
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
        cls.csv_run = subprocess.run(
            [str(cls.binary), "--write-csv", str(cls.csv_path)],
            text=True, capture_output=True, check=False,
        )
        cls.plot_run = subprocess.run(
            [os.fspath(shutil.which("python3") or "python3"), str(PLOTTER),
             str(cls.csv_path), str(cls.svg_path)],
            text=True, capture_output=True, check=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def value(self, prefix: str) -> str:
        matches = [line.removeprefix(prefix) for line in self.lines if line.startswith(prefix)]
        self.assertEqual(len(matches), 1, f"expected one line beginning {prefix!r}")
        return matches[0]

    def entropy(self, probability: str) -> float:
        tail = self.value(f"entropy:p={probability},")
        return float(tail.removeprefix("bits="))

    def test_01_binary_built(self) -> None:
        self.assertTrue(self.binary.is_file())

    def test_02_normal_run_succeeds(self) -> None:
        self.assertEqual(self.normal.returncode, 0, self.normal.stderr)

    def test_03_contract(self) -> None:
        self.assertEqual(self.value("contract="), "ranges_m:(N,),frame=sensor,unit=m,N>0,finite,nonnegative")

    def test_04_by_value_copies_four_elements(self) -> None:
        self.assertIn("by_value_copies=4", self.value("copy_demo:"))

    def test_05_const_reference_copies_none(self) -> None:
        self.assertIn("by_const_ref_copies=0", self.value("copy_demo:"))

    def test_06_both_sums_match(self) -> None:
        self.assertIn("sum_value=4.400000,sum_ref=4.400000", self.value("copy_demo:"))

    def test_07_alias_changes_original(self) -> None:
        self.assertEqual(self.value("alias_demo:"), "original_m=2.500000,alias_m=2.500000")

    def test_08_pointer_observes_element(self) -> None:
        self.assertIn("index=2,value_m=1.200000", self.value("pointer_demo:"))

    def test_09_pointer_mean(self) -> None:
        self.assertIn("mean_m=1.100000", self.value("pointer_demo:"))

    def test_10_entropy_endpoints_zero(self) -> None:
        self.assertAlmostEqual(self.entropy("0.000000"), 0.0)
        self.assertAlmostEqual(self.entropy("1.000000"), 0.0)

    def test_11_entropy_fair_coin_one_bit(self) -> None:
        self.assertAlmostEqual(self.entropy("0.500000"), 1.0)

    def test_12_entropy_is_symmetric(self) -> None:
        self.assertAlmostEqual(self.entropy("0.250000"), self.entropy("0.750000"))

    def test_13_success_marker(self) -> None:
        self.assertIn("SUCCESS: references, observers, and Bernoulli entropy validated", self.lines)

    def test_14_internal_self_tests(self) -> None:
        result = subprocess.run([str(self.binary), "--self-test"], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "SELF_TEST_OK: 14 checks")

    def test_15_csv_has_expected_rows(self) -> None:
        self.assertEqual(self.csv_run.returncode, 0, self.csv_run.stderr)
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[5], {"p": "0.500000", "entropy_bits": "1.000000"})

    def test_16_svg_created_and_labelled(self) -> None:
        self.assertEqual(self.plot_run.returncode, 0, self.plot_run.stderr)
        svg = self.svg_path.read_text(encoding="utf-8")
        self.assertIn("Bernoulli entropy", svg)
        self.assertIn("entropy (bit)", svg)

    def test_17_unknown_argument_rejected(self) -> None:
        result = subprocess.run([str(self.binary), "--unknown"], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown or incomplete argument", result.stderr)

    def test_18_missing_csv_argument_rejected(self) -> None:
        result = subprocess.run([str(self.binary), "--write-csv"], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
