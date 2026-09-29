from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "materials" / "day020"
BUNDLED_CMAKE = Path(
    "/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake"
)
DEFAULT_EIGEN = Path("/tmp/day020-eigen")


def cmake_path() -> str:
    discovered = shutil.which("cmake")
    if discovered:
        return discovered
    if BUNDLED_CMAKE.is_file():
        return str(BUNDLED_CMAKE)
    raise unittest.SkipTest("CMake 3.20+ is not installed or discoverable")


def eigen_dir() -> Path:
    configured = os.environ.get("DAY020_EIGEN_DIR")
    candidate = Path(configured) if configured else DEFAULT_EIGEN
    if (candidate / "Eigen" / "Dense").is_file():
        return candidate
    raise unittest.SkipTest(
        "Eigen 3 headers missing; set DAY020_EIGEN_DIR to a directory containing Eigen/Dense"
    )


def run(*args: object, cwd: Path | None = None, check: bool = True):
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd or ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


class Day020BlackBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="day020-test-")
        cls.build = Path(cls.temp.name) / "build"
        run(
            cmake_path(),
            "-S",
            SOURCE,
            "-B",
            cls.build,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DEIGEN3_INCLUDE_DIR={eigen_dir()}",
        )
        run(cmake_path(), "--build", cls.build, "--parallel", "2")
        cls.demo = cls.build / "eigen_mi_demo"
        cls.output = run(cls.demo).stdout

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def metric(self, key: str) -> float:
        for line in self.output.splitlines():
            if line.startswith(key + "="):
                return float(line.split("=", 1)[1])
        self.fail(f"missing metric: {key}")

    def test_01_out_of_source_cache(self):
        self.assertTrue((self.build / "CMakeCache.txt").is_file())
        self.assertFalse((SOURCE / "CMakeCache.txt").exists())

    def test_02_static_library_exists(self):
        candidates = list(self.build.rglob("*day020_core*.a"))
        self.assertTrue(candidates)

    def test_03_demo_success(self):
        self.assertIn("SUCCESS", self.output)
        self.assertIn("design_shape=(5,2)", self.output)
        self.assertIn("top_block_shape=(3,2) top_block_sum=6.000000", self.output)

    def test_04_eigen_version(self):
        self.assertIn("eigen_version=3.4.0", self.output)

    def test_05_qr_slope(self):
        self.assertAlmostEqual(self.metric("qr_slope"), 1.99, places=6)

    def test_06_svd_slope(self):
        self.assertAlmostEqual(self.metric("svd_slope"), 1.99, places=6)

    def test_07_solver_agreement(self):
        self.assertLessEqual(self.metric("beta_max_abs_diff"), 1e-12)

    def test_08_rmse(self):
        self.assertAlmostEqual(self.metric("rmse_m"), 0.146287, places=6)

    def test_09_condition_number(self):
        self.assertAlmostEqual(self.metric("condition_number"), 4.738720, places=6)

    def test_10_entropy_foundation(self):
        self.assertAlmostEqual(self.metric("h_x_bits"), 1.0, places=6)
        self.assertAlmostEqual(self.metric("h_joint_bits"), 1.811278, places=6)
        self.assertAlmostEqual(self.metric("h_x_given_y_bits"), 0.811278, places=6)

    def test_11_three_mi_definitions(self):
        self.assertAlmostEqual(self.metric("mi_conditional_bits"), 0.188722, places=6)
        self.assertAlmostEqual(self.metric("mi_entropies_bits"), 0.188722, places=6)
        self.assertAlmostEqual(self.metric("mi_kl_bits"), 0.188722, places=6)
        self.assertLessEqual(self.metric("mi_three_way_error"), 1e-12)

    def test_12_supervised_baselines(self):
        self.assertAlmostEqual(self.metric("match_rule_accuracy"), 0.75, places=6)
        self.assertAlmostEqual(self.metric("majority_y_accuracy"), 0.5, places=6)

    def test_13_pairing_failure(self):
        self.assertAlmostEqual(self.metric("shifted_mi_bits"), 0.0, places=6)
        self.assertAlmostEqual(self.metric("shifted_match_accuracy"), 0.5, places=6)

    def test_14_program_self_test(self):
        self.assertIn("SELF_TEST_OK: 19 checks", run(self.demo, "--self-test").stdout)

    def test_15_ctest(self):
        combined = run(cmake_path(), "--build", self.build, "--target", "test")
        self.assertIn("100% tests passed", combined.stdout + combined.stderr)

    def test_16_csv_evidence(self):
        csv_path = Path(self.temp.name) / "pairing.csv"
        run(self.demo, "--csv", csv_path)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["case"] for row in rows], ["paired", "shift_y_by_2"])
        self.assertAlmostEqual(float(rows[0]["mi_kl_bits"]), 0.188721875541)
        self.assertAlmostEqual(float(rows[1]["mi_kl_bits"]), 0.0)

    def test_17_numpy_report(self):
        report_path = Path(self.temp.name) / "numpy.json"
        run("conda", "run", "-n", "pyt", "python", SOURCE / "verify_numpy.py", "--json", report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(report["slope"], self.metric("svd_slope"), places=6)
        self.assertAlmostEqual(
            report["paired"]["mi_kl_bits"], self.metric("mi_kl_bits"), places=6
        )

    def test_18_in_source_build_rejected(self):
        copy_root = Path(self.temp.name) / "in-source-copy"
        shutil.copytree(SOURCE, copy_root)
        result = run(
            cmake_path(),
            "-S",
            copy_root,
            "-B",
            copy_root,
            f"-DEIGEN3_INCLUDE_DIR={eigen_dir()}",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("In-source builds are disabled", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
