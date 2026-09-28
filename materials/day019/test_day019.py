from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "materials" / "day019"
BUNDLED_CMAKE = Path(
    "/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake"
)


def cmake_path() -> str:
    discovered = shutil.which("cmake")
    if discovered:
        return discovered
    if BUNDLED_CMAKE.is_file():
        return str(BUNDLED_CMAKE)
    raise unittest.SkipTest("CMake 3.20+ is not installed or discoverable")


def run(*args: str, cwd: Path | None = None, check: bool = True):
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd or ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


class Day019BlackBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="day019-test-")
        cls.build = Path(cls.temp.name) / "build"
        run(cmake_path(), "-S", SOURCE, "-B", cls.build, "-DCMAKE_BUILD_TYPE=Release")
        run(cmake_path(), "--build", cls.build, "--parallel", "2")
        cls.demo = cls.build / "divergence_demo"
        cls.output = run(cls.demo).stdout

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def metric(self, key: str) -> float:
        for line in self.output.splitlines():
            if line.startswith(key + "="):
                return float(line.split("=", 1)[1])
        self.fail(f"missing metric: {key}")

    def test_01_configure_cache_is_out_of_source(self):
        self.assertTrue((self.build / "CMakeCache.txt").is_file())
        self.assertFalse((SOURCE / "CMakeCache.txt").exists())

    def test_02_library_artifact_exists(self):
        candidates = list(self.build.rglob("*divergence*.a"))
        self.assertTrue(candidates)
        self.assertTrue(all(path.stat().st_size > 0 for path in candidates))

    def test_03_executable_exists(self):
        self.assertTrue(self.demo.is_file())
        self.assertTrue(os.access(self.demo, os.X_OK))

    def test_04_demo_success(self):
        self.assertIn("SUCCESS", self.output)
        self.assertIn("p_shape=(2) q_shape=(2)", self.output)

    def test_05_entropy(self):
        self.assertAlmostEqual(self.metric("entropy_p_bits"), 1.0, places=6)

    def test_06_cross_entropy(self):
        self.assertAlmostEqual(
            self.metric("cross_entropy_pq_bits"), 1.207519, places=6
        )

    def test_07_kl_direction_one(self):
        self.assertAlmostEqual(self.metric("kl_pq_bits"), 0.207519, places=6)

    def test_08_kl_direction_two(self):
        self.assertAlmostEqual(self.metric("kl_qp_bits"), 0.188722, places=6)

    def test_09_kl_is_asymmetric(self):
        self.assertGreater(self.metric("kl_asymmetry_bits"), 0.0)

    def test_10_cross_entropy_identity(self):
        self.assertLessEqual(self.metric("identity_error_bits"), 1e-12)

    def test_11_js_value(self):
        self.assertAlmostEqual(self.metric("js_bits"), 0.048795, places=6)

    def test_12_support_mismatch_is_infinite(self):
        self.assertIn("support_mismatch_kl_bits=inf", self.output)

    def test_13_self_test(self):
        self.assertIn("SELF_TEST_OK: 11 checks", run(self.demo, "--self-test").stdout)

    def test_14_ctest(self):
        result = run(cmake_path(), "--build", self.build, "--target", "test")
        combined = result.stdout + result.stderr
        self.assertIn("100% tests passed", combined)

    def test_15_csv_scan(self):
        csv_path = Path(self.temp.name) / "smoothing.csv"
        run(self.demo, "--csv", csv_path)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["kl_pq_bits"], "inf")
        finite_kl = [float(row["kl_pq_bits"]) for row in rows[1:]]
        finite_js = [float(row["js_bits"]) for row in rows]
        self.assertTrue(all(math.isfinite(value) for value in finite_kl + finite_js))
        self.assertTrue(all(a > b for a, b in zip(finite_kl, finite_kl[1:])))
        self.assertTrue(all(a > b for a, b in zip(finite_js, finite_js[1:])))

    def test_16_in_source_build_is_rejected(self):
        copy_root = Path(self.temp.name) / "in-source-copy"
        shutil.copytree(SOURCE, copy_root)
        result = run(cmake_path(), "-S", copy_root, "-B", copy_root, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("In-source builds are disabled", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
