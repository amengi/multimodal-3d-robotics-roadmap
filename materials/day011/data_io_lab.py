"""Validated data I/O plus small gradient and log-sum-exp demonstrations.

Run from the repository root:
    python -m materials.day011.data_io_lab --output /tmp/day011-output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POSE_FIELDS = [
    "frame_id",
    "timestamp_s",
    "tx_m",
    "ty_m",
    "tz_m",
    "yaw_rad",
]


def _finite_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the schema and return the unchanged mapping."""
    required = {"schema_version", "camera", "point_cloud", "pose"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")

    camera = config["camera"]
    if camera.get("frame") != "camera_optical":
        raise ValueError("camera.frame must be camera_optical")
    width = camera.get("width_px")
    height = camera.get("height_px")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("camera image size must be positive integers in pixels")
    k = _finite_array(camera.get("K_px"), "camera.K_px", (3, 3))
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0.0, 0.0, 1.0]):
        raise ValueError("K_px must have positive focal lengths and last row [0, 0, 1]")

    if config["point_cloud"] != {"frame": "world", "unit": "m", "shape": "N,3"}:
        raise ValueError("point_cloud metadata must declare world frame, metres, and N,3")
    if config["pose"] != {"convention": "T_world_camera", "translation_unit": "m", "yaw_unit": "rad"}:
        raise ValueError("pose metadata must declare T_world_camera, m, and rad")
    return config


def demo_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "camera": {
            "frame": "camera_optical",
            "width_px": 8,
            "height_px": 6,
            "K_px": [[120.0, 0.0, 3.5], [0.0, 120.0, 2.5], [0.0, 0.0, 1.0]],
        },
        "point_cloud": {"frame": "world", "unit": "m", "shape": "N,3"},
        "pose": {"convention": "T_world_camera", "translation_unit": "m", "yaw_unit": "rad"},
    }


def demo_points() -> np.ndarray:
    """Eight cuboid corners in world coordinates, metres."""
    return np.array(
        [[x, y, z] for x in (0.0, 1.23456789) for y in (-0.5, 0.5) for z in (2.0, 2.75)],
        dtype=np.float64,
    )


def demo_image(height: int = 6, width: int = 8) -> np.ndarray:
    """Small RGB calibration pattern with shape (H,W,3), values in [0,1]."""
    yy, xx = np.mgrid[:height, :width]
    return np.stack(
        [xx / max(width - 1, 1), yy / max(height - 1, 1), (xx + yy) % 2], axis=-1
    ).astype(np.float32)


def write_bundle(output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    config = validate_config(demo_config())
    paths = {
        "config": output / "config.json",
        "poses": output / "poses.csv",
        "points": output / "points.npy",
        "image": output / "calibration.png",
    }
    with paths["config"].open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    rows = [
        [0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1, 0.1, 0.1, -0.02, 0.0, math.radians(5.0)],
        [2, 0.2, 0.2, -0.03, 0.01, math.radians(10.0)],
    ]
    with paths["poses"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(POSE_FIELDS)
        writer.writerows(rows)

    np.save(paths["points"], demo_points(), allow_pickle=False)
    plt.imsave(paths["image"], demo_image())
    return paths


def read_bundle(output: Path) -> dict[str, Any]:
    with (output / "config.json").open(encoding="utf-8") as handle:
        config = validate_config(json.load(handle))

    with (output / "poses.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != POSE_FIELDS:
            raise ValueError(f"poses.csv header must be {POSE_FIELDS}")
        poses = list(reader)
    if not poses:
        raise ValueError("poses.csv must contain at least one data row")
    numeric = np.array(
        [[float(row[field]) for field in POSE_FIELDS[1:]] for row in poses], dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("poses.csv contains NaN or infinity")
    if not np.all(np.diff(numeric[:, 0]) > 0):
        raise ValueError("pose timestamps must be strictly increasing")

    points = np.load(output / "points.npy", allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points.npy must be finite with shape (N,3)")
    if points.dtype != np.float64:
        raise ValueError("points.npy must use float64 metres")

    image = plt.imread(output / "calibration.png")
    expected_hw = (config["camera"]["height_px"], config["camera"]["width_px"])
    if image.ndim != 3 or image.shape[:2] != expected_hw or image.shape[2] not in (3, 4):
        raise ValueError(f"image must have H,W={expected_hw} and RGB/RGBA channels")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("decoded PNG values must be finite and normalized to [0,1]")
    return {"config": config, "poses": numeric, "points": points, "image": image[..., :3]}


def precision_sweep(points: np.ndarray, output: Path, decimals: Iterable[int] = (3, 6, 9)) -> list[dict[str, Any]]:
    """Change only CSV decimal precision; report error and file size."""
    results: list[dict[str, Any]] = []
    for digits in decimals:
        path = output / f"points_{digits}dp.csv"
        np.savetxt(path, points, delimiter=",", fmt=f"%.{digits}f", header="x_m,y_m,z_m", comments="")
        restored = np.loadtxt(path, delimiter=",", skiprows=1)
        results.append(
            {
                "decimal_places": digits,
                "max_abs_error_m": float(np.max(np.abs(restored - points))),
                "file_bytes": path.stat().st_size,
            }
        )
    return results


def stable_logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite 1-D array")
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def chain_value_and_grad(x: float) -> tuple[float, float]:
    """For u=3x-1 and f=u^2, return f and df/dx=2u*3."""
    u = 3.0 * x - 1.0
    return u * u, 6.0 * u


def binary_nll_and_grad(logit: float, target: int) -> tuple[float, float]:
    if target not in (0, 1) or not math.isfinite(logit):
        raise ValueError("target must be 0/1 and logit finite")
    loss = float(np.logaddexp(0.0, logit) - target * logit)
    if logit >= 0:
        probability = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        probability = exp_logit / (1.0 + exp_logit)
    return loss, probability - target


def run(output: Path) -> dict[str, Any]:
    paths = write_bundle(output)
    loaded = read_bundle(output)
    sweep = precision_sweep(loaded["points"], output)
    f_value, analytic_grad = chain_value_and_grad(2.0)
    epsilon = 1e-6
    plus, _ = chain_value_and_grad(2.0 + epsilon)
    minus, _ = chain_value_and_grad(2.0 - epsilon)
    numerical_grad = (plus - minus) / (2.0 * epsilon)
    large_logits = np.array([1000.0, 999.0, 998.0])
    stable_lse = stable_logsumexp(large_logits)
    nll, nll_grad = binary_nll_and_grad(1000.0, 1)

    report = {
        "files": {name: path.name for name, path in paths.items()},
        "shapes": {
            "poses_numeric": list(loaded["poses"].shape),
            "points_world_m": list(loaded["points"].shape),
            "image_rgb": list(loaded["image"].shape),
        },
        "npy_roundtrip_max_abs_error_m": float(np.max(np.abs(loaded["points"] - demo_points()))),
        "csv_precision_sweep": sweep,
        "chain_rule": {
            "x": 2.0,
            "f": f_value,
            "analytic_grad": analytic_grad,
            "numerical_grad": numerical_grad,
            "absolute_difference": abs(analytic_grad - numerical_grad),
        },
        "numerical_stability": {
            "stable_logsumexp": stable_lse,
            "binary_nll_logit_1000_target_1": nll,
            "binary_nll_gradient": nll_grad,
        },
    }
    report_path = output / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("materials/day011/output"))
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("SUCCESS: all files reloaded and validated")


if __name__ == "__main__":
    main()
