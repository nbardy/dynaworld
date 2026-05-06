from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return data


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def powerfoam_row(output_dir: Path, *, label: str) -> dict[str, Any]:
    best = load_json(output_dir / "best_metrics.json")
    config = load_json(output_dir / "resolved_config.json")
    metrics = best["metrics"]
    calibration = str(config.get("render", {}).get("eval_color_calibration", "none"))
    raw_psnr = metric(metrics, "uncalibrated_heldout_eval_psnr")
    raw_ssim = metric(metrics, "uncalibrated_heldout_eval_ssim")
    raw_l1 = metric(metrics, "uncalibrated_heldout_eval_l1")
    if calibration == "none":
        raw_psnr = metric(metrics, "heldout_eval_psnr")
        raw_ssim = metric(metrics, "heldout_eval_ssim")
        raw_l1 = metric(metrics, "heldout_eval_l1")
    return {
        "label": label,
        "representation": "powerfoam_metal",
        "metric_semantics": "raw" if calibration == "none" else "calibrated_with_raw_disclosed",
        "output_dir": rel(output_dir),
        "config": rel(output_dir / "resolved_config.json"),
        "step": int(best["step"]),
        "steps": int(config["train"]["steps"]),
        "render_size": int(config["render"]["render_size"]),
        "primitive_count": int(config["model"]["cells"]),
        "train_frames": int(config["data"]["max_frames"]),
        "eval_psnr": metric(metrics, "eval_psnr"),
        "eval_ssim": metric(metrics, "eval_ssim"),
        "eval_l1": metric(metrics, "eval_l1"),
        "heldout_eval_psnr": metric(metrics, "heldout_eval_psnr"),
        "heldout_eval_ssim": metric(metrics, "heldout_eval_ssim"),
        "heldout_eval_l1": metric(metrics, "heldout_eval_l1"),
        "raw_heldout_eval_psnr": raw_psnr,
        "raw_heldout_eval_ssim": raw_ssim,
        "raw_heldout_eval_l1": raw_l1,
        "eval_color_calibration": calibration,
        "renderer": config["render"].get("renderer"),
        "camera_projection": config["render"].get("camera_projection"),
        "lens_model": config["camera"].get("lens_model"),
    }


def splat_row(output_dir: Path, *, config_path: Path) -> dict[str, Any]:
    metrics = load_json(output_dir / "metrics.json")
    config = load_json(output_dir / "config.json")
    return {
        "label": "matched_free_dynamic_3dgs",
        "representation": "free_dynamic_3dgs",
        "metric_semantics": "raw",
        "output_dir": rel(output_dir),
        "config": rel(config_path),
        "steps": int(config["train"]["steps"]),
        "render_size": int(config["render"]["render_size"]),
        "primitive_count": int(config["model"]["num_splats"]),
        "train_frames": int(config["data"]["max_frames"]),
        "eval_psnr": metric(metrics, "eval_psnr"),
        "eval_ssim": metric(metrics, "eval_ssim"),
        "eval_l1": metric(metrics, "eval_l1"),
        "heldout_eval_psnr": metric(metrics, "heldout_eval_psnr"),
        "heldout_eval_ssim": metric(metrics, "heldout_eval_ssim"),
        "heldout_eval_l1": metric(metrics, "heldout_eval_l1"),
        "raw_heldout_eval_psnr": metric(metrics, "heldout_eval_psnr"),
        "raw_heldout_eval_ssim": metric(metrics, "heldout_eval_ssim"),
        "raw_heldout_eval_l1": metric(metrics, "heldout_eval_l1"),
        "train_loop_elapsed_s": metric(metrics, "train_loop_elapsed_s"),
        "renderer": config["render"].get("renderer"),
        "camera_projection": config["render"].get("camera_projection"),
        "lens_model": config["camera"].get("lens_model"),
        "heldout_pose_is_calibrated": metric(metrics, "heldout_pose_is_calibrated"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare nearest0040 PowerFoam and matched free-dynamic 3DGS rows.")
    parser.add_argument("--powerfoam-raw-output", required=True)
    parser.add_argument("--powerfoam-calibrated-output", required=True)
    parser.add_argument("--splat-output", required=True)
    parser.add_argument("--splat-config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    powerfoam_raw_output = ROOT / args.powerfoam_raw_output
    powerfoam_calibrated_output = ROOT / args.powerfoam_calibrated_output
    splat_output = ROOT / args.splat_output
    splat_config = ROOT / args.splat_config
    rows = [
        powerfoam_row(powerfoam_raw_output, label="powerfoam_raw_nearest0040"),
        powerfoam_row(powerfoam_calibrated_output, label="powerfoam_eval_rgb_calibrated_nearest0040"),
        splat_row(splat_output, config_path=splat_config),
    ]
    report = {
        "comparison": "nearest0040_8cam_holdout0040_powerfoam_vs_free_dynamic_3dgs",
        "rows": rows,
        "caveats": [
            "PowerFoam raw/calibrated rows use OPENCV_FISHEYE; the current splat trainer materializes pinhole CameraSpec objects.",
            "The splat row uses fast_mac for local runtime; older gauge-field splat baselines used the dense PyTorch renderer.",
            "The calibrated PowerFoam row is not raw representation quality; compare raw_heldout_eval_* fields when judging raw PowerFoam.",
        ],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
