from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUDA_SUMMARY = ROOT / "outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json"
DEFAULT_METAL_OUTPUT = (
    ROOT / "outputs/powerfoam_metal/local_mac_powerfoam_metal_cuda_micro_match_randominit_64_4f_256cells_5step"
)
DEFAULT_OUTPUT = ROOT / "outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/cuda_vs_metal_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run_by_name(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for run in summary.get("runs", []):
        if isinstance(run, dict) and run.get("name") == name:
            return run
    raise KeyError(f"CUDA summary has no run named {name!r}")


def cuda_lane(summary: dict[str, Any], name: str) -> dict[str, Any]:
    run = run_by_name(summary, name)
    metrics = run.get("metrics", {})
    return {
        "name": name,
        "status": run.get("status"),
        "eval": metrics.get("eval", {}),
        "warm_timing_excluding_step0": metrics.get("warm_timing_excluding_step0", {}),
        "timing": metrics.get("timing", {}),
        "dynamic": metrics.get("dynamic", {}),
        "model": metrics.get("model", {}),
    }


def metal_lane(metal_output: Path) -> dict[str, Any]:
    best = load_json(metal_output / "best_metrics.json")
    resolved = load_json(metal_output / "resolved_config.json")
    train_history = read_jsonl(metal_output / "train_metrics_history.jsonl")
    metrics = best.get("metrics", {})
    return {
        "name": "powerfoam_metal_micro_match",
        "status": "ok",
        "best_step": best.get("step"),
        "best_metric_name": best.get("best_metric_name"),
        "eval": {
            key: metrics.get(key)
            for key in ("eval_l1", "eval_mse", "eval_psnr", "eval_ssim")
            if key in metrics
        },
        "train_elapsed_s": None if not train_history else train_history[-1].get("elapsed_s"),
        "model": {
            "points": resolved.get("model", {}).get("cells"),
            "num_texel_sites": resolved.get("model", {}).get("num_texel_sites"),
            "sv_dof": resolved.get("model", {}).get("sv_dof"),
        },
        "settings": {
            "video_path": resolved.get("data", {}).get("video_path"),
            "frames": resolved.get("data", {}).get("max_frames"),
            "size": resolved.get("render", {}).get("render_size"),
            "background": resolved.get("render", {}).get("background"),
            "iterations": resolved.get("train", {}).get("steps"),
            "adjacency_mode": resolved.get("model", {}).get("adjacency_mode"),
            "init_from_video": resolved.get("model", {}).get("init_from_video"),
            "color_init_mode": resolved.get("model", {}).get("color_init_mode"),
            "render_backend": "raytrace" if resolved.get("render", {}).get("use_raytrace") else "other",
            "device": resolved.get("train", {}).get("device"),
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def diff_eval(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    a_eval = a.get("eval", {})
    b_eval = b.get("eval", {})
    diffs: dict[str, float | None] = {}
    for key in ("eval_psnr", "eval_ssim", "eval_l1", "eval_mse"):
        if key in a_eval and key in b_eval and a_eval[key] is not None and b_eval[key] is not None:
            diffs[f"{a['name']}_minus_{b['name']}_{key}"] = float(a_eval[key]) - float(b_eval[key])
        else:
            diffs[f"{a['name']}_minus_{b['name']}_{key}"] = None
    return diffs


def matched_contract(cuda_summary: dict[str, Any], metal: dict[str, Any]) -> dict[str, Any]:
    cuda_clip = cuda_summary.get("clip", {})
    cuda_settings = cuda_summary.get("settings", {})
    metal_settings = metal.get("settings", {})
    metal_background = metal_settings.get("background")
    metal_black_background = (
        isinstance(metal_background, list)
        and len(metal_background) == 3
        and all(float(channel) == 0.0 for channel in metal_background)
    )
    return {
        "same_source_clip": metal_settings.get("video_path") == cuda_clip.get("path"),
        "same_frame_count": metal_settings.get("frames") == cuda_clip.get("frames"),
        "same_render_size": metal_settings.get("size") == cuda_clip.get("size"),
        "same_step_count": metal_settings.get("iterations") == cuda_settings.get("iterations"),
        "same_point_count": metal.get("model", {}).get("points") == cuda_settings.get("points"),
        "same_texel_site_count": metal.get("model", {}).get("num_texel_sites")
        == cuda_settings.get("num_texel_sites"),
        "same_sv_dof": metal.get("model", {}).get("sv_dof") == cuda_settings.get("sv_dof"),
        "metal_random_init": metal_settings.get("init_from_video") is False
        and metal_settings.get("color_init_mode") == "random",
        "same_fixed_black_background": bool(cuda_settings.get("fixed_black_background")) and metal_black_background,
    }


def build_report(cuda_summary: dict[str, Any], metal: dict[str, Any], *, cuda_path: Path, metal_output: Path) -> dict[str, Any]:
    official = cuda_lane(cuda_summary, "official_static_cuda")
    dynamic = cuda_lane(cuda_summary, "dynamic_feature_foam_cuda")
    contract = matched_contract(cuda_summary, metal)
    return {
        "schema_version": "powerfoam_cuda_metal_smoke_comparison_v1",
        "status": "ok" if all(bool(value) for value in contract.values()) else "mismatch",
        "purpose": "Smoke-scale comparison only; not a paper-quality or held-out validation result.",
        "sources": {
            "cuda_summary": rel(cuda_path),
            "metal_output": rel(metal_output),
        },
        "matched_contract": contract,
        "lanes": {
            "official_static_cuda": official,
            "dynamic_feature_foam_cuda": dynamic,
            "powerfoam_metal_micro_match": metal,
        },
        "comparisons": {
            **diff_eval(dynamic, official),
            **diff_eval(metal, official),
            **diff_eval(metal, dynamic),
        },
        "caveats": [
            "CUDA lanes run official upstream PowerFoam/Warp on Modal L40S; Metal lane runs the local Metal trainer on MPS.",
            "The matched contract checks clip, frame count, render size, step count, point/cell count, texel-site count, SV DoF, random init, and fixed black background.",
            "The dynamic CUDA fork is a time-conditioned texel_sv_rgb residual, not a true F32 Warp feature-accumulation kernel fork.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a smoke-scale PowerFoam CUDA-vs-Metal comparison JSON.")
    parser.add_argument("--cuda-summary", type=Path, default=DEFAULT_CUDA_SUMMARY)
    parser.add_argument("--metal-output", type=Path, default=DEFAULT_METAL_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cuda_summary = load_json(args.cuda_summary)
    metal = metal_lane(args.metal_output)
    report = build_report(cuda_summary, metal, cuda_path=args.cuda_summary, metal_output=args.metal_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": rel(args.output), "status": report["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
