"""Re-evaluate both frozen worlds on the same cameras; retain raw cam04 pixels."""
from __future__ import annotations

import gc
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

from config_utils import load_config_file
from paper_training_protocol import resolve_paper_training_protocol
from research_experiments.paper_runner_suite.run_unified_paper_ablation import (
    live_resource_snapshot, require_live_resources, run_checked_with_peak_rss,
    source_provenance, wandb_file_identity,
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(config_path, *, worker=False, output_dir=None):
    cfg = load_config_file(config_path)
    base = load_config_file(cfg["sampling_config"])
    variant = next(v for v in base["variants"] if v["name"] == cfg["variant"])
    raw_protocol = load_config_file(variant["protocol"])
    protocol = resolve_paper_training_protocol(raw_protocol)
    out = (Path(output_dir) if output_dir else Path(cfg["output_dir"]) / "evaluation").resolve()
    runs = {"two_camera": Path(cfg["reference_run"]),
        "cam04_only": Path(cfg["output_dir"]) / "robust_l1"}
    if not worker:
        if out.exists():
            raise FileExistsError(f"Preserve previous evaluation: {out}")
        snapshot = live_resource_snapshot()
        require_live_resources(snapshot, protocol)
        out.mkdir(parents=True)
        write(out / "preflight.json", snapshot)
        paths = set(read(runs["cam04_only"] / "source_identity.json")["bound_files"])
        paths.update([str(Path(__file__).resolve().relative_to(Path.cwd())), str(Path(config_path)),
            "src/train/paper_local_resources.py", "src/train/losses.py", "src/train/perceptual_metrics.py"])
        write(out / "source_identity.json", {**source_provenance(), "bound_files": {p: sha(p) for p in paths}})
        def timeout(signum, frame):
            raise TimeoutError("unchanged 600-second evaluation limit")
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(base["timeout_seconds"])
        receipt = run_checked_with_peak_rss([sys.executable, "-u", str(Path(__file__).resolve()),
            config_path, "--worker", "--output-dir", str(out)], cwd=Path.cwd(),
            rss_limit_bytes=protocol.local_resources["process_tree_rss_limit_bytes"],
            protocol=protocol, output_root=out)
        signal.alarm(0)
        write(out / "resource_receipt.json", receipt)
        return

    os.environ.update(STAR_UVT_TILE_CAPACITY=str(variant["tile_capacity"]),
        STAR_UVT_TILE_T=str(variant["tile_t"]), WANDB_MODE="offline", WANDB_DIR=str(out.resolve()))
    import torch
    import wandb
    from star_uvt_runtime import ensure_star_uvt_on_path
    from paper_local_resources import configure_local_mps
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from research_project.benchmarks import multicam_heldout_compare as c
    from torch_gsplat_bridge_star_uvt import UVTRenderConfig
    configure_local_mps(raw_protocol, "mps")
    torch.set_num_threads(2)
    meta = read(runs["two_camera"] / "world_tubes/run_meta.json")
    baseline = load_config_file(meta["baseline_config"])
    bundle = c.load_multicam_video_bundle(data_cfg=meta["config_data"],
        camera_cfg={**baseline["camera"], "rig_init": "neural_3d_video"},
        target_size=protocol.final_stage.image_size.as_list(), device=torch.device("mps"),
        frame_device=torch.device("cpu"), defer_video_frames=True)
    providers = {split: c.PaperMulticamTargetProvider(getattr(bundle, split + "_frame_sources"),
        cache_capacity_frames=8) for split in ("train", "heldout")}
    dataset = c.paper_dataset_bundle_identity(bundle, image_size=protocol.final_stage.image_size,
        decoded_frame_identities={split + "_frames": provider.tensor_content_identity(chunk_frames=16)
            for split, provider in providers.items()})
    assert dataset == meta["paper_dataset_bundle"]
    render = UVTRenderConfig(height=96, width=128, frames=32,
        tile_capacity=variant["tile_capacity"], tile_t=variant["tile_t"])
    rows = {}
    for name, folder in runs.items():
        original = read(folder / "world_tubes/comparison_report.json")
        assert original["meta"]["paper_dataset_bundle"] == dataset
        assert original["meta"]["paper_evaluator"] == c.paper_evaluator_contract()
        native = original["meta"]["star_uvt_native_extension"]
        assert sha(native["path"]) == native["sha256"]
        checkpoint = original["star_uvt"]["final_world_checkpoint"]
        model, loaded = c._load_frozen_world_checkpoint(Path(checkpoint["path"]), device=torch.device("mps"),
            expected_file_sha256=checkpoint["sha256"], expected_world_state_sha256=checkpoint["world_state_sha256"],
            expected_full_frames=32)
        result = c.eval_world_tubes(model, bundle, backend="metal_tile", camera_projection="dataset_lens",
            camera_sequence_mode="static_view", segment_frames=4, synthetic_pan_x=0, synthetic_pan_y=0,
            synthetic_dolly_z=0, synthetic_zoom=0, synthetic_principal_x=0, synthetic_principal_y=0,
            render_config=render, chunk_frames=2, media_max_frames=32,
            train_target_provider=providers["train"], heldout_target_provider=providers["heldout"])
        raw_path = out / (name + "_cam04.pt")
        target, rendered = result["train_rows"][0]
        torch.save({"target": target, "prediction": rendered.rgb}, raw_path)
        for split in ("train", "heldout"):
            c.save_first_row_media(out, name + "_" + split, result[split + "_rows"], fps=30)
        stats = c.world_tube_metal_stats(model, bundle, camera_projection="dataset_lens",
            camera_sequence_mode="static_view", segment_frames=4, render_config=render, chunk_frames=2)
        assert all(row["stats"]["overflow_tile_count"] == 0 for row in stats["rows"])
        rows[name] = {"checkpoint": loaded, "metrics": result["metrics"],
            "per_camera": {split: dict(zip(getattr(bundle, split + "_camera_names"),
                result[split + "_view_metrics"], strict=True)) for split in ("train", "heldout")},
            "metal_stats": stats, "raw_cam04": {"path": str(raw_path), "sha256": sha(raw_path)},
            "source_report": str(folder / "world_tubes/comparison_report.json"),
            "source_report_sha256": sha(folder / "world_tubes/comparison_report.json")}
        del model, result, target, rendered
        gc.collect()
        torch.mps.empty_cache()
    report = {"scope": "fixed final checkpoints; cam04-only vs two-camera optimization with shared two-camera initialization",
        "publication_eligible": False, "rows": rows, "dataset_identity": dataset,
        "evaluator": c.paper_evaluator_contract(), "native_library_sha256": native["sha256"],
        "source": read(out / "source_identity.json")}
    write(out / "report.json", report)
    run = wandb.init(project="dynaworld", name="world-tube-single-camera-control-20260913", mode="offline",
        dir=str(out), config={"config": cfg, "report_sha256": sha(out / "report.json"), "source": report["source"]},
        tags=["world_tubes", "diagnostic", "single_camera", "frozen_checkpoint"])
    run.log({f"{name}/{camera}/{metric}": value for name, row in rows.items()
        for split in row["per_camera"].values() for camera, metrics in split.items() for metric, value in metrics.items()})
    for path in out.glob("*.png"):
        run.log({path.stem: wandb.Image(str(path))})
    for path in out.glob("*.mp4"):
        run.log({path.stem: wandb.Video(str(path), format="mp4")})
    run.save(str(out / "report.json"), base_path=str(out))
    run_id, run_dir = run.id, run.dir
    run.finish()
    write(out / "wandb_identity.json", {"run_id": run_id, "mode": "offline", "finish_called": True,
        "report_sha256": sha(out / "report.json"), "run_file": wandb_file_identity(run_dir, run_id)})
    print(json.dumps({name: row["per_camera"] for name, row in rows.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    main(args.config, worker=args.worker, output_dir=args.output_dir)
