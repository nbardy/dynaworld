"""Execution-only Metal batching control; run with the retained guarded launcher."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys

from config_utils import load_config_file


def main(config_path: str) -> None:
    cfg = load_config_file(config_path)
    frozen = load_config_file(cfg["frozen_config"])
    capacity = json.loads(Path(cfg["capacity_report"]).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    training = json.loads((Path(frozen["source_run"]) / "comparison_report.json").read_text())["star_uvt"]
    meta = json.loads((Path(frozen["source_run"]) / "run_meta.json").read_text())
    identity = training["final_world_checkpoint"]
    assert identity["sha256"] == frozen["expected_checkpoint_sha256"]
    assert identity["world_state_sha256"] == frozen["expected_world_state_sha256"]
    assert training["steps"] == 800 and training["stopped_reason"] is None
    assert cfg["temporal_tile_size"] == training["tile_t"] == 1
    os.environ.update(STAR_UVT_TILE_CAPACITY=str(training["tile_capacity"]), STAR_UVT_TILE_T="1")

    import torch
    import wandb
    from paper_local_resources import configure_local_mps
    from star_uvt_runtime import ensure_star_uvt_on_path

    torch.set_num_threads(2)
    configure_local_mps(load_config_file(frozen["resource_protocol"]), "mps")
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from research_project.benchmarks import multicam_heldout_compare as c
    from torch_gsplat_bridge_star_uvt import UVTRenderConfig

    model, loaded = c._load_frozen_world_checkpoint(
        Path(identity["path"]), device=torch.device("mps"),
        expected_file_sha256=frozen["expected_checkpoint_sha256"],
        expected_world_state_sha256=frozen["expected_world_state_sha256"],
        expected_full_frames=frozen["full_frames"],
    )
    baseline = load_config_file(meta["baseline_config"])
    bundle = c.load_multicam_video_bundle(
        data_cfg=meta["config_data"], camera_cfg={**baseline["camera"], "rig_init": "neural_3d_video"},
        target_size=tuple(frozen["image_size"]), device=torch.device("mps"),
        frame_device=torch.device("cpu"), defer_video_frames=True,
    )
    render = UVTRenderConfig(
        height=frozen["image_size"][0], width=frozen["image_size"][1], frames=frozen["full_frames"],
        **{key: training[key] for key in ["tile_capacity", "tile_x", "tile_y", "tile_t", "alpha_mode"]},
    )
    assert frozen["wandb_enabled"] and frozen["wandb_mode"] == "offline"
    run = wandb.init(project=frozen["wandb_project"], mode="offline", dir=str(out),
                     name="frozen-world-device-chunks", config={**frozen, "device_chunk_control": cfg},
                     tags=[*frozen["wandb_tags"], "device_chunk_control"])
    progress = {"scope": "local execution layout; no optimizer or publication claim", "status": "running", "rows": []}
    try:
        for chunk in cfg["device_chunk_sizes"]:
            for frames in cfg["frame_counts"]:
                packed = next(row for row in capacity["rows"] if row["chunk"] == chunk and row["frames"] == frames)
                if packed["overflow_tiles"]:
                    # This is a retained negative packing result, not a passing render.
                    progress["rows"].append({"chunk": chunk, "frames": frames, "accepted": False,
                        "status": "capacity_rejected", "packing": packed})
                    (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                    run.log({"chunk": chunk, "frames": frames, "capacity_rejected": True,
                             "max_tile_count": packed["max_tile_count"]})
                    print(f"Retained capacity rejection: chunk {chunk}, F{frames}, {packed['max_tile_count']}/256 slots", flush=True)
                    continue
                row_out = out / f"chunk{chunk}_frames{frames}"
                row_out.mkdir(exist_ok=True)
                provider = c.PaperMulticamTargetProvider(bundle.heldout_frame_sources, cache_capacity_frames=cfg["cpu_cache_frames"])
                print(f"Starting device chunk {chunk}, F{frames}, temporal tile {render.tile_t}", flush=True)
                row = c.frozen_world_replay_compiled_report(
                    model, bundle, render_config=render, camera_projection=frozen["camera_projection"],
                    out_dir=row_out, max_frames=frames, resident_chunk_frames=chunk, checkpoint=loaded,
                    verify_selected_time_slice_parity=frames < frozen["full_frames"],
                    heldout_target_provider=provider,
                    timing_warmups=frozen["timing_warmups"], timing_repeats=frozen["timing_repeats"],
                )
                assert row["retained_storage_bytes"]["compiled"]["artifact"]["sha256"] == packed["atlas_sha256"]
                (row_out / "report.json").write_text(json.dumps(row, indent=2) + "\n")
                (row_out / "target_provider_accounting.json").write_text(json.dumps(provider.accounting(), indent=2) + "\n")
                progress["rows"].append({"chunk": chunk, "frames": frames, "path": str(row_out / "report.json"), "accepted": row["accepted"]})
                (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                run.log({"chunk": chunk, "frames": frames, "accepted": row["accepted"],
                         "timing": row["timing_benchmark"]["summary_s"], "image": row["image"], "gradient": row["gradient"]})
                run.save(str(row_out / "report.json"), base_path=str(out))
                run.save(str(row_out / "target_provider_accounting.json"), base_path=str(out))
                print(json.dumps({**progress["rows"][-1], "checks": row["checks"]}), flush=True)
                if not row["accepted"]:
                    raise RuntimeError(f"Numerical contract failed at chunk {chunk}, F{frames}; stopping larger batches")
                del provider, row
                gc.collect()
                torch.mps.empty_cache()
        progress["all_rows_accepted"] = all(row["accepted"] for row in progress["rows"])
        progress["status"] = "complete" if progress["all_rows_accepted"] else "complete_with_capacity_rejections"
    except Exception as exc:
        progress.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        run.save(str(out / "progress.json"), base_path=str(out))
        run.summary.update({"status": progress["status"], "completed_rows": len(progress["rows"])})
        (out / "wandb_identity.json").write_text(json.dumps({"id": run.id, "mode": "offline", "dir": run.dir}, indent=2) + "\n")
        run.finish()


if __name__ == "__main__":
    main(sys.argv[1])
