"""Same frozen world, ordinary native UVT renderer, different execution batches.

Launch through the retained resource-guarded launcher. This diagnostic retains
negative outcomes and only measures warmed throughput after image/VJP parity.
"""
from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from config_utils import load_config_file


def main(config_path: str) -> None:
    cfg = load_config_file(config_path)
    frozen = load_config_file(cfg["frozen_config"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    source = Path(frozen["source_run"])
    training = json.loads((source / "comparison_report.json").read_text())["star_uvt"]
    meta = json.loads((source / "run_meta.json").read_text())
    # The existing native wrapper specializes these settings at import time.
    os.environ.update(STAR_UVT_TILE_CAPACITY=str(training["tile_capacity"]),
                      STAR_UVT_TILE_T=str(training["tile_t"]))

    import torch
    import wandb
    from paper_local_resources import configure_local_mps
    from paper_multicam_targets import PaperMulticamTargetProvider, load_grouped_frozen_target_frames
    from star_uvt_runtime import ensure_star_uvt_on_path
    from research_experiments.paper_runner_suite.run_unified_paper_ablation import FROZEN_WORLD_ACCEPTANCE
    from research_experiments.paper_runner_suite.run_frozen_world_replay_compiled import timing_summary

    torch.set_num_threads(2)
    configure_local_mps(load_config_file(frozen["resource_protocol"]), "mps")
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from research_project.benchmarks import multicam_heldout_compare as c
    from torch_gsplat_bridge_star_uvt import UVTRenderConfig

    model, checkpoint = c._load_frozen_world_checkpoint(
        Path(training["final_world_checkpoint"]["path"]), device=torch.device("mps"),
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
    assert render.frames == 32 and render.tile_t == 1 and render.tile_capacity == 256
    assert render.alpha_mode == "peak_splat" and cfg["frame_batch_sizes"][0] == 1
    camera_K = c.select_view_K(bundle.heldout_K, 0)
    camera_w2c = c.select_view_w2c(bundle.heldout_w2c, 0)
    lens_model, distortion = c.select_lens(
        bundle.heldout_lens_models, bundle.heldout_distortions, 0,
        camera_projection=frozen["camera_projection"],
    )
    indices = tuple(range(render.frames))
    element_count = render.frames * render.height * render.width * 3
    provider = PaperMulticamTargetProvider(bundle.heldout_frame_sources, cache_capacity_frames=cfg["cpu_target_cache_frames"])
    assert frozen["wandb_enabled"] and frozen["wandb_mode"] == "offline"
    run = wandb.init(project=frozen["wandb_project"], mode="offline", dir=str(out),
                     name="same-world-native-uvt-batches", config={**frozen, "native_control": cfg},
                     tags=[*frozen["wandb_tags"], "native_uvt_batch_control"])
    report = {"scope": "same frozen static-camera world, contiguous 32 times, native UVT batches",
              "status": "running", "config": cfg, "checkpoint": checkpoint,
              "camera": bundle.heldout_camera_names[0], "frame_indices": list(indices),
              "image_size": frozen["image_size"], "acceptance": FROZEN_WORLD_ACCEPTANCE,
              "correctness": {}, "timing": {}, "publication_eligible": False}

    def boundary(phases, key, started):
        torch.mps.synchronize()
        stopped = time.perf_counter()
        phases[key] += stopped - started
        return stopped

    def execute(batch: int, *, retain: bool):
        model.zero_grad(set_to_none=True)
        phases = dict.fromkeys(["project_bin_render", "target_cpu_load", "target_transfer", "loss", "backward"], 0.0)
        outputs = []
        loss_value = 0.0
        target_hash = hashlib.sha256()
        memory = c.DeviceMemorySampler(torch.device("mps"))
        memory.start()
        try:
            for start in range(0, render.frames, batch):
                stop = min(start + batch, render.frames)
                torch.mps.synchronize()
                tick = time.perf_counter()
                local = replace(render, frames=stop-start)
                projected = c.project_world_tube_sequence(
                    model, camera_K, camera_w2c, local,
                    camera_projection=frozen["camera_projection"], lens_model=lens_model,
                    distortion=distortion,
                    full_frames=render.frames, frame_start=start,
                )
                image = c.render_projected_sequence(projected, local, backend="metal_tile",
                    reduction_mode="index_add", sample_emission_mode="direct_atomic").rgb
                tick = boundary(phases, "project_bin_render", tick)
                loss = 0
                # Targets stay in requests <= LRU8. Full-batch outputs and loss
                # residuals are intentionally resident and included in memory.
                target_chunk = min(batch, provider.cache_capacity_frames)
                for target_start in range(start, stop, target_chunk):
                    target_stop = min(target_start + target_chunk, stop)
                    host = load_grouped_frozen_target_frames(provider, indices,
                        start=target_start, stop=target_stop, chunk_frames=target_chunk)
                    tick = boundary(phases, "target_cpu_load", tick)
                    target = host.permute(0, 2, 3, 1).contiguous().to("mps")
                    tick = boundary(phases, "target_transfer", tick)
                    loss = loss + torch.sqrt((image[target_start-start:target_stop-start]-target).square()+1.0e-6).sum()/element_count
                    tick = boundary(phases, "loss", tick)
                    if retain:
                        target_hash.update(memoryview(host.contiguous().numpy()).cast("B"))
                    del host, target
                    # Retention/hashing is outside the performance segments.
                    tick = time.perf_counter()
                loss.backward()
                boundary(phases, "backward", tick)
                if retain:
                    outputs.append(image.detach().cpu())
                    loss_value += float(loss.detach().cpu())
                del projected, image, loss
        finally:
            memory.stop()
        payload = None
        if retain:
            grads, covered = c._world_parameter_gradients(model)
            payload = {"rgb": torch.cat(outputs), "gradients": grads, "covered": covered,
                       "loss": loss_value, "target_sha256": target_hash.hexdigest()}
        model.zero_grad(set_to_none=True)
        return phases, memory.stats(), payload

    try:
        reference = None
        for batch in cfg["frame_batch_sizes"]:
            print(f"Correctness: native UVT batch {batch}", flush=True)
            _, memory, payload = execute(batch, retain=True)
            artifact = out / f"batch{batch}_correctness.pt"
            torch.save(payload, artifact)
            if reference is None:
                reference = payload
            gradient = c._gradient_comparison(reference["gradients"], payload["gradients"],
                replay_covered=reference["covered"], compiled_covered=payload["covered"])
            delta = payload["rgb"]-reference["rgb"]
            checks = {"image_matches": float(delta.abs().max()) <= FROZEN_WORLD_ACCEPTANCE["image_max_abs_error"],
                      "loss_matches": abs(payload["loss"]-reference["loss"]) <= FROZEN_WORLD_ACCEPTANCE["loss_absolute_delta"],
                      "gradient_matches": gradient["global_normalized_l2_error"] <= FROZEN_WORLD_ACCEPTANCE["gradient_global_normalized_l2_error"],
                      "per_parameter_matches": gradient["max_parameter_normalized_l2_error"] <= FROZEN_WORLD_ACCEPTANCE["gradient_max_parameter_normalized_l2_error"],
                      "gradient_coverage": gradient["gradient_coverage_matches"] and len(payload["covered"]) == len(checkpoint["parameter_names"]),
                      "gradient_nonzero": min(gradient["replay_l2_norm"], gradient["compiled_l2_norm"]) > FROZEN_WORLD_ACCEPTANCE["min_world_vjp_l2_norm"],
                      "targets_match": payload["target_sha256"] == reference["target_sha256"]}
            report["correctness"][str(batch)] = {"checks": checks, "accepted": all(checks.values()),
                "max_rgb_error": float(delta.abs().max()), "loss_delta": abs(payload["loss"]-reference["loss"]),
                "gradient": gradient, "memory": memory, "artifact": str(artifact),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
            print(json.dumps({"batch": batch, **report["correctness"][str(batch)]}), flush=True)
            run.save(str(artifact), base_path=str(out))
            if not all(checks.values()):
                report["status"] = "parity_failed"
                return
            del payload, delta
            gc.collect()
            torch.mps.empty_cache()
        del reference
        for batch in cfg["frame_batch_sizes"]:
            for _ in range(frozen["timing_warmups"]): execute(batch, retain=False)
        samples = {str(batch): [] for batch in cfg["frame_batch_sizes"]}
        for repeat in range(frozen["timing_repeats"]):
            order = cfg["frame_batch_sizes"][repeat:] + cfg["frame_batch_sizes"][:repeat]
            for batch in order:
                phases, memory, _ = execute(batch, retain=False)
                samples[str(batch)].append({"phases_s": phases, "memory": memory})
        report["timing"] = {batch: {"samples": values,
            "evaluator_plus_backward_s": timing_summary([x["phases_s"]["project_bin_render"]+x["phases_s"]["backward"] for x in values]),
            "full_s": timing_summary([sum(x["phases_s"].values()) for x in values])} for batch, values in samples.items()}
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="execution_failed", failure=f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        report["target_provider_accounting"] = provider.accounting()
        report["world_state_after_sha256"] = c._world_state_digest(c.snapshot_world_tube_state(model),
            metadata=c._world_state_metadata(model, frame_count=render.frames, representation=model.representation_name))
        (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        run.summary.update({"status": report["status"], "timing": report["timing"]})
        run.save(str(out/"report.json"), base_path=str(out))
        (out/"wandb_identity.json").write_text(json.dumps({"id": run.id, "mode": "offline", "dir": run.dir}, indent=2)+"\n")
        run.finish()


if __name__ == "__main__":
    main(sys.argv[1])
