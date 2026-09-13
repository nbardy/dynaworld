"""Verify the matched local loss experiment from retained CPU artifacts.

Recompute first-step loss/derivatives, world hashes and sampler schedules;
check source, budget, complete evaluation, resources and offline W&B backing.
Final images are not re-rendered here. Acceptance records a valid measurement,
whether or not MSE improves quality; it is not publication acceptance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from config_utils import load_config_file
from paper_training_protocol import (
    PaperSampleScheduleDigest, SpacetimeEpochSampler, paper_stage_for_step,
    resolve_paper_training_protocol,
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def world(identity):
    assert sha(identity["path"]) == identity["sha256"]
    payload = torch.load(identity["path"], map_location="cpu", weights_only=True)
    metadata = {k: v for k, v in payload.items() if k not in {"schema_version", "state_dict", "world_state_sha256"}}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    assert set(payload["state_dict"]) == set(metadata["parameter_names"])
    for name, tensor in sorted(payload["state_dict"].items()):
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    assert digest.hexdigest() == payload["world_state_sha256"] == identity["world_state_sha256"]
    return payload


def loss_probe(path, name):
    probe = torch.load(path, map_location="cpu", weights_only=True)
    assert probe["loss_name"] == name and list(probe["residual"].shape) == [2, 48, 64, 3]
    residual = probe["residual"].numpy().astype(np.float64)
    if name == "robust_l1":
        terms = np.sqrt(residual * residual + 1e-6)
        gradient = residual / (terms * residual.size)
    else:
        terms = residual * residual
        gradient = 2 * residual / residual.size
    actual = probe["gradient"].numpy().astype(np.float64)
    value_error = abs(float(probe["value"]) - float(terms.mean()))
    gradient_error = float(np.linalg.norm(actual - gradient) / np.linalg.norm(gradient))
    assert np.isfinite(actual).all() and value_error <= 1e-7 and gradient_error <= 1e-6
    return probe, {"value": float(probe["value"]), "value_absolute_error": value_error,
        "gradient_relative_l2_error": gradient_error, "sha256": sha(path)}


def offline_records(path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(path)
    config, history = {}, {}
    try:
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            for item in [*record.run.config.update, *record.config.update]:
                config[item.key or "/".join(item.nested_key)] = json.loads(item.value_json)
            for item in record.history.item:
                history[item.key or "/".join(item.nested_key)] = json.loads(item.value_json)
    finally:
        store.close()
    return config, history


def offline_backing(path, report, source):
    identity = read(path)
    assert identity["mode"] == "offline" and identity["remote_identity"]["finish_called"]
    assert identity["comparison_report_sha256"] == json_hash(report)
    assert identity["source_digest"] == json_hash(source)
    assert sha(identity["run_file"]["path"]) == identity["run_file"]["sha256"]
    config, history = offline_records(identity["run_file"]["path"])
    assert config["comparison_report_sha256"] == json_hash(report)
    assert config["source"]["bound_files"] == source["bound_files"]
    assert config["paper_evaluator"] == report["meta"]["paper_evaluator"]
    for logged, metric in [("train/psnr", "eval_psnr"), ("train/ssim", "eval_ssim"),
                           ("heldout/psnr", "heldout_eval_psnr"), ("heldout/ssim", "heldout_eval_ssim"),
                           ("heldout/lpips", "heldout_eval_lpips")]:
        assert history[logged] == report["star_uvt"]["metrics"][metric]
    assert history["cost/optimizer_steps"] == 800
    assert all(f"media/{split}_view/path" in history for split in ("train", "heldout"))
    return identity["run_id"]


def verify(config_path):
    cfg = load_config_file(config_path)
    base = load_config_file(cfg["sampling_config"])
    variant = next(v for v in base["variants"] if v["name"] == cfg["variant"])
    protocol = resolve_paper_training_protocol(load_config_file(variant["protocol"]))
    out = Path(cfg["output_dir"])
    assert cfg["photometric_losses"] == ["robust_l1", "mse"] and protocol.steps == 800
    reports, sources, commands, initials, probes, rows = [], [], [], [], [], []
    for name in cfg["photometric_losses"]:
        folder = out / name
        report = read(folder / "world_tubes/comparison_report.json")
        lane, meta = report["star_uvt"], report["meta"]
        diagnostic = report["diagnostic_photometric_loss"]
        assert diagnostic["name"] == name and diagnostic["calls"] == lane["steps"] == protocol.steps
        assert diagnostic["publication_eligible"] is False and lane["stopped_reason"] is None
        assert all(lane[k] == 0 for k in ("multiscale_loss_weight", "crop_loss_weight", "sequence_consistency_weight"))
        assert meta["device"] == "mps" and meta["only_lane"] == "world_tubes"
        assert meta["train_cameras"] == ["cam04", "cam09"] and meta["heldout_cameras"] == ["cam06"]
        assert meta["paper_evaluator"]["evaluation_set"] == "all_declared_train_and_heldout_frames"
        assert lane["optimizer_train_view_indices"] == [0, 1] and lane["optimizer_frame_indices"] == list(range(32))
        assert all(row["stats"]["overflow_tile_count"] == 0 for row in lane["metal_stats"]["rows"])
        assert len(lane["metal_stats"]["rows"]) == 3
        source = read(folder / "source_identity.json")
        command = read(folder / "command.json")
        command[command.index("--out-dir") + 1] = "<output>"
        for path, digest in source["bound_files"].items():
            archived = out / "after" / path
            assert sha(archived if archived.exists() else path) == digest, f"Missing or changed execution source: {path}"
        native = meta["star_uvt_native_extension"]
        assert native["sha256"] == sha(native["path"])
        initial = world(diagnostic["initial_world"])
        assert diagnostic["initial_world"] == read(folder / "initial_world_identity.json")
        final = world(lane["final_world_checkpoint"])
        assert initial["world_state_sha256"] != final["world_state_sha256"]
        assert final["active_tube_count"] == final["tube_count"] == 2048
        latest = read(folder / "world_tubes/training_progress/latest.json")
        assert latest["step"] == 800 and latest["checkpoint"]["world_state_sha256"] == final["world_state_sha256"]
        checkpoints = sorted((folder / "world_tubes/training_progress").glob("step_*.pt"))
        assert [int(p.stem.removeprefix("step_")) for p in checkpoints] == [1, *range(10, 801, 10)]
        probe, probe_check = loss_probe(folder / "first_loss_probe.pt", name)
        assert probe_check["sha256"] == diagnostic["first_probe_sha256"]
        assert lane["logs"][0]["recon_loss"] == float(probe["value"])
        sampler = SpacetimeEpochSampler(view_count=2, frame_indices=list(range(32)),
            batch_size=protocol.stages[0].frames_per_step, same_time_count=protocol.same_time_count,
            local_time_count=protocol.local_time_count, local_time_radius=protocol.local_time_radius,
            seed=base["seed"] + protocol.sampler_seed_offset)
        digest = PaperSampleScheduleDigest(sampler_seed=sampler.seed)
        for step in range(protocol.steps):
            stage = paper_stage_for_step(protocol.stages, step)
            digest.record(step=step, stage=stage, batch=sampler.next_batch(stage.frames_per_step))
        assert digest.snapshot() == lane["paper_protocol"]["sample_schedule"]
        cost = lane["paper_protocol"]["cost"]
        assert cost["optimizer_steps"] == 800 and cost["target_frames"] == cost["rasterized_frames"] == 1600
        assert cost["target_pixels"] == cost["rasterized_pixels"] == 19292160
        resources = read(folder / "resource_receipt.json")
        assert resources["guard_tripped"] is False
        limits = resources["local_resources"]["limits"]
        assert limits == protocol.local_resources
        assert 0 < resources["local_resources"]["peak_process_tree_and_launcher_rss_bytes"] <= limits["process_tree_rss_limit_bytes"]
        assert resources["local_resources"]["peak_host_swap_growth_bytes"] <= limits["max_swap_growth_bytes"]
        assert resources["local_resources"]["peak_output_bytes"] <= limits["output_limit_bytes"]
        assert cost["sampled_peak_current_allocated_bytes"] <= limits["mps_allocator_limit_bytes"]
        assert 0 < lane["train_loop_elapsed_s"] < protocol.max_train_seconds == base["timeout_seconds"]
        metrics = lane["metrics"]
        for prefix in ("eval", "heldout_eval"):
            assert abs(metrics[prefix + "_psnr"] + 10 * math.log10(metrics[prefix + "_mse"])) < 1e-8
        assert all(math.isfinite(v) for v in metrics.values())
        media = list((folder / "world_tubes").glob("*.png")) + list((folder / "world_tubes").glob("*.mp4"))
        assert len(media) == 4 and all(p.stat().st_size > 0 for p in media)
        run_id = offline_backing(folder / "world_tubes/wandb_identity.json", report, source)
        rows.append({"loss": name, "metrics": metrics, "train_wall_s": lane["train_loop_elapsed_s"],
            "wandb_offline_id": run_id, "first_loss_probe": probe_check, "resources": resources,
            "report": str(folder / "world_tubes/comparison_report.json"),
            "report_sha256": sha(folder / "world_tubes/comparison_report.json"),
            "checkpoint_sha256": lane["final_world_checkpoint"]["sha256"],
            "world_state_sha256": final["world_state_sha256"], "checkpoint_count": len(checkpoints),
            "media_sha256": {str(p): sha(p) for p in media}})
        reports.append(report); sources.append(source); commands.append(command)
        initials.append(initial); probes.append(probe)
    assert commands[0] == commands[1] and sources[0]["bound_files"] == sources[1]["bound_files"]
    for key in ("paper_dataset_bundle", "paper_evaluator", "star_uvt_native_extension", "paper_runtime"):
        assert reports[0]["meta"][key] == reports[1]["meta"][key], key
    for key in ("sample_schedule", "stages", "sampling", "kernel"):
        assert reports[0]["star_uvt"]["paper_protocol"][key] == reports[1]["star_uvt"]["paper_protocol"][key]
    assert initials[0]["world_state_sha256"] == initials[1]["world_state_sha256"]
    assert all(torch.equal(value, initials[1]["state_dict"][key]) for key, value in initials[0]["state_dict"].items())
    assert torch.equal(probes[0]["residual"], probes[1]["residual"])
    # Archive exactly the execution-bound files; refuse to replace different bytes.
    for path, digest in sources[0]["bound_files"].items():
        archive = out / "after" / path
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            assert sha(archive) == digest
        else:
            archive.write_bytes(Path(path).read_bytes())
        assert sha(archive) == digest
    result = {"accepted": True, "publication_eligible": False, "rows": rows,
        "scope": "one-seed fixed-budget loss substitution; final metrics are retained evaluator output, not an independent re-render",
        "shared_initial_world_sha256": initials[0]["world_state_sha256"],
        "same_initial_residual_exact": True, "same_bound_sources": sources[0]["bound_files"],
        "live_source_differences_from_archive": [p for p, digest in sources[0]["bound_files"].items() if sha(p) != digest],
        "sample_schedule": reports[0]["star_uvt"]["paper_protocol"]["sample_schedule"],
        "metric_delta_mse_minus_robust_l1": {key: rows[1]["metrics"][key] - rows[0]["metrics"][key]
            for key in ("eval_psnr", "eval_mse", "eval_ssim", "heldout_eval_psnr", "heldout_eval_mse", "heldout_eval_ssim", "heldout_eval_lpips")},
        "limits": ["Atomic backward is nondeterministic; one run per loss is not a seed/stability study.",
            "Unchanged regularization weights have different relative strength after the loss substitution.",
            "Heldout cam06 has been used for development and is exploratory validation."],
        "verifier_sha256": sha(__file__)}
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accepted": True, "rows": [{"loss": r["loss"], "metrics": r["metrics"]} for r in rows],
        "delta": result["metric_delta_mse_minus_robust_l1"]}, indent=2))


if __name__ == "__main__":
    verify(sys.argv[1])
