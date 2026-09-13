"""Independent sampler/exposure, checkpoint and raw cam04 quality checks."""
from pathlib import Path
import json
import math
import sys

import numpy as np
import torch

from config_utils import load_config_file
from paper_training_protocol import PaperSampleScheduleDigest, SpacetimeEpochSampler, paper_stage_for_step, resolve_paper_training_protocol
from research_experiments.paper_runner_suite.verify_world_tube_loss_control import (
    read, sha, world, loss_probe, offline_backing, offline_records,
)


def verify(config_path, evaluation_dir=None):
    cfg = load_config_file(config_path)
    cfg.setdefault("control_label", "cam04_only")
    cfg.setdefault("photometric_gradient_view_indices", None)
    masked = cfg["photometric_gradient_view_indices"] is not None
    assert cfg["optimizer_train_views"] == ("all" if masked else "first_only")
    assert cfg["photometric_losses"] == ["robust_l1"]
    assert not masked or cfg["photometric_gradient_view_indices"] == [0]
    control = cfg["control_label"]
    base = load_config_file(cfg["sampling_config"])
    variant = next(v for v in base["variants"] if v["name"] == cfg["variant"])
    protocol = resolve_paper_training_protocol(load_config_file(variant["protocol"]))
    out = Path(cfg["output_dir"])
    evaluation_path = Path(evaluation_dir) if evaluation_dir else out / "evaluation"
    reference, trained = Path(cfg["reference_run"]), out / "robust_l1"
    reports = {name: read(folder / "world_tubes/comparison_report.json")
        for name, folder in (("two_camera", reference), (control, trained))}
    evaluation = read(evaluation_path / "report.json")
    sources = {name: read(folder / "source_identity.json")
        for name, folder in (("two_camera", reference), (control, trained))}
    shared = sources["two_camera"]["bound_files"].keys() & sources[control]["bound_files"].keys()
    changes = [p for p in shared if sources["two_camera"]["bound_files"][p] != sources[control]["bound_files"][p]]
    assert changes == ["research_experiments/paper_runner_suite/run_world_tube_loss_control.py"]
    commands = [read(folder / "command.json") for folder in (reference, trained)]
    for command in commands:
        command[command.index("--out-dir") + 1] = "<output>"
        if "--uvt-optimizer-train-views" in command:
            index = command.index("--uvt-optimizer-train-views")
            assert command[index + 1] == cfg["optimizer_train_views"]
            del command[index:index + 2]
    assert commands[0] == commands[1]
    initial = [world(reports[name]["diagnostic_photometric_loss"]["initial_world"]) for name in reports]
    assert initial[0]["world_state_sha256"] == initial[1]["world_state_sha256"]
    assert all(torch.equal(v, initial[1]["state_dict"][k]) for k, v in initial[0]["state_dict"].items())
    probe_tensors, probe = loss_probe(trained / "first_loss_probe.pt", "robust_l1")
    assert probe["sha256"] == reports[control]["diagnostic_photometric_loss"]["first_probe_sha256"]
    mask_checks, observed_batches = None, None
    if masked:
        diagnostic = reports[control]["diagnostic_photometric_loss"]
        assert diagnostic["photometric_gradient_view_indices"] == [0]
        assert diagnostic["sampled_images"] == 1600 and diagnostic["photometric_gradient_images"] == 800
        assert sha(trained / "photometric_batches.json") == diagnostic["photometric_batches_sha256"]
        observed_batches = read(trained / "photometric_batches.json")
        assert len(observed_batches) == 800
        original_probe = torch.load(reference / "first_loss_probe.pt", map_location="cpu", weights_only=True)
        assert torch.equal(original_probe["residual"], probe_tensors["residual"])
        mask = torch.tensor(probe_tensors["gradient_view_mask"])
        assert probe_tensors["sample_view_indices"] == observed_batches[0]["view_indices"]
        assert probe_tensors["sample_frame_indices"] == observed_batches[0]["frame_indices"]
        assert mask.tolist() == [v == 0 for v in observed_batches[0]["view_indices"]]
        assert torch.equal(original_probe["gradient"][mask], probe_tensors["gradient"][mask])
        assert torch.count_nonzero(probe_tensors["gradient"][~mask]) == 0
        mask_checks = {"initial_residual_exact": True, "retained_cam04_derivative_bit_identical": True,
            "excluded_cam09_derivative_exactly_zero": True, "observed_batch_count": 800,
            "first_batch_views": observed_batches[0]["view_indices"], "normalization": "original full two-image batch"}
    exposures, gradient_exposures, raw, metrics = {}, {}, {}, {}
    reevaluation_metric_roundoff = {}
    for name, report in reports.items():
        lane, row = report["star_uvt"], evaluation["rows"][name]
        expected_views = [0, 1] if name == "two_camera" or masked else [0]
        assert lane["optimizer_train_view_indices"] == expected_views
        assert lane["steps"] == protocol.steps == lane["paper_protocol"]["cost"]["optimizer_steps"] == 800
        assert lane["stopped_reason"] is None and report["diagnostic_photometric_loss"]["calls"] == 800
        assert all(lane[k] == 0 for k in ("multiscale_loss_weight", "crop_loss_weight", "sequence_consistency_weight"))
        assert report["meta"]["paper_dataset_bundle"] == evaluation["dataset_identity"]
        assert report["meta"]["paper_evaluator"] == evaluation["evaluator"]
        assert report["meta"]["star_uvt_native_extension"]["sha256"] == evaluation["native_library_sha256"]
        assert sha(row["source_report"]) == row["source_report_sha256"]
        saved = world(lane["final_world_checkpoint"])
        assert row["checkpoint"]["loaded_from_input_checkpoint"]
        assert row["checkpoint"]["world_state_sha256"] == saved["world_state_sha256"]
        assert row["checkpoint"]["sha256"] == lane["final_world_checkpoint"]["sha256"]
        assert all(x["stats"]["overflow_tile_count"] == 0 for x in row["metal_stats"]["rows"])
        assert all(x["stats"]["overflow_tile_count"] == 0 for x in lane["metal_stats"]["rows"])
        sampler = SpacetimeEpochSampler(view_count=len(expected_views), frame_indices=list(range(32)),
            batch_size=2, same_time_count=protocol.same_time_count, local_time_count=0,
            local_time_radius=0, seed=base["seed"] + protocol.sampler_seed_offset)
        digest = PaperSampleScheduleDigest(sampler_seed=sampler.seed)
        counts = np.zeros((2, 32), dtype=np.int64)
        for step in range(800):
            stage = paper_stage_for_step(protocol.stages, step)
            batch = sampler.next_batch(stage.frames_per_step)
            digest.record(step=step, stage=stage, batch=batch)
            if masked and name == control:
                observed = observed_batches[step]
                assert observed["view_indices"] == [s.view_index for s in batch.samples]
                assert observed["frame_indices"] == [s.frame_index for s in batch.samples]
                assert observed["gradient_view_mask"] == [s.view_index == 0 for s in batch.samples]
                assert observed["residual_shape"] == [2, *stage.image_size.as_list(), 3]
            for sample in batch.samples:
                counts[expected_views[sample.view_index], sample.frame_index] += 1
        assert digest.snapshot() == lane["paper_protocol"]["sample_schedule"]
        assert counts.sum() == lane["paper_protocol"]["cost"]["target_frames"] == 1600
        assert lane["paper_protocol"]["cost"]["target_pixels"] == 19292160
        exposures[name] = counts.tolist()
        active_counts = counts.copy()
        if masked and name == control:
            active_counts[1] = 0
        gradient_exposures[name] = active_counts.tolist()
        assert sha(row["raw_cam04"]["path"]) == row["raw_cam04"]["sha256"]
        raw[name] = torch.load(row["raw_cam04"]["path"], map_location="cpu", weights_only=True)
        assert all(list(t.shape) == [32, 96, 128, 3] and t.dtype == torch.float32 and torch.isfinite(t).all()
            for t in raw[name].values())
        diff = raw[name]["prediction"].numpy().astype(np.float64).clip(0, 1) - raw[name]["target"].numpy().astype(np.float64)
        mse = float(np.square(diff).mean())
        camera = row["per_camera"]["train"]["cam04"]
        assert abs(camera["eval_mse"] - mse) < 1e-8
        assert abs(camera["eval_psnr"] + 10 * math.log10(mse)) < 1e-5
        assert abs(camera["eval_l1"] - np.abs(diff).mean()) < 1e-8
        assert abs(row["metrics"]["eval_mse"] - np.mean([m["eval_mse"] for m in row["per_camera"]["train"].values()])) < 1e-10
        for key, value in row["metrics"].items():
            if any(key.endswith(suffix) for suffix in ("_psnr", "_ssim", "_l1", "_mse", "_lpips")):
                assert abs(value - lane["metrics"][key]) < 1e-6, (name, key)
        metrics[name] = row["per_camera"]
    assert np.array_equal(exposures["two_camera"], np.full((2, 32), 25))
    expected_control = np.full((2, 32), 25) if masked else np.stack([np.full(32, 50), np.zeros(32)])
    assert np.array_equal(exposures[control], expected_control)
    assert np.array_equal(gradient_exposures[control], np.stack([np.full(32, 25 if masked else 50), np.zeros(32)]))
    if masked:
        assert reports["two_camera"]["star_uvt"]["paper_protocol"]["sample_schedule"] == reports[control]["star_uvt"]["paper_protocol"]["sample_schedule"]
    assert torch.equal(raw["two_camera"]["target"], raw[control]["target"])
    for split in ("train", "heldout"):
        recorded = reports[control]["diagnostic_view_evaluation"][split]
        assert recorded.keys() == metrics[control][split].keys()
        for camera, values in recorded.items():
            assert values.keys() == metrics[control][split][camera].keys()
            for key, value in values.items():
                repeated = metrics[control][split][camera][key]
                # CPU parallel float64 reduction differs by two ULPs on identical
                # retained pixels. Permit scalar roundoff, not a changed image fit.
                ulps = abs(repeated - value) / max(math.ulp(value), math.ulp(repeated))
                assert math.isfinite(ulps) and ulps <= 4, (split, camera, key, ulps)
                reevaluation_metric_roundoff[f"{split}/{camera}/{key}"] = ulps
    resources = {}
    live_source_differences = set()
    for name, folder in (("training", trained), ("evaluation", evaluation_path)):
        receipt = read(folder / "resource_receipt.json")
        assert receipt["guard_tripped"] is False and receipt["local_resources"]["limits"] == protocol.local_resources
        for key, limit in (("peak_process_tree_and_launcher_rss_bytes", "process_tree_rss_limit_bytes"),
            ("peak_host_swap_growth_bytes", "max_swap_growth_bytes"), ("peak_output_bytes", "output_limit_bytes")):
            assert receipt["local_resources"][key] <= protocol.local_resources[limit]
        resources[name] = receipt
        source = read(folder / "source_identity.json")
        for path, value in source["bound_files"].items():
            archive = out / "after" / path
            assert sha(archive if archive.exists() else path) == value
            if sha(path) != value:
                live_source_differences.add(path)
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                assert sha(archive) == value
            else:
                archive.write_bytes(Path(path).read_bytes())
    train_id = offline_backing(trained / "world_tubes/wandb_identity.json", reports[control], sources[control])
    identity = read(evaluation_path / "wandb_identity.json")
    assert identity["finish_called"] and identity["mode"] == "offline"
    assert identity["report_sha256"] == sha(evaluation_path / "report.json")
    assert sha(identity["run_file"]["path"]) == identity["run_file"]["sha256"]
    config, history = offline_records(identity["run_file"]["path"])
    assert config["report_sha256"] == identity["report_sha256"] and config["source"] == evaluation["source"]
    for name, splits in metrics.items():
        for cameras in splits.values():
            for camera, values in cameras.items():
                assert all(history[f"{name}/{camera}/{key}"] == value for key, value in values.items())
    summary = {"accepted": True, "publication_eligible": False, "metrics": metrics,
        "scope": ("single-seed cam09 photometric-gradient removal with exact original sample schedule and cam04 normalization"
            if masked else "single-seed fixed-total-budget view restriction; cam04 exposure doubles and cam09 remains initialization-exposed"),
        "cam04_psnr_gain": metrics[control]["train"]["cam04"]["eval_psnr"] - metrics["two_camera"]["train"]["cam04"]["eval_psnr"],
        "same_initial_world_sha256": initial[0]["world_state_sha256"], "per_camera_frame_exposures": exposures,
        "per_camera_photometric_gradient_exposures": gradient_exposures, "photometric_mask_checks": mask_checks,
        "reevaluation_metric_roundoff_ulps": reevaluation_metric_roundoff,
        "live_source_differences_from_archive": sorted(live_source_differences),
        "changed_common_bound_files": changes, "independent_loss_probe": probe, "resources": resources,
        "wandb_offline": {"training": train_id, "evaluation": identity["run_id"]},
        "evaluation_dir": str(evaluation_path),
        "evaluation_report_sha256": sha(evaluation_path / "report.json"), "verifier_sha256": sha(__file__)}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"accepted": True, "metrics": metrics, "cam04_psnr_gain": summary["cam04_psnr_gain"]}, indent=2))


if __name__ == "__main__":
    verify(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
