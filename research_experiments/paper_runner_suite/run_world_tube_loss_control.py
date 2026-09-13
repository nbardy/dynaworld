"""Matched local loss substitution, using the existing guarded training recipe.

The worker replaces only this benchmark module's photometric function. It
records the first actual residual/loss/derivative and pre-optimizer world;
production renderer, sampler, evaluator and regularization stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys

from config_utils import load_config_file
from paper_training_protocol import resolve_paper_training_protocol
from research_experiments.paper_runner_suite.run_unified_paper_ablation import (
    _comparison_wandb_log, comparison_command, live_resource_snapshot,
    require_live_resources, run_checked_with_peak_rss, source_provenance,
)


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def main(config_path: str, loss_name: str, *, worker: bool = False) -> None:
    cfg = load_config_file(config_path)
    assert loss_name in cfg["photometric_losses"] and loss_name in {"robust_l1", "mse"}
    base = load_config_file(cfg["sampling_config"])
    variant = next(v for v in base["variants"] if v["name"] == cfg["variant"])
    protocol = resolve_paper_training_protocol(load_config_file(variant["protocol"]))
    root = Path.cwd()
    out = root / cfg["output_dir"] / loss_name
    lane = out / "world_tubes"
    assert base["logging"]["wandb_enabled"] and base["logging"]["wandb_mode"] == "offline"
    if not worker:
        if out.exists():
            raise FileExistsError(f"Preserve prior attempts; choose a fresh output root: {out}")
        lane.mkdir(parents=True)
        snapshot = live_resource_snapshot()
        require_live_resources(snapshot, protocol)
        write(out / "preflight.json", snapshot)
        command = comparison_command(Path(variant["protocol"]).resolve(), protocol, base["seed"], lane,
            backward_policy=base["backward_policy"], device=base["device"], only_lane="world_tubes",
            allow_local_mps_execution=True)
        command[command.index("--uvt-init-sampling")+1] = variant["init_sampling"]
        for key, flag in [("tile_capacity", "--uvt-tile-capacity"), ("tile_t", "--uvt-tile-t"),
                          ("init_depth", "--init-depth"), ("init_precision_xy", "--uvt-init-precision-xy")]:
            command.extend([flag, str(variant[key])])
        write(out / "command.json", command)
        bound = [Path(__file__).resolve(), Path(config_path).resolve(), Path(cfg["sampling_config"]).resolve(),
                 Path(variant["protocol"]).resolve(), Path(command[1]), Path(command[command.index("--baseline-config")+1])]
        star = root / "third_party/fast-mac-gsplat/variants/star_uvt_v0"
        bound += [star / p for p in ["csrc/metal/star_uvt_kernels.metal", "research_project/trainer_harness/world_tube.py",
            "research_project/trainer_harness/tile_metal_autograd.py", "torch_gsplat_bridge_star_uvt/rasterize.py",
            "torch_gsplat_bridge_star_uvt/_C.cpython-311-darwin.so"]]
        bound += [root / p for p in ["src/train/paper_training_protocol.py", "src/train/paper_training_types.py",
            "src/train/paper_multicam_targets.py", "src/train/multicam_video_data.py", "src/train/multicam_val_data.py",
            "research_experiments/gauge_fields/common.py", "research_experiments/paper_runner_suite/run_unified_paper_ablation.py"]]
        source = source_provenance()
        source["bound_files"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound}
        write(out / "source_identity.json", source)
        def timeout(signum, frame):
            raise TimeoutError("unchanged 600-second loss-control wall limit")
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(base["timeout_seconds"])
        receipt = run_checked_with_peak_rss([sys.executable, "-u", str(Path(__file__).resolve()), config_path,
            loss_name, "--worker"], cwd=root, rss_limit_bytes=protocol.local_resources["process_tree_rss_limit_bytes"],
            protocol=protocol, output_root=out)
        signal.alarm(0)
        write(out / "resource_receipt.json", receipt)
        return

    os.environ.update(WANDB_MODE="offline", WANDB_DIR=str(out),
        STAR_UVT_TILE_CAPACITY=str(variant["tile_capacity"]), STAR_UVT_TILE_T=str(variant["tile_t"]))
    import torch
    from star_uvt_runtime import ensure_star_uvt_on_path
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from research_project.benchmarks import multicam_heldout_compare as c
    initial = []
    original_init = c.WorldTubeModel.__init__
    original_loss = c.robust_l1
    calls = 0

    def capture_initial(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        assert not initial, "This control expects one world construction"
        initial.append(c._save_frozen_world_checkpoint(model, lane / "initial_world.pt",
            frame_count=model.frames, representation=model.representation_name))
        write(out / "initial_world_identity.json", initial[0])

    def photometric(residual):
        nonlocal calls
        calls += 1
        value = original_loss(residual) if loss_name == "robust_l1" else residual.square().mean()
        if calls == 1:
            grad = torch.autograd.grad(value, residual, retain_graph=True)[0]
            torch.save({"residual": residual.detach().cpu(), "value": value.detach().cpu(),
                "gradient": grad.detach().cpu(), "loss_name": loss_name}, out / "first_loss_probe.pt")
        return value

    c.WorldTubeModel.__init__ = capture_initial
    c.robust_l1 = photometric
    command = json.loads((out / "command.json").read_text())
    sys.argv = command[1:]
    c.main()
    report = json.loads((lane / "comparison_report.json").read_text())
    assert report["star_uvt"]["steps"] == protocol.steps == calls
    assert report["star_uvt"]["stopped_reason"] is None
    for name in ["multiscale_loss_weight", "crop_loss_weight", "sequence_consistency_weight"]:
        assert report["star_uvt"][name] == 0
    report["diagnostic_photometric_loss"] = {
        "name": loss_name, "calls": calls, "initial_world": initial[0],
        "first_probe_sha256": hashlib.sha256((out / "first_loss_probe.pt").read_bytes()).hexdigest(),
        "substitution_scope": "module-local robust_l1 symbol; auxiliary photometric weights are zero",
        "formula": "mean(sqrt(residual^2+1e-6))" if loss_name == "robust_l1" else "mean(residual^2)",
        "publication_eligible": False,
    }
    write(lane / "comparison_report.json", report)
    identity = _comparison_wandb_log(report, protocol, lane_name="world_tubes", seed=base["seed"],
        report_dir=out, wandb_mode="offline", execution_source=json.loads((out / "source_identity.json").read_text()))
    print(json.dumps({"photometric_loss": loss_name, "metrics": report["star_uvt"]["metrics"], "wandb": identity}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], worker="--worker" in sys.argv[3:])
