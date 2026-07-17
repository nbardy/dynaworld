from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
PROBE_PATH = (
    DYNAWORLD
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "star_uvt_v0"
    / "research_project"
    / "benchmarks"
    / "uvt_train_step_timing_probe.py"
)


def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def _load_probe_module() -> types.ModuleType:
    stub_names = [
        "torch_gsplat_bridge_star_uvt",
        "research_project",
        "research_project.trainer_harness",
        "research_project.trainer_harness.data",
        "research_project.trainer_harness.model",
        "research_project.trainer_harness.tile_metal_autograd",
    ]
    original_modules = {name: sys.modules.get(name) for name in stub_names}
    bridge = types.ModuleType("torch_gsplat_bridge_star_uvt")

    class DummyConfig:
        pass

    bridge.UVTRenderConfig = DummyConfig
    for name in (
        "direct_atomic_backward",
        "direct_fixedpoint_backward",
        "direct_split_fixedpoint_backward",
        "direct_serial_backward",
        "render_uvt_tubes",
        "stable_backward_samples",
        "stable_backward_samples_with_keys",
        "tile_pair_backward_samples",
        "tile_pair_backward_samples_compensated",
        "tile_pair_atomic_backward",
        "tile_pair_fixedpoint_backward",
        "tile_pair_grouped_backward_samples",
        "tile_pair_parallel_backward_samples",
        "tile_pair_reduced_backward",
        "tile_pair_reduced_parallel_backward",
        "tile_pair_scanline_backward_samples",
        "tile_pair_sharedsort_backward_samples",
        "tile_pair_suffix_backward_samples",
        "tile_pair_suffix_reduced_backward",
        "tile_pair_target_bounds_backward_samples",
    ):
        setattr(bridge, name, _noop)
    data = types.ModuleType("research_project.trainer_harness.data")
    data.load_video_target = _noop
    model = types.ModuleType("research_project.trainer_harness.model")
    model.ScreenTimeTubeModel = object
    autograd = types.ModuleType("research_project.trainer_harness.tile_metal_autograd")
    autograd.render_uvt_tubes_metal_tile_backward = _noop
    sys.modules["torch_gsplat_bridge_star_uvt"] = bridge
    sys.modules["research_project"] = types.ModuleType("research_project")
    sys.modules["research_project.trainer_harness"] = types.ModuleType("research_project.trainer_harness")
    sys.modules["research_project.trainer_harness.data"] = data
    sys.modules["research_project.trainer_harness.model"] = model
    sys.modules["research_project.trainer_harness.tile_metal_autograd"] = autograd
    try:
        spec = importlib.util.spec_from_file_location("uvt_train_step_timing_probe_frame_fit_test", PROBE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load spec for {PROBE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class StarUvtTimingProbeFrameFitTests(unittest.TestCase):
    def test_repeat_mode_expands_loaded_target_with_honest_metadata(self) -> None:
        probe = _load_probe_module()
        target = torch.arange(2, dtype=torch.float32).reshape(2, 1, 1, 1)

        fitted, meta = probe._fit_loaded_video_target_frames(
            target,
            requested_frame_count=5,
            allow_repeat_loaded_frames=True,
        )

        self.assertEqual(fitted[:, 0, 0, 0].tolist(), [0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertEqual(meta["requested_frame_count"], 5)
        self.assertEqual(meta["loaded_frame_count"], 2)
        self.assertTrue(meta["repeat_loaded_frames"])
        self.assertTrue(meta["repeat_loaded_frames_used"])
        self.assertEqual(meta["repeat_loaded_frames_scope"], "video_target")

    def test_missing_repeat_mode_rejects_short_loaded_target(self) -> None:
        probe = _load_probe_module()
        target = torch.zeros((2, 1, 1, 1), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "pass --repeat-loaded-frames"):
            probe._fit_loaded_video_target_frames(
                target,
                requested_frame_count=5,
                allow_repeat_loaded_frames=False,
            )


if __name__ == "__main__":
    unittest.main()
