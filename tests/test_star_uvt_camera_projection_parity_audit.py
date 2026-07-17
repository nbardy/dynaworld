from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "star_uvt_v0"
    / "research_project"
    / "benchmarks"
    / "camera_projection_parity_audit.py"
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("camera_projection_parity_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neural_3d_audit_uses_poses_bounds_calibration(monkeypatch) -> None:
    audit = load_audit_module()
    expected_K = torch.tensor(
        [[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    calls = []

    def fake_camera(record, camera_name, *, H, W, device):
        calls.append((record["dataset"], camera_name, H, W, device.type))
        return expected_K, torch.eye(4)

    monkeypatch.setattr(audit, "neural_3d_camera_from_poses_bounds", fake_camera)

    row = audit.audit_camera(
        {"dataset": "neural_3d_video"},
        "cam06",
        target_size=64,
        grid_size=5,
    )

    assert calls == [("neural_3d_video", "cam06", 64, 64, "cpu")]
    assert row["camera"] == "cam06"
    assert row["lens_model"] == "pinhole"
    assert row["distortion"] == []
    assert row["max_shift_px"] == 0.0
