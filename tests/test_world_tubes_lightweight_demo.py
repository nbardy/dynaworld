from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.paper_runner_suite import (
    run_world_tubes_lightweight_demo as demo,
)


def _fake_native_binary() -> dict[str, object]:
    return {
        "path": "third_party/fast-mac-gsplat/variants/star_uvt_v0/"
        "torch_gsplat_bridge_star_uvt/_C.test.so",
        "bytes": 1,
        "sha256": "0" * 64,
        "python_abi_suffix": ".test.so",
        "import_verified": True,
        "used_for_demo_computation": False,
        "role": (
            "packaging preflight; this bounded demo intentionally executes the "
            "CPU reference route and never invokes MPS"
        ),
    }


def test_lightweight_demo_executes_real_replay_and_compiled_routes() -> None:
    report, _replay_image, _compiled_image = demo.build_demo_report(
        native_binary=_fake_native_binary(),
        frames=4,
        image_size=8,
    )

    assert demo.verify_demo_report(report) == []
    assert report["accepted"] is True
    assert report["execution"]["device"] == "cpu"
    assert report["execution"]["training_steps"] == 0
    assert report["execution"]["wandb"] == "not_imported_or_used"
    assert report["same_world"]["matches"] is True
    assert report["forward"]["max_abs_error"] <= 1.0e-5
    assert report["vjp"]["global_normalized_l2_error"] <= 1.0e-5
    assert report["vjp"]["replay_l2_norm"] > 0.0
    assert report["vjp"]["compiled_l2_norm"] > 0.0
    assert report["structure"]["per_frame_replay"]["tile_cell_count"] == 4
    assert report["structure"]["compiled_interval_atlas"]["tile_cell_count"] == 1
    assert report["structure"]["compiled_to_replay_interval_entry_ratio"] == 0.25
    assert report["structure"]["dense_sample_count_matches"] is True


def test_lightweight_demo_writes_and_verifies_manifest_and_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        demo,
        "require_star_uvt_native_binary",
        lambda: _fake_native_binary(),
    )

    report, manifest = demo.run_demo(
        tmp_path,
        frames=4,
        image_size=8,
    )

    assert report["accepted"] is True
    assert demo.verify_demo_directory(tmp_path) == []
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "demo_manifest.json").is_file()
    image_path = tmp_path / manifest["artifacts"]["comparison_image"]["path"]
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
    loaded_manifest = json.loads(
        (tmp_path / "demo_manifest.json").read_text(encoding="utf-8")
    )
    assert loaded_manifest["runtime"]["wandb"] == "not_used"
    assert loaded_manifest["runtime"]["mps"] == "not_used"


def test_lightweight_demo_verifier_rejects_tampered_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        demo,
        "require_star_uvt_native_binary",
        lambda: _fake_native_binary(),
    )
    _report, manifest = demo.run_demo(tmp_path, frames=4, image_size=8)
    image_path = tmp_path / manifest["artifacts"]["comparison_image"]["path"]
    image_path.write_bytes(image_path.read_bytes() + b"tampered")

    errors = demo.verify_demo_directory(tmp_path)

    assert any("comparison_image byte identity drifted" in error for error in errors)


def test_lightweight_demo_verifier_rejects_stale_forward_claim() -> None:
    report, _replay_image, _compiled_image = demo.build_demo_report(
        native_binary=_fake_native_binary(),
        frames=4,
        image_size=8,
    )
    report["forward"]["max_abs_error"] = 0.5

    errors = demo.verify_demo_report(report)

    assert any("checks do not match" in error for error in errors)
    assert any("accepted does not match" in error for error in errors)


def test_lightweight_demo_fails_clearly_without_native_binary(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "torch_gsplat_bridge_star_uvt"
    package_dir.mkdir()

    with pytest.raises(RuntimeError) as exc_info:
        demo.require_star_uvt_native_binary(
            package_dir,
            verify_import=False,
        )

    message = str(exc_info.value)
    assert "requires a STAR UVT native binary compatible" in message
    assert "setup.py build_ext --inplace" in message


@pytest.mark.parametrize(
    ("frames", "image_size", "message"),
    (
        (demo.MAX_FRAMES + 1, 8, "frames must be"),
        (4, demo.MAX_IMAGE_SIZE + 1, "image_size must be"),
    ),
)
def test_lightweight_demo_enforces_bounded_resource_contract(
    frames: int,
    image_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        demo.build_demo_report(
            native_binary=_fake_native_binary(),
            frames=frames,
            image_size=image_size,
        )
