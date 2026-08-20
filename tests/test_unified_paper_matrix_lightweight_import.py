from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_matrix_dry_import_does_not_import_torch_or_wandb() -> None:
    script = """
import sys
import research_experiments.paper_runner_suite.run_unified_paper_matrix as matrix
assert "torch" not in sys.modules
assert "wandb" not in sys.modules
assert matrix.DEFAULT_OUT_DIR.name == "2026-07-28_world_tubes_submission_matrix_schema2"
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(ROOT / "src" / "train"),
                str(ROOT),
            )
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_matrix_cli_dry_run_emits_canonical_seven_keys_without_heavy_imports() -> None:
    runner = (
        ROOT
        / "research_experiments"
        / "paper_runner_suite"
        / "run_unified_paper_matrix.py"
    )
    matrix = (
        ROOT
        / "src"
        / "train_configs"
        / "paper_protocols"
        / "world_tubes_submission_matrix_v1.jsonc"
    )
    script = f"""
import runpy
import sys
sys.argv = [{str(runner)!r}, "--matrix", {str(matrix)!r}]
runpy.run_path({str(runner)!r}, run_name="__main__")
assert "torch" not in sys.modules
assert "wandb" not in sys.modules
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(ROOT / "src" / "train"),
                str(ROOT),
            )
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["out_dir"].endswith(
        "outputs/benchmarks/"
        "2026-07-28_world_tubes_submission_matrix_schema2"
    )
    assert [run["key"] for run in report["runs"]] == [
        "coffee_martini_full_300f_progressive_512_v1/seed_17/fast_exploration",
        "coffee_martini_full_300f_progressive_512_v1/seed_29/fast_exploration",
        "coffee_martini_full_300f_progressive_512_v1/seed_43/fast_exploration",
        (
            "coffee_martini_full_300f_fixed_512_pixel_matched_v1/"
            "seed_17/fast_exploration"
        ),
        (
            "coffee_martini_full_300f_fixed_512_pixel_matched_v1/"
            "seed_29/fast_exploration"
        ),
        (
            "coffee_martini_full_300f_fixed_512_pixel_matched_v1/"
            "seed_43/fast_exploration"
        ),
        (
            "coffee_martini_full_300f_progressive_global_shuffle_512_v1/"
            "seed_17/fast_exploration"
        ),
    ]


def test_lpips_asset_gate_hashes_exact_cached_bytes_without_importing_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_was_loaded = "torch" in sys.modules
    import paper_training_protocol as protocol

    trunk_bytes = b"alexnet-trunk-fixture"
    linear_bytes = b"lpips-linear-fixture"
    trunk_name = "alexnet-fixture.pth"
    torch_home = tmp_path / "torch"
    trunk_path = torch_home / "hub" / "checkpoints" / trunk_name
    linear_root = tmp_path / "lpips"
    linear_path = linear_root / "weights" / "v0.1" / "alex.pth"
    trunk_path.parent.mkdir(parents=True)
    linear_path.parent.mkdir(parents=True)
    trunk_path.write_bytes(trunk_bytes)
    linear_path.write_bytes(linear_bytes)
    monkeypatch.setattr(
        protocol,
        "LPIPS_ALEXNET_TRUNK",
        {
            "filename": trunk_name,
            "bytes": len(trunk_bytes),
            "sha256": hashlib.sha256(trunk_bytes).hexdigest(),
        },
    )
    monkeypatch.setattr(
        protocol,
        "LPIPS_ALEX_V01_LINEAR",
        {
            "resource": "weights/v0.1/alex.pth",
            "bytes": len(linear_bytes),
            "sha256": hashlib.sha256(linear_bytes).hexdigest(),
        },
    )
    monkeypatch.setattr(
        protocol.importlib.metadata,
        "version",
        lambda name: "fixture" if name == "lpips" else None,
    )

    status = protocol.lpips_alex_asset_status(
        torch_home=torch_home,
        lpips_package_root=linear_root,
    )

    assert status["status"] == "pass"
    assert all(status["checks"].values())
    assert ("torch" in sys.modules) is torch_was_loaded

    trunk_path.write_bytes(b"drifted")
    drifted = protocol.lpips_alex_asset_status(
        torch_home=torch_home,
        lpips_package_root=linear_root,
    )
    assert drifted["status"] == "rejected"
    assert drifted["checks"]["alexnet_trunk_exact"] is False


def test_wandb_preflight_checks_local_credentials_without_importing_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_was_loaded = "torch" in sys.modules
    wandb_was_loaded = "wandb" in sys.modules
    from research_experiments.paper_runner_suite import (
        run_unified_paper_matrix as matrix,
    )

    monkeypatch.setenv("WANDB_API_KEY", "present-but-never-reported")
    monkeypatch.setattr(
        matrix.importlib.util,
        "find_spec",
        lambda name: object() if name == "wandb" else None,
    )
    monkeypatch.setattr(
        matrix.importlib.metadata,
        "version",
        lambda name: "fixture" if name == "wandb" else None,
    )

    readiness = matrix.wandb_local_readiness("online")

    assert readiness["status"] == "pass"
    assert readiness["credential_source"] == "environment"
    assert "present-but-never-reported" not in repr(readiness)
    assert readiness["connectivity"]["requested"] is False
    assert ("wandb" in sys.modules) is wandb_was_loaded
    assert ("torch" in sys.modules) is torch_was_loaded
