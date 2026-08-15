from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch

from research_experiments.world_foam_lane2 import (
    adaptive_material_basis_selection as ablation,
)
from research_experiments.world_foam_lane2 import (
    verify_adaptive_material_basis_selection as verifier,
)


def _metrics(loss: float) -> dict[str, float]:
    return {
        "loss": loss,
        "beta_mse": 0.75 * loss,
        "rgb_mse": 0.25 * loss,
        "max_beta_abs_error": loss**0.5,
        "max_rgb_abs_error": (0.25 * loss) ** 0.5,
    }


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    candidates: list[dict[str, object]] = []
    for target_index, target in enumerate(ablation.TARGETS):
        for seed in ablation.SEEDS:
            for mode in ablation.MODES:
                if target.family == "m3":
                    loss = 1.0e-6 if mode == ablation.M3 else 1.0e-3
                elif target.family == "m5":
                    loss = 1.0e-3 if mode == ablation.M3 else 1.0e-6
                else:
                    m3_wins = target_index % 2 == 0
                    loss = (
                        2.0e-5
                        if (mode == ablation.M3) == m3_wins
                        else 3.0e-5
                    )
                candidates.append(
                    {
                        "target": target.name,
                        "target_family": target.family,
                        "mode": mode.name,
                        "seed": seed,
                        "serialized_material_scalars": 6,
                        "serialized_material_bytes_float32": 24,
                        "basis_tag_bits": 1,
                        "trainable_scalars": 6,
                        "density_controls": [0.1, 0.2, 0.3],
                        "color_front": [0.2, 0.3, 0.4],
                        "color_back": [0.2, 0.3, 0.4],
                        "splits": {
                            "train": _metrics(1.1 * loss),
                            "selection": _metrics(loss),
                            "heldout": _metrics(loss),
                        },
                        "diagnostic_cpu_wall_seconds": 0.01,
                    }
                )
    selections = ablation._selection_rows(candidates)
    aggregates = ablation._aggregates(candidates, selections)
    checks = ablation._checks(candidates, selections, aggregates)
    assert all(checks.values())
    protocol = ablation.protocol_payload(
        steps=verifier.PAPER_STEPS,
        learning_rate=verifier.PAPER_LEARNING_RATE,
        refinement_steps=verifier.PAPER_REFINEMENT_STEPS,
    )
    report: dict[str, object] = {
        "schema_version": ablation.SCHEMA_VERSION,
        "suite_id": ablation.SUITE_ID,
        "accepted": True,
        "claim_scope": ["synthetic fixture"],
        "promotion": {"eligible_to_replace_p0_in_native_renderer": False},
        "dtype": "float64",
        "device": "cpu",
        "timing_is_paper_evidence": False,
        "git": {"commit": "0" * 40, "dirty": False},
        "source_sha256": ablation._source_hashes(),
        "protocol": protocol,
        "protocol_sha256": ablation._sha256_json(protocol),
        "candidate_rows": candidates,
        "selection_rows": selections,
        "aggregates": aggregates,
        "checks": checks,
    }
    summary = tmp_path / "summary.json"
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    assets = {
        "worldfoam_adaptive_material_basis.svg": figure_dir / "worldfoam_adaptive_material_basis.svg",
        "adaptive_material_basis_table.md": tmp_path / "adaptive_material_basis_table.md",
        "adaptive_material_basis_table.tex": tmp_path / "adaptive_material_basis_table.tex",
    }
    for name, path in assets.items():
        path.write_text(f"fixture {name}\n", encoding="utf-8")
    report["assets"] = {
        name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for name, path in assets.items()
    }
    return report, summary


def test_adaptive_basis_verifier_accepts_complete_receipt(tmp_path: Path) -> None:
    report, summary = _fixture(tmp_path)
    assert verifier.verify_report(report, summary_path=summary) == []


def test_adaptive_basis_verifier_rejects_selected_mode_tamper(tmp_path: Path) -> None:
    report, summary = _fixture(tmp_path)
    tampered = copy.deepcopy(report)
    row = tampered["selection_rows"][0]
    row["selected_mode"] = (
        ablation.M5.name
        if row["selected_mode"] == ablation.M3.name
        else ablation.M3.name
    )
    errors = verifier.verify_report(tampered, summary_path=summary)
    assert any("selected_mode" in error for error in errors)


def test_target_oracle_and_chord_splits_are_well_formed() -> None:
    all_intervals = (
        set(ablation.DEFAULT_INTERVALS)
        | set(ablation.SELECTION_INTERVALS)
        | set(ablation.HELDOUT_INTERVALS)
    )
    assert len(all_intervals) == (
        len(ablation.DEFAULT_INTERVALS)
        + len(ablation.SELECTION_INTERVALS)
        + len(ablation.HELDOUT_INTERVALS)
    )
    beta, moment = ablation.independent_target_outputs(
        ablation.TARGETS[-1],
        ablation.SELECTION_INTERVALS,
    )
    assert beta.shape == (len(ablation.SELECTION_INTERVALS),)
    assert moment.shape == (len(ablation.SELECTION_INTERVALS), 3)
    assert torch.isfinite(beta).all() and torch.isfinite(moment).all()
    assert ((0.0 < beta) & (beta < 1.0)).all()
