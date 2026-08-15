from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from research_experiments.paper_runner_suite import (
    verify_ordered_transfer_ablation as verifier,
)


def test_checked_in_ordered_transfer_evidence_is_bounded_and_verified() -> None:
    result = verifier.verify()

    assert result["status"] == "pass"
    assert all(result["checks"].values())
    assert (
        result["results"]["selective_16_atom_smoke"]["fallback_tile_count"]
        == 10
    )
    assert (
        result["results"]["dense_199_atom_negative_control"][
            "fallback_tile_count"
        ]
        == 64
    )
    assert result["claim_limits"] == {
        "public_quality_or_speed_ablation_complete": False,
        "projective_retained_fiber_supported": False,
        "adaptive_error_certified_quadrature": False,
        "dense_scene_hybrid_selective": False,
        "accepted_claim": (
            "bounded native forward/VJP correctness, small-fixture "
            "selectivity with oracle metric parity, and a retained "
            "dense-scene negative selectivity control"
        ),
    }


def test_ordered_transfer_verifier_rejects_inconsistent_fallback_counts(
    tmp_path: Path,
) -> None:
    native_path = tmp_path / "native.json"
    shutil.copy2(verifier.DEFAULT_NATIVE_GATE, native_path)
    smoke_root = tmp_path / "smokes"
    for relative_path in verifier.REPORT_PATHS.values():
        destination = smoke_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(verifier.DEFAULT_SMOKE_ROOT / relative_path, destination)

    hybrid_path = smoke_root / verifier.REPORT_PATHS["hybrid_16"]
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
    hybrid["star_uvt"]["physical_visibility"]["fallback_tile_count"] = 11
    hybrid_path.write_text(json.dumps(hybrid), encoding="utf-8")

    with pytest.raises(ValueError, match="fallback fraction"):
        verifier.verify(
            native_gate_path=native_path,
            smoke_root=smoke_root,
        )
