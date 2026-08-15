from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from research_experiments.paper_runner_suite.import_world_tubes_variable_camera_schema_v2 import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_LOCAL_EVIDENCE_DIR,
    DEFAULT_PUBLICATION_SVG,
    FROZEN_CONTRACT,
    LOCAL_SUMMARY_NAME,
    MOVING_DENSITY_COMPONENT,
    CompatibilityImportError,
    _canonical_json_sha256,
    _json_bytes,
    apply_import_to_bundle,
    render_variable_assets,
    render_publication_svg,
    verify_frozen_report_bytes,
    verify_handoff,
    verify_local_import,
    verify_rendered_assets,
)


def _accepted_raw() -> bytes:
    path = DEFAULT_LOCAL_EVIDENCE_DIR / LOCAL_SUMMARY_NAME
    assert path.is_file(), "run the checked-in schema-v2 importer to materialize evidence"
    return path.read_bytes()


def _contract_for_payload(payload: bytes):
    return replace(FROZEN_CONTRACT, raw_sha256=hashlib.sha256(payload).hexdigest())


def _write_valid_ledger(path: Path, ledger: dict) -> None:
    ledger.pop("ledger_sha256", None)
    ledger["ledger_sha256"] = _canonical_json_sha256(ledger)
    path.write_bytes(_json_bytes(ledger))


def test_exact_local_import_is_schema_v2_clean_and_receipted() -> None:
    report, receipt = verify_local_import()
    assert report["schema_version"] == 2
    assert report["source"]["repository_dirty"] is False
    assert report["source"]["star_uvt_dirty"] is False
    assert report["summary"]["last_accepted_half_span_degrees"] == 170.0
    assert report["summary"]["first_death_half_span_degrees"] == 179.5
    assert receipt["compatibility_boundary"] == {
        "decoder": "pinned_schema_v2_compatibility_importer",
        "dirty_schema_v1_178_179_candidate_imported": False,
        "forbidden_decoder": "current_schema_v1_variable_camera_runner",
        "source_schema_version": 2,
    }


def test_schema_v1_candidate_cannot_be_relabelled_as_frozen_schema_v2() -> None:
    report = json.loads(_accepted_raw())
    report["schema_version"] = 1
    report["observed_motion_half_spans_degrees"][-2:] = [178.0, 179.0]
    payload = _json_bytes(report)
    with pytest.raises(CompatibilityImportError, match="schema_version 2"):
        verify_frozen_report_bytes(payload, contract=_contract_for_payload(payload))


def test_clean_source_and_implementation_manifest_are_semantic_gates() -> None:
    report = json.loads(_accepted_raw())
    dirty = copy.deepcopy(report)
    dirty["source"]["repository_dirty"] = True
    dirty_payload = _json_bytes(dirty)
    with pytest.raises(CompatibilityImportError, match="source start is not exact and clean"):
        verify_frozen_report_bytes(
            dirty_payload,
            contract=_contract_for_payload(dirty_payload),
        )

    changed_manifest = copy.deepcopy(report)
    changed_manifest["implementation"]["source_files"][0]["sha256"] = "0" * 64
    manifest_payload = _json_bytes(changed_manifest)
    with pytest.raises(CompatibilityImportError, match="source file manifest drifted"):
        verify_frozen_report_bytes(
            manifest_payload,
            contract=_contract_for_payload(manifest_payload),
        )


def test_exact_table_and_svg_bytes_match_the_paper_freeze() -> None:
    report = verify_frozen_report_bytes(_accepted_raw())
    assets = render_variable_assets(report)
    assert verify_rendered_assets(assets) == dict(sorted(FROZEN_CONTRACT.rendered_assets.items()))
    assert b"179.5" in assets["variable_camera_table.md"]
    assert b"178" not in assets["variable_camera_table.md"]
    assert b"first death" in assets["variable_camera_closure_death.svg"]


def test_publication_svg_is_separate_and_only_anchors_label_inward() -> None:
    report = verify_frozen_report_bytes(_accepted_raw())
    exact = render_variable_assets(report)["variable_camera_closure_death.svg"]
    publication = render_publication_svg(exact)
    assert hashlib.sha256(exact).hexdigest() == FROZEN_CONTRACT.rendered_assets[
        "variable_camera_closure_death.svg"
    ]
    assert publication != exact
    assert b'x="862.00" y="74.0" text-anchor="end"' in publication
    assert b'x="875.00" y="74.0"' in exact
    assert DEFAULT_PUBLICATION_SVG.read_bytes() == publication


def test_handoff_verifier_checks_declared_file_bytes(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "superproject": {"commit": "a" * 40},
        "star_uvt": {"commit": "b" * 40},
        "accepted_retained_jobs": ["variable_camera_closure_death_curve"],
    }
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipts = {"MANIFEST.json": manifest_digest}
    sums = f"{manifest_digest}  MANIFEST.json\n".encode()
    (tmp_path / "SHA256SUMS").write_bytes(sums)
    contract = replace(
        FROZEN_CONTRACT,
        handoff_superproject_commit="a" * 40,
        handoff_star_uvt_commit="b" * 40,
        handoff_sha256s_sha256=hashlib.sha256(sums).hexdigest(),
        handoff_receipts=receipts,
    )
    audit = verify_handoff(tmp_path, contract=contract)
    assert audit["all_receipts_verified"] is True

    manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CompatibilityImportError, match="target digest drifted"):
        verify_handoff(tmp_path, contract=contract)


def test_bundle_patch_preserves_existing_moving_density_and_other_failures(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(DEFAULT_BUNDLE_DIR, bundle)
    ledger_path = bundle / "evidence_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["components"][MOVING_DENSITY_COMPONENT]["preserve_sentinel"] = "yes"
    ledger["missing_runtime_inputs"].append(
        {
            "component": "unrelated_required_gate",
            "status": "missing",
            "expected_summary": "never/delete/me.json",
        }
    )
    _write_valid_ledger(ledger_path, ledger)

    result = apply_import_to_bundle(
        bundle_dir=bundle,
        local_evidence_dir=DEFAULT_LOCAL_EVIDENCE_DIR,
    )
    updated = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert result["bundle_submission_ready"] is False
    assert updated["components"][MOVING_DENSITY_COMPONENT]["preserve_sentinel"] == "yes"
    assert any(
        item.get("component") == "unrelated_required_gate"
        for item in updated["missing_runtime_inputs"]
    )
    assert not any(
        item.get("component") == "variable_camera_closure_death"
        for item in updated["missing_runtime_inputs"]
    )


def test_bundle_patch_restores_missing_moving_density_gate(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(DEFAULT_BUNDLE_DIR, bundle)
    ledger_path = bundle / "evidence_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["components"].pop(MOVING_DENSITY_COMPONENT)
    ledger["missing_runtime_inputs"] = [
        item
        for item in ledger["missing_runtime_inputs"]
        if item.get("component") != MOVING_DENSITY_COMPONENT
    ]
    _write_valid_ledger(ledger_path, ledger)

    apply_import_to_bundle(
        bundle_dir=bundle,
        local_evidence_dir=DEFAULT_LOCAL_EVIDENCE_DIR,
    )
    updated = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert updated["components"][MOVING_DENSITY_COMPONENT]["status"] == "missing"
    assert any(
        item.get("component") == MOVING_DENSITY_COMPONENT
        for item in updated["missing_runtime_inputs"]
    )
    assert updated["submission_ready"] is False
