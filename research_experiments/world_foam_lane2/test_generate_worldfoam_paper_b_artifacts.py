from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from research_experiments.world_foam_lane2.generate_worldfoam_paper_b_artifacts import (
    DEFAULT_ADAPTIVE_MATERIAL,
    DEFAULT_COMPILED_LIE,
    DEFAULT_CONSTANT_TRANSFER,
    DEFAULT_MATERIAL_FIT,
    DEFAULT_MATERIAL_PARITY,
    DEFAULT_SYNTHETIC_VISIBILITY,
    EXPECTED_ADAPTIVE_MATERIAL_ASSETS,
    EXPECTED_VISIBILITY_FIGURES,
    build_bundle,
    default_specs,
    verify_bundle_dir,
    write_bundle,
)
from research_experiments.world_foam_lane2.test_verify_worldfoam_public_quality_ablation_v2 import (  # noqa: E501
    _complete_artifact as _complete_g4_artifact,
)
from research_experiments.world_foam_lane2.test_verify_worldfoam_training_memory_ablation import (  # noqa: E501
    _artifact as _complete_g6_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "research_experiments"
    / "world_foam_lane2"
    / "generate_worldfoam_paper_b_artifacts.py"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _ledger_by_id(bundle: object) -> dict[str, dict[str, object]]:
    return {
        record["evidence_id"]: record
        for record in bundle.ledger["records"]
    }


def test_current_bundle_accepts_full_cpu_g0_g3_and_rejects_stale_compiled_lie() -> None:
    bundle = build_bundle()
    ledger = _ledger_by_id(bundle)

    assert bundle.complete is False
    assert ledger["m0_m5_segment_parity"]["status"] == "accepted"
    assert ledger["m3_m5_partial_chord_fit"]["status"] == "accepted"
    assert ledger["adaptive_m3_m5_basis_selection"]["status"] == "accepted"
    assert ledger["adaptive_m3_m5_basis_selection"]["numeric_rows_emitted"] == 2
    assert ledger["constant_density_ordered_transfer"]["status"] == "accepted"
    assert ledger["synthetic_visibility_g0_g3"]["status"] == "accepted"
    assert ledger["synthetic_visibility_g0_g3"]["numeric_rows_emitted"] == 9
    assert ledger["compiled_lie_frame_density"]["status"] == "rejected"
    assert ledger["compiled_lie_frame_density"]["numeric_rows_emitted"] == 0
    assert any(
        "schema_version" in error
        for error in ledger["compiled_lie_frame_density"]["errors"]
    )
    assert len(bundle.foundation_rows) == 16
    assert {
        row["evidence_id"] for row in bundle.foundation_rows
    }.isdisjoint({"compiled_lie_frame_density"})
    historical_metal = next(
        row
        for row in bundle.foundation_rows
        if row["row_id"] == "m0_m5_metal_segment_parity"
    )
    assert historical_metal["verdict"] == (
        "accepted_historical_source_hash_checked_metal"
    )
    assert "not current trainer runtime" in historical_metal["claim_scope"]
    adaptive_loss = next(
        row
        for row in bundle.foundation_rows
        if row["row_id"] == "adaptive_m3_m5_mean_loss"
    )
    assert adaptive_loss["value_1"] == pytest.approx(0.3134053578693319)
    assert adaptive_loss["value_2"] == pytest.approx(1.0)
    adaptive_selection = next(
        row
        for row in bundle.foundation_rows
        if row["row_id"] == "adaptive_m3_m5_selection_accuracy"
    )
    assert adaptive_selection["value_1"] == pytest.approx(1.0)
    assert adaptive_selection["value_2"] == pytest.approx(1.0)
    assert "no native" in adaptive_selection["claim_scope"]
    gates = {row["gate"]: row for row in bundle.gate_status["gates"]}
    assert gates["material_ablation"]["status"] == "accepted_cpu_synthetic"
    assert gates["G0"]["status"] == "accepted_cpu_synthetic"
    assert gates["G3"]["status"] == "accepted_cpu_synthetic"
    assert bundle.gate_status["claims"]["synthetic_cpu_g0_g3"] is True
    assert bundle.gate_status["claims"]["adaptive_material_basis_cpu"] is True
    for name in EXPECTED_ADAPTIVE_MATERIAL_ASSETS:
        source = DEFAULT_ADAPTIVE_MATERIAL.parent / (
            "figures" if name.endswith(".svg") else ""
        ) / name
        assert bundle.files[name] == source.read_bytes()
    for name in EXPECTED_VISIBILITY_FIGURES:
        assert bundle.files[name] == (
            DEFAULT_SYNTHETIC_VISIBILITY.parent / "figures" / name
        ).read_bytes()


def test_bundle_claims_and_required_placeholders_remain_fail_closed() -> None:
    bundle = build_bundle()
    claims = bundle.gate_status["claims"]
    assert claims["synthetic_cpu_g0_g3"] is True
    assert claims["adaptive_material_basis_cpu"] is True
    assert all(
        claims[key] is False
        for key in (
            "native_memory_fit",
            "public_quality",
            "public_or_native_visibility_advantage",
            "official_cuda_warp_parity",
            "state_of_the_art",
        )
    )
    gates = {row["gate"]: row for row in bundle.gate_status["gates"]}
    for gate in ("G4", "G6"):
        assert gates[gate]["status"] == "not_measured"

    for name in (
        "g4_public_quality_placeholder.svg",
        "g6_native_memory_placeholder.svg",
    ):
        assert b"NOT MEASURED" in bundle.files[name]
    assert b"native-memory or public-quality evidence" in bundle.files[
        "foundation_table.md"
    ]
    assert b"ACCEPTED \xe2\x80\x94 CPU SYNTHETIC ONLY" in bundle.files[
        "foundation_table.md"
    ]


def test_all_svg_outputs_are_valid_and_self_describing() -> None:
    bundle = build_bundle()
    for name, payload in bundle.files.items():
        if not name.endswith(".svg"):
            continue
        root = ET.fromstring(payload)
        assert root.tag.endswith("svg"), name
        children = list(root)
        assert any(
            child.tag.endswith("title") and (child.text or "").strip()
            for child in children
        ), name
        assert any(
            child.tag.endswith("desc") and (child.text or "").strip()
            for child in children
        ), name


def test_foundation_error_figure_uses_readable_publication_labels() -> None:
    figure = build_bundle().files["foundation_error_summary.svg"].decode("utf-8")
    for label in (
        "CPU segment integral",
        "CPU segment VJP",
        "Metal segment forward",
        "Metal segment VJP",
        "Constant-density render",
        "Constant-density VJP",
    ):
        assert label in figure
    assert "m0_m5_cpu_segment_parity" not in figure
    assert "finite_difference_vjp_normalized_error" not in figure


def test_publication_tex_fragment_has_complete_claim_scoped_environment() -> None:
    bundle = build_bundle()
    fragment = bundle.files["foundation_table.tex"].decode("utf-8")
    for environment in ("table*", "tabular"):
        assert fragment.count(f"\\begin{{{environment}}}") == 1
        assert fragment.count(f"\\end{{{environment}}}") == 1
    assert "\\caption{" in fragment
    assert "\\label{tab:worldfoam-foundation}" in fragment
    assert "CPU synthetic only" in fragment
    assert "NOT MEASURED: public quality" in fragment
    assert "NOT MEASURED: native memory" in fragment
    assert "Metal parity (historical)" in fragment
    assert "Adaptive M3/M5 mean loss" in fragment
    assert "Adaptive M3/M5 selection" in fragment
    assert "do not authorize native material promotion" in fragment


def test_compact_synthetic_table_is_verifier_derived_and_scope_limited() -> None:
    fragment = build_bundle().files["synthetic_visibility_table.tex"].decode(
        "utf-8"
    )
    assert "37.9252 dB" in fragment
    assert "82.2477$\\times$" in fragment
    assert "528.953$\\times$" in fragment
    assert "3.32998e-07 / 0.305335" in fragment
    assert "4.35542e-04 / 5.05269e-04" in fragment
    assert "does not report native runtime" in fragment
    assert "trained public-data quality" in fragment


def test_visibility_verifier_rejects_tampered_full_matrix_without_rows_or_figures(
    tmp_path: Path,
) -> None:
    source = json.loads(
        DEFAULT_SYNTHETIC_VISIBILITY.read_text(encoding="utf-8")
    )
    tampered = copy.deepcopy(source)
    tampered["baseline_rows"][0]["rgb_mse"] *= 4.0
    target = tmp_path / "summary.json"
    _write_json(target, tampered)
    shutil.copytree(
        DEFAULT_SYNTHETIC_VISIBILITY.parent / "figures",
        tmp_path / "figures",
    )

    bundle = build_bundle(
        default_specs(synthetic_visibility=target)
    )
    record = _ledger_by_id(bundle)["synthetic_visibility_g0_g3"]
    assert record["status"] == "rejected"
    assert record["numeric_rows_emitted"] == 0
    assert any(
        "rgb_psnr_db does not match rgb_mse" in error
        for error in record["errors"]
    )
    assert all(
        row["evidence_id"] != "synthetic_visibility_g0_g3"
        for row in bundle.foundation_rows
    )
    assert set(EXPECTED_VISIBILITY_FIGURES).isdisjoint(bundle.files)
    gates = {row["gate"]: row for row in bundle.gate_status["gates"]}
    assert gates["G3"]["status"] == "missing_or_rejected"
    assert bundle.gate_status["claims"]["synthetic_cpu_g0_g3"] is False


def test_joint_g0_g3_claim_requires_constant_transfer_and_visibility(
    tmp_path: Path,
) -> None:
    bundle = build_bundle(
        default_specs(constant_transfer=tmp_path / "missing-transfer.json")
    )
    gates = {row["gate"]: row for row in bundle.gate_status["gates"]}
    assert gates["G0"]["status"] == "missing_or_rejected"
    assert gates["G3"]["status"] == "accepted_cpu_synthetic"
    assert bundle.gate_status["claims"]["synthetic_cpu_g0_g3"] is False


def test_visibility_verifier_rejects_figure_byte_drift(tmp_path: Path) -> None:
    target = tmp_path / "summary.json"
    target.write_bytes(DEFAULT_SYNTHETIC_VISIBILITY.read_bytes())
    shutil.copytree(
        DEFAULT_SYNTHETIC_VISIBILITY.parent / "figures",
        tmp_path / "figures",
    )
    figure = tmp_path / "figures" / EXPECTED_VISIBILITY_FIGURES[0]
    figure.write_bytes(figure.read_bytes() + b"\n")

    bundle = build_bundle(
        default_specs(synthetic_visibility=target)
    )
    record = _ledger_by_id(bundle)["synthetic_visibility_g0_g3"]
    assert record["status"] == "rejected"
    assert any("figure hash mismatch" in error for error in record["errors"])
    assert set(EXPECTED_VISIBILITY_FIGURES).isdisjoint(bundle.files)


def test_adaptive_material_verifier_rejects_tampered_aggregate_without_rows_or_assets(
    tmp_path: Path,
) -> None:
    source = json.loads(DEFAULT_ADAPTIVE_MATERIAL.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(source)
    tampered["aggregates"]["adaptive_to_best_fixed_ratio"] = 9.0
    target = tmp_path / "summary.json"
    _write_json(target, tampered)
    for name in EXPECTED_ADAPTIVE_MATERIAL_ASSETS:
        source_path = DEFAULT_ADAPTIVE_MATERIAL.parent / (
            "figures" if name.endswith(".svg") else ""
        ) / name
        target_path = tmp_path / (
            "figures" if name.endswith(".svg") else ""
        ) / name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    bundle = build_bundle(default_specs(adaptive_material=target))
    record = _ledger_by_id(bundle)["adaptive_m3_m5_basis_selection"]
    assert record["status"] == "rejected"
    assert record["numeric_rows_emitted"] == 0
    assert all(
        row["evidence_id"] != "adaptive_m3_m5_basis_selection"
        for row in bundle.foundation_rows
    )
    assert set(EXPECTED_ADAPTIVE_MATERIAL_ASSETS).isdisjoint(bundle.files)
    gates = {row["gate"]: row for row in bundle.gate_status["gates"]}
    assert gates["material_ablation"]["status"] == "missing_or_rejected"
    assert bundle.gate_status["claims"]["adaptive_material_basis_cpu"] is False
    assert gates["G4"]["status"] == "not_measured"
    assert gates["G6"]["status"] == "not_measured"


def test_material_parity_verifier_rejects_broadened_claim_or_source_drift(
    tmp_path: Path,
) -> None:
    original = json.loads(DEFAULT_MATERIAL_PARITY.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(original)
    tampered["claim_scope"].append("native trainer fits in memory")
    tampered_path = tmp_path / "tampered_material.json"
    _write_json(tampered_path, tampered)

    bundle = build_bundle(
        default_specs(material_parity=tampered_path)
    )
    record = _ledger_by_id(bundle)["m0_m5_segment_parity"]
    assert record["status"] == "rejected"
    assert record["numeric_rows_emitted"] == 0
    assert any("claim limits" in error for error in record["errors"])
    assert all(
        row["evidence_id"] != "m0_m5_segment_parity"
        for row in bundle.foundation_rows
    )

    tampered["claim_scope"] = original["claim_scope"]
    tampered["source_sha256"] = "0" * 64
    _write_json(tampered_path, tampered)
    source_drift = build_bundle(
        default_specs(material_parity=tampered_path)
    )
    record = _ledger_by_id(source_drift)["m0_m5_segment_parity"]
    assert record["status"] == "rejected"
    assert any("source hash" in error for error in record["errors"])


def test_written_bundle_is_deterministic_and_hash_verified(tmp_path: Path) -> None:
    bundle = build_bundle()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = write_bundle(bundle, first)
    second_manifest = write_bundle(bundle, second)

    assert first_manifest == second_manifest
    assert verify_bundle_dir(first) == []
    assert verify_bundle_dir(second) == []
    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files

    (first / "foundation_table.csv").write_text("tampered\n", encoding="utf-8")
    failures = verify_bundle_dir(first)
    assert any("foundation_table.csv" in failure for failure in failures)


def test_verifier_reextracts_results_after_attacker_rebinds_manifest(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "rebound"
    write_bundle(build_bundle(), out_dir)
    assert verify_bundle_dir(out_dir) == []

    foundation_path = out_dir / "foundation_rows.json"
    foundation = json.loads(foundation_path.read_text(encoding="utf-8"))
    foundation["rows"][0]["value_1"] = float(
        foundation["rows"][0]["value_1"]
    ) + 1.0
    _write_json(foundation_path, foundation)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rebound_bytes = foundation_path.read_bytes()
    rebound_row = next(
        row
        for row in manifest["files"]
        if row["path"] == "foundation_rows.json"
    )
    rebound_row["bytes"] = len(rebound_bytes)
    rebound_row["sha256"] = hashlib.sha256(rebound_bytes).hexdigest()
    ledger = json.loads(
        (out_dir / "evidence_ledger.json").read_text(encoding="utf-8")
    )
    gate_status = json.loads(
        (out_dir / "gate_status.json").read_text(encoding="utf-8")
    )
    manifest["content_sha256"] = _canonical_sha256(
        {
            "complete": manifest["complete"],
            "ledger_sha256": _canonical_sha256(ledger),
            "gate_status_sha256": _canonical_sha256(gate_status),
            "foundation_rows_sha256": _canonical_sha256(foundation["rows"]),
            "files": manifest["files"],
        }
    )
    _write_json(manifest_path, manifest)

    failures = verify_bundle_dir(out_dir)
    assert any(
        "foundation rows differ from regenerated evidence" in failure
        for failure in failures
    )


def test_complete_g4_g6_evidence_replaces_placeholders_and_promotes_only_bound_claims(
    tmp_path: Path,
) -> None:
    g4_path = tmp_path / "g4.json"
    g6_path = tmp_path / "g6.json"
    _write_json(g4_path, _complete_g4_artifact())
    g6_artifact, _config, _contract = _complete_g6_artifact()
    _write_json(g6_path, g6_artifact)

    bundle = build_bundle(
        default_specs(
            g4_public_quality=g4_path,
            g6_native_memory=g6_path,
        )
    )
    assert bundle.complete is True
    assert bundle.gate_status["paper_ready"] is True
    assert bundle.gate_status["iclr_ready"] is True
    assert bundle.gate_status["claims"]["public_quality"] is True
    assert bundle.gate_status["claims"]["native_memory_fit"] is True
    assert bundle.gate_status["claims"]["state_of_the_art"] is False
    assert not any("placeholder" in name for name in bundle.files)
    for name in (
        "g4_public_quality_table.tex",
        "g4_public_quality.svg",
        "g6_native_memory_table.tex",
        "g6_native_memory_scaling.svg",
    ):
        assert name in bundle.files
        assert b"NOT MEASURED" not in bundle.files[name]
        if name.endswith(".svg"):
            root = ET.fromstring(bundle.files[name])
            assert any(child.tag.endswith("title") for child in root)
            assert any(child.tag.endswith("desc") for child in root)
    ledger = _ledger_by_id(bundle)
    assert ledger["g4_public_quality"]["numeric_rows_emitted"] == 36
    assert ledger["g6_native_memory"]["numeric_rows_emitted"] == 21

    output = tmp_path / "complete_bundle"
    write_bundle(build_bundle(), output)
    assert (output / "g4_public_quality_placeholder.svg").is_file()
    assert (output / "g6_native_memory_placeholder.svg").is_file()
    write_bundle(bundle, output)
    assert not (output / "g4_public_quality_placeholder.svg").exists()
    assert not (output / "g6_native_memory_placeholder.svg").exists()
    assert verify_bundle_dir(output) == []


def test_verifier_binds_retained_input_bytes(tmp_path: Path) -> None:
    retained = tmp_path / "material_fit.json"
    retained.write_bytes(DEFAULT_MATERIAL_FIT.read_bytes())
    bundle = build_bundle(default_specs(material_fit=retained))
    out_dir = tmp_path / "bundle"
    write_bundle(bundle, out_dir)
    assert verify_bundle_dir(out_dir) == []

    payload = json.loads(retained.read_text(encoding="utf-8"))
    payload["passed"] = False
    _write_json(retained, payload)
    failures = verify_bundle_dir(out_dir)
    assert any("retained input changed" in failure for failure in failures)


def test_verifier_binds_visibility_dependency_bytes(tmp_path: Path) -> None:
    retained = tmp_path / "input" / "summary.json"
    retained.parent.mkdir()
    retained.write_bytes(DEFAULT_SYNTHETIC_VISIBILITY.read_bytes())
    shutil.copytree(
        DEFAULT_SYNTHETIC_VISIBILITY.parent / "figures",
        retained.parent / "figures",
    )
    bundle = build_bundle(
        default_specs(synthetic_visibility=retained)
    )
    out_dir = tmp_path / "bundle"
    write_bundle(bundle, out_dir)
    assert verify_bundle_dir(out_dir) == []

    dependency = retained.parent / "figures" / EXPECTED_VISIBILITY_FIGURES[0]
    dependency.write_bytes(dependency.read_bytes() + b"\n")
    failures = verify_bundle_dir(out_dir)
    assert any("dependency source changed" in failure for failure in failures)


def test_verifier_binds_adaptive_material_dependency_bytes(tmp_path: Path) -> None:
    retained = tmp_path / "input" / "summary.json"
    retained.parent.mkdir()
    retained.write_bytes(DEFAULT_ADAPTIVE_MATERIAL.read_bytes())
    for name in EXPECTED_ADAPTIVE_MATERIAL_ASSETS:
        source = DEFAULT_ADAPTIVE_MATERIAL.parent / (
            "figures" if name.endswith(".svg") else ""
        ) / name
        target = retained.parent / (
            "figures" if name.endswith(".svg") else ""
        ) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    bundle = build_bundle(default_specs(adaptive_material=retained))
    out_dir = tmp_path / "bundle"
    write_bundle(bundle, out_dir)
    assert verify_bundle_dir(out_dir) == []

    dependency = retained.parent / "figures" / (
        "worldfoam_adaptive_material_basis.svg"
    )
    dependency.write_bytes(dependency.read_bytes() + b"\n")
    failures = verify_bundle_dir(out_dir)
    assert any("dependency source changed" in failure for failure in failures)


def test_cli_bootstraps_imports_and_requires_incomplete_acknowledgement(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    help_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--allow-incomplete" in help_result.stdout

    rejected = subprocess.run(
        [sys.executable, str(SCRIPT), "--out-dir", str(tmp_path / "rejected")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert rejected.returncode == 1, rejected.stderr
    rejected_payload = json.loads(rejected.stdout)
    assert rejected_payload["status"] == "incomplete"
    assert rejected_payload["native_memory_fit_claimed"] is False
    assert rejected_payload["public_quality_claimed"] is False

    accepted = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(tmp_path / "accepted"),
            "--allow-incomplete",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert accepted.returncode == 0, accepted.stderr
    accepted_payload = json.loads(accepted.stdout)
    assert accepted_payload["status"] == "incomplete"
    assert verify_bundle_dir(tmp_path / "accepted") == []


@pytest.mark.parametrize(
    "path",
    (
        DEFAULT_ADAPTIVE_MATERIAL,
        DEFAULT_MATERIAL_PARITY,
        DEFAULT_MATERIAL_FIT,
        DEFAULT_COMPILED_LIE,
        DEFAULT_CONSTANT_TRANSFER,
        DEFAULT_SYNTHETIC_VISIBILITY,
    ),
)
def test_default_inputs_are_real_json_objects(path: Path) -> None:
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
