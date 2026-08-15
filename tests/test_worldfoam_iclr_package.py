from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research_experiments.paper_runner_suite.generate_worldfoam_concept_figures import (
    write_figures,
)
from research_experiments.paper_runner_suite.verify_worldfoam_iclr_package import (
    BUILD_KIND,
    DEFAULT_BIBLIOGRAPHY,
    DEFAULT_DRAFT,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_VENUE_DIR,
    OFFICIAL_TEMPLATE_URL,
    PAPER_ID,
    REQUIRED_CONCEPT_FIGURES,
    REQUIRED_EVIDENCE_FIGURES,
    REQUIRED_TABLE_FRAGMENTS,
    REQUIRED_VISUAL_QA_CHECKS,
    ROOT,
    verify_evidence_bundle,
    verify_iclr_package,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_probe(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _accepted_g6_report(_path: Path) -> dict[str, object]:
    return {
        "accepted": True,
        "observed_row_count": 12,
        "observed_control_row_count": 9,
    }


def _accepted_g4_report(_path: Path) -> dict[str, object]:
    return {
        "accepted": True,
        "observed_row_count": 36,
        "observed_scene_count": 3,
        "observed_seed_count": 3,
        "observed_route_count": 4,
    }


def _write_complete_fixture(tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "repository"
    paper_dir = repository / "research_notes" / "worldfoam_paper"
    evidence = paper_dir / "generated" / "foundation_v1"
    concepts = paper_dir / "figures"
    venue = paper_dir / "venue" / "iclr2027"
    evidence.mkdir(parents=True)
    venue.mkdir(parents=True)
    write_figures(concepts)

    draft = paper_dir / "WORLD_FOAM_ICLR_MAIN_DRAFT.md"
    generated_markers = "\n".join(
        f"<!-- GENERATED-TABLE:{name} -->" for name in REQUIRED_TABLE_FRAGMENTS
    )
    draft.write_text(
        "---\nauthor: Anonymous\n---\n\n"
        "A source-bound result [@kerbl2023].\n\n"
        f"{generated_markers}\n",
        encoding="utf-8",
    )
    bibliography = paper_dir / "WORLD_FOAM_REFERENCES.bib"
    bibliography.write_text(
        "@inproceedings{kerbl2023, title={3D Gaussian Splatting}}\n",
        encoding="utf-8",
    )

    raw_dir = repository / "outputs" / "paper_b"
    raw_dir.mkdir(parents=True)
    g4_raw = raw_dir / "public_quality_g4.json"
    g6_raw = raw_dir / "worldfoam_training_memory_ablation.json"
    _write_json(
        g4_raw,
        {
            "status": "measured",
            "proxy_or_test_artifact": False,
            "measurement_is_simulated": False,
            "public_quality_evidence": True,
        },
    )
    _write_json(
        g6_raw,
        {
            "status": "measured",
            "proxy_or_test_artifact": False,
            "measurement_is_simulated": False,
            "public_quality_evidence": False,
        },
    )

    gate_status = {
        "paper_ready": True,
        "iclr_ready": True,
        "gates": [
            {"gate": gate, "status": "accepted"}
            for gate in ("G0", "G1", "G2", "G3", "G4", "G6")
        ],
    }
    ledger = {
        "records": [
            {
                "evidence_id": "public_quality_g4",
                "gate": "G4",
                "status": "accepted",
                "numeric_rows_emitted": 36,
                "verifier": "strict_public_quality_v1",
                "scope": "fresh public heldout quality ablation",
                "errors": [],
                "path": str(g4_raw.relative_to(repository)),
                "sha256": _sha256(g4_raw),
            },
            {
                "evidence_id": "native_memory_work_g6",
                "gate": "G6",
                "status": "accepted",
                "numeric_rows_emitted": 21,
                "verifier": "verify_worldfoam_training_memory_ablation",
                "scope": "fresh-process production native memory ablation",
                "errors": [],
                "path": str(g6_raw.relative_to(repository)),
                "sha256": _sha256(g6_raw),
            },
        ]
    }
    _write_json(evidence / "gate_status.json", gate_status)
    _write_json(evidence / "evidence_ledger.json", ledger)
    for name in REQUIRED_TABLE_FRAGMENTS:
        (evidence / name).write_text(
            "% verifier-generated WorldFoam table\n"
            "\\begin{tabular}{ll} accepted & evidence \\\\ \\end{tabular}\n",
            encoding="utf-8",
        )
    for name in REQUIRED_EVIDENCE_FIGURES:
        (evidence / name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700">'
            "<text>accepted measured evidence</text></svg>\n",
            encoding="utf-8",
        )
    evidence_files = [
        evidence / "gate_status.json",
        evidence / "evidence_ledger.json",
        *(evidence / name for name in REQUIRED_TABLE_FRAGMENTS),
        *(evidence / name for name in REQUIRED_EVIDENCE_FIGURES),
    ]
    evidence_manifest = {
        "schema_version": 2,
        "generator": "worldfoam_paper_b_foundation_artifacts",
        "complete": True,
        "claims": {
            "synthetic_cpu_g0_g3": True,
            "public_quality": True,
            "native_memory_fit": True,
        },
        "files": [
            {
                "path": str(path.relative_to(evidence)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in evidence_files
        ],
    }
    _write_json(evidence / "manifest.json", evidence_manifest)

    style = venue / "iclr2027_conference.sty"
    style.write_text(
        "\\ProvidesPackage{iclr2027_conference}[2026/08/15 ICLR 2027]\n",
        encoding="utf-8",
    )
    archive = venue / "iclr2027.zip"
    archive.write_bytes(b"official-worldfoam-template-fixture")
    statement = venue / "AI_USE_STATEMENT.md"
    statement.write_text(
        "# AI use statement\n\n"
        "Generative-AI tools assisted with code review, experiment orchestration, "
        "mathematical checking, and manuscript editing. The authors reviewed and "
        "verified every generated change and remain fully responsible for the "
        "code, proofs, experiments, and claims in this submission.\n",
        encoding="utf-8",
    )

    figure_dir = venue / "figures"
    figure_dir.mkdir()
    sources = [
        *(concepts / name for name in REQUIRED_CONCEPT_FIGURES),
        *(evidence / name for name in REQUIRED_EVIDENCE_FIGURES),
    ]
    exports: list[dict[str, object]] = []
    export_paths: list[Path] = []
    for source in sources:
        export = figure_dir / f"{source.stem}.png"
        export.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 2048)
        export_paths.append(export)
        exports.append(
            {
                "source": str(source.relative_to(repository)),
                "source_sha256": _sha256(source),
                "export": str(export.relative_to(venue)),
                "export_sha256": _sha256(export),
                "export_bytes": export.stat().st_size,
            }
        )

    main_tex = venue / "main.tex"
    lines = [
        "\\documentclass{article}",
        "\\usepackage{iclr2027_conference}",
        f"% WORLD_FOAM_SOURCE_SHA256: {_sha256(draft)}",
        f"% WORLD_FOAM_BIBLIOGRAPHY_SHA256: {_sha256(bibliography)}",
        f"% WORLD_FOAM_EVIDENCE_MANIFEST_SHA256: {_sha256(evidence / 'manifest.json')}",
        f"% WORLD_FOAM_AI_USE_STATEMENT_SHA256: {_sha256(statement)}",
    ]
    lines.extend(f"\\input{{{evidence / name}}}" for name in REQUIRED_TABLE_FRAGMENTS)
    lines.extend(f"\\includegraphics{{figures/{path.name}}}" for path in export_paths)
    lines.append("\\label{worldfoam-main-end}")
    main_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    recorder = venue / "main.fls"
    recorder_inputs = [
        main_tex,
        style,
        bibliography,
        *(evidence / name for name in REQUIRED_TABLE_FRAGMENTS),
        *export_paths,
    ]
    recorder.write_text(
        "".join(f"INPUT {path.resolve()}\n" for path in recorder_inputs),
        encoding="utf-8",
    )
    auxiliary = venue / "main.aux"
    auxiliary.write_text(
        "\\newlabel{worldfoam-main-end}{{7}{9}{}{section.7}{}}\n",
        encoding="utf-8",
    )
    pdf = venue / "main.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"0" * 2048)
    package_manifest = {
        "schema_version": 1,
        "paper_id": PAPER_ID,
        "build_kind": BUILD_KIND,
        "venue": "ICLR",
        "venue_year": 2027,
        "anonymous_submission": True,
        "package_status": "submission_candidate",
        "submission_ready": True,
        "template": {
            "status": "acquired_official",
            "source_url": OFFICIAL_TEMPLATE_URL,
            "retrieved_at": "2026-08-15",
            "archive_path": archive.name,
            "archive_sha256": _sha256(archive),
            "style_path": style.name,
            "style_sha256": _sha256(style),
        },
        "entrypoint": main_tex.name,
        "recorder": recorder.name,
        "auxiliary": auxiliary.name,
        "compiled_pdf": pdf.name,
        "ai_use_statement": statement.name,
        "main_text_pages": 9,
        "total_pages": 11,
        "source_bindings": {
            "concise_draft": {
                "path": str(draft.relative_to(repository)),
                "sha256": _sha256(draft),
                "bytes": draft.stat().st_size,
            },
            "bibliography": {
                "path": str(bibliography.relative_to(repository)),
                "sha256": _sha256(bibliography),
                "bytes": bibliography.stat().st_size,
            },
            "evidence_manifest": {
                "path": str((evidence / "manifest.json").relative_to(repository)),
                "sha256": _sha256(evidence / "manifest.json"),
                "bytes": (evidence / "manifest.json").stat().st_size,
            },
            "ai_use_statement": {
                "path": str(statement.relative_to(repository)),
                "sha256": _sha256(statement),
                "bytes": statement.stat().st_size,
            },
        },
        "asset_exports": exports,
        "pdf_visual_qa": {
            "status": "accepted",
            "pdf_sha256": _sha256(pdf),
            "page_count": 11,
            "inspected_pages": list(range(1, 12)),
            "reviewed_at": "2026-08-15",
            "checks": {key: True for key in REQUIRED_VISUAL_QA_CHECKS},
        },
    }
    _write_json(venue / "package_manifest.json", package_manifest)

    pdfinfo = tmp_path / "pdfinfo"
    pdffonts = tmp_path / "pdffonts"
    _write_probe(
        pdfinfo,
        "printf 'Pages: 11\\nEncrypted: no\\nPage size: 612 x 792 pts (letter)\\n'\n",
    )
    _write_probe(
        pdffonts,
        "printf 'CMR10 Type 1 Builtin yes yes yes 1 0\\n'\n",
    )
    return {
        "repository": repository,
        "draft": draft,
        "bibliography": bibliography,
        "evidence": evidence,
        "concepts": concepts,
        "venue": venue,
        "pdfinfo": pdfinfo,
        "pdffonts": pdffonts,
        "g4_raw": g4_raw,
        "g6_raw": g6_raw,
    }


def _verify_fixture(paths: dict[str, object]) -> dict[str, object]:
    return verify_iclr_package(
        venue_dir=paths["venue"],
        draft_path=paths["draft"],
        bibliography_path=paths["bibliography"],
        evidence_dir=paths["evidence"],
        concept_dir=paths["concepts"],
        repository=paths["repository"],
        require_clean_source=False,
        evidence_verifier=lambda _path: [],
        g4_artifact_verifier=_accepted_g4_report,
        g6_artifact_verifier=_accepted_g6_report,
        pdfinfo_command=str(paths["pdfinfo"]),
        pdffonts_command=str(paths["pdffonts"]),
    )


def test_current_foundation_is_valid_but_g6_reports_zero_measured_rows() -> None:
    errors, audit = verify_evidence_bundle(DEFAULT_EVIDENCE_DIR, repository=ROOT)

    assert audit["foundation_integrity_verified"] is True
    assert audit["verified_measured_rows"]["G6"] == 0
    assert audit["verified_measured_rows"]["G4"] == 0
    assert any("evidence manifest is incomplete" in error for error in errors)
    assert any("no independently accepted G6 record" in error for error in errors)
    assert any("placeholder artifact" in error for error in errors)


def test_checked_in_venue_scaffold_is_truthful_and_fail_closed() -> None:
    manifest = json.loads(
        (DEFAULT_VENUE_DIR / "package_manifest.json").read_text(encoding="utf-8")
    )
    template = manifest["template"]

    assert manifest["package_status"] == "scaffold_blocked"
    assert manifest["submission_ready"] is False
    assert template["status"] == "unavailable"
    assert template["source_url"] == OFFICIAL_TEMPLATE_URL
    assert template["http_status"] == 404
    assert template["archive_sha256"] is None
    assert template["style_sha256"] is None
    assert not (DEFAULT_VENUE_DIR / template["archive_path"]).exists()
    assert not (DEFAULT_VENUE_DIR / template["style_path"]).exists()

    expected_sources = {
        "concise_draft": DEFAULT_DRAFT,
        "bibliography": DEFAULT_BIBLIOGRAPHY,
        "evidence_manifest": DEFAULT_EVIDENCE_DIR / "manifest.json",
        "ai_use_statement": DEFAULT_VENUE_DIR / "AI_USE_STATEMENT.md",
    }
    for key, path in expected_sources.items():
        record = manifest["source_bindings"][key]
        assert (ROOT / record["path"]).resolve() == path.resolve()
        assert record["sha256"] == _sha256(path)
        assert record["bytes"] == path.stat().st_size

    main_tex = (DEFAULT_VENUE_DIR / "main.tex").read_text(encoding="utf-8")
    active_tex = "\n".join(
        line for line in main_tex.splitlines() if not line.lstrip().startswith("%")
    )
    assert "\\PackageError{worldfoam-venue}" in active_tex
    assert "\\input{../../generated/foundation_v1/synthetic_visibility_table.tex}" in active_tex
    assert "g4_public_quality" not in active_tex.lower()
    assert "g6_native_memory" not in active_tex.lower()

    audit = verify_iclr_package(require_clean_source=False)
    assert audit["accepted"] is False
    assert any("package_status must be submission_candidate" in error for error in audit["errors"])
    assert any("submission_ready=true" in error for error in audit["errors"])
    assert any("template status must be acquired_official" in error for error in audit["errors"])
    assert any("template archive is missing" in error for error in audit["errors"])
    assert any("official ICLR style is missing" in error for error in audit["errors"])
    assert any("fail-closed scaffold stop" in error for error in audit["errors"])
    assert any("no independently accepted G4 record" in error for error in audit["errors"])
    assert any("no independently accepted G6 record" in error for error in audit["errors"])


def test_complete_fixture_binds_generated_results_assets_and_official_pdf(
    tmp_path: Path,
) -> None:
    paths = _write_complete_fixture(tmp_path)
    audit = _verify_fixture(paths)

    assert audit["errors"] == []
    assert audit["accepted"] is True
    assert audit["inputs"]["evidence"]["verified_measured_rows"] == {
        "G4": 36,
        "G6": 21,
    }


def test_gate_rejects_proxy_evidence_hand_copied_results_and_generic_pdf(
    tmp_path: Path,
) -> None:
    paths = _write_complete_fixture(tmp_path)
    g6_raw = paths["g6_raw"]
    payload = json.loads(g6_raw.read_text(encoding="utf-8"))
    payload["proxy_or_test_artifact"] = True
    _write_json(g6_raw, payload)
    ledger_path = paths["evidence"] / "evidence_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["records"][1]["sha256"] = _sha256(g6_raw)
    ledger["records"][1]["scope"] = "dry-run proxy memory test"
    _write_json(ledger_path, ledger)
    draft = paths["draft"]
    draft.write_text(
        draft.read_text(encoding="utf-8")
        + "\n| Method | PSNR | RSS peak |\n|---|---:|---:|\n| fake | 99 | 1 |\n",
        encoding="utf-8",
    )
    package_manifest_path = paths["venue"] / "package_manifest.json"
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    generic_pdf = paths["venue"] / "worldfoam_iclr_generic_qa.pdf"
    generic_pdf.write_bytes((paths["venue"] / "main.pdf").read_bytes())
    package_manifest["compiled_pdf"] = generic_pdf.name
    _write_json(package_manifest_path, package_manifest)

    audit = _verify_fixture(paths)

    assert audit["accepted"] is False
    assert audit["inputs"]["evidence"]["verified_measured_rows"]["G6"] == 0
    assert any("dry-run/test/proxy evidence" in error for error in audit["errors"])
    assert any("hand-copied result table" in error for error in audit["errors"])
    assert any("generic QA PDFs" in error for error in audit["errors"])
