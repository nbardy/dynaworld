from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from research_experiments.paper_runner_suite.generate_world_tubes_concept_figures import (
    write_figures,
)
from research_experiments.paper_runner_suite.generate_world_tubes_paper_artifacts import (
    ARTIFACT_FILENAMES,
)
from research_experiments.paper_runner_suite.verify_world_tubes_iclr_package import (
    DEFAULT_BIBLIOGRAPHY,
    DEFAULT_DRAFT,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_VENUE_DIR,
    OFFICIAL_TEMPLATE_URL,
    REQUIRED_CONCEPT_FIGURES,
    REQUIRED_EVIDENCE_FIGURES,
    REQUIRED_EVIDENCE_COMPONENTS,
    REQUIRED_TABLE_FRAGMENTS,
    REQUIRED_VISUAL_QA_CHECKS,
    audit_pdf,
    parse_pdffonts,
    parse_pdfinfo,
    verify_ai_use_statement,
    verify_iclr_package,
    verify_source_cleanliness,
)
from research_experiments.paper_runner_suite.import_world_tubes_variable_camera_schema_v2 import (
    FROZEN_CONTRACT,
    verify_local_import,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_complete_evidence_bundle(path: Path) -> None:
    path.mkdir(parents=True)
    for filename in ARTIFACT_FILENAMES:
        if filename == "evidence_ledger.json":
            continue
        artifact = path / filename
        if artifact.suffix == ".json":
            artifact.write_text("{}\n", encoding="utf-8")
        elif artifact.suffix == ".svg":
            artifact.write_text("<svg><text>accepted evidence</text></svg>\n", encoding="utf-8")
        else:
            artifact.write_text("accepted evidence\n", encoding="utf-8")
    _report, variable_receipt = verify_local_import()
    components = {
        name: {
            "status": "accepted",
            "accepted": True,
            "input": f"accepted-fixture/{name}.json",
        }
        for name in REQUIRED_EVIDENCE_COMPONENTS
    }
    components["variable_camera_closure_death"].update(
        {
            "input_sha256": FROZEN_CONTRACT.raw_sha256,
            "compatibility_import": {
                "receipt_payload_sha256": variable_receipt[
                    "receipt_payload_sha256"
                ],
                "uses_current_schema_v1_runner": False,
            },
        }
    )
    ledger = {
        "status": "complete",
        "submission_ready": True,
        "readiness_scope": "evidence_artifact_bundle_only",
        "manuscript_package_required": True,
        "components": components,
    }
    ledger["ledger_sha256"] = _canonical_digest(ledger)
    _write_json(path / "evidence_ledger.json", ledger)
    artifacts = [
        {
            "path": filename,
            "bytes": (path / filename).stat().st_size,
            "sha256": _sha256(path / filename),
        }
        for filename in ARTIFACT_FILENAMES
    ]
    manifest = {
        "schema_version": 1,
        "generator": "world_tubes_paper_artifacts",
        "status": "complete",
        "submission_ready": True,
        "readiness_scope": "evidence_artifact_bundle_only",
        "manuscript_package_required": True,
        "ledger_sha256": ledger["ledger_sha256"],
        "artifacts": artifacts,
    }
    manifest["manifest_payload_sha256"] = _canonical_digest(manifest)
    _write_json(path / "artifact_manifest.json", manifest)


def _write_probe(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_pdf_probe_parsers_and_rejections(tmp_path: Path) -> None:
    info = parse_pdfinfo(
        "Pages: 12\nEncrypted: no\nPage size: 612 x 792 pts (letter)\n"
    )
    fonts = parse_pdffonts(
        "name type encoding emb sub uni object ID\n"
        "------------------------------------------\n"
        "CMR10 Type 1 Builtin yes yes yes 1 0\n"
        "Bad Type 3 Custom no no no 2 0\n"
    )

    assert info["Pages"] == "12"
    assert [font["name"] for font in fonts] == ["CMR10", "Bad"]

    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"0" * 2048)
    pdfinfo = tmp_path / "pdfinfo"
    pdffonts = tmp_path / "pdffonts"
    _write_probe(
        pdfinfo,
        "printf 'Pages: 12\\nEncrypted: no\\nPage size: 595 x 842 pts\\n'\n",
    )
    _write_probe(
        pdffonts,
        "printf 'Bad Type 3 Custom no no no 2 0\\n'\n",
    )

    errors, audit = audit_pdf(
        pdf,
        expected_total_pages=12,
        main_text_pages=10,
        pdfinfo_command=str(pdfinfo),
        pdffonts_command=str(pdffonts),
    )

    assert audit["page_count"] == 12
    assert any("main_text_pages" in error for error in errors)
    assert any("not US Letter" in error for error in errors)
    assert any("not embedded" in error for error in errors)
    assert any("Type 3" in error for error in errors)


def test_ai_use_statement_requires_actual_disclosure(tmp_path: Path) -> None:
    statement = tmp_path / "AI_USE_STATEMENT.md"
    statement.write_text("TODO: add wording\n", encoding="utf-8")
    errors, _audit = verify_ai_use_statement(statement)
    assert errors

    statement.write_text(
        "# AI use statement\n\n"
        "Generative-AI tools assisted with code review, experiment "
        "orchestration, mathematical checking, and manuscript editing. "
        "The authors reviewed and verified every generated change and remain "
        "fully responsible for the code, proofs, experiments, and claims.\n",
        encoding="utf-8",
    )
    errors, audit = verify_ai_use_statement(statement)
    assert errors == []
    assert audit["sha256"] == _sha256(statement)


def test_source_cleanliness_checks_both_repositories(tmp_path: Path) -> None:
    repositories = (tmp_path / "main", tmp_path / "star")
    for repository in repositories:
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=World Tubes Test",
                "-c",
                "user.email=world-tubes@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )

    errors, audit = verify_source_cleanliness(*repositories)
    assert errors == []
    assert all(record["clean"] for record in audit.values())

    (repositories[1] / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    errors, audit = verify_source_cleanliness(*repositories)
    assert errors == ["STAR UVT repository is dirty"]
    assert audit["STAR UVT repository"]["porcelain_count"] == 1


def test_complete_venue_fixture_binds_sources_evidence_figures_and_pdf(
    tmp_path: Path,
) -> None:
    draft = tmp_path / "WORLD_TUBES_ICLR_MAIN_DRAFT.md"
    draft.write_text(
        "---\nauthor: Anonymous\n---\n\nA cited result [@kerbl2023].\n",
        encoding="utf-8",
    )
    bibliography = tmp_path / "WORLD_TUBES_REFERENCES.bib"
    bibliography.write_text(
        "@inproceedings{kerbl2023, title={3D Gaussian Splatting}}\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence"
    _write_complete_evidence_bundle(evidence)
    concepts = tmp_path / "concepts"
    write_figures(concepts)

    venue = tmp_path / "venue"
    figure_dir = venue / "figures"
    figure_dir.mkdir(parents=True)
    style = venue / "iclr2027_conference.sty"
    style.write_text(
        "\\ProvidesPackage{iclr2027_conference}[2026/08/15 ICLR 2027]\n",
        encoding="utf-8",
    )
    archive = venue / "iclr2027.zip"
    archive.write_bytes(b"official-template-fixture")
    statement = venue / "AI_USE_STATEMENT.md"
    statement.write_text(
        "# AI use statement\n\n"
        "Generative-AI tools assisted with code review, experiment "
        "orchestration, mathematical checking, and manuscript editing. "
        "The authors reviewed and verified every generated change and remain "
        "fully responsible for all code, proofs, experiments, and claims.\n",
        encoding="utf-8",
    )

    source_figures = [
        *(concepts / name for name in REQUIRED_CONCEPT_FIGURES),
        *(evidence / name for name in REQUIRED_EVIDENCE_FIGURES),
    ]
    exports: list[dict[str, str]] = []
    export_paths: list[Path] = []
    for source in source_figures:
        export = figure_dir / f"{source.stem}.png"
        export.write_bytes(b"\x89PNG\r\n\x1a\n" + source.read_bytes())
        export_paths.append(export)
        exports.append({"source": str(source), "export": str(export.relative_to(venue))})

    main_tex = venue / "main.tex"
    lines = [
        "\\documentclass{article}",
        "\\usepackage{iclr2027_conference}",
        f"% WORLD_TUBES_SOURCE_SHA256: {_sha256(draft)}",
        f"% WORLD_TUBES_BIBLIOGRAPHY_SHA256: {_sha256(bibliography)}",
        (
            "% WORLD_TUBES_EVIDENCE_MANIFEST_SHA256: "
            f"{_sha256(evidence / 'artifact_manifest.json')}"
        ),
        f"% WORLD_TUBES_AI_USE_STATEMENT_SHA256: {_sha256(statement)}",
    ]
    lines.extend(f"\\input{{{evidence / name}}}" for name in REQUIRED_TABLE_FRAGMENTS)
    lines.extend(f"\\includegraphics{{figures/{path.name}}}" for path in export_paths)
    lines.append("\\label{world-tubes-main-end}")
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
        "\\newlabel{world-tubes-main-end}{{7}{9}{}{section.7}{}}\n",
        encoding="utf-8",
    )
    pdf = venue / "main.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"0" * 2048)
    pdf_digest = _sha256(pdf)
    manifest = {
        "schema_version": 1,
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
                "path": str(draft),
                "sha256": _sha256(draft),
                "bytes": draft.stat().st_size,
            },
            "bibliography": {
                "path": str(bibliography),
                "sha256": _sha256(bibliography),
                "bytes": bibliography.stat().st_size,
            },
            "evidence_manifest": {
                "path": str(evidence / "artifact_manifest.json"),
                "sha256": _sha256(evidence / "artifact_manifest.json"),
                "bytes": (evidence / "artifact_manifest.json").stat().st_size,
            },
            "ai_use_statement": {
                "path": str(statement),
                "sha256": _sha256(statement),
                "bytes": statement.stat().st_size,
            },
        },
        "asset_exports": exports,
        "pdf_visual_qa": {
            "status": "accepted",
            "pdf_sha256": pdf_digest,
            "page_count": 11,
            "inspected_pages": list(range(1, 12)),
            "reviewed_at": "2026-08-15",
            "checks": {key: True for key in REQUIRED_VISUAL_QA_CHECKS},
        },
    }
    _write_json(venue / "package_manifest.json", manifest)
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

    audit = verify_iclr_package(
        venue_dir=venue,
        draft_path=draft,
        bibliography_path=bibliography,
        evidence_dir=evidence,
        concept_dir=concepts,
        require_clean_source=False,
        pdfinfo_command=str(pdfinfo),
        pdffonts_command=str(pdffonts),
    )

    assert audit["errors"] == []
    assert audit["accepted"] is True
    assert audit["pdf_audit"]["page_count"] == 11

    archive.write_bytes(b"tampered-template-fixture")
    archive_rejected = verify_iclr_package(
        venue_dir=venue,
        draft_path=draft,
        bibliography_path=bibliography,
        evidence_dir=evidence,
        concept_dir=concepts,
        require_clean_source=False,
        pdfinfo_command=str(pdfinfo),
        pdffonts_command=str(pdffonts),
    )
    assert any(
        "template archive hash does not match" in error
        for error in archive_rejected["errors"]
    )
    archive.write_bytes(b"official-template-fixture")

    accepted_tex = main_tex.read_text(encoding="utf-8")
    theorem_input = f"\\input{{{evidence / REQUIRED_TABLE_FRAGMENTS[0]}}}"
    main_tex.write_text(
        accepted_tex.replace(theorem_input, f"% {theorem_input}"),
        encoding="utf-8",
    )
    commented = verify_iclr_package(
        venue_dir=venue,
        draft_path=draft,
        bibliography_path=bibliography,
        evidence_dir=evidence,
        concept_dir=concepts,
        require_clean_source=False,
        pdfinfo_command=str(pdfinfo),
        pdffonts_command=str(pdffonts),
    )
    assert any(
        "does not input evidence fragment" in error
        for error in commented["errors"]
    )
    main_tex.write_text(accepted_tex, encoding="utf-8")

    evidence_manifest = evidence / "artifact_manifest.json"
    accepted_manifest = evidence_manifest.read_text(encoding="utf-8")
    evidence_manifest.write_text(accepted_manifest + "\n", encoding="utf-8")
    stale_binding = verify_iclr_package(
        venue_dir=venue,
        draft_path=draft,
        bibliography_path=bibliography,
        evidence_dir=evidence,
        concept_dir=concepts,
        require_clean_source=False,
        pdfinfo_command=str(pdfinfo),
        pdffonts_command=str(pdffonts),
    )
    assert any(
        "source_bindings.evidence_manifest.sha256 is stale" in error
        for error in stale_binding["errors"]
    )
    evidence_manifest.write_text(accepted_manifest, encoding="utf-8")

    (evidence / "frozen_scaling_table.tex").write_text(
        "NOT SUBMISSION-READY\n",
        encoding="utf-8",
    )
    rejected = verify_iclr_package(
        venue_dir=venue,
        draft_path=draft,
        bibliography_path=bibliography,
        evidence_dir=evidence,
        concept_dir=concepts,
        require_clean_source=False,
        pdfinfo_command=str(pdfinfo),
        pdffonts_command=str(pdffonts),
    )
    assert rejected["accepted"] is False
    assert any("placeholder token" in error for error in rejected["errors"])


def test_checked_in_venue_scaffold_is_truthful_and_fail_closed() -> None:
    manifest = json.loads(
        (DEFAULT_VENUE_DIR / "package_manifest.json").read_text(encoding="utf-8")
    )
    template = manifest["template"]

    assert manifest["package_status"] == "scaffold_blocked"
    assert manifest["submission_ready"] is False
    assert template["status"] == "unavailable"
    assert template["source_url"] == OFFICIAL_TEMPLATE_URL
    assert template["archive_sha256"] is None
    assert template["style_sha256"] is None
    assert not (DEFAULT_VENUE_DIR / template["archive_path"]).exists()
    assert not (DEFAULT_VENUE_DIR / template["style_path"]).exists()

    expected_sources = {
        "concise_draft": DEFAULT_DRAFT,
        "bibliography": DEFAULT_BIBLIOGRAPHY,
        "evidence_manifest": DEFAULT_EVIDENCE_DIR / "artifact_manifest.json",
        "ai_use_statement": DEFAULT_VENUE_DIR / "AI_USE_STATEMENT.md",
    }
    for key, path in expected_sources.items():
        record = manifest["source_bindings"][key]
        assert (Path(__file__).resolve().parents[1] / record["path"]).resolve() == path.resolve()
        assert record["sha256"] == _sha256(path)
        assert record["bytes"] == path.stat().st_size

    main_tex = (DEFAULT_VENUE_DIR / "main.tex").read_text(encoding="utf-8")
    active_tex = "\n".join(
        line for line in main_tex.splitlines() if not line.lstrip().startswith("%")
    )
    assert "\\PackageError{world-tubes-venue}" in active_tex
    assert "\\input{../../generated/schema_v2/theorem_table.tex}" in active_tex
    assert "\\input{../../generated/schema_v2/variable_camera_table.tex}" in active_tex
    for filename in ("frozen_scaling_table.tex", "public_context_table.tex"):
        assert filename not in active_tex

    audit = verify_iclr_package(require_clean_source=False)
    assert audit["accepted"] is False
    assert any("package_status must be submission_candidate" in error for error in audit["errors"])
    assert any("submission_ready=true" in error for error in audit["errors"])
    assert any("template status must be acquired_official" in error for error in audit["errors"])
    assert any("template archive is missing" in error for error in audit["errors"])
    assert any("official ICLR style is missing" in error for error in audit["errors"])
    assert any("fail-closed scaffold stop" in error for error in audit["errors"])
    assert any("placeholder token" in error for error in audit["errors"])
