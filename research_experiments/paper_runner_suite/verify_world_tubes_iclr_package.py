from __future__ import annotations

"""Fail-closed verifier for the concise World Tubes ICLR submission package.

This verifier is deliberately separate from the evidence-bundle generator.
The generator certifies numeric inputs; this module certifies that a venue
package consumes those inputs and that the rendered PDF is the inspected,
anonymous ICLR artifact built from the current concise manuscript source.

The expected venue tree is::

    research_notes/gauged_uvt_trace_atlas/paper/venue/iclr2027/
      package_manifest.json
      iclr2027.zip
      iclr2027_conference.sty
      main.tex
      main.fls
      main.aux
      main.pdf
      AI_USE_STATEMENT.md
      figures/*.pdf (or *.png)

``package_manifest.json`` records official-template provenance, the main-text
page budget, portable exports of all required concept/evidence figures, and a
page-by-page visual-QA acknowledgement.  The generated ``main.tex`` must carry
four source-binding comments (reported verbatim in verifier errors when
missing): concise Markdown, bibliography, evidence manifest, and AI statement.

No option weakens a publication gate.  Missing evidence, dirty source, an
absent venue tree, a stale PDF, or an incomplete audit always exits non-zero.
The optional JSON report records hashes and failures; it does not promote an
incomplete package.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAPER_DIR = (
    ROOT / "research_notes" / "gauged_uvt_trace_atlas" / "paper"
)
DEFAULT_DRAFT = PAPER_DIR / "WORLD_TUBES_ICLR_MAIN_DRAFT.md"
DEFAULT_BIBLIOGRAPHY = PAPER_DIR / "WORLD_TUBES_REFERENCES.bib"
DEFAULT_EVIDENCE_DIR = PAPER_DIR / "generated" / "schema_v2"
DEFAULT_CONCEPT_DIR = PAPER_DIR / "figures"
DEFAULT_VENUE_DIR = PAPER_DIR / "venue" / "iclr2027"
DEFAULT_STAR_REPOSITORY = ROOT / "third_party" / "fast-mac-gsplat"

PACKAGE_SCHEMA_VERSION = 1
VENUE_NAME = "ICLR"
VENUE_YEAR = 2027
MAX_MAIN_TEXT_PAGES = 9
READY_PACKAGE_STATUS = "submission_candidate"
READY_TEMPLATE_STATUS = "acquired_official"
OFFICIAL_TEMPLATE_URL = (
    "https://github.com/ICLR/Master-Template/raw/master/iclr2027.zip"
)

REQUIRED_TABLE_FRAGMENTS = (
    "theorem_table.tex",
    "frozen_scaling_table.tex",
    "variable_camera_table.tex",
    "public_context_table.tex",
)
REQUIRED_EVIDENCE_FIGURES = (
    "frozen_scaling.svg",
    "variable_camera_closure_death.svg",
    "public_heldout_quality.svg",
    "public_cost_and_storage.svg",
)
REQUIRED_CONCEPT_FIGURES = (
    "world_tubes_system_overview.svg",
    "world_tubes_projective_compiler.svg",
)
REQUIRED_VISUAL_QA_CHECKS = (
    "no_clipped_content",
    "readable_labels",
    "resolved_cross_references",
    "no_placeholders",
    "anonymous_submission",
    "captions_and_error_bars_checked",
)
REQUIRED_EVIDENCE_COMPONENTS = (
    "theorem_correctness",
    "variable_camera_closure_death",
    "frozen_world_scaling",
    "moving_camera_density",
    "public_context",
)
FORBIDDEN_SUBMISSION_TOKENS = (
    "NOT SUBMISSION-READY",
    "No numeric rows emitted",
    "TODO",
    "TBD",
)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _resolve_inside(base: Path, value: Any, *, label: str) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, f"{label} must be a non-empty path"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None, f"{label} escapes the venue directory: {value}"
    return candidate, None


def _resolve_source(
    value: Any,
    *,
    label: str,
    allowed_roots: Sequence[Path] = (ROOT,),
) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, f"{label} must be a non-empty path"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not any(
        candidate == root.resolve() or root.resolve() in candidate.parents
        for root in allowed_roots
    ):
        return None, f"{label} is outside the allowed source roots: {value}"
    return candidate, None


def _read_text(path: Path, label: str, errors: list[str]) -> str:
    if not path.is_file():
        errors.append(f"{label} is missing: {_display_path(path)}")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"{label} is not UTF-8: {_display_path(path)}")
        return ""


def _verify_citations(draft: str, bibliography: str) -> list[str]:
    errors: list[str] = []
    citation_keys = set(
        re.findall(r"@([A-Za-z0-9][A-Za-z0-9_.:+-]*)", draft)
    )
    bibliography_keys = set(
        re.findall(
            r"@[A-Za-z]+\s*\{\s*([^,\s]+)",
            bibliography,
            flags=re.IGNORECASE,
        )
    )
    for key in sorted(citation_keys - bibliography_keys):
        errors.append(f"bibliography is missing cited key: {key}")
    if not citation_keys:
        errors.append("concise manuscript contains no citation keys")
    return errors


def _strip_comments(text: str) -> str:
    without_html = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return "\n".join(
        line for line in without_html.splitlines() if not line.lstrip().startswith("%")
    )


def _strip_tex_comments(text: str) -> str:
    return "\n".join(
        re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines()
    )


def verify_ai_use_statement(path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    text = _read_text(path, "AI-use statement", errors)
    visible = _strip_comments(text).strip()
    if text and len(visible) < 120:
        errors.append("AI-use statement is too short to be an author-approved disclosure")
    if text and not re.search(
        r"(?:generative[- ]AI|artificial intelligence|large language model|AI tools?)",
        visible,
        flags=re.IGNORECASE,
    ):
        errors.append("AI-use statement does not identify generative-AI use")
    if text and not re.search(
        r"author(?:s)?[^.]{0,160}(?:responsib|review|verif|accountable)",
        visible,
        flags=re.IGNORECASE,
    ):
        errors.append("AI-use statement does not state author responsibility or review")
    if text and not re.search(
        r"(?:code|experiment|mathemat|manuscript|editing|orchestration)",
        visible,
        flags=re.IGNORECASE,
    ):
        errors.append("AI-use statement does not describe how the tools were used")
    if text and re.search(r"\b(?:TODO|TBD|placeholder)\b", visible, re.IGNORECASE):
        errors.append("AI-use statement still contains placeholder language")
    return errors, {
        "path": _display_path(path),
        "sha256": _sha256(path) if path.is_file() else None,
        "bytes": path.stat().st_size if path.is_file() else None,
    }


def parse_pdfinfo(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def parse_pdffonts(output: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    row_pattern = re.compile(
        r"^(?P<name>\S+)\s+(?P<type>.+?)\s+(?P<encoding>\S+)\s+"
        r"(?P<embedded>yes|no)\s+(?P<subset>yes|no)\s+"
        r"(?P<unicode>yes|no)\s+\d+\s+\d+\s*$",
        flags=re.IGNORECASE,
    )
    for line in output.splitlines():
        match = row_pattern.match(line.strip())
        if match:
            rows.append({key: value for key, value in match.groupdict().items()})
    return rows


def _run_probe(command: str, pdf_path: Path) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            [command, str(pdf_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return -1, "", str(error)
    return result.returncode, result.stdout, result.stderr


def audit_pdf(
    pdf_path: Path,
    *,
    expected_total_pages: Any,
    main_text_pages: Any,
    pdfinfo_command: str = "pdfinfo",
    pdffonts_command: str = "pdffonts",
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    audit: dict[str, Any] = {
        "path": _display_path(pdf_path),
        "sha256": None,
        "bytes": None,
        "page_count": None,
        "main_text_pages": main_text_pages,
        "fonts": [],
    }
    if not pdf_path.is_file():
        errors.append(f"compiled PDF is missing: {_display_path(pdf_path)}")
        return errors, audit
    audit["sha256"] = _sha256(pdf_path)
    audit["bytes"] = pdf_path.stat().st_size
    if pdf_path.stat().st_size < 1024:
        errors.append("compiled PDF is implausibly small")

    returncode, stdout, stderr = _run_probe(pdfinfo_command, pdf_path)
    if returncode != 0:
        errors.append(f"pdfinfo failed ({returncode}): {stderr.strip()}")
        info: dict[str, str] = {}
    else:
        info = parse_pdfinfo(stdout)
    try:
        page_count = int(info.get("Pages", ""))
    except ValueError:
        page_count = 0
    if page_count < 1:
        errors.append("pdfinfo did not report a positive page count")
    else:
        audit["page_count"] = page_count
    if isinstance(expected_total_pages, bool) or not isinstance(expected_total_pages, int):
        errors.append("package manifest total_pages must be an integer")
    elif page_count and expected_total_pages != page_count:
        errors.append(
            f"compiled PDF page count {page_count} does not match manifest "
            f"total_pages {expected_total_pages}"
        )
    if isinstance(main_text_pages, bool) or not isinstance(main_text_pages, int):
        errors.append("package manifest main_text_pages must be an integer")
    elif not 1 <= main_text_pages <= MAX_MAIN_TEXT_PAGES:
        errors.append(
            f"main_text_pages must be between 1 and {MAX_MAIN_TEXT_PAGES}"
        )
    allowed_encryption_descriptions = {
        "no",
        "no (print:yes copy:yes change:yes addnotes:yes algorithm:none)",
    }
    if info.get("Encrypted", "no").lower() not in allowed_encryption_descriptions:
        errors.append("compiled PDF must not be encrypted")
    page_size = info.get("Page size", "")
    page_match = re.search(r"([0-9.]+)\s+x\s+([0-9.]+)\s+pts", page_size)
    if not page_match:
        errors.append("pdfinfo did not report a parseable page size")
    else:
        width, height = (float(page_match.group(1)), float(page_match.group(2)))
        if abs(width - 612.0) > 1.0 or abs(height - 792.0) > 1.0:
            errors.append(
                f"compiled PDF is not US Letter size: {width:g} x {height:g} pts"
            )

    returncode, stdout, stderr = _run_probe(pdffonts_command, pdf_path)
    if returncode != 0:
        errors.append(f"pdffonts failed ({returncode}): {stderr.strip()}")
        fonts: list[dict[str, str]] = []
    else:
        fonts = parse_pdffonts(stdout)
    audit["fonts"] = fonts
    if not fonts:
        errors.append("pdffonts reported no auditable fonts")
    for font in fonts:
        if font["embedded"].lower() != "yes":
            errors.append(f"PDF font is not embedded: {font['name']}")
        if "type 3" in font["type"].lower():
            errors.append(f"PDF uses forbidden Type 3 font: {font['name']}")
    return errors, audit


def _git_status(repository: Path) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, str(error)
    if result.returncode != 0:
        return None, result.stderr.strip() or f"git exited {result.returncode}"
    return result.stdout, None


def verify_source_cleanliness(
    repository: Path = ROOT,
    star_repository: Path = DEFAULT_STAR_REPOSITORY,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    audit: dict[str, Any] = {}
    for label, path in (
        ("repository", repository),
        ("STAR UVT repository", star_repository),
    ):
        status, error = _git_status(path)
        audit[label] = {
            "path": _display_path(path),
            "clean": status == "" if status is not None else False,
            "porcelain_count": len(status.splitlines()) if status else 0,
            "porcelain_sha256": (
                hashlib.sha256(status.encode("utf-8")).hexdigest()
                if status
                else hashlib.sha256(b"").hexdigest()
            ),
            "porcelain_sample": status.splitlines()[:20] if status else [],
        }
        if error:
            errors.append(f"could not inspect {label} cleanliness: {error}")
        elif status:
            errors.append(f"{label} is dirty")
    return errors, audit


def _artifact_manifest_entries(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return {}
    return {
        str(record.get("path")): record
        for record in artifacts
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }


def verify_evidence_bundle(evidence_dir: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    try:
        from research_experiments.paper_runner_suite.generate_world_tubes_paper_artifacts import (
            verify_bundle_dir,
        )
    except Exception as error:  # pragma: no cover - import failure is a user-facing gate
        return [f"could not import evidence verifier: {error}"], {}
    errors.extend(verify_bundle_dir(evidence_dir, require_complete=True))
    manifest_path = evidence_dir / "artifact_manifest.json"
    ledger_path = evidence_dir / "evidence_ledger.json"
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"could not load evidence artifact manifest: {error}")
        manifest = {}
    try:
        ledger = _load_json(ledger_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"could not load evidence ledger: {error}")
        ledger = {}
    components = ledger.get("components")
    if not isinstance(components, Mapping):
        errors.append("evidence ledger components must be an object")
        components = {}
    for name in REQUIRED_EVIDENCE_COMPONENTS:
        component = components.get(name)
        if not isinstance(component, Mapping):
            errors.append(f"evidence ledger is missing required component: {name}")
        elif component.get("accepted") is not True:
            errors.append(f"required evidence component is not accepted: {name}")

    variable_import_audit: dict[str, Any] = {}
    try:
        from research_experiments.paper_runner_suite.import_world_tubes_variable_camera_schema_v2 import (
            DEFAULT_LOCAL_EVIDENCE_DIR,
            FROZEN_CONTRACT,
            verify_local_import,
        )

        _report, receipt = verify_local_import(DEFAULT_LOCAL_EVIDENCE_DIR)
        variable = components.get("variable_camera_closure_death", {})
        compatibility = (
            variable.get("compatibility_import", {})
            if isinstance(variable, Mapping)
            else {}
        )
        if compatibility.get("receipt_payload_sha256") != receipt.get(
            "receipt_payload_sha256"
        ):
            errors.append(
                "variable-camera evidence ledger does not bind the verified "
                "schema-v2 compatibility receipt"
            )
        if variable.get("input_sha256") != FROZEN_CONTRACT.raw_sha256:
            errors.append(
                "variable-camera evidence ledger does not bind the exact clean "
                "schema-v2 raw artifact"
            )
        if compatibility.get("uses_current_schema_v1_runner") is not False:
            errors.append(
                "variable-camera schema-v2 evidence must not use the current "
                "schema-v1 runner"
            )
        variable_import_audit = {
            "status": "accepted",
            "source_artifact_sha256": FROZEN_CONTRACT.raw_sha256,
            "receipt_payload_sha256": receipt.get("receipt_payload_sha256"),
            "uses_current_schema_v1_runner": False,
        }
    except Exception as error:  # pragma: no cover - user-facing provenance gate
        errors.append(f"variable-camera schema-v2 compatibility import is invalid: {error}")
        variable_import_audit = {"status": "invalid", "error": str(error)}
    entries = _artifact_manifest_entries(manifest)
    for filename in (*REQUIRED_TABLE_FRAGMENTS, *REQUIRED_EVIDENCE_FIGURES):
        if filename not in entries:
            errors.append(f"evidence manifest does not bind required artifact: {filename}")
        path = evidence_dir / filename
        if not path.is_file():
            errors.append(f"required evidence artifact is missing: {_display_path(path)}")
            continue
        if filename.endswith((".tex", ".svg")):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_SUBMISSION_TOKENS:
                if token in text:
                    errors.append(f"{filename} contains incomplete placeholder token: {token}")
    audit = {
        "directory": _display_path(evidence_dir),
        "manifest": _display_path(manifest_path),
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "ledger_sha256": _sha256(ledger_path) if ledger_path.is_file() else None,
        "submission_ready": manifest.get("submission_ready"),
        "status": manifest.get("status"),
        "variable_camera_schema_v2_import": variable_import_audit,
    }
    return sorted(set(errors)), audit


def verify_concept_figures(concept_dir: Path) -> tuple[list[str], dict[str, Any]]:
    try:
        from research_experiments.paper_runner_suite.generate_world_tubes_concept_figures import (
            verify_figure_dir,
        )
    except Exception as error:  # pragma: no cover - import failure is a user-facing gate
        return [f"could not import concept-figure verifier: {error}"], {}
    errors = verify_figure_dir(concept_dir)
    figures = []
    for filename in REQUIRED_CONCEPT_FIGURES:
        path = concept_dir / filename
        figures.append(
            {
                "path": _display_path(path),
                "sha256": _sha256(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return sorted(set(errors)), {"figures": figures}


def _binding_line(label: str, digest: str) -> str:
    return f"% WORLD_TUBES_{label}_SHA256: {digest}"


def _recorder_inputs(recorder_path: Path, venue_dir: Path) -> tuple[list[Path], list[str]]:
    errors: list[str] = []
    text = _read_text(recorder_path, "LaTeX recorder file", errors)
    inputs: list[Path] = []
    for line in text.splitlines():
        if not line.startswith("INPUT "):
            continue
        value = line[6:].strip()
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = venue_dir / candidate
        inputs.append(candidate.resolve())
    if text and not inputs:
        errors.append("LaTeX recorder contains no INPUT records")
    return inputs, errors


def _contains_recorded_path(inputs: Sequence[Path], path: Path) -> bool:
    resolved = path.resolve()
    return any(candidate == resolved for candidate in inputs)


def _verify_visual_qa(
    record: Any,
    *,
    pdf_audit: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ["package manifest pdf_visual_qa must be an object"]
    if record.get("status") != "accepted":
        errors.append("PDF visual QA status must be accepted")
    if record.get("pdf_sha256") != pdf_audit.get("sha256"):
        errors.append("PDF visual QA is not bound to the current compiled PDF")
    if record.get("page_count") != pdf_audit.get("page_count"):
        errors.append("PDF visual QA page_count does not match the compiled PDF")
    page_count = pdf_audit.get("page_count")
    inspected = record.get("inspected_pages")
    expected_pages = list(range(1, int(page_count) + 1)) if isinstance(page_count, int) else []
    if inspected != expected_pages:
        errors.append("PDF visual QA must list every rendered page exactly once")
    checks = record.get("checks")
    if not isinstance(checks, Mapping):
        errors.append("PDF visual QA checks must be an object")
    else:
        for key in REQUIRED_VISUAL_QA_CHECKS:
            if checks.get(key) is not True:
                errors.append(f"PDF visual QA check is not accepted: {key}")
    reviewed_at = record.get("reviewed_at")
    try:
        date.fromisoformat(str(reviewed_at))
    except ValueError:
        errors.append("PDF visual QA reviewed_at must be an ISO date")
    return errors


def _main_text_page_from_aux(path: Path, errors: list[str]) -> int | None:
    text = _read_text(path, "LaTeX auxiliary file", errors)
    if not text:
        return None
    match = re.search(
        r"\\newlabel\{world-tubes-main-end\}\{\{[^}]*\}\{([0-9]+)\}",
        text,
    )
    if not match:
        errors.append(
            "LaTeX auxiliary file does not resolve label world-tubes-main-end"
        )
        return None
    return int(match.group(1))


def _verify_manifest_header(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        errors.append(
            f"package manifest schema_version must be {PACKAGE_SCHEMA_VERSION}"
        )
    if manifest.get("venue") != VENUE_NAME:
        errors.append(f"package manifest venue must be {VENUE_NAME}")
    if manifest.get("venue_year") != VENUE_YEAR:
        errors.append(f"package manifest venue_year must be {VENUE_YEAR}")
    if manifest.get("anonymous_submission") is not True:
        errors.append("package manifest must declare anonymous_submission=true")
    if manifest.get("package_status") != READY_PACKAGE_STATUS:
        errors.append(
            "package manifest package_status must be "
            f"{READY_PACKAGE_STATUS}"
        )
    if manifest.get("submission_ready") is not True:
        errors.append("package manifest must declare submission_ready=true")
    return errors


def _verify_source_bindings(
    manifest: Mapping[str, Any],
    *,
    draft_path: Path,
    bibliography_path: Path,
    evidence_manifest_path: Path,
    ai_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Verify the package manifest's byte-exact bindings to authored inputs.

    The comments in ``main.tex`` make stale generated TeX visible.  The
    manifest bindings are a separate, machine-readable provenance contract;
    neither may stand in for the other.
    """

    errors: list[str] = []
    records = manifest.get("source_bindings")
    if not isinstance(records, Mapping):
        return ["package manifest source_bindings must be an object"], {}

    expected = {
        "concise_draft": draft_path,
        "bibliography": bibliography_path,
        "evidence_manifest": evidence_manifest_path,
        "ai_use_statement": ai_path,
    }
    audit: dict[str, Any] = {}
    allowed_roots = tuple(
        dict.fromkeys(
            (
                ROOT.resolve(),
                draft_path.parent.resolve(),
                bibliography_path.parent.resolve(),
                evidence_manifest_path.parent.resolve(),
                ai_path.parent.resolve(),
            )
        )
    )
    for key, expected_path in expected.items():
        record = records.get(key)
        if not isinstance(record, Mapping):
            errors.append(f"source_bindings.{key} must be an object")
            continue
        bound_path, path_error = _resolve_source(
            record.get("path"),
            label=f"source_bindings.{key}.path",
            allowed_roots=allowed_roots,
        )
        if path_error:
            errors.append(path_error)
        elif bound_path != expected_path.resolve():
            errors.append(
                f"source_bindings.{key}.path does not identify the verified input"
            )

        digest = record.get("sha256")
        byte_count = record.get("bytes")
        if not _is_sha256(digest):
            errors.append(f"source_bindings.{key}.sha256 is invalid")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            errors.append(f"source_bindings.{key}.bytes must be a nonnegative integer")
        if expected_path.is_file():
            actual_digest = _sha256(expected_path)
            actual_bytes = expected_path.stat().st_size
            if _is_sha256(digest) and digest != actual_digest:
                errors.append(f"source_bindings.{key}.sha256 is stale")
            if isinstance(byte_count, int) and not isinstance(byte_count, bool):
                if byte_count != actual_bytes:
                    errors.append(f"source_bindings.{key}.bytes is stale")
            audit[key] = {
                "path": _display_path(expected_path),
                "sha256": actual_digest,
                "bytes": actual_bytes,
            }
        else:
            errors.append(
                f"source binding input is missing: {_display_path(expected_path)}"
            )
            audit[key] = {
                "path": _display_path(expected_path),
                "sha256": None,
                "bytes": None,
            }

    unknown = sorted(set(records) - set(expected))
    if unknown:
        errors.append(
            "package manifest source_bindings has unknown entries: "
            + ", ".join(unknown)
        )
    return errors, audit


def _load_package_manifest(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, [f"venue package manifest is missing: {_display_path(path)}"]
    try:
        return _load_json(path), []
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {}, [f"could not load venue package manifest: {error}"]


def verify_iclr_package(
    *,
    venue_dir: Path = DEFAULT_VENUE_DIR,
    draft_path: Path = DEFAULT_DRAFT,
    bibliography_path: Path = DEFAULT_BIBLIOGRAPHY,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    concept_dir: Path = DEFAULT_CONCEPT_DIR,
    repository: Path = ROOT,
    star_repository: Path = DEFAULT_STAR_REPOSITORY,
    require_clean_source: bool = True,
    pdfinfo_command: str = "pdfinfo",
    pdffonts_command: str = "pdffonts",
) -> dict[str, Any]:
    """Return a deterministic audit dictionary; ``accepted`` is fail-closed."""

    venue_dir = venue_dir.resolve()
    errors: list[str] = []
    draft = _read_text(draft_path, "concise manuscript", errors)
    bibliography = _read_text(bibliography_path, "bibliography", errors)
    if draft and "WORLD_TUBES_ICLR_MAIN_DRAFT" not in draft_path.name:
        errors.append("venue verifier is not bound to the concise ICLR manuscript")
    if draft and not re.search(r"^author:\s*Anonymous\s*$", draft, re.MULTILINE):
        errors.append("concise manuscript front matter is not anonymous")
    if "| Contract | Metric | Result | Gate |" in draft:
        errors.append(
            "concise manuscript still contains a hand-copied theorem table; "
            "the venue package must consume theorem_table.tex"
        )
    errors.extend(_verify_citations(draft, bibliography))

    evidence_errors, evidence_audit = verify_evidence_bundle(evidence_dir)
    concept_errors, concept_audit = verify_concept_figures(concept_dir)
    errors.extend(evidence_errors)
    errors.extend(concept_errors)

    manifest_path = venue_dir / "package_manifest.json"
    manifest, manifest_errors = _load_package_manifest(manifest_path)
    errors.extend(manifest_errors)
    errors.extend(_verify_manifest_header(manifest))

    template = manifest.get("template")
    if not isinstance(template, Mapping):
        errors.append("package manifest template must be an object")
        template = {}
    if template.get("source_url") != OFFICIAL_TEMPLATE_URL:
        errors.append("package manifest does not record the official ICLR 2027 template URL")
    template_status = template.get("status")
    if template_status != READY_TEMPLATE_STATUS:
        errors.append(
            "official ICLR 2027 template status must be "
            f"{READY_TEMPLATE_STATUS}; found {template_status!r}"
        )
    archive_sha256 = template.get("archive_sha256")
    if not _is_sha256(archive_sha256):
        errors.append("package manifest template.archive_sha256 is invalid")
    try:
        date.fromisoformat(str(template.get("retrieved_at")))
    except ValueError:
        errors.append("package manifest template.retrieved_at must be an ISO date")

    path_specs = (
        (
            "archive_path",
            template.get("archive_path", "iclr2027.zip"),
            "official ICLR template archive",
        ),
        ("style_path", template.get("style_path", "iclr2027_conference.sty"), "ICLR style"),
        ("entrypoint", manifest.get("entrypoint", "main.tex"), "venue TeX entrypoint"),
        ("recorder", manifest.get("recorder", "main.fls"), "LaTeX recorder file"),
        ("auxiliary", manifest.get("auxiliary", "main.aux"), "LaTeX auxiliary file"),
        ("compiled_pdf", manifest.get("compiled_pdf", "main.pdf"), "compiled PDF"),
        (
            "ai_use_statement",
            manifest.get("ai_use_statement", "AI_USE_STATEMENT.md"),
            "AI-use statement",
        ),
    )
    resolved_paths: dict[str, Path] = {}
    for key, value, label in path_specs:
        path, error = _resolve_inside(venue_dir, value, label=label)
        if error:
            errors.append(error)
        elif path is not None:
            resolved_paths[key] = path

    archive_path = resolved_paths.get("archive_path", venue_dir / "iclr2027.zip")
    style_path = resolved_paths.get("style_path", venue_dir / "iclr2027_conference.sty")
    main_tex_path = resolved_paths.get("entrypoint", venue_dir / "main.tex")
    recorder_path = resolved_paths.get("recorder", venue_dir / "main.fls")
    auxiliary_path = resolved_paths.get("auxiliary", venue_dir / "main.aux")
    pdf_path = resolved_paths.get("compiled_pdf", venue_dir / "main.pdf")
    ai_path = resolved_paths.get("ai_use_statement", venue_dir / "AI_USE_STATEMENT.md")

    if not archive_path.is_file():
        errors.append(
            "official ICLR template archive is missing: "
            f"{_display_path(archive_path)}"
        )
    elif _is_sha256(archive_sha256) and archive_sha256 != _sha256(archive_path):
        errors.append("official ICLR template archive hash does not match package manifest")

    style = _read_text(style_path, "official ICLR style", errors)
    if style and not re.search(r"ICLR", style, re.IGNORECASE):
        errors.append("style file does not identify ICLR")
    if style and "2027" not in style:
        errors.append("style file does not identify the 2027 venue year")
    recorded_style_sha256 = template.get("style_sha256")
    if not _is_sha256(recorded_style_sha256):
        errors.append("package manifest template.style_sha256 is invalid")
    elif style_path.is_file() and recorded_style_sha256 != _sha256(style_path):
        errors.append("official ICLR style hash does not match package manifest")

    ai_errors, ai_audit = verify_ai_use_statement(ai_path)
    errors.extend(ai_errors)
    source_binding_errors, source_binding_audit = _verify_source_bindings(
        manifest,
        draft_path=draft_path,
        bibliography_path=bibliography_path,
        evidence_manifest_path=evidence_dir / "artifact_manifest.json",
        ai_path=ai_path,
    )
    errors.extend(source_binding_errors)
    main_tex = _read_text(main_tex_path, "venue TeX entrypoint", errors)
    active_main_tex = _strip_tex_comments(main_tex)
    style_stem = style_path.stem
    if main_tex and not re.search(
        rf"\\usepackage(?:\[[^\]]*\])?\{{[^}}]*\b{re.escape(style_stem)}\b[^}}]*\}}",
        active_main_tex,
    ):
        errors.append(f"venue TeX does not load official style {style_stem}")
    if main_tex and "\\iclrfinalcopy" in active_main_tex:
        errors.append("anonymous submission must not enable \\iclrfinalcopy")
    if main_tex and "\\PackageError{world-tubes-venue}" in active_main_tex:
        errors.append("venue TeX still contains the fail-closed scaffold stop")
    if main_tex and "\\label{world-tubes-main-end}" not in active_main_tex:
        errors.append(
            "venue TeX must label the end of main text as world-tubes-main-end"
        )

    source_bindings = {
        "SOURCE": _sha256(draft_path) if draft_path.is_file() else "missing",
        "BIBLIOGRAPHY": (
            _sha256(bibliography_path) if bibliography_path.is_file() else "missing"
        ),
        "EVIDENCE_MANIFEST": (
            _sha256(evidence_dir / "artifact_manifest.json")
            if (evidence_dir / "artifact_manifest.json").is_file()
            else "missing"
        ),
        "AI_USE_STATEMENT": _sha256(ai_path) if ai_path.is_file() else "missing",
    }
    for label, digest in source_bindings.items():
        expected = _binding_line(label, digest)
        if expected not in main_tex:
            errors.append(f"venue TeX is missing source binding: {expected}")

    for filename in REQUIRED_TABLE_FRAGMENTS:
        if not re.search(
            rf"\\input\{{[^}}]*{re.escape(filename)}\}}",
            active_main_tex,
        ):
            errors.append(f"venue TeX does not input evidence fragment: {filename}")

    asset_exports = manifest.get("asset_exports")
    if not isinstance(asset_exports, list):
        errors.append("package manifest asset_exports must be a list")
        asset_exports = []
    required_sources = {
        *(concept_dir / name for name in REQUIRED_CONCEPT_FIGURES),
        *(evidence_dir / name for name in REQUIRED_EVIDENCE_FIGURES),
    }
    exported_sources: dict[Path, Path] = {}
    for index, record in enumerate(asset_exports):
        if not isinstance(record, Mapping):
            errors.append(f"asset_exports[{index}] must be an object")
            continue
        source, source_error = _resolve_source(
            record.get("source"),
            label=f"asset_exports[{index}].source",
            allowed_roots=(ROOT, concept_dir, evidence_dir),
        )
        export, export_error = _resolve_inside(
            venue_dir,
            record.get("export"),
            label=f"asset_exports[{index}].export",
        )
        for error in (source_error, export_error):
            if error:
                errors.append(error)
        if source is None or export is None:
            continue
        if export.suffix.lower() not in {".pdf", ".png"}:
            errors.append(f"portable asset export must be PDF or PNG: {_display_path(export)}")
        if source in exported_sources:
            errors.append(f"asset source is exported more than once: {_display_path(source)}")
        exported_sources[source] = export
        if not source.is_file():
            errors.append(f"asset source is missing: {_display_path(source)}")
        if not export.is_file():
            errors.append(f"portable asset export is missing: {_display_path(export)}")
        elif export.name not in active_main_tex:
            errors.append(f"venue TeX does not reference portable asset: {export.name}")
    for source in sorted(required_sources):
        if source.resolve() not in {path.resolve() for path in exported_sources}:
            errors.append(
                "package manifest does not export required figure: "
                f"{_display_path(source)}"
            )

    main_text_page = _main_text_page_from_aux(auxiliary_path, errors)
    if (
        main_text_page is not None
        and manifest.get("main_text_pages") != main_text_page
    ):
        errors.append(
            "package manifest main_text_pages does not match the resolved "
            "world-tubes-main-end label"
        )
    pdf_errors, pdf_audit = audit_pdf(
        pdf_path,
        expected_total_pages=manifest.get("total_pages"),
        main_text_pages=main_text_page,
        pdfinfo_command=pdfinfo_command,
        pdffonts_command=pdffonts_command,
    )
    errors.extend(pdf_errors)
    errors.extend(_verify_visual_qa(manifest.get("pdf_visual_qa"), pdf_audit=pdf_audit))

    recorder_inputs, recorder_errors = _recorder_inputs(recorder_path, venue_dir)
    errors.extend(recorder_errors)
    required_recorded_inputs = [
        main_tex_path,
        style_path,
        bibliography_path,
        *(evidence_dir / name for name in REQUIRED_TABLE_FRAGMENTS),
        *exported_sources.values(),
    ]
    for path in required_recorded_inputs:
        if path.is_file() and not _contains_recorded_path(recorder_inputs, path):
            errors.append(f"LaTeX recorder does not bind input: {_display_path(path)}")

    if pdf_path.is_file():
        build_inputs = [path for path in required_recorded_inputs if path.is_file()]
        if build_inputs and pdf_path.stat().st_mtime_ns < max(
            path.stat().st_mtime_ns for path in build_inputs
        ):
            errors.append("compiled PDF is older than one or more package inputs")

    if require_clean_source:
        cleanliness_errors, source_audit = verify_source_cleanliness(
            repository,
            star_repository,
        )
        errors.extend(cleanliness_errors)
    else:
        source_audit = {"skipped": True}

    audit_payload: dict[str, Any] = {
        "schema_version": 1,
        "verifier": "world_tubes_iclr_package",
        "venue": VENUE_NAME,
        "venue_year": VENUE_YEAR,
        "accepted": not errors,
        "errors": sorted(set(errors)),
        "inputs": {
            "concise_draft": {
                "path": _display_path(draft_path),
                "sha256": _sha256(draft_path) if draft_path.is_file() else None,
            },
            "bibliography": {
                "path": _display_path(bibliography_path),
                "sha256": (
                    _sha256(bibliography_path) if bibliography_path.is_file() else None
                ),
            },
            "evidence": evidence_audit,
            "concept_figures": concept_audit,
            "venue_manifest": {
                "path": _display_path(manifest_path),
                "sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
            },
            "ai_use_statement": ai_audit,
            "source_bindings": source_binding_audit,
            "source_cleanliness": source_audit,
        },
        "pdf_audit": pdf_audit,
    }
    audit_payload["audit_payload_sha256"] = _canonical_json_sha256(audit_payload)
    return audit_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the concise World Tubes ICLR 2027 submission package."
    )
    parser.add_argument("--venue-dir", type=Path, default=DEFAULT_VENUE_DIR)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--bibliography", type=Path, default=DEFAULT_BIBLIOGRAPHY)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--concept-dir", type=Path, default=DEFAULT_CONCEPT_DIR)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--pdfinfo-command", default="pdfinfo")
    parser.add_argument("--pdffonts-command", default="pdffonts")
    args = parser.parse_args()

    audit = verify_iclr_package(
        venue_dir=args.venue_dir,
        draft_path=args.draft,
        bibliography_path=args.bibliography,
        evidence_dir=args.evidence_dir,
        concept_dir=args.concept_dir,
        pdfinfo_command=args.pdfinfo_command,
        pdffonts_command=args.pdffonts_command,
    )
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not audit["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
