#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed verifier for the concise WorldFoam Paper-B ICLR package.

The WorldFoam foundation-bundle verifier answers a deliberately narrower
question: are the currently retained synthetic/foundation artifacts intact?
It intentionally accepts an *incomplete* bundle whose G4 public-quality and
G6 native-memory panels say ``NOT MEASURED``.  This module is the higher-level
submission gate.  It never promotes that foundation acceptance into paper
readiness.

Acceptance requires all of the following at once:

* the concise anonymous manuscript and its bibliography;
* an independently verified, complete evidence bundle with accepted G4 and G6
  records and generated table/figure artifacts;
* deterministic WorldFoam concept figures;
* an official ICLR 2027 venue package whose TeX is hash-bound to those inputs;
* portable, hash-bound concept/evidence exports actually consumed by LaTeX;
* an author-reviewed AI-use statement;
* a US-Letter, font-embedded, page-budgeted PDF and page-complete visual audit;
* clean source provenance.

There is no command-line option that relaxes a publication gate.  The Python
API exposes dependency injection only for focused fixtures; the CLI always
uses the repository's real evidence verifier and source-cleanliness check.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_experiments.paper_runner_suite.verify_world_tubes_iclr_package import (  # noqa: E402
    audit_pdf,
    verify_ai_use_statement,
)


PAPER_DIR = ROOT / "research_notes" / "worldfoam_paper"
DEFAULT_DRAFT = PAPER_DIR / "WORLD_FOAM_ICLR_MAIN_DRAFT.md"
DEFAULT_BIBLIOGRAPHY = PAPER_DIR / "WORLD_FOAM_REFERENCES.bib"
DEFAULT_EVIDENCE_DIR = PAPER_DIR / "generated" / "foundation_v1"
DEFAULT_CONCEPT_DIR = PAPER_DIR / "figures"
DEFAULT_VENUE_DIR = PAPER_DIR / "venue" / "iclr2027"

PACKAGE_SCHEMA_VERSION = 1
PAPER_ID = "worldfoam-paper-b"
BUILD_KIND = "official_iclr2027_submission"
VENUE_NAME = "ICLR"
VENUE_YEAR = 2027
MAX_MAIN_TEXT_PAGES = 9
READY_PACKAGE_STATUS = "submission_candidate"
READY_TEMPLATE_STATUS = "acquired_official"
OFFICIAL_TEMPLATE_URL = (
    "https://github.com/ICLR/Master-Template/raw/master/iclr2027.zip"
)

REQUIRED_TABLE_FRAGMENTS = (
    "synthetic_visibility_table.tex",
    "g6_native_memory_table.tex",
    "g4_public_quality_table.tex",
)
REQUIRED_EVIDENCE_FIGURES = (
    "worldfoam_synthetic_depth_convergence.svg",
    "worldfoam_synthetic_crossing_flicker.svg",
    "worldfoam_synthetic_adaptive_fallback.svg",
    "material_family_loss.svg",
    "g6_native_memory_scaling.svg",
    "g4_public_quality.svg",
)
REQUIRED_CONCEPT_FIGURES = (
    "worldfoam_representation_split.svg",
    "worldfoam_ray_fiber_atlas.svg",
)
REQUIRED_ACCEPTED_GATES = ("G0", "G1", "G2", "G3", "G4", "G6")
REQUIRED_PROMOTED_CLAIMS = (
    "synthetic_cpu_g0_g3",
    "public_quality",
    "native_memory_fit",
)
REQUIRED_VISUAL_QA_CHECKS = (
    "no_clipped_content",
    "readable_labels",
    "resolved_cross_references",
    "no_placeholders",
    "anonymous_submission",
    "captions_and_error_bars_checked",
)
FORBIDDEN_ARTIFACT_PATTERNS = (
    re.compile(r"\bNOT\s+MEASURED\b", re.IGNORECASE),
    re.compile(r"\bNOT\s+SUBMISSION[- ]READY\b", re.IGNORECASE),
    re.compile(r"\b(?:TODO|TBD)\b", re.IGNORECASE),
    re.compile(r"\bPLACEHOLDER\b", re.IGNORECASE),
)
UNRESOLVED_DRAFT_GATE_PATTERNS = (
    re.compile(r"<!--\s*ARTIFACT-GATE:", re.IGNORECASE),
    re.compile(r"<!--\s*AI-USE-GATE", re.IGNORECASE),
)
RESULT_TABLE_HEADER_TOKENS = frozenset(
    {
        "psnr",
        "ssim",
        "lpips",
        "l1",
        "mse",
        "rss",
        "mps",
        "memory",
        "runtime",
        "wall time",
        "loss",
        "metric",
        "result",
        "peak",
        "bytes",
    }
)


EvidenceVerifier = Callable[[Path], list[str]]
GateArtifactVerifier = Callable[[Path], Mapping[str, Any]]


def _display_path(path: Path, *, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
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


def _read_text(path: Path, label: str, errors: list[str]) -> str:
    if not path.is_file():
        errors.append(f"{label} is missing: {_display_path(path)}")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"{label} is not UTF-8: {_display_path(path)}")
        return ""


def _resolve_inside(
    base: Path,
    value: Any,
    *,
    label: str,
) -> tuple[Path | None, str | None]:
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
    repository: Path,
    value: Any,
    *,
    label: str,
) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, f"{label} must be a non-empty path"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repository / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(repository.resolve())
    except ValueError:
        return None, f"{label} is outside the source repository: {value}"
    return candidate, None


def _strip_tex_comments(text: str) -> str:
    return "\n".join(
        re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines()
    )


def _verify_citations(draft: str, bibliography: str) -> list[str]:
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
    errors = [
        f"bibliography is missing cited key: {key}"
        for key in sorted(citation_keys - bibliography_keys)
    ]
    if not citation_keys:
        errors.append("concise manuscript contains no citation keys")
    return errors


def _markdown_result_table_headers(markdown: str) -> list[str]:
    """Return result-like Markdown table headers that bypass generated TeX."""

    lines = markdown.splitlines()
    headers: list[str] = []
    in_fence = False
    for index, line in enumerate(lines[:-1]):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        separator = lines[index + 1].strip()
        if not re.fullmatch(r"\|[\s:|-]+\|", separator):
            continue
        cells = [cell.strip().lower() for cell in stripped.strip("|").split("|")]
        if any(
            token == cell or token in cell
            for cell in cells
            for token in RESULT_TABLE_HEADER_TOKENS
        ):
            headers.append(stripped)
    return headers


def _artifact_manifest_entries(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("path")): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }


def _status_is_accepted(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith("accepted")


def _default_evidence_verifier(path: Path) -> list[str]:
    try:
        from research_experiments.world_foam_lane2.generate_worldfoam_paper_b_artifacts import (  # noqa: E501
            verify_bundle_dir,
        )
    except Exception as error:  # pragma: no cover - user-facing import gate
        return [f"could not import Paper-B evidence verifier: {error}"]
    return verify_bundle_dir(path)


def _default_g6_artifact_verifier(path: Path) -> Mapping[str, Any]:
    from research_experiments.world_foam_lane2.verify_worldfoam_training_memory_ablation import (  # noqa: E501
        verify_artifact_file,
    )

    return verify_artifact_file(path)


def _default_g4_artifact_verifier(path: Path) -> Mapping[str, Any]:
    from research_experiments.world_foam_lane2.verify_worldfoam_public_quality_ablation_v2 import (  # noqa: E501
        verify_artifact_file,
    )

    return verify_artifact_file(path)


def _verify_promoted_gate_record(
    gate: str,
    records: Sequence[Any],
    *,
    repository: Path,
    g4_artifact_verifier: GateArtifactVerifier,
    g6_artifact_verifier: GateArtifactVerifier,
) -> tuple[list[str], int]:
    errors: list[str] = []
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and gate in str(record.get("gate", "")).split("/")
        and record.get("status") == "accepted"
    ]
    if not matches:
        return [f"evidence ledger has no independently accepted {gate} record"], 0
    verified_measured_rows = 0
    for index, record in enumerate(matches):
        label = f"{gate} accepted record {index}"
        record_errors: list[str] = []
        rows = record.get("numeric_rows_emitted")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            record_errors.append(f"{label} emits no measured numeric rows")
        if not isinstance(record.get("verifier"), str) or not record["verifier"].strip():
            record_errors.append(f"{label} has no independent verifier identity")
        if record.get("errors") != []:
            record_errors.append(f"{label} retains verifier errors")
        evidence_description = " ".join(
            str(record.get(key, ""))
            for key in ("evidence_id", "path", "scope", "verifier")
        )
        if re.search(
            r"(?:^|[\W_])(?:dry[-_ ]?run|smoke|test|proxy|fake[-_ ]?native|"
            r"source[-_ ]?only)(?:$|[\W_])",
            evidence_description,
            flags=re.IGNORECASE,
        ):
            record_errors.append(
                f"{label} is labelled as dry-run/test/proxy evidence"
            )
        source, source_error = _resolve_source(
            repository,
            record.get("path"),
            label=f"{label}.path",
        )
        if source_error:
            record_errors.append(source_error)
            errors.extend(record_errors)
            continue
        assert source is not None
        if not source.is_file():
            record_errors.append(
                f"{label} source artifact is missing: {_display_path(source)}"
            )
        elif record.get("sha256") != _sha256(source):
            record_errors.append(f"{label} source artifact hash changed")
        else:
            try:
                payload = _load_json(source)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                record_errors.append(f"{label} source artifact is not valid JSON: {error}")
                payload = {}
            if payload:
                if payload.get("proxy_or_test_artifact") is not False:
                    record_errors.append(
                        f"{label} does not explicitly reject proxy/test origin"
                    )
                if payload.get("measurement_is_simulated") is not False:
                    record_errors.append(
                        f"{label} does not explicitly reject simulated measurement"
                    )
                if payload.get("status") not in {"measured", "accepted"}:
                    record_errors.append(f"{label} is not a measured artifact")
                if gate == "G4" and payload.get("public_quality_evidence") is not True:
                    record_errors.append(
                        f"{label} does not declare public_quality_evidence=true"
                    )
                if gate == "G4":
                    try:
                        report = dict(g4_artifact_verifier(source))
                    except Exception as error:  # pragma: no cover - fail-closed boundary
                        record_errors.append(
                            f"{label} public-quality verifier raised "
                            f"{type(error).__name__}: {error}"
                        )
                        report = {}
                    if report.get("accepted") is not True:
                        record_errors.append(
                            f"{label} independent G4 verifier did not accept the artifact"
                        )
                    observed_g4_rows = report.get(
                        "row_count", report.get("observed_row_count")
                    )
                    if observed_g4_rows != 36:
                        record_errors.append(
                            f"{label} must contain the complete 36-row public matrix"
                        )
                    if rows != 36:
                        record_errors.append(
                            f"{label} ledger numeric_rows_emitted must equal 36"
                        )
                if gate == "G6":
                    try:
                        report = dict(g6_artifact_verifier(source))
                    except Exception as error:  # pragma: no cover - fail-closed boundary
                        record_errors.append(
                            f"{label} native verifier raised {type(error).__name__}: {error}"
                        )
                        report = {}
                    if report.get("accepted") is not True:
                        record_errors.append(
                            f"{label} native fresh-process verifier did not accept the artifact"
                        )
                    observed_primary = report.get("observed_row_count")
                    observed_control = report.get("observed_control_row_count")
                    if observed_primary != 12 or observed_control != 9:
                        record_errors.append(
                            f"{label} must contain 12 primary plus 9 control measured rows"
                        )
                    if rows != 21:
                        record_errors.append(
                            f"{label} ledger numeric_rows_emitted must equal 21"
                        )
        errors.extend(record_errors)
        if not record_errors and isinstance(rows, int):
            verified_measured_rows += rows
    return errors, verified_measured_rows


def verify_evidence_bundle(
    evidence_dir: Path,
    *,
    repository: Path = ROOT,
    bundle_verifier: EvidenceVerifier | None = None,
    g4_artifact_verifier: GateArtifactVerifier | None = None,
    g6_artifact_verifier: GateArtifactVerifier | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Verify foundation integrity, then enforce the stricter paper boundary."""

    errors: list[str] = []
    verifier = _default_evidence_verifier if bundle_verifier is None else bundle_verifier
    try:
        errors.extend(
            f"evidence bundle verifier: {error}"
            for error in verifier(evidence_dir)
        )
    except Exception as error:  # pragma: no cover - fail-closed API boundary
        errors.append(
            "evidence bundle verifier raised "
            f"{type(error).__name__}: {error}"
        )

    manifest_path = evidence_dir / "manifest.json"
    gate_status_path = evidence_dir / "gate_status.json"
    ledger_path = evidence_dir / "evidence_ledger.json"
    payloads: dict[str, dict[str, Any]] = {}
    for label, path in (
        ("manifest", manifest_path),
        ("gate status", gate_status_path),
        ("evidence ledger", ledger_path),
    ):
        try:
            payloads[label] = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"could not load Paper-B {label}: {error}")
            payloads[label] = {}

    manifest = payloads["manifest"]
    gate_status = payloads["gate status"]
    ledger = payloads["evidence ledger"]
    if manifest.get("complete") is not True:
        errors.append("Paper-B evidence manifest is incomplete")
    claims = manifest.get("claims")
    if not isinstance(claims, Mapping):
        errors.append("Paper-B evidence manifest claims are missing")
        claims = {}
    for claim in REQUIRED_PROMOTED_CLAIMS:
        if claims.get(claim) is not True:
            errors.append(f"Paper-B evidence claim is not promoted: {claim}")

    gate_rows = gate_status.get("gates")
    if not isinstance(gate_rows, list):
        errors.append("Paper-B gate-status rows are missing")
        gate_rows = []
    by_gate = {
        str(row.get("gate")): row
        for row in gate_rows
        if isinstance(row, Mapping)
    }
    for gate in REQUIRED_ACCEPTED_GATES:
        if not _status_is_accepted(by_gate.get(gate, {}).get("status")):
            errors.append(f"Paper-B gate is not accepted: {gate}")
    if gate_status.get("paper_ready") is not True:
        errors.append("Paper-B gate status does not declare paper_ready=true")
    if gate_status.get("iclr_ready") is not True:
        errors.append("Paper-B gate status does not declare iclr_ready=true")

    records = ledger.get("records")
    if not isinstance(records, list):
        errors.append("Paper-B evidence ledger records are missing")
        records = []
    measured_rows_by_gate: dict[str, int] = {}
    resolved_g4_verifier = (
        _default_g4_artifact_verifier
        if g4_artifact_verifier is None
        else g4_artifact_verifier
    )
    resolved_g6_verifier = (
        _default_g6_artifact_verifier
        if g6_artifact_verifier is None
        else g6_artifact_verifier
    )
    for gate in ("G4", "G6"):
        gate_errors, measured_rows = _verify_promoted_gate_record(
            gate,
            records,
            repository=repository,
            g4_artifact_verifier=resolved_g4_verifier,
            g6_artifact_verifier=resolved_g6_verifier,
        )
        errors.extend(gate_errors)
        measured_rows_by_gate[gate] = measured_rows

    entries = _artifact_manifest_entries(manifest)
    if not entries:
        errors.append("Paper-B evidence manifest files are missing")
    for name in (*REQUIRED_TABLE_FRAGMENTS, *REQUIRED_EVIDENCE_FIGURES):
        if name not in entries:
            errors.append(f"evidence manifest does not bind required artifact: {name}")
        path = evidence_dir / name
        if not path.is_file():
            errors.append(f"required evidence artifact is missing: {_display_path(path)}")
            continue
        row = entries.get(name, {})
        if row.get("sha256") != _sha256(path):
            errors.append(f"evidence artifact hash disagrees with manifest: {name}")
        if row.get("bytes") != path.stat().st_size:
            errors.append(f"evidence artifact byte size disagrees with manifest: {name}")
        if path.suffix.lower() in {".tex", ".svg", ".md", ".csv", ".json"}:
            text = _read_text(path, f"evidence artifact {name}", errors)
            for pattern in FORBIDDEN_ARTIFACT_PATTERNS:
                if pattern.search(text):
                    errors.append(
                        f"evidence artifact contains incomplete placeholder content: {name}"
                    )
                    break
    for name in entries:
        if "placeholder" in name.lower():
            errors.append(f"evidence bundle still contains placeholder artifact: {name}")

    audit = {
        "directory": _display_path(evidence_dir),
        "manifest": _display_path(manifest_path),
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "gate_status_sha256": (
            _sha256(gate_status_path) if gate_status_path.is_file() else None
        ),
        "ledger_sha256": _sha256(ledger_path) if ledger_path.is_file() else None,
        "foundation_integrity_verified": not any(
            error.startswith("evidence bundle verifier:")
            or error.startswith("evidence bundle verifier raised")
            for error in errors
        ),
        "complete": manifest.get("complete"),
        "claims": dict(claims),
        "gates": {
            gate: by_gate.get(gate, {}).get("status")
            for gate in REQUIRED_ACCEPTED_GATES
        },
        "verified_measured_rows": measured_rows_by_gate,
    }
    return sorted(set(errors)), audit


def verify_concept_figures(concept_dir: Path) -> tuple[list[str], dict[str, Any]]:
    try:
        from research_experiments.paper_runner_suite.generate_worldfoam_concept_figures import (  # noqa: E501
            verify_figure_dir,
        )
    except Exception as error:  # pragma: no cover - user-facing import gate
        return [f"could not import WorldFoam concept-figure verifier: {error}"], {}
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
    return f"% WORLD_FOAM_{label}_SHA256: {digest}"


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


def _verify_visual_qa(record: Any, *, pdf_audit: Mapping[str, Any]) -> list[str]:
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
    expected_pages = (
        list(range(1, page_count + 1)) if isinstance(page_count, int) else []
    )
    if record.get("inspected_pages") != expected_pages:
        errors.append("PDF visual QA must list every rendered page exactly once")
    checks = record.get("checks")
    if not isinstance(checks, Mapping):
        errors.append("PDF visual QA checks must be an object")
    else:
        for key in REQUIRED_VISUAL_QA_CHECKS:
            if checks.get(key) is not True:
                errors.append(f"PDF visual QA check is not accepted: {key}")
    try:
        date.fromisoformat(str(record.get("reviewed_at")))
    except ValueError:
        errors.append("PDF visual QA reviewed_at must be an ISO date")
    return errors


def _main_text_page_from_aux(path: Path, errors: list[str]) -> int | None:
    text = _read_text(path, "LaTeX auxiliary file", errors)
    if not text:
        return None
    match = re.search(
        r"\\newlabel\{worldfoam-main-end\}\{\{[^}]*\}\{([0-9]+)\}",
        text,
    )
    if not match:
        errors.append("LaTeX auxiliary file does not resolve label worldfoam-main-end")
        return None
    return int(match.group(1))


def _load_package_manifest(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, [f"venue package manifest is missing: {_display_path(path)}"]
    try:
        return _load_json(path), []
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {}, [f"could not load venue package manifest: {error}"]


def _verify_manifest_header(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "paper_id": PAPER_ID,
        "build_kind": BUILD_KIND,
        "venue": VENUE_NAME,
        "venue_year": VENUE_YEAR,
        "anonymous_submission": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            errors.append(f"package manifest {key} must be {value!r}")
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
    repository: Path,
    draft_path: Path,
    bibliography_path: Path,
    evidence_manifest_path: Path,
    ai_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Verify exact machine-readable bindings to the authored package inputs."""

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
    for key, expected_path in expected.items():
        record = records.get(key)
        if not isinstance(record, Mapping):
            errors.append(f"source_bindings.{key} must be an object")
            continue
        bound_path, path_error = _resolve_source(
            repository,
            record.get("path"),
            label=f"source_bindings.{key}.path",
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
                "path": _display_path(expected_path, root=repository),
                "sha256": actual_digest,
                "bytes": actual_bytes,
            }
        else:
            errors.append(
                "source binding input is missing: "
                f"{_display_path(expected_path, root=repository)}"
            )
            audit[key] = {"path": str(expected_path), "sha256": None, "bytes": None}

    unknown = sorted(set(records) - set(expected))
    if unknown:
        errors.append(
            "package manifest source_bindings has unknown entries: "
            + ", ".join(unknown)
        )
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
) -> tuple[list[str], dict[str, Any]]:
    status, error = _git_status(repository)
    audit = {
        "path": _display_path(repository),
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
        return [f"could not inspect repository cleanliness: {error}"], audit
    if status:
        return ["repository is dirty"], audit
    return [], audit


def _verify_portable_export(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"portable asset export is missing: {_display_path(path)}"]
    if path.stat().st_size < 1024:
        errors.append(f"portable asset export is implausibly small: {_display_path(path)}")
    prefix = path.read_bytes()[:8]
    if path.suffix.lower() == ".png" and prefix != b"\x89PNG\r\n\x1a\n":
        errors.append(f"portable PNG has an invalid signature: {_display_path(path)}")
    elif path.suffix.lower() == ".pdf" and not prefix.startswith(b"%PDF-"):
        errors.append(f"portable PDF has an invalid signature: {_display_path(path)}")
    return errors


def verify_iclr_package(
    *,
    venue_dir: Path = DEFAULT_VENUE_DIR,
    draft_path: Path = DEFAULT_DRAFT,
    bibliography_path: Path = DEFAULT_BIBLIOGRAPHY,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    concept_dir: Path = DEFAULT_CONCEPT_DIR,
    repository: Path = ROOT,
    require_clean_source: bool = True,
    evidence_verifier: EvidenceVerifier | None = None,
    g4_artifact_verifier: GateArtifactVerifier | None = None,
    g6_artifact_verifier: GateArtifactVerifier | None = None,
    pdfinfo_command: str = "pdfinfo",
    pdffonts_command: str = "pdffonts",
) -> dict[str, Any]:
    """Return a deterministic package audit; acceptance is strictly fail-closed."""

    venue_dir = venue_dir.resolve()
    repository = repository.resolve()
    errors: list[str] = []

    draft = _read_text(draft_path, "concise manuscript", errors)
    bibliography = _read_text(bibliography_path, "bibliography", errors)
    if draft and draft_path.name != "WORLD_FOAM_ICLR_MAIN_DRAFT.md":
        errors.append("venue verifier is not bound to the concise WorldFoam ICLR manuscript")
    if draft and not re.search(r"^author:\s*Anonymous\s*$", draft, re.MULTILINE):
        errors.append("concise manuscript front matter is not anonymous")
    errors.extend(_verify_citations(draft, bibliography))
    for pattern in UNRESOLVED_DRAFT_GATE_PATTERNS:
        if pattern.search(draft):
            errors.append(
                "concise manuscript still contains unresolved artifact/AI gate markers"
            )
            break
    for table in REQUIRED_TABLE_FRAGMENTS:
        marker = f"<!-- GENERATED-TABLE:{table}"
        if marker not in draft:
            errors.append(f"concise manuscript is missing generated-table marker: {table}")
    for header in _markdown_result_table_headers(draft):
        errors.append(
            "concise manuscript contains a hand-copied result table instead of a "
            f"generated fragment: {header}"
        )

    evidence_errors, evidence_audit = verify_evidence_bundle(
        evidence_dir,
        repository=repository,
        bundle_verifier=evidence_verifier,
        g4_artifact_verifier=g4_artifact_verifier,
        g6_artifact_verifier=g6_artifact_verifier,
    )
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

    if pdf_path != (venue_dir / "main.pdf").resolve():
        errors.append(
            "official WorldFoam package compiled_pdf must be venue/iclr2027/main.pdf; "
            "generic QA PDFs are not submission artifacts"
        )
    if re.search(r"(?:generic|qa|draft)", pdf_path.name, re.IGNORECASE):
        errors.append("generic QA/draft PDF cannot satisfy the official package gate")

    if not archive_path.is_file():
        errors.append(
            "official ICLR template archive is missing: "
            f"{_display_path(archive_path, root=repository)}"
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
        repository=repository,
        draft_path=draft_path,
        bibliography_path=bibliography_path,
        evidence_manifest_path=evidence_dir / "manifest.json",
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
    if main_tex and "\\PackageError{worldfoam-venue}" in active_main_tex:
        errors.append("venue TeX still contains the fail-closed scaffold stop")
    if main_tex and "\\label{worldfoam-main-end}" not in active_main_tex:
        errors.append("venue TeX must label the end of main text as worldfoam-main-end")
    for pattern in FORBIDDEN_ARTIFACT_PATTERNS:
        if pattern.search(active_main_tex):
            errors.append("venue TeX contains visible placeholder/incomplete content")
            break

    source_bindings = {
        "SOURCE": _sha256(draft_path) if draft_path.is_file() else "missing",
        "BIBLIOGRAPHY": (
            _sha256(bibliography_path) if bibliography_path.is_file() else "missing"
        ),
        "EVIDENCE_MANIFEST": (
            _sha256(evidence_dir / "manifest.json")
            if (evidence_dir / "manifest.json").is_file()
            else "missing"
        ),
        "AI_USE_STATEMENT": _sha256(ai_path) if ai_path.is_file() else "missing",
    }
    for label, digest in source_bindings.items():
        expected = _binding_line(label, digest)
        if expected not in main_tex:
            errors.append(f"venue TeX is missing source binding: {expected}")

    for filename in REQUIRED_TABLE_FRAGMENTS:
        if not re.search(rf"\\input\{{[^}}]*{re.escape(filename)}\}}", active_main_tex):
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
            repository,
            record.get("source"),
            label=f"asset_exports[{index}].source",
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
        if "placeholder" in source.name.lower() or "placeholder" in export.name.lower():
            errors.append("placeholder asset cannot enter the official venue package")
        if source in exported_sources:
            errors.append(f"asset source is exported more than once: {_display_path(source)}")
        exported_sources[source] = export
        if not source.is_file():
            errors.append(f"asset source is missing: {_display_path(source)}")
        else:
            if record.get("source_sha256") != _sha256(source):
                errors.append(f"asset source hash mismatch: {_display_path(source)}")
        errors.extend(_verify_portable_export(export))
        if export.is_file():
            if record.get("export_sha256") != _sha256(export):
                errors.append(f"portable asset export hash mismatch: {_display_path(export)}")
            if record.get("export_bytes") != export.stat().st_size:
                errors.append(f"portable asset export byte size mismatch: {_display_path(export)}")
            if export.name not in active_main_tex:
                errors.append(f"venue TeX does not reference portable asset: {export.name}")
    exported_resolved = {path.resolve() for path in exported_sources}
    for source in sorted(required_sources):
        if source.resolve() not in exported_resolved:
            errors.append(
                "package manifest does not export required figure: "
                f"{_display_path(source)}"
            )

    main_text_page = _main_text_page_from_aux(auxiliary_path, errors)
    if main_text_page is not None and manifest.get("main_text_pages") != main_text_page:
        errors.append(
            "package manifest main_text_pages does not match the resolved "
            "worldfoam-main-end label"
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
        cleanliness_errors, source_audit = verify_source_cleanliness(repository)
        errors.extend(cleanliness_errors)
    else:
        source_audit = {"skipped": True}

    audit_payload: dict[str, Any] = {
        "schema_version": 1,
        "verifier": "worldfoam_iclr_package",
        "paper_id": PAPER_ID,
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
            "source_bindings": source_binding_audit,
            "ai_use_statement": ai_audit,
            "source_cleanliness": source_audit,
        },
        "pdf_audit": pdf_audit,
    }
    audit_payload["audit_payload_sha256"] = _canonical_json_sha256(audit_payload)
    return audit_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the concise WorldFoam Paper-B ICLR 2027 package."
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
