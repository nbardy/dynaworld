from __future__ import annotations

"""Import the accepted clean World Tubes variable-camera schema-v2 artifact.

This is intentionally a compatibility importer, not a second experiment
runner.  The accepted paper-freeze report predates the current variable-camera
runner's schema-v1 contract.  Passing those bytes through the current runner
would either reject valid old fields or, worse, tempt a caller to relabel the
newer dirty 178/179-degree diagnostic.  This module instead pins the exact
accepted bytes, clean paired source commits, implementation manifest, handoff
receipts, and deterministic table/figure bytes.

The importer launches no renderer, training workload, Torch import, or Git
command.  It changes only the variable-camera component of an existing paper
artifact bundle, preserves every other component, and restores the stronger
moving-camera-density requirement when an older current-worktree bundle lacks
it.  Overall submission readiness remains fail-closed.
"""

import argparse
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
IMPORTER_PATH = Path(__file__).resolve()
DEFAULT_PAPER_FREEZE_ROOT = ROOT.with_name(f"{ROOT.name}-paper-freeze")
DEFAULT_HANDOFF_DIR = DEFAULT_PAPER_FREEZE_ROOT / "output" / "paper_handoff"
DEFAULT_FROZEN_RAW = (
    DEFAULT_PAPER_FREEZE_ROOT
    / "outputs"
    / "benchmarks"
    / "2026-07-28_world_tubes_variable_camera_closure_death_curve"
    / "summary.json"
)
DEFAULT_FROZEN_ASSET_DIR = (
    DEFAULT_PAPER_FREEZE_ROOT
    / "research_notes"
    / "gauged_uvt_trace_atlas"
    / "paper"
    / "generated"
    / "schema_v2"
)
DEFAULT_LOCAL_EVIDENCE_DIR = (
    ROOT
    / "artifacts"
    / "paper_evidence"
    / "world_tubes_variable_camera_schema_v2_clean"
)
DEFAULT_BUNDLE_DIR = (
    ROOT
    / "research_notes"
    / "gauged_uvt_trace_atlas"
    / "paper"
    / "generated"
    / "schema_v2"
)
DEFAULT_PUBLICATION_SVG = (
    ROOT
    / "research_notes"
    / "gauged_uvt_trace_atlas"
    / "paper"
    / "figures"
    / "world_tubes_variable_camera_closure_death_publication.svg"
)

VARIABLE_COMPONENT = "variable_camera_closure_death"
MOVING_DENSITY_COMPONENT = "moving_camera_density"
LOCAL_SUMMARY_NAME = "summary.json"
LOCAL_RECEIPT_NAME = "compatibility_import_receipt.json"


@dataclass(frozen=True)
class FrozenContract:
    raw_sha256: str
    source_repository_commit: str
    source_star_uvt_commit: str
    handoff_superproject_commit: str
    handoff_star_uvt_commit: str
    handoff_sha256s_sha256: str
    handoff_receipts: Mapping[str, str]
    implementation_sources: tuple[tuple[str, str], ...]
    implementation_manifest_sha256: str
    rendered_assets: Mapping[str, str]


FROZEN_CONTRACT = FrozenContract(
    raw_sha256="118f26857a1c51262f6d8b0a33d55ee037dc19a07713ce318aaab9878d5df198",
    source_repository_commit="33a64aa44efd430f56eb284915aa47b3e5ec2b7d",
    source_star_uvt_commit="6c9945258fb1b31c43418857eb5ead98e588fd77",
    handoff_superproject_commit="d5b0db58c3038f25d14a5412b4cfe170c65eb3b8",
    handoff_star_uvt_commit="6c9945258fb1b31c43418857eb5ead98e588fd77",
    handoff_sha256s_sha256="38e2a02cd72f8b3381dadaa135c1f9460c76d84f091647cc9bb8e580577053b5",
    handoff_receipts={
        "README.md": "2498a69c7805baedc309e78582d79852fdad29c865f23d67c2bb551da36fb169",
        "MANIFEST.json": "b3790a2c9c528b628d73322b2592baff60550ee858db58ed2bb7ec4f3b9062d6",
        "source/dynaworld-paper-freeze.bundle": "eb580b0468c25c9640ec1d36cdd5dfee05d0bf30d2a11d1bc2720c52c810d48f",
        "source/fast-mac-world-tubes-star-freeze.bundle": "98ef22bd62d56a82bf210aa3d596a3754a29c6d42e70863896f5db3e97c69eb3",
        "artifacts/variable_camera/summary.json": "118f26857a1c51262f6d8b0a33d55ee037dc19a07713ce318aaab9878d5df198",
        "artifacts/variable_camera/summary.md": "f393833a9d14bbace0ec5d7b6e9cf2330e39d436a091ec987536b4ab4c7fc3ef",
        "assets/native/powerfoam._C.cpython-311-darwin.so": "9197d05e712621b893224b85fa8e76a63c6896daeddf9cc7e729dce8acf11f14",
        "assets/native/star_uvt._C.cpython-311-darwin.so": "452dbd0b09d8be890f0d79913d63bcfef9a41d37e4912be0003523d4ac7ad89d",
        "assets/native/v5._C.cpython-311-darwin.so": "018245d206dd430b1709e35aeef881769a330c37af9d15851d4c3e9ad9c6532b",
        "assets/lpips/alexnet-owt-7be5be79.pth": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
        "assets/lpips/lpips-v0.1-alex-linear.pth": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    },
    implementation_sources=(
        (
            "research_experiments/star_uvt_feature_tubes/"
            "projective_variable_camera_closure_death_curve.py",
            "418427de2e590adfad061fa91eb7a86fd1505654c103f3ed6f56f7a69064faf2",
        ),
        (
            "third_party/fast-mac-gsplat/variants/star_uvt_v0/"
            "torch_gsplat_bridge_star_uvt/__init__.py",
            "d94412648f7329c0419d5e8ca316ec9f1c6dfdb3bc70824f0090bffa9cdb1111",
        ),
        (
            "third_party/fast-mac-gsplat/variants/star_uvt_v0/"
            "torch_gsplat_bridge_star_uvt/projective_trace.py",
            "ecff7d5246a8a89cb7d2b8fe955cd244f6538660e9125ea599a1590e7644c8f7",
        ),
    ),
    implementation_manifest_sha256="5ce27c6672825dacd6e96ccf697423ad34df4dac26a4cc3daa4d2d51aab4719a",
    rendered_assets={
        "variable_camera_table.md": "75adb7e011b19436a9caff3a8af22fe9525e9365b1118fd8fbe1dd2a8eb0c72a",
        "variable_camera_table.tex": "154cedbbaefb73c94d0be3e0df1accc49599a996f9579348893a9ac96f56ce30",
        "variable_camera_closure_death.svg": "a1e1a29c77578f1a8beb1520a69e4653ba241e804ac256d4ee59d89dae14cb43",
    },
)


MOVING_CAMERA_DENSITY_INPUT = (
    "outputs/benchmarks/world_tubes_frozen_world_moving_camera_v1/"
    "coffee_martini_full_300f_progressive_512_v1/seed_17/summary.json"
)
MOVING_CAMERA_DENSITY_CONTRACT = {
    "benchmark": "world_tubes_frozen_world_moving_camera_v1",
    "protocol": "coffee_martini_full_300f_progressive_512_v1",
    "seed": 17,
    "frame_counts": [8, 16, 32, 64],
    "image_size": [256, 256],
    "camera_program": {
        "schema_version": 1,
        "mode": "bounded_yaw_projective_first_order_v1",
        "path_scope": "bounded_open_path",
        "yaw_start_degrees": -22.5,
        "yaw_end_degrees": 22.5,
        "yaw_total_degrees": 45.0,
        "sampling": "uniform_closed_interval",
        "frame_counts": [8, 16, 32, 64],
        "image_size": [256, 256],
        "compiler_chart_policy": "single_midpoint_first_order",
        "multi_chart_gauge_compiler": False,
    },
    "timing_warmups": 1,
    "timing_repeats": 5,
    "publication_thresholds": {
        "max_expensive_unresolved_fallback_fraction_exclusive": 0.02,
        "max_f32_continuous_to_sliced_reference_ratio": 0.4,
        "max_heavy_structural_ratio_t32_t8": 1.1,
        "max_image_p999_abs_error": 0.00784313725490196,
        "max_inference_break_even_frame_count": 8,
        "max_interaction_memory_ratio_t32_t8": 1.1,
        "max_loss_absolute_delta": 0.00001,
        "max_lpips_delta": 0.001,
        "max_training_break_even_frame_count": 16,
        "max_world_vjp_global_normalized_l2_error": 0.00001,
        "max_world_vjp_max_parameter_normalized_l2_error": 0.00001,
        "min_certified_stable_or_event_aligned_fraction": 0.98,
        "min_f32_forward_speedup": 2.0,
        "min_f32_reverse_speedup": 2.0,
        "min_f32_total_speedup": 1.7,
        "min_image_psnr_db": 50.0,
    },
    "complete_negative_is_retained": True,
}


class CompatibilityImportError(ValueError):
    """Raised when any frozen provenance or compatibility invariant drifts."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(value: Any, *, ensure_ascii: bool = False) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
        ).encode("utf-8")
    ).hexdigest()


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompatibilityImportError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise CompatibilityImportError(f"{label} must be a JSON object")
    return value


def _load_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    try:
        return _load_json_bytes(path.read_bytes(), label=label or str(path))
    except OSError as error:
        raise CompatibilityImportError(f"could not read {label or path}: {error}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompatibilityImportError(message)


def _parse_sha256s(payload: bytes) -> dict[str, str]:
    records: dict[str, str] = {}
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise CompatibilityImportError("handoff SHA256SUMS is not UTF-8") from error
    for index, line in enumerate(lines, start=1):
        fields = line.split("  ", 1)
        _require(len(fields) == 2, f"malformed SHA256SUMS line {index}")
        digest, raw_path = fields
        path = Path(raw_path)
        _require(
            len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
            f"invalid digest on SHA256SUMS line {index}",
        )
        _require(
            not path.is_absolute() and ".." not in path.parts and str(path) == raw_path,
            f"unsafe path on SHA256SUMS line {index}",
        )
        _require(raw_path not in records, f"duplicate SHA256SUMS path: {raw_path}")
        records[raw_path] = digest
    return records


def verify_handoff(
    handoff_dir: Path,
    *,
    contract: FrozenContract = FROZEN_CONTRACT,
) -> dict[str, Any]:
    """Verify every file named by the frozen handoff receipt."""

    handoff_dir = handoff_dir.resolve()
    sums_path = handoff_dir / "SHA256SUMS"
    _require(sums_path.is_file(), f"handoff SHA256SUMS is missing: {sums_path}")
    sums_payload = sums_path.read_bytes()
    _require(
        _bytes_sha256(sums_payload) == contract.handoff_sha256s_sha256,
        "handoff SHA256SUMS bytes do not match the frozen receipt",
    )
    records = _parse_sha256s(sums_payload)
    _require(
        records == dict(contract.handoff_receipts),
        "handoff SHA256SUMS file set or declared digests drifted",
    )
    for relative_path, expected_digest in records.items():
        path = handoff_dir / relative_path
        _require(path.is_file(), f"handoff receipt target is missing: {relative_path}")
        _require(
            _file_sha256(path) == expected_digest,
            f"handoff receipt target digest drifted: {relative_path}",
        )

    manifest = _load_json(handoff_dir / "MANIFEST.json", label="handoff manifest")
    superproject = manifest.get("superproject")
    star_uvt = manifest.get("star_uvt")
    _require(isinstance(superproject, Mapping), "handoff superproject record is missing")
    _require(isinstance(star_uvt, Mapping), "handoff STAR record is missing")
    _require(
        superproject.get("commit") == contract.handoff_superproject_commit,
        "handoff superproject commit drifted",
    )
    _require(
        star_uvt.get("commit") == contract.handoff_star_uvt_commit,
        "handoff STAR commit drifted",
    )
    _require(
        manifest.get("accepted_retained_jobs")
        == ["variable_camera_closure_death_curve"],
        "handoff accepted job list drifted",
    )
    return {
        "sha256s_sha256": contract.handoff_sha256s_sha256,
        "manifest_sha256": records["MANIFEST.json"],
        "all_receipts_verified": True,
        "verified_receipts": dict(sorted(records.items())),
        "superproject_commit": contract.handoff_superproject_commit,
        "star_uvt_commit": contract.handoff_star_uvt_commit,
    }


def verify_frozen_report_bytes(
    payload: bytes,
    *,
    contract: FrozenContract = FROZEN_CONTRACT,
) -> dict[str, Any]:
    """Decode only the pinned legacy schema-v2 contract.

    The current schema-v1 runner is deliberately not imported by this module.
    """

    digest = _bytes_sha256(payload)
    _require(digest == contract.raw_sha256, "frozen raw summary SHA-256 mismatch")
    report = _load_json_bytes(payload, label="frozen variable-camera summary")
    _require(report.get("schema_version") == 2, "frozen report must use schema_version 2")
    _require(
        report.get("benchmark") == "world_tubes_variable_camera_closure_death_curve",
        "frozen report benchmark mismatch",
    )
    _require(
        report.get("scope") == "bounded_synthetic_variable_camera_closure_death_curve",
        "frozen report scope mismatch",
    )

    expected_source = {
        "repository_commit": contract.source_repository_commit,
        "repository_dirty": False,
        "star_uvt_commit": contract.source_star_uvt_commit,
        "star_uvt_dirty": False,
    }
    _require(report.get("source") == expected_source, "frozen source start is not exact and clean")
    _require(report.get("source_finish") == expected_source, "frozen source finish drifted")
    _require(
        report.get("source_policy")
        == {"dirty_source_allowed": False, "paper_evidence_eligible": True},
        "frozen source policy is not paper eligible",
    )

    implementation = report.get("implementation")
    _require(isinstance(implementation, Mapping), "implementation manifest is missing")
    expected_sources = [
        {"path": path, "sha256": digest_value}
        for path, digest_value in contract.implementation_sources
    ]
    _require(
        implementation.get("source_files") == expected_sources,
        "implementation source file manifest drifted",
    )
    source_manifest_digest = _canonical_json_sha256(expected_sources, ensure_ascii=True)
    _require(
        source_manifest_digest == contract.implementation_manifest_sha256,
        "frozen contract implementation manifest constant is inconsistent",
    )
    _require(
        implementation.get("source_manifest_sha256") == source_manifest_digest,
        "implementation source manifest digest mismatch",
    )

    acceptance = report.get("acceptance")
    _require(isinstance(acceptance, Mapping), "frozen acceptance record is missing")
    _require(acceptance.get("accepted") is True, "frozen report is not accepted")
    _require(
        acceptance.get("label") == "accepted_bounded_closure_death_gate",
        "frozen acceptance label mismatch",
    )
    _require(
        acceptance.get("claim_scope")
        == "bounded open-path variable-camera projective atlas; not a 360/720-degree transition claim",
        "frozen claim scope drifted",
    )
    _require(acceptance.get("reasons") == [], "frozen acceptance has failure reasons")

    rows = report.get("rows")
    summary = report.get("summary")
    _require(isinstance(rows, list) and len(rows) == 12, "frozen report must contain 12 rows")
    _require(isinstance(summary, Mapping), "frozen summary record is missing")
    expected_spans = [5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 150.0, 170.0, 179.5]
    _require(
        report.get("observed_motion_half_spans_degrees") == expected_spans,
        "frozen observed motion sweep drifted",
    )
    _require(report.get("unexecuted_motion_half_spans_degrees") == [], "frozen sweep is incomplete")
    _require(report.get("sweep_termination") == "completed_requested_sweep", "frozen sweep termination drifted")
    _require(
        [row.get("motion_half_span_degrees") for row in rows] == expected_spans,
        "frozen row spans drifted",
    )
    _require(
        all(row.get("row_status") == "evaluated_certified" for row in rows[:-1]),
        "frozen closure row status drifted",
    )
    terminal = rows[-1]
    _require(
        terminal.get("row_status") == "compiler_unresolved_terminal"
        and terminal.get("regime") == "death"
        and terminal.get("accepted") is False,
        "frozen terminal compiler-death row drifted",
    )
    _require(
        terminal.get("unresolved_chart_count") == 2
        and terminal.get("unresolved_chart_reasons") == ["depth_residual"],
        "frozen terminal death witness drifted",
    )
    _require(
        summary.get("last_accepted_half_span_degrees") == 170.0
        and summary.get("first_death_half_span_degrees") == 179.5
        and summary.get("accepted_count") == 11
        and summary.get("death_count") == 1
        and summary.get("terminal_compiler_death") is True,
        "frozen closure/death summary drifted",
    )
    # Explicitly exclude the newer dirty candidate's distinguishing rows.
    _require(178.0 not in expected_spans and 179.0 not in expected_spans, "dirty candidate spans leaked into the frozen contract")
    return report


def _latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(old, new)
    return text


def _svg_axes(
    *,
    width: int,
    height: int,
    title: str,
    y_label: str,
    x_label: str,
) -> tuple[list[str], tuple[float, float, float, float]]:
    left, right, top, bottom = 90.0, 30.0, 58.0, 72.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f'<rect width="{width}" height="{height}" fill="white"/>',
        (
            f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" '
            f'font-size="18" font-weight="bold">{html.escape(title)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" '
            'stroke="#111827"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}" stroke="#111827"/>'
        ),
        (
            f'<text x="{width / 2:.1f}" y="{height - 16}" '
            f'text-anchor="middle" font-size="13">{html.escape(x_label)}</text>'
        ),
        (
            f'<text x="20" y="{height / 2:.1f}" '
            f'transform="rotate(-90 20 {height / 2:.1f})" '
            f'text-anchor="middle" font-size="13">{html.escape(y_label)}</text>'
        ),
    ]
    return elements, (left, top, plot_width, plot_height)


def render_variable_assets(report: Mapping[str, Any]) -> dict[str, bytes]:
    rows = list(report["rows"])
    summary = report["summary"]
    markdown = [
        "# Variable-camera closure/death curve",
        "",
        "| Half span (deg) | Regime | Compiler | Charts (accepted/unresolved) "
        "| Events (support/visibility) "
        "| Fallback samples | Invalid samples | Image max error | VJP rel. L2 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{tabular}{rllrrrrrr}",
        r"\toprule",
        (
            r"Half span & Regime & Compiler & Charts (A/U) & Events (S/V) & Fallback "
            r"& Invalid & Image err. & VJP rel. \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        terminal = row.get("row_status") == "compiler_unresolved_terminal"
        cells = (
            f"{float(row['motion_half_span_degrees']):.4g}",
            str(row["regime"]),
            "unresolved" if terminal else "certified",
            f"{int(row['accepted_chart_count'])}/{int(row['unresolved_chart_count'])}",
            "—" if terminal else f"{int(row['support_event_count'])}/{int(row['visibility_event_count'])}",
            "—" if terminal else f"{float(row['fallback_sample_fraction']):.4f}",
            "—" if terminal else f"{float(row['invalid_sample_fraction']):.4f}",
            "—" if terminal else f"{float(row['image_max_abs_error']):.3g}",
            "—" if terminal else f"{float(row['world_vjp_rel_l2_max']):.3g}",
        )
        markdown.append("| " + " | ".join(cells) + " |")
        latex.append(" & ".join(_latex_escape(cell) for cell in cells) + r" \\")
    markdown.extend(
        (
            "",
            "The terminal unresolved row is a compiler-certificate death event. "
            "It is not lowered or rendered, so image and VJP cells are intentionally blank.",
        )
    )
    latex.extend((r"\bottomrule", r"\end{tabular}"))

    rows = sorted(rows, key=lambda row: float(row["motion_half_span_degrees"]))
    evaluated_rows = [row for row in rows if row.get("row_status") == "evaluated_certified"]
    width, height = 900, 460
    elements, (left, top, plot_width, plot_height) = _svg_axes(
        width=width,
        height=height,
        title="Bounded variable-camera atlas closure and death",
        y_label="Fraction / normalized error",
        x_label="Camera motion half-span (degrees)",
    )
    min_x = float(rows[0]["motion_half_span_degrees"])
    max_x = float(rows[-1]["motion_half_span_degrees"])
    x_span = max(1.0e-12, max_x - min_x)
    max_error = max((float(row["image_max_abs_error"]) for row in evaluated_rows), default=0.0)
    for key, label, color, normalizer in (
        ("fallback_sample_fraction", "Fallback fraction", "#2563eb", 1.0),
        ("image_max_abs_error", "Image max error (normalized)", "#dc2626", max(max_error, 1.0e-12)),
    ):
        points = []
        for row in evaluated_rows:
            value = min(1.0, max(0.0, float(row[key]) / normalizer))
            x = left + (float(row["motion_half_span_degrees"]) - min_x) / x_span * plot_width
            y = top + plot_height - value * plot_height
            points.append(f"{x:.2f},{y:.2f}")
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        elements.append(
            f'<polyline class="data-line" points="{" ".join(points)}" '
            f'fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_x = left + 15 + (215 if key == "image_max_abs_error" else 0)
        elements.extend(
            (
                f'<line x1="{legend_x}" y1="{top - 18}" x2="{legend_x + 28}" '
                f'y2="{top - 18}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 35}" y="{top - 13}" font-size="11">{html.escape(label)}</text>',
            )
        )
    first_death = summary.get("first_death_half_span_degrees")
    _require(isinstance(first_death, (int, float)) and math.isfinite(float(first_death)), "first death is not finite")
    death_x = left + (float(first_death) - min_x) / x_span * plot_width
    elements.extend(
        (
            f'<line x1="{death_x:.2f}" y1="{top}" x2="{death_x:.2f}" '
            f'y2="{top + plot_height}" stroke="#7f1d1d" stroke-width="2" stroke-dasharray="6 5"/>',
            f'<text x="{death_x + 5:.2f}" y="{top + 16}" font-size="11" fill="#7f1d1d">first death</text>',
        )
    )
    elements.append("</svg>")
    return {
        "variable_camera_table.md": ("\n".join(markdown) + "\n").encode("utf-8"),
        "variable_camera_table.tex": ("\n".join(latex) + "\n").encode("utf-8"),
        "variable_camera_closure_death.svg": ("\n".join(elements) + "\n").encode("utf-8"),
    }


def verify_rendered_assets(
    assets: Mapping[str, bytes],
    *,
    contract: FrozenContract = FROZEN_CONTRACT,
) -> dict[str, str]:
    digests = {name: _bytes_sha256(payload) for name, payload in assets.items()}
    _require(digests == dict(contract.rendered_assets), "deterministic frozen table/SVG bytes drifted")
    return dict(sorted(digests.items()))


def render_publication_svg(exact_svg: bytes) -> bytes:
    """Derive a non-clipped display SVG without mutating frozen evidence.

    The paper-freeze SVG puts a left-anchored label five pixels from the right
    viewBox boundary.  The accepted data and all geometry stay unchanged; only
    that label is anchored inward in this separately named publication asset.
    """

    _require(
        _bytes_sha256(exact_svg)
        == FROZEN_CONTRACT.rendered_assets[
            "variable_camera_closure_death.svg"
        ],
        "publication rendering source is not the exact frozen evidence SVG",
    )
    text = exact_svg.decode("utf-8")
    source = (
        '<text x="875.00" y="74.0" font-size="11" '
        'fill="#7f1d1d">first death</text>'
    )
    replacement = (
        '<text x="862.00" y="74.0" text-anchor="end" font-size="11" '
        'fill="#7f1d1d">first death</text>'
    )
    _require(text.count(source) == 1, "frozen first-death label geometry drifted")
    return text.replace(source, replacement).encode("utf-8")


def _receipt_payload(
    *,
    handoff_audit: Mapping[str, Any],
    rendered_digests: Mapping[str, str],
    publication_svg_sha256: str,
    contract: FrozenContract,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "world_tubes_variable_camera_schema_v2_compatibility_import",
        "status": "accepted",
        "compatibility_boundary": {
            "source_schema_version": 2,
            "decoder": "pinned_schema_v2_compatibility_importer",
            "forbidden_decoder": "current_schema_v1_variable_camera_runner",
            "dirty_schema_v1_178_179_candidate_imported": False,
        },
        "source_artifact": {
            "path": f"artifacts/paper_evidence/world_tubes_variable_camera_schema_v2_clean/{LOCAL_SUMMARY_NAME}",
            "sha256": contract.raw_sha256,
        },
        "source_provenance": {
            "repository_commit": contract.source_repository_commit,
            "repository_dirty": False,
            "star_uvt_commit": contract.source_star_uvt_commit,
            "star_uvt_dirty": False,
            "implementation_manifest_sha256": contract.implementation_manifest_sha256,
        },
        "handoff": dict(handoff_audit),
        "rendered_assets": dict(sorted(rendered_digests.items())),
        "publication_rendering": {
            "path": (
                "research_notes/gauged_uvt_trace_atlas/paper/figures/"
                "world_tubes_variable_camera_closure_death_publication.svg"
            ),
            "sha256": publication_svg_sha256,
            "derived_from": "variable_camera_closure_death.svg",
            "derived_from_sha256": contract.rendered_assets[
                "variable_camera_closure_death.svg"
            ],
            "data_or_geometry_changed": False,
            "display_only_change": "anchor first-death label inward",
        },
        "importer": {
            "path": "research_experiments/paper_runner_suite/import_world_tubes_variable_camera_schema_v2.py",
            "sha256": _file_sha256(IMPORTER_PATH),
            "launches_renderer_or_training": False,
            "uses_current_schema_v1_runner": False,
        },
    }
    payload["receipt_payload_sha256"] = _canonical_json_sha256(payload)
    return payload


def verify_local_import(
    local_evidence_dir: Path = DEFAULT_LOCAL_EVIDENCE_DIR,
    *,
    contract: FrozenContract = FROZEN_CONTRACT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = local_evidence_dir / LOCAL_SUMMARY_NAME
    receipt_path = local_evidence_dir / LOCAL_RECEIPT_NAME
    _require(summary_path.is_file(), f"imported schema-v2 summary is missing: {summary_path}")
    _require(receipt_path.is_file(), f"compatibility receipt is missing: {receipt_path}")
    report = verify_frozen_report_bytes(summary_path.read_bytes(), contract=contract)
    receipt = _load_json(receipt_path, label="compatibility import receipt")
    digest = receipt.get("receipt_payload_sha256")
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    _require(digest == _canonical_json_sha256(payload), "compatibility receipt payload digest is invalid")
    _require(receipt.get("status") == "accepted", "compatibility receipt is not accepted")
    boundary = receipt.get("compatibility_boundary")
    _require(isinstance(boundary, Mapping), "compatibility boundary is missing")
    _require(boundary.get("source_schema_version") == 2, "compatibility receipt schema drifted")
    _require(boundary.get("decoder") == "pinned_schema_v2_compatibility_importer", "compatibility decoder drifted")
    _require(boundary.get("forbidden_decoder") == "current_schema_v1_variable_camera_runner", "forbidden decoder marker drifted")
    _require(boundary.get("dirty_schema_v1_178_179_candidate_imported") is False, "dirty schema-v1 candidate was relabelled")
    source = receipt.get("source_artifact")
    _require(isinstance(source, Mapping) and source.get("sha256") == contract.raw_sha256, "receipt raw source binding drifted")
    provenance = receipt.get("source_provenance")
    _require(
        isinstance(provenance, Mapping)
        and provenance.get("repository_commit") == contract.source_repository_commit
        and provenance.get("star_uvt_commit") == contract.source_star_uvt_commit
        and provenance.get("repository_dirty") is False
        and provenance.get("star_uvt_dirty") is False
        and provenance.get("implementation_manifest_sha256") == contract.implementation_manifest_sha256,
        "receipt source provenance drifted",
    )
    handoff = receipt.get("handoff")
    _require(isinstance(handoff, Mapping), "receipt handoff audit is missing")
    _require(handoff.get("all_receipts_verified") is True, "handoff receipts were not all verified")
    _require(handoff.get("sha256s_sha256") == contract.handoff_sha256s_sha256, "handoff receipt-file binding drifted")
    _require(handoff.get("verified_receipts") == dict(sorted(contract.handoff_receipts.items())), "handoff verified receipt set drifted")
    rendered = render_variable_assets(report)
    rendered_digests = verify_rendered_assets(rendered, contract=contract)
    _require(receipt.get("rendered_assets") == rendered_digests, "receipt rendered asset binding drifted")
    exact_svg = rendered["variable_camera_closure_death.svg"]
    publication_svg = render_publication_svg(exact_svg)
    publication = receipt.get("publication_rendering")
    _require(isinstance(publication, Mapping), "publication rendering receipt is missing")
    _require(
        publication.get("derived_from_sha256")
        == contract.rendered_assets["variable_camera_closure_death.svg"],
        "publication rendering frozen-source binding drifted",
    )
    _require(
        publication.get("sha256") == _bytes_sha256(publication_svg),
        "publication rendering digest drifted",
    )
    _require(
        publication.get("data_or_geometry_changed") is False,
        "publication rendering may not change evidence data or geometry",
    )
    _require(
        DEFAULT_PUBLICATION_SVG.is_file()
        and DEFAULT_PUBLICATION_SVG.read_bytes() == publication_svg,
        "publication rendering file is missing or stale",
    )
    importer = receipt.get("importer")
    _require(isinstance(importer, Mapping), "receipt importer record is missing")
    _require(importer.get("uses_current_schema_v1_runner") is False, "receipt permits the current schema-v1 runner")
    _require(importer.get("sha256") == _file_sha256(IMPORTER_PATH), "compatibility importer source drifted")
    return report, receipt


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _variable_component(report: Mapping[str, Any], receipt: Mapping[str, Any], summary_path: Path) -> dict[str, Any]:
    return {
        "status": "accepted",
        "accepted": True,
        "input": _display_path(summary_path),
        "input_sha256": FROZEN_CONTRACT.raw_sha256,
        "errors": [],
        "thresholds": report.get("thresholds"),
        "summary": dict(report["summary"]),
        "rows": list(report["rows"]),
        "compatibility_import": {
            "receipt": _display_path(summary_path.with_name(LOCAL_RECEIPT_NAME)),
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "source_schema_version": 2,
            "decoder": "pinned_schema_v2_compatibility_importer",
            "uses_current_schema_v1_runner": False,
            "source_repository_commit": FROZEN_CONTRACT.source_repository_commit,
            "source_star_uvt_commit": FROZEN_CONTRACT.source_star_uvt_commit,
            "implementation_manifest_sha256": FROZEN_CONTRACT.implementation_manifest_sha256,
            "handoff_sha256s_sha256": FROZEN_CONTRACT.handoff_sha256s_sha256,
        },
    }


def _moving_density_missing_component() -> dict[str, Any]:
    return {
        "accepted": False,
        "verified": False,
        "input": MOVING_CAMERA_DENSITY_INPUT,
        "errors": [],
        "rows": [],
        "status": "missing",
    }


def _moving_density_missing_input() -> dict[str, Any]:
    return {
        "component": MOVING_DENSITY_COMPONENT,
        "status": "missing",
        "expected_summary": MOVING_CAMERA_DENSITY_INPUT,
        "validation_errors": [],
        "required_contract": MOVING_CAMERA_DENSITY_CONTRACT,
    }


def _verify_payload_digest(payload: Mapping[str, Any], field: str, label: str) -> None:
    recorded = payload.get(field)
    value = dict(payload)
    value.pop(field, None)
    _require(recorded == _canonical_json_sha256(value), f"{label} payload digest is invalid before import")


def _write_ledger_markdown(bundle: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# World Tubes submission evidence ledger",
        "",
        f"Overall evidence-bundle status: **{bundle['status']}**.",
        "",
        "This ledger covers generated evidence artifacts only. Venue conversion and the manuscript-package gate remain required.",
        "",
        "| Component | Status | Accepted | Input |",
        "|---|---|---:|---|",
    ]
    for name, component in bundle["components"].items():
        input_path = component.get("input", component.get("matrix_path", ""))
        lines.append(
            f"| {name} | {component['status']} | {'yes' if component['accepted'] else 'no'} | `{input_path}` |"
        )
    public = bundle["components"].get("public_context", {})
    lines.extend(("", "## Public matrix slots", "", "| # | Role | Protocol | Seed | Policy | Status |", "|---:|---|---|---:|---|---|"))
    for slot in public.get("slots", []):
        lines.append(
            f"| {slot['ordinal']} | {slot['role']} | {slot['protocol_name']} | {slot['seed']} | {slot['world_tubes_backward_policy']} | {slot['status']} |"
        )
        for error in slot.get("errors", []):
            lines.append(f"|  |  | validation error: {error} |  |  |  |")
    if bundle["missing_runtime_inputs"]:
        lines.extend(("", "## Missing runtime inputs", ""))
        for item in bundle["missing_runtime_inputs"]:
            label = item.get("run_key", item["component"])
            lines.append(f"- `{label}`: {item['status']} — `{item['expected_summary']}`")
    else:
        lines.extend(("", "All declared runtime inputs are accepted.", ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_bundle_manifest(bundle_dir: Path, ledger: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = bundle_dir / "artifact_manifest.json"
    manifest = _load_json(manifest_path, label="paper artifact manifest")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "paper artifact manifest has no artifacts")
    refreshed = []
    for record in artifacts:
        _require(isinstance(record, Mapping), "paper artifact manifest record is invalid")
        name = record.get("path")
        _require(isinstance(name, str) and Path(name).name == name, "unsafe paper artifact path")
        path = bundle_dir / name
        _require(path.is_file(), f"paper artifact is missing after import: {name}")
        refreshed.append({"path": name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    manifest["status"] = ledger["status"]
    manifest["submission_ready"] = ledger["submission_ready"]
    manifest["ledger_sha256"] = ledger["ledger_sha256"]
    manifest["artifacts"] = refreshed
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = _canonical_json_sha256(manifest)
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest


def apply_import_to_bundle(
    *,
    bundle_dir: Path,
    local_evidence_dir: Path,
) -> dict[str, Any]:
    report, receipt = verify_local_import(local_evidence_dir)
    ledger_path = bundle_dir / "evidence_ledger.json"
    ledger = _load_json(ledger_path, label="paper evidence ledger")
    _verify_payload_digest(ledger, "ledger_sha256", "paper evidence ledger")
    components = ledger.get("components")
    _require(isinstance(components, dict), "paper evidence ledger components are invalid")
    components[VARIABLE_COMPONENT] = _variable_component(
        report,
        receipt,
        local_evidence_dir / LOCAL_SUMMARY_NAME,
    )
    if MOVING_DENSITY_COMPONENT not in components:
        components[MOVING_DENSITY_COMPONENT] = _moving_density_missing_component()

    missing = ledger.get("missing_runtime_inputs")
    _require(isinstance(missing, list), "paper evidence missing-input ledger is invalid")
    missing = [item for item in missing if item.get("component") != VARIABLE_COMPONENT]
    if components[MOVING_DENSITY_COMPONENT].get("accepted") is not True and not any(
        item.get("component") == MOVING_DENSITY_COMPONENT for item in missing
    ):
        missing.append(_moving_density_missing_input())
    ledger["missing_runtime_inputs"] = missing
    accepted = all(component.get("accepted") is True for component in components.values())
    ledger["status"] = "complete" if accepted else "incomplete"
    ledger["submission_ready"] = accepted
    imports = ledger.setdefault("compatibility_imports", {})
    _require(isinstance(imports, dict), "paper evidence compatibility_imports is invalid")
    imports[VARIABLE_COMPONENT] = {
        "receipt": _display_path(local_evidence_dir / LOCAL_RECEIPT_NAME),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "source_artifact_sha256": FROZEN_CONTRACT.raw_sha256,
        "source_schema_version": 2,
        "uses_current_schema_v1_runner": False,
    }
    ledger.pop("ledger_sha256", None)
    ledger["ledger_sha256"] = _canonical_json_sha256(ledger)

    assets = render_variable_assets(report)
    verify_rendered_assets(assets)
    for name, payload in assets.items():
        (bundle_dir / name).write_bytes(payload)
    ledger_path.write_bytes(_json_bytes(ledger))
    _write_ledger_markdown(ledger, bundle_dir / "evidence_ledger.md")
    (bundle_dir / "missing_runtime_inputs.json").write_bytes(
        _json_bytes(
            {
                "schema_version": 1,
                "submission_ready": ledger["submission_ready"],
                "readiness_scope": ledger["readiness_scope"],
                "manuscript_package_required": ledger["manuscript_package_required"],
                "inputs": ledger["missing_runtime_inputs"],
            }
        )
    )
    manifest = _refresh_bundle_manifest(bundle_dir, ledger)
    return {
        "accepted_component": VARIABLE_COMPONENT,
        "source_artifact_sha256": FROZEN_CONTRACT.raw_sha256,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "bundle_status": ledger["status"],
        "bundle_submission_ready": ledger["submission_ready"],
        "moving_camera_density_preserved": MOVING_DENSITY_COMPONENT in components,
        "ledger_sha256": ledger["ledger_sha256"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "rendered_assets": dict(FROZEN_CONTRACT.rendered_assets),
    }


def import_from_paper_freeze(
    *,
    handoff_dir: Path = DEFAULT_HANDOFF_DIR,
    frozen_raw: Path = DEFAULT_FROZEN_RAW,
    frozen_asset_dir: Path = DEFAULT_FROZEN_ASSET_DIR,
    local_evidence_dir: Path = DEFAULT_LOCAL_EVIDENCE_DIR,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
) -> dict[str, Any]:
    handoff_audit = verify_handoff(handoff_dir)
    handoff_raw = handoff_dir / "artifacts" / "variable_camera" / "summary.json"
    handoff_payload = handoff_raw.read_bytes()
    raw_payload = frozen_raw.read_bytes()
    _require(raw_payload == handoff_payload, "paper-freeze raw and handoff raw bytes disagree")
    report = verify_frozen_report_bytes(raw_payload)
    assets = render_variable_assets(report)
    rendered_digests = verify_rendered_assets(assets)
    publication_svg = render_publication_svg(
        assets["variable_camera_closure_death.svg"]
    )
    for name, expected_digest in FROZEN_CONTRACT.rendered_assets.items():
        reference = frozen_asset_dir / name
        _require(reference.is_file(), f"frozen generated reference asset is missing: {reference}")
        _require(_file_sha256(reference) == expected_digest, f"frozen generated reference asset drifted: {name}")
        _require(reference.read_bytes() == assets[name], f"generated compatibility asset bytes disagree: {name}")

    local_evidence_dir.mkdir(parents=True, exist_ok=True)
    (local_evidence_dir / LOCAL_SUMMARY_NAME).write_bytes(raw_payload)
    DEFAULT_PUBLICATION_SVG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_PUBLICATION_SVG.write_bytes(publication_svg)
    receipt = _receipt_payload(
        handoff_audit=handoff_audit,
        rendered_digests=rendered_digests,
        publication_svg_sha256=_bytes_sha256(publication_svg),
        contract=FROZEN_CONTRACT,
    )
    (local_evidence_dir / LOCAL_RECEIPT_NAME).write_bytes(_json_bytes(receipt))
    verify_local_import(local_evidence_dir)
    audit = apply_import_to_bundle(
        bundle_dir=bundle_dir,
        local_evidence_dir=local_evidence_dir,
    )
    audit["handoff_all_receipts_verified"] = handoff_audit["all_receipts_verified"]
    audit["source_repository_commit"] = FROZEN_CONTRACT.source_repository_commit
    audit["source_star_uvt_commit"] = FROZEN_CONTRACT.source_star_uvt_commit
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the exact clean World Tubes variable-camera schema-v2 paper evidence."
    )
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--frozen-raw", type=Path, default=DEFAULT_FROZEN_RAW)
    parser.add_argument("--frozen-asset-dir", type=Path, default=DEFAULT_FROZEN_ASSET_DIR)
    parser.add_argument("--local-evidence-dir", type=Path, default=DEFAULT_LOCAL_EVIDENCE_DIR)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument(
        "--verify-local",
        action="store_true",
        help="Verify the already imported raw artifact and receipt without reading paper-freeze.",
    )
    args = parser.parse_args()
    if args.verify_local:
        report, receipt = verify_local_import(args.local_evidence_dir.resolve())
        result = {
            "status": "accepted",
            "source_artifact_sha256": FROZEN_CONTRACT.raw_sha256,
            "row_count": len(report["rows"]),
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        }
    else:
        result = import_from_paper_freeze(
            handoff_dir=args.handoff_dir.resolve(),
            frozen_raw=args.frozen_raw.resolve(),
            frozen_asset_dir=args.frozen_asset_dir.resolve(),
            local_evidence_dir=args.local_evidence_dir.resolve(),
            bundle_dir=args.bundle_dir.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
