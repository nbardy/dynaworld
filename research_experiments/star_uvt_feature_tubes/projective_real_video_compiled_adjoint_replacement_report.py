from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, SCRIPT_DIR, STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projective_real_video_acceptance_envelope_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_ACCEPTANCE_ENVELOPE_OUT_DIR,
    verify_real_video_acceptance_envelope_report,
)
from projective_real_video_broad10_quality_tether_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BROAD10_QUALITY_OUT_DIR,
    verify_real_video_broad10_quality_tether_report,
)
from projective_real_video_multiscene_media_tether_report import (  # noqa: E402
    verify_real_video_multiscene_media_tether_report,
)
from projective_real_video_multiscene_trainer_matrix import (  # noqa: E402
    verify_real_video_multiscene_trainer_matrix_report,
)
from projective_real_video_timing_protocol_acceptance_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_TIMING_PROTOCOL_OUT_DIR,
    verify_real_video_timing_protocol_acceptance_report,
)
from projective_shared_work_goal_audit import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_SHARED_WORK_OUT_DIR,
    verify_shared_work_goal_audit,
)
from report_artifacts import write_report_json, write_report_text  # noqa: E402


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement"
)
DEFAULT_BROAD10_TRAINER_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10"
    / "summary.json"
)
DEFAULT_BROAD10_CASE_DIR = DEFAULT_BROAD10_TRAINER_REPORT.parent / "cases"
DEFAULT_BROAD10_QUALITY_REPORT = DEFAULT_BROAD10_QUALITY_OUT_DIR / "summary.json"
DEFAULT_BROAD10_MEDIA_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_broad10_media_tether"
    / "summary.json"
)
DEFAULT_ACCEPTANCE_ENVELOPE_REPORT = DEFAULT_ACCEPTANCE_ENVELOPE_OUT_DIR / "summary.json"
DEFAULT_TIMING_PROTOCOL_REPORT = DEFAULT_TIMING_PROTOCOL_OUT_DIR / "summary.json"
DEFAULT_SHARED_WORK_REPORT = DEFAULT_SHARED_WORK_OUT_DIR / "summary.json"

TRAINER_SOURCE = ROOT / "src" / "train" / "star_uvt_feature_overfit_trainer.py"
HARNESS_SOURCE = STAR_UVT_ROOT / "research_project" / "trainer_harness" / "tile_metal_autograd.py"
PROJECTIVE_TRACE_SOURCE = STAR_UVT_ROOT / "torch_gsplat_bridge_star_uvt" / "projective_trace.py"

EVIDENCE_ORDER = (
    "real_video_broad10_trainer_matrix",
    "real_video_broad10_quality_tether",
    "real_video_broad10_media_tether",
    "real_video_acceptance_envelope",
    "real_video_timing_protocol_acceptance",
    "shared_work",
)

GRAD_FLAGS = (
    "raw_feature_grad_seen",
    "center_uv_grad_seen",
    "center_t_grad_seen",
    "velocity_uv_grad_seen",
    "raw_precision_grad_seen",
    "raw_opacity_grad_seen",
    "colorizer_grad_seen",
)

Verifier = Callable[[dict[str, Any]], list[str]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, verifier: Verifier) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(path),
        "benchmark": report.get("benchmark"),
        "status": report.get("status"),
        "verifier_errors": verifier(report),
        "summary": report.get("summary", {}),
    }


def _summary(report: dict[str, Any], key: str) -> dict[str, Any]:
    return report["evidence"][key]["summary"]


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _all_case_rows(case_payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows = case_payloads.get("rows")
    return rows if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else []


def _all_case_flag(rows: list[dict[str, Any]], key: str) -> bool:
    return bool(rows) and all(row.get(key) is True for row in rows)


def _source_contract() -> dict[str, Any]:
    trainer_source = TRAINER_SOURCE.read_text(encoding="utf-8") if TRAINER_SOURCE.exists() else ""
    harness_source = HARNESS_SOURCE.read_text(encoding="utf-8") if HARNESS_SOURCE.exists() else ""
    projective_trace_source = (
        PROJECTIVE_TRACE_SOURCE.read_text(encoding="utf-8") if PROJECTIVE_TRACE_SOURCE.exists() else ""
    )
    checks = {
        "trainer_selects_projective_interval_route": (
            "elif projective_interval_runtime_enabled:" in trainer_source
            and "_render_projective_interval_feature_tubes_autograd(" in trainer_source
            and "feature_state.render()" in trainer_source
        ),
        "harness_defines_interval_autograd_function": (
            "class _ProjectiveCellIntervalBackward(torch.autograd.Function):" in harness_source
        ),
        "harness_forward_calls_interval_metal": (
            "return render_projective_trace_cell_interval_atlas_metal(" in harness_source
        ),
        "harness_backward_calls_direct_interval_metal_vjp": (
            "grads = direct_backward_projective_trace_cell_interval_atlas_metal(" in harness_source
        ),
        "harness_wrapper_applies_static_compiled_atlas": (
            "_ProjectiveCellIntervalBackward.apply(" in harness_source
            and "_static_projective_cell_atlas(atlas)" in harness_source
        ),
        "bridge_backward_treats_visibility_and_bins_as_constants": (
            "Visibility order and" in projective_trace_source
            and "tile membership are treated as compiled atlas constants" in projective_trace_source
        ),
        "bridge_forward_consumes_interval_compressed_atlas": (
            "hot-path shape that consumes the interval-compressed atlas object directly"
            in projective_trace_source
        ),
    }
    return {
        "source_paths": {
            "trainer": str(TRAINER_SOURCE),
            "trainer_harness": str(HARNESS_SOURCE),
            "projective_trace": str(PROJECTIVE_TRACE_SOURCE),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _case_path(case_dir: Path, row: dict[str, Any]) -> Path:
    return case_dir / f"{row['scene_id']}_{row['policy']}.json"


def _positive_timing(payload: dict[str, Any], key: str) -> bool:
    mean = payload.get("mean_timing_ms")
    steps = payload.get("step_timings_ms")
    if not isinstance(mean, dict) or not isinstance(steps, list) or not steps:
        return False
    mean_value = mean.get(key)
    if not isinstance(mean_value, int | float) or not math.isfinite(float(mean_value)) or float(mean_value) <= 0.0:
        return False
    for step in steps:
        if not isinstance(step, dict):
            return False
        value = step.get(key)
        if not isinstance(value, int | float) or not math.isfinite(float(value)) or float(value) <= 0.0:
            return False
    return True


def _case_payload_row(case_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = _case_path(case_dir, row)
    payload = _load_json(path) if path.exists() else {}
    grad_flags = {name: payload.get(name) is True for name in GRAD_FLAGS}
    policy = str(row.get("policy"))
    measured = policy == "measured"
    return {
        "scene_id": str(row.get("scene_id")),
        "policy": policy,
        "path": str(path),
        "exists": path.exists(),
        "pass": payload.get("pass") is True,
        "loss_decreased": payload.get("loss_decreased") is True,
        "projective_interval_enabled": payload.get("projective_interval_enabled") is True,
        "projective_interval_runtime_enabled": payload.get("projective_interval_runtime_enabled") is True,
        "projective_interval_fallback_render_mode": str(payload.get("projective_interval_fallback_render_mode")),
        "projective_interval_alpha_render_mode": str(payload.get("projective_interval_alpha_render_mode")),
        "projective_interval_cache_fallback_marks": int(
            payload.get("projective_interval_cache_fallback_marks") or 0
        ),
        "projective_interval_cache_visibility_stratifications": int(
            payload.get("projective_interval_cache_visibility_stratifications") or 0
        ),
        "projective_interval_cache_support_rebins": int(
            payload.get("projective_interval_cache_support_rebins") or 0
        ),
        "projective_interval_cache_stale_refreshes": int(
            payload.get("projective_interval_cache_stale_refreshes") or 0
        ),
        "projective_interval_cache_rebuilds": int(payload.get("projective_interval_cache_rebuilds") or 0),
        "projective_interval_cache_live_updates": int(payload.get("projective_interval_cache_live_updates") or 0),
        "projective_interval_refresh_policy": str(payload.get("projective_interval_refresh_policy")),
        "requested_render_mode": str(payload.get("requested_render_mode")),
        "effective_render_mode": str(payload.get("effective_render_mode")),
        "kernel_backward_mode": str(payload.get("kernel_backward_mode")),
        "feature_target_image_vjp_mode": str(payload.get("feature_target_image_vjp_mode")),
        "rgb_loss_weight": float(payload.get("rgb_loss_weight") or 0.0),
        "feature_target_enabled": payload.get("feature_target_enabled") is True,
        "feature_target_loss_weight": float(payload.get("feature_target_loss_weight") or 0.0),
        "mode_fallback_required": payload.get("mode_fallback_required") is True,
        "gradient_flags": grad_flags,
        "all_gradient_flags_present": all(grad_flags.values()),
        "backward_timing_present": _positive_timing(payload, "backward_ms"),
        "render_forward_timing_present": _positive_timing(payload, "render_forward_ms"),
        "uses_rgb_direct_loss": (
            float(payload.get("rgb_loss_weight") or 0.0) > 0.0
            and payload.get("feature_target_enabled") is not True
            and float(payload.get("feature_target_loss_weight") or 0.0) == 0.0
        ),
        "uses_projective_interval_main_path": (
            payload.get("projective_interval_enabled") is True
            and payload.get("projective_interval_runtime_enabled") is True
            and str(payload.get("feature_target_image_vjp_mode")) == "autograd"
            and str(payload.get("projective_interval_fallback_render_mode")) == "mixed"
            and int(payload.get("projective_interval_cache_fallback_marks") or 0) == 0
            and int(payload.get("projective_interval_cache_visibility_stratifications") or 0) == 0
            and int(payload.get("projective_interval_cache_support_rebins") or 0) == 0
            and (not measured or int(payload.get("projective_interval_cache_stale_refreshes") or 0) == 0)
        ),
        "base_mode_is_practical_direct_atomic": (
            str(payload.get("requested_render_mode")) == "feature_direct_atomic"
            and str(payload.get("effective_render_mode")) == "feature_direct_atomic"
            and str(payload.get("kernel_backward_mode")) == "direct_atomic"
            and payload.get("mode_fallback_required") is not True
        ),
    }


def _case_payloads(trainer_report: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    rows = trainer_report.get("rows")
    if not isinstance(rows, list):
        rows = []
    case_rows = [_case_payload_row(case_dir, row) for row in rows if isinstance(row, dict)]
    return {
        "case_dir": str(case_dir),
        "expected_case_count": len(rows),
        "rows": case_rows,
        "missing_paths": [row["path"] for row in case_rows if row.get("exists") is not True],
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    trainer = _summary(report, "real_video_broad10_trainer_matrix")
    quality = _summary(report, "real_video_broad10_quality_tether")
    media = _summary(report, "real_video_broad10_media_tether")
    acceptance = _summary(report, "real_video_acceptance_envelope")
    timing_protocol = _summary(report, "real_video_timing_protocol_acceptance")
    shared = _summary(report, "shared_work")
    case_rows = _all_case_rows(report.get("case_payloads", {}))
    source_contract = report.get("source_contract", {})
    source_checks_pass = bool(source_contract.get("all_checks_pass"))
    all_underlying = all(
        isinstance(report["evidence"].get(key), dict)
        and report["evidence"][key].get("status") == "ok"
        and report["evidence"][key].get("verifier_errors") == []
        and isinstance(report["evidence"][key].get("summary"), dict)
        for key in EVIDENCE_ORDER
    )
    measured_rows = [row for row in case_rows if row.get("policy") == "measured"]
    cadence_rows = [row for row in case_rows if row.get("policy") == "cadence"]
    broad_context_passes = (
        int(trainer["distinct_youtube_id_count"]) >= 10
        and int(trainer["row_count"]) >= 20
        and int(quality["distinct_youtube_id_count"]) >= 10
        and int(media["distinct_youtube_id_count"]) >= 10
        and int(acceptance["broad_frame_count_count"]) >= 4
        and bool(timing_protocol["final_timing_protocol_accepted"])
    )
    clean_cache_and_support = (
        bool(trainer["all_rows_fallback_free"])
        and bool(trainer["all_rows_visibility_stratification_free"])
        and int(trainer["max_measured_support_rebins"]) == 0
        and int(trainer["max_measured_stale_refreshes"]) == 0
        and float(trainer["max_measured_vs_cadence_rebuild_ratio"]) <= 0.5
    )
    shared_work_passes = (
        float(shared["orbit_payload_growth_ratio"]) <= 0.20
        and float(shared["trained_shared_to_replay_interval_growth_ratio"]) <= 0.25
        and float(shared["max_trained_final_backward_ms_ratio"]) <= 0.25
    )
    all_case_payloads_present = (
        int(report.get("case_payloads", {}).get("expected_case_count") or 0) == len(case_rows)
        and not report.get("case_payloads", {}).get("missing_paths")
    )
    all_case_training_ok = (
        _all_case_flag(case_rows, "pass")
        and _all_case_flag(case_rows, "loss_decreased")
        and _all_case_flag(case_rows, "all_gradient_flags_present")
        and _all_case_flag(case_rows, "backward_timing_present")
        and _all_case_flag(case_rows, "render_forward_timing_present")
    )
    all_case_path_ok = (
        _all_case_flag(case_rows, "uses_projective_interval_main_path")
        and _all_case_flag(case_rows, "uses_rgb_direct_loss")
        and _all_case_flag(case_rows, "base_mode_is_practical_direct_atomic")
    )
    measured_cache_reuse_ok = (
        bool(measured_rows)
        and bool(cadence_rows)
        and all(int(row["projective_interval_cache_rebuilds"]) == 1 for row in measured_rows)
        and all(int(row["projective_interval_cache_live_updates"]) >= 3 for row in measured_rows)
        and all(int(row["projective_interval_cache_rebuilds"]) >= 2 for row in cadence_rows)
    )
    final_accepted = (
        all_underlying
        and source_checks_pass
        and broad_context_passes
        and clean_cache_and_support
        and shared_work_passes
        and all_case_payloads_present
        and all_case_training_ok
        and all_case_path_ok
        and measured_cache_reuse_ok
        and bool(quality["all_gradient_flags_present"])
        and bool(media["all_gradient_flags_present"])
        and bool(quality["all_measured_psnr_improves"])
        and bool(media["all_measured_psnr_improves"])
    )
    return {
        "underlying_report_count": len(EVIDENCE_ORDER),
        "all_underlying_verifiers_pass": all_underlying,
        "source_contract_checks_pass": source_checks_pass,
        "broad10_trainer_distinct_youtube_id_count": int(trainer["distinct_youtube_id_count"]),
        "broad10_trainer_row_count": int(trainer["row_count"]),
        "broad10_quality_distinct_youtube_id_count": int(quality["distinct_youtube_id_count"]),
        "broad10_media_distinct_youtube_id_count": int(media["distinct_youtube_id_count"]),
        "broad_frame_count_count": int(acceptance["broad_frame_count_count"]),
        "timing_protocol_accepted": bool(timing_protocol["final_timing_protocol_accepted"]),
        "broad_context_passes": broad_context_passes,
        "clean_cache_and_support": clean_cache_and_support,
        "max_measured_support_rebins": int(trainer["max_measured_support_rebins"]),
        "max_measured_stale_refreshes": int(trainer["max_measured_stale_refreshes"]),
        "max_measured_vs_cadence_rebuild_ratio": float(trainer["max_measured_vs_cadence_rebuild_ratio"]),
        "case_payload_count": len(case_rows),
        "expected_case_payload_count": int(report.get("case_payloads", {}).get("expected_case_count") or 0),
        "missing_case_payload_count": len(report.get("case_payloads", {}).get("missing_paths") or []),
        "all_case_payloads_present": all_case_payloads_present,
        "all_case_training_ok": all_case_training_ok,
        "all_case_path_ok": all_case_path_ok,
        "all_cases_projective_interval_main_path": _all_case_flag(case_rows, "uses_projective_interval_main_path"),
        "all_cases_rgb_direct_loss": _all_case_flag(case_rows, "uses_rgb_direct_loss"),
        "all_cases_base_mode_direct_atomic": _all_case_flag(case_rows, "base_mode_is_practical_direct_atomic"),
        "all_cases_gradient_flags_present": _all_case_flag(case_rows, "all_gradient_flags_present"),
        "all_cases_backward_timing_present": _all_case_flag(case_rows, "backward_timing_present"),
        "all_cases_render_forward_timing_present": _all_case_flag(case_rows, "render_forward_timing_present"),
        "measured_case_count": len(measured_rows),
        "cadence_case_count": len(cadence_rows),
        "measured_cache_reuse_ok": measured_cache_reuse_ok,
        "quality_media_tethers_preserve_gradients": (
            bool(quality["all_gradient_flags_present"]) and bool(media["all_gradient_flags_present"])
        ),
        "quality_media_tethers_improve_psnr": (
            bool(quality["all_measured_psnr_improves"]) and bool(media["all_measured_psnr_improves"])
        ),
        "shared_work_passes": shared_work_passes,
        "orbit_payload_growth_ratio": float(shared["orbit_payload_growth_ratio"]),
        "trained_shared_to_replay_interval_growth_ratio": float(
            shared["trained_shared_to_replay_interval_growth_ratio"]
        ),
        "max_trained_final_backward_ms_ratio": float(shared["max_trained_final_backward_ms_ratio"]),
        "final_compiled_adjoint_replacement_accepted": final_accepted,
        "compiled_trainer_replacement_gap": 0 if final_accepted else 1,
        "does_not_prove_completion": report.get("does_not_prove_completion") is True,
    }


def verify_real_video_compiled_adjoint_replacement_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_compiled_adjoint_replacement":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("proves_compiled_adjoint_replacement") is not True:
        errors.append("proves_compiled_adjoint_replacement must be true")
    if report.get("does_not_prove_completion") is not True:
        errors.append("does_not_prove_completion must remain true")
    theory_contract = report.get("theory_contract")
    theory_text = theory_contract.lower() if isinstance(theory_contract, str) else ""
    for phrase in (
        "sensor-time trace atlas",
        "interval metal direct vjp",
        "compiled constants",
        "not deterministic compact static-star promotion",
        "does not prove full goal completion",
    ):
        if phrase not in theory_text:
            errors.append(f"theory_contract must include {phrase!r}")

    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
        return errors
    for key in EVIDENCE_ORDER:
        row = evidence.get(key)
        if not isinstance(row, dict):
            errors.append(f"evidence {key} must be an object")
            continue
        if row.get("status") != "ok":
            errors.append(f"evidence {key} status must be ok, got {row.get('status')!r}")
        if row.get("verifier_errors"):
            errors.append(f"evidence {key} verifier failed: {row.get('verifier_errors')}")
        if not isinstance(row.get("summary"), dict):
            errors.append(f"evidence {key} summary must be an object")
    source_contract = report.get("source_contract")
    if not isinstance(source_contract, dict):
        errors.append("source_contract must be an object")
        return errors
    checks = source_contract.get("checks")
    if not isinstance(checks, dict):
        errors.append("source_contract.checks must be an object")
        return errors
    for key, value in checks.items():
        if value is not True:
            errors.append(f"source contract check {key} must be true")
    if source_contract.get("all_checks_pass") is not True:
        errors.append("source_contract all_checks_pass must be true")

    case_payloads = report.get("case_payloads")
    if not isinstance(case_payloads, dict):
        errors.append("case_payloads must be an object")
        return errors
    case_rows = _all_case_rows(case_payloads)
    if not case_rows:
        errors.append("case_payloads.rows must be non-empty")
    if case_payloads.get("missing_paths"):
        errors.append(f"case payloads must all exist, missing {case_payloads.get('missing_paths')!r}")
    for row in case_rows:
        label = f"{row.get('scene_id')} {row.get('policy')}"
        for key in (
            "exists",
            "pass",
            "loss_decreased",
            "uses_projective_interval_main_path",
            "uses_rgb_direct_loss",
            "base_mode_is_practical_direct_atomic",
            "all_gradient_flags_present",
            "backward_timing_present",
            "render_forward_timing_present",
        ):
            if row.get(key) is not True:
                errors.append(f"{label} case {key} must be true")
        if row.get("projective_interval_fallback_render_mode") != "mixed":
            errors.append(f"{label} fallback render mode must be mixed")
        if _finite_int(row.get("projective_interval_cache_fallback_marks"), f"{label} fallback marks", errors) != 0:
            errors.append(f"{label} fallback marks must be zero")
        if _finite_int(row.get("projective_interval_cache_visibility_stratifications"), f"{label} visibility strata", errors) != 0:
            errors.append(f"{label} visibility stratifications must be zero")
        if _finite_int(row.get("projective_interval_cache_support_rebins"), f"{label} support rebins", errors) != 0:
            errors.append(f"{label} support rebins must be zero")
        if row.get("policy") == "measured":
            if _finite_int(row.get("projective_interval_cache_stale_refreshes"), f"{label} stale refreshes", errors) != 0:
                errors.append(f"{label} stale refreshes must be zero")
            if _finite_int(row.get("projective_interval_cache_live_updates"), f"{label} live updates", errors) < 3:
                errors.append(f"{label} measured live updates must be at least three")
            if _finite_int(row.get("projective_interval_cache_rebuilds"), f"{label} rebuilds", errors) != 1:
                errors.append(f"{label} measured rebuilds must be one")

    trainer = _summary(report, "real_video_broad10_trainer_matrix")
    quality = _summary(report, "real_video_broad10_quality_tether")
    media = _summary(report, "real_video_broad10_media_tether")
    acceptance = _summary(report, "real_video_acceptance_envelope")
    timing = _summary(report, "real_video_timing_protocol_acceptance")
    shared = _summary(report, "shared_work")
    if _finite_int(trainer.get("distinct_youtube_id_count"), "trainer distinct sources", errors) < 10:
        errors.append("compiled replacement trainer evidence must cover at least ten source-distinct videos")
    if _finite_int(trainer.get("row_count"), "trainer row count", errors) < 20:
        errors.append("compiled replacement trainer evidence must cover at least twenty rows")
    for key in (
        "all_rows_pass",
        "all_rows_loss_decreased",
        "all_rows_fallback_free",
        "all_rows_visibility_stratification_free",
        "all_measured_loss_matches_cadence",
    ):
        if trainer.get(key) is not True:
            errors.append(f"trainer summary {key} must be true")
    if _finite_int(trainer.get("max_measured_support_rebins"), "trainer support rebins", errors) != 0:
        errors.append("trainer support rebins must be zero")
    if _finite_int(trainer.get("max_measured_stale_refreshes"), "trainer stale refreshes", errors) != 0:
        errors.append("trainer stale refreshes must be zero")
    if _finite_float(trainer.get("max_measured_vs_cadence_rebuild_ratio"), "trainer rebuild ratio", errors) > 0.5:
        errors.append("trainer measured/cadence rebuild ratio must stay at or below 0.5")
    if _finite_int(quality.get("distinct_youtube_id_count"), "quality distinct sources", errors) < 10:
        errors.append("quality tether must cover ten source-distinct videos")
    if _finite_int(media.get("distinct_youtube_id_count"), "media distinct sources", errors) < 10:
        errors.append("media tether must cover ten source-distinct videos")
    for label, tether in (("quality", quality), ("media", media)):
        if tether.get("all_gradient_flags_present") is not True:
            errors.append(f"{label} tether gradient flags must be present")
        if tether.get("all_measured_psnr_improves") is not True:
            errors.append(f"{label} tether measured PSNR must improve")
    if _finite_int(acceptance.get("broad_frame_count_count"), "broad frame-count count", errors) < 4:
        errors.append("acceptance envelope must retain at least four frame counts")
    if timing.get("final_timing_protocol_accepted") is not True:
        errors.append("timing protocol must be accepted")
    if _finite_float(shared.get("orbit_payload_growth_ratio"), "orbit payload ratio", errors) > 0.20:
        errors.append("orbit payload growth ratio must pass shared-work threshold")
    if _finite_float(shared.get("trained_shared_to_replay_interval_growth_ratio"), "trained interval ratio", errors) > 0.25:
        errors.append("trained interval growth ratio must pass shared-work threshold")
    if _finite_float(shared.get("max_trained_final_backward_ms_ratio"), "backward ratio", errors) > 0.25:
        errors.append("backward shared-work ratio must pass threshold")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
                errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
        elif actual != expected_value:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    if summary.get("final_compiled_adjoint_replacement_accepted") is not True:
        errors.append("final compiled-adjoint replacement must be accepted")
    if summary.get("compiled_trainer_replacement_gap") != 0:
        errors.append("compiled_trainer_replacement_gap must be zero")
    if summary.get("does_not_prove_completion") is not True:
        errors.append("summary must preserve does_not_prove_completion")
    return errors


def assert_real_video_compiled_adjoint_replacement_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_compiled_adjoint_replacement_report(report)
    if errors:
        raise AssertionError("real-video compiled-adjoint replacement failed:\n- " + "\n- ".join(errors))


def run_report(
    *,
    broad10_trainer_report: Path = DEFAULT_BROAD10_TRAINER_REPORT,
    broad10_case_dir: Path = DEFAULT_BROAD10_CASE_DIR,
    broad10_quality_report: Path = DEFAULT_BROAD10_QUALITY_REPORT,
    broad10_media_report: Path = DEFAULT_BROAD10_MEDIA_REPORT,
    acceptance_envelope_report: Path = DEFAULT_ACCEPTANCE_ENVELOPE_REPORT,
    timing_protocol_report: Path = DEFAULT_TIMING_PROTOCOL_REPORT,
    shared_work_report: Path = DEFAULT_SHARED_WORK_REPORT,
) -> dict[str, Any]:
    trainer_report = _load_json(broad10_trainer_report)
    evidence = {
        "real_video_broad10_trainer_matrix": {
            "path": str(broad10_trainer_report),
            "benchmark": trainer_report.get("benchmark"),
            "status": trainer_report.get("status"),
            "verifier_errors": verify_real_video_multiscene_trainer_matrix_report(trainer_report),
            "summary": trainer_report.get("summary", {}),
        },
        "real_video_broad10_quality_tether": _artifact(
            broad10_quality_report,
            verify_real_video_broad10_quality_tether_report,
        ),
        "real_video_broad10_media_tether": _artifact(
            broad10_media_report,
            verify_real_video_multiscene_media_tether_report,
        ),
        "real_video_acceptance_envelope": _artifact(
            acceptance_envelope_report,
            verify_real_video_acceptance_envelope_report,
        ),
        "real_video_timing_protocol_acceptance": _artifact(
            timing_protocol_report,
            verify_real_video_timing_protocol_acceptance_report,
        ),
        "shared_work": _artifact(shared_work_report, verify_shared_work_goal_audit),
    }
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_compiled_adjoint_replacement",
        "base_domain": "source-distinct real-video projective interval trainer replacement",
        "theory_contract": (
            "This report proves the current practical Sensor-Time Trace Atlas trainer replacement: the real-video "
            "trainer uses compiled projective interval traces, and the renderer autograd path lowers to the interval "
            "Metal direct VJP with visibility order and tile membership treated as compiled constants. It is not "
            "deterministic compact static-STAR promotion; it is the practical direct-atomic RGB trainer route backed "
            "by the compiled interval adjoint. This does not prove full goal completion."
        ),
        "proves_compiled_adjoint_replacement": True,
        "does_not_prove_completion": True,
        "evidence": evidence,
        "source_contract": _source_contract(),
        "case_payloads": _case_payloads(trainer_report, broad10_case_dir),
        "summary": {},
    }
    report["summary"] = summarize(report)
    errors = verify_real_video_compiled_adjoint_replacement_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def write_report(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report_json(out_dir / "summary.json", report)
    lines = [
        "# STAR UVT Real-Video Compiled-Adjoint Replacement",
        "",
        "This is a compiled-adjoint replacement acceptance artifact, not a full goal-completion claim.",
        "",
        "## Summary",
        "",
        f"- accepted: {report['summary']['final_compiled_adjoint_replacement_accepted']}",
        f"- source contract checks pass: {report['summary']['source_contract_checks_pass']}",
        f"- broad10 trainer sources: {report['summary']['broad10_trainer_distinct_youtube_id_count']}",
        f"- broad frame-count count: {report['summary']['broad_frame_count_count']}",
        f"- case payloads: {report['summary']['case_payload_count']}",
        f"- all cases use projective interval main path: {report['summary']['all_cases_projective_interval_main_path']}",
        f"- all cases expose renderer gradients: {report['summary']['all_cases_gradient_flags_present']}",
        f"- measured cache reuse ok: {report['summary']['measured_cache_reuse_ok']}",
        f"- replacement gap: {report['summary']['compiled_trainer_replacement_gap']}",
        "",
    ]
    write_report_text(out_dir / "summary.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--broad10-trainer-report", type=Path, default=DEFAULT_BROAD10_TRAINER_REPORT)
    parser.add_argument("--broad10-case-dir", type=Path, default=DEFAULT_BROAD10_CASE_DIR)
    parser.add_argument("--broad10-quality-report", type=Path, default=DEFAULT_BROAD10_QUALITY_REPORT)
    parser.add_argument("--broad10-media-report", type=Path, default=DEFAULT_BROAD10_MEDIA_REPORT)
    parser.add_argument("--acceptance-envelope-report", type=Path, default=DEFAULT_ACCEPTANCE_ENVELOPE_REPORT)
    parser.add_argument("--timing-protocol-report", type=Path, default=DEFAULT_TIMING_PROTOCOL_REPORT)
    parser.add_argument("--shared-work-report", type=Path, default=DEFAULT_SHARED_WORK_REPORT)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        assert_real_video_compiled_adjoint_replacement_report(report)
        print(f"verified {args.verify_report}")
        return
    report = run_report(
        broad10_trainer_report=args.broad10_trainer_report,
        broad10_case_dir=args.broad10_case_dir,
        broad10_quality_report=args.broad10_quality_report,
        broad10_media_report=args.broad10_media_report,
        acceptance_envelope_report=args.acceptance_envelope_report,
        timing_protocol_report=args.timing_protocol_report,
        shared_work_report=args.shared_work_report,
    )
    assert_real_video_compiled_adjoint_replacement_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
