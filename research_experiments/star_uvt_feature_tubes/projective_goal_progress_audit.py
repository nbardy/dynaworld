from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_experiments.star_uvt_feature_tubes.projective_bundle_gauge_gradient_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BUNDLE_GRADIENT_OUT_DIR,
    verify_bundle_gauge_gradient_report,
)
from research_experiments.star_uvt_feature_tubes.projective_bundle_gauge_invariance_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BUNDLE_INVARIANCE_OUT_DIR,
    verify_bundle_gauge_invariance_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_gauge_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_OUT_DIR,
    verify_camera_family_gauge_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_gauge_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_OUT_DIR,
    verify_camera_family_2d_gauge_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_METAL_LOWERING_OUT_DIR,
    verify_camera_family_2d_metal_lowering_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_chain_rule_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_METAL_CHAIN_RULE_OUT_DIR,
    verify_camera_family_2d_metal_chain_rule_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_materialized_batch_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_MATERIALIZED_BATCH_OUT_DIR,
    verify_camera_family_2d_materialized_batch_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_native_eval_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_NATIVE_EVAL_OUT_DIR,
    verify_camera_family_2d_native_eval_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_native_interval_forward_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_FORWARD_OUT_DIR,
    verify_camera_family_2d_native_interval_forward_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_native_interval_backward_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_BACKWARD_OUT_DIR,
    verify_camera_family_2d_native_interval_backward_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_tile_order_reuse_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_REUSE_OUT_DIR,
    verify_camera_family_2d_tile_order_reuse_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_tile_order_strata_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_STRATA_OUT_DIR,
    verify_camera_family_2d_tile_order_strata_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_active_set_strata_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_ACTIVE_SET_STRATA_OUT_DIR,
    verify_camera_family_2d_active_set_strata_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_active_set_distribution_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_ACTIVE_SET_DISTRIBUTION_OUT_DIR,
    verify_real_active_set_distribution_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_shared_work_scaling import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_2D_SHARED_WORK_OUT_DIR,
    verify_camera_family_2d_shared_work_scaling_report,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_shared_work_scaling import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_CAMERA_FAMILY_SHARED_WORK_OUT_DIR,
    verify_camera_family_shared_work_scaling_report,
)
from research_experiments.star_uvt_feature_tubes.projective_interval_trainer_frame_scaling_benchmark import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_INTERVAL_TRAINER_OUT_DIR,
    verify_interval_trainer_frame_scaling_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_trainer_frame_scaling_benchmark import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_TRAINER_OUT_DIR,
    verify_real_video_trainer_frame_scaling_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_guarded_support_matrix_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_GUARDED_SUPPORT_MATRIX_OUT_DIR,
    verify_real_video_guarded_support_matrix_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_trainer_matrix import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_MULTISCENE_TRAINER_MATRIX_OUT_DIR,
    verify_real_video_multiscene_trainer_matrix_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_frame_scaling_matrix import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_MULTISCENE_FRAME_SCALING_MATRIX_OUT_DIR,
    verify_real_video_multiscene_frame_scaling_matrix_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_extended_frame_scaling_diagnostic_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_FRAME_SCALING_DIAGNOSTIC_OUT_DIR,
    verify_extended_frame_scaling_diagnostic_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_quality_tether_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_MULTISCENE_QUALITY_TETHER_OUT_DIR,
    verify_real_video_multiscene_quality_tether_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_extended_quality_tether_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_QUALITY_TETHER_OUT_DIR,
    verify_real_video_multiscene_extended_quality_tether_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_media_tether_report import (  # noqa: E402
    CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE,
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_MULTISCENE_MEDIA_TETHER_OUT_DIR,
    verify_real_video_multiscene_media_tether_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_acceptance_envelope_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_ACCEPTANCE_ENVELOPE_OUT_DIR,
    verify_real_video_acceptance_envelope_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_timing_variance_envelope_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_TIMING_VARIANCE_ENVELOPE_OUT_DIR,
    verify_real_video_timing_variance_envelope_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_compiled_adjoint_replacement_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_COMPILED_ADJOINT_REPLACEMENT_OUT_DIR,
    verify_real_video_compiled_adjoint_replacement_report,
)
from research_experiments.star_uvt_feature_tubes.projective_shared_work_goal_audit import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_SHARED_WORK_OUT_DIR,
    verify_shared_work_goal_current_acceptance,
    verify_shared_work_goal_audit,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_goal_progress_audit"
DEFAULT_BUNDLE_INVARIANCE_REPORT = DEFAULT_BUNDLE_INVARIANCE_OUT_DIR / "summary.json"
DEFAULT_BUNDLE_GRADIENT_REPORT = DEFAULT_BUNDLE_GRADIENT_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_REPORT = DEFAULT_CAMERA_FAMILY_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_REPORT = DEFAULT_CAMERA_FAMILY_2D_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_METAL_LOWERING_REPORT = DEFAULT_CAMERA_FAMILY_2D_METAL_LOWERING_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_METAL_CHAIN_RULE_REPORT = DEFAULT_CAMERA_FAMILY_2D_METAL_CHAIN_RULE_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_MATERIALIZED_BATCH_REPORT = DEFAULT_CAMERA_FAMILY_2D_MATERIALIZED_BATCH_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_NATIVE_EVAL_REPORT = DEFAULT_CAMERA_FAMILY_2D_NATIVE_EVAL_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_FORWARD_REPORT = (
    DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_FORWARD_OUT_DIR / "summary.json"
)
DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_BACKWARD_REPORT = (
    DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_BACKWARD_OUT_DIR / "summary.json"
)
DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_REUSE_REPORT = DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_REUSE_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_STRATA_REPORT = DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_STRATA_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_ACTIVE_SET_STRATA_REPORT = (
    DEFAULT_CAMERA_FAMILY_2D_ACTIVE_SET_STRATA_OUT_DIR / "summary.json"
)
DEFAULT_REAL_ACTIVE_SET_DISTRIBUTION_REPORT = DEFAULT_REAL_ACTIVE_SET_DISTRIBUTION_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_SHARED_WORK_REPORT = DEFAULT_CAMERA_FAMILY_SHARED_WORK_OUT_DIR / "summary.json"
DEFAULT_CAMERA_FAMILY_2D_SHARED_WORK_REPORT = DEFAULT_CAMERA_FAMILY_2D_SHARED_WORK_OUT_DIR / "summary.json"
DEFAULT_INTERVAL_TRAINER_REPORT = DEFAULT_INTERVAL_TRAINER_OUT_DIR / "summary.json"
DEFAULT_REAL_VIDEO_TRAINER_REPORT = DEFAULT_REAL_VIDEO_TRAINER_OUT_DIR / "summary.json"
DEFAULT_REAL_VIDEO_GUARDED_SUPPORT_MATRIX_REPORT = (
    DEFAULT_REAL_VIDEO_GUARDED_SUPPORT_MATRIX_OUT_DIR / "summary.json"
)
DEFAULT_REAL_VIDEO_MULTISCENE_TRAINER_MATRIX_REPORT = (
    DEFAULT_REAL_VIDEO_MULTISCENE_TRAINER_MATRIX_OUT_DIR / "summary.json"
)
DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_TRAINER_MATRIX_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5"
    / "summary.json"
)
DEFAULT_REAL_VIDEO_MULTISCENE_FRAME_SCALING_MATRIX_REPORT = (
    DEFAULT_REAL_VIDEO_MULTISCENE_FRAME_SCALING_MATRIX_OUT_DIR / "summary.json"
)
DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_FRAME_SCALING_DIAGNOSTIC_REPORT = (
    DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_FRAME_SCALING_DIAGNOSTIC_OUT_DIR / "summary.json"
)
DEFAULT_REAL_VIDEO_MULTISCENE_QUALITY_TETHER_REPORT = (
    DEFAULT_REAL_VIDEO_MULTISCENE_QUALITY_TETHER_OUT_DIR / "summary.json"
)
DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_QUALITY_TETHER_REPORT = (
    DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_QUALITY_TETHER_OUT_DIR / "summary.json"
)
DEFAULT_REAL_VIDEO_MULTISCENE_MEDIA_TETHER_REPORT = (
    DEFAULT_REAL_VIDEO_MULTISCENE_MEDIA_TETHER_OUT_DIR / "summary.json"
)
DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_MEDIA_TETHER_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether"
    / "summary.json"
)
DEFAULT_REAL_VIDEO_ACCEPTANCE_ENVELOPE_REPORT = DEFAULT_REAL_VIDEO_ACCEPTANCE_ENVELOPE_OUT_DIR / "summary.json"
DEFAULT_REAL_VIDEO_TIMING_VARIANCE_ENVELOPE_REPORT = (
    DEFAULT_REAL_VIDEO_TIMING_VARIANCE_ENVELOPE_OUT_DIR / "summary.json"
)
DEFAULT_REAL_VIDEO_COMPILED_ADJOINT_REPLACEMENT_REPORT = (
    DEFAULT_REAL_VIDEO_COMPILED_ADJOINT_REPLACEMENT_OUT_DIR / "summary.json"
)
DEFAULT_SHARED_WORK_REPORT = DEFAULT_SHARED_WORK_OUT_DIR / "summary.json"

PROVEN_REQUIREMENT_IDS = (
    "formal_goal_contract",
    "fiber_gauge_trace_invariant",
    "clean_fiber_derivatives",
    "local_camera_family_bundle_math",
    "local_camera_family_2d_bundle_math",
    "local_camera_family_shared_metadata",
    "local_camera_family_2d_shared_metadata",
    "local_camera_family_2d_metal_slice_lowering",
    "local_camera_family_2d_metal_shared_backward",
    "local_camera_family_2d_metal_single_launch_materialized",
    "local_camera_family_2d_metal_native_family_eval",
    "local_camera_family_2d_metal_native_interval_forward",
    "local_camera_family_2d_metal_native_interval_backward",
    "local_camera_family_2d_tile_order_reuse",
    "local_camera_family_2d_tile_order_strata",
    "local_camera_family_2d_active_set_strata",
    "real_video_active_set_distribution",
    "metal_time_shared_forward_backward",
    "finite_exposure_rolling_fallback",
    "compiled_adjoint_trainer_smoke",
    "real_video_trainer_smoke",
    "real_video_guarded_support_matrix",
    "real_video_multiscene_trainer_matrix",
    "real_video_multiscene_extended_trainer_matrix",
    "real_video_multiscene_frame_scaling_matrix",
    "real_video_multiscene_extended_frame_scaling_diagnostic",
    "real_video_multiscene_quality_tether",
    "real_video_multiscene_extended_quality_tether",
    "real_video_multiscene_media_tether",
    "real_video_multiscene_extended_media_tether",
    "real_video_acceptance_envelope",
    "real_video_timing_variance_envelope",
    "real_video_compiled_adjoint_replacement",
    "sublinear_world_side_work_proxy",
)
OPEN_REQUIREMENT_ID = "full_goal_completion"


Verifier = Callable[[dict[str, Any]], list[str]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _require_close(
    left: Any,
    right: Any,
    label: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    left_value = _finite_float(left, f"{label} left", errors)
    right_value = _finite_float(right, f"{label} right", errors)
    if abs(left_value - right_value) > atol:
        errors.append(f"{label} must match across artifacts: {left_value!r} != {right_value!r}")


def _artifact(path: Path, verifier: Verifier) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(path),
        "benchmark": report.get("benchmark"),
        "status": report.get("status"),
        "verifier_errors": verifier(report),
        "summary": report.get("summary", {}),
    }


def _shared_work_artifact(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(path),
        "benchmark": report.get("benchmark"),
        "status": report.get("status"),
        "verifier_errors": verify_shared_work_goal_audit(report),
        "current_input_errors": verify_shared_work_goal_current_acceptance(report),
        "summary": report.get("summary", {}),
    }


def _requirement(
    requirement_id: str,
    status: str,
    statement: str,
    evidence: list[str],
    *,
    gaps: list[str] | None = None,
) -> dict[str, Any]:
    row = {
        "id": requirement_id,
        "status": status,
        "statement": statement,
        "evidence": evidence,
    }
    if gaps:
        row["gaps"] = gaps
    return row


def summarize(requirements: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    shared = evidence["shared_work"]["summary"]
    bundle_value = evidence["bundle_invariance"]["summary"]
    bundle_grad = evidence["bundle_gradient"]["summary"]
    camera_family = evidence["camera_family"]["summary"]
    camera_family_2d = evidence["camera_family_2d"]["summary"]
    camera_family_2d_metal_lowering = evidence["camera_family_2d_metal_lowering"]["summary"]
    camera_family_2d_metal_chain_rule = evidence["camera_family_2d_metal_chain_rule"]["summary"]
    camera_family_2d_materialized_batch = evidence["camera_family_2d_materialized_batch"]["summary"]
    camera_family_2d_native_eval = evidence["camera_family_2d_native_eval"]["summary"]
    camera_family_2d_native_interval_forward = evidence["camera_family_2d_native_interval_forward"]["summary"]
    camera_family_2d_native_interval_backward = evidence["camera_family_2d_native_interval_backward"]["summary"]
    camera_family_2d_tile_order_reuse = evidence["camera_family_2d_tile_order_reuse"]["summary"]
    camera_family_2d_tile_order_strata = evidence["camera_family_2d_tile_order_strata"]["summary"]
    camera_family_2d_active_set_strata = evidence["camera_family_2d_active_set_strata"]["summary"]
    real_active_set_distribution = evidence["real_active_set_distribution"]["summary"]
    camera_family_shared = evidence["camera_family_shared_work"]["summary"]
    camera_family_2d_shared = evidence["camera_family_2d_shared_work"]["summary"]
    trainer = evidence["trainer_interval"]["summary"]
    real_video_trainer = evidence["trainer_real_video"]["summary"]
    real_video_guarded_support = evidence["real_video_guarded_support_matrix"]["summary"]
    real_video_multiscene = evidence["real_video_multiscene_trainer_matrix"]["summary"]
    real_video_multiscene_extended = evidence["real_video_multiscene_extended_trainer_matrix"]["summary"]
    real_video_multiscene_frame_scaling = evidence["real_video_multiscene_frame_scaling_matrix"]["summary"]
    real_video_multiscene_extended_frame_scaling = evidence[
        "real_video_multiscene_extended_frame_scaling_diagnostic"
    ]["summary"]
    real_video_multiscene_quality = evidence["real_video_multiscene_quality_tether"]["summary"]
    real_video_multiscene_extended_quality = evidence["real_video_multiscene_extended_quality_tether"][
        "summary"
    ]
    real_video_multiscene_media = evidence["real_video_multiscene_media_tether"]["summary"]
    real_video_multiscene_extended_media = evidence["real_video_multiscene_extended_media_tether"]["summary"]
    real_video_acceptance_envelope = evidence["real_video_acceptance_envelope"]["summary"]
    real_video_timing_variance_envelope = evidence["real_video_timing_variance_envelope"]["summary"]
    real_video_compiled_adjoint_replacement = evidence["real_video_compiled_adjoint_replacement"]["summary"]
    status_counts = {
        "proved": sum(1 for row in requirements if row.get("status") == "proved"),
        "open": sum(1 for row in requirements if row.get("status") == "open"),
        "failed": sum(1 for row in requirements if row.get("status") == "failed"),
    }
    return {
        "overall_status": "in_progress",
        "proved_requirement_count": status_counts["proved"],
        "open_requirement_count": status_counts["open"],
        "failed_requirement_count": status_counts["failed"],
        "is_goal_complete": False,
        "max_bundle_value_rel_error": bundle_value["max_rel_error"],
        "max_bundle_gradient_rel_error": bundle_grad["max_gradient_rel_error"],
        "max_camera_family_value_rel_error": camera_family["max_value_rel_error"],
        "max_camera_family_primitive_gradient_rel_error": camera_family["max_primitive_gradient_rel_error"],
        "camera_family_q_gradient_rel_error": camera_family["q_gradient_rel_error"],
        "max_camera_family_2d_value_rel_error": camera_family_2d["max_value_rel_error"],
        "max_camera_family_2d_primitive_gradient_rel_error": camera_family_2d[
            "max_primitive_gradient_rel_error"
        ],
        "camera_family_2d_q_phase_gradient_rel_error": camera_family_2d["q_phase_gradient_rel_error"],
        "camera_family_2d_q_height_gradient_rel_error": camera_family_2d["q_height_gradient_rel_error"],
        "camera_family_2d_metal_lowering_family_payload_ratio": camera_family_2d_metal_lowering[
            "family_to_replay_payload_ratio"
        ],
        "camera_family_2d_metal_lowering_peak_slice_payload_ratio": camera_family_2d_metal_lowering[
            "peak_slice_to_replay_payload_ratio"
        ],
        "camera_family_2d_metal_lowering_min_grad_coeff_abs_sum": camera_family_2d_metal_lowering[
            "min_grad_coeff_abs_sum"
        ],
        "camera_family_2d_metal_chain_rule_shared_gradient_ratio": camera_family_2d_metal_chain_rule[
            "shared_to_replay_gradient_payload_ratio"
        ],
        "camera_family_2d_metal_chain_rule_max_fd_rel_error": camera_family_2d_metal_chain_rule[
            "max_finite_difference_rel_error"
        ],
        "camera_family_2d_metal_chain_rule_shared_grad_abs_sum": camera_family_2d_metal_chain_rule[
            "shared_family_grad_abs_sum"
        ],
        "camera_family_2d_materialized_batch_forward_launch_ratio": camera_family_2d_materialized_batch[
            "forward_launch_ratio"
        ],
        "camera_family_2d_materialized_batch_backward_launch_ratio": camera_family_2d_materialized_batch[
            "backward_launch_ratio"
        ],
        "camera_family_2d_materialized_batch_payload_ratio": camera_family_2d_materialized_batch[
            "materialized_to_replay_trace_payload_ratio"
        ],
        "camera_family_2d_materialized_batch_family_payload_ratio": camera_family_2d_materialized_batch[
            "family_to_materialized_trace_payload_ratio"
        ],
        "camera_family_2d_materialized_batch_max_image_abs_error": camera_family_2d_materialized_batch[
            "max_batched_vs_slice_image_abs_error"
        ],
        "camera_family_2d_materialized_batch_max_shared_grad_rel_error": camera_family_2d_materialized_batch[
            "max_batched_vs_slice_shared_grad_rel_error"
        ],
        "camera_family_2d_native_eval_family_payload_ratio": camera_family_2d_native_eval[
            "family_coeff_to_materialized_coeff_payload_ratio"
        ],
        "camera_family_2d_native_eval_family_plus_q_payload_ratio": camera_family_2d_native_eval[
            "family_plus_q_basis_to_materialized_coeff_payload_ratio"
        ],
        "camera_family_2d_native_eval_max_value_rel_error": camera_family_2d_native_eval[
            "native_eval_max_rel_error"
        ],
        "camera_family_2d_native_eval_max_family_grad_rel_error": camera_family_2d_native_eval[
            "native_grad_family_max_rel_error"
        ],
        "camera_family_2d_native_eval_max_q_basis_grad_rel_error": camera_family_2d_native_eval[
            "native_grad_q_basis_max_rel_error"
        ],
        "camera_family_2d_native_interval_forward_payload_ratio": camera_family_2d_native_interval_forward[
            "family_forward_to_materialized_trace_payload_ratio"
        ],
        "camera_family_2d_native_interval_forward_coeff_payload_ratio": camera_family_2d_native_interval_forward[
            "family_coeff_to_materialized_trace_payload_ratio"
        ],
        "camera_family_2d_native_interval_forward_max_image_rel_error": camera_family_2d_native_interval_forward[
            "native_family_forward_max_rel_error"
        ],
        "camera_family_2d_native_interval_backward_gradient_payload_ratio": camera_family_2d_native_interval_backward[
            "native_family_gradient_to_materialized_gradient_payload_ratio"
        ],
        "camera_family_2d_native_interval_backward_coeff_gradient_payload_ratio": camera_family_2d_native_interval_backward[
            "native_family_coeff_gradient_to_materialized_gradient_payload_ratio"
        ],
        "camera_family_2d_native_interval_backward_max_family_grad_rel_error": camera_family_2d_native_interval_backward[
            "native_family_interval_backward_max_family_grad_rel_error"
        ],
        "camera_family_2d_native_interval_backward_max_q_basis_grad_rel_error": camera_family_2d_native_interval_backward[
            "native_family_interval_backward_max_q_basis_grad_rel_error"
        ],
        "camera_family_2d_tile_order_reuse_metadata_ratio": camera_family_2d_tile_order_reuse[
            "shared_to_materialized_tile_order_metadata_ratio"
        ],
        "camera_family_2d_tile_order_reuse_materialized_growth": camera_family_2d_tile_order_reuse[
            "materialized_tile_order_metadata_growth"
        ],
        "camera_family_2d_tile_order_reuse_shared_growth": camera_family_2d_tile_order_reuse[
            "shared_tile_order_metadata_growth"
        ],
        "camera_family_2d_tile_order_reuse_min_union_depth_order_gap": camera_family_2d_tile_order_reuse[
            "min_union_depth_order_gap"
        ],
        "camera_family_2d_tile_order_strata_metadata_ratio": camera_family_2d_tile_order_strata[
            "shared_to_materialized_tile_order_metadata_ratio"
        ],
        "camera_family_2d_tile_order_strata_materialized_growth": camera_family_2d_tile_order_strata[
            "materialized_tile_order_metadata_growth"
        ],
        "camera_family_2d_tile_order_strata_shared_growth": camera_family_2d_tile_order_strata[
            "shared_tile_order_metadata_growth"
        ],
        "camera_family_2d_tile_order_strata_count": camera_family_2d_tile_order_strata[
            "order_stratum_count"
        ],
        "camera_family_2d_tile_order_strata_min_union_depth_order_gap": camera_family_2d_tile_order_strata[
            "min_stratum_union_depth_order_gap"
        ],
        "camera_family_2d_active_set_strata_metadata_ratio": camera_family_2d_active_set_strata[
            "shared_to_materialized_tile_order_metadata_ratio"
        ],
        "camera_family_2d_active_set_strata_materialized_growth": camera_family_2d_active_set_strata[
            "materialized_tile_order_metadata_growth"
        ],
        "camera_family_2d_active_set_strata_shared_growth": camera_family_2d_active_set_strata[
            "shared_tile_order_metadata_growth"
        ],
        "camera_family_2d_active_set_strata_count": camera_family_2d_active_set_strata[
            "active_set_stratum_count"
        ],
        "camera_family_2d_active_set_strata_min_union_depth_order_gap": camera_family_2d_active_set_strata[
            "min_active_set_union_depth_order_gap"
        ],
        "real_active_set_distribution_artifact_count": real_active_set_distribution["artifact_count"],
        "real_active_set_distribution_row_count": real_active_set_distribution["row_count"],
        "real_active_set_distribution_max_group_dense_ratio": real_active_set_distribution[
            "max_active_set_group_to_dense_tile_pair_ratio"
        ],
        "real_active_set_distribution_max_cells_per_group": real_active_set_distribution[
            "max_cells_per_active_set_group"
        ],
        "real_active_set_distribution_max_cell_group_ratio": real_active_set_distribution[
            "max_cell_to_active_set_group_ratio"
        ],
        "camera_family_shared_final_payload_ratio": camera_family_shared["final_payload_ratio"],
        "camera_family_shared_final_chart_ratio": camera_family_shared["final_chart_ratio"],
        "camera_family_shared_payload_growth": camera_family_shared["family_payload_growth"],
        "camera_family_replay_payload_growth": camera_family_shared["per_q_replay_payload_growth"],
        "camera_family_shared_max_fit_uv_error_px": camera_family_shared["max_family_fit_uv_error_px"],
        "camera_family_2d_shared_final_payload_ratio": camera_family_2d_shared["final_payload_ratio"],
        "camera_family_2d_shared_final_chart_ratio": camera_family_2d_shared["final_chart_ratio"],
        "camera_family_2d_shared_payload_growth": camera_family_2d_shared["family_payload_growth"],
        "camera_family_2d_replay_payload_growth": camera_family_2d_shared["per_q_replay_payload_growth"],
        "camera_family_2d_shared_max_fit_uv_error_px": camera_family_2d_shared["max_family_fit_uv_error_px"],
        "trainer_interval_max_no_first_step_ratio": max(trainer["measured_vs_cadence_no_first_step_ms_ratios"]),
        "trainer_interval_max_rebuild_ratio": max(trainer["measured_vs_cadence_rebuild_ratios"]),
        "trainer_interval_loss_match_delta": trainer["max_measured_vs_cadence_end_loss_abs_delta"],
        "trainer_real_video_max_no_first_step_ratio": max(
            real_video_trainer["measured_vs_cadence_no_first_step_ms_ratios"]
        ),
        "trainer_real_video_max_rebuild_ratio": max(real_video_trainer["measured_vs_cadence_rebuild_ratios"]),
        "trainer_real_video_loss_match_delta": real_video_trainer["max_measured_vs_cadence_end_loss_abs_delta"],
        "real_video_guarded_support_artifact_count": real_video_guarded_support["artifact_count"],
        "real_video_guarded_support_guarded_artifact_count": real_video_guarded_support["guarded_artifact_count"],
        "real_video_guarded_support_default_rebins": real_video_guarded_support[
            "default_measured_support_rebins"
        ],
        "real_video_guarded_support_guarded_rebins": real_video_guarded_support[
            "guarded_measured_support_rebins"
        ],
        "real_video_guarded_support_guarded_stale_refreshes": real_video_guarded_support[
            "guarded_measured_stale_refreshes"
        ],
        "real_video_guarded_support_max_no_first_ratio": real_video_guarded_support[
            "max_guarded_measured_no_first_ratio"
        ],
        "real_video_guarded_support_max_rebuild_ratio": real_video_guarded_support[
            "max_guarded_measured_rebuild_ratio"
        ],
        "real_video_multiscene_scene_count": real_video_multiscene["scene_count"],
        "real_video_multiscene_row_count": real_video_multiscene["row_count"],
        "real_video_multiscene_distinct_youtube_id_count": real_video_multiscene[
            "distinct_youtube_id_count"
        ],
        "real_video_multiscene_max_no_first_ratio": real_video_multiscene[
            "max_measured_vs_cadence_no_first_step_ms_ratio"
        ],
        "real_video_multiscene_max_rebuild_ratio": real_video_multiscene[
            "max_measured_vs_cadence_rebuild_ratio"
        ],
        "real_video_multiscene_max_loss_delta": real_video_multiscene[
            "max_measured_vs_cadence_end_loss_abs_delta"
        ],
        "real_video_multiscene_max_support_rebins": real_video_multiscene[
            "max_measured_support_rebins"
        ],
        "real_video_multiscene_max_stale_refreshes": real_video_multiscene[
            "max_measured_stale_refreshes"
        ],
        "real_video_multiscene_extended_scene_count": real_video_multiscene_extended["scene_count"],
        "real_video_multiscene_extended_row_count": real_video_multiscene_extended["row_count"],
        "real_video_multiscene_extended_distinct_youtube_id_count": real_video_multiscene_extended[
            "distinct_youtube_id_count"
        ],
        "real_video_multiscene_extended_max_motion_score": real_video_multiscene_extended[
            "max_motion_score"
        ],
        "real_video_multiscene_extended_max_no_first_ratio": real_video_multiscene_extended[
            "max_measured_vs_cadence_no_first_step_ms_ratio"
        ],
        "real_video_multiscene_extended_max_rebuild_ratio": real_video_multiscene_extended[
            "max_measured_vs_cadence_rebuild_ratio"
        ],
        "real_video_multiscene_extended_max_loss_delta": real_video_multiscene_extended[
            "max_measured_vs_cadence_end_loss_abs_delta"
        ],
        "real_video_multiscene_extended_max_support_rebins": real_video_multiscene_extended[
            "max_measured_support_rebins"
        ],
        "real_video_multiscene_extended_max_stale_refreshes": real_video_multiscene_extended[
            "max_measured_stale_refreshes"
        ],
        "real_video_multiscene_frame_scaling_scene_count": real_video_multiscene_frame_scaling[
            "scene_count"
        ],
        "real_video_multiscene_frame_scaling_row_count": real_video_multiscene_frame_scaling[
            "row_count"
        ],
        "real_video_multiscene_frame_scaling_frame_growth_factor": real_video_multiscene_frame_scaling[
            "frame_growth_factor"
        ],
        "real_video_multiscene_frame_scaling_max_no_first_ratio": real_video_multiscene_frame_scaling[
            "max_measured_vs_cadence_no_first_step_ms_ratio"
        ],
        "real_video_multiscene_frame_scaling_max_rebuild_ratio": real_video_multiscene_frame_scaling[
            "max_measured_vs_cadence_rebuild_ratio"
        ],
        "real_video_multiscene_frame_scaling_max_no_first_growth_ratio": real_video_multiscene_frame_scaling[
            "max_measured_no_first_growth_vs_frame_growth_ratio"
        ],
        "real_video_multiscene_frame_scaling_max_rebuild_growth": real_video_multiscene_frame_scaling[
            "max_measured_cache_rebuild_growth"
        ],
        "real_video_multiscene_frame_scaling_max_support_rebins": real_video_multiscene_frame_scaling[
            "max_measured_support_rebins"
        ],
        "real_video_multiscene_frame_scaling_max_stale_refreshes": real_video_multiscene_frame_scaling[
            "max_measured_stale_refreshes"
        ],
        "real_video_multiscene_extended_frame_scaling_scene_count": real_video_multiscene_extended_frame_scaling[
            "source_scene_count"
        ],
        "real_video_multiscene_extended_frame_scaling_row_count": real_video_multiscene_extended_frame_scaling[
            "source_row_count"
        ],
        "real_video_multiscene_extended_frame_scaling_max_no_first_ratio": real_video_multiscene_extended_frame_scaling[
            "max_measured_vs_cadence_no_first_step_ms_ratio"
        ],
        "real_video_multiscene_extended_frame_scaling_max_growth_ratio": real_video_multiscene_extended_frame_scaling[
            "max_measured_no_first_growth_vs_frame_growth_ratio"
        ],
        "real_video_multiscene_extended_frame_scaling_max_rebuild_ratio": real_video_multiscene_extended_frame_scaling[
            "max_measured_vs_cadence_rebuild_ratio"
        ],
        "real_video_multiscene_extended_frame_scaling_max_support_rebins": real_video_multiscene_extended_frame_scaling[
            "max_measured_support_rebins"
        ],
        "real_video_multiscene_extended_frame_scaling_expected_timing_failures": real_video_multiscene_extended_frame_scaling[
            "strict_failure_count"
        ],
        "real_video_multiscene_quality_pair_count": real_video_multiscene_quality["pair_count"],
        "real_video_multiscene_quality_max_loss_curve_delta": real_video_multiscene_quality[
            "max_abs_loss_curve_delta"
        ],
        "real_video_multiscene_quality_max_end_psnr_delta": real_video_multiscene_quality[
            "max_end_psnr_abs_delta"
        ],
        "real_video_multiscene_quality_min_psnr_gain": real_video_multiscene_quality[
            "min_measured_psnr_gain"
        ],
        "real_video_multiscene_quality_min_end_psnr": real_video_multiscene_quality[
            "min_measured_end_psnr"
        ],
        "real_video_multiscene_extended_quality_scene_count": real_video_multiscene_extended_quality[
            "scene_count"
        ],
        "real_video_multiscene_extended_quality_pair_count": real_video_multiscene_extended_quality[
            "pair_count"
        ],
        "real_video_multiscene_extended_quality_distinct_source_count": real_video_multiscene_extended_quality[
            "source_distinct_youtube_id_count"
        ],
        "real_video_multiscene_extended_quality_max_loss_curve_delta": real_video_multiscene_extended_quality[
            "max_abs_loss_curve_delta"
        ],
        "real_video_multiscene_extended_quality_max_end_psnr_delta": real_video_multiscene_extended_quality[
            "max_end_psnr_abs_delta"
        ],
        "real_video_multiscene_extended_quality_min_psnr_gain": real_video_multiscene_extended_quality[
            "min_measured_psnr_gain"
        ],
        "real_video_multiscene_media_pair_count": real_video_multiscene_media["pair_count"],
        "real_video_multiscene_media_max_contact_sheet_delta": real_video_multiscene_media[
            "max_abs_contact_sheet_delta"
        ],
        "real_video_multiscene_media_max_contact_sheet_target_pred_mse_delta": real_video_multiscene_media[
            "max_contact_sheet_target_pred_mse_delta"
        ],
        "real_video_multiscene_media_max_contact_sheet_payload_loss_delta": real_video_multiscene_media[
            "max_contact_sheet_payload_loss_abs_delta"
        ],
        "real_video_multiscene_media_min_contact_sheet_target_std": real_video_multiscene_media[
            "min_contact_sheet_target_std"
        ],
        "real_video_multiscene_media_min_contact_sheet_pred_std": real_video_multiscene_media[
            "min_contact_sheet_pred_std"
        ],
        "real_video_multiscene_media_max_loss_curve_delta": real_video_multiscene_media[
            "max_abs_loss_curve_delta"
        ],
        "real_video_multiscene_media_max_final_rgb_loss_delta": real_video_multiscene_media[
            "max_final_full_rgb_loss_abs_delta"
        ],
        "real_video_multiscene_media_max_no_first_ratio": real_video_multiscene_media[
            "max_measured_vs_cadence_no_first_step_ms_ratio"
        ],
        "real_video_multiscene_media_max_rebuild_ratio": real_video_multiscene_media[
            "max_measured_vs_cadence_rebuild_ratio"
        ],
        "real_video_multiscene_extended_media_pair_count": real_video_multiscene_extended_media[
            "pair_count"
        ],
        "real_video_multiscene_extended_media_max_contact_sheet_delta": real_video_multiscene_extended_media[
            "max_abs_contact_sheet_delta"
        ],
        "real_video_multiscene_extended_media_max_contact_sheet_payload_loss_delta": real_video_multiscene_extended_media[
            "max_contact_sheet_payload_loss_abs_delta"
        ],
        "real_video_multiscene_extended_media_min_contact_sheet_target_std": real_video_multiscene_extended_media[
            "min_contact_sheet_target_std"
        ],
        "real_video_multiscene_extended_media_min_contact_sheet_pred_std": real_video_multiscene_extended_media[
            "min_contact_sheet_pred_std"
        ],
        "real_video_multiscene_extended_media_max_loss_curve_delta": real_video_multiscene_extended_media[
            "max_abs_loss_curve_delta"
        ],
        "real_video_multiscene_extended_media_max_final_rgb_loss_delta": real_video_multiscene_extended_media[
            "max_final_full_rgb_loss_abs_delta"
        ],
        "real_video_multiscene_extended_media_max_no_first_ratio": real_video_multiscene_extended_media[
            "max_measured_vs_cadence_no_first_step_ms_ratio"
        ],
        "real_video_multiscene_extended_media_max_rebuild_ratio": real_video_multiscene_extended_media[
            "max_measured_vs_cadence_rebuild_ratio"
        ],
        "real_video_acceptance_envelope_functional_scene_count": real_video_acceptance_envelope[
            "functional_scene_count"
        ],
        "real_video_acceptance_envelope_media_scene_count": real_video_acceptance_envelope[
            "media_scene_count"
        ],
        "real_video_acceptance_envelope_broad10_media_distinct_youtube_id_count": real_video_acceptance_envelope[
            "broad10_media_distinct_youtube_id_count"
        ],
        "real_video_acceptance_envelope_broad_media_distinct_youtube_id_count": real_video_acceptance_envelope[
            "broad_media_distinct_youtube_id_count"
        ],
        "real_video_acceptance_envelope_broad10_quality_distinct_youtube_id_count": real_video_acceptance_envelope[
            "broad10_quality_distinct_youtube_id_count"
        ],
        "real_video_acceptance_envelope_broad_quality_distinct_youtube_id_count": real_video_acceptance_envelope[
            "broad_quality_distinct_youtube_id_count"
        ],
        "real_video_acceptance_envelope_all_underlying_verifiers_pass": real_video_acceptance_envelope[
            "all_underlying_verifiers_pass"
        ],
        "real_video_acceptance_envelope_all_functional_rows_pass": real_video_acceptance_envelope[
            "all_functional_rows_pass"
        ],
        "real_video_acceptance_envelope_all_quality_tethers_match": real_video_acceptance_envelope[
            "all_quality_tethers_match"
        ],
        "real_video_acceptance_envelope_all_media_tethers_match": real_video_acceptance_envelope[
            "all_media_tethers_match"
        ],
        "real_video_acceptance_envelope_max_support_rebins": real_video_acceptance_envelope[
            "max_support_rebins"
        ],
        "real_video_acceptance_envelope_max_rebuild_ratio": real_video_acceptance_envelope[
            "max_rebuild_ratio"
        ],
        "real_video_acceptance_envelope_min_quality_psnr_gain": real_video_acceptance_envelope[
            "min_quality_psnr_gain"
        ],
        "real_video_acceptance_envelope_expected_timing_failures": real_video_acceptance_envelope[
            "extended_frame_scaling_expected_timing_failure_count"
        ],
        "real_video_acceptance_envelope_max_extended_timing_growth_overage": real_video_acceptance_envelope[
            "max_extended_timing_growth_overage"
        ],
        "real_video_acceptance_envelope_bq4_fresh_process_pair_count": real_video_acceptance_envelope[
            "bq4_fresh_process_pair_count"
        ],
        "real_video_acceptance_envelope_bq4_fresh_process_acceptance_status": real_video_acceptance_envelope[
            "bq4_fresh_process_timing_acceptance_status"
        ],
        "real_video_acceptance_envelope_bq4_fresh_process_post_warmup_median_no_first_ratio": real_video_acceptance_envelope[
            "bq4_fresh_process_post_warmup_median_no_first_ratio"
        ],
        "real_video_acceptance_envelope_bq4_fresh_process_post_warmup_median_projective_total_ratio": real_video_acceptance_envelope[
            "bq4_fresh_process_post_warmup_median_projective_total_ratio"
        ],
        "real_video_acceptance_envelope_bq4_fresh_process_post_warmup_median_feature_state_update_ratio": real_video_acceptance_envelope[
            "bq4_fresh_process_post_warmup_median_feature_state_update_ratio"
        ],
        "real_video_acceptance_envelope_bq4_fresh_process_max_projective_total_ratio": real_video_acceptance_envelope[
            "bq4_fresh_process_max_projective_total_ratio"
        ],
        "real_video_acceptance_envelope_bq4_fresh_process_no_first_bump_count": real_video_acceptance_envelope[
            "bq4_fresh_process_no_first_bump_count"
        ],
        "real_video_acceptance_envelope_strict_timing_win_claimed": real_video_acceptance_envelope[
            "strict_timing_win_claimed"
        ],
        "real_video_acceptance_envelope_fresh_process_median_timing_win_claimed": real_video_acceptance_envelope[
            "fresh_process_median_timing_win_claimed"
        ],
        "real_video_acceptance_envelope_does_not_prove_completion": real_video_acceptance_envelope[
            "does_not_prove_completion"
        ],
        "real_video_timing_variance_source_scene_count": real_video_timing_variance_envelope[
            "source_scene_count"
        ],
        "real_video_timing_variance_strict_failure_count": real_video_timing_variance_envelope[
            "strict_failure_count"
        ],
        "real_video_timing_variance_all_cache_support_clean": real_video_timing_variance_envelope[
            "all_cache_support_clean"
        ],
        "real_video_timing_variance_workload_explains_miss_count": real_video_timing_variance_envelope[
            "workload_explains_render_forward_miss_count"
        ],
        "real_video_timing_variance_drop_spike_render_forward_ratio": real_video_timing_variance_envelope[
            "drop_spike_render_forward_ratio"
        ],
        "real_video_timing_variance_fresh_process_status": real_video_timing_variance_envelope[
            "fresh_process_timing_acceptance_status"
        ],
        "real_video_timing_variance_fresh_process_median_no_first_ratio": real_video_timing_variance_envelope[
            "fresh_process_median_no_first_ratio"
        ],
        "real_video_timing_variance_fresh_process_median_projective_total_ratio": real_video_timing_variance_envelope[
            "fresh_process_median_projective_total_ratio"
        ],
        "real_video_timing_variance_fresh_process_median_feature_state_update_ratio": real_video_timing_variance_envelope[
            "fresh_process_median_feature_state_update_ratio"
        ],
        "real_video_timing_variance_strict_timing_win_claimed": real_video_timing_variance_envelope[
            "strict_timing_win_claimed"
        ],
        "real_video_timing_variance_does_not_prove_completion": real_video_timing_variance_envelope[
            "does_not_prove_completion"
        ],
        "real_video_compiled_adjoint_replacement_accepted": real_video_compiled_adjoint_replacement[
            "final_compiled_adjoint_replacement_accepted"
        ],
        "real_video_compiled_adjoint_replacement_gap": real_video_compiled_adjoint_replacement[
            "compiled_trainer_replacement_gap"
        ],
        "real_video_compiled_adjoint_case_payload_count": real_video_compiled_adjoint_replacement[
            "case_payload_count"
        ],
        "real_video_compiled_adjoint_source_contract_checks_pass": real_video_compiled_adjoint_replacement[
            "source_contract_checks_pass"
        ],
        "real_video_compiled_adjoint_all_cases_projective_interval_main_path": real_video_compiled_adjoint_replacement[
            "all_cases_projective_interval_main_path"
        ],
        "orbit_payload_growth_ratio": shared["orbit_payload_growth_ratio"],
        "orbit_final_payload_ratio": shared["orbit_final_payload_ratio"],
        "orbit_final_trace_ratio": shared["orbit_final_trace_ratio"],
        "max_trained_final_interval_entry_ratio": shared["max_trained_final_interval_entry_ratio"],
        "max_trained_final_trace_count_ratio": shared["max_trained_final_trace_count_ratio"],
        "trained_shared_to_replay_interval_growth_ratio": shared[
            "trained_shared_to_replay_interval_growth_ratio"
        ],
        "max_trained_final_forward_ms_ratio": shared["max_trained_final_forward_ms_ratio"],
        "max_trained_final_backward_ms_ratio": shared["max_trained_final_backward_ms_ratio"],
        "exposure_forward_max_metal_abs_error": shared["exposure_forward_max_metal_abs_error"],
        "exposure_backward_max_metal_grad_rel_error": shared["exposure_backward_max_metal_grad_rel_error"],
        "exposure_mixed_fallback_max_grad_rel_error": shared["exposure_mixed_fallback_max_grad_rel_error"],
    }


def run_report(
    *,
    bundle_invariance_report: Path = DEFAULT_BUNDLE_INVARIANCE_REPORT,
    bundle_gradient_report: Path = DEFAULT_BUNDLE_GRADIENT_REPORT,
    camera_family_report: Path = DEFAULT_CAMERA_FAMILY_REPORT,
    camera_family_2d_report: Path = DEFAULT_CAMERA_FAMILY_2D_REPORT,
    camera_family_2d_metal_lowering_report: Path = DEFAULT_CAMERA_FAMILY_2D_METAL_LOWERING_REPORT,
    camera_family_2d_metal_chain_rule_report: Path = DEFAULT_CAMERA_FAMILY_2D_METAL_CHAIN_RULE_REPORT,
    camera_family_2d_materialized_batch_report: Path = DEFAULT_CAMERA_FAMILY_2D_MATERIALIZED_BATCH_REPORT,
    camera_family_2d_native_eval_report: Path = DEFAULT_CAMERA_FAMILY_2D_NATIVE_EVAL_REPORT,
    camera_family_2d_native_interval_forward_report: Path = DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_FORWARD_REPORT,
    camera_family_2d_native_interval_backward_report: Path = DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_BACKWARD_REPORT,
    camera_family_2d_tile_order_reuse_report: Path = DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_REUSE_REPORT,
    camera_family_2d_tile_order_strata_report: Path = DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_STRATA_REPORT,
    camera_family_2d_active_set_strata_report: Path = DEFAULT_CAMERA_FAMILY_2D_ACTIVE_SET_STRATA_REPORT,
    real_active_set_distribution_report: Path = DEFAULT_REAL_ACTIVE_SET_DISTRIBUTION_REPORT,
    camera_family_shared_work_report: Path = DEFAULT_CAMERA_FAMILY_SHARED_WORK_REPORT,
    camera_family_2d_shared_work_report: Path = DEFAULT_CAMERA_FAMILY_2D_SHARED_WORK_REPORT,
    trainer_interval_report: Path = DEFAULT_INTERVAL_TRAINER_REPORT,
    trainer_real_video_report: Path = DEFAULT_REAL_VIDEO_TRAINER_REPORT,
    real_video_guarded_support_matrix_report: Path = DEFAULT_REAL_VIDEO_GUARDED_SUPPORT_MATRIX_REPORT,
    real_video_multiscene_trainer_matrix_report: Path = DEFAULT_REAL_VIDEO_MULTISCENE_TRAINER_MATRIX_REPORT,
    real_video_multiscene_extended_trainer_matrix_report: Path = (
        DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_TRAINER_MATRIX_REPORT
    ),
    real_video_multiscene_frame_scaling_matrix_report: Path = DEFAULT_REAL_VIDEO_MULTISCENE_FRAME_SCALING_MATRIX_REPORT,
    real_video_multiscene_extended_frame_scaling_diagnostic_report: Path = (
        DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_FRAME_SCALING_DIAGNOSTIC_REPORT
    ),
    real_video_multiscene_quality_tether_report: Path = DEFAULT_REAL_VIDEO_MULTISCENE_QUALITY_TETHER_REPORT,
    real_video_multiscene_extended_quality_tether_report: Path = (
        DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_QUALITY_TETHER_REPORT
    ),
    real_video_multiscene_media_tether_report: Path = DEFAULT_REAL_VIDEO_MULTISCENE_MEDIA_TETHER_REPORT,
    real_video_multiscene_extended_media_tether_report: Path = (
        DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_MEDIA_TETHER_REPORT
    ),
    real_video_acceptance_envelope_report: Path = DEFAULT_REAL_VIDEO_ACCEPTANCE_ENVELOPE_REPORT,
    real_video_timing_variance_envelope_report: Path = DEFAULT_REAL_VIDEO_TIMING_VARIANCE_ENVELOPE_REPORT,
    real_video_compiled_adjoint_replacement_report: Path = (
        DEFAULT_REAL_VIDEO_COMPILED_ADJOINT_REPLACEMENT_REPORT
    ),
    shared_work_report: Path = DEFAULT_SHARED_WORK_REPORT,
) -> dict[str, Any]:
    evidence = {
        "bundle_invariance": _artifact(bundle_invariance_report, verify_bundle_gauge_invariance_report),
        "bundle_gradient": _artifact(bundle_gradient_report, verify_bundle_gauge_gradient_report),
        "camera_family": _artifact(camera_family_report, verify_camera_family_gauge_report),
        "camera_family_2d": _artifact(camera_family_2d_report, verify_camera_family_2d_gauge_report),
        "camera_family_2d_metal_lowering": _artifact(
            camera_family_2d_metal_lowering_report,
            verify_camera_family_2d_metal_lowering_report,
        ),
        "camera_family_2d_metal_chain_rule": _artifact(
            camera_family_2d_metal_chain_rule_report,
            verify_camera_family_2d_metal_chain_rule_report,
        ),
        "camera_family_2d_materialized_batch": _artifact(
            camera_family_2d_materialized_batch_report,
            verify_camera_family_2d_materialized_batch_report,
        ),
        "camera_family_2d_native_eval": _artifact(
            camera_family_2d_native_eval_report,
            verify_camera_family_2d_native_eval_report,
        ),
        "camera_family_2d_native_interval_forward": _artifact(
            camera_family_2d_native_interval_forward_report,
            verify_camera_family_2d_native_interval_forward_report,
        ),
        "camera_family_2d_native_interval_backward": _artifact(
            camera_family_2d_native_interval_backward_report,
            verify_camera_family_2d_native_interval_backward_report,
        ),
        "camera_family_2d_tile_order_reuse": _artifact(
            camera_family_2d_tile_order_reuse_report,
            verify_camera_family_2d_tile_order_reuse_report,
        ),
        "camera_family_2d_tile_order_strata": _artifact(
            camera_family_2d_tile_order_strata_report,
            verify_camera_family_2d_tile_order_strata_report,
        ),
        "camera_family_2d_active_set_strata": _artifact(
            camera_family_2d_active_set_strata_report,
            verify_camera_family_2d_active_set_strata_report,
        ),
        "real_active_set_distribution": _artifact(
            real_active_set_distribution_report,
            verify_real_active_set_distribution_report,
        ),
        "camera_family_shared_work": _artifact(
            camera_family_shared_work_report,
            verify_camera_family_shared_work_scaling_report,
        ),
        "camera_family_2d_shared_work": _artifact(
            camera_family_2d_shared_work_report,
            verify_camera_family_2d_shared_work_scaling_report,
        ),
        "trainer_interval": _artifact(trainer_interval_report, verify_interval_trainer_frame_scaling_report),
        "trainer_real_video": _artifact(trainer_real_video_report, verify_real_video_trainer_frame_scaling_report),
        "real_video_guarded_support_matrix": _artifact(
            real_video_guarded_support_matrix_report,
            verify_real_video_guarded_support_matrix_report,
        ),
        "real_video_multiscene_trainer_matrix": _artifact(
            real_video_multiscene_trainer_matrix_report,
            verify_real_video_multiscene_trainer_matrix_report,
        ),
        "real_video_multiscene_extended_trainer_matrix": _artifact(
            real_video_multiscene_extended_trainer_matrix_report,
            verify_real_video_multiscene_trainer_matrix_report,
        ),
        "real_video_multiscene_frame_scaling_matrix": _artifact(
            real_video_multiscene_frame_scaling_matrix_report,
            verify_real_video_multiscene_frame_scaling_matrix_report,
        ),
        "real_video_multiscene_extended_frame_scaling_diagnostic": _artifact(
            real_video_multiscene_extended_frame_scaling_diagnostic_report,
            verify_extended_frame_scaling_diagnostic_report,
        ),
        "real_video_multiscene_quality_tether": _artifact(
            real_video_multiscene_quality_tether_report,
            verify_real_video_multiscene_quality_tether_report,
        ),
        "real_video_multiscene_extended_quality_tether": _artifact(
            real_video_multiscene_extended_quality_tether_report,
            verify_real_video_multiscene_extended_quality_tether_report,
        ),
        "real_video_multiscene_media_tether": _artifact(
            real_video_multiscene_media_tether_report,
            verify_real_video_multiscene_media_tether_report,
        ),
        "real_video_multiscene_extended_media_tether": _artifact(
            real_video_multiscene_extended_media_tether_report,
            verify_real_video_multiscene_media_tether_report,
        ),
        "real_video_acceptance_envelope": _artifact(
            real_video_acceptance_envelope_report,
            verify_real_video_acceptance_envelope_report,
        ),
        "real_video_timing_variance_envelope": _artifact(
            real_video_timing_variance_envelope_report,
            verify_real_video_timing_variance_envelope_report,
        ),
        "real_video_compiled_adjoint_replacement": _artifact(
            real_video_compiled_adjoint_replacement_report,
            verify_real_video_compiled_adjoint_replacement_report,
        ),
        "shared_work": _shared_work_artifact(shared_work_report),
    }
    requirements = [
        _requirement(
            "formal_goal_contract",
            "proved",
            "The active objective is represented as a camera-path compiler for spacetime primitives, not as frame/video caching.",
            ["shared_work.theory_contract", "GOAL_META_KEY_MATH memory contract"],
        ),
        _requirement(
            "fiber_gauge_trace_invariant",
            "proved",
            "The UVT trace is the fiber pushforward pi_* Gamma^* world_primitive and is invariant under monotone screen-fiber gauge changes with the correct measure Jacobian.",
            ["bundle_invariance"],
        ),
        _requirement(
            "clean_fiber_derivatives",
            "proved",
            "Primitive gradients through the fiber-gauged trace agree across depth and log-depth gauges and pass finite-difference control.",
            ["bundle_gradient"],
        ),
        _requirement(
            "local_camera_family_bundle_math",
            "proved",
            "The fiber-gauge value and derivative invariant extends from one camera path to a one-parameter local camera family over Q x Omega x T.",
            ["camera_family"],
        ),
        _requirement(
            "local_camera_family_2d_bundle_math",
            "proved",
            "The fiber-gauge value and derivative invariant extends to a two-parameter local camera family over Q2 x Omega x T, including both q_phase and q_height derivatives.",
            ["camera_family_2d"],
        ),
        _requirement(
            "local_camera_family_shared_metadata",
            "proved",
            "A one-parameter local Q x T chart stores projection metadata once while per-q replay grows linearly in q samples.",
            ["camera_family_shared_work"],
        ),
        _requirement(
            "local_camera_family_2d_shared_metadata",
            "proved",
            "A two-parameter local Q2 x T chart stores projection metadata once while per-q-pair replay grows with sampled camera-family pairs.",
            ["camera_family_2d_shared_work"],
        ),
        _requirement(
            "local_camera_family_2d_metal_slice_lowering",
            "proved",
            "A two-parameter local Q2 camera-family trace chart can be sliced into the existing Omega x T interval Metal forward/backward path with constant peak slice payload.",
            ["camera_family_2d_metal_lowering"],
        ),
        _requirement(
            "local_camera_family_2d_metal_shared_backward",
            "proved",
            "Per-slice interval Metal VJPs over a two-parameter Q2 camera-family grid accumulate into one shared Q2 family adjoint with finite-difference-checked chain rule.",
            ["camera_family_2d_metal_chain_rule"],
        ),
        _requirement(
            "local_camera_family_2d_metal_single_launch_materialized",
            "proved",
            "All sampled Q2 camera-family slices can be materialized into one interval Metal launch for forward/backward, matching per-slice images and shared gradients while intentionally retaining per-q trace payload.",
            ["camera_family_2d_materialized_batch"],
        ),
        _requirement(
            "local_camera_family_2d_metal_native_family_eval",
            "proved",
            "The Metal shader can evaluate Q2 family trace coefficients and VJPs directly from shared family coefficients plus q-basis values, before full renderer integration.",
            ["camera_family_2d_native_eval"],
        ),
        _requirement(
            "local_camera_family_2d_metal_native_interval_forward",
            "proved",
            "The interval Metal renderer can composite and visibility-sort Q2 family traces directly from shared family coefficients plus q-basis values on the forward path, without materializing per-q coefficient traces.",
            ["camera_family_2d_native_interval_forward"],
        ),
        _requirement(
            "local_camera_family_2d_metal_native_interval_backward",
            "proved",
            "The interval Metal renderer can run the Q2 family-trace VJP directly into shared family coefficients and q-basis values, with compiled visibility/order held fixed.",
            ["camera_family_2d_native_interval_backward"],
        ),
        _requirement(
            "local_camera_family_2d_tile_order_reuse",
            "proved",
            "When Q2 tile membership and depth order are stable, one shared tile/order topology plus q-index applicability replaces one materialized tile/order cell per sampled q-pair.",
            ["camera_family_2d_tile_order_reuse"],
        ),
        _requirement(
            "local_camera_family_2d_tile_order_strata",
            "proved",
            "When Q2 depth order changes across camera-family coordinates, sampled q-pairs compress into a small set of certified tile/order strata instead of one topology record per q-pair.",
            ["camera_family_2d_tile_order_strata"],
        ),
        _requirement(
            "local_camera_family_2d_active_set_strata",
            "proved",
            "When Q2 support/culling changes the active primitive set across camera-family coordinates, sampled q-pairs compress into a small set of certified active-set topology strata instead of one record per q-pair.",
            ["camera_family_2d_active_set_strata"],
        ),
        _requirement(
            "real_video_active_set_distribution",
            "proved",
            "Checked-in high-motion real-video projective interval atlases expose bounded, fallback-free active-set topology distributions rather than relying only on synthetic q-family active-set strata.",
            ["real_active_set_distribution"],
        ),
        _requirement(
            "metal_time_shared_forward_backward",
            "proved",
            "A Metal-backed projective interval renderer/VJP shares compiled tile-time work across many frames on orbit and trained high-motion artifacts.",
            ["shared_work.orbit", "shared_work.trained", "shared_work.exposure_backward"],
        ),
        _requirement(
            "finite_exposure_rolling_fallback",
            "proved",
            "Finite-exposure and rolling-shutter evaluation/backward are lowered through shared sensor-time samples, with differentiable fallback on visibility-ambiguous strata.",
            [
                "shared_work.exposure_quadrature",
                "shared_work.exposure_backward",
                "shared_work.exposure_mixed_fallback_backward",
            ],
        ),
        _requirement(
            "compiled_adjoint_trainer_smoke",
            "proved",
            "The actual projective-interval run_training route reuses a live compiled atlas across frame counts, matches cadence losses, keeps interval Metal VJP active, and lowers cache rebuilds.",
            ["trainer_interval"],
        ),
        _requirement(
            "real_video_trainer_smoke",
            "proved",
            "The high-motion real-video projective-interval run_training route preserves the trainer-smoke contract on checked-in video frames.",
            ["trainer_real_video"],
        ),
        _requirement(
            "real_video_guarded_support_matrix",
            "proved",
            "The high-motion real-video trainer route has a guarded-support matrix where slack-budgeted guard certificates eliminate measured support rebins and stale refreshes while preserving live-cache reuse.",
            ["real_video_guarded_support_matrix"],
        ),
        _requirement(
            "real_video_multiscene_trainer_matrix",
            "proved",
            "The guarded projective-interval trainer contract holds across a small source-distinct real-video matrix, matching cadence losses while cutting rebuilds and keeping support rebins, stale refreshes, overflow, fallback, and visibility stratifications at zero.",
            ["real_video_multiscene_trainer_matrix"],
        ),
        _requirement(
            "real_video_multiscene_extended_trainer_matrix",
            "proved",
            "The guarded projective-interval trainer contract functionally extends to a five-source real-video matrix including higher-motion bike and FPV clips, with cadence-loss agreement, lower rebuild count, zero support churn, and zero fallback/overflow/visibility stratification.",
            ["real_video_multiscene_extended_trainer_matrix"],
        ),
        _requirement(
            "real_video_multiscene_frame_scaling_matrix",
            "proved",
            "The guarded projective-interval trainer contract holds across source-distinct real-video frame growth, with measured cache rebuilds flat from 4 to 16 frames and measured timing growth below frame growth while preserving cadence losses and zero support churn.",
            ["real_video_multiscene_frame_scaling_matrix"],
        ),
        _requirement(
            "real_video_multiscene_extended_frame_scaling_diagnostic",
            "proved",
            "A five-source real-video frame-growth diagnostic preserves cadence loss, flat rebuild count, zero support churn, fallback-free rows, and exact expected strict timing failures, without claiming a timing win.",
            ["real_video_multiscene_extended_frame_scaling_diagnostic"],
        ),
        _requirement(
            "real_video_multiscene_quality_tether",
            "proved",
            "The measured live-cache route is quality-tethered to the cadence full-rebuild route across the source-distinct frame-scaling cases, with matching loss curves, matching end PSNR, positive PSNR gains, and gradient-flow flags preserved.",
            ["real_video_multiscene_quality_tether"],
        ),
        _requirement(
            "real_video_multiscene_extended_quality_tether",
            "proved",
            "The measured live-cache route is quality-tethered to the cadence full-rebuild route across the five-source extended trainer matrix, with matching loss curves, matching end PSNR, positive PSNR gains, and gradient-flow flags preserved.",
            ["real_video_multiscene_extended_quality_tether"],
        ),
        _requirement(
            "real_video_multiscene_media_tether",
            "proved",
            "The measured live-cache route is media-tethered to the cadence full-rebuild route across source-distinct real-video clips through the actual contact-sheet writer, with pixel-identical contact sheets, matching final full-RGB media loss, and preserved gradient-flow flags.",
            ["real_video_multiscene_media_tether"],
        ),
        _requirement(
            "real_video_multiscene_extended_media_tether",
            "proved",
            "The measured live-cache route is media-tethered to the cadence full-rebuild route across the five-source extended real-video matrix through the actual contact-sheet writer, with pixel-identical contact sheets, matching final full-RGB media loss, and preserved gradient-flow flags.",
            ["real_video_multiscene_extended_media_tether"],
        ),
        _requirement(
            "real_video_acceptance_envelope",
            "proved",
            "The focused real-video acceptance envelope consolidates source-distinct functional trainer rows, frame scaling, five-source timing diagnostics, cadence quality tethers, broad10 quality/media tethers, real media tethers, and the Bq4 fresh-process median timing gate while explicitly preserving the non-completion scope.",
            ["real_video_acceptance_envelope"],
        ),
        _requirement(
            "real_video_timing_variance_envelope",
            "proved",
            "The focused real-video timing-variance envelope consolidates strict timing failures, render-forward diagnostics, Bq4 traced reruns, warm-state caveats, and fresh-process median acceptance while separating cache/support correctness from MPS process variance.",
            ["real_video_timing_variance_envelope"],
        ),
        _requirement(
            "real_video_compiled_adjoint_replacement",
            "proved",
            "The practical real-video trainer replacement uses the projective interval route with interval Metal forward and interval Metal direct VJP, preserving gradients, cache reuse, and clean support/fallback behavior across the broad10 trainer payloads.",
            ["real_video_compiled_adjoint_replacement"],
        ),
        _requirement(
            "sublinear_world_side_work_proxy",
            "proved",
            "World-side payload, trace, interval-entry, and backward/forward ratios are sublinear versus per-frame replay on the saved orbit and trained high-motion checks.",
            ["shared_work.summary"],
        ),
        _requirement(
            OPEN_REQUIREMENT_ID,
            "open",
            "The original all-night goal is not yet fully complete across broad real-scene renderer acceptance.",
            ["this_audit"],
            gaps=[
                "The current proof is a set of focused artifacts, checked-in high-motion probes, source-distinct and broad10 real-video matrices/tethers, frame-count breadth, fresh-process median timing acceptance, the compiled-adjoint replacement artifact, and shared-work audits. A final completion audit still has to decide whether that evidence is sufficient for the active all-night goal.",
                "The Q2 camera-family Metal evidence now includes slice lowering, shared-backward accumulation, one single-launch materialized batch, native family trace eval/VJP, native family interval forward compositing, native family interval backward/VJP, stable-topology tile/order reuse, a two-strata order-change metadata certificate, a three-strata active-set metadata certificate, checked-in high-motion active-set distribution evidence, real-video guarded-support/multiscene matrices, broad10 quality/media tethers, timing-protocol acceptance, and the compiled-adjoint replacement artifact. Remaining work is the top-level completion decision, not a known missing local math/kernel row.",
            ],
        ),
    ]
    report = {
        "status": "in_progress",
        "benchmark": "star_uvt_projective_goal_progress_audit",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "key_math": "UVT trace = pi_* Gamma^* world_primitive",
        "theory": "STAR UVT is one local gauge expression of a camera-ray bundle atlas",
        "evidence": evidence,
        "requirements": requirements,
        "summary": summarize(requirements, evidence),
    }
    errors = verify_projective_goal_progress_audit(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _assert_summary_close(summary: dict[str, Any], expected: dict[str, Any], key: str, errors: list[str]) -> None:
    actual = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    elif actual != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")


def _compare_current_value(
    saved: Any,
    current: Any,
    label: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    if isinstance(current, dict):
        if not isinstance(saved, dict):
            errors.append(f"saved goal-progress report differs from current inputs at {label}: expected object")
            return
        for key, current_value in current.items():
            child_label = f"{label}.{key}" if label else str(key)
            _compare_current_value(saved.get(key), current_value, child_label, errors, atol=atol)
        return
    if isinstance(current, list):
        if not isinstance(saved, list) or len(saved) != len(current):
            errors.append(
                f"saved goal-progress report differs from current inputs at {label}: "
                f"expected list length {len(current)}, got {len(saved) if isinstance(saved, list) else type(saved).__name__}"
            )
            return
        for idx, (saved_value, current_value) in enumerate(zip(saved, current, strict=True)):
            _compare_current_value(saved_value, current_value, f"{label}[{idx}]", errors, atol=atol)
        return
    if isinstance(current, float):
        if not isinstance(saved, int | float) or abs(float(saved) - current) > atol:
            errors.append(
                f"saved goal-progress report differs from current inputs at {label}: "
                f"expected {current!r}, got {saved!r}"
            )
        return
    if saved != current:
        errors.append(
            f"saved goal-progress report differs from current inputs at {label}: "
            f"expected {current!r}, got {saved!r}"
        )


def verify_projective_goal_progress_audit(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "in_progress":
        errors.append(f"status must remain in_progress until the full goal is proven, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_goal_progress_audit":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    for key, phrase in (
        ("goal", "fast 2D rasters"),
        ("meta_goal", "share projection"),
        ("key_math", "pi_* Gamma^*"),
        ("theory", "camera-ray bundle atlas"),
    ):
        if not isinstance(report.get(key), str) or phrase not in report[key]:
            errors.append(f"{key} must preserve the active memory phrase {phrase!r}")

    evidence = report.get("evidence")
    requirements = report.get("requirements")
    summary = report.get("summary")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
        return errors
    if not isinstance(requirements, list):
        errors.append("requirements must be a list")
        return errors
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors

    for name in (
        "bundle_invariance",
        "bundle_gradient",
        "camera_family",
        "camera_family_2d",
        "camera_family_2d_metal_lowering",
        "camera_family_2d_metal_chain_rule",
        "camera_family_2d_materialized_batch",
        "camera_family_2d_native_eval",
        "camera_family_2d_native_interval_forward",
        "camera_family_2d_native_interval_backward",
        "camera_family_2d_tile_order_reuse",
        "camera_family_2d_tile_order_strata",
        "camera_family_2d_active_set_strata",
        "real_active_set_distribution",
        "camera_family_shared_work",
        "camera_family_2d_shared_work",
        "trainer_interval",
        "trainer_real_video",
        "real_video_guarded_support_matrix",
        "real_video_multiscene_trainer_matrix",
        "real_video_multiscene_extended_trainer_matrix",
        "real_video_multiscene_frame_scaling_matrix",
        "real_video_multiscene_extended_frame_scaling_diagnostic",
        "real_video_multiscene_quality_tether",
        "real_video_multiscene_extended_quality_tether",
        "real_video_multiscene_media_tether",
        "real_video_multiscene_extended_media_tether",
        "real_video_acceptance_envelope",
        "real_video_timing_variance_envelope",
        "real_video_compiled_adjoint_replacement",
        "shared_work",
    ):
        row = evidence.get(name)
        if not isinstance(row, dict):
            errors.append(f"evidence {name} must be an object")
            continue
        if not isinstance(row.get("path"), str) or not row["path"]:
            errors.append(f"evidence {name} path must be nonempty")
        if row.get("status") != "ok":
            errors.append(f"evidence {name} status must be ok, got {row.get('status')!r}")
        if row.get("verifier_errors"):
            errors.append(f"evidence {name} verifier failed: {row.get('verifier_errors')}")
        if name == "shared_work":
            current_input_errors = row.get("current_input_errors")
            if not isinstance(current_input_errors, list):
                errors.append("evidence shared_work current_input_errors must be an empty list")
            elif current_input_errors:
                errors.append(f"evidence shared_work current-input acceptance failed: {current_input_errors}")
        if not isinstance(row.get("summary"), dict):
            errors.append(f"evidence {name} summary must be an object")

    by_id: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(requirements):
        if not isinstance(row, dict):
            errors.append(f"requirement {idx} must be an object")
            continue
        requirement_id = row.get("id")
        if not isinstance(requirement_id, str) or not requirement_id:
            errors.append(f"requirement {idx} id must be nonempty")
            continue
        if requirement_id in by_id:
            errors.append(f"duplicate requirement id {requirement_id!r}")
        by_id[requirement_id] = row
        if row.get("status") not in {"proved", "open", "failed"}:
            errors.append(f"requirement {requirement_id} has invalid status {row.get('status')!r}")
        if not isinstance(row.get("statement"), str) or not row["statement"]:
            errors.append(f"requirement {requirement_id} statement must be nonempty")
        if not isinstance(row.get("evidence"), list) or not row["evidence"]:
            errors.append(f"requirement {requirement_id} must list evidence")

    for requirement_id in PROVEN_REQUIREMENT_IDS:
        if by_id.get(requirement_id, {}).get("status") != "proved":
            errors.append(f"requirement {requirement_id} must be proved by current evidence")
    open_row = by_id.get(OPEN_REQUIREMENT_ID)
    if not isinstance(open_row, dict) or open_row.get("status") != "open":
        errors.append("full_goal_completion must remain open in this progress audit")
    elif not isinstance(open_row.get("gaps"), list) or len(open_row["gaps"]) < 2:
        errors.append("full_goal_completion must list concrete remaining gaps")

    if isinstance(evidence.get("bundle_invariance"), dict):
        bundle_summary = evidence["bundle_invariance"].get("summary", {})
        if isinstance(bundle_summary, dict):
            if _finite_float(bundle_summary.get("max_rel_error"), "bundle_invariance max_rel_error", errors) > 1.0e-9:
                errors.append("bundle gauge value invariant must stay below 1e-9 rel error")
            if (
                _finite_float(
                    bundle_summary.get("min_bad_no_jacobian_rel_error"),
                    "bundle_invariance min_bad_no_jacobian_rel_error",
                    errors,
                )
                < 0.05
            ):
                errors.append("bundle gauge missing-Jacobian control must stay visibly wrong")
    if isinstance(evidence.get("bundle_gradient"), dict):
        gradient_summary = evidence["bundle_gradient"].get("summary", {})
        if isinstance(gradient_summary, dict):
            if _finite_float(gradient_summary.get("max_gradient_rel_error"), "bundle_gradient max_gradient_rel_error", errors) > 2.0e-6:
                errors.append("bundle gauge gradient invariant must stay below 2e-6 rel error")
            if _finite_float(gradient_summary.get("finite_difference_mean_x_rel_error"), "bundle_gradient finite_difference_mean_x_rel_error", errors) > 2.0e-6:
                errors.append("bundle gauge finite-difference check must stay below 2e-6 rel error")
    if isinstance(evidence.get("camera_family"), dict):
        camera_summary = evidence["camera_family"].get("summary", {})
        if isinstance(camera_summary, dict):
            if _finite_float(camera_summary.get("max_value_rel_error"), "camera_family max_value_rel_error", errors) > 2.0e-6:
                errors.append("camera-family value gauge invariant must stay below 2e-6 rel error")
            if (
                _finite_float(
                    camera_summary.get("max_primitive_gradient_rel_error"),
                    "camera_family max_primitive_gradient_rel_error",
                    errors,
                )
                > 2.0e-6
            ):
                errors.append("camera-family primitive gradient invariant must stay below 2e-6 rel error")
            if _finite_float(camera_summary.get("q_gradient_rel_error"), "camera_family q_gradient_rel_error", errors) > 2.0e-6:
                errors.append("camera-family q gradient invariant must stay below 2e-6 rel error")
            if (
                _finite_float(
                    camera_summary.get("q_finite_difference_rel_error"),
                    "camera_family q_finite_difference_rel_error",
                    errors,
                )
                > 2.0e-6
            ):
                errors.append("camera-family q finite-difference check must stay below 2e-6 rel error")
    if isinstance(evidence.get("camera_family_2d"), dict):
        camera_2d_summary = evidence["camera_family_2d"].get("summary", {})
        if isinstance(camera_2d_summary, dict):
            if (
                _finite_float(
                    camera_2d_summary.get("max_value_rel_error"),
                    "camera_family_2d max_value_rel_error",
                    errors,
                )
                > 2.0e-6
            ):
                errors.append("2D camera-family value gauge invariant must stay below 2e-6 rel error")
            if (
                _finite_float(
                    camera_2d_summary.get("max_primitive_gradient_rel_error"),
                    "camera_family_2d max_primitive_gradient_rel_error",
                    errors,
                )
                > 2.0e-6
            ):
                errors.append("2D camera-family primitive gradient invariant must stay below 2e-6 rel error")
            for axis in ("q_phase", "q_height"):
                if (
                    _finite_float(
                        camera_2d_summary.get(f"{axis}_gradient_rel_error"),
                        f"camera_family_2d {axis}_gradient_rel_error",
                        errors,
                    )
                    > 2.0e-6
                ):
                    errors.append(f"2D camera-family {axis} gradient invariant must stay below 2e-6 rel error")
                if (
                    _finite_float(
                        camera_2d_summary.get(f"{axis}_finite_difference_rel_error"),
                        f"camera_family_2d {axis}_finite_difference_rel_error",
                        errors,
                    )
                    > 2.0e-6
                ):
                    errors.append(f"2D camera-family {axis} finite-difference check must stay below 2e-6 rel error")
    if isinstance(evidence.get("camera_family_2d_metal_lowering"), dict):
        metal_lowering_summary = evidence["camera_family_2d_metal_lowering"].get("summary", {})
        if isinstance(metal_lowering_summary, dict):
            if (
                _finite_float(
                    metal_lowering_summary.get("family_to_replay_payload_ratio"),
                    "camera_family_2d_metal_lowering family_to_replay_payload_ratio",
                    errors,
                )
                >= 0.35
            ):
                errors.append("2D camera-family Metal lowering family/replay payload ratio must stay below 0.35")
            if (
                _finite_float(
                    metal_lowering_summary.get("peak_slice_to_replay_payload_ratio"),
                    "camera_family_2d_metal_lowering peak_slice_to_replay_payload_ratio",
                    errors,
                )
                >= 0.10
            ):
                errors.append("2D camera-family Metal lowering peak slice/replay ratio must stay below 0.10")
            if (
                _finite_float(
                    metal_lowering_summary.get("min_grad_coeff_abs_sum"),
                    "camera_family_2d_metal_lowering min_grad_coeff_abs_sum",
                    errors,
                )
                <= 1.0e-6
            ):
                errors.append("2D camera-family Metal lowering must produce nonzero coefficient gradients")
    if isinstance(evidence.get("camera_family_2d_metal_chain_rule"), dict):
        metal_chain_summary = evidence["camera_family_2d_metal_chain_rule"].get("summary", {})
        if isinstance(metal_chain_summary, dict):
            if (
                _finite_float(
                    metal_chain_summary.get("shared_to_replay_gradient_payload_ratio"),
                    "camera_family_2d_metal_chain_rule shared_to_replay_gradient_payload_ratio",
                    errors,
                )
                >= 0.30
            ):
                errors.append("2D camera-family Metal chain-rule shared/replay gradient ratio must stay below 0.30")
            if (
                _finite_float(
                    metal_chain_summary.get("max_finite_difference_rel_error"),
                    "camera_family_2d_metal_chain_rule max_finite_difference_rel_error",
                    errors,
                )
                >= 1.0e-3
            ):
                errors.append("2D camera-family Metal chain-rule finite-difference error must stay below 1e-3")
            if (
                _finite_float(
                    metal_chain_summary.get("shared_family_grad_abs_sum"),
                    "camera_family_2d_metal_chain_rule shared_family_grad_abs_sum",
                    errors,
                )
                <= 1.0e-6
            ):
                errors.append("2D camera-family Metal chain-rule must produce nonzero shared-family gradients")
    if isinstance(evidence.get("camera_family_2d_materialized_batch"), dict):
        materialized_summary = evidence["camera_family_2d_materialized_batch"].get("summary", {})
        if isinstance(materialized_summary, dict):
            if (
                _finite_float(
                    materialized_summary.get("forward_launch_ratio"),
                    "camera_family_2d_materialized_batch forward_launch_ratio",
                    errors,
                )
                >= 0.10
            ):
                errors.append("2D camera-family materialized batch forward launch ratio must stay below 0.10")
            if (
                _finite_float(
                    materialized_summary.get("backward_launch_ratio"),
                    "camera_family_2d_materialized_batch backward_launch_ratio",
                    errors,
                )
                >= 0.10
            ):
                errors.append("2D camera-family materialized batch backward launch ratio must stay below 0.10")
            if (
                abs(
                    _finite_float(
                        materialized_summary.get("materialized_to_replay_trace_payload_ratio"),
                        "camera_family_2d_materialized_batch materialized_to_replay_trace_payload_ratio",
                        errors,
                    )
                    - 1.0
                )
                > 1.0e-6
            ):
                errors.append("2D camera-family materialized batch must keep materialized/replay trace payload at 1.0")
            if (
                _finite_float(
                    materialized_summary.get("family_to_materialized_trace_payload_ratio"),
                    "camera_family_2d_materialized_batch family_to_materialized_trace_payload_ratio",
                    errors,
                )
                >= 0.35
            ):
                errors.append("2D camera-family materialized batch family/materialized payload ratio must stay below 0.35")
            if (
                _finite_float(
                    materialized_summary.get("max_batched_vs_slice_image_abs_error"),
                    "camera_family_2d_materialized_batch max_batched_vs_slice_image_abs_error",
                    errors,
                )
                > 1.0e-5
            ):
                errors.append("2D camera-family materialized batch image agreement must stay below 1e-5")
            if (
                _finite_float(
                    materialized_summary.get("max_batched_vs_slice_shared_grad_rel_error"),
                    "camera_family_2d_materialized_batch max_batched_vs_slice_shared_grad_rel_error",
                    errors,
                )
                > 1.0e-6
            ):
                errors.append("2D camera-family materialized batch shared-gradient rel error must stay below 1e-6")
    if isinstance(evidence.get("camera_family_2d_native_eval"), dict):
        native_eval_summary = evidence["camera_family_2d_native_eval"].get("summary", {})
        if isinstance(native_eval_summary, dict):
            if (
                _finite_float(
                    native_eval_summary.get("family_coeff_to_materialized_coeff_payload_ratio"),
                    "camera_family_2d_native_eval family_coeff_to_materialized_coeff_payload_ratio",
                    errors,
                )
                >= 0.30
            ):
                errors.append("2D camera-family native eval family/materialized coefficient ratio must stay below 0.30")
            if (
                _finite_float(
                    native_eval_summary.get("family_plus_q_basis_to_materialized_coeff_payload_ratio"),
                    "camera_family_2d_native_eval family_plus_q_basis_to_materialized_coeff_payload_ratio",
                    errors,
                )
                >= 0.65
            ):
                errors.append("2D camera-family native eval family-plus-q/materialized coefficient ratio must stay below 0.65")
            if (
                _finite_float(
                    native_eval_summary.get("native_eval_max_rel_error"),
                    "camera_family_2d_native_eval native_eval_max_rel_error",
                    errors,
                )
                > 1.0e-6
            ):
                errors.append("2D camera-family native eval value rel error must stay below 1e-6")
            if (
                _finite_float(
                    native_eval_summary.get("native_grad_family_max_rel_error"),
                    "camera_family_2d_native_eval native_grad_family_max_rel_error",
                    errors,
                )
                > 2.0e-5
            ):
                errors.append("2D camera-family native eval family-gradient rel error must stay below 2e-5")
            if (
                _finite_float(
                    native_eval_summary.get("native_grad_q_basis_max_rel_error"),
                    "camera_family_2d_native_eval native_grad_q_basis_max_rel_error",
                    errors,
                )
                > 2.0e-5
            ):
                errors.append("2D camera-family native eval q-basis-gradient rel error must stay below 2e-5")
    if isinstance(evidence.get("camera_family_2d_native_interval_forward"), dict):
        native_forward_summary = evidence["camera_family_2d_native_interval_forward"].get("summary", {})
        if isinstance(native_forward_summary, dict):
            if (
                _finite_float(
                    native_forward_summary.get("family_forward_to_materialized_trace_payload_ratio"),
                    "camera_family_2d_native_interval_forward family_forward_to_materialized_trace_payload_ratio",
                    errors,
                )
                >= 0.50
            ):
                errors.append("2D camera-family native interval forward payload ratio must stay below 0.50")
            if (
                _finite_float(
                    native_forward_summary.get("family_coeff_to_materialized_trace_payload_ratio"),
                    "camera_family_2d_native_interval_forward family_coeff_to_materialized_trace_payload_ratio",
                    errors,
                )
                >= 0.25
            ):
                errors.append("2D camera-family native interval forward coeff/materialized payload ratio must stay below 0.25")
            if (
                _finite_float(
                    native_forward_summary.get("native_family_forward_max_rel_error"),
                    "camera_family_2d_native_interval_forward native_family_forward_max_rel_error",
                    errors,
                )
                > 1.0e-6
            ):
                errors.append("2D camera-family native interval forward image rel error must stay below 1e-6")
            if (
                _finite_float(
                    native_forward_summary.get("native_family_image_abs_sum"),
                    "camera_family_2d_native_interval_forward native_family_image_abs_sum",
                    errors,
                )
                <= 0.0
            ):
                errors.append("2D camera-family native interval forward image support must be nonzero")
    if isinstance(evidence.get("camera_family_2d_native_interval_backward"), dict):
        native_backward_summary = evidence["camera_family_2d_native_interval_backward"].get("summary", {})
        if isinstance(native_backward_summary, dict):
            if (
                _finite_float(
                    native_backward_summary.get("native_family_gradient_to_materialized_gradient_payload_ratio"),
                    "camera_family_2d_native_interval_backward native_family_gradient_to_materialized_gradient_payload_ratio",
                    errors,
                )
                >= 0.35
            ):
                errors.append("2D camera-family native interval backward gradient payload ratio must stay below 0.35")
            if (
                _finite_float(
                    native_backward_summary.get("native_family_coeff_gradient_to_materialized_gradient_payload_ratio"),
                    "camera_family_2d_native_interval_backward native_family_coeff_gradient_to_materialized_gradient_payload_ratio",
                    errors,
                )
                >= 0.15
            ):
                errors.append("2D camera-family native interval backward coeff-gradient payload ratio must stay below 0.15")
            if (
                _finite_float(
                    native_backward_summary.get("native_family_interval_backward_max_family_grad_rel_error"),
                    "camera_family_2d_native_interval_backward native_family_interval_backward_max_family_grad_rel_error",
                    errors,
                )
                > 1.0e-5
            ):
                errors.append("2D camera-family native interval backward family-gradient rel error must stay below 1e-5")
            if (
                _finite_float(
                    native_backward_summary.get("native_family_interval_backward_max_q_basis_grad_rel_error"),
                    "camera_family_2d_native_interval_backward native_family_interval_backward_max_q_basis_grad_rel_error",
                    errors,
                )
                > 1.0e-5
            ):
                errors.append("2D camera-family native interval backward q-basis-gradient rel error must stay below 1e-5")
            if (
                _finite_float(
                    native_backward_summary.get("native_family_grad_abs_sum"),
                    "camera_family_2d_native_interval_backward native_family_grad_abs_sum",
                    errors,
                )
                <= 0.0
            ):
                errors.append("2D camera-family native interval backward family-gradient support must be nonzero")
            if (
                _finite_float(
                    native_backward_summary.get("native_q_basis_grad_abs_sum"),
                    "camera_family_2d_native_interval_backward native_q_basis_grad_abs_sum",
                    errors,
                )
                <= 0.0
            ):
                errors.append("2D camera-family native interval backward q-basis-gradient support must be nonzero")
    if isinstance(evidence.get("camera_family_2d_tile_order_reuse"), dict):
        tile_order_summary = evidence["camera_family_2d_tile_order_reuse"].get("summary", {})
        if isinstance(tile_order_summary, dict):
            if (
                _finite_float(
                    tile_order_summary.get("shared_to_materialized_tile_order_metadata_ratio"),
                    "camera_family_2d_tile_order_reuse shared_to_materialized_tile_order_metadata_ratio",
                    errors,
                )
                >= 0.20
            ):
                errors.append("2D camera-family tile/order shared/materialized metadata ratio must stay below 0.20")
            if (
                _finite_float(
                    tile_order_summary.get("materialized_tile_order_metadata_growth"),
                    "camera_family_2d_tile_order_reuse materialized_tile_order_metadata_growth",
                    errors,
                )
                < 25.0
            ):
                errors.append("2D camera-family materialized tile/order metadata growth must expose sampled-Q scaling")
            if (
                _finite_float(
                    tile_order_summary.get("shared_tile_order_metadata_growth"),
                    "camera_family_2d_tile_order_reuse shared_tile_order_metadata_growth",
                    errors,
                )
                > 1.05
            ):
                errors.append("2D camera-family shared tile/order metadata growth must stay near constant")
            if tile_order_summary.get("expanded_topology_matches_materialized") is not True:
                errors.append("2D camera-family shared tile/order topology must expand to materialized topology")
            if tile_order_summary.get("stable_union_depth_order") is not True:
                errors.append("2D camera-family shared tile/order depth certificate must remain stable")
            if (
                _finite_float(
                    tile_order_summary.get("min_union_depth_order_gap"),
                    "camera_family_2d_tile_order_reuse min_union_depth_order_gap",
                    errors,
                )
                <= 0.25
            ):
                errors.append("2D camera-family tile/order union depth gap must stay positive")
    if isinstance(evidence.get("camera_family_2d_tile_order_strata"), dict):
        tile_order_strata_summary = evidence["camera_family_2d_tile_order_strata"].get("summary", {})
        if isinstance(tile_order_strata_summary, dict):
            if (
                _finite_float(
                    tile_order_strata_summary.get("shared_to_materialized_tile_order_metadata_ratio"),
                    "camera_family_2d_tile_order_strata shared_to_materialized_tile_order_metadata_ratio",
                    errors,
                )
                >= 0.25
            ):
                errors.append("2D camera-family tile/order strata shared/materialized metadata ratio must stay below 0.25")
            if (
                _finite_float(
                    tile_order_strata_summary.get("materialized_tile_order_metadata_growth"),
                    "camera_family_2d_tile_order_strata materialized_tile_order_metadata_growth",
                    errors,
                )
                < 25.0
            ):
                errors.append("2D camera-family tile/order strata materialized growth must expose sampled-Q scaling")
            if (
                _finite_float(
                    tile_order_strata_summary.get("shared_tile_order_metadata_growth"),
                    "camera_family_2d_tile_order_strata shared_tile_order_metadata_growth",
                    errors,
                )
                > 2.05
            ):
                errors.append("2D camera-family tile/order strata shared growth must track strata count")
            if int(tile_order_strata_summary.get("order_stratum_count") or 0) != 2:
                errors.append("2D camera-family tile/order strata count must stay two in this smoke")
            if tile_order_strata_summary.get("expanded_topology_matches_materialized") is not True:
                errors.append("2D camera-family tile/order strata must expand to materialized topology")
            if tile_order_strata_summary.get("all_strata_depth_order_stable") is not True:
                errors.append("2D camera-family tile/order strata depth certificates must remain stable")
            if (
                _finite_float(
                    tile_order_strata_summary.get("min_stratum_union_depth_order_gap"),
                    "camera_family_2d_tile_order_strata min_stratum_union_depth_order_gap",
                    errors,
                )
                <= 0.20
            ):
                errors.append("2D camera-family tile/order strata union depth gap must stay positive")
    if isinstance(evidence.get("camera_family_2d_active_set_strata"), dict):
        active_set_strata_summary = evidence["camera_family_2d_active_set_strata"].get("summary", {})
        if isinstance(active_set_strata_summary, dict):
            if (
                _finite_float(
                    active_set_strata_summary.get("shared_to_materialized_tile_order_metadata_ratio"),
                    "camera_family_2d_active_set_strata shared_to_materialized_tile_order_metadata_ratio",
                    errors,
                )
                >= 0.25
            ):
                errors.append("2D camera-family active-set strata shared/materialized metadata ratio must stay below 0.25")
            if (
                _finite_float(
                    active_set_strata_summary.get("materialized_tile_order_metadata_growth"),
                    "camera_family_2d_active_set_strata materialized_tile_order_metadata_growth",
                    errors,
                )
                < 25.0
            ):
                errors.append("2D camera-family active-set strata materialized growth must expose sampled-Q scaling")
            if (
                _finite_float(
                    active_set_strata_summary.get("shared_tile_order_metadata_growth"),
                    "camera_family_2d_active_set_strata shared_tile_order_metadata_growth",
                    errors,
                )
                > 3.05
            ):
                errors.append("2D camera-family active-set strata shared growth must track strata count")
            if int(active_set_strata_summary.get("active_set_stratum_count") or 0) != 3:
                errors.append("2D camera-family active-set strata count must stay three in this smoke")
            if active_set_strata_summary.get("expanded_topology_matches_materialized") is not True:
                errors.append("2D camera-family active-set strata must expand to materialized topology")
            if active_set_strata_summary.get("all_active_set_strata_depth_order_stable") is not True:
                errors.append("2D camera-family active-set strata depth certificates must remain stable")
            if (
                _finite_float(
                    active_set_strata_summary.get("min_active_set_union_depth_order_gap"),
                    "camera_family_2d_active_set_strata min_active_set_union_depth_order_gap",
                    errors,
                )
                <= 0.20
            ):
                errors.append("2D camera-family active-set strata union depth gap must stay positive")
    if isinstance(evidence.get("real_active_set_distribution"), dict):
        real_active_set_summary = evidence["real_active_set_distribution"].get("summary", {})
        if isinstance(real_active_set_summary, dict):
            if int(real_active_set_summary.get("artifact_count") or 0) < 3:
                errors.append("real active-set distribution must cover at least three artifacts")
            if int(real_active_set_summary.get("row_count") or 0) < 9:
                errors.append("real active-set distribution must cover at least nine rows")
            if real_active_set_summary.get("all_underlying_verifiers_pass") is not True:
                errors.append("real active-set distribution underlying verifiers must pass")
            if real_active_set_summary.get("all_source_videos_exist") is not True:
                errors.append("real active-set distribution source videos must exist")
            if real_active_set_summary.get("all_fallback_free") is not True:
                errors.append("real active-set distribution rows must be fallback-free")
            if (
                _finite_float(
                    real_active_set_summary.get("max_active_set_group_to_dense_tile_pair_ratio"),
                    "real_active_set_distribution max_active_set_group_to_dense_tile_pair_ratio",
                    errors,
                )
                >= 0.05
            ):
                errors.append("real active-set distribution group/dense-tile-pair ratio must stay below 0.05")
            if int(real_active_set_summary.get("max_cells_per_active_set_group") or 0) > 3:
                errors.append("real active-set distribution max cells per active-set group must stay <= 3")
            if (
                _finite_float(
                    real_active_set_summary.get("max_cell_to_active_set_group_ratio"),
                    "real_active_set_distribution max_cell_to_active_set_group_ratio",
                    errors,
                )
                > 1.5
            ):
                errors.append("real active-set distribution cell/group ratio must stay bounded")
    if isinstance(evidence.get("camera_family_shared_work"), dict):
        family_shared_summary = evidence["camera_family_shared_work"].get("summary", {})
        if isinstance(family_shared_summary, dict):
            if (
                _finite_float(
                    family_shared_summary.get("final_payload_ratio"),
                    "camera_family_shared_work final_payload_ratio",
                    errors,
                )
                >= 0.30
            ):
                errors.append("camera-family shared payload ratio must stay below 0.30")
            if (
                _finite_float(
                    family_shared_summary.get("final_chart_ratio"),
                    "camera_family_shared_work final_chart_ratio",
                    errors,
                )
                >= 0.15
            ):
                errors.append("camera-family shared chart ratio must stay below 0.15")
            if (
                _finite_float(
                    family_shared_summary.get("family_payload_growth"),
                    "camera_family_shared_work family_payload_growth",
                    errors,
                )
                > 1.05
            ):
                errors.append("camera-family shared payload growth must stay near constant")
            if (
                _finite_float(
                    family_shared_summary.get("per_q_replay_payload_growth"),
                    "camera_family_shared_work per_q_replay_payload_growth",
                    errors,
                )
                < 4.0
            ):
                errors.append("camera-family per-q replay payload growth must stay visible")
            if (
                _finite_float(
                    family_shared_summary.get("max_family_fit_uv_error_px"),
                    "camera_family_shared_work max_family_fit_uv_error_px",
                    errors,
                )
                > 0.50
            ):
                errors.append("camera-family shared QxT fit residual must stay below 0.50 px")
    if isinstance(evidence.get("camera_family_2d_shared_work"), dict):
        family_2d_shared_summary = evidence["camera_family_2d_shared_work"].get("summary", {})
        if isinstance(family_2d_shared_summary, dict):
            if (
                _finite_float(
                    family_2d_shared_summary.get("final_payload_ratio"),
                    "camera_family_2d_shared_work final_payload_ratio",
                    errors,
                )
                >= 0.15
            ):
                errors.append("2D camera-family shared payload ratio must stay below 0.15")
            if (
                _finite_float(
                    family_2d_shared_summary.get("final_chart_ratio"),
                    "camera_family_2d_shared_work final_chart_ratio",
                    errors,
                )
                >= 0.05
            ):
                errors.append("2D camera-family shared chart ratio must stay below 0.05")
            if (
                _finite_float(
                    family_2d_shared_summary.get("family_payload_growth"),
                    "camera_family_2d_shared_work family_payload_growth",
                    errors,
                )
                > 1.05
            ):
                errors.append("2D camera-family shared payload growth must stay near constant")
            if (
                _finite_float(
                    family_2d_shared_summary.get("per_q_replay_payload_growth"),
                    "camera_family_2d_shared_work per_q_replay_payload_growth",
                    errors,
                )
                < 16.0
            ):
                errors.append("2D camera-family per-q-pair replay payload growth must stay visible")
            if (
                _finite_float(
                    family_2d_shared_summary.get("max_family_fit_uv_error_px"),
                    "camera_family_2d_shared_work max_family_fit_uv_error_px",
                    errors,
                )
                > 0.50
            ):
                errors.append("2D camera-family shared Q2xT fit residual must stay below 0.50 px")
    if isinstance(evidence.get("trainer_interval"), dict):
        trainer_summary = evidence["trainer_interval"].get("summary", {})
        if isinstance(trainer_summary, dict):
            if trainer_summary.get("measured_all_pass") is not True:
                errors.append("trainer interval measured rows must all pass")
            if trainer_summary.get("measured_all_no_overflow") is not True:
                errors.append("trainer interval measured rows must have no overflow")
            if trainer_summary.get("all_measured_loss_matches_cadence") is not True:
                errors.append("trainer interval measured losses must match cadence losses")
            if (
                _finite_float(
                    trainer_summary.get("max_measured_vs_cadence_end_loss_abs_delta"),
                    "trainer_interval max loss delta",
                    errors,
                )
                >= 1.0e-5
            ):
                errors.append("trainer interval measured/cadence loss delta must stay below 1e-5")
            no_first_ratios = trainer_summary.get("measured_vs_cadence_no_first_step_ms_ratios")
            if not isinstance(no_first_ratios, list) or not no_first_ratios:
                errors.append("trainer interval must report measured/cadence no-first-step timing ratios")
            elif any(_finite_float(value, "trainer_interval no_first ratio", errors) >= 1.0 for value in no_first_ratios):
                errors.append("trainer interval measured no-first-step timings must beat cadence")
            rebuild_ratios = trainer_summary.get("measured_vs_cadence_rebuild_ratios")
            if not isinstance(rebuild_ratios, list) or not rebuild_ratios:
                errors.append("trainer interval must report measured/cadence rebuild ratios")
            elif any(_finite_float(value, "trainer_interval rebuild ratio", errors) >= 1.0 for value in rebuild_ratios):
                errors.append("trainer interval measured rebuild ratios must stay below 1")
    if isinstance(evidence.get("trainer_real_video"), dict):
        trainer_summary = evidence["trainer_real_video"].get("summary", {})
        if isinstance(trainer_summary, dict):
            if trainer_summary.get("measured_all_pass") is not True:
                errors.append("real-video trainer measured rows must all pass")
            if trainer_summary.get("measured_all_no_overflow") is not True:
                errors.append("real-video trainer measured rows must have no overflow")
            if trainer_summary.get("all_measured_loss_matches_cadence") is not True:
                errors.append("real-video trainer measured losses must match cadence losses")
            if (
                _finite_float(
                    trainer_summary.get("max_measured_vs_cadence_end_loss_abs_delta"),
                    "trainer_real_video max loss delta",
                    errors,
                )
                >= 1.0e-5
            ):
                errors.append("real-video trainer measured/cadence loss delta must stay below 1e-5")
            no_first_ratios = trainer_summary.get("measured_vs_cadence_no_first_step_ms_ratios")
            if not isinstance(no_first_ratios, list) or not no_first_ratios:
                errors.append("real-video trainer must report measured/cadence no-first-step timing ratios")
            elif any(
                _finite_float(value, "trainer_real_video no_first ratio", errors) >= 1.0
                for value in no_first_ratios
            ):
                errors.append("real-video trainer measured no-first-step timings must beat cadence")
            rebuild_ratios = trainer_summary.get("measured_vs_cadence_rebuild_ratios")
            if not isinstance(rebuild_ratios, list) or not rebuild_ratios:
                errors.append("real-video trainer must report measured/cadence rebuild ratios")
            elif any(
                _finite_float(value, "trainer_real_video rebuild ratio", errors) >= 1.0
                for value in rebuild_ratios
            ):
                errors.append("real-video trainer measured rebuild ratios must stay below 1")
    if isinstance(evidence.get("real_video_guarded_support_matrix"), dict):
        guarded_summary = evidence["real_video_guarded_support_matrix"].get("summary", {})
        if isinstance(guarded_summary, dict):
            if guarded_summary.get("all_underlying_verifiers_pass") is not True:
                errors.append("real-video guarded support matrix underlying verifiers must pass")
            if guarded_summary.get("all_guarded_support_verifiers_pass") is not True:
                errors.append("real-video guarded support matrix guarded verifiers must pass")
            if guarded_summary.get("all_source_videos_exist") is not True:
                errors.append("real-video guarded support matrix source videos must exist")
            if int(guarded_summary.get("guarded_artifact_count") or 0) < 4:
                errors.append("real-video guarded support matrix must cover at least four guard paddings")
            if int(guarded_summary.get("default_measured_support_rebins") or 0) <= 0:
                errors.append("real-video guarded support matrix must preserve the unguarded support-churn control")
            if int(guarded_summary.get("guarded_measured_support_rebins") or 0) != 0:
                errors.append("real-video guarded support matrix must eliminate guarded support rebins")
            if int(guarded_summary.get("guarded_measured_stale_refreshes") or 0) != 0:
                errors.append("real-video guarded support matrix must eliminate guarded stale refreshes")
            if (
                _finite_float(
                    guarded_summary.get("max_guarded_measured_no_first_ratio"),
                    "real_video_guarded_support max no-first ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video guarded support matrix no-first timings must beat cadence")
            if (
                _finite_float(
                    guarded_summary.get("max_guarded_measured_rebuild_ratio"),
                    "real_video_guarded_support max rebuild ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video guarded support matrix rebuild ratios must stay below cadence")
    if isinstance(evidence.get("real_video_multiscene_trainer_matrix"), dict):
        multiscene_summary = evidence["real_video_multiscene_trainer_matrix"].get("summary", {})
        if isinstance(multiscene_summary, dict):
            if int(multiscene_summary.get("scene_count") or 0) < 3:
                errors.append("real-video multiscene trainer matrix must cover at least three scenes")
            if int(multiscene_summary.get("distinct_youtube_id_count") or 0) < 3:
                errors.append("real-video multiscene trainer matrix must cover source-distinct videos")
            if multiscene_summary.get("all_source_videos_exist") is not True:
                errors.append("real-video multiscene trainer matrix source videos must exist")
            if multiscene_summary.get("all_rows_pass") is not True:
                errors.append("real-video multiscene trainer matrix rows must all pass")
            if multiscene_summary.get("all_rows_loss_decreased") is not True:
                errors.append("real-video multiscene trainer matrix rows must all decrease loss")
            if multiscene_summary.get("all_rows_no_overflow") is not True:
                errors.append("real-video multiscene trainer matrix rows must have no overflow")
            if multiscene_summary.get("all_rows_fallback_free") is not True:
                errors.append("real-video multiscene trainer matrix rows must be fallback-free")
            if multiscene_summary.get("all_rows_visibility_stratification_free") is not True:
                errors.append("real-video multiscene trainer matrix rows must avoid visibility stratification")
            if multiscene_summary.get("all_measured_loss_matches_cadence") is not True:
                errors.append("real-video multiscene trainer matrix measured losses must match cadence")
            if (
                _finite_float(
                    multiscene_summary.get("max_measured_vs_cadence_end_loss_abs_delta"),
                    "real_video_multiscene max loss delta",
                    errors,
                )
                >= 1.0e-5
            ):
                errors.append("real-video multiscene trainer matrix loss delta must stay below 1e-5")
            if int(multiscene_summary.get("max_measured_support_rebins") or 0) != 0:
                errors.append("real-video multiscene trainer matrix must eliminate measured support rebins")
            if int(multiscene_summary.get("max_measured_stale_refreshes") or 0) != 0:
                errors.append("real-video multiscene trainer matrix must eliminate measured stale refreshes")
            if (
                _finite_float(
                    multiscene_summary.get("max_measured_vs_cadence_no_first_step_ms_ratio"),
                    "real_video_multiscene max no-first ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video multiscene trainer matrix no-first timings must beat cadence")
            if (
                _finite_float(
                    multiscene_summary.get("max_measured_vs_cadence_rebuild_ratio"),
                    "real_video_multiscene max rebuild ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video multiscene trainer matrix rebuild ratios must stay below cadence")
            if (
                _finite_float(
                    multiscene_summary.get("max_measured_support_tail_alpha_bound"),
                    "real_video_multiscene max support tail alpha bound",
                    errors,
                )
                > 1.0e-3
            ):
                errors.append("real-video multiscene trainer matrix support tail bound must stay below 1e-3")
    if isinstance(evidence.get("real_video_multiscene_extended_trainer_matrix"), dict):
        extended_summary = evidence["real_video_multiscene_extended_trainer_matrix"].get("summary", {})
        if isinstance(extended_summary, dict):
            if int(extended_summary.get("scene_count") or 0) < 5:
                errors.append("extended real-video multiscene trainer matrix must cover at least five scenes")
            if int(extended_summary.get("distinct_youtube_id_count") or 0) < 5:
                errors.append("extended real-video multiscene trainer matrix must cover source-distinct videos")
            if extended_summary.get("all_source_videos_exist") is not True:
                errors.append("extended real-video multiscene trainer matrix source videos must exist")
            if extended_summary.get("all_rows_pass") is not True:
                errors.append("extended real-video multiscene trainer matrix rows must all pass")
            if extended_summary.get("all_rows_loss_decreased") is not True:
                errors.append("extended real-video multiscene trainer matrix rows must all decrease loss")
            if extended_summary.get("all_rows_no_overflow") is not True:
                errors.append("extended real-video multiscene trainer matrix rows must have no overflow")
            if extended_summary.get("all_rows_fallback_free") is not True:
                errors.append("extended real-video multiscene trainer matrix rows must be fallback-free")
            if extended_summary.get("all_rows_visibility_stratification_free") is not True:
                errors.append("extended real-video multiscene trainer matrix rows must avoid visibility stratification")
            if extended_summary.get("all_measured_loss_matches_cadence") is not True:
                errors.append("extended real-video multiscene trainer matrix measured losses must match cadence")
            if (
                _finite_float(
                    extended_summary.get("max_motion_score"),
                    "extended real_video_multiscene max motion score",
                    errors,
                )
                < 7.0
            ):
                errors.append("extended real-video multiscene trainer matrix must include a high-motion scene")
            if (
                _finite_float(
                    extended_summary.get("max_measured_vs_cadence_end_loss_abs_delta"),
                    "extended real_video_multiscene max loss delta",
                    errors,
                )
                >= 1.0e-5
            ):
                errors.append("extended real-video multiscene trainer matrix loss delta must stay below 1e-5")
            if int(extended_summary.get("max_measured_support_rebins") or 0) != 0:
                errors.append("extended real-video multiscene trainer matrix must eliminate measured support rebins")
            if int(extended_summary.get("max_measured_stale_refreshes") or 0) != 0:
                errors.append("extended real-video multiscene trainer matrix must eliminate measured stale refreshes")
            if (
                _finite_float(
                    extended_summary.get("max_measured_vs_cadence_rebuild_ratio"),
                    "extended real_video_multiscene max rebuild ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("extended real-video multiscene trainer matrix rebuild ratios must stay below cadence")
            _finite_float(
                extended_summary.get("max_measured_vs_cadence_no_first_step_ms_ratio"),
                "extended real_video_multiscene max no-first ratio",
                errors,
            )
    if isinstance(evidence.get("real_video_multiscene_frame_scaling_matrix"), dict):
        scaling_summary = evidence["real_video_multiscene_frame_scaling_matrix"].get("summary", {})
        if isinstance(scaling_summary, dict):
            if int(scaling_summary.get("scene_count") or 0) < 3:
                errors.append("real-video multiscene frame-scaling matrix must cover at least three scenes")
            if int(scaling_summary.get("frame_count_count") or 0) < 3:
                errors.append("real-video multiscene frame-scaling matrix must cover at least three frame counts")
            if int(scaling_summary.get("distinct_youtube_id_count") or 0) < 3:
                errors.append("real-video multiscene frame-scaling matrix must cover source-distinct videos")
            if scaling_summary.get("all_source_videos_exist") is not True:
                errors.append("real-video multiscene frame-scaling matrix source videos must exist")
            if scaling_summary.get("all_rows_pass") is not True:
                errors.append("real-video multiscene frame-scaling matrix rows must all pass")
            if scaling_summary.get("all_rows_loss_decreased") is not True:
                errors.append("real-video multiscene frame-scaling matrix rows must all decrease loss")
            if scaling_summary.get("all_rows_no_overflow") is not True:
                errors.append("real-video multiscene frame-scaling matrix rows must have no overflow")
            if scaling_summary.get("all_rows_fallback_free") is not True:
                errors.append("real-video multiscene frame-scaling matrix rows must be fallback-free")
            if scaling_summary.get("all_rows_visibility_stratification_free") is not True:
                errors.append("real-video multiscene frame-scaling matrix rows must avoid visibility stratification")
            if scaling_summary.get("all_measured_loss_matches_cadence") is not True:
                errors.append("real-video multiscene frame-scaling matrix measured losses must match cadence")
            if (
                _finite_float(
                    scaling_summary.get("frame_growth_factor"),
                    "real_video_multiscene_frame_scaling frame growth factor",
                    errors,
                )
                < 4.0
            ):
                errors.append("real-video multiscene frame-scaling matrix must cover at least 4x frame growth")
            if (
                _finite_float(
                    scaling_summary.get("max_measured_vs_cadence_end_loss_abs_delta"),
                    "real_video_multiscene_frame_scaling max loss delta",
                    errors,
                )
                >= 1.0e-5
            ):
                errors.append("real-video multiscene frame-scaling matrix loss delta must stay below 1e-5")
            if int(scaling_summary.get("max_measured_support_rebins") or 0) != 0:
                errors.append("real-video multiscene frame-scaling matrix must eliminate measured support rebins")
            if int(scaling_summary.get("max_measured_stale_refreshes") or 0) != 0:
                errors.append("real-video multiscene frame-scaling matrix must eliminate measured stale refreshes")
            if (
                _finite_float(
                    scaling_summary.get("max_measured_vs_cadence_no_first_step_ms_ratio"),
                    "real_video_multiscene_frame_scaling max no-first ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video multiscene frame-scaling matrix no-first timings must beat cadence")
            if (
                _finite_float(
                    scaling_summary.get("max_measured_vs_cadence_rebuild_ratio"),
                    "real_video_multiscene_frame_scaling max rebuild ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video multiscene frame-scaling matrix rebuild ratios must stay below cadence")
            if (
                _finite_float(
                    scaling_summary.get("max_measured_cache_rebuild_growth"),
                    "real_video_multiscene_frame_scaling max rebuild growth",
                    errors,
                )
                > 1.0
            ):
                errors.append("real-video multiscene frame-scaling matrix measured rebuilds must not grow")
            if (
                _finite_float(
                    scaling_summary.get("max_measured_no_first_growth_vs_frame_growth_ratio"),
                    "real_video_multiscene_frame_scaling max no-first/frame growth ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video multiscene frame-scaling matrix timing growth must stay below frame growth")
            if (
                _finite_float(
                    scaling_summary.get("max_measured_support_tail_alpha_bound"),
                    "real_video_multiscene_frame_scaling max support tail alpha bound",
                    errors,
                )
                > 1.0e-3
            ):
                errors.append("real-video multiscene frame-scaling matrix support tail bound must stay below 1e-3")
    if isinstance(evidence.get("real_video_multiscene_extended_frame_scaling_diagnostic"), dict):
        diagnostic_summary = evidence["real_video_multiscene_extended_frame_scaling_diagnostic"].get("summary", {})
        if isinstance(diagnostic_summary, dict):
            if int(diagnostic_summary.get("source_scene_count") or 0) < 5:
                errors.append("extended frame-scaling diagnostic must cover at least five scenes")
            if int(diagnostic_summary.get("source_distinct_youtube_id_count") or 0) < 5:
                errors.append("extended frame-scaling diagnostic must cover five source-distinct videos")
            if int(diagnostic_summary.get("source_row_count") or 0) < 30:
                errors.append("extended frame-scaling diagnostic must cover at least 30 rows")
            if diagnostic_summary.get("strict_failed_only_expected_timing") is not True:
                errors.append("extended frame-scaling diagnostic must fail only the expected timing gates")
            if diagnostic_summary.get("no_first_timing_win") is not False:
                errors.append("extended frame-scaling diagnostic must not claim a timing win")
            if diagnostic_summary.get("all_source_videos_exist") is not True:
                errors.append("extended frame-scaling diagnostic source videos must exist")
            if diagnostic_summary.get("all_rows_pass") is not True:
                errors.append("extended frame-scaling diagnostic rows must pass")
            if diagnostic_summary.get("all_rows_loss_decreased") is not True:
                errors.append("extended frame-scaling diagnostic rows must decrease loss")
            if diagnostic_summary.get("all_rows_no_overflow") is not True:
                errors.append("extended frame-scaling diagnostic rows must have no overflow")
            if diagnostic_summary.get("all_rows_fallback_free") is not True:
                errors.append("extended frame-scaling diagnostic rows must be fallback-free")
            if diagnostic_summary.get("all_rows_visibility_stratification_free") is not True:
                errors.append("extended frame-scaling diagnostic rows must avoid visibility stratification")
            if diagnostic_summary.get("all_measured_loss_matches_cadence") is not True:
                errors.append("extended frame-scaling diagnostic measured losses must match cadence")
            if int(diagnostic_summary.get("max_measured_support_rebins") or 0) != 0:
                errors.append("extended frame-scaling diagnostic must have zero support rebins")
            if int(diagnostic_summary.get("max_measured_stale_refreshes") or 0) != 0:
                errors.append("extended frame-scaling diagnostic must have zero stale refreshes")
            if (
                _finite_float(
                    diagnostic_summary.get("max_measured_vs_cadence_rebuild_ratio"),
                    "extended frame-scaling diagnostic max rebuild ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("extended frame-scaling diagnostic rebuild ratio must stay below cadence")
            if (
                _finite_float(
                    diagnostic_summary.get("max_measured_cache_rebuild_growth"),
                    "extended frame-scaling diagnostic max rebuild growth",
                    errors,
                )
                > 1.0
            ):
                errors.append("extended frame-scaling diagnostic rebuild count must not grow")
            if (
                _finite_float(
                    diagnostic_summary.get("max_measured_vs_cadence_end_loss_abs_delta"),
                    "extended frame-scaling diagnostic max loss delta",
                    errors,
                )
                >= 1.0e-5
            ):
                errors.append("extended frame-scaling diagnostic loss delta must stay below 1e-5")
    if isinstance(evidence.get("real_video_multiscene_quality_tether"), dict):
        quality_summary = evidence["real_video_multiscene_quality_tether"].get("summary", {})
        if isinstance(quality_summary, dict):
            if int(quality_summary.get("scene_count") or 0) < 3:
                errors.append("real-video multiscene quality tether must cover at least three scenes")
            if int(quality_summary.get("frame_count_count") or 0) < 3:
                errors.append("real-video multiscene quality tether must cover at least three frame counts")
            if int(quality_summary.get("pair_count") or 0) < 9:
                errors.append("real-video multiscene quality tether must cover at least nine case pairs")
            if quality_summary.get("all_case_files_exist") is not True:
                errors.append("real-video multiscene quality tether case files must exist")
            if quality_summary.get("all_rows_pass") is not True:
                errors.append("real-video multiscene quality tether rows must pass")
            if quality_summary.get("all_gradient_flags_present") is not True:
                errors.append("real-video multiscene quality tether must preserve gradient-flow flags")
            if quality_summary.get("all_measured_loss_curves_match_cadence") is not True:
                errors.append("real-video multiscene quality tether loss curves must match cadence")
            if quality_summary.get("all_measured_end_psnr_matches_cadence") is not True:
                errors.append("real-video multiscene quality tether end PSNR must match cadence")
            if quality_summary.get("all_measured_psnr_improves") is not True:
                errors.append("real-video multiscene quality tether measured PSNR must improve")
            if (
                _finite_float(
                    quality_summary.get("max_abs_loss_curve_delta"),
                    "real_video_multiscene_quality max loss curve delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("real-video multiscene quality tether loss-curve delta must stay below 1e-8")
            if (
                _finite_float(
                    quality_summary.get("max_end_psnr_abs_delta"),
                    "real_video_multiscene_quality max end psnr delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("real-video multiscene quality tether end-PSNR delta must stay below 1e-8")
            if (
                _finite_float(
                    quality_summary.get("min_measured_psnr_gain"),
                    "real_video_multiscene_quality min psnr gain",
                    errors,
                )
                <= 0.0
            ):
                errors.append("real-video multiscene quality tether PSNR gains must be positive")
    if isinstance(evidence.get("real_video_multiscene_extended_quality_tether"), dict):
        quality_summary = evidence["real_video_multiscene_extended_quality_tether"].get("summary", {})
        if isinstance(quality_summary, dict):
            if int(quality_summary.get("source_scene_count") or 0) < 5:
                errors.append("extended real-video quality tether source matrix must cover at least five scenes")
            if int(quality_summary.get("source_distinct_youtube_id_count") or 0) < 5:
                errors.append("extended real-video quality tether must cover at least five source videos")
            if int(quality_summary.get("scene_count") or 0) < 5:
                errors.append("extended real-video quality tether must cover at least five scenes")
            if int(quality_summary.get("pair_count") or 0) < 5:
                errors.append("extended real-video quality tether must cover at least five case pairs")
            if quality_summary.get("all_case_files_exist") is not True:
                errors.append("extended real-video quality tether case files must exist")
            if quality_summary.get("all_rows_pass") is not True:
                errors.append("extended real-video quality tether rows must pass")
            if quality_summary.get("all_rows_error_free") is not True:
                errors.append("extended real-video quality tether rows must be error-free")
            if quality_summary.get("all_gradient_flags_present") is not True:
                errors.append("extended real-video quality tether must preserve gradient-flow flags")
            if quality_summary.get("all_measured_loss_curves_match_cadence") is not True:
                errors.append("extended real-video quality tether loss curves must match cadence")
            if quality_summary.get("all_measured_rgb_loss_curves_match_cadence") is not True:
                errors.append("extended real-video quality tether RGB-loss curves must match cadence")
            if quality_summary.get("all_measured_end_psnr_matches_cadence") is not True:
                errors.append("extended real-video quality tether end PSNR must match cadence")
            if quality_summary.get("all_measured_psnr_improves") is not True:
                errors.append("extended real-video quality tether measured PSNR must improve")
            if quality_summary.get("all_measured_loss_decreases") is not True:
                errors.append("extended real-video quality tether measured loss must decrease")
            if (
                _finite_float(
                    quality_summary.get("max_abs_loss_curve_delta"),
                    "extended real_video_multiscene_quality max loss curve delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("extended real-video quality tether loss-curve delta must stay below 1e-8")
            if (
                _finite_float(
                    quality_summary.get("max_abs_rgb_loss_curve_delta"),
                    "extended real_video_multiscene_quality max rgb loss curve delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("extended real-video quality tether RGB-loss-curve delta must stay below 1e-8")
            if (
                _finite_float(
                    quality_summary.get("max_end_psnr_abs_delta"),
                    "extended real_video_multiscene_quality max end psnr delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("extended real-video quality tether end-PSNR delta must stay below 1e-8")
            if (
                _finite_float(
                    quality_summary.get("min_measured_psnr_gain"),
                    "extended real_video_multiscene_quality min psnr gain",
                    errors,
                )
                <= 0.0
            ):
                errors.append("extended real-video quality tether PSNR gains must be positive")
    if isinstance(evidence.get("real_video_multiscene_media_tether"), dict):
        media_summary = evidence["real_video_multiscene_media_tether"].get("summary", {})
        if isinstance(media_summary, dict):
            if int(media_summary.get("scene_count") or 0) < 3:
                errors.append("real-video multiscene media tether must cover at least three scenes")
            if int(media_summary.get("distinct_youtube_id_count") or 0) < 3:
                errors.append("real-video multiscene media tether must cover source-distinct videos")
            if int(media_summary.get("pair_count") or 0) < 3:
                errors.append("real-video multiscene media tether must cover at least three media pairs")
            if media_summary.get("all_source_videos_exist") is not True:
                errors.append("real-video multiscene media tether source videos must exist")
            if media_summary.get("all_case_rows_pass") is not True:
                errors.append("real-video multiscene media tether case rows must pass")
            if media_summary.get("all_contact_sheets_exist") is not True:
                errors.append("real-video multiscene media tether contact sheets must exist")
            if media_summary.get("all_contact_sheet_pixels_match_cadence") is not True:
                errors.append("real-video multiscene media tether contact sheets must pixel-match cadence")
            if media_summary.get("all_contact_sheet_layouts_valid") is not True:
                errors.append("real-video multiscene media tether contact-sheet layouts must be valid")
            if media_summary.get("all_contact_sheet_metrics_match_payload") is not True:
                errors.append("real-video multiscene media tether contact-sheet MSE must match payload loss")
            if media_summary.get("all_contact_sheet_rows_nontrivial") is not True:
                errors.append("real-video multiscene media tether contact-sheet rows must be nontrivial")
            if media_summary.get("all_final_full_rgb_losses_match_cadence") is not True:
                errors.append("real-video multiscene media tether final full-RGB losses must match cadence")
            if media_summary.get("all_gradient_flags_present") is not True:
                errors.append("real-video multiscene media tether must preserve gradient-flow flags")
            if media_summary.get("all_measured_psnr_improves") is not True:
                errors.append("real-video multiscene media tether measured PSNR must improve")
            if media_summary.get("all_rows_no_overflow") is not True:
                errors.append("real-video multiscene media tether rows must have no overflow")
            if media_summary.get("all_rows_fallback_free") is not True:
                errors.append("real-video multiscene media tether rows must be fallback-free")
            if media_summary.get("all_rows_visibility_stratification_free") is not True:
                errors.append("real-video multiscene media tether rows must avoid visibility stratification")
            if (
                _finite_float(
                    media_summary.get("max_abs_contact_sheet_delta"),
                    "real_video_multiscene_media max contact-sheet delta",
                    errors,
                )
                != 0.0
            ):
                errors.append("real-video multiscene media tether contact-sheet delta must be zero")
            if (
                _finite_float(
                    media_summary.get("max_contact_sheet_target_pred_mse_delta"),
                    "real_video_multiscene_media max contact-sheet target/pred MSE delta",
                    errors,
                )
                > 1.0e-12
            ):
                errors.append("real-video multiscene media tether contact-sheet target/pred MSE must match cadence")
            if (
                _finite_float(
                    media_summary.get("max_contact_sheet_payload_loss_abs_delta"),
                    "real_video_multiscene_media max contact-sheet payload loss delta",
                    errors,
                )
                > CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE
            ):
                errors.append("real-video multiscene media tether contact-sheet MSE must match payload loss")
            if (
                _finite_float(
                    media_summary.get("min_contact_sheet_target_std"),
                    "real_video_multiscene_media min contact-sheet target std",
                    errors,
                )
                <= 1.0e-6
            ):
                errors.append("real-video multiscene media tether contact-sheet target rows must be nontrivial")
            if (
                _finite_float(
                    media_summary.get("min_contact_sheet_pred_std"),
                    "real_video_multiscene_media min contact-sheet pred std",
                    errors,
                )
                <= 1.0e-6
            ):
                errors.append("real-video multiscene media tether contact-sheet pred rows must be nontrivial")
            if (
                _finite_float(
                    media_summary.get("max_abs_loss_curve_delta"),
                    "real_video_multiscene_media max loss curve delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("real-video multiscene media tether loss-curve delta must stay below 1e-8")
            if (
                _finite_float(
                    media_summary.get("max_final_full_rgb_loss_abs_delta"),
                    "real_video_multiscene_media max final RGB loss delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("real-video multiscene media tether final RGB loss delta must stay below 1e-8")
            if (
                _finite_float(
                    media_summary.get("max_measured_vs_cadence_rebuild_ratio"),
                    "real_video_multiscene_media max rebuild ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video multiscene media tether measured rebuild ratio must stay below cadence")
    if isinstance(evidence.get("real_video_multiscene_extended_media_tether"), dict):
        media_summary = evidence["real_video_multiscene_extended_media_tether"].get("summary", {})
        if isinstance(media_summary, dict):
            if int(media_summary.get("scene_count") or 0) < 5:
                errors.append("extended real-video media tether must cover at least five scenes")
            if int(media_summary.get("distinct_youtube_id_count") or 0) < 5:
                errors.append("extended real-video media tether must cover five source-distinct videos")
            if int(media_summary.get("pair_count") or 0) < 5:
                errors.append("extended real-video media tether must cover at least five media pairs")
            if media_summary.get("all_source_videos_exist") is not True:
                errors.append("extended real-video media tether source videos must exist")
            if media_summary.get("all_case_rows_pass") is not True:
                errors.append("extended real-video media tether case rows must pass")
            if media_summary.get("all_contact_sheets_exist") is not True:
                errors.append("extended real-video media tether contact sheets must exist")
            if media_summary.get("all_contact_sheet_pixels_match_cadence") is not True:
                errors.append("extended real-video media tether contact sheets must pixel-match cadence")
            if media_summary.get("all_contact_sheet_layouts_valid") is not True:
                errors.append("extended real-video media tether contact-sheet layouts must be valid")
            if media_summary.get("all_contact_sheet_metrics_match_payload") is not True:
                errors.append("extended real-video media tether contact-sheet MSE must match payload loss")
            if media_summary.get("all_contact_sheet_rows_nontrivial") is not True:
                errors.append("extended real-video media tether contact-sheet rows must be nontrivial")
            if media_summary.get("all_final_full_rgb_losses_match_cadence") is not True:
                errors.append("extended real-video media tether final full-RGB losses must match cadence")
            if media_summary.get("all_gradient_flags_present") is not True:
                errors.append("extended real-video media tether must preserve gradient-flow flags")
            if media_summary.get("all_measured_psnr_improves") is not True:
                errors.append("extended real-video media tether measured PSNR must improve")
            if media_summary.get("all_rows_no_overflow") is not True:
                errors.append("extended real-video media tether rows must have no overflow")
            if media_summary.get("all_rows_fallback_free") is not True:
                errors.append("extended real-video media tether rows must be fallback-free")
            if media_summary.get("all_rows_visibility_stratification_free") is not True:
                errors.append("extended real-video media tether rows must avoid visibility stratification")
            if (
                _finite_float(
                    media_summary.get("max_abs_contact_sheet_delta"),
                    "extended real_video_multiscene_media max contact-sheet delta",
                    errors,
                )
                != 0.0
            ):
                errors.append("extended real-video media tether contact-sheet delta must be zero")
            if (
                _finite_float(
                    media_summary.get("max_contact_sheet_target_pred_mse_delta"),
                    "extended real_video_multiscene_media max contact-sheet target/pred MSE delta",
                    errors,
                )
                > 1.0e-12
            ):
                errors.append("extended real-video media tether contact-sheet target/pred MSE must match cadence")
            if (
                _finite_float(
                    media_summary.get("max_contact_sheet_payload_loss_abs_delta"),
                    "extended real_video_multiscene_media max contact-sheet payload loss delta",
                    errors,
                )
                > CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE
            ):
                errors.append("extended real-video media tether contact-sheet MSE must match payload loss")
            if (
                _finite_float(
                    media_summary.get("min_contact_sheet_target_std"),
                    "extended real_video_multiscene_media min contact-sheet target std",
                    errors,
                )
                <= 1.0e-6
            ):
                errors.append("extended real-video media tether contact-sheet target rows must be nontrivial")
            if (
                _finite_float(
                    media_summary.get("min_contact_sheet_pred_std"),
                    "extended real_video_multiscene_media min contact-sheet pred std",
                    errors,
                )
                <= 1.0e-6
            ):
                errors.append("extended real-video media tether contact-sheet pred rows must be nontrivial")
            if (
                _finite_float(
                    media_summary.get("max_abs_loss_curve_delta"),
                    "extended real_video_multiscene_media max loss curve delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("extended real-video media tether loss-curve delta must stay below 1e-8")
            if (
                _finite_float(
                    media_summary.get("max_final_full_rgb_loss_abs_delta"),
                    "extended real_video_multiscene_media max final RGB loss delta",
                    errors,
                )
                > 1.0e-8
            ):
                errors.append("extended real-video media tether final RGB loss delta must stay below 1e-8")
            if (
                _finite_float(
                    media_summary.get("max_measured_vs_cadence_rebuild_ratio"),
                    "extended real_video_multiscene_media max rebuild ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("extended real-video media tether measured rebuild ratio must stay below cadence")
    if isinstance(evidence.get("real_video_acceptance_envelope"), dict):
        envelope_summary = evidence["real_video_acceptance_envelope"].get("summary", {})
        if isinstance(envelope_summary, dict):
            if int(envelope_summary.get("functional_scene_count") or 0) < 5:
                errors.append("real-video acceptance envelope must cover at least five functional scenes")
            if int(envelope_summary.get("media_scene_count") or 0) < 5:
                errors.append("real-video acceptance envelope must cover at least five media scenes")
            if int(envelope_summary.get("broad_media_distinct_youtube_id_count") or 0) < 10:
                errors.append("real-video acceptance envelope must cover at least 10 broad media sources")
            if int(envelope_summary.get("broad_quality_distinct_youtube_id_count") or 0) < 10:
                errors.append("real-video acceptance envelope must cover at least 10 broad quality sources")
            if envelope_summary.get("all_underlying_verifiers_pass") is not True:
                errors.append("real-video acceptance envelope underlying verifiers must pass")
            if envelope_summary.get("all_functional_rows_pass") is not True:
                errors.append("real-video acceptance envelope functional rows must pass")
            if envelope_summary.get("all_quality_tethers_match") is not True:
                errors.append("real-video acceptance envelope quality tethers must match cadence")
            if envelope_summary.get("all_media_tethers_match") is not True:
                errors.append("real-video acceptance envelope media tethers must match cadence")
            if envelope_summary.get("all_support_churn_zero") is not True:
                errors.append("real-video acceptance envelope support churn must stay zero")
            if envelope_summary.get("all_rebuild_ratios_at_most_half") is not True:
                errors.append("real-video acceptance envelope rebuild ratios must stay at or below half")
            if (
                _finite_float(
                    envelope_summary.get("max_rebuild_ratio"),
                    "real_video_acceptance_envelope max rebuild ratio",
                    errors,
                )
                > 0.5
            ):
                errors.append("real-video acceptance envelope max rebuild ratio must stay at or below 0.5")
            if (
                _finite_float(
                    envelope_summary.get("min_quality_psnr_gain"),
                    "real_video_acceptance_envelope min quality PSNR gain",
                    errors,
                )
                <= 0.0
            ):
                errors.append("real-video acceptance envelope min quality PSNR gain must be positive")
            if int(envelope_summary.get("extended_frame_scaling_expected_timing_failure_count") or 0) != 2:
                errors.append("real-video acceptance envelope must preserve the two expected timing failures")
            if envelope_summary.get("bq4_fresh_process_timing_acceptance_status") != "pass":
                errors.append("real-video acceptance envelope Bq4 fresh-process median timing must pass")
            if int(envelope_summary.get("bq4_fresh_process_post_warmup_pair_count") or 0) < 4:
                errors.append("real-video acceptance envelope Bq4 fresh-process gate must keep post-warmup pairs")
            if int(envelope_summary.get("bq4_fresh_process_no_first_bump_count") or 0) != 0:
                errors.append("real-video acceptance envelope Bq4 fresh-process gate must have zero no-first bumps")
            for key in (
                "bq4_fresh_process_post_warmup_median_no_first_ratio",
                "bq4_fresh_process_post_warmup_median_projective_total_ratio",
                "bq4_fresh_process_post_warmup_median_feature_state_update_ratio",
            ):
                if _finite_float(envelope_summary.get(key), f"real_video_acceptance_envelope {key}", errors) > 1.0:
                    errors.append(f"real-video acceptance envelope {key} must stay at or below cadence")
            if envelope_summary.get("strict_timing_win_claimed") is not False:
                errors.append("real-video acceptance envelope must not claim a strict timing win")
            if envelope_summary.get("fresh_process_median_timing_win_claimed") is not True:
                errors.append("real-video acceptance envelope must claim only the fresh-process median timing win")
            if envelope_summary.get("does_not_prove_completion") is not True:
                errors.append("real-video acceptance envelope must preserve non-completion scope")
    if isinstance(evidence.get("real_video_timing_variance_envelope"), dict):
        timing_summary = evidence["real_video_timing_variance_envelope"].get("summary", {})
        if isinstance(timing_summary, dict):
            if int(timing_summary.get("source_scene_count") or 0) < 5:
                errors.append("real-video timing variance envelope must cover at least five source scenes")
            if timing_summary.get("all_underlying_verifiers_pass") is not True:
                errors.append("real-video timing variance envelope underlying verifiers must pass")
            if timing_summary.get("strict_failed_only_expected_timing") is not True:
                errors.append("real-video timing variance envelope strict failures must stay expected timing misses")
            if int(timing_summary.get("strict_failure_count") or 0) != 2:
                errors.append("real-video timing variance envelope must preserve the two strict timing failures")
            if timing_summary.get("all_timing_miss_pairs_cache_clean") is not True:
                errors.append("real-video timing variance envelope timing misses must keep cache state clean")
            if timing_summary.get("all_timing_miss_pairs_support_clean") is not True:
                errors.append("real-video timing variance envelope timing misses must keep support state clean")
            if timing_summary.get("all_cache_support_clean") is not True:
                errors.append("real-video timing variance envelope all cache/support rows must stay clean")
            if int(timing_summary.get("workload_explains_render_forward_miss_count") or 0) != 0:
                errors.append("real-video timing variance envelope must not explain misses by tile workload changes")
            if timing_summary.get("all_no_first_misses_tile_stats_identical") is not True:
                errors.append("real-video timing variance envelope no-first misses must keep tile stats identical")
            if timing_summary.get("all_render_forward_misses_tile_stats_identical") is not True:
                errors.append("real-video timing variance envelope render-forward misses must keep tile stats identical")
            if timing_summary.get("all_no_first_misses_single_spike_driven") is not True:
                errors.append("real-video timing variance envelope no-first misses must remain single-spike driven")
            if (
                _finite_float(
                    timing_summary.get("drop_spike_render_forward_ratio"),
                    "real_video_timing_variance_envelope drop_spike_render_forward_ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video timing variance envelope drop-spike render-forward ratio must stay below cadence")
            if timing_summary.get("bq4_traced_spike_reproduced") is not False:
                errors.append("real-video timing variance envelope Bq4 traced spike must remain unreproduced")
            if (
                _finite_float(
                    timing_summary.get("bq4_rerun_max_no_first_ratio"),
                    "real_video_timing_variance_envelope bq4_rerun_max_no_first_ratio",
                    errors,
                )
                >= 1.0
            ):
                errors.append("real-video timing variance envelope Bq4 rerun max no-first ratio must stay below cadence")
            if int(timing_summary.get("bq4_repeat_no_first_spike_count") or 0) != 0:
                errors.append("real-video timing variance envelope Bq4 repeat no-first spike count must stay zero")
            if int(timing_summary.get("bq4_sequence_no_first_bump_count") or 0) != 0:
                errors.append("real-video timing variance envelope Bq4 sequence no-first bump count must stay zero")
            if int(timing_summary.get("bq4_policy_projective_bump_count") or 0) < 1:
                errors.append("real-video timing variance envelope Bq4 policy order must preserve projective bumps")
            if timing_summary.get("fresh_process_timing_acceptance_status") != "pass":
                errors.append("real-video timing variance envelope fresh-process acceptance must pass")
            if int(timing_summary.get("fresh_process_post_warmup_pair_count") or 0) < 4:
                errors.append("real-video timing variance envelope fresh-process gate must keep post-warmup pairs")
            for key in (
                "fresh_process_median_no_first_ratio",
                "fresh_process_median_projective_total_ratio",
                "fresh_process_median_feature_state_update_ratio",
            ):
                if (
                    _finite_float(
                        timing_summary.get(key),
                        f"real_video_timing_variance_envelope {key}",
                        errors,
                    )
                    > 1.0
                ):
                    errors.append(f"real-video timing variance envelope {key} must stay at or below cadence")
            if timing_summary.get("strict_timing_win_claimed") is not False:
                errors.append("real-video timing variance envelope must not claim a strict timing win")
            if timing_summary.get("does_not_prove_completion") is not True:
                errors.append("real-video timing variance envelope must preserve non-completion scope")
    if isinstance(evidence.get("real_video_compiled_adjoint_replacement"), dict):
        compiled_summary = evidence["real_video_compiled_adjoint_replacement"].get("summary", {})
        if isinstance(compiled_summary, dict):
            if compiled_summary.get("final_compiled_adjoint_replacement_accepted") is not True:
                errors.append("compiled-adjoint replacement must be accepted")
            if int(compiled_summary.get("compiled_trainer_replacement_gap") or 0) != 0:
                errors.append("compiled-adjoint replacement gap must be zero")
            if compiled_summary.get("source_contract_checks_pass") is not True:
                errors.append("compiled-adjoint replacement source contract must pass")
            if compiled_summary.get("all_cases_projective_interval_main_path") is not True:
                errors.append("compiled-adjoint replacement cases must use projective interval main path")
            if compiled_summary.get("all_cases_gradient_flags_present") is not True:
                errors.append("compiled-adjoint replacement cases must preserve renderer gradients")
            if int(compiled_summary.get("case_payload_count") or 0) < 20:
                errors.append("compiled-adjoint replacement must retain at least twenty case payloads")
    if isinstance(evidence.get("real_video_acceptance_envelope"), dict) and isinstance(
        evidence.get("real_video_timing_variance_envelope"),
        dict,
    ):
        envelope_summary = evidence["real_video_acceptance_envelope"].get("summary", {})
        timing_summary = evidence["real_video_timing_variance_envelope"].get("summary", {})
        if isinstance(envelope_summary, dict) and isinstance(timing_summary, dict):
            if (
                envelope_summary.get("bq4_fresh_process_timing_acceptance_status")
                != timing_summary.get("fresh_process_timing_acceptance_status")
            ):
                errors.append(
                    "Bq4 fresh-process timing status must match across acceptance and variance envelopes"
                )
            if int(envelope_summary.get("bq4_fresh_process_post_warmup_pair_count") or 0) != int(
                timing_summary.get("fresh_process_post_warmup_pair_count") or 0
            ):
                errors.append(
                    "Bq4 fresh-process post-warmup pair count must match across acceptance and variance envelopes"
                )
            for envelope_key, timing_key, label in (
                (
                    "bq4_fresh_process_post_warmup_median_no_first_ratio",
                    "fresh_process_median_no_first_ratio",
                    "Bq4 fresh-process median no-first ratio",
                ),
                (
                    "bq4_fresh_process_post_warmup_median_projective_total_ratio",
                    "fresh_process_median_projective_total_ratio",
                    "Bq4 fresh-process median projective-total ratio",
                ),
                (
                    "bq4_fresh_process_post_warmup_median_feature_state_update_ratio",
                    "fresh_process_median_feature_state_update_ratio",
                    "Bq4 fresh-process median feature-state-update ratio",
                ),
            ):
                _require_close(
                    envelope_summary.get(envelope_key),
                    timing_summary.get(timing_key),
                    label,
                    errors,
                )
    if isinstance(evidence.get("shared_work"), dict):
        shared_summary = evidence["shared_work"].get("summary", {})
        if isinstance(shared_summary, dict):
            if _finite_float(shared_summary.get("orbit_payload_growth_ratio"), "shared_work orbit_payload_growth_ratio", errors) >= 0.20:
                errors.append("orbit payload growth ratio must stay below 0.20")
            if _finite_float(shared_summary.get("trained_shared_to_replay_interval_growth_ratio"), "shared_work trained_shared_to_replay_interval_growth_ratio", errors) >= 0.25:
                errors.append("trained shared/replay interval growth ratio must stay below 0.25")
            if _finite_float(shared_summary.get("max_trained_final_backward_ms_ratio"), "shared_work max_trained_final_backward_ms_ratio", errors) >= 0.25:
                errors.append("trained backward ratio must stay below 0.25")
            if _finite_float(shared_summary.get("exposure_mixed_fallback_max_grad_rel_error"), "shared_work exposure_mixed_fallback_max_grad_rel_error", errors) > 5.0e-3:
                errors.append("mixed fallback gradient rel error must stay below 5e-3")

    try:
        expected_summary = summarize(requirements, evidence)
        for key in expected_summary:
            _assert_summary_close(summary, expected_summary, key, errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")

    if summary.get("is_goal_complete") is not False:
        errors.append("summary is_goal_complete must be false until a requirement-level completion audit passes")
    return errors


def assert_projective_goal_progress_audit(report: dict[str, Any]) -> None:
    errors = verify_projective_goal_progress_audit(report)
    if errors:
        raise AssertionError("projective goal progress audit failed:\n- " + "\n- ".join(errors))


def verify_projective_goal_progress_current_acceptance(
    saved_report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> list[str]:
    """Check that a saved goal-progress report still matches current inputs."""

    errors = [f"saved report: {error}" for error in verify_projective_goal_progress_audit(saved_report)]
    current = run_report() if current_report is None else current_report
    errors.extend(f"current input report: {error}" for error in verify_projective_goal_progress_audit(current))
    if errors:
        return errors
    for key in (
        "benchmark",
        "status",
        "goal",
        "meta_goal",
        "key_math",
        "theory",
        "evidence",
        "requirements",
        "summary",
    ):
        _compare_current_value(saved_report.get(key), current.get(key), key, errors)
    return errors


def assert_projective_goal_progress_current_acceptance(
    saved_report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> None:
    errors = verify_projective_goal_progress_current_acceptance(saved_report, current_report=current_report)
    if errors:
        raise AssertionError("projective goal-progress current-input acceptance failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Goal Progress Audit",
        "",
        "This audit maps the active thread objective to current saved evidence. It is intentionally not a completion claim.",
        "",
        "## Memory Contract",
        "",
        "```text",
        f"goal       {report['goal']}",
        f"meta-goal  {report['meta_goal']}",
        f"key math   {report['key_math']}",
        f"theory     {report['theory']}",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Requirements",
        "",
        "| id | status | statement |",
        "| --- | --- | --- |",
    ]
    for row in report["requirements"]:
        lines.append(f"| {row['id']} | {row['status']} | {row['statement']} |")
    lines.extend(
        [
            "",
            "## Remaining Gaps",
            "",
        ]
    )
    for row in report["requirements"]:
        if row.get("status") == "open":
            for gap in row.get("gaps", []):
                lines.append(f"- {gap}")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "```json",
            json.dumps(report["evidence"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, default=None)
    parser.add_argument("--bundle-invariance-report", type=Path, default=DEFAULT_BUNDLE_INVARIANCE_REPORT)
    parser.add_argument("--bundle-gradient-report", type=Path, default=DEFAULT_BUNDLE_GRADIENT_REPORT)
    parser.add_argument("--camera-family-report", type=Path, default=DEFAULT_CAMERA_FAMILY_REPORT)
    parser.add_argument("--camera-family-2d-report", type=Path, default=DEFAULT_CAMERA_FAMILY_2D_REPORT)
    parser.add_argument(
        "--camera-family-2d-metal-lowering-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_METAL_LOWERING_REPORT,
    )
    parser.add_argument(
        "--camera-family-2d-metal-chain-rule-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_METAL_CHAIN_RULE_REPORT,
    )
    parser.add_argument(
        "--camera-family-2d-materialized-batch-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_MATERIALIZED_BATCH_REPORT,
    )
    parser.add_argument(
        "--camera-family-2d-native-eval-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_NATIVE_EVAL_REPORT,
    )
    parser.add_argument(
        "--camera-family-2d-native-interval-forward-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_FORWARD_REPORT,
    )
    parser.add_argument(
        "--camera-family-2d-native-interval-backward-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_NATIVE_INTERVAL_BACKWARD_REPORT,
    )
    parser.add_argument(
        "--camera-family-2d-tile-order-reuse-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_REUSE_REPORT,
    )
    parser.add_argument(
        "--camera-family-2d-tile-order-strata-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_TILE_ORDER_STRATA_REPORT,
    )
    parser.add_argument(
        "--camera-family-2d-active-set-strata-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_ACTIVE_SET_STRATA_REPORT,
    )
    parser.add_argument(
        "--real-active-set-distribution-report",
        type=Path,
        default=DEFAULT_REAL_ACTIVE_SET_DISTRIBUTION_REPORT,
    )
    parser.add_argument("--camera-family-shared-work-report", type=Path, default=DEFAULT_CAMERA_FAMILY_SHARED_WORK_REPORT)
    parser.add_argument(
        "--camera-family-2d-shared-work-report",
        type=Path,
        default=DEFAULT_CAMERA_FAMILY_2D_SHARED_WORK_REPORT,
    )
    parser.add_argument("--trainer-interval-report", type=Path, default=DEFAULT_INTERVAL_TRAINER_REPORT)
    parser.add_argument("--trainer-real-video-report", type=Path, default=DEFAULT_REAL_VIDEO_TRAINER_REPORT)
    parser.add_argument(
        "--real-video-guarded-support-matrix-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_GUARDED_SUPPORT_MATRIX_REPORT,
    )
    parser.add_argument(
        "--real-video-multiscene-trainer-matrix-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_MULTISCENE_TRAINER_MATRIX_REPORT,
    )
    parser.add_argument(
        "--real-video-multiscene-extended-trainer-matrix-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_TRAINER_MATRIX_REPORT,
    )
    parser.add_argument(
        "--real-video-multiscene-frame-scaling-matrix-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_MULTISCENE_FRAME_SCALING_MATRIX_REPORT,
    )
    parser.add_argument(
        "--real-video-multiscene-extended-frame-scaling-diagnostic-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_FRAME_SCALING_DIAGNOSTIC_REPORT,
    )
    parser.add_argument(
        "--real-video-multiscene-quality-tether-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_MULTISCENE_QUALITY_TETHER_REPORT,
    )
    parser.add_argument(
        "--real-video-multiscene-extended-quality-tether-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_QUALITY_TETHER_REPORT,
    )
    parser.add_argument(
        "--real-video-multiscene-media-tether-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_MULTISCENE_MEDIA_TETHER_REPORT,
    )
    parser.add_argument(
        "--real-video-multiscene-extended-media-tether-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_MULTISCENE_EXTENDED_MEDIA_TETHER_REPORT,
    )
    parser.add_argument(
        "--real-video-acceptance-envelope-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_ACCEPTANCE_ENVELOPE_REPORT,
    )
    parser.add_argument(
        "--real-video-timing-variance-envelope-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_TIMING_VARIANCE_ENVELOPE_REPORT,
    )
    parser.add_argument(
        "--real-video-compiled-adjoint-replacement-report",
        type=Path,
        default=DEFAULT_REAL_VIDEO_COMPILED_ADJOINT_REPLACEMENT_REPORT,
    )
    parser.add_argument("--shared-work-report", type=Path, default=DEFAULT_SHARED_WORK_REPORT)
    parser.add_argument(
        "--verify-current-inputs",
        action="store_true",
        help="also require the saved goal-progress report to match a fresh audit of the current default inputs",
    )
    args = parser.parse_args()

    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        if args.verify_current_inputs:
            assert_projective_goal_progress_current_acceptance(report)
            print(f"verified {args.verify_report} against current inputs")
        else:
            assert_projective_goal_progress_audit(report)
            print(f"verified {args.verify_report}")
        return

    if args.verify_current_inputs:
        report = _load_json(args.out_dir / "summary.json")
        assert_projective_goal_progress_current_acceptance(report)
        print(f"verified {args.out_dir / 'summary.json'} against current inputs")
        return

    report = run_report(
        bundle_invariance_report=args.bundle_invariance_report,
        bundle_gradient_report=args.bundle_gradient_report,
        camera_family_report=args.camera_family_report,
        camera_family_2d_report=args.camera_family_2d_report,
        camera_family_2d_metal_lowering_report=args.camera_family_2d_metal_lowering_report,
        camera_family_2d_metal_chain_rule_report=args.camera_family_2d_metal_chain_rule_report,
        camera_family_2d_materialized_batch_report=args.camera_family_2d_materialized_batch_report,
        camera_family_2d_native_eval_report=args.camera_family_2d_native_eval_report,
        camera_family_2d_native_interval_forward_report=args.camera_family_2d_native_interval_forward_report,
        camera_family_2d_native_interval_backward_report=args.camera_family_2d_native_interval_backward_report,
        camera_family_2d_tile_order_reuse_report=args.camera_family_2d_tile_order_reuse_report,
        camera_family_2d_tile_order_strata_report=args.camera_family_2d_tile_order_strata_report,
        camera_family_2d_active_set_strata_report=args.camera_family_2d_active_set_strata_report,
        real_active_set_distribution_report=args.real_active_set_distribution_report,
        camera_family_shared_work_report=args.camera_family_shared_work_report,
        camera_family_2d_shared_work_report=args.camera_family_2d_shared_work_report,
        trainer_interval_report=args.trainer_interval_report,
        trainer_real_video_report=args.trainer_real_video_report,
        real_video_guarded_support_matrix_report=args.real_video_guarded_support_matrix_report,
        real_video_multiscene_trainer_matrix_report=args.real_video_multiscene_trainer_matrix_report,
        real_video_multiscene_extended_trainer_matrix_report=(
            args.real_video_multiscene_extended_trainer_matrix_report
        ),
        real_video_multiscene_frame_scaling_matrix_report=args.real_video_multiscene_frame_scaling_matrix_report,
        real_video_multiscene_extended_frame_scaling_diagnostic_report=(
            args.real_video_multiscene_extended_frame_scaling_diagnostic_report
        ),
        real_video_multiscene_quality_tether_report=args.real_video_multiscene_quality_tether_report,
        real_video_multiscene_extended_quality_tether_report=(
            args.real_video_multiscene_extended_quality_tether_report
        ),
        real_video_multiscene_media_tether_report=args.real_video_multiscene_media_tether_report,
        real_video_multiscene_extended_media_tether_report=args.real_video_multiscene_extended_media_tether_report,
        real_video_acceptance_envelope_report=args.real_video_acceptance_envelope_report,
        real_video_timing_variance_envelope_report=args.real_video_timing_variance_envelope_report,
        real_video_compiled_adjoint_replacement_report=args.real_video_compiled_adjoint_replacement_report,
        shared_work_report=args.shared_work_report,
    )
    assert_projective_goal_progress_audit(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
