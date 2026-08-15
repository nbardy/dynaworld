"""Exact receipt contract for bounded real-native G4 route smokes.

These receipts are runtime capability evidence, never paper rows.  A smoke
must use the sealed public target/ray provider and the production route ABI,
but it intentionally consumes one target pixel so that native wiring can be
attested without launching a publication-scale ablation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


RUNTIME_SMOKE_SCHEMA_VERSION = 1
RUNTIME_SMOKE_KIND = "public-quality-route-runtime-smoke-v1"
RUNTIME_SMOKE_STATUS = "runtime_verified"
RUNTIME_SMOKE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "route",
        "lane",
        "execution_mode",
        "backend",
        "real_native",
        "native_extension_attested",
        "fake_native",
        "source_only",
        "procedural_target",
        "public_target_provider",
        "paper_evidence_eligible",
        "smoke",
        "device",
        "optimizer_steps",
        "train_render_count",
        "backward_passes",
        "optimizer_updates",
        "heldout_render_count",
        "finite_train_loss",
        "finite_gradients",
        "parameter_update_observed",
        "finite_heldout_rgb",
        "target_pixels",
        "rasterized_pixels",
        "parameter_count",
        "parameter_bytes",
        "sampled_peak_process_rss_bytes",
        "sampled_peak_mps_driver_allocated_bytes",
        "elapsed_s",
        "representation_sha256_before",
        "representation_sha256_after",
        "executor_receipt_sha256",
        "native_receipt_sha256",
        "source_receipt_sha256",
    }
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_public_quality_runtime_smoke(
    receipt: Mapping[str, Any],
    *,
    context: Any,
) -> dict[str, Any]:
    """Validate one common route receipt against its sealed row context."""

    if not isinstance(receipt, Mapping):
        raise TypeError("public-quality runtime smoke receipt must be a mapping")
    value = dict(receipt)
    if set(value) != set(RUNTIME_SMOKE_KEYS):
        missing = sorted(set(RUNTIME_SMOKE_KEYS) - set(value))
        unexpected = sorted(set(value) - set(RUNTIME_SMOKE_KEYS))
        raise ValueError(
            "public-quality runtime smoke keys changed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    exact = {
        "schema_version": RUNTIME_SMOKE_SCHEMA_VERSION,
        "kind": RUNTIME_SMOKE_KIND,
        "status": RUNTIME_SMOKE_STATUS,
        "route": context.request.route,
        "lane": context.route_spec["lane"],
        "execution_mode": context.route_spec["execution_mode"],
        "backend": context.route_spec["backend"],
        "real_native": True,
        "native_extension_attested": False,
        "fake_native": False,
        "source_only": False,
        "procedural_target": False,
        "public_target_provider": True,
        "paper_evidence_eligible": False,
        "smoke": True,
        "device": "mps",
        "optimizer_steps": 1,
        "train_render_count": 1,
        "backward_passes": 1,
        "optimizer_updates": 1,
        "heldout_render_count": 1,
        "finite_train_loss": True,
        "finite_gradients": True,
        "parameter_update_observed": True,
        "finite_heldout_rgb": True,
        "target_pixels": 1,
    }
    drift = [key for key, expected in exact.items() if value.get(key) != expected]
    if drift:
        raise ValueError(
            "public-quality runtime smoke differs from the exact production contract: "
            + ", ".join(drift)
        )
    for key in (
        "rasterized_pixels",
        "parameter_count",
        "parameter_bytes",
    ):
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError(f"runtime smoke {key} must be a positive integer")
    for key in (
        "sampled_peak_process_rss_bytes",
        "sampled_peak_mps_driver_allocated_bytes",
    ):
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"runtime smoke {key} must be a nonnegative integer")
    elapsed = value["elapsed_s"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
    ):
        raise ValueError("runtime smoke elapsed_s must be finite and positive")
    for key in (
        "representation_sha256_before",
        "representation_sha256_after",
        "executor_receipt_sha256",
        "native_receipt_sha256",
        "source_receipt_sha256",
    ):
        if not _sha256(value[key]):
            raise ValueError(f"runtime smoke {key} is not a SHA-256 digest")
    if value["representation_sha256_before"] == value["representation_sha256_after"]:
        raise ValueError("runtime smoke did not observe a parameter update")
    return value


__all__ = [
    "RUNTIME_SMOKE_KEYS",
    "RUNTIME_SMOKE_KIND",
    "RUNTIME_SMOKE_SCHEMA_VERSION",
    "RUNTIME_SMOKE_STATUS",
    "canonical_sha256",
    "validate_public_quality_runtime_smoke",
]
