"""Verify the saved WorldFoam partial-chord material-fit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPECTED_MODES = (
    "M0_P0_CONSTANT",
    "M1_P0_AFFINE_RGB",
    "M2_POSITIVE_BERNSTEIN_P1",
    "M3_POSITIVE_BERNSTEIN_P2",
    "M4_LOG_P1",
    "M5_CONVEX_LOG_P2",
)
DIRECT_TARGET = "positive_p2_hump"
LOG_TARGET = "convex_log_p2_hump"


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-18)


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = (
        "finite_element_material_fit.py",
        "finite_element_material_transfer.py",
    )
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in names
    }


def verify_artifact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if payload.get("schema_version") != 2:
        failures.append("schema_version must be 2")
    if not payload.get("passed"):
        failures.append("artifact passed flag is false")
    if payload.get("source_sha256") != _source_hashes():
        failures.append("source hashes do not match the current material gate")

    seeds = tuple(int(seed) for seed in payload.get("seeds", ()))
    rows = payload.get("rows", ())
    expected_row_count = 2 * len(EXPECTED_MODES) * len(seeds)
    if not seeds:
        failures.append("at least one seed is required")
    if len(rows) != expected_row_count:
        failures.append(
            f"expected {expected_row_count} rows, found {len(rows)}"
        )

    keys = {
        (row.get("target"), row.get("mode"), int(row.get("seed", -1)))
        for row in rows
    }
    expected_keys = {
        (target, mode, seed)
        for target in (DIRECT_TARGET, LOG_TARGET)
        for mode in EXPECTED_MODES
        for seed in seeds
    }
    if keys != expected_keys:
        failures.append("row target/mode/seed matrix is incomplete or duplicated")

    numeric_keys = (
        "loss",
        "train_loss",
        "heldout_loss",
        "beta_mse",
        "rgb_mse",
        "max_beta_abs_error",
        "max_rgb_abs_error",
    )
    for index, row in enumerate(rows):
        if not all(
            math.isfinite(float(row.get(key, math.nan))) for key in numeric_keys
        ):
            failures.append(f"row {index} contains a non-finite metric")
        if not _close(row.get("loss", math.nan), row.get("heldout_loss", math.nan)):
            failures.append(f"row {index} canonical loss is not heldout loss")

    train_intervals = {tuple(interval) for interval in payload.get("train_intervals", ())}
    heldout_intervals = {
        tuple(interval) for interval in payload.get("heldout_intervals", ())
    }
    if not train_intervals or not heldout_intervals:
        failures.append("train and heldout interval sets must both be nonempty")
    if train_intervals & heldout_intervals:
        failures.append("train and heldout interval sets overlap")

    medians = payload.get("medians", {})
    for target in (DIRECT_TARGET, LOG_TARGET):
        for mode in EXPECTED_MODES:
            selected = [
                float(row["heldout_loss"])
                for row in rows
                if row.get("target") == target and row.get("mode") == mode
            ]
            saved = (
                medians.get(target, {}).get(mode, {}).get("heldout_loss", math.nan)
            )
            if not selected or not _close(statistics.median(selected), saved):
                failures.append(f"median mismatch for {target}/{mode}")

    try:
        direct = medians[DIRECT_TARGET]
        log = medians[LOG_TARGET]
        direct_m3 = float(direct["M3_POSITIVE_BERNSTEIN_P2"]["heldout_loss"])
        direct_m5 = float(direct["M5_CONVEX_LOG_P2"]["heldout_loss"])
        log_m3 = float(log["M3_POSITIVE_BERNSTEIN_P2"]["heldout_loss"])
        log_m5 = float(log["M5_CONVEX_LOG_P2"]["heldout_loss"])
        if not direct_m3 <= 0.01 * direct_m5:
            failures.append("M3 does not beat M5 100x on held-out positive-P2")
        if not log_m5 <= 0.01 * log_m3:
            failures.append("M5 does not beat M3 100x on held-out log-P2")
    except (KeyError, TypeError, ValueError):
        failures.append("missing M3/M5 heldout medians")

    scalars = payload.get("material_scalars", {})
    if scalars.get("M3_POSITIVE_BERNSTEIN_P2") != scalars.get(
        "M5_CONVEX_LOG_P2"
    ):
        failures.append("M3 and M5 are not serialized-byte matched")
    promotion = payload.get("promotion", {})
    if promotion.get("winner") is not None:
        failures.append("synthetic gate must not declare a universal winner")
    if promotion.get("eligible_for_native_4d_integration") is not False:
        failures.append("native-4D integration must remain gated")

    result = {
        "artifact": str(path),
        "verified": not failures,
        "row_count": len(rows),
        "seed_count": len(seeds),
        "failures": failures,
    }
    if failures:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path(
            "artifacts/foundation_gates/"
            "worldfoam_material_value_fit_cpu_20260727.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(verify_artifact(args.artifact), sort_keys=True))


if __name__ == "__main__":
    main()
