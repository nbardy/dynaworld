"""Independent verifier for the WorldFoam adaptive M3/M5 CPU ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from research_experiments.world_foam_lane2.adaptive_material_basis_selection import (
    DEFAULT_INTERVALS,
    HELDOUT_INTERVALS,
    M3,
    M5,
    SCHEMA_VERSION,
    SEEDS,
    SELECTION_INTERVALS,
    SUITE_ID,
    TARGETS,
    _source_hashes,
)


PAPER_STEPS = 500
PAPER_LEARNING_RATE = 0.04
PAPER_REFINEMENT_STEPS = 30


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _close(left: float, right: float, *, tolerance: float = 1.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=1.0e-15)


def _mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    return sum(values) / len(values)


def _candidate_index(
    rows: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    expected_targets = {target.name: target for target in TARGETS}
    for row_index, row in enumerate(rows):
        try:
            key = (str(row["target"]), int(row["seed"]), str(row["mode"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"candidate row {row_index} has an invalid key")
            continue
        if key in result:
            errors.append(f"duplicate candidate key: {key}")
            continue
        if key[0] not in expected_targets:
            errors.append(f"candidate row has unknown target: {key[0]}")
        elif row.get("target_family") != expected_targets[key[0]].family:
            errors.append(f"candidate target family mismatch: {key}")
        if key[1] not in SEEDS:
            errors.append(f"candidate row has unexpected seed: {key}")
        if key[2] not in {M3.name, M5.name}:
            errors.append(f"candidate row has unexpected mode: {key}")
        if row.get("serialized_material_scalars") != 6:
            errors.append(f"candidate serialized scalar count mismatch: {key}")
        if row.get("serialized_material_bytes_float32") != 24:
            errors.append(f"candidate serialized byte count mismatch: {key}")
        if row.get("basis_tag_bits") != 1:
            errors.append(f"candidate basis tag mismatch: {key}")
        splits = row.get("splits")
        if not isinstance(splits, Mapping):
            errors.append(f"candidate splits missing: {key}")
        else:
            for split_name in ("train", "selection", "heldout"):
                metrics = splits.get(split_name)
                if not isinstance(metrics, Mapping):
                    errors.append(f"candidate split missing: {key} {split_name}")
                    continue
                for metric_name in (
                    "loss",
                    "beta_mse",
                    "rgb_mse",
                    "max_beta_abs_error",
                    "max_rgb_abs_error",
                ):
                    value = metrics.get(metric_name)
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        errors.append(
                            f"candidate metric is not finite: {key} {split_name}.{metric_name}"
                        )
                if all(
                    isinstance(metrics.get(name), (int, float))
                    for name in ("loss", "beta_mse", "rgb_mse")
                ) and not _close(
                    float(metrics["loss"]),
                    float(metrics["beta_mse"]) + float(metrics["rgb_mse"]),
                ):
                    errors.append(f"candidate split loss decomposition mismatch: {key} {split_name}")
        result[key] = row
    expected_keys = {
        (target.name, seed, mode.name)
        for target in TARGETS
        for seed in SEEDS
        for mode in (M3, M5)
    }
    missing = expected_keys - set(result)
    extra = set(result) - expected_keys
    if missing:
        errors.append(f"candidate rows missing {len(missing)} required keys")
    if extra:
        errors.append(f"candidate rows contain {len(extra)} extra keys")
    return result


def _verify_selections(
    rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[tuple[str, int, str], Mapping[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    target_family = {target.name: target.family for target in TARGETS}
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    recomputed: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        try:
            key = (str(row["target"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"selection row {row_index} has an invalid key")
            continue
        if key in indexed:
            errors.append(f"duplicate selection key: {key}")
            continue
        indexed[key] = row
        if key[0] not in target_family or key[1] not in SEEDS:
            errors.append(f"unexpected selection key: {key}")
            continue
        try:
            m3 = candidates[(key[0], key[1], M3.name)]
            m5 = candidates[(key[0], key[1], M5.name)]
        except KeyError:
            continue
        m3_selection = float(m3["splits"]["selection"]["loss"])
        m5_selection = float(m5["splits"]["selection"]["loss"])
        m3_heldout = float(m3["splits"]["heldout"]["loss"])
        m5_heldout = float(m5["splits"]["heldout"]["loss"])
        selected_mode = M3.name if m3_selection <= m5_selection else M5.name
        oracle_mode = M3.name if m3_heldout <= m5_heldout else M5.name
        selected_heldout = m3_heldout if selected_mode == M3.name else m5_heldout
        oracle_heldout = min(m3_heldout, m5_heldout)
        family = target_family[key[0]]
        expected = {
            "target": key[0],
            "target_family": family,
            "seed": key[1],
            "selected_mode": selected_mode,
            "oracle_mode": oracle_mode,
            "selection_correct_for_pure_family": (
                family == "hybrid"
                or selected_mode == (M3.name if family == "m3" else M5.name)
            ),
            "m3_selection_loss": m3_selection,
            "m5_selection_loss": m5_selection,
            "m3_heldout_loss": m3_heldout,
            "m5_heldout_loss": m5_heldout,
            "selected_heldout_loss": selected_heldout,
            "oracle_heldout_loss": oracle_heldout,
            "selection_regret": selected_heldout - oracle_heldout,
        }
        for name, expected_value in expected.items():
            actual = row.get(name)
            if isinstance(expected_value, float):
                if not isinstance(actual, (int, float)) or not _close(actual, expected_value):
                    errors.append(f"selection value mismatch: {key} {name}")
            elif actual != expected_value:
                errors.append(f"selection value mismatch: {key} {name}")
        recomputed.append(expected)
    expected_keys = {(target.name, seed) for target in TARGETS for seed in SEEDS}
    if set(indexed) != expected_keys:
        errors.append("selection row key set mismatch")
    return recomputed


def _recompute_aggregates(
    candidates: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fixed = {
        mode.name: _mean(
            row["splits"]["heldout"]["loss"]
            for row in candidates
            if row["mode"] == mode.name
        )
        for mode in (M3, M5)
    }
    adaptive = _mean(row["selected_heldout_loss"] for row in selections)
    oracle = _mean(row["oracle_heldout_loss"] for row in selections)
    pure = [row for row in selections if row["target_family"] != "hybrid"]
    hybrid = [row for row in selections if row["target_family"] == "hybrid"]
    family_losses: dict[str, dict[str, float]] = {}
    for family in ("m3", "m5", "hybrid"):
        family_rows = [row for row in selections if row["target_family"] == family]
        family_losses[family] = {
            "m3_mean_heldout_loss": _mean(row["m3_heldout_loss"] for row in family_rows),
            "m5_mean_heldout_loss": _mean(row["m5_heldout_loss"] for row in family_rows),
            "adaptive_mean_heldout_loss": _mean(
                row["selected_heldout_loss"] for row in family_rows
            ),
        }
    return {
        "mean_heldout_loss": {
            M3.name: fixed[M3.name],
            M5.name: fixed[M5.name],
            "adaptive_validation_selection": adaptive,
            "heldout_oracle_selection": oracle,
        },
        "adaptive_to_best_fixed_ratio": adaptive / min(fixed.values()),
        "adaptive_to_oracle_ratio": adaptive / max(oracle, 1.0e-30),
        "pure_family_selection_accuracy": _mean(
            1.0 if row["selection_correct_for_pure_family"] else 0.0 for row in pure
        ),
        "selection_oracle_agreement": _mean(
            1.0 if row["selected_mode"] == row["oracle_mode"] else 0.0
            for row in selections
        ),
        "mean_selection_regret": _mean(row["selection_regret"] for row in selections),
        "max_selection_regret": max(float(row["selection_regret"]) for row in selections),
        "hybrid_selected_mode_counts": {
            M3.name: sum(row["selected_mode"] == M3.name for row in hybrid),
            M5.name: sum(row["selected_mode"] == M5.name for row in hybrid),
        },
        "family_losses": family_losses,
    }


def _compare_nested(
    actual: Any,
    expected: Any,
    *,
    path: str,
    errors: list[str],
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            errors.append(f"{path} key set mismatch")
            return
        for key, value in expected.items():
            _compare_nested(actual[key], value, path=f"{path}.{key}", errors=errors)
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not _close(actual, expected):
            errors.append(f"{path} mismatch")
        return
    if actual != expected:
        errors.append(f"{path} mismatch")


def verify_report(
    report: Mapping[str, Any],
    *,
    summary_path: Path,
    require_current_source: bool = True,
) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if report.get("suite_id") != SUITE_ID:
        errors.append("suite_id mismatch")
    if report.get("accepted") is not True:
        errors.append("report is not accepted")
    if report.get("device") != "cpu" or report.get("dtype") != "float64":
        errors.append("device/dtype contract mismatch")
    if report.get("timing_is_paper_evidence") is not False:
        errors.append("diagnostic CPU timing must not be paper evidence")
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping):
        errors.append("protocol is missing")
    else:
        if protocol.get("steps") != PAPER_STEPS:
            errors.append("paper protocol steps mismatch")
        if not _close(protocol.get("learning_rate", math.nan), PAPER_LEARNING_RATE):
            errors.append("paper protocol learning rate mismatch")
        if protocol.get("refinement_steps") != PAPER_REFINEMENT_STEPS:
            errors.append("paper protocol refinement steps mismatch")
        if protocol.get("train_intervals") != [list(value) for value in DEFAULT_INTERVALS]:
            errors.append("train interval split mismatch")
        if protocol.get("selection_intervals") != [list(value) for value in SELECTION_INTERVALS]:
            errors.append("selection interval split mismatch")
        if protocol.get("heldout_intervals") != [list(value) for value in HELDOUT_INTERVALS]:
            errors.append("heldout interval split mismatch")
        if report.get("protocol_sha256") != _sha256_json(protocol):
            errors.append("protocol_sha256 mismatch")

    candidate_rows = report.get("candidate_rows")
    selection_rows = report.get("selection_rows")
    if not isinstance(candidate_rows, list):
        errors.append("candidate_rows must be a list")
        candidate_rows = []
    if not isinstance(selection_rows, list):
        errors.append("selection_rows must be a list")
        selection_rows = []
    candidates = _candidate_index(candidate_rows, errors)
    recomputed_selections = _verify_selections(selection_rows, candidates, errors)
    if len(candidates) == len(TARGETS) * len(SEEDS) * 2 and len(recomputed_selections) == len(TARGETS) * len(SEEDS):
        aggregates = _recompute_aggregates(candidate_rows, recomputed_selections)
        _compare_nested(report.get("aggregates"), aggregates, path="aggregates", errors=errors)
        losses = aggregates["mean_heldout_loss"]
        family = aggregates["family_losses"]
        expected_checks = {
            "expected_candidate_row_count": True,
            "expected_selection_row_count": True,
            "all_metrics_finite": True,
            "split_interval_lists_are_distinct": len(
                set(DEFAULT_INTERVALS) | set(SELECTION_INTERVALS) | set(HELDOUT_INTERVALS)
            )
            == len(DEFAULT_INTERVALS) + len(SELECTION_INTERVALS) + len(HELDOUT_INTERVALS),
            "m3_m5_matched_24_byte_payload": True,
            "pure_family_selection_accuracy_at_least_95pct": float(
                aggregates["pure_family_selection_accuracy"]
            )
            >= 0.95,
            "adaptive_beats_or_matches_best_fixed_mean": float(
                losses["adaptive_validation_selection"]
            )
            <= min(float(losses[M3.name]), float(losses[M5.name])),
            "adaptive_within_25pct_of_heldout_oracle": float(
                aggregates["adaptive_to_oracle_ratio"]
            )
            <= 1.25,
            "m3_wins_m3_family": float(family["m3"]["m3_mean_heldout_loss"])
            < float(family["m3"]["m5_mean_heldout_loss"]),
            "m5_wins_m5_family": float(family["m5"]["m5_mean_heldout_loss"])
            < float(family["m5"]["m3_mean_heldout_loss"]),
        }
        if report.get("checks") != expected_checks:
            errors.append("checks mismatch")
        if not all(expected_checks.values()):
            errors.append("one or more independently recomputed acceptance checks failed")

    source_hashes = report.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        errors.append("source_sha256 is missing")
    elif require_current_source and dict(source_hashes) != _source_hashes():
        errors.append("current source hashes do not match the artifact")

    assets = report.get("assets")
    if not isinstance(assets, Mapping):
        errors.append("assets manifest is missing")
    else:
        expected_names = {
            "worldfoam_adaptive_material_basis.svg",
            "adaptive_material_basis_table.md",
            "adaptive_material_basis_table.tex",
        }
        if set(assets) != expected_names:
            errors.append("asset name set mismatch")
        for name, receipt in assets.items():
            path = summary_path.parent / ("figures" if name.endswith(".svg") else "") / name
            if not path.is_file():
                errors.append(f"asset is missing: {name}")
                continue
            if not isinstance(receipt, Mapping):
                errors.append(f"asset receipt is invalid: {name}")
                continue
            if receipt.get("sha256") != _sha256_file(path):
                errors.append(f"asset sha256 mismatch: {name}")
            if receipt.get("bytes") != path.stat().st_size:
                errors.append(f"asset byte count mismatch: {name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--allow-source-drift", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    errors = verify_report(
        report,
        summary_path=args.summary,
        require_current_source=not args.allow_source_drift,
    )
    payload = {
        "accepted": not errors,
        "summary": str(args.summary),
        "errors": errors,
        "candidate_row_count": len(report.get("candidate_rows", [])),
        "selection_row_count": len(report.get("selection_rows", [])),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
