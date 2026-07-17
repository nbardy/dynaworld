"""CodeT/AlphaCode-style candidate selection utilities.

The selector consumes an already-executed candidate/probe matrix. It does not
run Codex and it does not execute tests; those are separate runner concerns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

PASS_STRINGS = {"1", "ok", "pass", "passed", "true", "yes"}
FAIL_STRINGS = {"0", "error", "fail", "failed", "false", "no", "timeout"}


@dataclass(frozen=True)
class Candidate:
    """Normalized view of one generated candidate."""

    candidate_id: str
    visible_pass: bool
    probes: Mapping[str, Any]
    metrics: Mapping[str, Any]
    hidden_pass: bool | None


@dataclass(frozen=True)
class ConsensusSet:
    """Candidates that pass the same generated-probe set."""

    consensus_id: str
    candidate_ids: tuple[str, ...]
    pass_vector: tuple[bool, ...]
    passed_probe_ids: tuple[str, ...]
    score: float
    representative_id: str

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

    @property
    def passed_probe_count(self) -> int:
        return len(self.passed_probe_ids)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in PASS_STRINGS:
            return True
        if normalized in FAIL_STRINGS:
            return False
    return bool(value)


def _hidden_pass(value: Any) -> bool | None:
    if value is None:
        return None
    return _as_bool(value)


def _candidate_id(raw: Mapping[str, Any], index: int) -> str:
    return str(raw.get("candidate_id") or raw.get("id") or f"cand_{index:06d}")


def normalize_candidates(rows: Sequence[Mapping[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for index, raw in enumerate(rows):
        candidates.append(
            Candidate(
                candidate_id=_candidate_id(raw, index),
                visible_pass=_as_bool(raw.get("visible_pass", True)),
                probes=dict(raw.get("probes") or raw.get("probe_results") or {}),
                metrics=dict(raw.get("metrics") or {}),
                hidden_pass=_hidden_pass(raw.get("hidden_pass")),
            )
        )
    return candidates


def infer_probe_ids(candidates: Iterable[Candidate]) -> list[str]:
    probe_ids: set[str] = set()
    for candidate in candidates:
        probe_ids.update(candidate.probes.keys())
    return sorted(probe_ids)


def _pass_vector(candidate: Candidate, probe_ids: Sequence[str]) -> tuple[bool, ...]:
    return tuple(_as_bool(candidate.probes.get(probe_id, False)) for probe_id in probe_ids)


def _support_weight(candidate_count: int, mode: str) -> float:
    if mode == "linear":
        return float(candidate_count)
    if mode == "log":
        return math.log1p(candidate_count)
    if mode == "cap1":
        return 1.0
    if mode != "sqrt":
        raise ValueError(f"unknown candidate_support_weight: {mode}")
    return math.sqrt(candidate_count)


def _metric_key(candidate: Candidate, metric_name: str | None, direction: str) -> tuple[float, str]:
    if not metric_name:
        return (0.0, candidate.candidate_id)
    raw_value = candidate.metrics.get(metric_name)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = math.inf if direction == "min" else -math.inf
    return (value if direction == "min" else -value, candidate.candidate_id)


def _representative(
    candidates: Sequence[Candidate],
    metric_name: str | None,
    metric_direction: str,
) -> Candidate:
    return min(candidates, key=lambda candidate: _metric_key(candidate, metric_name, metric_direction))


def build_consensus_sets(
    candidates: Sequence[Candidate],
    probe_ids: Sequence[str],
    *,
    metric_name: str | None = None,
    metric_direction: str = "min",
    candidate_support_weight: str = "sqrt",
) -> list[ConsensusSet]:
    grouped: dict[tuple[bool, ...], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[_pass_vector(candidate, probe_ids)].append(candidate)

    consensus_sets: list[ConsensusSet] = []
    for index, (pass_vector, group) in enumerate(grouped.items()):
        passed_probe_ids = tuple(
            probe_id for probe_id, passed in zip(probe_ids, pass_vector, strict=True) if passed
        )
        support = _support_weight(len(group), candidate_support_weight)
        score = support * len(passed_probe_ids)
        representative = _representative(group, metric_name, metric_direction)
        consensus_sets.append(
            ConsensusSet(
                consensus_id=f"consensus_{index:03d}",
                candidate_ids=tuple(sorted(candidate.candidate_id for candidate in group)),
                pass_vector=pass_vector,
                passed_probe_ids=passed_probe_ids,
                score=score,
                representative_id=representative.candidate_id,
            )
        )

    return sorted(
        consensus_sets,
        key=lambda consensus: (
            -consensus.score,
            -consensus.passed_probe_count,
            -consensus.candidate_count,
            consensus.representative_id,
        ),
    )


def _candidate_by_id(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def select_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget: int = 1,
    probe_ids: Sequence[str] | None = None,
    visible_only: bool = True,
    metric_name: str | None = None,
    metric_direction: str = "min",
    candidate_support_weight: str = "sqrt",
) -> tuple[list[str], list[ConsensusSet], list[Candidate]]:
    if budget < 1:
        raise ValueError("selection budget must be >= 1")
    candidates = normalize_candidates(rows)
    eligible = [candidate for candidate in candidates if candidate.visible_pass or not visible_only]
    resolved_probe_ids = list(probe_ids) if probe_ids is not None else infer_probe_ids(eligible)
    consensus_sets = build_consensus_sets(
        eligible,
        resolved_probe_ids,
        metric_name=metric_name,
        metric_direction=metric_direction,
        candidate_support_weight=candidate_support_weight,
    )

    by_id = _candidate_by_id(eligible)
    selected: list[str] = []
    for consensus in consensus_sets:
        if len(selected) >= budget:
            break
        selected.append(consensus.representative_id)

    if len(selected) < budget:
        for consensus in consensus_sets:
            group = [by_id[candidate_id] for candidate_id in consensus.candidate_ids]
            ordered_group = sorted(group, key=lambda candidate: _metric_key(candidate, metric_name, metric_direction))
            for candidate in ordered_group:
                if candidate.candidate_id in selected:
                    continue
                selected.append(candidate.candidate_id)
                if len(selected) >= budget:
                    break
            if len(selected) >= budget:
                break

    return selected, consensus_sets, eligible


def _has_hidden_labels(candidates: Sequence[Candidate]) -> bool:
    return any(candidate.hidden_pass is not None for candidate in candidates)


def _hidden_success(candidate_ids: Iterable[str], candidates: Sequence[Candidate]) -> bool | None:
    by_id = _candidate_by_id(candidates)
    labels = [by_id[candidate_id].hidden_pass for candidate_id in candidate_ids if candidate_id in by_id]
    if not labels or all(label is None for label in labels):
        return None
    return any(label is True for label in labels)


def _visible_false_positive_rate(candidates: Sequence[Candidate]) -> float | None:
    labelled = [candidate for candidate in candidates if candidate.visible_pass and candidate.hidden_pass is not None]
    if not labelled:
        return None
    false_positives = sum(1 for candidate in labelled if candidate.hidden_pass is False)
    return false_positives / len(labelled)


def consensus_to_dict(consensus: ConsensusSet) -> dict[str, Any]:
    return {
        "consensus_id": consensus.consensus_id,
        "candidate_ids": list(consensus.candidate_ids),
        "candidate_count": consensus.candidate_count,
        "passed_probe_ids": list(consensus.passed_probe_ids),
        "passed_probe_count": consensus.passed_probe_count,
        "score": consensus.score,
        "representative_id": consensus.representative_id,
        "pass_vector": list(consensus.pass_vector),
    }


def build_selection_report(matrix: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(matrix.get("candidates") or [])
    if not rows:
        raise ValueError("matrix must contain a non-empty candidates list")

    budget = int(matrix.get("selection_budget", matrix.get("budget", 1)))
    metric = dict(matrix.get("primary_metric") or {})
    metric_name = metric.get("name")
    metric_direction = str(metric.get("direction", "min"))
    support_weight = str(matrix.get("candidate_support_weight", "sqrt"))
    visible_only = _as_bool(matrix.get("visible_only", True))
    probe_ids = matrix.get("probe_ids")

    selected, consensus_sets, eligible = select_candidates(
        rows,
        budget=budget,
        probe_ids=probe_ids,
        visible_only=visible_only,
        metric_name=metric_name,
        metric_direction=metric_direction,
        candidate_support_weight=support_weight,
    )
    all_candidates = normalize_candidates(rows)
    selected_hidden_success = _hidden_success(selected, all_candidates)
    oracle_hidden_success = (
        any(candidate.hidden_pass is True for candidate in all_candidates)
        if _has_hidden_labels(all_candidates)
        else None
    )
    ranker_gap = None
    if oracle_hidden_success is not None and selected_hidden_success is not None:
        ranker_gap = int(oracle_hidden_success) - int(selected_hidden_success)

    return {
        "task_id": matrix.get("task_id", "unknown"),
        "selection_budget": budget,
        "probe_ids": list(probe_ids) if probe_ids is not None else infer_probe_ids(eligible),
        "eligible_count": len(eligible),
        "total_candidate_count": len(all_candidates),
        "selected_candidate_ids": selected,
        "consensus_sets": [consensus_to_dict(consensus) for consensus in consensus_sets],
        "oracle_hidden_success": oracle_hidden_success,
        "selected_hidden_success": selected_hidden_success,
        "ranker_gap": ranker_gap,
        "visible_false_positive_rate": _visible_false_positive_rate(all_candidates),
        "selection_method": {
            "kind": "codet_dual_agreement",
            "candidate_support_weight": support_weight,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "visible_only": visible_only,
        },
    }
