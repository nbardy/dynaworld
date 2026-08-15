"""Adaptive per-cell M3/M5 material-basis ablation for WorldFoam.

The earlier matched-payload gate showed that positive Bernstein P2 (M3) and
convex log-P2 (M5) are complementary: each exactly fits its own generating
family and loses on the other.  This experiment tests the resulting decision
rule without leaking the target family.  Both candidates are fitted on the
same train chords, the basis tag is selected on a disjoint validation chord
set, and the selected candidate is evaluated on a third held-out chord set.

This is a deterministic float64 CPU material-capacity ablation.  It is not a
native renderer, image-quality, speed, or memory-scaling result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Iterable, Mapping, Sequence

import torch

from research_experiments.world_foam_lane2.finite_element_material_fit import (
    DEFAULT_INTERVALS,
    HELDOUT_INTERVALS,
    FittedMaterial,
    MATERIAL_SCALARS,
    evaluate_material_field,
)
from research_experiments.world_foam_lane2.finite_element_material_transfer import (
    MaterialMode,
)


SCHEMA_VERSION = 1
SUITE_ID = "worldfoam-adaptive-m3-m5-basis-selection-v1"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/benchmarks/2026-08-15_worldfoam_adaptive_material_basis_cpu"
)
DTYPE = torch.float64
M3 = MaterialMode.M3_POSITIVE_BERNSTEIN_P2
M5 = MaterialMode.M5_CONVEX_LOG_P2
MODES = (M3, M5)
SEEDS = (17, 29, 43)
SELECTION_INTERVALS = (
    (0.02, 0.16),
    (0.07, 0.29),
    (0.15, 0.44),
    (0.28, 0.61),
    (0.52, 0.88),
    (0.71, 0.99),
)
_SIMPSON_SUBINTERVALS = 2048


@dataclass(frozen=True)
class TargetSpec:
    name: str
    family: str
    m3_controls: tuple[float, float, float]
    m5_controls: tuple[float, float, float]
    m3_mixture_weight: float
    length: float
    color: tuple[float, float, float]


TARGETS = (
    TargetSpec(
        "m3_peak_center",
        "m3",
        (0.08, 3.20, 0.18),
        (12.0, -12.0, 3.0),
        1.0,
        1.70,
        (0.82, 0.27, 0.11),
    ),
    TargetSpec(
        "m3_broad_center",
        "m3",
        (0.15, 2.40, 0.35),
        (8.0, -5.6, 1.2),
        1.0,
        1.35,
        (0.19, 0.71, 0.38),
    ),
    TargetSpec(
        "m3_back_heavy",
        "m3",
        (0.05, 1.80, 0.65),
        (16.0, -22.4, 7.0),
        1.0,
        1.90,
        (0.28, 0.36, 0.84),
    ),
    TargetSpec(
        "m3_sharp_center",
        "m3",
        (0.40, 3.80, 0.12),
        (6.0, -9.0, 3.2),
        1.0,
        1.55,
        (0.74, 0.22, 0.56),
    ),
    TargetSpec(
        "m5_peak_center",
        "m5",
        (0.08, 3.20, 0.18),
        (12.0, -12.0, 3.0),
        0.0,
        1.70,
        (0.18, 0.73, 0.36),
    ),
    TargetSpec(
        "m5_front_peak",
        "m5",
        (0.15, 2.40, 0.35),
        (8.0, -5.6, 1.2),
        0.0,
        1.35,
        (0.77, 0.31, 0.16),
    ),
    TargetSpec(
        "m5_back_peak",
        "m5",
        (0.05, 1.80, 0.65),
        (16.0, -22.4, 7.0),
        0.0,
        1.90,
        (0.25, 0.42, 0.88),
    ),
    TargetSpec(
        "m5_broad_back",
        "m5",
        (0.40, 3.80, 0.12),
        (6.0, -9.0, 3.2),
        0.0,
        1.55,
        (0.67, 0.19, 0.61),
    ),
    TargetSpec(
        "hybrid_m5_dominant",
        "hybrid",
        (0.10, 2.80, 0.25),
        (11.0, -13.2, 3.8),
        0.20,
        1.60,
        (0.21, 0.76, 0.49),
    ),
    TargetSpec(
        "hybrid_m5_leaning",
        "hybrid",
        (0.25, 2.20, 0.42),
        (9.0, -7.2, 1.8),
        0.40,
        1.45,
        (0.81, 0.38, 0.13),
    ),
    TargetSpec(
        "hybrid_m3_leaning",
        "hybrid",
        (0.12, 3.40, 0.30),
        (14.0, -19.6, 6.1),
        0.60,
        1.80,
        (0.31, 0.28, 0.86),
    ),
    TargetSpec(
        "hybrid_m3_dominant",
        "hybrid",
        (0.32, 3.00, 0.10),
        (7.0, -9.8, 3.6),
        0.80,
        1.50,
        (0.71, 0.17, 0.53),
    ),
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve()
    paths = (
        here,
        here.with_name("finite_element_material_fit.py"),
        here.with_name("finite_element_material_transfer.py"),
    )
    return {str(path.relative_to(here.parents[2])): _sha256_file(path) for path in paths}


def _target_density(target: TargetSpec, xi: torch.Tensor) -> torch.Tensor:
    m3 = torch.tensor(target.m3_controls, dtype=DTYPE)
    one_minus = 1.0 - xi
    rho_m3 = (
        one_minus.square() * m3[0]
        + 2.0 * xi * one_minus * m3[1]
        + xi.square() * m3[2]
    )
    a, b, c = torch.tensor(target.m5_controls, dtype=DTYPE)
    rho_m5 = torch.exp(-(a * xi.square() + b * xi + c))
    weight = float(target.m3_mixture_weight)
    return weight * rho_m3 + (1.0 - weight) * rho_m5


def independent_target_outputs(
    target: TargetSpec,
    intervals: Iterable[tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Composite-Simpson target oracle independent of both fitted evaluators."""

    betas: list[torch.Tensor] = []
    moments: list[torch.Tensor] = []
    color = torch.tensor(target.color, dtype=DTYPE)
    for start, stop in intervals:
        if not 0.0 <= start < stop <= 1.0:
            raise ValueError("all intervals must satisfy 0 <= start < stop <= 1")
        xi = torch.linspace(
            float(start),
            float(stop),
            _SIMPSON_SUBINTERVALS + 1,
            dtype=DTYPE,
        )
        density = _target_density(target, xi)
        step = (float(stop) - float(start)) / _SIMPSON_SUBINTERVALS
        integral = (step / 3.0) * (
            density[0]
            + density[-1]
            + 4.0 * density[1:-1:2].sum()
            + 2.0 * density[2:-1:2].sum()
        )
        beta = torch.exp(-float(target.length) * integral)
        betas.append(beta)
        moments.append((1.0 - beta) * color)
    return torch.stack(betas), torch.stack(moments)


def _stack_outputs(transfers: Sequence[object]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([transfer.element.beta for transfer in transfers]),
        torch.stack([transfer.element.m for transfer in transfers]),
    )


def _split_metrics(
    target: TargetSpec,
    mode: MaterialMode,
    model: FittedMaterial,
    intervals: Sequence[tuple[float, float]],
) -> dict[str, float]:
    target_beta, target_rgb = independent_target_outputs(target, intervals)
    color_front, color_back = model.colors()
    predicted_beta, predicted_rgb = _stack_outputs(
        evaluate_material_field(
            mode,
            model.density_controls(),
            torch.tensor(target.length, dtype=DTYPE),
            color_front,
            color_back,
            intervals,
        )
    )
    beta_error = predicted_beta - target_beta
    rgb_error = predicted_rgb - target_rgb
    beta_mse = float(beta_error.square().mean().detach())
    rgb_mse = float(rgb_error.square().mean().detach())
    return {
        "loss": beta_mse + rgb_mse,
        "beta_mse": beta_mse,
        "rgb_mse": rgb_mse,
        "max_beta_abs_error": float(beta_error.abs().max().detach()),
        "max_rgb_abs_error": float(rgb_error.abs().max().detach()),
    }


def fit_candidate(
    target: TargetSpec,
    mode: MaterialMode,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    refinement_steps: int,
) -> dict[str, object]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    model = FittedMaterial(mode, seed)
    train_beta, train_rgb = independent_target_outputs(target, DEFAULT_INTERVALS)
    length = torch.tensor(target.length, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    started_at = time.perf_counter()

    def training_loss() -> torch.Tensor:
        color_front, color_back = model.colors()
        predicted_beta, predicted_rgb = _stack_outputs(
            evaluate_material_field(
                mode,
                model.density_controls(),
                length,
                color_front,
                color_back,
                DEFAULT_INTERVALS,
            )
        )
        return (
            (predicted_beta - train_beta).square().mean()
            + (predicted_rgb - train_rgb).square().mean()
        )

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss()
        loss.backward()
        optimizer.step()

    if refinement_steps:
        refinement = torch.optim.LBFGS(
            model.parameters(),
            max_iter=int(refinement_steps),
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            refinement.zero_grad(set_to_none=True)
            loss = training_loss()
            loss.backward()
            return loss

        refinement.step(closure)

    with torch.no_grad():
        splits = {
            "train": _split_metrics(target, mode, model, DEFAULT_INTERVALS),
            "selection": _split_metrics(target, mode, model, SELECTION_INTERVALS),
            "heldout": _split_metrics(target, mode, model, HELDOUT_INTERVALS),
        }
        color_front, color_back = model.colors()
        density_controls = model.density_controls()
    return {
        "target": target.name,
        "target_family": target.family,
        "mode": mode.name,
        "seed": int(seed),
        "serialized_material_scalars": MATERIAL_SCALARS[mode],
        "serialized_material_bytes_float32": 4 * MATERIAL_SCALARS[mode],
        "basis_tag_bits": 1,
        "trainable_scalars": sum(parameter.numel() for parameter in model.parameters()),
        "density_controls": density_controls.tolist(),
        "color_front": color_front.tolist(),
        "color_back": color_back.tolist(),
        "splits": splits,
        "diagnostic_cpu_wall_seconds": time.perf_counter() - started_at,
    }


def _mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    return sum(values) / len(values)


def _selection_rows(candidate_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (str(row["target"]), int(row["seed"]), str(row["mode"])): row
        for row in candidate_rows
    }
    selections: list[dict[str, object]] = []
    for target in TARGETS:
        for seed in SEEDS:
            m3 = indexed[(target.name, seed, M3.name)]
            m5 = indexed[(target.name, seed, M5.name)]
            m3_selection = float(m3["splits"]["selection"]["loss"])
            m5_selection = float(m5["splits"]["selection"]["loss"])
            selected = m3 if m3_selection <= m5_selection else m5
            m3_heldout = float(m3["splits"]["heldout"]["loss"])
            m5_heldout = float(m5["splits"]["heldout"]["loss"])
            oracle = m3 if m3_heldout <= m5_heldout else m5
            selected_heldout = float(selected["splits"]["heldout"]["loss"])
            oracle_heldout = float(oracle["splits"]["heldout"]["loss"])
            selections.append(
                {
                    "target": target.name,
                    "target_family": target.family,
                    "seed": seed,
                    "selected_mode": selected["mode"],
                    "oracle_mode": oracle["mode"],
                    "selection_correct_for_pure_family": (
                        target.family == "hybrid"
                        or selected["mode"] == (M3.name if target.family == "m3" else M5.name)
                    ),
                    "m3_selection_loss": m3_selection,
                    "m5_selection_loss": m5_selection,
                    "m3_heldout_loss": m3_heldout,
                    "m5_heldout_loss": m5_heldout,
                    "selected_heldout_loss": selected_heldout,
                    "oracle_heldout_loss": oracle_heldout,
                    "selection_regret": selected_heldout - oracle_heldout,
                }
            )
    return selections


def _aggregates(
    candidate_rows: Sequence[Mapping[str, object]],
    selections: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_mode = {
        mode.name: _mean(
            float(row["splits"]["heldout"]["loss"])
            for row in candidate_rows
            if row["mode"] == mode.name
        )
        for mode in MODES
    }
    adaptive = _mean(float(row["selected_heldout_loss"]) for row in selections)
    oracle = _mean(float(row["oracle_heldout_loss"]) for row in selections)
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
            M3.name: by_mode[M3.name],
            M5.name: by_mode[M5.name],
            "adaptive_validation_selection": adaptive,
            "heldout_oracle_selection": oracle,
        },
        "adaptive_to_best_fixed_ratio": adaptive / min(by_mode.values()),
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


def _checks(
    candidate_rows: Sequence[Mapping[str, object]],
    selections: Sequence[Mapping[str, object]],
    aggregates: Mapping[str, object],
) -> dict[str, bool]:
    losses = aggregates["mean_heldout_loss"]
    family = aggregates["family_losses"]
    return {
        "expected_candidate_row_count": len(candidate_rows) == len(TARGETS) * len(SEEDS) * 2,
        "expected_selection_row_count": len(selections) == len(TARGETS) * len(SEEDS),
        "all_metrics_finite": all(
            math.isfinite(float(metric))
            for row in candidate_rows
            for split in ("train", "selection", "heldout")
            for metric in row["splits"][split].values()
        ),
        "split_interval_lists_are_distinct": len(
            set(DEFAULT_INTERVALS) | set(SELECTION_INTERVALS) | set(HELDOUT_INTERVALS)
        )
        == len(DEFAULT_INTERVALS) + len(SELECTION_INTERVALS) + len(HELDOUT_INTERVALS),
        "m3_m5_matched_24_byte_payload": MATERIAL_SCALARS[M3] == MATERIAL_SCALARS[M5] == 6,
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


def protocol_payload(
    *,
    steps: int,
    learning_rate: float,
    refinement_steps: int,
) -> dict[str, object]:
    return {
        "suite_id": SUITE_ID,
        "targets": [asdict(target) for target in TARGETS],
        "train_intervals": [list(interval) for interval in DEFAULT_INTERVALS],
        "selection_intervals": [list(interval) for interval in SELECTION_INTERVALS],
        "heldout_intervals": [list(interval) for interval in HELDOUT_INTERVALS],
        "seeds": list(SEEDS),
        "candidate_modes": [mode.name for mode in MODES],
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "refinement_steps": int(refinement_steps),
        "target_oracle": f"composite_simpson_{_SIMPSON_SUBINTERVALS}_subintervals",
        "selection_rule": "minimum_disjoint_selection_chord_loss",
        "payload": "six_float32_material_scalars_plus_one_basis_tag_bit",
    }


def run_ablation(
    *,
    steps: int = 500,
    learning_rate: float = 0.04,
    refinement_steps: int = 30,
) -> dict[str, object]:
    candidates = [
        fit_candidate(
            target,
            mode,
            seed=seed,
            steps=steps,
            learning_rate=learning_rate,
            refinement_steps=refinement_steps,
        )
        for target in TARGETS
        for seed in SEEDS
        for mode in MODES
    ]
    selections = _selection_rows(candidates)
    aggregates = _aggregates(candidates, selections)
    checks = _checks(candidates, selections, aggregates)
    root = Path(__file__).resolve().parents[2]
    protocol = protocol_payload(
        steps=steps,
        learning_rate=learning_rate,
        refinement_steps=refinement_steps,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "accepted": all(checks.values()),
        "claim_scope": [
            "synthetic per-cell M3/M5 basis selection on disjoint train/selection/heldout chords",
            "matched six-float material payloads plus one basis-tag bit",
            "not public image quality, native integration, renderer speed, or memory scaling",
        ],
        "promotion": {
            "adaptive_basis_selection_supported": all(checks.values()),
            "eligible_to_replace_p0_in_native_renderer": False,
            "reason": (
                "selection is synthetic material evidence; native/public gates remain separate"
            ),
        },
        "dtype": "float64",
        "device": "cpu",
        "timing_is_paper_evidence": False,
        "git": _git_metadata(root),
        "source_sha256": _source_hashes(),
        "protocol": protocol,
        "protocol_sha256": _sha256_json(protocol),
        "candidate_rows": candidates,
        "selection_rows": selections,
        "aggregates": aggregates,
        "checks": checks,
    }


def write_markdown_table(report: Mapping[str, object], path: Path) -> None:
    losses = report["aggregates"]["mean_heldout_loss"]
    lines = [
        "# Adaptive M3/M5 held-out material selection",
        "",
        "| Policy | Mean held-out loss |",
        "| --- | ---: |",
        f"| Fixed M3 | {float(losses[M3.name]):.8e} |",
        f"| Fixed M5 | {float(losses[M5.name]):.8e} |",
        f"| Validation-selected M3/M5 | {float(losses['adaptive_validation_selection']):.8e} |",
        f"| Held-out oracle (diagnostic ceiling) | {float(losses['heldout_oracle_selection']):.8e} |",
        "",
        (
            "Pure-family selection accuracy: "
            f"{100.0 * float(report['aggregates']['pure_family_selection_accuracy']):.2f}%"
        ),
        "",
        "Synthetic material-capacity evidence only; timing is diagnostic.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_table(report: Mapping[str, object], path: Path) -> None:
    losses = report["aggregates"]["mean_heldout_loss"]
    rows = (
        ("Fixed M3", float(losses[M3.name])),
        ("Fixed M5", float(losses[M5.name])),
        ("Validation-selected M3/M5", float(losses["adaptive_validation_selection"])),
        ("Held-out oracle (diagnostic)", float(losses["heldout_oracle_selection"])),
    )
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Policy & Mean held-out loss \\",
        r"\midrule",
        *(f"{label} & {value:.3e} \\\\" for label, value in rows),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(report: Mapping[str, object], path: Path) -> None:
    losses = report["aggregates"]["mean_heldout_loss"]
    values = (
        ("Fixed M3", float(losses[M3.name]), "#4C78A8"),
        ("Fixed M5", float(losses[M5.name]), "#F58518"),
        ("Adaptive", float(losses["adaptive_validation_selection"]), "#54A24B"),
        ("Oracle", float(losses["heldout_oracle_selection"]), "#B279A2"),
    )
    scores = [-math.log10(max(value, 1.0e-16)) for _, value, _ in values]
    maximum = max(scores) * 1.08
    bars: list[str] = []
    for index, ((label, value, color), score) in enumerate(zip(values, scores)):
        x = 108 + index * 142
        height = 260.0 * score / maximum
        y = 354.0 - height
        bars.extend(
            (
                f'<rect x="{x}" y="{y:.3f}" width="92" height="{height:.3f}" fill="{color}"/>',
                f'<text x="{x + 46}" y="378" text-anchor="middle" class="label">{label}</text>',
                f'<text x="{x + 46}" y="{max(88.0, y - 8):.3f}" text-anchor="middle" class="value">{value:.2e}</text>',
            )
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440" viewBox="0 0 720 440">
<title>Adaptive WorldFoam M3/M5 basis selection</title>
<desc>Mean held-out material loss for fixed M3, fixed M5, validation-selected adaptive basis, and a diagnostic held-out oracle.</desc>
<style>.title{{font:700 22px sans-serif;fill:#111}}.axis{{font:14px sans-serif;fill:#333}}.label{{font:14px sans-serif;fill:#222}}.value{{font:12px monospace;fill:#111}}</style>
<rect width="720" height="440" fill="white"/>
<text x="360" y="34" text-anchor="middle" class="title">Adaptive per-cell material basis selection</text>
<text x="24" y="220" transform="rotate(-90 24 220)" text-anchor="middle" class="axis">-log10(mean held-out loss), higher is better</text>
<line x1="74" y1="354" x2="686" y2="354" stroke="#222" stroke-width="1.5"/>
{''.join(bars)}
<text x="360" y="418" text-anchor="middle" class="axis">Disjoint train / selection / held-out partial chords; float64 CPU</text>
</svg>'''
    path.write_text(svg + "\n", encoding="utf-8")


def add_assets(report: dict[str, object], output_dir: Path) -> None:
    figure = output_dir / "figures" / "worldfoam_adaptive_material_basis.svg"
    markdown = output_dir / "adaptive_material_basis_table.md"
    tex = output_dir / "adaptive_material_basis_table.tex"
    figure.parent.mkdir(parents=True, exist_ok=True)
    write_svg(report, figure)
    write_markdown_table(report, markdown)
    write_tex_table(report, tex)
    report["assets"] = {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in (figure, markdown, tex)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--refinement-steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 1 or args.refinement_steps < 0 or args.learning_rate <= 0.0:
        parser.error("steps and learning rate must be positive; refinement steps nonnegative")
    torch.set_num_threads(1)
    report = run_ablation(
        steps=args.steps,
        learning_rate=args.learning_rate,
        refinement_steps=args.refinement_steps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    add_assets(report, args.output_dir)
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "summary": str(summary),
                "aggregates": report["aggregates"],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
