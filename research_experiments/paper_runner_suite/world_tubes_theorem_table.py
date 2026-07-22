from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
STAR_REPORTS = ROOT / "research_experiments" / "star_uvt_feature_tubes"
for path in (ROOT, STAR_REPORTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projective_bundle_gauge_gradient_report import verify_bundle_gauge_gradient_report  # noqa: E402
from projective_bundle_gauge_invariance_report import verify_bundle_gauge_invariance_report  # noqa: E402
from projective_decisive_demo_report import verify_projective_decisive_demo_report  # noqa: E402
from projective_exposure_rolling_backward_report import verify_exposure_rolling_backward_report  # noqa: E402
from projective_exposure_rolling_mixed_fallback_backward_report import (  # noqa: E402
    verify_mixed_fallback_backward_report,
)
from projective_exposure_rolling_quadrature_report import verify_exposure_rolling_quadrature_report  # noqa: E402
from projective_orbit_fixed_chart_scaling_benchmark import verify_orbit_fixed_chart_scaling_report  # noqa: E402
from projective_visibility_stress_suite import verify_projective_visibility_stress_suite  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-22_world_tubes_theorem_table"
DEFAULT_REPORTS = {
    "gauge_value": ROOT / "outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json",
    "gauge_gradient": ROOT / "outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json",
    "decisive_demo": ROOT / "outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json",
    "visibility": ROOT / "outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json",
    "exposure": ROOT / "outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.json",
    "exposure_backward": ROOT / "outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.json",
    "mixed_fallback_backward": ROOT / "outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.json",
    "scaling": ROOT / "outputs/benchmarks/2026-07-22_world_tubes_same_representation_scaling_f4_128_cap256/summary.json",
}
VERIFIERS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "gauge_value": verify_bundle_gauge_invariance_report,
    "gauge_gradient": verify_bundle_gauge_gradient_report,
    "decisive_demo": verify_projective_decisive_demo_report,
    "visibility": verify_projective_visibility_stress_suite,
    "exposure": verify_exposure_rolling_quadrature_report,
    "exposure_backward": verify_exposure_rolling_backward_report,
    "mixed_fallback_backward": verify_mixed_fallback_backward_report,
    "scaling": verify_orbit_fixed_chart_scaling_report,
}


def _load_verified(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"theorem evidence is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    errors = VERIFIERS[name](report)
    if errors:
        raise ValueError(f"{name} theorem evidence failed:\n- " + "\n- ".join(errors))
    return report


def build_table(paths: dict[str, Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    reports = {name: _load_verified(name, path) for name, path in paths.items()}
    gauge_value = reports["gauge_value"]["summary"]
    gauge_gradient = reports["gauge_gradient"]["summary"]
    demo = reports["decisive_demo"]["summary"]
    visibility = reports["visibility"]["summary"]
    exposure = reports["exposure"]["summary"]
    exposure_backward = reports["exposure_backward"]["summary"]
    mixed = reports["mixed_fallback_backward"]["summary"]
    scaling = reports["scaling"]["summary"]
    rows = [
        {
            "claim": "Fiber value is gauge invariant",
            "metric": "max relative error",
            "value": gauge_value["max_rel_error"],
            "acceptance": "<= 1e-10",
            "source": "gauge_value",
        },
        {
            "claim": "Fiber gradient is gauge invariant",
            "metric": "max gradient relative error",
            "value": gauge_gradient["max_gradient_rel_error"],
            "acceptance": "<= 1e-9",
            "source": "gauge_gradient",
        },
        {
            "claim": "Compiled atlas matches dense/replay image",
            "metric": "max absolute image error",
            "value": demo["max_image_abs_error_vs_reference"],
            "acceptance": "<= 1e-5",
            "source": "decisive_demo",
        },
        {
            "claim": "Visibility crossing is repaired by stratification",
            "metric": "max accepted quality error",
            "value": max(
                float(row["quality_error"])
                for row in reports["visibility"]["rows"]
                if row["case_id"] in {"clean_orbit_ordered", "crossing_stratified", "forced_fallback_ambiguous"}
            ),
            "acceptance": "<= 1e-5",
            "source": "visibility",
        },
        {
            "claim": "Finite exposure / rolling shutter forward parity",
            "metric": "max Metal absolute error",
            "value": exposure["max_metal_abs_error"],
            "acceptance": "<= 1e-5",
            "source": "exposure",
        },
        {
            "claim": "Finite exposure / rolling shutter gradient parity",
            "metric": "max Metal gradient relative error",
            "value": exposure_backward["max_metal_grad_rel_error"],
            "acceptance": "<= 1e-5",
            "source": "exposure_backward",
        },
        {
            "claim": "Mixed fallback preserves gradients",
            "metric": "max mixed gradient relative error",
            "value": mixed["max_mixed_grad_rel_error"],
            "acceptance": "<= 1e-5",
            "source": "mixed_fallback_backward",
        },
        {
            "claim": "Bounded-orbit chart reuses payload across F=4..128",
            "metric": "final fixed/replay trace ratio",
            "value": scaling["last_fixed_vs_per_frame_trace_ratio"],
            "acceptance": "< 0.25",
            "source": "scaling",
        },
        {
            "claim": "Bounded-orbit compiled forward is faster at F=128",
            "metric": "final fixed/replay forward ratio",
            "value": scaling["last_fixed_vs_per_frame_forward_ms_ratio"],
            "acceptance": "< 0.5",
            "source": "scaling",
        },
        {
            "claim": "Bounded-orbit compiled backward is faster at F=128",
            "metric": "final fixed/replay backward ratio",
            "value": scaling["last_fixed_vs_per_frame_backward_ms_ratio"],
            "acceptance": "< 0.5",
            "source": "scaling",
        },
    ]
    return {
        "status": "complete",
        "scope": "bounded event-certified projective chart segments; no 360/720 multi-chart claim",
        "rows": rows,
        "sources": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        "summary": {
            "row_count": len(rows),
            "all_sources_verified": True,
            "frame_counts": reports["scaling"]["frame_counts"],
            "full_orbit_multigauge_claim": False,
        },
    }


def write_table(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    header = "| Claim | Metric | Value | Acceptance | Source |\n|---|---|---:|---:|---|"
    lines = ["# World Tubes theorem and correctness table", "", report["scope"], "", header]
    latex = [
        r"\begin{tabular}{p{0.34\linewidth}p{0.25\linewidth}rr}",
        r"\toprule",
        r"Claim & Metric & Value & Acceptance \\",
        r"\midrule",
    ]
    for row in report["rows"]:
        value = f"{float(row['value']):.6g}"
        lines.append(f"| {row['claim']} | {row['metric']} | {value} | {row['acceptance']} | {row['source']} |")
        latex_acceptance = row["acceptance"].replace("<=", r"$\le$").replace("<", "$<$")
        latex.append(
            f"{row['claim']} & {row['metric']} & {value} & {latex_acceptance} \\\\"
        )
    latex.extend((r"\bottomrule", r"\end{tabular}"))
    (out_dir / "theorem_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "theorem_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("summary", {}).get("all_sources_verified") is not True:
            raise ValueError("theorem table report is incomplete")
        print(f"verified {args.verify_report}")
        return
    report = build_table()
    write_table(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
