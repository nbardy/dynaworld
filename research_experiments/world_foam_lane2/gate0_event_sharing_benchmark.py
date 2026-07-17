from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gate0_beam_toy import ToyConfig, run


def parse_frame_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in value.split(",") if part.strip())
    if not counts:
        raise ValueError("provide at least one frame count")
    return counts


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("provide at least one float value")
    return values


def run_sweep(args: argparse.Namespace, *, camera_velocity_x: float) -> dict[str, Any]:
    label = f"camera_velocity_x_{camera_velocity_x:g}"
    safe_label = label.replace("-", "neg_").replace(".", "p")
    config = ToyConfig(
        frame_counts=parse_frame_counts(args.frame_counts),
        u_samples=args.u_samples,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        camera_velocity_x=camera_velocity_x,
        invalid_epsilon=args.invalid_epsilon,
    )
    toy = run(config)
    backward_rows = [
        {
            "frames": row["frames"],
            "per_frame_replay_event_sum": row["per_frame_event_sum"],
            "beam_replay_event_sum": row["beam_slab_event_sum"],
            "backward_replay_ratio": row["event_sharing_ratio"],
        }
        for row in toy["rows"]
    ]
    return {
        "name": safe_label,
        "camera_velocity_x": camera_velocity_x,
        "rows": toy["rows"],
        "backward_replay_rows": backward_rows,
        "growth": toy["growth"],
        "all_rows_zero_missing": all(int(row["missing_sample_events"]) == 0 for row in toy["rows"]),
        "sublinear_event_growth": bool(toy["growth"]["sublinear_event_growth"]),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    sweeps = [run_sweep(args, camera_velocity_x=value) for value in parse_float_list(args.camera_velocities)]
    primary = sweeps[0]
    return {
        "benchmark": "world_foam_lane2_gate0_event_sharing",
        "status": "toy_forward_event_count_only",
        "backward_status": "event_replay_accounting_only_no_gradients",
        "world_foam_rows": primary["rows"],
        "world_foam_backward_replay_rows": primary["backward_replay_rows"],
        "growth": primary["growth"],
        "sweeps": sweeps,
        "comparison_contract": {
            "per_frame_world_foam": "summed power-boundary events at sampled frame times",
            "beam_world_foam": "shared screen-time slab candidate boundary events",
            "star_uvt": "not run here; compare against star_uvt_v0 pair_ratio and render timing under matched data",
            "dynamic_splats": "not run here; compare against fast_mac dynamic splat per-frame pair/render/backward timing under matched data",
        },
        "acceptance": {
            "requires_zero_missing_sample_events": True,
            "requires_sublinear_event_growth": True,
            "requires_two_sweeps": True,
            "all_rows_zero_missing": all(bool(sweep["all_rows_zero_missing"]) for sweep in sweeps),
            "sublinear_event_growth": all(bool(sweep["sublinear_event_growth"]) for sweep in sweeps),
            "backward_replay_sublinear_event_growth": all(
                bool(sweep["sublinear_event_growth"]) for sweep in sweeps
            ),
            "sweep_count": len(sweeps),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Foam Lane 2 Gate 0 event-sharing benchmark.")
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--u-samples", type=int, default=17)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.25)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument("--camera-velocities", default="0.35,0.7")
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
