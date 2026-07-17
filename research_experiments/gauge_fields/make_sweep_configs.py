from __future__ import annotations

import argparse
from pathlib import Path


from common import (  # noqa: E402
    DYNAWORLD_ROOT,
    clone_jsonable,
    parse_csv_bools,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    resolve_dynaworld_path,
    write_generated_jsonc,
)
from config_utils import load_config_file  # noqa: E402
from train import gauge_config  # noqa: E402


def format_float_token(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def format_bool_token(value: bool) -> str:
    return "trace_norm" if value else "trace_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate material-surfel capacity/radius/alpha sweep configs.")
    parser.add_argument(
        "--base-config",
        default="src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc",
    )
    parser.add_argument(
        "--output-dir",
        default="src/train_configs/generated_gauge_fields_sweeps",
    )
    parser.add_argument("--elements", default="1024,2048,4096")
    parser.add_argument("--radii", default="0.05,0.07,0.09")
    parser.add_argument("--alpha-logits", default="-1.2,0.0")
    parser.add_argument("--support-modes", default="screen_disk,derived_support_metric,rank_adaptive_metric")
    parser.add_argument("--incidence-modes", default="projected_conic")
    parser.add_argument("--derived-support-scales", default="0.035")
    parser.add_argument("--derived-support-floors", default="0.0001")
    parser.add_argument("--derived-support-weight-taus", default="0.0")
    parser.add_argument("--derived-support-normalize-trace-values", default="true")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = gauge_config(load_config_file(resolve_dynaworld_path(args.base_config)))
    output_dir = resolve_dynaworld_path(args.output_dir)

    written = []
    for support_mode in parse_csv_strings(args.support_modes):
        for incidence_mode in parse_csv_strings(args.incidence_modes):
            for elements in parse_csv_ints(args.elements):
                for radius in parse_csv_floats(args.radii):
                    for alpha_logit in parse_csv_floats(args.alpha_logits):
                        if support_mode == "derived_support_metric":
                            derived_scales = parse_csv_floats(args.derived_support_scales)
                            derived_floors = parse_csv_floats(args.derived_support_floors)
                            derived_weight_taus = parse_csv_floats(args.derived_support_weight_taus)
                            derived_normalize_trace_values = parse_csv_bools(
                                args.derived_support_normalize_trace_values
                            )
                        else:
                            derived_scales = [float(base_config["model"].get("derived_support_scale", 0.035))]
                            derived_floors = [float(base_config["model"].get("derived_support_floor", 1e-4))]
                            derived_weight_taus = [float(base_config["model"].get("derived_support_weight_tau", 0.0))]
                            derived_normalize_trace_values = [
                                bool(base_config["model"].get("derived_support_normalize_trace", True))
                            ]

                        for derived_scale in derived_scales:
                            for derived_floor in derived_floors:
                                for derived_weight_tau in derived_weight_taus:
                                    for derived_normalize_trace in derived_normalize_trace_values:
                                        cfg = clone_jsonable(base_config)
                                        cfg.setdefault("model", {})
                                        cfg.setdefault("render", {})
                                        cfg.setdefault("train", {})
                                        cfg.setdefault("logging", {})
                                        cfg["model"]["support_mode"] = support_mode
                                        cfg["render"]["incidence_mode"] = incidence_mode
                                        cfg["model"]["num_elements"] = int(elements)
                                        cfg["model"]["init_radius"] = float(radius)
                                        cfg["model"]["init_alpha_logit"] = float(alpha_logit)
                                        cfg["train"]["steps"] = int(args.steps)
                                        if support_mode == "derived_support_metric":
                                            cfg["model"]["derived_support_scale"] = float(derived_scale)
                                            cfg["model"]["derived_support_floor"] = float(derived_floor)
                                            cfg["model"]["derived_support_weight_tau"] = float(derived_weight_tau)
                                            cfg["model"]["derived_support_normalize_trace"] = bool(derived_normalize_trace)

                                        run_slug = (
                                            f"gauge_fields_{support_mode}_{incidence_mode}_motion_128_16f_"
                                            f"{elements}el-r{format_float_token(radius)}-"
                                            f"a{format_float_token(alpha_logit)}"
                                        )
                                        derived_tags = []
                                        if support_mode == "derived_support_metric":
                                            run_slug += (
                                                f"-ds{format_float_token(derived_scale)}"
                                                f"-df{format_float_token(derived_floor)}"
                                                f"-dt{format_float_token(derived_weight_tau)}"
                                                f"-{format_bool_token(derived_normalize_trace)}"
                                            )
                                            derived_tags = [
                                                "derived-support-calibration",
                                                f"ds{format_float_token(derived_scale)}",
                                                f"df{format_float_token(derived_floor)}",
                                                f"dt{format_float_token(derived_weight_tau)}",
                                                format_bool_token(derived_normalize_trace),
                                            ]
                                        run_slug += f"-{int(args.steps)}step"
                                        cfg["logging"]["wandb_run_name"] = run_slug
                                        cfg["logging"]["wandb_mode"] = args.wandb_mode
                                        cfg["logging"]["log_to_wandb"] = not bool(args.disable_wandb)
                                        cfg["logging"]["output_dir"] = f"outputs/gauge_fields/sweeps/{run_slug}"
                                        cfg["logging"]["wandb_tags"] = [
                                            "gauge-fields",
                                            support_mode,
                                            "support-mode",
                                            incidence_mode,
                                            "incidence-mode",
                                            "coverage-sweep",
                                            "128px",
                                            "16f",
                                            f"{int(args.steps)}step",
                                            f"{elements}el",
                                            f"r{format_float_token(radius)}",
                                            f"a{format_float_token(alpha_logit)}",
                                            *derived_tags,
                                        ]

                                        path = output_dir / f"local_mac_{run_slug}.jsonc"
                                        write_generated_jsonc(
                                            path,
                                            cfg,
                                            generated_by="research_experiments/gauge_fields/make_sweep_configs.py",
                                        )
                                        written.append(path)

    for path in written:
        try:
            print(path.relative_to(DYNAWORLD_ROOT))
        except ValueError:
            print(path)


if __name__ == "__main__":
    main()
