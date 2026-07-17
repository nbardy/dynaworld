from __future__ import annotations

from common import parse_gauge_matrix_args, run_gauge_matrix


RUNS = [
    {
        "name": "free_dynamic_3dgs",
        "script": "research_experiments/gauge_fields/train_splat_baseline.py",
        "config": "src/train_configs/local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc",
    },
    {
        "name": "screen_disk_projected_conic",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_screen_disk_128_16f_2048el.jsonc",
    },
    {
        "name": "rank_adaptive_metric_projected_conic",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_128_16f_2048el.jsonc",
    },
    {
        "name": "rank_adaptive_metric_ray_gaussian_line_mass",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_mass_128_16f_2048el.jsonc",
    },
    {
        "name": "rank_adaptive_metric_ray_gaussian_line_mass_candidate_tiled",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_mass_candidate_tiled_128_16f_2048el.jsonc",
    },
    {
        "name": "rank_adaptive_metric_ray_gaussian_line_peak",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_peak_128_16f_2048el.jsonc",
    },
]


def parse_args():
    return parse_gauge_matrix_args(
        description="Run the DeepView incidence-mode benchmark matrix.",
        default_output_root="outputs/gauge_fields/multicam_deepview_incidence_matrix_80step",
    )


def main() -> None:
    run_gauge_matrix(RUNS, parse_args())


if __name__ == "__main__":
    main()
