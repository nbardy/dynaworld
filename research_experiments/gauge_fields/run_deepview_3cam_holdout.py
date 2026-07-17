from __future__ import annotations

from common import parse_gauge_matrix_args, run_gauge_matrix


BASE_CONFIG = (
    "src/train_configs/"
    "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_128_16f_2048el.jsonc"
)

SPLAT_BASE_CONFIG = (
    "src/train_configs/"
    "local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_free_dynamic_3dgs_128_16f_2048splats.jsonc"
)

STATIC_SPLAT_CONFIG = (
    "src/train_configs/"
    "local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_static_3dgs_128_16f_2048splats.jsonc"
)


RUNS = [
    {
        "name": "free_dynamic_3dgs",
        "script": "research_experiments/gauge_fields/train_splat_baseline.py",
        "config": SPLAT_BASE_CONFIG,
        "extra_args": [],
    },
    {
        "name": "static_3dgs",
        "script": "research_experiments/gauge_fields/train_splat_baseline.py",
        "config": STATIC_SPLAT_CONFIG,
        "extra_args": [],
    },
    {
        "name": "screen_disk_projected_conic",
        "script": "research_experiments/gauge_fields/train.py",
        "config": BASE_CONFIG,
        "extra_args": ["--support-mode", "screen_disk"],
    },
    {
        "name": "rank_adaptive_metric_projected_conic",
        "script": "research_experiments/gauge_fields/train.py",
        "config": BASE_CONFIG,
        "extra_args": [],
    },
    {
        "name": "rank_adaptive_metric_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "derived_support_metric_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_derived_support_metric_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "derived_support_metric_ray_gaussian_line_mass_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_derived_support_metric_ray_gaussian_line_mass_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "derived_support_metric_scale_0p025_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_derived_support_metric_scale_0p025_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "derived_support_metric_scale_0p050_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_derived_support_metric_scale_0p050_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "screen_disk_2048_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_screen_disk_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "transported_world_ball_2048_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_transported_world_ball_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "rank_adaptive_metric_2048_multiview_init_delayed_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_delayed_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "screen_disk_8192_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_screen_disk_multiview_init_pair_x_128_16f_8192el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "transported_world_ball_8192_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_transported_world_ball_multiview_init_pair_x_128_16f_8192el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "rank_adaptive_metric_7516_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_128_16f_7516el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "rank_adaptive_metric_ray_gaussian_line_mass_candidate_tiled",
        "script": "research_experiments/gauge_fields/train.py",
        "config": BASE_CONFIG,
        "extra_args": [
            "--incidence-mode",
            "ray_gaussian_line_mass",
            "--line-candidate-mode",
            "projected_bbox",
        ],
    },
]


def parse_args():
    return parse_gauge_matrix_args(
        description="Run DeepView train-2-cameras/test-1-camera gauge matrix.",
        default_output_root="outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step",
    )


def main() -> None:
    run_gauge_matrix(RUNS, parse_args())


if __name__ == "__main__":
    main()
