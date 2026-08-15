# World Tubes submission evidence ledger

Overall evidence-bundle status: **incomplete**.

This ledger covers generated evidence artifacts only. Venue conversion and the manuscript-package gate remain required.

| Component | Status | Accepted | Input |
|---|---|---:|---|
| frozen_world_scaling | missing | no | `outputs/benchmarks/world_tubes_frozen_world_replay_compiled_v1/coffee_martini_full_300f_progressive_512_v1/seed_17/summary.json` |
| moving_camera_density | missing | no | `outputs/benchmarks/world_tubes_frozen_world_moving_camera_v1/coffee_martini_full_300f_progressive_512_v1/seed_17/summary.json` |
| public_context | missing | no | `src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc` |
| theorem_correctness | accepted | yes | `outputs/benchmarks/2026-07-22_world_tubes_theorem_table/summary.json` |
| variable_camera_closure_death | accepted | yes | `artifacts/paper_evidence/world_tubes_variable_camera_schema_v2_clean/summary.json` |

## Public matrix slots

| # | Role | Protocol | Seed | Policy | Status |
|---:|---|---|---:|---|---|
| 0 | primary_progressive | coffee_martini_full_300f_progressive_512_v1 | 17 | fast_exploration | missing |
| 1 | primary_progressive | coffee_martini_full_300f_progressive_512_v1 | 29 | fast_exploration | missing |
| 2 | primary_progressive | coffee_martini_full_300f_progressive_512_v1 | 43 | fast_exploration | missing |
| 3 | pixel_matched_control | coffee_martini_full_300f_fixed_512_pixel_matched_v1 | 17 | fast_exploration | missing |
| 4 | pixel_matched_control | coffee_martini_full_300f_fixed_512_pixel_matched_v1 | 29 | fast_exploration | missing |
| 5 | pixel_matched_control | coffee_martini_full_300f_fixed_512_pixel_matched_v1 | 43 | fast_exploration | missing |
| 6 | sampler_control | coffee_martini_full_300f_progressive_global_shuffle_512_v1 | 17 | fast_exploration | missing |

## Missing runtime inputs

- `public_context_matrix_summary`: missing — `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2/matrix_summary.json`
- `coffee_martini_full_300f_progressive_512_v1/seed_17/fast_exploration`: missing — `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2/coffee_martini_full_300f_progressive_512_v1/seed_17/run_summary.json`
- `coffee_martini_full_300f_progressive_512_v1/seed_29/fast_exploration`: missing — `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2/coffee_martini_full_300f_progressive_512_v1/seed_29/run_summary.json`
- `coffee_martini_full_300f_progressive_512_v1/seed_43/fast_exploration`: missing — `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2/coffee_martini_full_300f_progressive_512_v1/seed_43/run_summary.json`
- `coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_17/fast_exploration`: missing — `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2/coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_17/run_summary.json`
- `coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_29/fast_exploration`: missing — `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2/coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_29/run_summary.json`
- `coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_43/fast_exploration`: missing — `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2/coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_43/run_summary.json`
- `coffee_martini_full_300f_progressive_global_shuffle_512_v1/seed_17/fast_exploration`: missing — `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2/coffee_martini_full_300f_progressive_global_shuffle_512_v1/seed_17/run_summary.json`
- `frozen_world_scaling`: missing — `outputs/benchmarks/world_tubes_frozen_world_replay_compiled_v1/coffee_martini_full_300f_progressive_512_v1/seed_17/summary.json`
- `moving_camera_density`: missing — `outputs/benchmarks/world_tubes_frozen_world_moving_camera_v1/coffee_martini_full_300f_progressive_512_v1/seed_17/summary.json`
