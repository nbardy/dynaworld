# Gauge Sweep Config Helpers

## Goal

Continue the trainer/interface cleanup by removing a live, narrow duplication in
the Gauge Fields experiment surface without touching trainer math.

## Change

- Added Gauge-local helpers in `research_experiments/gauge_fields/common.py`:
  - `parse_csv_strings(...)`
  - `parse_csv_ints(...)`
  - `parse_csv_floats(...)`
  - `parse_csv_bools(...)`
  - `clone_jsonable(...)`
  - `write_generated_jsonc(...)`
- Routed `research_experiments/gauge_fields/make_sweep_configs.py` through
  those helpers.
- Kept sweep-specific concerns local to the generator:
  - float/bool slug formatting
  - support/incidence/radius/alpha loop structure
  - derived-support tag construction
  - config patching and output filenames

## Validation

Commands run from the Dynaworld root:

```bash
rtk .venv/bin/python -m py_compile research_experiments/gauge_fields/common.py research_experiments/gauge_fields/make_sweep_configs.py
rtk uv run python research_experiments/gauge_fields/make_sweep_configs.py --help
rtk rm -rf /tmp/dynaworld_gauge_sweep_helper
rtk uv run python research_experiments/gauge_fields/make_sweep_configs.py --output-dir /tmp/dynaworld_gauge_sweep_helper --elements 16 --radii 0.05 --alpha-logits 0.0 --support-modes screen_disk --incidence-modes projected_conic --steps 1 --disable-wandb --wandb-mode offline
```

The generation smoke wrote:

```text
/tmp/dynaworld_gauge_sweep_helper/local_mac_gauge_fields_screen_disk_projected_conic_motion_128_16f_16el-r0p05-a0-1step.jsonc
```

Spot checks confirmed:

- generated header is still present
- `model.num_elements = 16`
- `model.init_radius = 0.05`
- `model.init_alpha_logit = 0.0`
- `train.steps = 1`
- `logging.log_to_wandb = false`
- `logging.wandb_mode = offline`

## Why This Matters

This keeps the cleanup pattern small and concrete: common parsing/artifact
helpers live in the Gauge common module, while experiment-specific sweep
semantics stay in the sweep script. No training objective, optimizer, renderer,
or result schema was changed.
