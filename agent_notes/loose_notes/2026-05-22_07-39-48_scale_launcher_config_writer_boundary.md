# Scale Launcher Config Writer Boundary

## Context

The scale/pretrain shell launchers had already been moved onto
`trainer_registry.resolve_config_for_arch(...)` and `run_config_dict(...)` for
embedded Python checks and in-memory probe runs. Two embedded Python snippets
still wrote complete patched config files with direct
`Path(...).write_text(json.dumps(...))`.

## Change

- `src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh`
  now imports `train_artifacts.write_json` in `write_sample_config(...)` and
  uses it for the per-record temporary config.
- `src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh`
  now imports `train_artifacts.write_json` in `write_smoke_config(...)` and
  uses it for the one-step smoke config.

## Deliberate Non-Changes

- The 1k smoke manifest copy stays as direct text writing because it is JSONL
  row copying, not a complete JSON object artifact.
- The launch scripts still choose configs and modes in shell. They are not being
  turned into Python entrypoints in this slice.

## Validation

- `bash -n` passed for both touched shell scripts.
- `bash src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh check`
  passed and exercised the patched per-record config writer.
- `bash src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh check`
  passed and resolved the 1k lazy manifest config.
- A standalone smoke-config writer probe exercised the same 1k smoke-config
  logic and confirmed both the smoke config and copied smoke manifest were
  written.
- Targeted `git diff --check` passed.

## Current State

This continues the launch/config boundary cleanup without changing trainer
semantics. The remaining shell-script cleanup should keep the same standard:
route complete reusable artifacts through helpers, leave stream/row/text
formats local when their semantics differ.
