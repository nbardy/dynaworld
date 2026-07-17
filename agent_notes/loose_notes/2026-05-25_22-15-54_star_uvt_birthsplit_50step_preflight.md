# STAR UVT Birth/Split 50-Step Preflight

Context:
    Softmax-GS now has enough implementation evidence to avoid a blind STAR or
    WorldFoam port: the dynamic-GS repeat/scale rows are mixed, not promoted.
    The practical next lane is back to STAR UVT support work, specifically the
    selected `K=8/r64/o0.4` birth/split support gate.

What changed:
    Added the 50-step continuation config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_50step_media.jsonc`.

    Added the preflight verifier:
    `research_experiments/star_uvt_feature_tubes/preflight_support_birthsplit_gate.py`.
    It checks the selected support-gate config values, required input artifact
    paths, and the feature-cache directory before launching a trainer.

    Added focused tests:
    `tests/test_star_uvt_support_birthsplit_preflight.py`.

Validation:
    `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_support_birthsplit_preflight.py -q`
    passed with `4 passed`.

    `PYTHONPATH=src/train .venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/preflight_support_birthsplit_gate.py`
    passed.

    Ran the preflight with `--allow-missing` so the blocked state is recorded:
    `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_r64_o04_50step_preflight/summary.md`.

Result:
    The 50-step config parses and matches the selected support gate:
    `feature_direct_gradcache_reduce_vec4`, `8192` tubes, tile cap `128`,
    `support_birth_split.enabled=true`, `center_strategy=farthest_xy`,
    `center_count=8`, `reallocate_tubes=32`, isotropic `64px` support,
    opacity `0.4`, target source `uncovered_brightness`, global step offset
    `1500`, and `train.steps=50`.

    The local workspace is blocked before launch on missing required inputs:
    `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`
    and
    `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt`.
    The checked-in video exists. The feature cache directory
    `outputs/feature_cache/star_uvt_feature_targets/vjepa2_1_vitb_256crop_64f`
    is also missing, recorded as a warning because it can be regenerated but
    may require a slow V-JEPA bake/download.

Decision:
    Do not launch the 50-step STAR continuation from this checkout until the
    missing checkpoints are restored or regenerated. Once they exist, run the
    preflight without `--allow-missing`, then launch the config and compare the
    50-step media/support output against the existing 20-step `r64/o0.4` row.
