# Dynaworld Baselines

This file is the canonical, append-friendly tracking sheet for our baselines.
It records:

- which task tier the baseline serves (fast dev probe vs. real novel-view probe)
- which baseline category within that tier (static gsplat, dynamic gsplat,
  token-only, video-encoder)
- the config that defines the run
- the train script that launches it
- the dataset config and the train/eval split
- the most recent W&B run id
- the metrics we ranked it on (PSNR, SSIM, L1, eval loss)
- the optimizer step count and wall-clock time it took
- the device the run was on
- the date the row was last refreshed

When you re-run a baseline with a meaningful change (new code, new init, new
sampler), do **not** silently overwrite the row. Add a new dated row beneath
it, and leave the older row in place so the trail is preserved. Use `loose_notes/`
for the chronological reasoning; this file is the standings table.

If a cell is unknown, mark it `TODO`. Do not backfill metrics by guessing —
empty cells are honest, fabricated cells are landmines.

## Task tiers

We have two task tiers right now. They answer different questions and should
not be compared against each other.

### Tier 1: Fast overfit (dev-loop sanity)

**Question**: Does the pipeline run end-to-end with current code? Does loss
go down on a single direction in a single video?

**Budget**: target ~30 s wall clock. Pure smoke. Source-view overfit only.

**Use it for**: validating a refactor, a renderer change, a config schema
migration, an init change. **Do not** use it to rank models or claim a baseline
beats another; it has zero novel-view signal and zero held-out cameras.

### Tier 2: Multicam novel-camera-angle probe (cheapest real probe)

**Question**: Does the learned representation render a held-out camera from
the same clip? This is the cheapest measurement that targets the actual task
we care about (novel view synthesis).

**Budget**: minutes to ~tens of minutes locally on MPS, depending on baseline
category. Eval is `heldout_eval_psnr` on a never-trained camera.

**Use it for**: ranking baseline categories, checking convergence, comparing
init/feature/architecture choices.

**Current dataset** (`multicam_val_v1_128_4fps_16f`,
`src/dataset_configs/multicam_val_v1_128_4fps_16f.jsonc`): 4 multicam samples,
one per source dataset (AIST gBR/sBM/d04, Neural3D `coffee_martini`,
ViVo `athlete_rows`, DeepView `03_Dog`).

**Target**: expand to ~20 multicam samples covering more scenes/dynamics.
Until then, treat numbers as a small-N probe, not a benchmark score. The
expansion contract lives in
`agent_notes/loose_notes/2026-04-28_16-59-07_stable_fair_benchmark_contract_for_dynaworld.md`.

## Baseline categories

We track four categories side by side, each at both Tier-1 and Tier-2 budgets
(where applicable). Each is a different "base case" — they differ in *what
information the model is allowed to use* and *what kind of primitive it
decodes*. Comparing across them is how we tell whether tokens/encoders earn
their keep over a simple per-frame 3DGS.

| Category | What it is | Why we track it |
|---|---|---|
| **Static gsplat** | Per-frame 3DGS, single shared static scene | Lower bound on capacity; tells us how much the held-out camera can be explained without any temporal modeling at all. |
| **Dynamic gsplat** | Per-frame 3DGS, one set of splats *per frame* | Strong local baseline. Optimizes splats directly, no encoder, no tokens. Currently the strongest on `heldout_eval_psnr` on DeepView 3-cam. |
| **Token-only (no encoder)** | Learned token bank → low-rank Gaussian decoder. No video features fed in. | Isolates how much the *decoder/head parameterization* contributes. Single-clip overfit can be solved from time alone, so this is the unconditioned control. |
| **Video encoder (V-JEPA)** | Video features → cross-attn → token bank → Gaussian head | Tests whether a frozen video backbone actually provides useful 3D/dynamic structure on top of the token decoder. Currently V-JEPA 2.1 ViT-B/384 precomputed. |

## Tier 1: Fast overfit (single video, ~30 s sanity)

**Train data**: `test_data/test_video_384_3fps.mp4` (single clip, no held-out).
**Eval**: source-view reconstruction on the same clip. **No novel-view
signal.**

| Category | Config | Script / how to launch | W&B | Steps | Wall | Eval/Loss | SSIM | PSNR | Device | Last refreshed |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| Token-only smoke | `src/train_configs/local_mac_overfit_video_token_smoke.jsonc` | `PYTHONPATH=src/train uv run python src/train/train.py src/train_configs/local_mac_overfit_video_token_smoke.jsonc` | TODO | 10 | TODO (~30 s target) | TODO | TODO | TODO | MPS | TODO |
| Tiny 30-clip token smoke | `src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc` | `PYTHONPATH=src/train uv run python src/train/train.py src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc` | TODO | TODO | TODO | TODO | TODO | TODO | MPS | TODO |

These rows exist to be re-run on demand, *not* to be ranked. If a smoke run
takes more than ~60 s, treat it as a regression in the dev loop.

## Tier 2: Multicam novel-camera-angle probe

### Tier 2a: DeepView 3-cam train2 / test1 (single scene, 80–250 steps)

**Train data**: 2 cameras of DeepView `03_Dog` (`camera_0001`, `camera_0015`).
**Eval**: held-out `camera_0040`. Same clip, novel viewpoint.
**Render/loss**: 128 px, 16 frames, 2048 primitives (gsplat baselines) or
8192 splats (token/encoder baselines).
**Manifest**: `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl`.

| Category | Config | W&B | Steps | Wall | Train PSNR | **Heldout PSNR** | Notes | Last refreshed |
|---|---|---|---:|---:|---:|---:|---|---|
| Static gsplat (per-frame 3DGS, static scene) | `src/train_configs/local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_static_3dgs_128_16f_2048splats.jsonc` | TODO | 80 | TODO | TODO | TODO | Run via `research_experiments/gauge_fields/run_deepview_3cam_holdout.py --only static_3dgs` | TODO |
| Dynamic gsplat (per-frame, free) | `src/train_configs/local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_free_dynamic_3dgs_128_16f_2048splats.jsonc` | offline (no-wandb run) | 80 | 184.31 s | 16.4423 | **13.2940** | Run via `run_deepview_3cam_holdout.py --only free_dynamic_3dgs`. From `loose_notes/2026-04-27_17-05-00_3cam_free_dynamic_3dgs_baseline.md`. Currently the strongest baseline on this split. | 2026-04-27 |
| Token-only (no encoder) | `src/train_configs/local_mac_ablate_time_static_dynamic_96_32_unconditioned_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc` | TODO (run on 3-cam split) | 1000 | TODO | TODO | TODO | The same unconditioned static/dynamic strong-init cell from the contract. Same-source variant got `qstqjup2` Eval/Loss 0.0588 SSIM 0.771; rerun on 3-cam split. | TODO |
| Video encoder (V-JEPA) | `src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_camera_clamp_video_implicit_128_fast_mac_8192splats_1000step.jsonc` | TODO (run on 3-cam split) | 1000 | TODO | TODO | TODO | Same-source clamped run is `yhezacn8` (Eval/Loss 0.0425, SSIM 0.848). Pending re-run on the actual 3-cam heldout split. | TODO |
| Video encoder (V-JEPA) + F=32 features + 256px (ultimate baseline) | `src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc` | TODO | 1000 | TODO | TODO | TODO | Combines V-JEPA 2.1 ViT-B/384 + static/dynamic 96/32 + cross-attn-4 + DeepView 3-cam train2/test1 + 256px render + 8192 splats + F=32 feature splatting + alpha-aware composition + random per-step bg + LN+kaiming+g4 colorize. Not yet run. | TODO |
| Video encoder (V-JEPA) + F=32 features + 256px, stable LR/camera clamp | `src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha_lr3e4_camclamp.jsonc` | [`iom0ibz8`](https://wandb.ai/nbardy/dynaworld/runs/iom0ibz8) | 1000 | 57m25s | 24.2203 | **8.6923** | Stable version of the ultimate config after aggressive run `nlz1057l` NaN'd around step 46. Train cameras reached PSNR 24.0275 / 24.4131, but heldout `camera_0040` degraded to PSNR 8.6923 / SSIM 0.0711; earlier validation was 9.3172 and the 120-step smoke was 10.5071. See `agent_notes/loose_notes/2026-04-30_22-04-39_vjepa_f32_multicam_heldout_baseline.md`. | 2026-04-30 |

### Tier 2b: Multicam val v1 (4 samples → target 20)

**Train data**: source camera per sample.
**Eval**: target cameras per sample (held-out).
**Manifest**: `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl`.

This tier is the cheapest *real* probe of the task we care about. Right now
the manifest only has 4 samples (one per source dataset). Expansion to ~20
samples is tracked in
`agent_notes/loose_notes/2026-04-28_16-59-07_stable_fair_benchmark_contract_for_dynaworld.md`.

| Category | Config | W&B | Steps | Wall | Source PSNR | **Heldout PSNR** | Heldout SSIM | N samples | Last refreshed |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Static gsplat | TODO (config TBD against `multicam_val_v1`) | TODO | TODO | TODO | TODO | TODO | TODO | 4 → 20 | TODO |
| Dynamic gsplat | TODO (config TBD against `multicam_val_v1`) | TODO | TODO | TODO | TODO | TODO | TODO | 4 → 20 | TODO |
| Token-only | TODO | TODO | TODO | TODO | TODO | TODO | TODO | 4 → 20 | TODO |
| Video encoder (V-JEPA) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | 4 → 20 | TODO |

### Same-source overfit reference (not a probe — diagnostic only)

These are single-clip overfit numbers that exist in the loose notes. They
**do not measure novel-view performance** — they are listed only so we know
what the same-source ceiling looks like for each category, which is useful
for sanity-checking the 3-cam heldout numbers.

All rows are same source-video, 128 px render/loss, 8192 splats, static/dynamic 96/32
where applicable. Pulled from `loose_notes/2026-04-27_16-46-57_static_dynamic_vjepa_matrix_completion.md`.

| Category | W&B | Steps | Wall | Eval/Loss | L1 | SSIM | **PSNR** | cam adj rot | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Unconditioned static/dynamic 96/32, 250 steps** | [`twh5to1q`](https://wandb.ai/nbardy/dynaworld/runs/twh5to1q) | 250 | **216 s (~3.6 min)** | 0.0762 | 0.0555 | 0.6815 | **21.36** | 0.0244 | **Fastest path to solid PSNR. No video embeddings at all.** |
| Unconditioned static/dynamic 96/32, 1000 steps | [`qstqjup2`](https://wandb.ai/nbardy/dynaworld/runs/qstqjup2) | 1000 | 912 s (~15 min) | 0.0588 | 0.0448 | 0.7706 | 23.03 | 0.0160 | Encoder-free, nearly matches the V-JEPA interrupted run at half the wall clock. |
| Local static/dynamic 96/32 (no V-JEPA), 250 steps | [`sc25ek8t`](https://wandb.ai/nbardy/dynaworld/runs/sc25ek8t) | 250 | 488 s | 0.1195 | 0.0779 | 0.4287 | 18.42 | 0.0159 | Local video encoder; weaker than the unconditioned token bank at this step count. |
| Local static/dynamic 96/32 (no V-JEPA), 1000 steps | [`x803a6ra`](https://wandb.ai/nbardy/dynaworld/runs/x803a6ra) | 1000 | 1587 s | 0.0781 | 0.0551 | 0.6599 | 21.49 | 0.0334 | |
| V-JEPA static/dynamic 96/32, 250 steps | [`oaor6um2`](https://wandb.ai/nbardy/dynaworld/runs/oaor6um2) | 250 | 810 s | 0.0881 | 0.0615 | 0.6109 | 20.29 | 0.1309 | First V-JEPA + split run. Cache prebake excluded from runtime. |
| V-JEPA static/dynamic 96/32, ~525 steps (interrupted) | [`mybv736f`](https://wandb.ai/nbardy/dynaworld/runs/mybv736f) | 520 | 2424 s | 0.0547 | 0.0413 | 0.7836 | 23.69 | 0.1827 | "Visually almost perfect." Cache-hit. |
| V-JEPA static/dynamic 96/32, 1000 steps | [`x4uc6va3`](https://wandb.ai/nbardy/dynaworld/runs/x4uc6va3) | 1000 | 1486 s | 0.0455 | 0.0360 | 0.8336 | 24.93 | 0.2909 | Best same-source unclamped. Larger camera adjacent motion is a flag. |
| V-JEPA static/dynamic 96/32, camera-clamped, 1000 steps | [`yhezacn8`](https://wandb.ai/nbardy/dynaworld/runs/yhezacn8) | 1000 | TODO | 0.0425 | TODO | 0.848 | 25.26 | clamped | Strongest same-source clamped result. |
| Free splats (raw splat-param gradient) | [`kttrbewl`](https://wandb.ai/nbardy/dynaworld/runs/kttrbewl) | TODO | TODO | 0.2633 | TODO | TODO | TODO | TODO | Raw splat-grad floor. |
| Token-only (no static/dynamic split) | [`xenc4w06`](https://wandb.ai/nbardy/dynaworld/runs/xenc4w06) | TODO | TODO | 0.1439 | TODO | TODO | TODO | TODO | Plain learned tokens without the static/dynamic split. |
| Local video encoder, strong init | [`bbk7maml`](https://wandb.ai/nbardy/dynaworld/runs/bbk7maml) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | |
| V-JEPA frozen, strong init (no split) | [`pwvybmao`](https://wandb.ai/nbardy/dynaworld/runs/pwvybmao) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | |

## Fastest path to a "solid" same-source PSNR (static/dynamic gsplat token split)

Question this answers: *if I want a passable PSNR on the static/dynamic
gsplat token-split recipe as quickly as possible, what should I run?*

These are all single-clip same-source overfits (Tier 1-style, not novel-view).
The numbers come from the diagnostic table above.

| Tier | Recipe | Wall | PSNR | SSIM | Eval/Loss | When to use |
|---|---|---:|---:|---:|---:|---|
| **Recommended fast solid** (no video embeds) | **Unconditioned static/dynamic 96/32, 250 steps** (`twh5to1q`) | **216 s (~3.6 min)** | **21.36** | **0.6815** | 0.0762 | Encoder-free token bank with the split + strong init. Fastest measured path to solid same-source PSNR. |
| Stronger no-encoder | Unconditioned static/dynamic 96/32, 1000 steps (`qstqjup2`) | 912 s (~15 min) | 23.03 | 0.7706 | 0.0588 | No V-JEPA, nearly matches the interrupted V-JEPA `mybv736f` run at half the wall clock. |
| First V-JEPA recipe | V-JEPA static/dynamic 96/32, 250 steps (`oaor6um2`) | 810 s (~14 min) | 20.29 | 0.6109 | 0.0881 | Slower **and** lower PSNR than the unconditioned 250-step row. Use only when the experiment specifically needs the V-JEPA conditioning path. |
| Visually "almost perfect" | V-JEPA static/dynamic 96/32, ~525 steps (`mybv736f`) | 2424 s (~40 min) | 23.69 | 0.7836 | 0.0547 | Clearly sharp media. |
| Best same-source overall | V-JEPA static/dynamic 96/32, 1000 steps (`x4uc6va3`) | 1486 s (~25 min) | 24.93 | 0.8336 | 0.0455 | Strongest unclamped result, but largest camera adjacent rotation (`0.29 deg`) — a flag. |
| Best clamped | V-JEPA static/dynamic 96/32 camera-clamped, 1000 steps (`yhezacn8`) | TODO | 25.26 | 0.848 | 0.0425 | Strongest same-source clamped result; camera motion bounded. |

Caveats:

- **All Tier-1 numbers are same-source overfit on a single clip**. PSNR 20+
  here does not mean PSNR 20+ on a held-out camera. None of these have been
  re-run on the Tier-2a 3-cam DeepView heldout split yet — that is the
  highest-priority TODO below.
- **Cache prebake adds wall time** on the *first* V-JEPA run for a given
  config (cache key
  `ablate-time-static-dynamic-96-32-vjepa2-1-vitb-384-small128-max16-v1`,
  ~6.8 MB cached tensor). Quoted wall is cache-hit runtime.
- **Unconditioned static/dynamic strong-init** (`qstqjup2`, 1000 steps,
  Eval/Loss 0.0588 / SSIM 0.771) closes most of the V-JEPA gap, suggesting a
  big chunk of the V-JEPA win is the split + init + decoder rather than the
  features themselves. Re-running the unconditioned recipe at 250 steps with
  wall-clock recorded would establish a faster, encoder-free fallback.

## Reruns needed (priority-ordered TODOs)

These are the runs missing from this file. They are listed roughly in
priority order — Tier 2 first because that is where the actual task signal
lives.

1. **Tier 2a static gsplat (per-frame static 3DGS)** on DeepView 3-cam train2/test1.
   Config exists; just needs a run via
   `research_experiments/gauge_fields/run_deepview_3cam_holdout.py --only static_3dgs`.
   ~1–3 min wall expected based on the dynamic variant (184 s at 80 steps).
2. **Tier 2a free dynamic gsplat with W&B enabled.** The recorded `13.2940`
   heldout PSNR run was `--no-wandb`; we should redo it once with wandb so
   the row has a link. Same script, drop `--no-wandb`.
3. **Tier 2a token-only (unconditioned static/dynamic strong-init).** Take
   the `qstqjup2` recipe and apply it to the 3-cam `multicam_val` split.
   This is the cheapest test of "does the token-split decoder generalize to
   a held-out camera at all, with no V-JEPA?"
4. **Tier 2a V-JEPA static/dynamic on 3-cam.** Same recipe as `yhezacn8`
   on the 3-cam multicam split. Direct apples-to-apples vs. the dynamic
   gsplat baseline on heldout PSNR.
5. **Tier 1 smoke wall-clock** for `local_mac_overfit_video_token_smoke.jsonc`
   and `local_mac_tiny_30_video_token_smoke.jsonc`. One-shot timing run; no
   metric ranking needed. Goal: confirm the dev-loop smoke is actually <60 s.
6. **Backfill missing same-source rows**: step counts and wall-clock for
   `xenc4w06` (token-only), `bbk7maml` (local encoder), `pwvybmao` (V-JEPA
   no-split), `kttrbewl` (free splats), `x4uc6va3` (V-JEPA 1000-step), and
   `yhezacn8` (V-JEPA clamped 1000-step). Pull from W&B run pages — these
   are recorded there but not transcribed into the loose notes.
7. **Tier 2b multicam_val v1**: pick configs and run all four categories on
   the existing 4-sample manifest before expanding to 20.

## Conventions for editing this file

- **Run id, not date, identifies a row**: keep the W&B id even if the metrics
  change later, so a future reader can re-pull the run from W&B.
- **Wall clock includes data loading and W&B logging**, not just training
  inner-loop time. Numbers should be reproducible from the listed
  config + script on the listed device.
- **Heldout PSNR is the primary selector for Tier 2**, not source PSNR.
- **If you ran the baseline, add the row.** If you only have a partial
  number, fill it and mark the rest TODO. Do not wait for a "complete" run.
- **Linked notes**: every refreshed row should cite the loose note that
  produced it, so the chronology and caveats are findable.
