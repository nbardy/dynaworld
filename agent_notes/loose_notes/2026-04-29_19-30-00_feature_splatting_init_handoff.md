# Feature Splatting Init — Handoff to Other Scientists

Date: 2026-04-29
Context: Adding F=32 feature splatting on top of the unconditioned token-GS
baseline. The basic pipeline is live (Codex's v5_features rasterizer +
trainer-side colorize MLP), but two open issues remain that need investigation.

## What's working

- F=32 forward + backward end-to-end. Run command:

  ```bash
  PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py \
    src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc
  ```

- v5_features rasterizer fork at `third_party/fast-mac-gsplat/variants/v5_features/`
  is built for Python 3.11 and dispatched in `src/train/renderers/fast_mac.py`
  by `rgbs.shape[-1]` (F=3 → v5, else → v5_features).

- Colorize MLP (1×1 Conv F→3 + sigmoid, optional hidden + GELU) at
  `src/train/colorize.py`. Knobs: `hidden_dim`, `activation`, `pre_norm`,
  `weight_init`, `weight_init_gain`, `view_condition`.

- PCA-of-features video logged at `Feature_PCA_Video` in W&B at image/video
  log intervals via `src/train/feature_pca_viz.py`.

## Open issues (the work that's outstanding)

### Issue 1: Splats stay clustered at center, don't spread

In all F=32 runs (default-kaiming, LN+orth-g3, LN+kaiming-g4 — see images
the user attached), the rendered splats cluster at the screen center and
don't migrate outward, *unlike the F=3 baseline which started clustered
and spread out fine within ~200 steps*.

**Most likely root cause** (full reasoning in
`2026-04-29_18-15-00_colorize_gray_collapse_hypotheses.md` + chat thread):
the F=3 path had a strong miss-pixel loss (`pixel = (1 - alpha) · white`,
loss ≈ 0.5–0.8 in dark GT regions) that drove splats outward via Gaussian
tail gradients. The F=32 path's colorize MLP maps any constant feature
input (which is what background pixels carry post-rasterizer) to a fixed
~gray RGB, collapsing miss-pixel loss to ~0.05–0.2. Splats lose the outward
gradient force.

**Proposed fix**: alpha-aware composition. Get the accumulated alpha mask
out of the rasterizer and composite white background in RGB space
*after* colorize:

```
splat_rgb = colorize(rendered_features)
final_rgb = alpha · splat_rgb + (1 - alpha) · [1, 1, 1]
```

Two implementation paths:
- Marker-channel hack: `feature_dim=33`, force channel 32 to 1.0 in head,
  set `feature_background[32] = 0.0`. After raster, channel 32 = alpha.
  Works without touching v5_features.
- Clean fix: extend v5_features to return alpha as a second output. Codex
  job.

### Issue 2: 10× more color params in F=32 — does LR need adjustment?

User raised the question. My read: probably not the primary issue, since
xyz/scale/opacity heads have *identical* parameter counts in F=32 vs F=3
and Adam normalizes per-parameter. But hasn't been measured. Confirm by
logging gradient norms.

## Tools written for diagnosing init / training health

### `src/train/init_diagnostics.py` (extended)

Existing functions kept untouched. Two new functions added:

- `post_colorize_image_diagnostics(rgb_image, *, logits=None)` — returns
  per-channel and per-pixel statistics on the post-colorize-MLP RGB output.
  Key metrics:
  - `PerPixelChroma/Mean` — std across (R,G,B) per pixel. Near 0 = grayscale.
  - `SpatialStdMean` — std across pixels per channel. Near 0 = uniform image.
  - `Range` — max - min over the image. Near 0 = collapsed.
  - `R/G/B/Std`, `R/G/B/Mean`, `R/G/B/Entropy01` — per-channel stats.
  - `Logit/AbsGt0p5Frac`, `Logit/AbsGt2Frac`, `Logit/AbsGt4Frac` — pre-sigmoid
    saturation fractions. The healthy band is `AbsGt2 ∈ [0.10, 0.30]` and
    `AbsGt4 < 0.05`. Larger AbsGt4 = many pixels in sigmoid's dead zone.

- `format_colorize_init_summary(metrics)` — formats the metrics into a
  one-line summary string.

### `src/train/probe_colorize_init.py`

Loads a config, builds the model + colorize, runs ONE forward + raster +
colorize end-to-end, prints metrics. Works for both F=3 and F=32 configs
(same metrics function — apples-to-apples).

```bash
PYTHONPATH=src/train uv run python src/train/probe_colorize_init.py \
  src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc \
  --seeds 0,1,2
```

### `src/train/probe_colorize_matrix.py`

Sweeps a hardcoded matrix of (`pre_norm`, `weight_init`, `weight_init_gain`,
`hidden_dim`) cells against the cached rendered features. Renders features
ONCE per seed (heavy step ~3 s on MPS), then iterates ~20 colorize-init
variants in sub-millisecond each. Total wall: ~10–15 s for 3 seeds × 20
cells.

```bash
PYTHONPATH=src/train uv run python src/train/probe_colorize_matrix.py \
  src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc \
  --seeds 0,1,2
```

To add a new cell: append a tuple to `CELLS` at the top of the script.

## Configs you can iterate on

All under `src/train_configs/`:

- `local_mac_unconditioned_tokens_fast.jsonc` — F=3 baseline (legacy RGB
  path, no feature splatting). Train command:

  ```bash
  PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py \
    src/train_configs/local_mac_unconditioned_tokens_fast.jsonc
  ```

  Recent run: `qstqjup2` (1000 steps, Eval/Loss 0.0588, SSIM 0.7706, PSNR 23.03).

- `local_mac_unconditioned_tokens_features_F32.jsonc` — F=32 with the user's
  current chosen knobs (currently `hidden_dim: null`, `activation: sigmoid`,
  `view_condition: none`). Default-kaiming colorize at this point.

- `local_mac_unconditioned_tokens_features_F32_LN_orth_g3.jsonc` — F=32 with
  LN + orthogonal init + gain=3. Recent 200-step run: `gwdmm5cc` (Loss 0.1818).

- `local_mac_unconditioned_tokens_features_F32_LN_kaiming_g4.jsonc` — F=32
  with LN + Kaiming + gain=4. Recent 200-step run: `2nz89pj3` (Loss 0.1118 —
  notably *better* than orth-g3 despite worse init metrics, likely because
  fewer pixels in sigmoid's saturated tail).

## Knobs that have been tweaked (and what we found)

### Colorize architecture (in `src/train/colorize.py`)

| Knob | Values tested | Finding |
|---|---|---|
| `pre_norm` | False, True | LN does most of the heavy lifting — 5× spatial std improvement at gain=1, by forcing colorize input to unit variance which is what Kaiming was designed for. |
| `weight_init` | "kaiming", "orthogonal" | Orthogonal gives ~2× per-pixel chroma at the same gain (decorrelates 3 output channels). But Kaiming with bigger gain trained better — ~3× lower deep-saturation (Logit|>4|=0.019 vs 0.055) means more pixels stay gradient-rich. |
| `weight_init_gain` | 1, 2, 3, 4, 5, 7, 10 | Healthy band: post-conv std ~1.5–3 (Logit|>2| ∈ [0.10, 0.30]). With LN giving unit-variance input: orth needs gain ~3, Kaiming needs gain ~4. Without LN: orth needs gain ~7, Kaiming hits saturation issues. |
| `hidden_dim` | None, 8, 16, 32 | At init: hidden costs ~10% spatial std (compounds variance loss through layers). At training: untested, but adds 6× per-pixel multiplies. **Decision: drop hidden** — the cost is real for browser inference and the init benefit is negative. |
| `activation` | "sigmoid", "identity" | Only sigmoid tested. Identity would lose the [0,1] clamp. |
| `view_condition` | "none", "camera_center_ray", "pixel_ray" | Added by user to colorize.py; not yet tested in matrix probe. View conditioning could give per-pixel direction info that breaks the colorize MLP's symmetry — worth a sweep. |

### Upstream init (in `local_mac_unconditioned_tokens_features_F32.jsonc`)

Most upstream knobs were inherited from the F=3 baseline and not retuned for F=32:

| Knob | F=3 default | F=32 status | Worth retuning? |
|---|---|---|---|
| `head_output_init_std` | 0.12 | unchanged | Maybe — increases per-splat feature magnitude. But LN renormalizes, so probably small effect on what colorize sees. |
| `opacity_init` | 0.1 | unchanged | Maybe — bigger opacity = more contribution per splat, more gradient signal. |
| `scale_init` | 0.02 | unchanged | **Likely useful as a band-aid for the splat-clustering issue** — bigger initial Gaussians have wider tails, more pixels generate nonzero gradient on each splat. Try 0.04, 0.06. Not a permanent fix. |
| `position_init_extent_coverage` | 0.9 | unchanged | Already spreads splats at init. Issue is they don't *stay* spread. |
| `rgb_init` | "uniform" | silently ignored at F!=3 (head outputs raw features, not RGB-bounded) | The F=3 path's per-splat color diversity at init isn't preserved. Could help to add an analog: random orthogonal feature directions per splat. Not implemented. |

## Open hypotheses worth testing

1. **Alpha-aware compositing fixes the splat-clustering issue** (most important).
   Test: marker-channel hack in the trainer. ~30 min of plumbing. If splats
   spread immediately, hypothesis confirmed. Then escalate to Codex for clean
   v5_features alpha output.

2. **Geometry gradient norms are weak in F=32** (validates the above).
   Add `grad.norm()` logging on `static_gaussian_heads.xyz_head.weight`,
   `scale_head`, `opacity_head`, `rgb_head`, `colorize.net` to `val_log`.
   Compare F=3 vs F=32 magnitudes. If F=32 xyz_head grad is e.g. 100× smaller
   than F=3, confirms the "weak miss-loss → weak geometry gradient" theory.

3. **Lower `alpha_threshold` makes Gaussian tails reach further at the cost
   of compute**. The current 1/255 cull means splats only get gradient from
   pixels within their visible footprint. Try 1/1024 or 1/4096. Diagnostic
   only — not a permanent fix.

4. **View conditioning helps** (`view_condition: pixel_ray`). The 3 extra
   ray-direction channels per pixel give the colorize MLP per-pixel signal
   that breaks the gray-collapse without needing LN. Untested in the matrix.

5. **Per-splat feature direction init (analog of `rgb_init=uniform`)**.
   Initialize `rgb_head.bias` for F=32 such that each splat's feature vector
   points in a different direction in 32-space at init. Might restore the
   per-splat color diversity that gave F=3 splats gradient signal.

## Files to read before picking this up

- This handoff: `2026-04-29_19-30-00_feature_splatting_init_handoff.md`
- Hypotheses doc: `2026-04-29_18-15-00_colorize_gray_collapse_hypotheses.md`
- Plan doc: `2026-04-29_17-30-00_feature_splatting_plan.md`
- Codex handoff: `2026-04-29_17-30-00_codex_handoff_v5_features_rasterizer.md`
- Build & run conventions: `AGENTS.md` ("Build & Run Conventions" section).
  Specifically the warning about not `cd`'ing into v5_features and the
  smoke-test rule.

## Don't do these things (lessons from this session)

- Don't add init knobs without re-running the matrix probe across both F=3
  and F=32 configs. The F=32 path has surprising failure modes (e.g.,
  saturation-driven gradient starvation, gray-background-collapse) that
  don't show in F=3.
- Don't trust `py_compile` as a smoke check. It catches import/syntax errors
  but misses tuple-arity mismatches and dataclass field renames. Always run
  the 1-step F=32 smoke after architecture-touching edits — see AGENTS.md.
- Don't `cd` into a fast-mac-gsplat variant directory mid-session. The
  variant `pyproject.toml` lacks a `[project]` table and `uv` aborts when
  it finds itself walking up from there. Use the subshell-parens pattern
  if you must `cd` for a build.
