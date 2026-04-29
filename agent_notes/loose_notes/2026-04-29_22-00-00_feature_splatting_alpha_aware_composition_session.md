# Feature Splatting + Alpha-Aware Composition: Full Session Note

Date: 2026-04-29 22:00
Context: Single long session adding F=32 feature splatting on top of the
unconditioned token-GS baseline (`local_mac_unconditioned_tokens_fast.jsonc`,
parent run `qstqjup2`). Started from a working v5_features rasterizer fork
delivered by Codex earlier the same day, hit a colorize-init gray collapse,
swept inits, ran two 200-step ablations, mis-diagnosed the win condition,
re-diagnosed the actual root cause as the colorize MLP absorbing
background-pixel supervision via Path B (bias shortcut), designed an
alpha-aware composition fix, handed off the rasterizer extension to Codex,
plumbed alpha through the trainer, and confirmed F=32 now beats the F=3
baseline at 400 steps.

## TL;DR

- F=32 feature splatting on top of unconditioned token-GS works end-to-end
  via alpha-aware composition: `final_rgb = α · colorize(features) + (1-α) · white`.
- The earlier "splats clustered at center, won't spread" failure across all
  F=32 init variants was structurally caused by the colorize MLP absorbing
  background-pixel supervision through its bias (Path B / shortcut learning),
  not by colorize-MLP weight init or upstream feature-head init.
- 400-step run `3reqcya9` (F=32, LN+Kaiming-g4, alpha-aware) lands at Eval/Loss
  0.0653, beating same-horizon F=3 baseline `azufyx9e` at 0.0749.
- The init-matrix sweep correctly identified the saturation-vs-engagement
  tradeoff between LN+orth-g3 and LN+Kaiming-g4 but mis-ranked them: the
  better init by static spatial-std lost in actual training because of
  deeper sigmoid-tail saturation (Logit|>4|).

## What feature splatting is here

Each splat outputs an `F`-channel feature vector (default `F=32`) instead of a
fixed RGB-3. The rasterizer accumulates an F-channel feature map per pixel.
After rasterization, a per-pixel 1×1-conv MLP (`FeatureToColor` in
`src/train/colorize.py`) maps features to RGB. PCA of the F-channel feature
buffer (`src/train/feature_pca_viz.py`) is logged at image/video intervals as
`Feature_PCA_Video` for diagnosis. The upstream model
(`UnconditionedTokenGSImplicitCamera`), token bank, static/dynamic split,
camera heads, time projector, loss, and projection math all stay the same.

## The arc of the session

Codex landed `third_party/fast-mac-gsplat/variants/v5_features/` earlier in
the day with full F-channel parity (`max_abs=0` at F=3 vs v5; gradient checks
clean at F={3, 8, 32}). The session opened with the trainer-side wiring
ready: feature head, colorize MLP, PCA viz, F=3 parity smoke. The first
F=32 run produced rich PCA features but a uniform gray-teal RGB image.
The diagnosis (`2026-04-29_18-15-00_colorize_gray_collapse_hypotheses.md`)
identified the failure as pre-sigmoid std ≈ 0.15 — sigmoid stuck in its
linear band — and proposed weight-gain, bias-spread, and identity-passthrough
hypotheses. Codex independently ran a one-step gradient probe
(`2026-04-29_19-29-37_codex_changes_handoff_for_claude.md`), found the F=32
default's xyz-grad was ~10× weaker than F=3, and patched the F=32 config
with `pre_norm=true`, `weight_init="kaiming"`, `weight_init_gain=2.0`. The
init was no longer gray, but splats still clustered.

We then built a fast init matrix probe (`probe_colorize_matrix.py`) that
renders features once per seed (~3 s on MPS) and iterates ~21 colorize-init
cells in sub-millisecond each, total ~15 s for 3 seeds. The matrix declared
LN+orth-g3 as the highest-spatial-std cell. We ran two 200-step training
ablations side by side: `gwdmm5cc` (LN+orth-g3, Loss 0.1818) and `2nz89pj3`
(LN+Kaiming-g4, Loss 0.1118). LN+Kaiming-g4 won despite worse init metrics —
the deciding factor was less deep-sigmoid saturation (Logit|>4|=0.019 vs
orth's 0.055), so more pixels stayed gradient-rich during training.

The user pointed out that the splat-clustering symptom predated all the init
sweeps: the original default-Kaiming F=32 run already clustered. So the init
work could only ever be a second-order fix. The real diagnosis emerged as
"the colorize MLP is absorbing background-pixel supervision via its bias"
(Path B in the discussion). With white background composited *in feature
space before colorize*, every miss-pixel feeds a constant feature vector
into the MLP, and the MLP has a learnable per-output bias that can absorb
all of that supervision into the bias by saturating the sigmoid toward
gray. The geometric supervision (push splats outward to fill missed pixels)
that the F=3 path got from `(1-α) · white` in RGB space is removed entirely.

The fix is to bypass the MLP for background pixels and composite white in
RGB space *after* colorize:

```
splat_rgb = colorize(rendered_features)   # what splats want to paint
final_rgb = α · splat_rgb + (1 - α) · white_RGB
```

Now `(1-α)` is structurally non-learnable and saturated (white in RGB is
literally the value the prediction must match in background regions), so
the only way to make a background pixel match a non-white GT pixel is to
push splats into it. The colorize MLP's gradient on a fully-empty pixel is
multiplied by `α=0` and vanishes — there's no bias-absorption path left.

We wrote `2026-04-29_20-30-00_codex_handoff_v5_features_alpha_output.md` for
Codex with the synthetic-channel equivalence proof: `accumulated_alpha =
1 - T_final` is identical to a feature channel where every splat has color
`1.0` and the background is `0.0`. The implementation strategy is to reuse
all existing per-channel feature backward code with the synthetic channel's
color-grad write skipped — no new gradient math. Codex landed it, returned
v5_features as `(features, alpha)`, and shipped Tests A–E (forward shape,
backward to geometry, parity vs synthetic-channel, combined-grad linearity,
F=3-vs-v5 parity).

We then plumbed alpha end-to-end through the trainer: `render_fast_mac_3dgs`
and `_batch` return the tuple, `render_gaussian_frames_alpha_aware` exposes
it without breaking the legacy single-tensor `render_gaussian_frames` API,
and `recon_backward` / `render_full_sequence` do the post-colorize
composition. Two new W&B videos were added: `Alpha_Mask_Video` (raw alpha
heatmap) and `Render_Composite_Video` (4-column GT|Pred|Alpha|Feature-PCA).
Run `3reqcya9` (F=32 alpha-aware 400 steps) hit Eval/Loss 0.0653, beating
F=3 reference `azufyx9e` (400 steps, Loss 0.0749). Splats spread visibly
across the frame instead of clustering at center.

A separate symptom remains, captured in a parallel `TODO/` document this
session: the alpha mask shows variance instead of converging to a clean
foreground/background split. White scene content (sky, light grass, clouds)
gets explained by the white background composite rather than splats, which
will hurt depth and novel-view synthesis.

## Files changed and what they do

- `src/train/colorize.py` — `FeatureToColor` 1×1-conv module with knobs
  `hidden_dim`, `activation`, `pre_norm` (LayerNorm), `weight_init`
  (`kaiming` | `orthogonal`), `weight_init_gain`, `view_condition`
  (`none` | `camera_center_ray` | `pixel_ray`), `detach_view_condition`. The
  `_run` helper accepts both `[B, F, H, W]` and `[B, T, F, H, W]`. A
  legacy-parity case (F=3, hidden=None, kaiming, gain=1.0, no LN, no view)
  identity-initializes the conv so the F=3 RGB baseline stays bit-equivalent.
- `src/train/feature_pca_viz.py` — `feature_pca_to_rgb([..., F, H, W])`
  → `[..., 3, H, W]`, detached, top-3 PCA components, per-channel min-max
  normalized to [0, 1], MPS→CPU SVD fallback because MPS SVD is unsupported.
- `src/train/init_diagnostics.py` — added `post_colorize_image_diagnostics`
  (PerPixelChroma, SpatialStdMean, Range, R/G/B Std/Mean/Entropy01,
  Logit/AbsGt0p5/2/4 saturation fractions) and
  `format_colorize_init_summary`. Existing `decoded_gaussian_init_diagnostics`
  unchanged.
- `src/train/probe_colorize_init.py` — single-config probe; loads config,
  builds model + colorize, runs one forward + raster + colorize, prints
  metrics across seeds. Works for both F=3 and F=32 with the same metrics.
- `src/train/probe_colorize_matrix.py` — sweeps a hardcoded `CELLS` matrix
  of ~21 cells (`pre_norm × weight_init × weight_init_gain × hidden_dim`)
  against cached rendered features. Renders features once per seed (heavy),
  then iterates colorize variants in sub-millisecond.
- `src/train/renderers/fast_mac.py` — `render_fast_mac_3dgs` and
  `_batch` now return `(features, alpha_or_None)`. F=3 returns
  `(image, None)` (legacy v5 path); F!=3 routes to v5_features and returns
  `(features, alpha_mask)`. The bridge unpacks the tuple defensively
  (`isinstance(rasterize_out, tuple)`) so partially-built v5_features states
  during dev don't break the F=3 path.
- `src/train/rendering.py` — added
  `render_gaussian_frames_alpha_aware(...) -> tuple[Tensor, Tensor | None]`
  for callers that need alpha. Existing `render_gaussian_frames` and
  `render_gaussian_frame` strip the alpha to preserve their legacy
  single-tensor return contract for non-alpha-aware callers.
- `src/train/train_video_token_implicit_dynamic.py` — large surface change:
  - `render_clip_sequence(...) -> tuple[Tensor, Tensor | None]` (was Tensor).
  - `recon_backward` chunked path: each chunk now does
    `chunk_renders = α · colorize(features) + (1-α) · 1.0` when alpha is
    present; falls back to plain colorize for None-alpha (F=3) chunks.
  - `recon_backward` non-chunked path mirrors the same composition.
  - `render_full_sequence` allocates `alpha_frames` lazily on the first
    chunk that returns a non-None alpha; stays `None` for F=3.
  - `validation_video_payload` adds `Alpha_Mask_Video` (single-channel α
    rendered as grayscale) and `Render_Composite_Video` (4-column
    GT | Pred | Alpha | Feature-PCA contact strip).
  - `KnownCameraTrainer.render_full_sequence` mirrors the same alpha-aware
    composition.
  - `MODEL_OPTION_DEFAULTS` gained `feature_dim`.
  - `Trainer.__init__` constructs `FeatureToColor` from `cfg["colorize"]`
    when `feature_dim != 3` (or for the legacy parity case at F=3) and
    adds its parameters to the optimizer.
- `third_party/fast-mac-gsplat/variants/v5_features/` — Codex extended this.
  v5_features now returns `(features, alpha_mask)` with full gradient
  correctness on means2d/conics/opacities from the alpha stream (verified via
  Tests A–E in their handback). Implemented via the synthetic-channel trick:
  internal (F+1)th channel with `c_i=1.0`, `bg=0.0`, color-grad write
  skipped. Bindings, schema strings, dispatch sigs, Metal kernels (4 of 7
  needed alpha; 2 binning kernels and the no-grad path were sufficient to
  update). This was Codex's work — we did not touch it.

New configs in `src/train_configs/`:

- `local_mac_unconditioned_tokens_features_F32.jsonc` — original F32 config;
  defaults eventually settled on `pre_norm=true`, `kaiming`, `gain=2.0` per
  Codex's gradient probe finding.
- `local_mac_unconditioned_tokens_features_F32_LN_orth_g3.jsonc` — 200-step
  ablation with LN + orthogonal init + gain=3.
- `local_mac_unconditioned_tokens_features_F32_LN_kaiming_g4.jsonc` — 200-step
  ablation with LN + Kaiming init + gain=4 (saturation-safer alternative).
- `local_mac_unconditioned_tokens_fast_400step.jsonc` — F=3 baseline at
  400 steps (apples-to-apples reference for the alpha-aware F=32 run).
- `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc` —
  F=32 with LN+kaiming-g4 + alpha-aware composition at 400 steps. The
  current best F=32 recipe.

## Configs that exist (one-line description each)

- `local_mac_unconditioned_tokens_features_F32.jsonc` — original F32 config
  (default-Kaiming colorize at first; gray collapse → Codex patched to
  LN+Kaiming-g2 mid-session).
- `local_mac_unconditioned_tokens_features_F32_LN_orth_g3.jsonc` — 200-step
  LN+orth-g3 ablation. Highest spatial-std at init in the matrix; lost in
  practice.
- `local_mac_unconditioned_tokens_features_F32_LN_kaiming_g4.jsonc` — 200-step
  LN+Kaiming-g4 ablation. Less init richness, less deep saturation; won.
- `local_mac_unconditioned_tokens_fast_400step.jsonc` — F=3 baseline at
  400 steps (reference for alpha-aware comparison).
- `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc` —
  F=32 with LN+Kaiming-g4 + alpha-aware composition at 400 steps. Winner.

## W&B runs from this session

| run id | config | steps | wall | final loss | notes |
|---|---|---:|---:|---:|---|
| `gwdmm5cc` | F=32 LN+orth-g3 | 200 | 31s | 0.1818 | clustered, lost to kaiming-g4 |
| `2nz89pj3` | F=32 LN+kaiming-g4 | 200 | 76s | 0.1118 | clustered but lower loss; less Logit\|>4\| saturation than orth |
| `azufyx9e` | F=3 400-step baseline | 400 | 76s | 0.0749 | clean apples-to-apples reference |
| `3reqcya9` | F=32 alpha-aware 400 | 400 | 65s | **0.0653** | beats F=3 baseline at same horizon |
| `qstqjup2` | (older) F=3 1000-step | 1000 | 912s | 0.0588 | longer-horizon F=3 reference |

## Key learnings to compress into key_learnings.md candidates

These are the dense bullets a future agent should carry forward. The user
should decide which to actually paste into `agent_notes/key_learnings.md`.

- **MLP cheating via Path B (parasitic gradient sink).** When background
  supervision flows through a learnable layer with a bias before reaching
  splat geometry, the bias absorbs that supervision (sigmoid saturates at
  ~gray, loss minimizes) and the geometric outward-push gradient vanishes.
  The fix is not bigger gradients or different init — it is removing the
  parasitic path. White-background-in-RGB-space matters because RGB white
  (1, 1, 1) is a structurally non-learnable saturated reference: the model
  literally cannot represent it any other way, so supervision in background
  regions is forced to flow through the splat-coverage stream
  (`α · splat_rgb`), which means through `α`, which means through means2d /
  conics / opacities. The moment a learnable layer sits between the loss and
  the alpha-blend, this invariant breaks.
- **LayerNorm + Kaiming does most of the colorize-init work.** Kaiming was
  designed under `Var(input) ≈ 1`; without LN, raw raster output has σ ≈ 0.2
  and Kaiming-init pre-sigmoid logits sit in sigmoid's linear band. LN forces
  σ_in = 1, which is the Kaiming calibration point. Without LN, you need
  gain ≈ 7 and end up in the saturation zone.
- **`Logit|>4|` matters more than `Logit|>2|` for trainability.** ~5% of
  pixels in `|z|>4` is enough to slow training meaningfully because their
  sigmoid gradient is ~0.018. We initially over-indexed on `Logit|>2|` and
  recommended orth+gain=3 over kaiming+gain=4 based on richer init metrics —
  kaiming actually trained better because of less deep saturation. Init
  metrics are necessary but not sufficient for picking a winner.
- **Hidden-layer colorize is a negative trade.** ~6× more multiplies/pixel
  for ~10% NEGATIVE init spatial std (variance compounds through layers).
  Don't add hidden unless training shows linear-only colorize is the
  capacity bottleneck. Browser inference cost is real.
- **Don't trust `py_compile` alone after a tuple-arity change.** Mid-cascade
  states are syntactically valid by construction; smoke-test must run the
  actual call graph (val_log path) before claiming done.
- **The synthetic-channel equivalence trick is the implementation strategy,
  not just a derivation aid.** `accumulated_alpha = 1 - T_final` is
  mathematically identical to a feature channel where every splat has
  `c_i = 1.0` and `bg = 0.0`. The kernel reuses all existing per-channel
  feature backward code with the synthetic channel's color-grad write
  skipped. No new gradient math. This made the v5_features alpha-output
  extension a single-PR plumbing job instead of a derivation pass.

## Things that worked

- Alpha-aware composition restored geometric supervision; F=32 now beats F=3
  baseline at 400 steps (Eval/Loss 0.0653 vs 0.0749).
- The matrix probe (`probe_colorize_matrix.py`) is fast: ~15 s for 3 seeds ×
  21 cells via feature caching. Good ROI for tight init iteration.
- Codex's synthetic-channel implementation in v5_features passed all five
  tests (A-E) on the first try. The math audit upfront paid off.

## Things that didn't work / went wrong

- The init matrix probe identified LN+orth-g3 as the highest-spatial-std
  cell; LN+Kaiming-g4 actually trained better because of less Logit|>4|
  saturation. The probe metrics are necessary but not sufficient.
- A mid-cascade state where `render_full_sequence` had been changed to
  return alpha but `validation_video_payload`'s unpack site hadn't been
  updated: `py_compile` clean, val_log(0) crashed at runtime. Fixed by
  always running the 1-step F=32 smoke after architecture-touching edits.
- `cd` into `third_party/fast-mac-gsplat/variants/v5_features/` for the
  build then forgetting to come back; uv tripped on the variant's bare
  `pyproject.toml` (no `[project]` table). Use the subshell-parens pattern
  (`( cd ... && build )`) so the parent shell stays in dynaworld root.
  Documented in `AGENTS.md` "Build & Run Conventions."
- Init knobs were tuned for ~30 s before the splat-clustering symptom was
  pinned to alpha composition. Time spent on init wasn't wasted — LN+Kaiming
  does help at runtime — but it was a second-order fix being applied to a
  first-order structural failure.

## What's still broken — open issues for next session

- **Alpha mask still shows variance instead of converging to a clean
  foreground/background split.** The model uses white background as a proxy
  for white scene content (sky, clouds, light-grass regions), so foreground
  regions that "look white" get reconstructed by the white background
  composite rather than by splat coverage. This blows holes in the Gaussian
  field where there should be splats, which will hurt depth maps and
  novel-view synthesis. See `TODO/alpha_white_background_cheating.md`
  (parallel doc being written this session).
- **No alpha L1 / total-variation prior in the loss.** Nothing currently
  penalizes alpha for taking arbitrary intermediate values; the model is
  free to drift into the white-background-as-cheap-proxy regime above.
- **Tier 2 numbers for F=32 alpha-aware.** All current F=32 runs are Tier 1
  same-source single-clip overfit. The actual question — does feature
  splatting improve held-out PSNR — is unanswered. The Tier 2a 3-cam
  DeepView holdout split is the cheapest probe.
- **F=3 parity with view conditioning enabled.** The `view_condition`
  knob (`camera_center_ray`, `pixel_ray`) was added for view-dependent
  effects but has not been A/B-tested on either F=3 or F=32 with training
  metrics.

## Pointers

- Codex handoff that produced the rasterizer alpha output:
  `agent_notes/loose_notes/2026-04-29_20-30-00_codex_handoff_v5_features_alpha_output.md`.
  Read it for the math + test specs Codex implemented against.
- Init-matrix probe data and three hypotheses doc:
  `agent_notes/loose_notes/2026-04-29_18-15-00_colorize_gray_collapse_hypotheses.md`.
- Init-handoff doc summarizing knobs and what was learned:
  `agent_notes/loose_notes/2026-04-29_19-30-00_feature_splatting_init_handoff.md`.
- Codex's gradient-probe diagnostic that motivated the Codex config change:
  `agent_notes/loose_notes/2026-04-29_19-29-37_codex_changes_handoff_for_claude.md`.
- Original feature-splatting plan: `agent_notes/loose_notes/2026-04-29_17-30-00_feature_splatting_plan.md`.
- v5_features fork notes: `agent_notes/loose_notes/2026-04-29_17-34-36_v5_features_rasterizer_fork.md`.
- Build & run conventions (don't `cd` into v5/v5_features, don't trust
  `py_compile`, run F=32 smoke after architecture edits): root `AGENTS.md`.
- Baselines table for context on `qstqjup2` and the unconditioned recipe:
  `BASELINES.md`.
