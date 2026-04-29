# Alpha-Mask White-Background Cheating

## Status / Summary

The F=32 feature-splatting trainer was unblocked by switching to alpha-aware
composition (`final_rgb = α · colorize(features) + (1 - α) · white`), which
made splats spread out and beat the F=3 baseline on training loss. The
`Alpha_Mask_Video` panel from run `3reqcya9`, however, shows that the alpha
mask still has meaningful frame-to-frame variance instead of converging to a
clean foreground/background silhouette. The model appears to exploit the
white background composite to "render" white-ish scene content (sky, clouds,
glints, white grass-highlights) by leaving alpha low in those regions and
letting `(1 - α) · white` carry the GT signal. This is a different cheating
mechanism than the MLP-bias cheat the alpha-aware fix originally addressed,
and it puts holes in the underlying 3D representation even though the L1 loss
looks healthy.

## What We Observed

- Reference run: `3reqcya9` (F=32 alpha-aware, 400 steps,
  https://wandb.ai/nbardy/dynaworld/runs/3reqcya9). Final training loss
  `0.0653`.
- Config: `src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc`.
- The `Alpha_Mask_Video` panel does not collapse to a stable foreground
  silhouette. There is significant variance across the frame, and the
  low-alpha regions visually align with bright/white parts of the GT scene
  rather than with regions outside the captured frustum or behind genuinely
  empty space.
- The `Render_Composite_Video` panel renders correctly because the white
  background composite fills in those low-alpha regions with white that
  happens to match the GT.
- Expected behavior: alpha should be ~1 wherever the GT has visible scene
  content (regardless of color) and ~0 only in genuinely empty regions.

## Why This Is Happening (Hypotheses, Ranked)

The composition formula in `recon_backward` inside
`src/train/train_video_token_implicit_dynamic.py` is:

```
final_rgb = alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * 1.0
```

Given a GT pixel that is close to white, the model has two paths to satisfy
the L1 loss:

1. **Honest path**: cover the pixel with splats, set their features so that
   `colorize(features) ≈ white`, and push α high. Learns geometry and
   appearance.
2. **Cheating path**: leave α low at that pixel and let `(1 - α) · 1.0`
   carry the signal. Wins the loss term without any splat present. **No
   geometric supervision flows to the splats from white-ish GT pixels.**

The cheating path is shorter (only α has to drop) than the honest path (α,
xyz, scale, features, and the colorize MLP all have to cooperate), so the
optimizer prefers it whenever the GT is close to white.

- **H1**: The GT contains bright/white content (sky, clouds, white grass
  highlights, glints) that the model reconstructs through `(1 - α) · white`.
  Same family of cheat as the MLP-bias cheat that the alpha-aware fix
  closed, just shifted to a different spot in the loss path.
- **H2**: The L1 loss treats `|0.95 - 1.0|` and `|0.55 - 0.5|` symmetrically,
  so the gradient magnitude on the cheating path equals that on the honest
  path. The optimizer has no reason to prefer the path that builds 3D
  structure.
- **H3**: White is a degenerate fixed point. Once the model starts using
  white background composite as the explanation for a white region, dropping
  α further is locally cost-free, and pushing α back up requires the
  colorize MLP and splat positions to first learn to reproduce that color.
  Local gradient never traverses the saddle.

## Why This Matters

1. **Depth maps will have holes.** Where there are no splats, there is no
   depth. Novel views with even small camera shifts will reveal voids where
   white scene content used to be.
2. **3D structure is missing**, not merely wrong. The Gaussians do not
   represent the white parts of the scene at all. This is worse than
   "wrong-colored splats present" because there is nothing to fix later.
3. **Generalization is broken.** The model exploits the specific quirk that
   the renderer composites against white. Any change to background color,
   lighting, or camera angle exposes the holes.
4. **Loss curves lie.** L1 converges nicely while the underlying geometry
   stays degenerate. We cannot trust loss-only metrics for this regime.

## Possible Fixes (Brainstorm, Not Decisions)

1. **Penalize low alpha in pixels where GT is non-degenerate.** Regularize
   alpha toward 1 wherever GT looks like real scene content (e.g., GT differs
   from the background color by more than some threshold). Risk: weighting is
   delicate; can over-pull alpha to 1 everywhere and re-create the original
   center-clump pathology.
2. **Drop the white background composite entirely.** Composite against a
   learnable per-frame background, or against the first frame's mean color.
   Removes the "free white" the model exploits. Risk: removes the geometric
   pressure that the alpha-aware fix added; might break the splat-spread we
   just won.
3. **Black background instead of white.** Replace
   `(1.0 - alpha_expanded) * 1.0` with `(1.0 - alpha_expanded) * 0.0`. The
   cheating path now produces black, so it only helps where GT is genuinely
   black. Most natural scenes have far less black content than white, so the
   incentive to cheat shrinks. Trade-off: low-alpha regions display
   differently in the composite video.
4. **Anti-cheating auxiliary loss.** Add a small term that compares
   `α · colorize(features)` (no background composite) directly to GT in
   regions where GT has scene content. Forces splats themselves to
   reconstruct GT; background composite only fills true gaps.
5. **Higher `opacity_init`.** Raise from `0.1` to `0.3` or `0.5` so splats
   start more confident. Less room for the optimizer to discover the
   cheating path early. Cheap to test; may need a paired LR adjustment.
6. **Alpha continuity / coverage regularizers.** Total-variation or
   anti-sparsity on alpha to encourage continuous coverage rather than
   patchy holes.
7. **Two-stage training.** First stage with white background to keep the
   spread fix; second stage with black background or anti-cheating loss to
   wash out the inherited cheating bias. Hacky but may work.
8. **GT-alpha supervision.** Where we have synthetic data with real alpha
   channels (Blender Open Movies pipeline), supervise alpha directly. Not
   available for natural video, but useful as a controlled-setting probe.

## Specific Experiments To Try, In Order

1. **Visualize the failure mode more clearly.** Extend the validation video
   panel to a 6-column composite: GT, splat-only `α · colorize(features)`,
   background-only `(1 - α) · white`, alpha mask, full composite,
   per-pixel L1. If H1 is right, the background-only column should carry
   most of the signal in white-ish GT regions.
2. **Black-background ablation** (Fix #3, tests H1). One-line edit in
   `recon_backward` inside
   `src/train/train_video_token_implicit_dynamic.py`: replace
   `(1.0 - alpha_expanded) * 1.0` with `(1.0 - alpha_expanded) * 0.0`. Run a
   variant of the F=32 alpha config for 400 steps, compare alpha mask
   convergence.
3. **Mean-color background**. Composite against the per-clip mean color
   instead of white. Removes the white-as-fixed-point issue without going to
   the opposite extreme. Slightly more involved than Fix #3.
4. **Anti-cheating loss term** (Fix #4). Add a small auxiliary loss that
   compares `α · colorize(features)` to GT, masked to regions where GT
   differs meaningfully from the background. 400-step run.
5. **`opacity_init` sweep**: `0.1 → 0.3 → 0.5`, paired with a small LR
   adjustment if needed. Cheap to sweep; quickest signal on H3.

For each experiment: log a fresh W&B run, save the `Alpha_Mask_Video`, and
visually verify whether the alpha mask becomes a cleaner foreground
silhouette.

## Definition of Done

The alpha mask in `Render_Composite_Video` looks like a clean opaque
silhouette of the GT scene content: high alpha wherever the GT shows scene,
low alpha only in genuinely empty regions (outside the frustum, behind
nothing). Quantitatively, per-pixel alpha should *not* correlate with
"GT pixel is near the background color".

A useful tracking metric:

- `alpha_to_gt_white_correlation` = correlation between `(1 - α)` and
  `GT_pixel ≈ white-ish`. High correlation = cheating. Track in
  `validation_video_payload` across the experiments above.

## Out Of Scope

- Do not bundle this with the broader feature-splatting / V-JEPA
  conditioning work. This is a focused alpha-mask convergence fix.
- Do not rewrite the alpha-aware composition formula until the simpler
  fixes (background color, opacity init, regularization) have been tried.

## Cross-References

- Original alpha-aware composition session note:
  `agent_notes/loose_notes/2026-04-29_22-00-00_feature_splatting_alpha_aware_composition_session.md`.
- F=32 feature-splatting init handoff:
  `agent_notes/loose_notes/2026-04-29_19-30-00_feature_splatting_init_handoff.md`.
- Codex rasterizer extension (v5_features alpha output):
  `agent_notes/loose_notes/2026-04-29_20-30-00_codex_handoff_v5_features_alpha_output.md`.
- Reference config:
  `src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc`.
- Composition site:
  `src/train/train_video_token_implicit_dynamic.py` (search `recon_backward`,
  `Alpha_Mask_Video`, `Render_Composite_Video`).
