# DGS-LRM comparison — same silhouette, different contract

Session on 2026-04-22. Read the DGS-LRM paper (Lin et al. 2025,
arXiv 2506.09997) after the user initially assumed we had converged on
"basically the same solution." Key finding: same outer silhouette, but
three load-bearing differences in the training contract. The framing is
not redundant with theirs.

## What DGS-LRM is

- **Task:** feed-forward prediction of deformable 3D Gaussians from
  posed monocular video. Real-time (~0.495s on A100 per clip).
- **Architecture:** 24-layer transformer over spatial-temporal video
  tokens (s × s × l cubes, MovieGen-style). Two-layer MLPs project
  tokens to per-pixel Gaussian parameters.
- **Representation:** per-pixel deformable 3D Gaussians. Each pixel
  predicts depth + RGB + rot + scale + opacity + a deformation vector
  per frame. Translation-only deformation.
- **Training data:** 40,000 Kubric scenes with 4 synchronized cameras,
  physics-simulated objects, ground-truth depth, camera, and 3D scene
  flow. Synthetic.
- **Training loss:** L = L_mse + 0.5·L_lpips + 10·L_depth + 10·L_flow.
  Dual-view sampling: input from one camera, supervision on another
  synchronized camera at the same timestamp.
- **Compute:** 64 H100s, 60k iterations across two resolution stages.
- **Inference contract:** posed monocular video → deformable splats.
  Deploy monocular even though training uses multi-view.

## What's the same as our plan

Outer silhouette, nothing more:

- Feed-forward video → splats, no per-scene optimization.
- Transformer over spatiotemporal video tokens.
- Splats as the output representation.
- Real-time inference goal.
- Deformation vectors for temporal dynamics.

This silhouette is the LRM-lineage default (PixelSplat / MVSplat / LRM /
4DGS-LRM / now DGS-LRM). Sharing a silhouette is not sharing a research
program.

## What's fundamentally different

Three axes on which their contract is the opposite of ours:

1. **Training data.** They require multi-view synthetic data with 4
   synchronized cameras. Our framing 3 contract trains on variable
   observation budgets drawn from real single-camera video with
   crop-as-extrinsic synthesizing the pseudo-multi-view signal;
   synthetic multi-view is EH-4 (optional strictly-additive curriculum),
   not the foundation.
2. **3D labels.** They use direct ground-truth depth (`λ_depth = 10`)
   and 3D scene flow (`λ_flow = 10`) — not optional; these terms
   dominate the loss. Our framing explicitly rejects requiring 3D
   labels: photometric on held-out observations is the primary
   supervision; 3D labels are optional aux targets only where they
   happen to exist.
3. **Representation.** Their Gaussians are per-pixel — tied to the
   encoder's input-frame pixel grid. That is by our vocabulary a
   **decoding expansion** representation: W travels with the encoder
   path; at inference they still feed video through the encoder to
   produce splats. Our framing requires a world-token representation
   that is self-sufficient (C1) — `Render(W, c, τ) → image` with
   no per-frame encoder features at deploy.

They get away with per-pixel + synthetic multi-view because the 3D
labels plus dual-view supervision let them skip the generative-
reconstruction contract entirely. Without those labels, per-pixel
collapses to single-view autoencoding.

## What's worth stealing

- **Their Kubric pipeline.** Synthetic multi-view is a free rich-budget
  source for our `𝒟_var`. Framing 3 EH-4. Use it; do not adopt their
  supervision.
- **Spatial-temporal cube tokenization.** s × s × l cubes (l=4) reduce
  token count 4× and made it "trainable at scale." Useful reference
  when we write the encoder architecture.
- **Dual-view sampling strategy as an ablation reference.** Their
  ablation shows that sampling two cameras at the same timestamp
  disambiguates motion from geometry. In our framing that maps to
  `𝒟_var` sampling same-τ different-c pairs into `H`. The empirical
  confirmation is useful even if their mechanism is different.
- **Reference-frame idea.** They pass K temporally distant frames in
  addition to the main input to improve geometry via larger baselines.
  Mechanism: longer-baseline frames serve as richer evidence. Equivalent
  in our framing to sampling long-baseline budgets into `O_r`.
- **Model-shape numbers.** 24 layers, 256×256 at 15/GPU batch, 512×512
  at 8/GPU. Useful for sizing our own experiments.

## What to reject

- **Per-pixel Gaussian representation.** Violates C1 (self-sufficient
  decode) — the splats only make sense relative to the input frame
  grid. We want world tokens.
- **Scene flow loss (λ=10) as primary supervision.** Requires labels
  we do not have at scale. At most, use on rich synthetic budgets
  only, and weight as an optional aux.
- **Ground-truth depth as primary supervision.** Same argument.
- **Translation-only deformation.** Their simplification is fine for
  their synthetic training distribution; real video has rotation and
  non-rigid change we cannot discard.
- **Their multi-view-required training setup.** Ours must train on
  monocular as the baseline; multi-view is an additive budget, not a
  foundation.

## One-line takeaway

> DGS-LRM is a per-pixel-splat LRM that trained on synthetic multi-view
> with 3D labels. We are building a world-token model that trains under
> a variable observation budget on real monocular video with no 3D
> labels. Same output modality, different research program.

## Trip-saver for future sessions

If a future agent reads DGS-LRM and thinks "oh we already converged on
this," go straight to the three differences above (training data, 3D
labels, representation). Those three are load-bearing and they are all
axes where framing 3 takes a different commitment from DGS-LRM. The
overlap is silhouette only.

Related paper to remember: "World-Consistent Video Diffusion with
Explicit 3D Modeling" (WVD) — generates 6D (RGB + XYZ) video. Also
silhouette-similar, also solves a different problem. Same pattern to
watch for.
