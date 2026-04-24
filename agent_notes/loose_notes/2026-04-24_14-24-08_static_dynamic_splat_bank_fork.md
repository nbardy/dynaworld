# Static/dynamic splat-bank fork

User asked to fork the video-token architecture toward a structural
static/dynamic capacity split:

- static splats are time-invariant by construction
- dynamic splats are a smaller residual bank with temporal motion,
  rotation, and opacity coefficients
- the split is not a semantic static/dynamic classifier
- dynamic capacity should be more expensive through rate pressure

Implementation direction:

- Created branch `codex/static-dynamic-splat-bank`.
- Added `StaticDynamicVideoTokenGSImplicitCamera` in
  `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`.
- Reused the shared `VideoEncoder`.
- Added separate learned query banks and cross-attention stacks for:
  camera tokens, static world tokens, and dynamic world tokens.
- Static bank is decoded once per clip and concatenated unchanged for
  every decoded time.
- Dynamic bank is decoded once into canonical Gaussians plus fixed
  temporal-basis coefficients:
  `A_mu`, `A_rot`, and `A_alpha`.
- Query time enters the dynamic bank only through fixed basis evaluation;
  it does not re-cross-attend into observation tokens.
- Kept current renderer RGB contract, so this fork uses persistent RGB
  color rather than full degree-1 SH in the render path.
- Added `GaussianSequence.auxiliary` to carry bank tensors needed for rate
  diagnostics and losses.
- Added rate terms in `train_video_token_implicit_dynamic.py`:
  static/dynamic alpha means and dynamic coefficient L1 penalties.
- Added local config
  `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_static_dynamic_banks.jsonc`
  with `96` static tokens and `32` dynamic tokens at `64` gaussians/token,
  producing `6144` static and `2048` dynamic splats.

Important caveat:

- This is a structural architecture fork on top of the existing
  implicit-camera trainer. It does not yet implement the richer
  diagnostics from the memo, such as static-only renders,
  dynamic-only renders, zero-motion renders, or parallax-bucket evals.

Revision after user pushback:

- Removed the separate `StaticDynamicVideoTokenGSImplicitCamera` class and
  its separate query banks. That was too much like a second baseline.
- Collapsed the split into the existing
  `DynamicVideoTokenGSImplicitCamera` path. The same learned query bank
  and cross-attention decoder now slice world tokens into static and
  dynamic ranges when `static_tokens` / `dynamic_tokens` are configured.
- Deleted the extra static/dynamic config preset. The split is now an
  opt-in mode of the existing video-token config rather than a new
  baseline entry point.

Branch consolidation:

- Fetched remotes and found local branches:
  `main`, `codex/register-taichi-mac-submodule`, and
  `codex/static-dynamic-splat-bank`.
- `main` was behind the codex branch stack by 45 commits.
- Fast-forwarded local `main` to `3c77a99 Add dataset ingest scaffolds`.
- Deleted duplicate local codex branches after confirming they were
  contained in `main`.
- Left `origin/main` and `origin/codex/register-taichi-mac-submodule`
  untouched. Local `main` is ahead of `origin/main` by 45 commits.
