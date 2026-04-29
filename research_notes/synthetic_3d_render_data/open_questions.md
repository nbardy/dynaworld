# Open Questions

Things we don't know yet about synthetic 3D render data inside DynaWorld.
List grows over time; resolved questions move to other files with the
answer.

## Contract / philosophy

- **What's the maximum synthetic-to-real ratio that doesn't violate the
  spirit of the contract?** Pretraining at 99% synthetic + 1% real probably
  trains on synthetic. Pretraining at 50/50 with synthetic loss decaying to
  zero by the end is fine. Where exactly is the line?

- **Is BEDLAM "synthetic" for contract purposes?** It's rendered, but it's
  rendered in Unreal with strong cloth/skin sim. Distribution-wise it's
  closer to real than to a Blender Foundation movie. Does it count as
  pretraining-only data, or is it real-enough for finetune?

- **Does DynaWorld's "world tokens decode to splats" view permit using
  synthetic to *evaluate* novel-camera consistency, or only to *probe* it?**
  The distinction: if the probe passes, can we report the result as a
  validation metric, or is it strictly internal CI?

## Practical / engineering

- **What's the right `CameraSpec` schema for synthetic-emitted cameras?**
  AIST has fixed extrinsics (c01–c09) with no calibration available locally.
  DeepView has fisheye + radial distortion. Neural 3D has LLFF-format
  poses_bounds. Synthetic could match any of them, but matching all of them
  doubles the export work. Pick one.

- **How do we make synthetic look enough like real to be useful, but not so
  similar that the model overfits to synthetic artifacts?** Probably:
  - Cycles for source render (richer materials than Eevee).
  - Domain-randomize camera FOV, exposure, rolling-shutter sim, motion
    blur, lens distortion, ISO noise.
  - Skip the "make it match a phone camera exactly" trap — real phone
    footage will dominate finetune.

- **Should synthetic clips include rolling-shutter simulation?** Most
  synthetic data ignores it. Real phone video has it heavily. If the model
  pretrains without it, finetune has to undo a wrong prior.

- **What's the minimum number of synthetic scenes for the camera-leakage
  test to be reliable?** Five Blender Foundation scenes? Ten? A hundred
  BlenderProc-randomized variants? Don't know yet.

## DynaWorld-specific architecture

- **Does the synthetic camera-leakage probe go before or after the
  paired-camera finetune?** Both? If only one: probably *before* (catches
  architectural bugs) and *after* (regression test).

- **Can we use synthetic as a contrastive signal — train the world tokens
  to be invariant to the *distribution* (real vs synthetic) while preserving
  scene content?** Adversarial discriminator on the tokens. Risk: the
  signal gets entangled with content, model learns to forget detail.

- **Does the bullet-time validation track (Nova feature) and the
  novel-view-synthesis validation track (DynaWorld research) want different
  synthetic sources?** Probably yes:
  - Nova bullet-time: skating/sports/action human motion. BEDLAM, Mixamo,
    AIST.
  - DynaWorld novel-view: scene diversity + camera diversity. Blender
    Foundation + BlenderProc.
  Confirm before building one shared pipeline that serves neither well.

## Asset / dataset questions

- **Are there license issues mixing AIST + DeepView + Neural 3D Video + ViVo
  + BEDLAM in the same training run?** Each has different terms. Confirm
  before any training run that emits a model intended for redistribution.

- **Does Adobe's Mixamo license permit using its animations to render
  training data for a commercially-deployed model?** The "free with Adobe
  account" wording is ambiguous about commercial training data
  pipelines. Check before relying on Mixamo as the primary motion source.

- **Are there better Blender-native rich animated datasets we haven't found
  yet?** Blender Foundation movies are ~10 productions. ArtStation /
  Blendswap / BlenderKit have larger catalogs but mixed quality. Worth
  doing a curation pass?

## Pipeline-state questions (re-check before acting)

These should be re-verified by reading the current state, not relying on
this doc:

- Has `data/v2/bullet-time/assets/spring.zip` been extracted yet?
- Has Unity been installed (the `which unity` check)?
- Has anyone wired the v2 Blender pipeline output into DynaWorld's
  `CameraSpec` adapter?
- Has BlenderProc been added to the v2 stack?
- Have any new Blender Foundation movies been downloaded into the empty
  stubs (`agent327/`, `bbb/`, `cosmos/`, `sintel/`, `bbb/`)?

If you're about to act on any of these, run the actual `ls` first.
