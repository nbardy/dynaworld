# Reviewer 02 — Architect's Critique

> Lens: long-term modularity, hidden coupling, scalability, type discipline.
> Migration cost is secondary; what matters is which design absorbs the next
> 12 months of features without becoming a tangle.

## TL;DR

- **Proposal 3 (typed dataflow) is the only one that encodes the project's
  load-bearing invariants in types**. The single shared `ComposeStage`
  consumed by both single-cam and multicam pipelines makes the Apr 29
  bug class (multicam silently bypassing alpha composition) structurally
  unrepresentable. The other two proposals make it a discipline.
- **Proposal 2 (composition-over-inheritance) is the right architectural
  diagnosis but the wrong factoring**. The seam is correct (six
  Protocols), but ownership of "alpha-aware composition + random_bg +
  colorize" inside `RenderHarness` reintroduces the very coupling the
  bug exposed: a `MultiViewRenderHarness` is again free to drift from
  `SingleViewRenderHarness` because the shared `_composite` helper sits
  inside an inheritance-by-convention base. It moves the bug surface
  from method-overrides to harness-implementations.
- **Proposal 1 (functional helpers) is architecturally inert**. It
  trades a few helpers for code-deletion velocity. Every future feature
  (V-JEPA distillation, depth, anti-cheating) lands by editing
  `recon_backward` and the multicam method in lock-step. The
  inheritance tree it preserves is the historical pattern that
  `key_learnings.md:18` warns about. It is a band-aid sized for "ship
  the multicam fix this week," not for 12 months.
- The product roadmap is dominated by **additive cross-cutting concerns**
  (distillation losses, depth output, anti-cheating bg policies, two-stage
  bg curricula, multi-resolution training). These are the cases where
  Proposal 3's "register a new stage / extend a bundle" wins decisively
  and Proposal 1's "edit `recon_backward`" gets exponentially worse.
- **Architect's bet: a hybrid of Proposal 3's bundle vocabulary and
  Proposal 2's strategy registry**. Proposal 3's bundles (especially
  `RenderedBundle`, `SingleCamComposedBundle`, `MultiCamComposedBundle`)
  are the right type spine; Proposal 2's named pipeline registry is the
  right ergonomic surface. Stages are the implementation primitive, but
  do NOT reify them into a generic Stage Protocol with a graph runtime.
  See "The architect's recommendation".

---

## Proposal 1: Functional helpers

### Module cohesion

The proposed `training_common/` package has one module per
responsibility and each helper has a single job:

| Module | Responsibility | One concept? |
|---|---|---|
| `render_bundle.py` | the `RenderedClipBundle` typed value | yes |
| `compose.py` | `compose_rendered_rgb` + bg materialization | yes |
| `colorize_factory.py` | extract colorize construction from `Trainer.__init__` | yes |
| `recon_loss.py` | the chunked-vs-mean normalization wrapper | yes |
| `video_logging.py` | column assembly for the W&B media payload | yes |
| `render_clip.py` | typed wrapper around `render_clip_sequence` | yes |

Each helper is cohesive in isolation. **The problem is the host.** The
trainer classes that call these helpers — `Trainer`,
`KnownCameraTrainer`, `PrecomputedFeatureImplicitTrainer`,
`MulticamPrecomputedFeatureImplicitTrainer` — remain god-objects whose
single responsibility is "everything." Cohesion at the leaf is good;
cohesion at the trunk is unaffected.

The helper package adds 6 small modules that are pure. The 2,072-line
`train_video_token_implicit_dynamic.py` becomes ~1,950 lines instead of
~2,072. Investigator 01's table of 30+ method overrides on the trainer
classes does not collapse.

### Coupling analysis

Proposal 1 is honest about preserving the existing coupling:

- **Trainers still own the optimizer, the colorize MLP, the W&B
  logging cadence, the validation loop, the model lifecycle, and the
  step counter.** All of these mutate from helper call sites:
  `compose_rendered_rgb` reads `self.colorize`; `recon_backward` calls
  `self.optimizer.step`; `validation_video_payload` reads
  `self.gt_video_logged`. The shared mutable state remains the
  trainer instance.
- **`render_clip_with_alpha` does a circular import inside the function
  body** to dodge the load-time cycle with
  `train_video_token_implicit_dynamic.render_clip_sequence`. This is
  not a coupling fix; it is a coupling concealment. A future move of
  `render_clip_sequence` into a different module breaks this helper
  silently because Python does not surface the cycle at import time.
- **`MulticamPrecomputedFeatureImplicitTrainer.step` continues to do its
  own `loss.backward()` while `Trainer.recon_backward` does chunked
  per-chunk `backward(retain_graph=...)` calls.** The proposal explicitly
  punts on unifying these. The classification "single-cam = chunked,
  multicam = single-shot" remains tribal knowledge baked into method
  overrides.

The structural coupling is moderate; the implicit coupling is the same
as today plus the new "circular-import-by-convention" inside helpers.

### Type discipline

Honors the project's typed-camera primitive style for one new value
(`RenderedClipBundle`) plus a small sum (`BackgroundPolicy`). The
bundle's `__post_init__` enforces shape invariants that today are
checked nowhere.

Where it falls short:

- **No type around the `LossOutput`.** Loss assembly stays as scalar
  tensors plumbed through method bodies. The chunked vs single-shot
  backward decision remains tribal.
- **No type around the validation payload.** Each trainer's
  `validation_video_payload` returns a `dict[str, Any]`. The
  proposal's `build_validation_video_payload` helper takes typed
  arguments but its return type is `dict[str, Any]` again.
- **`RenderedClipBundle.cameras` is `tuple[Any, ...]`** — explicitly
  punts on tightening to `tuple[CameraSpec, ...]`.
- **Doesn't fix the `feature_dim` silent-drop bug** in
  `FreeGaussianBankImplicitCamera` (investigator 05) because it
  doesn't reach into model construction.

This is a 1-of-3 bundle adoption. The investment in types stops at
render output.

### Composability — future feature thought experiments

| Future feature | Lands as |
|---|---|
| **Feature distillation (V-JEPA teacher-MSE)** | New helper `compute_distillation_loss(features, teacher_features)` plus an edit to `recon_backward` to call it and add to `regularizer_loss`. **Also requires editing `multicam_recon_loss` separately**. The bundle would need a new field for `teacher_features`, or it ride-alongs through the trainer. |
| **Anti-cheating alpha loss** | New helper `compute_alpha_supervision_loss(alpha, ...)` plus edits to `recon_backward` and `multicam_recon_loss`. The chunked-backward strategy breaks here: alpha supervision per-chunk needs cross-chunk awareness. |
| **Depth output** | Add `depth: Tensor \| None` to `RenderedClipBundle`. Edit `compose_rendered_rgb` to keep depth side-channel. Edit each `validation_video_payload` to log depth panels. Edit each trainer's `render_full_sequence` to plumb depth. **Touches every trainer file.** |
| **Two-stage training (white bg phase 1, black bg phase 2)** | New `BackgroundPolicy` variant. Edit `recon_backward` to read schedule. Edit `multicam_recon_loss` similarly. The proposal's `BackgroundPolicy` sum already accommodates this; the trainer-side wiring duplicates. |
| **Multi-resolution training (256px hot + 64px smoke)** | Already supported via config. No code change. |

The pattern is consistent: **each feature requires editing two trainer
methods in lock-step** (single-cam + multicam) plus the eval path.
That is exactly the failure mode the Apr 29 bug exposed.

### Testability

Helper-level. Six pure-function tests, each catching a specific
falsifiable invariant (alpha-blend math, shape mismatches, frozen
dataclass). Good targeted defenses.

What it cannot test:

- **Trainer-level orchestration.** `recon_backward` is a 28-line
  method with branches (chunk loop, last-chunk regularizer, preview
  capture). It can be smoke-tested but not unit-tested without
  fakes for half the trainer state.
- **Cross-trainer parity.** "single-cam and multicam apply the same
  composition" is checked only by smoke tests on two configs.
- **The bug class itself.** A future "MyNewMulticamTrainer" subclass
  that overrides `multicam_recon_loss` and forgets to call the
  helper would not be caught by any test.

### Hidden coupling

- `compose_rendered_rgb` lives in `training_common.compose` but
  imports `FeatureToColor` lazily via `torch.nn.Module`-typed param to
  avoid a cycle. The forward pass assumes `FeatureToColor.forward`
  takes `view_dirs=...` keyword. **A future colorize-API change
  silently breaks the helper at runtime, not at import.**
- `render_clip_with_alpha` does a function-local import of
  `render_clip_sequence` and `viewport_cameras` from
  `train_video_token_implicit_dynamic`. **Renaming the parent file
  breaks this transparently.**
- The bg-sampling-per-chunk vs per-step semantic shift (Risk 2 in the
  proposal) is acknowledged but unresolved. Whether the helper or the
  caller owns "once per step" is not nailed down.
- **`Trainer.colorize` is read by helpers.** The factory builds it,
  the trainer holds it, multiple helpers call it. Ownership is
  fractional.

### What it gets right architecturally

- **Acknowledges the project's existing typed-primitive grain** with
  `RenderedClipBundle` validating shapes at construction.
- **Names the seam** (compose, render-with-alpha, recon-loss-norm,
  bg-policy, video-payload) precisely. The investigator reports
  identified these; the proposal articulates them as concrete
  signatures.
- **Doesn't try to do the universe.** The scope is the alpha-aware
  multicam fix plus a few cleanups. Honest about its limits.

### What it gets wrong architecturally

- **Leaves the trainer hierarchy in place**, contradicting the
  project's stated learning that "shared payload contracts are
  cleaner than shared trainer inheritance" (`key_learnings.md:18`).
  Helpers are payload contracts; the trainer hierarchy is the
  inheritance the note warns about. The proposal keeps both.
- **Does not encode `is_eval` or `training` mode in types.** The
  asymmetry "training: random bg, eval: white bg" lives at call
  sites; a future eval that wants to A/B random vs white is a
  scattered edit, not a config knob.
- **Backward strategy stays as method dispatch.** `Trainer.recon_backward`
  vs `MulticamPrecomputedFeatureImplicitTrainer.step` remain two
  different ways to call `.backward()`. The proposal is explicit
  that this is out of scope.
- **Does not address `KnownCameraTrainer`'s structural drift.** The
  bug is fixed, but the trainer class lives on with its own `step`,
  `sample_clip`, `initial_step_result`, `render_full_sequence`,
  `run`. Investigator 01 documents this as a 6-method override; the
  proposal patches one of those overrides (`initial_step_result`)
  and leaves the others.

### Survives the next 12 months?

No. The core feature roadmap is additive (distillation, depth,
anti-cheating, two-stage bg) and each one requires editing the two
trainer methods in parallel. By feature 4 or 5, `recon_backward`
becomes a 100-line god-method and `multicam_recon_loss` becomes a
45-line shadow. The helpers stay clean; the trainers do not. The
codebase ages like the current trainer file did between Q4 2025 (clean
enough to clone-and-modify) and Apr 30 2026 (2,072 lines, three latent
bugs, multicam diverged).

This proposal is the right shape for "ship the multicam fix in a
week." It is the wrong shape for "12 months of feature additions."

---

## Proposal 2: Composition over inheritance

### Module cohesion

The six Protocols carve the right joints:

| Module | Responsibility |
|---|---|
| `ClipSampler` | data loading + clip selection + view sampling + cache lifecycle |
| `RenderHarness` | viewport + rasterize + colorize + alpha composition + bg sampling |
| `LossFn` | recon + camera reg + bank-rate + rig + backward strategy decision |
| `Validator` | held-out metric computation |
| `VideoLogger` | W&B media + scalar payload assembly |
| `ModelBuilder` | model variant dispatch |

Five of these are clean single-concept modules. **`RenderHarness` owns
two concepts**: the rasterizer call AND the post-render compose
step. The proposal explicitly defends this — "composition is not
orthogonal to rendering." That defense is partially correct (they
share cameras and view-dirs) but partially mistaken: they have
different lifetimes. Rendering is per-call; composition policy
(random vs fixed bg) varies independently per-train-vs-eval and per-
ablation. Bundling them makes "what bg did this run use?" a
harness-implementation question rather than a pipeline question.

The `LossFn` ownership of `BackwardStrategy` is the second slightly
muddled boundary. Backward is a property of how the optimizer wants
to consume gradients, not of how the loss was assembled. Putting
`backward_strategy` on `LossOutput` ties two concerns that the project
might want to vary independently in 12 months (gradient
accumulation, fp16 loss scaling, multi-optimizer sequencing).

### Coupling analysis

- **`render_harness.bind_colorize(self.colorize)` is acknowledged as a
  circular-dependency hack.** The model-builder builds colorize, the
  trainer injects into the harness. This is the kind of spooky
  late-binding the proposal otherwise avoids; it leaks the
  construction order into the runtime contract.
- **`ClipSampler.extra_param_groups()` is a cross-strategy
  back-channel** so the multicam sampler can register the camera-rig
  param group on the trainer's optimizer. Why does the *sampler*
  know about the optimizer? Because the camera rig is sampler-owned
  data that needs to be optimized. This couples the sampler to
  optimizer construction — a leak.
- **`StrategyTuple` is a Cartesian product the registry has to
  validate**. Six strategies × ~3 implementations each = ~729
  notional combos, ~5–8 valid. The proposal's mitigation is a
  per-tuple `validate()` guard at trainer init. Architecturally:
  this is the registry doing typing the type system can't. A future
  contributor will write a "valid in spirit" combo that fails at
  init with a runtime KeyError.
- **`_RenderHarnessBase._composite` is called by harness implementations
  via `self._composite(...)`** but is "NOT inheritance — just a
  shared file." This is inheritance-by-convention. Three implementations
  read each other's expectations of the `_composite` contract.
  When (not if) someone overrides one and forgets the other, the
  proposal's central claim ("structurally impossible") becomes
  "structurally improbable but achievable."
- **`PipelineContext`-equivalent state is implicit.** The Trainer
  shell holds `model`, `colorize`, `optimizer`, `device`. Each
  strategy reaches into the Trainer for the bits it needs. The
  Trainer is now a thin god-object instead of a thick one — better,
  but still the integration point.

### Type discipline

Significantly better than Proposal 1:

- `ClipBatch`, `RenderedClipBundle`, `LossOutput`, `ValidationPayload`,
  `StepResult` are all typed dataclasses with documented
  invariants.
- `BackwardStrategy = SingleShot | Chunked` is a sum type the trainer
  pattern-matches on.
- `Protocol`s replace base classes; `runtime_checkable=True` gives a
  startup smoke.

Where it falls short:

- **`RenderedClipBundle.rgb` accepts a `[T, 3, H, W]` OR a
  `[V, T, 3, H, W]` tensor.** The proposal mentions this in passing
  but doesn't split the type. A `LossFn` that consumes the bundle
  has to branch on shape.
- **`RenderedClipBundle.cameras` is
  `tuple[CameraSpec, ...] | tuple[tuple[CameraSpec, ...], ...]`** —
  same union, same branching cost.
- **`LossOutput.breakdown` is `Mapping[str, Tensor]`** — string keys,
  no enforcement that "recon_loss" is present, no schema.

The bundles do half the type-safety work. The other half is dispatch
on bundle shape, which the proposal hand-waves.

### Composability — future feature thought experiments

| Future feature | Lands as |
|---|---|
| **Feature distillation** | New `LossFn` strategy `DistillationLoss` that wraps `StandardLoss` and adds the distillation term. Register in `LOSS_FN_REGISTRY`. Pipeline override `loss_fn: distillation_loss`. **Clean.** |
| **Anti-cheating alpha loss** | New `LossFn` variant. Same shape as above. **Clean.** |
| **Depth output** | Extend `RenderHarness.render` to return `RenderedClipBundle` with optional `depth` field. Every harness implementation must add depth plumbing. The shared `_composite` helper needs a depth pass-through. The `Validator` needs a depth panel. The `VideoLogger` needs a depth panel. **Touches three strategies.** |
| **Two-stage bg curriculum** | The `RenderHarness` owns bg sampling. Adding a phase-aware bg policy means editing the harness implementation OR introducing a new `BackgroundPolicy` strategy that the harness consumes — but bg policy is not currently a strategy. **Either edit harness or introduce a 7th strategy.** |
| **Multi-resolution training** | Already supported. |

Feature distillation and anti-cheating loss compose cleanly through
`LossFn`. Depth requires changes across `RenderHarness`, `Validator`,
`VideoLogger`. **Bg policy as a phase-dependent thing requires either
swapping harness mid-training or introducing a new strategy.**

### Testability

Per-strategy unit tests are unlocked. Each strategy is testable
against a fake trainer-state. The proposal's test plan
(`test_render_harness_composition.py`,
`test_loss_fn_chunked_vs_singleshot.py`) is genuinely good.

`test_multicam_loss_no_silent_skip.py` is the most interesting test:
mock-patching `_composite` with `side_effect=AssertionError` to
verify the multicam path goes through the shared kernel. This is a
falsifiable architectural-invariant test, exactly the shape the
project's testing guide endorses.

### Hidden coupling

- **`_RenderHarnessBase._composite` is shared by import not
  inheritance.** Three harnesses import the same module and call
  `self._composite(...)` — but `self` is each harness, and the method
  resolution is by Python's MRO. If `MultiViewRenderHarness`
  overrides `_composite` (legal), the contract breaks silently.
- **`bind_colorize` must be called exactly once before any
  `render()`.** Proposal acknowledges this; doesn't enforce it.
  The trainer guarantees the order, but a future test that
  constructs a harness directly will hit a `None` colorize and
  produce confusing failures.
- **Strategy combos are Cartesian product validated by registry.**
  An ablation engineer who writes
  `strategies: {render_harness: multi_view_alpha_aware,
  loss_fn: l1_dssim_singlecam}` produces a runtime error at step 1,
  not at config parse. Mitigation is the registry guard, but the
  guard is a string-based table.
- **Sampler exposes `extra_param_groups()` so the trainer can plumb
  rig parameters into the optimizer.** This is the optimizer
  reaching into the sampler's data. Architecturally, the rig
  should belong to the model (it's a parameter), not the sampler
  (which is data).

### What it gets right architecturally

- **Names the joints correctly.** Six strategies cover the trainer's
  actual concerns. A new contributor reading the strategy registry
  knows where every concept lives.
- **Honors `key_learnings.md:18`.** The `Protocol` discipline is
  structural, not nominal. There is no shared `BaseTrainer`. Each
  strategy is an independent unit.
- **Backward strategy as data.** `LossOutput.backward_strategy =
  SingleShot | Chunked` makes the chunked-vs-single-shot decision
  visible in the type system. A multicam config that wants chunked
  backward gets it by returning `Chunked(...)` from its `LossFn`.
- **The named pipeline registry** is the right ergonomic surface.
  Configs say `pipeline: multicam_vjepa_alpha`, not "which `.sh`."

### What it gets wrong architecturally

- **`RenderHarness` owns too much.** Render + colorize + compose +
  bg-sample is four concerns. The Apr 29 bug was about
  multicam-specific compose drifting from single-cam compose. The
  proposal solves "drift between harnesses" by demanding all
  harnesses inherit-by-convention from a shared `_composite` helper
  — i.e. it is back to depending on discipline.
- **Strategy combos are unconstrained at the type level.** The
  registry validates combos at runtime. The type system thinks any
  six-tuple is fine.
- **The bundle types don't split single-cam vs multicam.** Same
  union problem as Proposal 1's `RenderedClipBundle.cameras`. Loss
  and validator have to branch on shape.
- **Bg policy is harness-internal**. Future "phase 1 white bg, phase
  2 black bg" curriculum is awkward.
- **`KnownCameraTrainer`'s special needs (zero-weight camera reg)
  collapse into "loss config option"** — at the cost of
  `LossConfig.with_camera_weights_zero()` being a method on the
  config, an idiom the codebase doesn't currently have.

### Survives the next 12 months?

Mostly yes, with caveats. Distillation and anti-cheating losses land
cleanly as new `LossFn`s. Depth is a 3-strategy edit (harness +
validator + logger). Bg curriculum is awkward. The strategy registry
absorbs new pipelines without modification.

The structural risk: when a new feature does NOT decompose along the
six existing strategy lines, the proposal has to choose between
adding a 7th strategy (and re-validating all combos) or stuffing the
new concern into the strategy that's "closest" — which is exactly
how `RenderHarness` ended up owning four concerns in this proposal.

The proposal is at the right grain for the next 6–12 months. It will
need a careful follow-up if/when the rasterizer changes
fundamentally (e.g. the v5/v5_features merge, a depth-aware variant,
or a depth-of-field branch).

---

## Proposal 3: Typed dataflow pipeline

### Module cohesion

Each stage has one concept:

| Stage | Concept | One? |
|---|---|---|
| `SampleStage` | pick clip frames + cameras + GT | yes |
| `MulticamSampleStage` | pick views + per-view clips | yes |
| `LiveEncoderModelInputStage` / `PrecomputedFeatureModelInputStage` | what the model sees | yes |
| `ForwardStage` / `KnownCameraForwardStage` | model.forward call | yes |
| `RenderStage` / `MultiViewRenderStage` | rasterize | yes |
| `ComposeStage` | colorize + alpha composite + bg sample | yes (3 sub-tasks but one purpose: produce final RGB) |
| `LossStage` | scalar loss + breakdown | yes |
| `BackwardStage` | call .backward | yes |
| `OptimizeStage` | call optimizer.step | yes |
| `MetricStage` | compute eval metrics | yes |
| `MediaPayloadStage` | build W&B panels | yes |

`ComposeStage` is the most-loaded but its three sub-tasks
(colorize MLP forward, alpha blend, bg sample) form one semantic
unit: "take the rasterizer output and produce a comparable RGB."
Splitting them would make compose look exactly like Proposal 1's
helper proliferation.

`PipelineContext` carries ~10 fields — the proposal acknowledges this
as a god-object risk (Risk 6). It is the right answer for "what
exceeds one step's lifetime" but it is the design's softest spot.

### Coupling analysis

- **Stages thread bundles through `parent` pointers.** `LossStage`
  reaches `bundle_in.parent.parent.parent` for `ClipBundle`. This
  is the proposal's biggest architectural smell. The chain is type-
  checkable but cognitively heavy. A future stage insertion shifts
  the chain — and every consumer that reaches up through `parent`
  breaks. **The fix is to flatten: instead of nested parents,
  bundles should carry the upstream payload directly.** The proposal
  half-acknowledges this when it offers
  `SingleCamComposedBundle` / `MultiCamComposedBundle` as a split
  rather than a union — but the parent chain is preserved.
- **`PipelineContext` is mutable and shared.** Stages access
  `ctx.optimizer`, `ctx.model`, `ctx.rng`, `ctx.cfg`,
  `ctx.feature_cache`. The proposal forbids stages from mutating
  context fields they don't own (only `optimizer.step()` and `rng`
  advance). This is a discipline contract; the type system doesn't
  enforce it.
- **The chunked-backward path is awkward.** Proposal explicitly
  flags Risk 4: "stages produce a single bundle out, not N." The
  workaround is a nested `ChunkedRecoBackwardPipeline`. This is the
  one concrete place where the design fights the grain.

The structural coupling is much lower than Proposal 1's or 2's
because every stage's input and output are typed bundles. The
implicit coupling is the parent-pointer chain plus
`PipelineContext`.

### Type discipline

By far the strongest of the three:

- **Every stage input and output is a typed dataclass with
  invariants** validated at construction.
- **`SingleCamComposedBundle` vs `MultiCamComposedBundle`** is the
  type-level discriminator that Proposals 1 and 2 lack. The
  Proposal 3 author names this explicitly: "I propose (b): explicit
  types, mechanical dispatch."
- **`alpha is None iff renderer cannot expose alpha`** is documented
  as an invariant on `RenderedBundle`, validated in `__post_init__`.
- **`final_rgb in [0, 1] when colorize activation is sigmoid`** is
  an invariant on `SingleCamComposedBundle`. (Validatable, though
  the proposal doesn't show the check.)
- **`Literal["fast_mac", "dense", "tiled", "taichi"]`** for renderer
  mode — type-narrowed dispatch.
- **`AuxiliaryDecode`** wraps the dynamic-split bookkeeping in a
  typed dataclass, replacing the `auxiliary: Mapping[str, Any]`
  in `GaussianSequence`.

Over-engineering risk:

- **10 bundle types** is at the boundary of "central artifact" vs
  "ceremony." Several bundles (`OptimizerBundle`, `BackwardBundle`)
  are mostly forwarding. The proposal's defense — "each bundle has
  a real consumer with a real consumed field" — is true today but
  strict. If `OptimizeStage` were merged into `BackwardStage`, the
  `OptimizerBundle` would vanish without loss.
- **`PipelineContext.dtype`** carries the model dtype. Why not on
  the bundles? Because every bundle would inherit it. But that
  invariant should be enforced at bundle construction in tests.

### Composability — future feature thought experiments

| Future feature | Lands as |
|---|---|
| **Feature distillation** | New `DistillationLossStage` slotted between `ComposeStage` and `LossStage`. Reads `bundle.parent.rendered_features` (pre-colorize features), compares to teacher targets in ctx. Or: extend `LossStage` to consume a teacher-target bundle field. **Clean: register a new stage in the pipeline registry.** |
| **Anti-cheating alpha loss** | New stage `AlphaSupervisionLossStage` consuming `SingleCamComposedBundle.alpha_used`. Slot before `LossStage` to add an alpha penalty into the LossBundle's `terms`. **Clean.** |
| **Depth output** | Extend `RenderedBundle` with `depth: Tensor \| None`. Extend `ComposedBundle` to pass through. Add a `DepthLossStage` if depth supervision is wanted. The MediaPayloadStage adds a depth panel. **Clean: extend bundle types, register stages.** |
| **Two-stage bg curriculum** | Edit `ComposeStage` to read `ctx.cfg.compose.bg_schedule` and select the bg policy by step. **Clean: one stage edit, no other change.** |
| **Multi-resolution training** | Already supported. |

This is where Proposal 3 outscales the others. **Every cross-cutting
concern lands as a bundle field extension plus a stage insertion.**
The bundle types absorb the new data; the stage list absorbs the new
behavior. Existing stages don't change.

### Testability

Each stage testable on synthetic bundles. Each bundle has invariant
tests. The pipeline-level integration test exercises a real fixture.

This is the highest-granularity test surface of the three:

- **Bundle invariant tests**: shape, alpha-vs-feature_dim
  consistency, parent-pointer integrity. Each is a falsifiable math
  test.
- **Per-stage unit tests**: feed synthetic input bundle, assert
  output bundle invariants. Possible because stages have no
  hidden state.
- **Pipeline smoke**: 5-step run on a small fixture, asserts loss
  decreases.
- **Architectural invariant**: `mock.patch` to verify the legacy
  alpha-stripping render entry is never called on the F=32 path.

### Hidden coupling

- **Parent pointer chain.** `LossStage._single_cam_recon` walks
  `bundle_in.parent.parent.parent`. Insert a stage and the index
  shifts. **This is the proposal's most fragile piece**.
- **`PipelineContext` is mutable.** Stages call `ctx.optimizer.step()`
  and `ctx.rng.manual_seed()`. The "stages only mutate optimizer
  and rng" rule is a comment, not a type.
- **`AuxiliaryDecode` requires `decoded.auxiliary.get("static_opacities")`
  etc.** This means the model's forward output schema is now coupled
  to `AuxiliaryDecode`. A new model variant that emits a different
  auxiliary key has to either (a) match the existing schema or
  (b) trigger an `AuxiliaryDecode` extension.
- **Chunked backward as a nested pipeline** is acknowledged as awkward.
  The choice between "every bundle is one-step" and "chunked-backward
  produces N bundles" is unresolved in the proposal.
- **Eval pipeline shares stages with train pipeline by class
  identity, not by composition.** The same `ComposeStage` instance
  is reused with `background_policy="white"` for eval vs
  `"random_per_step"` for train. This is correct, but means the
  stage's constructor knob is the only knob; per-step behavior
  variance has to be a stage-level config, not bundle field.

### What it gets right architecturally

- **Bundles are the central artifact.** Once they compile, the
  pipeline is mostly mechanical. Investigator 02's "what is the
  seam?" question gets a real answer: the seam is each bundle's
  schema.
- **The bug class becomes structurally impossible.** A multicam
  pipeline goes through the same `ComposeStage` as single-cam.
  There is no multicam-specific compose path to forget to update.
  This is the deepest claim of the three proposals; the only one
  that converts the bug class from "discipline" to "type
  unrepresentable."
- **Pipeline = data, not code.** New trainers are registry entries.
  The 12-month roadmap is "register more pipelines and stages,"
  not "edit `Trainer`."
- **Train and eval share the bundle vocabulary.** `MetricStage`
  consumes the same `SingleCamComposedBundle` that `LossStage`
  consumes; the difference is "compute metrics" vs "compute
  loss," not "different render path."

### What it gets wrong architecturally

- **Parent-pointer chain.** Should be a flat dependency graph of
  bundles (e.g. each bundle carries the upstream `ClipBundle` by
  reference at top level, not via `parent.parent.parent`). The
  proposal has this issue and acknowledges it indirectly through
  Risk 7 ("over-engineering").
- **10 bundles is at the upper limit of comprehensibility**. Two of
  them (`BackwardBundle`, `OptimizerBundle`) could collapse into one
  `PostUpdateBundle` without losing invariant coverage. The author
  flags this risk.
- **`PipelineContext` IS a god-object** despite the proposal's
  defense. It carries model + optimizer + rng + config +
  feature_cache + multicam_bundle + camera_rig + sequences. The
  defense — "these have lifetimes longer than one step" — is
  correct but not exhaustive. Some of these (camera_rig, multicam_bundle)
  are arguably sampler-owned data; some (feature_cache) are
  build-once-then-readonly.
- **Chunked backward path is unresolved.** The proposal is honest
  about this; the architect notes it as a concrete cost of the
  design.

### Survives the next 12 months?

Yes, with the strongest claim of any proposal. **Each future feature
the user named lands as a stage insertion or a bundle field
extension; existing stages are not modified.** The bundle types
absorb the data evolution; the pipeline registry absorbs the
behavior evolution.

The two architectural risks the design carries forward:
1. The parent-pointer chain is fragile; a future refactor should
   flatten it.
2. `PipelineContext` will accumulate fields. A few will be moved
   into bundles (e.g. `feature_cache` could become a one-shot
   ingestion stage that produces a typed `FeatureCacheBundle`).

Both of these are tractable mid-flight refactors. Neither is
existential.

---

## Comparison table

| Dimension | Proposal 1 | Proposal 2 | Proposal 3 |
|---|---|---|---|
| **Cohesion (1–5)** | 4 — leaf helpers cohesive; trunk trainers untouched. | 3.5 — five strategies clean, RenderHarness owns four concerns. | 4.5 — each stage is one concept; ComposeStage stretches but stays unified. |
| **Coupling discipline (1–5)** | 2 — circular imports, shared trainer mutable state, lock-step edits. | 3 — `bind_colorize` hack, sampler-knows-optimizer leak, MRO-by-convention `_composite`. | 4 — bundles flow through types; parent-pointer chain is the only smell. |
| **Type-driven invariants (1–5)** | 2.5 — one bundle, one sum type, dict[str, Any] elsewhere. | 3.5 — five bundles, BackwardStrategy sum, but unions for single/multi. | 4.5 — 10 bundles, full split for single/multicam, Literal-typed dispatch. |
| **Composability for new losses (1–5)** | 2 — every loss = edit two trainer methods. | 4 — new `LossFn` strategy. | 4.5 — new `LossStage` plugin. |
| **Composability for new render outputs (depth, etc.) (1–5)** | 1.5 — touches every trainer. | 3 — touches 3 strategies. | 4 — extend `RenderedBundle`, register new stage. |
| **Composability for cross-cutting policy (bg curriculum) (1–5)** | 1.5 — scattered edits. | 2.5 — harness-internal or new strategy. | 4.5 — one stage's constructor knob. |
| **Test surface granularity** | helpers (6 unit tests). | strategies (~7 unit + integration). | stages + bundles + pipeline (~15+ tests across 3 levels). |
| **New-feature absorption** | medium → degrades fast. | high. | highest. |
| **Onboarding cost for a new contributor** | low (helpers + existing trainer). | medium (six Protocols, named pipelines). | medium-high (10 bundle types, parent chain, PipelineContext). |
| **Future-proofing for V-JEPA distillation** | medium — edit `recon_backward`. | high — new LossFn. | high — new stage. |
| **Future-proofing for depth output** | low — touches every trainer. | medium — RenderHarness + Validator + VideoLogger. | high — extend bundle, register stage. |
| **Future-proofing for anti-cheating alpha** | low — edit two methods. | medium — new LossFn variant. | high — new stage. |
| **Future-proofing for multicam-at-256px-1000-step** | medium — multicam still its own trainer. | high — multicam = strategy combo. | high — multicam = pipeline registry entry. |
| **Risk of accumulating special cases** | high — trainer hierarchy preserved. | medium — strategy combos can sprawl. | low — bundle vocabulary forces uniform shape. |
| **Risk of over-engineering** | very low. | medium — six strategies, registry validation. | medium-high — 10 bundles, parent chain, context god-object. |
| **Honors `key_learnings.md:18` (no shared BaseTrainer)** | partial — no new base, but trainer hierarchy stays. | yes — Protocol-based, structural. | yes — no trainer class hierarchy at all. |
| **Aligns with project's typed-camera primitive grain** | partial — one bundle. | yes — five bundles. | yes — full bundle vocabulary. |
| **Migration cost (architect's secondary concern)** | smallest. | medium. | largest. |

---

## The architect's recommendation

**Proposal 3's bundle vocabulary plus Proposal 2's named pipeline
registry, with Proposal 1's six-helpers as the migration on-ramp.**

In detail:

### Take from Proposal 3

- **The bundle types**: `RenderedBundle`, `ComposedBundle` (split as
  `SingleCamComposedBundle` / `MultiCamComposedBundle`), `LossBundle`,
  `AuxiliaryDecode`, `ValidationBundle`. These are the type spine.
- **The single shared `ComposeStage`** consumed by single-cam and
  multicam pipelines. This is the structural fix for the Apr 29
  bug class.
- **The `is_eval` flag and `Literal[...]` typed enums** for renderer
  mode and bg policy. These narrow dispatch in the type system.
- **The bundle `__post_init__` invariants**: alpha shape consistency,
  feature_dim consistency, parent-pointer integrity.

### Take from Proposal 2

- **Named pipeline registry** as the ergonomic surface. Configs say
  `pipeline: multicam_vjepa_alpha`; ablations override individual
  stages.
- **The 6-stage decomposition** as the conceptual carving (sample,
  forward, render, compose, loss, optimize). Use it as the standard
  shape for all pipelines.
- **`Protocol` for stages** instead of a generic `Stage[BundleIn,
  BundleOut]` graph runtime. Stages are concrete callables; the
  pipeline is a list of typed callables that compose by Python's
  type system, not a DAG executor.

### Take from Proposal 1

- **The migration on-ramp**: extract `RenderedClipBundle` and
  `compose_rendered_rgb` first as helpers, before the full bundle
  vocabulary lands. This gives the team a path that ships the
  Apr 29 multicam fix in week 1, without committing to the full
  pipeline rewrite.

### Reject from Proposal 3

- **The deep parent-pointer chain.** Bundles should carry upstream
  data by name reference (e.g. `LossBundle.clip` directly), not via
  `bundle.parent.parent.parent`. Flat is better than nested for
  type-checking and refactoring.
- **The `PipelineContext` god-object**. Split it: model + optimizer
  in one (`ModelHandle`), feature_cache + sequences in another
  (`DataHandle`), config in a third (`ResolvedConfig`). Each stage
  receives only the handles it needs.
- **Generic `Stage[BundleIn, BundleOut]` Protocol**. Don't reify
  stages into a graph runtime. A pipeline is a list of typed
  callables; runtime composition is just function calls in
  sequence. The bundle types do the type-checking.

### Reject from Proposal 2

- **`RenderHarness` ownership of compose + bg sampling**. Compose
  is its own stage; bg sampling is a stage-config knob. Splitting
  them makes "what bg curriculum is this run using?" a pipeline
  question, not a harness implementation question.
- **`bind_colorize` late-binding hack.** The colorize MLP belongs
  in the `ComposeStage` constructor, not bound after the fact.
- **`extra_param_groups()` on the sampler.** The camera rig is a
  parameter; it belongs with the model, not the sampler.

### Reject from Proposal 1

- **Preserving the trainer hierarchy.** `Trainer`,
  `KnownCameraTrainer`, `PrecomputedFeatureImplicitTrainer`,
  `MulticamPrecomputedFeatureImplicitTrainer` should collapse into
  one trainer + a strategy/pipeline picker.
- **Circular import inside helpers.** The dependency on
  `train_video_token_implicit_dynamic.render_clip_sequence` should
  be resolved by moving `render_clip_sequence` to a neutral module
  in week 1.

### Why this hybrid

1. **The roadmap features the user named are all additive:**
   distillation, depth, anti-cheating, two-stage bg, V-JEPA scaling,
   live web viewer integration. Each is a new stage or a bundle
   field extension. Proposal 3's bundle vocabulary is the only one
   that absorbs them without modifying central code.
2. **The Apr 29 bug class is type-fixable**, and only Proposal 3
   makes it type-unrepresentable. Proposals 1 and 2 fix the
   instance; Proposal 3 fixes the class.
3. **The project's existing typed-camera primitive grain
   (`C2W`, `Intrinsics3x3`, `QuatWXYZ`) extends naturally** to
   `RenderedBundle`, `ComposedBundle`, `LossBundle`. Proposal 3 is
   the explicit continuation of that pattern.
4. **`key_learnings.md:18` is honored** by all three; only Proposal
   3 honors it most fully (no trainer hierarchy at all).
5. **The named pipeline registry from Proposal 2 is the right
   ergonomic surface** for an architecture-driven config style.
   Configs name a pipeline; the registry resolves to a stage list.
6. **Proposal 1's helpers are the right week-1 migration** because
   they ship the bug fix without committing to the full rewrite. PR
   1 of the migration plan is "extract `RenderedBundle` and
   `compose_rendered_rgb` as Proposal 1 describes." PRs 2–N
   incrementally reify the stage decomposition.

---

## How each proposal extends to the future roadmap

| Feature | Proposal 1 | Proposal 2 | Proposal 3 |
|---|---|---|---|
| **Feature distillation (V-JEPA teacher MSE)** | New helper `compute_distillation_loss(features, teacher)` + edits to `recon_backward` AND `multicam_recon_loss`. | New `DistillationLoss` strategy implementing `LossFn`; pipeline override. | New `DistillationLossStage` slotted before `LossStage` or merged into `LossStage`'s breakdown; reads pre-colorize features from `RenderedBundle`. |
| **Anti-cheating alpha loss** | New helper `compute_alpha_supervision_loss` + same two edits. Chunked-backward complicates. | New `AlphaSupervisionLoss` `LossFn` variant; or `LossFn` decorator wrapping `StandardLoss`. | New `AlphaLossStage`; reads `SingleCamComposedBundle.alpha_used`; emits a term in `LossBundle.terms`. |
| **Depth output** | Extend `RenderedClipBundle` (`depth: Tensor \| None`); helper `compose_with_depth`; edit each trainer's `render_full_sequence` AND `validation_video_payload` AND `recon_backward`. | Extend `RenderHarness` Protocol return; every harness implementation adds depth plumbing; `Validator` + `VideoLogger` extended. | Extend `RenderedBundle.depth: Tensor \| None`; new optional `DepthLossStage`; `MediaPayloadStage` learns depth panel. |
| **Two-stage bg curriculum (white phase 1 / black phase 2)** | New `BackgroundPolicy` variant; edit `recon_backward` + `multicam_recon_loss` to read step. | Either edit `RenderHarness` impls or introduce 7th `BgPolicyStrategy`. | `ComposeStage(background_policy="schedule")` constructor knob + reads `ctx.step` from PipelineContext. |
| **Multi-resolution training (256px hot + 64px smoke)** | Already config-driven. | Already config-driven. | Already config-driven. |
| **Held-out novel-view PSNR as primary benchmark** | Already on multicam trainer; extend `validation_video_payload`. | `HeldoutCameraValidator` strategy. | Already in `MULTICAM_VJEPA_ALPHA_EVAL` pipeline registry entry. |
| **Live web viewer integration (browser export)** | Already there as `export_browser_bundle` method on Trainer. | Trainer's `_maybe_export_browser_bundle()` reads `ClipSampler.eval_sequences()`. | New `ExportStage` consuming the final `OptimizerBundle`; or end-of-run hook outside the pipeline. |
| **Anti-cheating: random rotation aug, random crop aug** | New augmentation helper called from `recon_backward`; edit multicam path. | New `AugmentingClipSampler` wrapping the base sampler. | New `AugmentationStage` between `SampleStage` and `ForwardStage`. |
| **Multi-stage training (e.g. freeze backbone in phase 2)** | Mid-run state-mutation in trainer's `step` method. | Mid-run strategy swap (questionable; `Trainer` constructs strategies once). | Mid-run pipeline swap (registry resolves to a different stage list); or a `phase: Literal[1, 2]` flag in PipelineContext that stages read. |

The composability story for Proposal 1 is "edit two methods per
feature." For Proposal 2 it is "implement one strategy per feature,
register it." For Proposal 3 it is "register one stage per feature,
extend a bundle if data flow changes."

---

## What I'd push back on

### To Proposal 1

- **"The trainer hierarchy stays."** Future
  `MulticamPrecomputedFeatureImplicitTrainer.step` will continue to
  override the parent and silently bypass new helpers, exactly as it
  did with alpha-aware composition. What in your design prevents
  that recurrence? Smoke tests catch the instance, not the class.
- **`render_clip_with_alpha`'s function-local circular import.** This
  is a coupling concealment. Why not move `render_clip_sequence` and
  `viewport_cameras` into a neutral module in step 1? The proposal
  punts: "deferred — it's a chunk of work and not load-bearing for
  the alpha-aware fix." But it IS load-bearing for not breaking the
  helper in 6 months.
- **No backward strategy unification.** Multicam still does a single
  `loss.backward()`; single-cam does chunked. When V-JEPA features
  go to F=64 and memory pressure forces multicam to chunk, where
  does the chunking logic live? Currently nowhere; this proposal
  defers.
- **`KnownCameraTrainer` stays.** Investigator 01 documents 6
  method overrides. You fix one (`initial_step_result`) via the
  helper. What about the other 5?

### To Proposal 2

- **`RenderHarness` owns four concerns.** Render + viewport +
  colorize + alpha-composite + bg-sample. Why is this not
  `RenderStage` + `ComposeStage`? The only justification given is
  "they share cameras" — that is a data-flow argument, not an
  ownership argument.
- **`PipelineContext`-equivalent state lives on the Trainer
  instance**, accessed by every strategy. What is the type contract
  for "what fields can a strategy read from Trainer?" Today it's
  any public attribute. Tomorrow a future strategy implementation
  reads `self.trainer.step_idx` and the implicit dependency graph
  grows.
- **`_RenderHarnessBase._composite` is inheritance-by-import.** Three
  harnesses share the `_composite` method via a "shared file." The
  type system doesn't enforce that a new `MyHarness` subclasses or
  imports the helper. Show me the runtime check that catches a
  harness which forgets to call `_composite`.
- **Strategy combos are Cartesian.** 729 nominal combos, ~6 valid.
  How does the registry guard scale when distillation, depth, and
  anti-cheating each add a new strategy implementation? At ~10
  implementations × 6 strategies = 1M combos.
- **Sampler exposes `extra_param_groups()`.** Why does the data
  loader know about the optimizer? The camera rig is a model
  parameter living in sampler scope by accident.

### To Proposal 3

- **The parent-pointer chain.** `LossStage._single_cam_recon` walks
  `bundle_in.parent.parent.parent`. Insert one stage and every
  consumer of `parent` shifts. Why not flat data — `LossBundle`
  carries `clip: ClipBundle` directly, not via `parent.parent.parent`?
- **10 bundle types is at the limit.** `OptimizerBundle` and
  `BackwardBundle` look like ceremony. Show me the test that breaks
  if they collapse into one `PostUpdateBundle`.
- **`PipelineContext` is a god-object.** It carries 10 fields with
  different lifetimes (model: per-run; rng: per-step; cfg: per-run;
  feature_cache: build-once-then-readonly; multicam_bundle: per-run
  but only relevant to multicam). Why not split into
  `ModelHandle`, `DataHandle`, `Config`, with each stage receiving
  only the handle it needs?
- **Chunked backward is unresolved.** The proposal mentions a
  `ChunkedRecoBackwardPipeline` as a sub-pipeline workaround. What
  does that look like concretely? Does it bring back the
  per-chunk-backward-with-retain_graph pattern or replace it?
- **The `Mapping[str, Tensor]` `terms` dict on `LossBundle`.** No
  type-level requirement that "recon" is present. Why not a typed
  `LossTerms` dataclass with required `recon` and optional named
  fields?

---

## How the proposals interact with the project's stated conventions

### `key_learnings.md:18` — "A single shared `BaseTrainer` would hide real differences"

| Proposal | Honors? |
|---|---|
| 1 | Partial. No new shared BaseTrainer is added. The existing trainer hierarchy is preserved, which is the historical pattern that the note's lesson came from. |
| 2 | Yes. `Protocol` is structural, not nominal. There is no shared BaseTrainer; there is one concrete `Trainer` shell that composes strategies. The differences live in named strategies. |
| 3 | Yes, most fully. There is no trainer class hierarchy at all — pipelines are stage lists. Differences live in the registry. |

The note specifically warned against shared trainer inheritance. All
three proposals avoid adding one, but Proposal 1 keeps the existing
one intact. Proposals 2 and 3 dismantle it.

### `AGENTS.md` — types as control flow

The project's CLAUDE.md prescribes "canonical types + type-directed
dispatch + one clean handler per type" with "no structural branching
inside handlers."

| Proposal | Aligns? |
|---|---|
| 1 | Partial. `BackgroundPolicy` is a clean sum type with type-dispatched handlers in `_materialize_background`. But `compose_rendered_rgb` has structural branching: `if colorize is None`, `if alpha is None`. These are encoded as conditions on optionality, not as a sum type. |
| 2 | Mostly. Strategies are type-directed dispatch (the registry picks the strategy by name → type). `BackwardStrategy = SingleShot \| Chunked` is a clean sum. But `RenderedClipBundle.rgb: Tensor \| Tensor[V,...]` is a union without explicit dispatch — consumers branch on shape. |
| 3 | Best alignment. `SingleCamComposedBundle` vs `MultiCamComposedBundle` is the type-level discriminator the project's style asks for. Stages dispatch by isinstance, which is structural-but-typed. The compose path's `if colorize is None` / `if alpha is None` branching remains, however. |

### Smoke-test rule

Every signature change requires a runtime smoke. The clearer the
typed bundles, the harder it is to silently misbehave between
smokes.

| Proposal | Smoke surface |
|---|---|
| 1 | Same as today plus 6 helper unit tests. Trainer-level smokes are the safety net. |
| 2 | Per-strategy unit tests + pipeline integration smokes. Smokes catch combo errors. |
| 3 | Bundle invariant tests + per-stage tests + pipeline integration smokes. Smokes catch fewer things because more is type-checked. |

Proposal 3 makes the smoke-test rule less load-bearing because the
type system handles more. Proposal 1 leaves smokes as the primary
defense.

### Config style — "JSONC configs, normalized once at load"

| Proposal | Config impact |
|---|---|
| 1 | Minimal. One factory (`build_colorize_module_from_config`); rest of config flows through unchanged. The 30+ required-but-not-defaulted keys (investigator 04) remain. |
| 2 | New top-level `pipeline:` field replacing dead `arch:`. Override `strategies:` block for ablation. The strategy registry handles config dispatch; the rest of the config schema stays. |
| 3 | Same as Proposal 2 plus a new `compose:` section for bg policy and view-condition. Pipelines are named in config. |

Both 2 and 3 fix investigator 04's "the `arch` field is dead
documentation" finding. Proposal 1 leaves it.

---

## Final architect's verdict

The user is in a research phase where the trainer is changing weekly
(four trainer files in flight, two of them very young). The product
roadmap is dominated by additive cross-cutting concerns. The bug class
that triggered this audit is exactly the one that types prevent
better than discipline.

**Adopt Proposal 3's bundle vocabulary as the spine. Adopt
Proposal 2's named pipeline registry as the ergonomic surface.
Use Proposal 1's helpers as the migration on-ramp.**

Concretely:

1. **Week 1**: Land Proposal 1's `RenderedClipBundle` and
   `compose_rendered_rgb` helpers. Fix the multicam bug. Move
   `render_clip_sequence` and `viewport_cameras` out of
   `train_video_token_implicit_dynamic.py` into a neutral module
   (no circular import).
2. **Weeks 2–4**: Reify Proposal 3's bundle vocabulary
   (`ClipBundle`, `DecodedBundle`, `RenderedBundle`,
   `SingleCamComposedBundle`, `MultiCamComposedBundle`,
   `LossBundle`). Bundles flow through the existing trainer
   methods; the methods become bundle-consuming bundle-producing
   thin wrappers.
3. **Weeks 5–8**: Extract stages as concrete classes (not a
   generic Stage Protocol). The trainer's `step()` becomes a
   list of stage calls. Adopt Proposal 2's named pipeline registry
   as the config surface.
4. **Weeks 9–12**: Collapse the trainer hierarchy. Delete the
   subclasses. Migrate legacy trainers
   (`dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`,
   `tokenGS.py`) into pipeline registry entries.
5. **Ongoing**: When a new feature lands (distillation, depth,
   anti-cheating), it lands as a new stage + bundle field extension.

The hybrid keeps Proposal 1's pragmatism for the bug fix, Proposal
3's type discipline for the long-term spine, and Proposal 2's
ergonomic pipeline names for the config surface. None of the three
proposers wrote this; this is the architect's synthesis.

If forced to pick exactly one of the three as-written:

- **Proposal 3** for a 12-month horizon. It is the only one whose
  abstractions absorb the additive roadmap without central
  edits. The migration cost is real but front-loaded, and the
  resulting architecture survives.
- **Proposal 2** for a 6-month horizon if Proposal 3's migration
  is not affordable. It honors `key_learnings.md:18` and gives
  good composability for losses, at the cost of weaker types and
  the `RenderHarness` ownership smell.
- **Proposal 1** only if the multicam fix must ship this week and
  the team will not commit to a structural rewrite. It is a
  competent band-aid; the architecture under it ages on the same
  trajectory it has been aging.

End of review.
