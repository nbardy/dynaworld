# Reviewer 01 — Pragmatist Critique

> Angle: I weight every proposal by the cost of a botched refactor against the
> cost of slightly imperfect code. The user just spent a chunk of session on a
> tuple-arity cascade bug. There are 96 configs in tree. `key_learnings.md:18`
> tells us we burned ourselves on shared trainer ancestry already. The MVP is
> "ship the fix, keep training models, do not stall." Anything that breaks
> mid-cascade is unacceptable.

## TL;DR

- **Recommend Proposal 1, immediately and unmodified for the first 5 steps.**
  It lands the actual reported multicam alpha bug in one or two PRs, touches
  ~2 trainer files plus 6 small new files, and invalidates zero configs. Every
  step is runnable end-to-end against an existing smoke config.
- **Reservations on Proposal 1:** it leaves the broader landscape mess
  intact (`dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`, the legacy
  function-style trainers, the "30+ required-but-never-defaulted keys" config
  smell, the silent `feature_dim` drop in `FreeGaussianBankImplicitCamera`).
  These are real, but they are not the bug we shipped.
- **Reservations on Proposal 2:** "one Trainer that takes six strategies" is
  a god-class wearing a Protocol mask. It violates `key_learnings.md:18` in
  spirit even if not in letter. Migration is 9 PRs and the cutover PR (PR 8)
  is where everything fights at once.
- **Reservations on Proposal 3:** ten frozen bundle types, a stage Protocol,
  a registry, two pipeline drivers, and a ~4400-line delete with ~1200 lines
  of new code is a textbook abandoned-rewrite. The chunked-backward story
  ("a nested ChunkedRecoBackwardPipeline") is hand-waved. The user is not
  going to ship this, the user is going to half-ship this.
- **What to do today:** land the `RenderedClipBundle` + `compose_rendered_rgb`
  + `sample_random_background` helpers from Proposal 1 (Steps 1-2). Then
  port the multicam trainer to call them (Step 5 — promote it ahead of
  Steps 3-4 because that's the actual bug). Then stop and decide whether
  the rest of Proposal 1 graduates or whether we need bigger guns.

---

## Proposal 1: Functional Helpers

### Migration cost

| Step | Files touched | Lines changed | Smoke required |
|---|---|---|---|
| 1: package skeleton | 2 new files | ~150 LOC new | unit tests only (no trainer touched) |
| 2: helpers | 3 new files | ~250 LOC new | unit tests only |
| 3: `Trainer.recon_backward` | 1 edit | -35 +25 | F=3 smoke + F=32 alpha smoke (1 step each) |
| 4: `initial_step_result` + method `render_full_sequence` | 1 edit | -60 +30 | adds known-camera 1-step smoke |
| 5: multicam trainer | 1 edit | -40 +30 | F=3 multicam smoke + F=32 alpha multicam smoke |
| 6: colorize factory | 1 edit | -25 +5 | every previous smoke |
| 7: video payload helper | 2 edits | -100 +30 | 1 full validation cycle each path |
| 8: dead-file deletion | -6 files | -50 LOC removed | global grep + import smoke |

Total estimate: ~400 lines added, ~310 lines removed across 2 trainer files
plus 6 new helper modules. Each step is < 1 hour of work plus its smoke. A
careful engineer can land Steps 1-5 in a single day.

### Risk surface

- **Risk 1: bg sampling drifts from per-step to per-chunk.** The proposal
  acknowledges this honestly and recommends "Path A" (preserve verbatim) for
  the migration. As long as the implementer reads the proposal and follows
  the recommendation, this is detectable by comparing loss curves to a recent
  W&B baseline. If they don't read carefully, the F=3 single-cam path will
  silently change behaviour for `microbatch` / `framewise` strategies.
  **Detectability: silent regression on a non-default config.** Mitigation:
  bake the "sample once, pass `custom_bg(tensor)`" pattern into Step 3's
  reference snippet so there's no decision to skip.
- **Risk 2: `RenderedClipBundle` `__post_init__` rejects cases the legacy
  code accepted.** `cameras` length must equal `features.shape[0]`. The
  legacy `render_clip_sequence` doesn't enforce this. If any caller passes
  a mismatched `cameras` (e.g. by accident, a slice-off-by-one), it now
  crashes loudly at the bundle constructor. **Detectability: loud crash at
  the call site that introduced the mismatch.** This is an upgrade.
- **Risk 3: local imports inside `render_clip_with_alpha` create import
  cycles when someone moves `render_clip_sequence`.** The proposal flags
  this with `# local import: existing module`. The future refactor that
  moves `render_clip_sequence` into a shared module breaks this without
  warning. **Detectability: import error at call time, not load time.**
  Mitigation: leave a `TODO(circular-import)` comment, fix when the bigger
  rendering refactor happens.
- **Risk 4: `train_ltx_feature_implicit_dynamic.py` deletion changes the
  stdout prefix for one config.** Trivial. Print prefix loss does not break
  anything, just slightly affects log greppability.
- **Risk 5: `Trainer.recon_backward` no longer broadcasts ONE bg tensor
  across chunks.** The proposal pushes bg sampling into
  `compose_rendered_rgb` per call, which is per-chunk. Multicam case adds
  `sample_random_background` outside the loop. Single-cam case does NOT
  adopt that pattern in the snippet; the snippet says `bg_policy =
  random_bg()` once and then calls `compose_rendered_rgb(... background=bg_policy)`
  per chunk. The `_RandomRGBBackground` class samples each call. **This
  is exactly Risk 1 again.** Spell out the fix in the snippet — pre-sample
  outside the loop, pass `custom_bg(tensor)`. Otherwise it is a silent
  behaviour change.

No cascading signature changes. No multi-file-in-concert edits. Each step
is locally reversible.

### Compatibility with project rules

- `key_learnings.md:18`: **Honored.** No new base trainer. Adds a typed
  payload (`RenderedClipBundle`) which is exactly what the note approves
  ("Shared payload contracts are cleaner than shared trainer inheritance").
- AGENTS.md smoke-test rule: **Honored.** Every step ends with a real
  `python <trainer>.py <config>` smoke that exercises the call graph.
  Not just `py_compile`.
- 96 configs preserved: **Yes.** Zero config files edited. The only
  config-adjacent change is dropping the empty `LTXFeatureImplicitTrainer`
  alias, which one config dispatches to — and the launcher script for that
  config already targets `train_precomputed_feature_implicit_dynamic.py`
  per Investigator 01. Risk is tiny.

### Time-to-first-value

After Step 1+2 (package + helpers): nothing visible yet, but the smoke tests
prove the helpers work in isolation. Probably 2 hours.

After Step 5 (multicam fix): **the actual bug is fixed.** F=32 alpha
multicam configs that crash today produce a finite loss. This is the
fastest of any of the three proposals to deliver this user-facing fix.
Estimated: 1 day from cold start.

After Step 8 (cleanup): same value, slightly tidier tree. ~3 days from cold
start.

### What it gets right

- **Honest about scope.** Says "doesn't unify the legacy trainers, doesn't
  fix the renderer-side dispatch, doesn't solve the config-schema mess,
  doesn't fix `feature_dim` silent drop in `FreeGaussianBankImplicitCamera`."
  Names what stays broken. Other proposals overstate what they "fix" by
  way of architecture.
- **The bundle/policy separation is clean.** `RenderedClipBundle` is the
  smallest possible typed surface. `BackgroundPolicy` as a sum type is
  exactly the right grain — not a `bool training` confusion, not a giant
  struct.
- **Test surface is minimal but real.** Six tests, all falsifiable. The
  `test_bundle_rejects_wrong_alpha_shape` test is a literal regression
  guard for the multicam tuple-as-tensor bug. That's a Good test by the
  AGENTS.md "tests should catch a future bug" rule.
- **Migration steps are independently runnable.** Mid-cascade states are
  not py_compile-clean-but-broken; they actually run a model.
- **Aligns with the user's existing TODO/trainer_landscape_unification.md.**
  Reading that TODO, Proposal 1 is essentially the user's own plan written
  out with code stubs. Lower friction in review.

### What it gets wrong

- **Doesn't fix the renderer's silent F-dispatch in `fast_mac.py`.**
  Investigator 03 calls this out as a structural smell. Proposal 1 lives
  one layer above it. A future rewrite that wants to remove the v5/v5_features
  channel-count branch has to redo work.
- **Leaves `KnownCameraTrainer` alive.** Pure inheritance, parent class
  `Trainer` still has all of the same `**` kwargs and overrideable methods.
  The class of bug ("subclass overrides `step` and silently bypasses the
  parent's safety net") that bit us in multicam is still possible in any
  future subclass.
- **`build_colorize_module_from_config` is the only piece that touches
  `cfg.get(default)` smells.** Investigator 04 documents 30+ other places.
  Proposal 1 explicitly punts.
- **Step 5 (the actual bug fix) is fifth.** I'd land it second or third —
  the prep steps (1, 2, 3, 4) are valuable but they don't ship the fix.
  Re-order: 1, 2, 5, 3, 4, 6, 7, 8.

### Verdict

**Ship it.** Lowest blast radius, fastest time-to-first-value, fewest
config invalidations, fewest cross-file edits. Best-aligned with the
user's existing TODO. Reorder the steps so the multicam fix lands first.

---

## Proposal 2: Composition Over Inheritance

### Migration cost

| PR | Scope | Files | LoC delta | Smoke |
|---|---|---|---|---|
| 1: lift `pick_device`/etc to `train_runtime.py` | 4 file edits | ~50 | existing trainers launch |
| 2: extract `VideoLogger` | new module + 4 trainer edits | ~400 | F=3 smoke + multicam smoke; W&B panels match |
| 3: extract `LossFn` (introduces `BackwardStrategy`) | new module + 4 trainer edits | ~600 | 30-step overfit single + multicam |
| 4: extract `RenderHarness` (folds composition + bg) | new module + 4 trainer edits | ~700 | full F=3 + F=32 alpha + multicam smokes |
| 5: extract `Validator` | new module + 4 trainer edits | ~400 | full validation cycle |
| 6: extract `ClipSampler` (feature-cache lifecycle) | new module + 4 trainer edits | ~800 | full matrix |
| 7: extract `ModelBuilder` | new module + 4 trainer edits | ~300 | every variant builds |
| 8: collapse trainer hierarchy + delete subclasses + delete `arch` field from configs | massive | ~2000 | every config in tree |
| 9: absorb `dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`, `tokenGS.py` | rewrite three legacy trainers as strategy combos | ~1500 | every legacy config |

Total estimate: roughly 6800 LoC delta across 9 PRs. PR 8 is the cutover
that touches every train config. PR 9 is a full rewrite of three legacy
trainers.

### Risk surface

- **PR 4 fights with PR 3.** `BackwardStrategy` is decided inside `LossFn`
  but the chunked-render-tensor lifetime is owned by `RenderHarness`. The
  proposal acknowledges this in §"6. The eager-concatenation render
  contract" but doesn't actually solve it. Naive eager concatenation
  raises peak memory by `T * 3 * H * W * dtype_bytes` per step on the
  framewise strategy. For 16f / 256px / fp16 that's 6 MB of pixels — fine —
  but the autograd graph for that concatenated tensor is the real cost,
  and it depends on `train_frame_count` and chunk size. **Detectability:
  silent OOM on the configs that exercise `framewise` strategy.**
  Mitigation: actually implement the chunked path before claiming it.
- **PR 8 is the cutover.** Every config in tree has `arch` deleted and
  `pipeline:` added. 96 configs. The proposal mentions "a small Python
  script" — not actually written. **Detectability: every old launcher
  script breaks at once.**
- **The "strategies are not inheritance" argument is technically true
  but operationally false.** A `Trainer` that accepts six strategies in
  its constructor and dispatches to them in `step()` IS a god-class. The
  Method Resolution Order (MRO) bug shape that bit multicam — "subclass
  overrides `step` and bypasses parent" — does not exist, true. But a new
  shape replaces it: "the multicam strategy combo is missing one of the
  six required strategies, or has a wrong-protocol-shaped one." The
  `runtime_checkable=True` mitigation catches some of this, but not the
  case where `MulticamLoss.compute()` returns the wrong `BackwardStrategy`
  variant. **Detectability: silent gradient-flow weirdness, not a crash.**
- **`render_harness.bind_colorize(self.colorize)` after construction.**
  The proposal acknowledges this as a "circular-dependency hack." Hacks
  in the trainer constructor are exactly where bugs hide. Either the
  ModelBuilder owns colorize, or RenderHarness owns colorize, but the
  current "build it in ModelBuilder, inject it via a setter into
  RenderHarness later" violates the strategy's invariant ("strategies are
  immutable after construction").
- **`StrategyTuple` has 6 fields in fixed positions.** Adding a 7th
  strategy (e.g. an explicit `BackgroundPolicy` strategy, which the
  proposal punts to a constructor arg on `RenderHarness`) means an
  N-place edit.
- **Configs change shape.** `pipeline:` field is new. `arch:` field is
  removed. `compose:` section is new (per the eval-bg-policy split).
  Mid-flight ablation configs that exist as branches will all need
  rewriting. **Cost: depends on config count in flight.**

### Compatibility with project rules

- `key_learnings.md:18`: **Disputed.** The proposal claims structural
  Protocols are not inheritance. Technically correct. Operationally,
  one `Trainer` class wired to six strategies is a single shared
  trainer. The note's spirit ("real differences should be visible, not
  hidden behind shared shape") is partially honored (strategies make
  differences explicit) and partially violated (the one Trainer class
  treats every variant identically through the protocol surface). I
  count this as **depends on intent**.
- AGENTS.md smoke-test rule: **Mostly honored.** PRs 1-7 each name a
  smoke. PR 8 says "every config in `src/train_configs/` runs at least
  one step" which is 96 smokes. That's optimistic to claim done in one
  PR.
- 96 configs preserved: **No.** Every config gains `pipeline:` and loses
  `arch:`. Even if the migration script is bug-free, every active
  branch's configs will diverge.

### Time-to-first-value

PRs 1-3 give nothing user-visible. They are pure refactors.

PR 4 fixes the bug — that's the fast-cycle equivalent of Proposal 1's
Step 5. Estimated: 1-2 weeks (4 PRs, with smoke tests, with code review).

PR 8+9 deliver the architecturally-clean state: ~6 weeks if it lands at
all.

### What it gets right

- **The right insight: composition is not orthogonal to rendering.** The
  prose under "How alpha-aware composition + random bg lives" is the
  best-articulated argument in any of the three proposals. Putting
  composition inside `RenderHarness` (not as its own strategy) is
  correct.
- **`mock.patch(target, side_effect=AssertionError)` test pattern is
  exactly the right architectural-invariant test the project uses
  elsewhere.** `tests/test_multicam_loss_no_silent_skip.py` is the
  right shape.
- **Names are good.** `ClipSampler`, `RenderHarness`, `LossFn`, etc.
  read fluently. They are what they sound like.
- **Honest about migration size.** "9 PRs" is stated. "PRs 1-3 alone
  fix the bug we just shipped" is a real escape hatch.

### What it gets wrong

- **The N×M strategy combination problem is real and the answer
  ("`PIPELINE_REGISTRY` validates compatible pairs") is hand-waved.**
  6 strategies × 3-5 implementations each is 729 raw combos and most
  are nonsensical. The proposal says "a `pipeline_registry.py::validate(tuple)`
  guard at trainer init that asserts compatible pairs" — but that guard
  is essentially encoding the same dispatch table that `arch` was
  supposed to encode. We're back to a discriminator field in a different
  hat.
- **`BackwardStrategy` lives on `LossOutput` but the Trainer's `step()`
  has a `match` statement on it.** The proposal's stated principle is
  "the chunked-backward decision lives in `LossFn`." Then why does the
  Trainer dispatch on it? Either fully encapsulate inside `LossFn` (have
  `LossFn` take the optimizer and call `.backward()` itself — gross), or
  drop the abstraction (Trainer always does single-shot — drops a real
  feature for memory-bounded training).
- **PR 9's "absorb the legacy trainers" is its own multi-week project
  pretending to be one PR.** `dynamicTokenGS.py` has weight-decay-aware
  param groups, lr_schedule, clip_grad_norm. Lifting these into the
  strategy framework requires rethinking the optimizer step. None of
  Proposal 2's strategies have a "post-backward grad-clip + LR-schedule
  step" hook. Adding one expands the protocol surface.

### Verdict

**Iterate.** Strong design, too much migration risk for the user's current
iteration tempo. If we end up wanting Proposal 2's structure 6 months from
now, we should land Proposal 1 first and then graduate. Specifically: PRs
1 and 2 (lift utilities + extract VideoLogger) are reasonable PRs even
without the rest; PRs 3-9 are a structural rewrite.

---

## Proposal 3: Typed Dataflow Pipeline

### Migration cost

| Step | Files | LoC delta | Smoke |
|---|---|---|---|
| 1: bundles + protocol + context (no behavior) | 4 new files | ~600 LOC new | bundle invariant unit tests |
| 2: single-cam F=3 + driver + side-by-side flag | ~12 new files; 1 trainer-file shim | ~1200 LOC new | 100-step diff vs old trainer |
| 3: single-cam F=32 alpha port | registry add + bundle invariant tests | ~100 | 400-step F=32 alpha; loss curve match within 1% |
| 4: multicam port (the bug fix) | 2 new stages + registry | ~300 | 16-step multicam smoke |
| 5: port remaining trainers; delete old; rewrite launchers | massive | ~2000 added; ~4400 removed | every legacy config |

Total stated: "roughly 4400 lines deleted, replaced by roughly 1200 lines."
Realistic estimate is 50% higher than stated for the new code (helper
threading, parent-pointer plumbing, error messages, edge cases). Realistic
PR count is 5-8 substantial PRs over 4-8 weeks.

### Risk surface

- **The chunked-backward problem is genuinely unsolved.** §"4. The chunked
  backward is awkward" admits it: "the cleanest answer is to make the
  chunked path a nested `ChunkedRecoBackwardPipeline` that produces N
  `BackwardBundle`s and merges them. The current code does
  `backward_loss.backward(retain_graph=not is_last_chunk)` inside a loop,
  which is naturally a sub-pipeline." This is hand-waved. A nested
  pipeline that re-renders + re-composites + re-loss-computes per chunk
  is fundamentally different from the current "render features once,
  loop chunks for backward" pattern. The current `Trainer.recon_backward`
  loops `render_clip_sequence` per chunk too — but it shares the
  `random_bg` across chunks, which the proposal's `ComposeStage` would
  re-sample inside each chunk's pipeline pass. **Detectability: subtle
  loss-curve drift on `framewise` and `microbatch` strategies, no crash.**
- **`bundle.parent.parent.parent` chain is a foot-gun.** §"_single_cam_recon"
  in `LossStage` walks `SingleCamComposed → Rendered → Decoded →
  ModelInput.parent (ClipBundle)`. The proposal calls this "explicit; no
  global context needed" — it's the opposite, it's a typed-but-fragile
  global context where the path's correctness depends on the upstream
  stage list. Any insertion of an intermediate stage breaks this chain
  silently. **Detectability: AttributeError or wrong-shape tensor at the
  loss stage, the third stage downstream from the actual bug.**
- **10 bundle types is a lot of memory work.** `ClipBundle`, `MulticamClipBundle`,
  `ModelInputBundle`, `DecodedBundle`, `RenderedBundle`, `MultiViewRenderedBundle`,
  `SingleCamComposedBundle`, `MultiCamComposedBundle`, `LossBundle`,
  `BackwardBundle`, `OptimizerBundle`, `StepResult`, `PreviewBundle`,
  `ValidationBundle`, `VideoLogBundle`, `AuxiliaryDecode`. That's 16
  frozen dataclasses to keep alive in the heap per step. Frozen
  dataclasses with tensors aren't free; `.parent` chains keep upstream
  bundles alive longer than they need to be, defeating the optimizer's
  ability to free intermediate tensor memory. **Detectability: silent
  MPS memory growth, no crash.**
- **PipelineContext is a god object even by the proposal's own
  admission.** It carries `model`, `colorize`, `optimizer`, `device`,
  `dtype`, `rng`, `cfg`, `is_eval`, `feature_cache`, `multicam_bundle`,
  `camera_rig`, `sequences`, `eval_sequences`. 13 fields. A god object
  that pretends to be a stage-local config is worse than one that admits
  it.
- **Step 2 introduces a `--use-pipeline` flag.** Now the codebase has
  TWO trainers per config — the old `Trainer.run` and the new
  `Pipeline.run_step` — and a flag to choose between them. This is
  exactly the kind of dual-system state the project's `dev` rules call
  out as a smell. The plan to remove it (Step 5) is "port remaining
  trainers; delete old files." That's 4400 LOC of deletion across N
  legacy trainers, all in one step. **Detectability: depends on whether
  Step 5 lands at all.**
- **Configs change.** `arch` removed, `pipeline:` added, new `compose:`
  section, every config rewritten. 96 configs.

### Compatibility with project rules

- `key_learnings.md:18`: **Honored in spirit.** No shared base class,
  pipeline is data. The note's intent (real differences should be
  explicit) is honored: the multicam pipeline's stage list shows exactly
  where it differs from the single-cam one.
- AGENTS.md smoke-test rule: **Mostly honored.** Each step has a smoke,
  but Step 5 is "every legacy config" which is many smokes lumped into
  one PR.
- 96 configs preserved: **No.** Every config rewritten.

### Time-to-first-value

Step 4 is when the bug is fixed. That's after Steps 1, 2, 3 — i.e. after
a complete single-cam pipeline runs end-to-end with parity. Estimated:
3-4 weeks.

Step 5 is the architectural payoff. Estimated: 6-10 weeks total to land
fully. Or, more realistically, ~80% landed and the legacy trainers
linger because their configs aren't worth porting.

### What it gets right

- **The bundle types are the cleanest stated typed surface of any of the
  three proposals.** If we were starting from scratch, this is what the
  trainer would look like.
- **"The Apr 29 bug class becomes structurally impossible" is true for
  the alpha-aware composition seam.** There IS one `ComposeStage`. The
  multicam pipeline DOES use it.
- **Per-stage tests on synthetic bundles is a real win.** You can test
  `ComposeStage` on a `RenderedBundle` directly without spinning up a
  trainer. This is impossible today.
- **The eval pipeline / training pipeline symmetry is elegant.** Same
  bundle vocabulary, swap `LossStage`+`BackwardStage`+`OptimizeStage`
  for `MetricStage`+`MediaPayloadStage`. The "training: random bg; eval:
  white bg" asymmetry becomes a single-flag difference.

### What it gets wrong

- **The chunked-backward design is unfinished.** The whole proposal
  rests on "stages are pure functions producing one bundle out per
  bundle in" — and that mental model breaks the moment you have to
  call `.backward()` N times with `retain_graph=True` on different
  scalars. The "nested ChunkedRecoBackwardPipeline" answer is a sketch,
  not a design.
- **`bundle.parent.parent.parent` is a worse abstraction than what we
  have.** Today, `Trainer.recon_backward` reaches `self.cfg`, `self.colorize`,
  `self.render_size` and that's plain old method dispatch. A walk down
  N parent pointers is "implicit context inverted into a typed chain."
  The same context has to thread somewhere.
- **The "delete 4400 lines, add 1200 lines" math is wrong.** Yes the
  trainer files shrink. But the bundle types, stage classes, registry,
  pipeline driver, eval pipeline, and bundle invariant tests easily add
  1500-2000 LoC of their own (the proposal estimates 1200). The legacy
  trainer paths (`dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`,
  `tokenGS.py`) are not "small registry entries"; they have their own
  optimizer schemes, eval payloads, model API call shapes. Each is its
  own port.
- **"The pipeline is data, not code" is true for the registry but not
  for the stage classes.** Every stage has a constructor with config-shaped
  args. Adding a knob (e.g. `compose.background_policy = "config"`)
  means editing a stage class, a registry entry, and a config schema.
  The "data, not code" claim only holds for the wiring layer.
- **Step 2's "side-by-side comparison flag" is exactly the dual-system
  smell the project's `dev` rules call out.**

### Verdict

**Shelve.** The design is correct in the limit, but the migration cost
is substantially higher than stated, the chunked-backward path is
unfinished, and the cutover needs ~6 weeks of focused work to land
without leaving dual systems. Given the user's iteration tempo, this is
an abandonment risk. Proposal 1 plus a future "graduate to bundles" pass
gets us 80% of the value at 10% of the cost.

---

## Comparison table

| Dimension | Proposal 1 | Proposal 2 | Proposal 3 |
|---|---|---|---|
| Time to first usable refactor (Steps 1-2 / PR 1-2) | ~2 hours | ~3-5 days | ~1-2 weeks |
| Time to bug fix landed | ~1 day (Step 5) | ~2 weeks (PR 4) | ~3-4 weeks (Step 4) |
| Lines changed at first checkpoint (helpers in tree, no trainer touched) | ~150 LOC | ~50 LOC (PR 1) | ~600 LOC |
| Lines added total | ~400 | ~3000 | ~1500-2000 |
| Lines deleted total | ~310 | ~4500 | ~4400 |
| Files added | 6 | ~10 | ~16 |
| Files deleted | 6 (all confirmed dead) | ~12 | ~12 |
| Trainer files edited concurrently | 2 | 4 → 1 collapse | All trainers replaced |
| Configs rewritten | 0 | 96 (PR 8) | 96 (Step 5) |
| Risk of silent regressions | low (per-chunk bg drift on framewise) | medium (BackwardStrategy / `match`, `bind_colorize` hack) | high (chunked-backward unfinished, parent-pointer chain, MPS memory growth) |
| Long-term maintainability | medium (legacy trainers untouched) | high (clean strategy surface, but god-class risk) | highest (typed dataflow, but 16 bundle types) |
| Onboarding cost for new contributor | low (read 6 small files) | medium (read 6 protocols + registry) | high (read 16 bundles + 9 stages + 2 drivers) |
| Aligns with `key_learnings.md:18` | yes | depends | yes |
| Solves the multicam alpha bug | yes (Step 5) | yes (PR 4) | yes (Step 4) |
| Fixes `KnownCameraTrainer.initial_step_result` tuple bug | yes | yes | yes |
| Fixes `FreeGaussianBankImplicitCamera` `feature_dim` silent drop | no | yes (PR 7) | maybe (depends on builder fix) |
| Touches the renderer's `fast_mac.py` F-dispatch | no | no | no (only wraps it) |
| Solves config-schema mess (96 configs, dead `arch`) | no | yes (PR 8) | yes (Step 5) |
| Adds new TODO debt | low (legacy trainers stay) | medium (PR 9 always pending) | high (Step 5 cutover always pending) |
| Mid-cascade states runnable end-to-end | yes (every step) | yes (PRs 1-7); PR 8 is cutover | yes (Step 1-4); Step 5 is cutover |
| Dual-system flag during migration | no | no | yes (`--use-pipeline`) |
| Abandonment-completion risk | low | medium-high | high |

---

## The pragmatist's recommendation

**Hybrid:** ship Proposal 1 unmodified for a Phase 1, with Step 5
reordered to be Step 3 (the multicam fix lands as soon as the helpers
exist).

After Phase 1 stabilizes (~1 week of training cycles, several W&B runs
on the actual research workload), make a deliberate decision:

- **Stop.** The remaining mess (legacy trainers, 96-config drift,
  `FreeGaussianBankImplicitCamera`'s silent `**_unused`) is documented
  in TODOs but not bleeding. Continue research.
- **Phase 2: graduate to Proposal 2's strategies.** Proposal 1's helpers
  become the bodies of `RenderHarness` and `LossFn` strategies. The
  trainer hierarchy doesn't collapse in Phase 2 — that's Phase 3. By
  this point, we have a real reason (e.g. a fifth trainer kind appears)
  to justify the abstraction cost.
- **Skip Phase 2, attempt Proposal 3.** Only if a major change forces
  the dataflow-typing question — e.g. a new modality adds a stage that
  doesn't fit the current shape. Without that forcing function, this is
  YAGNI.

Concretely:

- **Phase 1 (now): Proposal 1, Steps 1, 2, 5, 3, 4, 6, 7, 8.**
- **Phase 2 (deferred): if a fifth trainer kind emerges, lift the
  helpers into `RenderHarness` / `LossFn` Protocols. Keep the trainer
  hierarchy.**
- **Phase 3 (probably never): if config drift becomes intolerable,
  consider Proposal 3's bundle types.**

What graduates: `RenderedClipBundle` (already a Proposal 1 dataclass),
`compose_rendered_rgb`, `compute_recon_loss`. They become the seam that
Proposal 2's strategies wrap.

What stays: the trainer subclass hierarchy, the legacy function-style
trainers, the `cfg.get(default)` smells outside `colorize`, the 96
configs.

---

## Concrete next-action checklist

For this week, ordered:

1. **Land Proposal 1 Step 1+2 in one PR.** New package
   `src/train/training_common/` with `render_bundle.py`, `compose.py`,
   `recon_loss.py`, `render_clip.py`. Add the 6 unit tests
   (`test_render_bundle.py`, `test_compose.py`,
   `test_sample_random_background.py`).
   - **Files:** 6 new files in `src/train/training_common/`,
     3 new files in `tests/`.
   - **Smoke:** `uv run pytest tests/test_render_bundle.py
     tests/test_compose.py tests/test_sample_random_background.py`.
   - **Done:** all 6 tests pass; no existing trainer touched; no config
     touched.

2. **Land Proposal 1 Step 5 (multicam fix) in the next PR.** Rewrite
   `MulticamPrecomputedFeatureImplicitTrainer.render_view_clip` to
   `render_view_clip_bundle` returning a `RenderedClipBundle`. Rewrite
   `multicam_recon_loss`, `initial_step_result`,
   `render_full_external_views` to use `compose_rendered_rgb` with a
   `custom_bg(sample_random_background(...))` shared across views.
   - **Files:** 1 trainer file edit
     (`src/train/train_multicam_precomputed_feature_implicit_dynamic.py`).
   - **Smoke 1:** `uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py
     src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc`
     for 2 steps. Finite loss + non-NaN preview.
   - **Smoke 2 (the actual bug fix):** `uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py
     src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc`
     with `train.steps=2` overridden. Previously this would have crashed
     at `reconstruction_loss_per_image(rendered, ...)` because `rendered`
     was a tuple. Now it produces a finite loss.
   - **Done:** both smokes finite + non-NaN; user runs a longer training
     cycle (100-400 steps) on a research-relevant config and confirms
     loss curve looks reasonable.

3. **Validate against a recent W&B baseline.** Pick the most recent
   F=32 alpha single-cam run (e.g. `3reqcya9` from the loose note), run
   1 step in the new code, compare scalar payload fields. They should
   match within RNG noise. If they diverge by more than ~1% on the
   first step, the per-chunk bg drift (Risk 1) has materialized — fix
   by pre-sampling outside the loop with `sample_random_background`
   and passing `custom_bg(tensor)`.

4. **Stop and decide.** With (1)-(3) landed, the load-bearing bug is
   fixed and the helpers exist. The user makes a call: graduate to
   Proposal 2 or stop here.

If steps 1-3 take more than 3 working days, that is a signal that
Proposal 1 is harder than it looks — not a signal to escalate to
Proposal 2.

---

## Risks that apply across all three proposals

### Tuple-arity cascades

All three introduce new bundle/tuple types whose construction has to
match the consumers' destructuring. The user already burned a session
on the `(features, alpha)` tuple-vs-tensor cascade. New cascades to
watch:

- **Proposal 1:** `RenderedClipBundle.__post_init__` validates shapes;
  bundle is frozen, has a clear constructor surface. **One new failure
  shape: passing a `(features, alpha)` raw tuple where a bundle is
  expected.** Detectable as `AttributeError: 'tuple' object has no
  attribute 'features'`. Loud crash. Mitigation built in.
- **Proposal 2:** `RenderHarness.render()` returns `RenderedClipBundle`
  with `rgb` and `pre_composite_features` as named fields. Same shape
  of mitigation. **New failure shape: a strategy combo where the
  `RenderHarness` returns the wrong leading-dim shape (e.g.
  `MultiViewRenderHarness` returns `[V, T, 3, H, W]` but the `LossFn`
  in the combo expects `[T, 3, H, W]`).** Detectable as a shape
  mismatch deep in the loss kernel.
- **Proposal 3:** 16 bundle types. Each has a `__post_init__`. Each
  consumer destructures named fields. **New failure shape:
  `bundle.parent.parent.parent` — the parent-pointer walk in the loss
  stage. If a stage is inserted that doesn't carry `parent`, the walk
  crashes 3 stages downstream of the bug.** Hardest of the three to
  reason about.

**Mitigation per proposal:** Proposal 1 wins here — fewest new bundle
types, simplest validation surface.

### Config compatibility

96 configs in tree, JSONC convention is canonical, no inheritance
between configs.

- **Proposal 1:** zero configs rewritten. Best case.
- **Proposal 2:** 96 configs rewritten in PR 8. Adds `pipeline:`,
  removes `arch:`, possibly adds `compose:` for the bg policy split.
- **Proposal 3:** 96 configs rewritten in Step 5. Adds `pipeline:`,
  removes `arch:`, adds `compose:`, possibly more.

Both Proposal 2 and Proposal 3 wave at "a small Python script" for the
migration. Neither writes the script. The 96 configs include
research-active branches; cross-branch merges become a nightmare.

**Mitigation:** Proposal 1 wins here — zero rewrites.

### Codex / external rasterizer boundary

The user noted that nothing should require Codex to redo the
v5_features rasterizer extension work. All three proposals respect
this — none of them edit
`third_party/fast-mac-gsplat/variants/v5_features/`. The
`rendering.py` / `renderers/fast_mac.py` boundary stays intact in
Proposal 1 (untouched), Proposal 2 (untouched), Proposal 3 (wrapped
by `RenderStage`).

**Mitigation:** all three respect the boundary.

### The `arch` field is dead code

Investigator 04 documents this: `arch` is read by `init_diagnostics.py`
and `probe_init_diagnostics.py` only, for logging. Production trainer
dispatch is by launcher script + `model.variant`.

- **Proposal 1:** does not touch `arch`. Keeps it as documentation.
  Possibly the right call — deleting a field across 96 configs requires
  PR-level coordination, and the field doesn't bleed.
- **Proposal 2:** deletes `arch` in PR 8 across 96 configs.
- **Proposal 3:** deletes `arch` in Step 5 across 96 configs.

**Mitigation:** Proposal 1 wins by leaving the dead field alone.

---

## What I'd push back on if I were the user

For Proposer 1:

- "Step 5 (the actual bug fix) is fifth out of eight steps. Why is the
  bug fix not Step 2 or 3? The user's working complaint is the multicam
  alpha gap, not the single-cam refactor. What does the smoke for Step 5
  cost if Steps 3-4 haven't landed?" — In particular, Step 5's snippets
  reference `self.colorize_view_condition` etc., which I think still
  exist on the trainer pre-Step-6, so I believe Step 5 is reorderable
  to second.
- "The single-cam `recon_backward` snippet (§Trainer.recon_backward)
  uses `bg_policy = random_bg()` once outside the loop, then passes it
  per-chunk to `compose_rendered_rgb`. But `_RandomRGBBackground` is
  empty + frozen — `_materialize_background` samples a fresh
  `torch.rand(3)` every call. That's per-chunk, not per-step. Either
  fix the snippet to pre-sample with `sample_random_background` and
  use `custom_bg(tensor)`, or admit the per-chunk drift is intentional
  and document why the loss curves still match."
- "What happens to the `bundle.cameras` field if a future
  `viewport_cameras` change makes the post-scaling cameras a tuple of
  CameraSpec but with extra metadata? The bundle is frozen, can't be
  evolved without breaking every caller."

For Proposer 2:

- "A `Trainer` that takes 6 strategies in its constructor and dispatches
  to them in `step()` is still a god-class. The MRO bug shape is
  replaced by a `StrategyTuple`-shape bug (the multicam combo has the
  wrong `LossFn`). How does the testability story differ from a
  base-class hierarchy? You list a `mock.patch` test that asserts the
  composition kernel is reached on the multicam path — that's the same
  kind of test that would catch a multicam subclass bypassing
  `recon_backward`. The structural-impossibility claim relies on
  `MultiViewRenderHarness` being the only path; what stops a future
  contributor from writing a `BypassMulticamHarness` strategy?"
- "PR 9 absorbs `dynamicTokenGS.py`'s `lr_schedule`, weight-decay-aware
  param groups, and `clip_grad_norm` into the strategy framework. None
  of the six protocols today have a "post-backward grad-clip" hook. How
  many additional protocols / strategy methods are needed to complete
  PR 9 honestly? Is the answer 'we drop those features'? If so, that's
  a real semantic loss for the prebaked-camera config family."
- "The `render_harness.bind_colorize(...)` post-construction injection
  is called a hack in the proposal. What goes wrong if `ModelBuilder`
  returns a tuple `(model, colorize)` and the trainer constructor
  passes both into `RenderHarness.__init__`? The 'circular dep' is the
  trainer's, not the harness's. It seems trivially fixable."

For Proposer 3:

- "Ten bundle types. Why not four (`Sample`, `Decoded`, `Composed`,
  `LossOut`)? `BackwardBundle`, `OptimizerBundle`, `LossBundle`,
  `ModelInputBundle` could collapse — they're each a shallow envelope
  around a tensor + bookkeeping. What does the extra granularity buy
  beyond ceremony?"
- "The chunked-backward solution ('a nested
  ChunkedRecoBackwardPipeline that produces N BackwardBundles and
  merges them') needs to be designed before this proposal can be
  evaluated. As stated, it implies the per-chunk pipeline runs the
  full Render+Compose+Loss N times — which changes memory and gradient
  semantics from today. Show the actual stage list for the chunked
  case, including what gets re-rendered vs. cached."
- "`bundle.parent.parent.parent` is the loss stage walking up the
  bundle chain to reach the original `ClipBundle.clip_frames`. If a
  future stage is inserted (e.g. an explicit `ResizeStage` that does
  GT resize), the parent chain depth changes and the loss stage
  silently breaks. Why is this better than `LossStage` taking GT as a
  named field on `ComposedBundle`?"
- "Step 2 introduces a `--use-pipeline` flag with both old and new
  trainer code in tree at the same time. The project's `dev` rules
  call out dual-system state as a smell. How long does this flag live?
  When does it get removed? If Step 5 stalls, does the flag become
  permanent?"

---

End of review.

agent_notes/apr_30th_clean/reviewer_01_pragmatist.md
