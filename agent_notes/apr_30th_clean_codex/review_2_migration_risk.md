# Review 2: Migration Risk, Smoke Coverage, Deletion Safety, Provenance

Reviewer:
    Reviewer 2

Scope:
    Review the three April 30 cleanup proposals for migration risk, smoke
    coverage, deletion safety, and experiment provenance.

Reviewed files:

- `agent_notes/apr_30th_clean_codex/proposal_a_shared_objective_pipeline.md`
- `agent_notes/apr_30th_clean_codex/proposal_b_viewbatch_recipe_registry.md`
- `agent_notes/apr_30th_clean_codex/proposal_c_cleanup_deletion_migration_plan.md`

Non-scope:
    I did not edit proposal docs, code, configs, `BASELINES.md`, or any other
    docs. This file is the only intended output.

## Executive Verdict

The three proposals are directionally right and mutually compatible:

- Proposal A correctly identifies the shared `RenderObjective` as the central
  safety boundary.
- Proposal B correctly turns `arch` into real recipe routing and makes
  `ViewBatch` the common single-cam/multicam data contract.
- Proposal C correctly treats deletion as the last phase, not the cleanup
  mechanism.

But implementation must be gated more strictly than the proposals currently
state. The highest-risk failure is not that the refactor fails loudly. The
highest-risk failure is that it trains a run that looks plausible while missing
random-background provenance, missing alpha composition, using train-view PSNR
as a held-out claim, or silently routing multicam F32 through the old broken
render/loss path.

Required pre-implementation changes:

1. Add explicit expected-fail route guards before any router redirects old
   commands.
2. Make random background config-visible before any new F32 comparison run.
3. Reorder `FeatureProvider.describe()` ahead of any multicam V-JEPA ultimate
   smoke, because that path depends on precomputed feature metadata before model
   construction.
4. Freeze baseline provenance in a small audit note before changing behavior.
5. Keep all legacy entrypoints as compatibility shims until route smokes and
   artifact checks pass.

## High-Risk Items

| Severity | Risk | Proposal coverage | Required gate |
|---|---|---|---|
| P0 | Multicam F32 silently trains without shared alpha/colorize/random-bg objective | A/B/C all identify the broken path | Add route validation that blocks `multicam_precomputed_feature_implicit_camera` with `feature_dim != 3` until the shared objective is actually wired. |
| P0 | Random background remains code-only, so W&B/config cannot prove what happened | A and C call this out clearly | Add normalized `losses.background` and W&B fields before rerunning or comparing F32 configs. |
| P0 | Router redirects old scripts before parity | B and C propose shims, but this needs a hard no-regression rule | `train.py explain` can land first; `train.py run` redirects only after route-specific smokes pass. |
| P0 | Baseline claims drift from source-view loss into novel-view claims | B mentions held-out as default selector; C has BASELINES rules | Require held-out metrics for any multicam or novel-view baseline row. Source/train-view metrics are diagnostics only. |
| P1 | Known-camera validation still passes a tuple to colorize/loss | A and C identify this exact stale path | Known-camera 1-step smoke must exercise `initial_step_result` and final validation before any script redirect. |
| P1 | Feature cache/model construction order breaks V-JEPA/LTX/Wan | B and C propose `FeatureProvider.describe()` | FeatureProvider extraction must happen before multicam V-JEPA ultimate, not after. |
| P1 | Deleting `dynamicTokenGS.py` breaks newer trainers that import utilities | B and C say keep/extract first | `rg` must prove no new trainer imports helpers from `dynamicTokenGS.py` before deletion or retirement. |
| P1 | Gauge-field route gets accidentally merged into video-token abstractions | B/C mark it external | Router should delegate, not import or normalize gauge internals in this cleanup. |
| P1 | Duplicate implicit-camera shims are deleted before old commands are replaced | C says one shim period | First turn them into shims that call the router and print deprecation; delete only after command/reference audit. |
| P2 | Random background nondeterminism makes parity thresholds brittle | A notes stochasticity | Use sanity ranges and artifact checks, not exact loss equality. Log mode, sample scope, seed, and optionally sampled RGB mean. |
| P2 | W&B media volume explodes for multicam alpha/PCA/composite videos | B leaves primary-heldout question open | Define primary train and primary heldout media as required; all-view media can be opt-in. |
| P2 | Stale probes and diagnostics keep old arch names | Investigator reports noted stale `probe_init_diagnostics.py` | Add probes to route audit or explicitly mark them legacy before implementation claims full coverage. |

## Proposal A Review: Shared Render Objective Pipeline

Strong points:

- It states the right invariant: only final RGB reaches reconstruction loss.
- It centralizes the exact fragile equation:

```python
rgb = alpha * splat_rgb + (1.0 - alpha) * background
```

- It treats random train background as policy, not a local tensor allocation.
- It explicitly says multicam must not merely unpack the tuple.
- It includes known-camera validation and held-out multicam video artifacts in
  the smoke matrix.

Migration risks to tighten:

1. The proposal should explicitly require route guards before objective
   migration. Today it says multicam F32 should pass after migration, but an
   implementer could try to make it "work" with local tuple unpacking. Add a
   pre-migration expected-fail state:

```text
feature_dim != 3 and route lacks feature_alpha objective -> fail with ValueError
```

2. The proposal should specify objective versioning. W&B and `BASELINES.md`
   should be able to distinguish:

```text
objective.version = "legacy_rgb_v0"
objective.version = "feature_alpha_white_v1"
objective.version = "feature_alpha_random_bg_v2"
```

Names do not need to be exact, but a version field is needed.

3. The proposal should add an RNG/provenance contract for `BackgroundPolicy`.
   Random background sampled once per step is correct, but a future agent needs
   to know whether it is seeded by global torch RNG, a passed generator, or an
   objective-owned generator. Suggested interface addition:

```python
@dataclass(frozen=True)
class BackgroundSample:
    rgb: torch.Tensor | None
    mode: BackgroundMode
    sample_scope: Literal["step", "view", "frame", "pixel"]
    seed: int | None
    step: int | None
```

4. Chunked backward needs one additional invariant. Proposal A already says pass
   the same background sample into each chunk. It should also require loss
   normalization to remain per-frame/per-view equivalent after chunking:

```text
sum(chunk_loss * chunk_frame_count) / total_frame_count
```

or prove the existing `reconstruction_loss_per_image(...).mean()` semantics are
unchanged. Without that, a refactor can silently reweight temporal chunks.

5. The objective must not own W&B lifecycle, optimizer state, feature cache, or
   model decode. Proposal A says this, but the API should enforce it by returning
   plain artifacts instead of W&B objects:

```python
@dataclass(frozen=True)
class ValidationArtifactBundle:
    tensors: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]
```

Then the logger converts to W&B media. This keeps the objective testable.

Concrete change request for Proposal A before implementation:

- Add an "expected-fail route validation" phase before Phase 2.
- Add `objective.version` and `BackgroundSample.seed/step` to provenance.
- Make chunk loss normalization an acceptance criterion.
- State that artifact builders return tensors/metadata, not W&B media objects.

## Proposal B Review: ViewBatch + TrainRecipe Registry

Strong points:

- It makes `arch` real instead of treating it as a label.
- It models single-cam and multicam as the same `ViewBatch` shape rather than
  inherited trainer subclasses.
- It preserves camera roles: condition, anchor, train target, held-out target.
- It correctly makes held-out target the selector for multicam baselines.
- It introduces `FeatureProvider.describe()` before model construction, which is
  the right fix for precomputed V-JEPA/LTX/Wan feature-channel mutation.

Migration risks to tighten:

1. `train.py --explain-routing` must not bake features or instantiate heavy
   extractors. If explain mode calls `FeatureProvider.describe()` and that method
   bakes V-JEPA/LTX/Wan outputs, explain-all-96-configs becomes too expensive
   and risks mutating caches during a read-only audit.

Required split:

```python
class FeatureProvider(Protocol):
    def describe_config(self, spec: FeatureProviderSpec) -> FeatureDescription | None: ...
    def describe_data(self, bundle: DataBundle) -> FeatureDescription: ...
    def load(self, batch: ViewBatch) -> Mapping[str, torch.Tensor] | torch.Tensor: ...
```

`describe_config()` is cheap and explain-safe. `describe_data()` may inspect or
warm cache and belongs in run/smoke, not read-only routing.

2. Feature cache keys need a stronger minimum field list. The proposal asks
   whether `render_size` or `model.size` belong in the key. That should not stay
   open before implementation. For precomputed video features, require:

```text
extractor_name
extractor_version_or_model_id
feature_layer_names
source_path or content hash
native frame ids
frame_times if used
condition camera id/name
input resize/crop policy
feature dtype/layout
```

`render_size` belongs only if the extractor reads resized render tensors rather
than source/native tensors. The current notes say some feature-bake paths still
read resized training tensors, so the provider must log which source it used.

3. The registry should have two commands with different safety semantics:

```bash
src/train/train.py explain CONFIG
src/train/train.py run CONFIG
```

`explain` can be added immediately. `run` should remain opt-in until the recipe
has passed its route smoke. Old entrypoints should not delegate to `run_compat`
for a recipe until that recipe is marked green.

4. The proposal should make expected-fail states first-class in routing reports.
   Example:

```text
route_status: blocked
reason: feature_dim=32 requires shared feature_alpha objective for multicam
old_command: ...
next_required_phase: migrate_multicam_objective
```

This is safer than letting a route resolve and then relying on a later smoke to
catch the bad path.

5. The model-program boundary needs one additional check for multicam:

```text
decoded.cameras may be model-owned, but TargetView.cameras must control target
rendering for external train/heldout views.
```

This is easy to regress if a future `ModelProgram.decode()` returns cameras and
the objective blindly uses `decoded.cameras` instead of `target.cameras`.

Concrete change request for Proposal B before implementation:

- Split explain-safe provider description from run-time/cache-warming provider
  description.
- Close the feature-cache-key open question with a minimum key schema.
- Add blocked-route reports as a formal routing state.
- Add a target-camera-vs-decoded-camera invariant for multicam objective calls.

## Proposal C Review: Cleanup, Deletion, And Staged Migration

Strong points:

- It has the best deletion discipline of the three proposals.
- It correctly says `BASELINES.md` is append-only.
- It records the successful single-cam random-background W&B run:

```text
https://wandb.ai/nbardy/dynaworld/runs/9gr2dm3v
```

- It calls out that this run's random background is not config-proven because it
  is hardcoded in trainer code.
- It keeps gauge-field experiments isolated.
- It says not to delete `dynamicTokenGS.py` before extracting utility imports.

Migration risks to tighten:

1. Phase 5 extracts `FeatureProvider` after Phase 4 migrates multicam. That order
   is risky for the actual desired ultimate config, because the ultimate config
   is both multicam and precomputed V-JEPA. Move FeatureProvider extraction before
   the multicam F32 ultimate pass, or split Phase 4:

```text
Phase 4a: Multicam RGB/F3 through ViewBatch + objective.
Phase 4b: FeatureProvider extraction and V-JEPA describe/load.
Phase 4c: Multicam F32 V-JEPA ultimate through shared objective.
```

2. Phase 1 says missing background defaults should preserve old behavior, but
   current F32 random background exists only in code. If the normalizer defaults
   missing `losses.background` to white for all old configs, it will regress the
   already-fixed F32 config unless that exact config is migrated in the same
   change.

Required action:

```text
Enumerate all current F32 feature-splatting configs and explicitly set
losses.background.train_mode = random_rgb for the ones intended to inherit
the April 30 fix.
```

At minimum include:

```text
src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc
src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc
```

and audit the other F32 configs under `src/train_configs`.

3. The deletion table is good, but deletion needs a command-reference audit:

```bash
rg "train_camera_implict_dynamic|train_image_encoder_implicit_camera_baseline|train_ltx_feature_implicit_dynamic" .
rg "dynamicTokenGS|tokenGS|train_camera_implicit_dynamic" src train_scripts research_experiments
```

Only delete after every reference is redirected or deliberately archival.

4. The W&B provenance plan should include the old entrypoint and old config path
   for every migrated run, not just new recipe fields. Suggested minimum:

```python
{
    "Route/OldEntrypoint": "...",
    "Route/NewEntrypoint": "src/train/train.py",
    "Route/CompatShim": True,
    "Route/Arch": "...",
    "Route/Recipe": "...",
    "Objective/Version": "...",
    "LossBackground/TrainMode": "...",
    "LossBackground/EvalMode": "...",
    "FeatureSplatting/FeatureDim": 32,
    "FeatureSplatting/AlphaAvailable": True,
}
```

5. The baseline audit should distinguish smoke artifacts from standing metrics.
   Proposal C says do not update `BASELINES.md` from a smoke unless it is a smoke
   row. I would make this stricter:

```text
Smoke results go in migration notes.
BASELINES.md rows require an intentional benchmark run.
```

If a smoke row is absolutely useful, put it in a separate "Migration Smokes"
section, not the standings table.

Concrete change request for Proposal C before implementation:

- Reorder FeatureProvider before multicam F32 ultimate.
- Add an explicit list/audit command for every current F32 feature config.
- Add command-reference audits to deletion criteria.
- Require migrated run W&B fields for old and new entrypoints.
- Keep smoke results out of standings by default.

## Required Smoke Matrix Revisions

The proposals already include a strong smoke matrix. It should be split into
three layers so future agents cannot accidentally treat "expected fail" as
"not tested".

### Layer 0: Pre-Migration Freeze

Purpose:
    Capture current behavior and known broken routes before refactor.

Required checks:

| Route | Expected result | Why |
|---|---|---|
| F3 single-cam 1-step | pass | Preserve legacy RGB route. |
| F32 single-cam alpha/random-bg 1-step | pass | Preserve the fixed feature-splat path. |
| Known-camera 1-step with validation | may fail today if stale tuple path is hit | Confirms the bug the migration must fix. |
| Precomputed V-JEPA single-cam 1-step | pass or documented cache issue | Protects feature-cache path. |
| Multicam F32 ultimate 1-step | expected fail/block | Proves we are not silently using the broken multicam tuple/raw-feature path. |
| Explain all 96 configs | pass after explain command lands | Proves registry coverage without behavior change. |

This layer should be documented in the cleanup notes, not used as a baseline
claim.

### Layer 1: Per-Phase Runtime Smokes

Purpose:
    Prove each migration slice works before moving to the next.

Required sequence:

1. `train.py explain` across all configs.
2. Background config visibility on F32 single-cam.
3. Single-cam F3 and F32 through `RenderObjective`.
4. Known-camera initial and final validation through `RenderObjective`.
5. Precomputed V-JEPA through `FeatureProvider`.
6. Multicam RGB/F3 through `ViewBatch` and shared objective.
7. Multicam F32 V-JEPA ultimate through `ViewBatch`, `FeatureProvider`, and
   shared objective.
8. Script shims through router.
9. Legacy adapters and gauge external delegates.

No deletion before step 8 is green.

### Layer 2: Artifact Assertions

Exit code is not enough. Every smoke should write a small machine-readable or
grep-friendly result with:

```python
{
    "exit_code": 0,
    "step_count": 1,
    "train_loss_logged": True,
    "validation_payload_logged": True,
    "expected_media_logged": True,
    "route_recipe_logged": True,
    "background_mode_logged": True,
}
```

F32-specific:

```python
{
    "feature_dim": 32,
    "colorize_enabled": True,
    "alpha_available": True,
    "no_raw_feature_loss": True,
    "alpha_mask_video_logged": True,
    "feature_pca_video_logged": True,
    "composite_video_logged": True,
}
```

Multicam-specific:

```python
{
    "train_view_metrics_logged": True,
    "heldout_metrics_logged": True,
    "heldout_render_video_logged": True,
    "heldout_alpha_video_logged_if_f32": True,
    "condition_camera_logged": True,
    "anchor_camera_logged": True,
    "heldout_camera_names_logged": True,
}
```

Suggested artifact check:

```bash
find wandb -maxdepth 5 -type d -name videos -newer <run_start_marker>
```

Then check for expected filenames:

```text
Alpha_Mask_Video
Feature_PCA_Video
Render_Composite_Video
Heldout*_Alpha_Mask_Video
Heldout*_Feature_PCA_Video
Heldout*_Render_Composite_Video
```

## Deletion Safety Rules

The cleanup should treat deletion as a reward for proven routing, not as a way
to force discipline.

### Safe To Delete Only After Shim Period

Candidates:

- `src/train/train_camera_implict_dynamic.py`
- `src/train/train_image_encoder_implicit_camera_baseline.py`
- `src/train/train_ltx_feature_implicit_dynamic.py`

Required before delete:

1. `rg` proves every script/config/doc reference is redirected or knowingly
   archival.
2. Old command path runs and prints a deprecation message.
3. Router `explain` reports old and new route.
4. One compatibility release/commit exists with the shim still present.

### Do Not Delete During This Refactor

Do not delete:

- `src/train/dynamicTokenGS.py`
- `src/train/tokenGS.py`
- tiled legacy trainers
- gauge-field train stack

Conditions before later retirement:

- All active configs route through adapters or are explicitly marked archival.
- Shared utilities have been extracted out of legacy files.
- New trainers no longer import legacy utilities.
- At least one smoke or documented archival decision exists per config family.

### Gauge-Field Isolation

Gauge-field recipes should be visible to the central router but not absorbed
into the video-token `ViewBatch`/`RenderObjective` migration. The gauge stack has
its own representation and held-out-camera benchmarking context. For this
cleanup, the correct interface is:

```python
@dataclass(frozen=True)
class ExternalRecipe:
    name: str
    explain: Callable[[ExperimentSpec], RoutingReport]
    run_legacy: Callable[[Path], int]
```

No gauge-field internals should be normalized by this pass.

## Experiment Provenance Requirements

The current single-cam F32 random-background run is important but imperfect
provenance:

```text
Run: 9gr2dm3v
URL: https://wandb.ai/nbardy/dynaworld/runs/9gr2dm3v
Config: src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc
Final: Loss 0.0665 / recon 0.0660
Caveat: random background was hardcoded in trainer code, not represented in config.
```

That caveat must travel into any migration notes. Do not use the W&B config
alone to prove random background was enabled for that run.

Future W&B runs must log:

```python
{
    "Route/Arch": spec.arch,
    "Route/Recipe": recipe.name,
    "Route/OldEntrypoint": old_entrypoint,
    "Route/NewEntrypoint": "src/train/train.py",
    "Objective/Version": objective.version,
    "Model/FeatureDim": feature_dim,
    "Renderer/Mode": renderer_mode,
    "LossBackground/TrainMode": background.train_mode,
    "LossBackground/EvalMode": background.eval_mode,
    "LossBackground/SampleScope": background.sample_scope,
    "LossBackground/SeedPolicy": background.seed_policy,
}
```

For F32 feature splatting:

```python
{
    "FeatureSplatting/ColorizeEnabled": True,
    "FeatureSplatting/ColorizeClass": colorize.__class__.__name__,
    "FeatureSplatting/ColorizePreNorm": colorize_cfg["pre_norm"],
    "FeatureSplatting/WeightInit": colorize_cfg["weight_init"],
    "FeatureSplatting/WeightInitGain": colorize_cfg["weight_init_gain"],
    "FeatureSplatting/AlphaAvailable": alpha is not None,
}
```

For multicam:

```python
{
    "Multicam/ConditionCamera": condition_camera_name,
    "Multicam/AnchorCamera": anchor_camera_name,
    "Multicam/TrainCameras": train_camera_names,
    "Multicam/HeldoutCameras": heldout_camera_names,
    "Multicam/PrimaryHeldoutMetric": "heldout/l1" or "heldout/psnr",
}
```

Baseline rules:

1. `BASELINES.md` is append-only.
2. Do not overwrite old run rows after migration.
3. Do not add "ultimate multicam F32" as complete until the migrated route runs.
4. Do not rank multicam baselines by train/source-view loss.
5. Any row claiming novel-view value needs held-out camera metrics and held-out
   media.

## Implementation Gate Checklist

Before code implementation begins, amend the plans or create an implementation
task list with these gates:

- [ ] `train.py explain CONFIG` exists and is read-only.
- [ ] All 96 configs explain to active, compat, legacy, or external route.
- [ ] F32 configs intended to use random background explicitly set normalized
      `losses.background`.
- [ ] W&B logging plan includes old/new route, objective version, feature dim,
      and background policy.
- [ ] Multicam F32 route is blocked until shared objective support is enabled.
- [ ] Known-camera stale tuple path has a dedicated smoke.
- [ ] `FeatureProvider.describe_config()` is explain-safe and does not bake.
- [ ] `FeatureProvider.describe_data()` or equivalent runs before model
      construction for precomputed routes.
- [ ] Feature cache keys include source, frame ids, camera role, extractor
      identity, and resize/crop policy.
- [ ] Gauge-field routes are external delegates.
- [ ] `dynamicTokenGS.py` utility imports are extracted before any retirement.
- [ ] Duplicate implicit-camera shims are converted to deprecation shims before
      deletion.
- [ ] Smoke artifacts are checked for expected videos, not just exit codes.
- [ ] `BASELINES.md` update rules are explicit and append-only.

## Final Recommendation

Proceed with the architecture, but do not start with a broad router rewrite.
Start with the boring safety layer:

1. `train.py explain` plus route status.
2. normalized background config and W&B provenance.
3. route guard that blocks unsupported F32 multicam.
4. single shared `RenderObjective` extraction for single-cam and known-camera.

Then migrate multicam through `ViewBatch` and the shared objective, with
`FeatureProvider` in place before the V-JEPA ultimate run. Only after those
smokes pass should scripts redirect. Only after redirected scripts pass should
true duplicate shims be deleted.

This sequencing preserves the active baselines, makes the random-background fix
auditable, and prevents the exact bug pattern that triggered the cleanup:
feature splatting fixed in one trainer while multicam and validation quietly use
a stale local render/loss path.
