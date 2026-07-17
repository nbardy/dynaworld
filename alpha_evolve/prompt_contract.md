# Prompt Contract

The local evolver should generate strict prompts for `codex exec`. The prompt
must make the problem, allowed files, evaluator, and final response schema
unambiguous.

## Prompt Skeleton

```text
You are editing a disposable candidate worktree for DynaWorld.

Problem:
  <one paragraph>

Current best:
  <metrics and path>

Known negative results:
  <bullets>

Allowed write paths:
  <paths>

Forbidden:
  <bullets>

Evaluator:
  Stage 0:
    <commands>
  Stage 1:
    <commands>
  Stage 2:
    <commands>

Task:
  Make one coherent implementation change that improves <primary metric>
  without violating the hard gates. Keep the diff small. Do not edit evaluator
  scripts unless explicitly listed in allowed paths.

Final response:
  - changed files
  - implementation summary
  - tests run
  - any expected evaluator risks
```

## Response Handling

Codex may edit files directly. The runner should ignore prose unless a patch
exists. After the run:

```bash
git diff --binary > "$CANDIDATE/patch.diff"
git diff --name-only > "$CANDIDATE/changed_files.txt"
```

If no patch exists, score as a rejected no-op even if the final message sounds
plausible.

## Prompt Inputs From Program Database

Include at most:

- 1 current elite for the same island
- 1 global elite
- 1 diverse near miss
- 1 failure with a clear lesson

Too much history makes Codex average over old ideas. The prompt should carry
the current target and the sharpest constraints, not the whole repo memory.

## Strict Non-Goals

Always spell out non-goals:

- Do not disable W&B for benchmark candidates.
- Do not change target frames, target size, tube count, or loss kind unless the
  microlib explicitly allows it.
- Do not edit `BASELINES.md` from a candidate worktree.
- Do not mutate user dirty worktree state.
- Do not add a broad framework when the microlib needs one helper.

## Example: STAR Feature Prompt Core

```text
Improve STAR UVT F32 feature backward.

Allowed paths:
  research_experiments/star_uvt_feature_tubes/
  src/train/train_star_uvt_feature_overfit.py
  third_party/fast-mac-gsplat/variants/star_uvt_v0/

Primary metric:
  Lower 64f/256px/32768t/F32 backward_ms than feature_direct_gradcache with
  zero overflow and parity within the current tolerances.

Hard rejects:
  - skip feature gradients
  - fixed first-three sigmoid only
  - no colorizer parameter gradients
  - hidden target/loss/frame-count changes
```

## Example: Mixed Scheduler Prompt Core

```text
Implement the smallest mixed same-view + heldout scheduler bridge.

Allowed paths:
  src/train/sequence_data.py
  src/train/multicam_video_data.py
  src/train/train_video_token_implicit_dynamic.py
  src/train_configs/<new smoke config>.jsonc
  tests/test_<focused>.py

Primary metric:
  One-step smoke exercises both same_view_recon and heldout_view_recon and logs
  both separately.

Hard rejects:
  - third manifest format
  - a large BaseTrainer abstraction
  - exact target observation visible to encoder
```
