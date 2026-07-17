# Microlib: Code Organization Behavior Helpers

## Problem

Reduce behavior forks around RGB composition, validation media, metrics, and log
cadence without building a large trainer framework.

## Why Now

`CODE_ORGANIZATION.md` names shared RGB composition and validation media as
high-priority cleanup. This is a good low-cost evolver target because tests can
protect behavior and the score can include LOC reduction plus smoke success.

## Allowed Edits

Likely surface:

- `src/train/pipeline/`
- `src/train/train_logging.py`
- trainer call sites that already use the pipeline helpers
- focused tests under `tests/`

Avoid broad trainer inheritance or PowerFoam/WorldFoam unification.

## Baseline

Current desired helper shape:

```text
compose_rendered_rgb(features, alpha, colorize, cameras, *, bg)
build_validation_media_bundle(gt, rendered, alpha=None, features=None, cameras=None)
```

The behavior contract matters more than line count.

## Evaluator Cascade

Stage 0:

- changed files are scoped
- no broad framework
- no semantic changes to config defaults

Stage 1:

- focused tests for RGB composition with F=3 and F32 feature colorization
- media builder tests include optional alpha/features

Stage 2:

- one canonical trainer smoke if call signatures changed
- no validation-media name regression

## Primary Metrics

- tests pass
- behavior branches reduced
- changed LOC modest
- same output keys/media names

## Hard Rejects

- Reshuffling lines without behavior coverage.
- Hiding same-view vs heldout semantics.
- Changing alpha/background composition.
- Adding a base trainer hierarchy.

## Promotion Gate

Promote only if the diff makes future behavior harder to fork, not merely
shorter.
