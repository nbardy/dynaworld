# Synthetic 3D Render Data — Research Notes

Captured 2026-04-25. Knowledge transfer from the Nova iOS-app side
(`~/git/nova/data/`) into the DynaWorld research context.

## Why this folder exists

DynaWorld's core training contract (per `DATASET_V1.md` and the postulates in
the top-level `README.md`) is:

> `Video <=> Video` is the only training data for world models that scales.
> No fake 3D labels. No synthetic ground truth.

This folder does **not** propose violating that contract. Instead it captures
where synthetic 3D rendering is genuinely useful in a world-model project that
otherwise rejects synthetic GT:

1. As a **probe / unit test** for novel-camera consistency (you can hold scene
   state fixed and vary the camera, which monocular video can't do).
2. As **structured pretraining pressure** before paired-camera finetune.
3. As a **camera-leakage stress test** — render the same scene from two cams,
   feed cam A's video tokens with cam B's camera token, see if the model can
   actually swap.
4. As **Nova app feature work** (deblur / stabilization / bullet-time training
   set generation) — adjacent project, same engines.

Where synthetic gets *banned* is as the primary `video <=> video` training
signal — we don't want the model learning the V-Ray / Cycles / Unreal style as
its idea of "real."

## Files in this folder

- **`README.md`** — this file. Orientation.
- **`pipelines_we_have.md`** — concrete state of the three Blender/Unity pipelines
  we own in `~/git/nova/data/v1/`, `v2/`, `v3/`. What works, what's blocked,
  what's a stub. Use this before proposing to *build* a synthetic pipeline —
  there are already two and a half.
- **`scene_sources.md`** — tiered catalog of where rich, animated, ready-to-render
  3D content actually lives (Blender Foundation, Unreal Fab, Unity samples,
  Polyhaven, Mixamo). Ranked by quality-per-effort for a Blender-based pipeline.
- **`synthetic_frameworks.md`** — Kubric vs BlenderProc vs InfiniGen vs
  Hypersim. The "Kubric is too synthetic" critique and why BlenderProc on top
  of real Blender Foundation scenes is the better path.
- **`human_motion_and_action_datasets.md`** — BEDLAM, AGORA, AIST++, Hi4D,
  SURREAL, plus sports-specific (SkatingVerse, FineGym, SportsMoT). Mostly
  rendered or real-captured; not the same as a "scene render" but they're the
  realistic alternative.
- **`dynaworld_relevance.md`** — how synthetic data could fit (or shouldn't)
  inside the DynaWorld training contract specifically. The four legitimate
  uses, and the things that would violate the contract.
- **`open_questions.md`** — what we don't know yet.

## How to use

- **Considering a new synthetic data effort?** Read `pipelines_we_have.md` first
  to avoid duplicating v2 Blender. Then `synthetic_frameworks.md` to pick
  Kubric vs BlenderProc vs InfiniGen.
- **Need rich animated content?** `scene_sources.md` ranks options for v2
  Blender specifically.
- **Worried this conflicts with `DATASET_V1.md`?** Read `dynaworld_relevance.md`
  — it spells out where synthetic fits and where it doesn't.
- **Adding a new finding?** Update the relevant file. If it's tactical, drop
  into `../../agent_notes/loose_notes/`. If it changes the strategic picture,
  update the README here.

## Cross-references

- **Nova catalog (master, more recent)**: `~/git/nova/data/CATALOG.md` — broader
  catalog including the iOS pipeline's perspective.
- **DynaWorld dataset contract**: `../../DATASET_V1.md` — the rule about no
  synthetic GT lives there.
- **Multi-camera GT track**: `../../DATASET_V1.md` §AIST/DeepView/Neural3D/ViVo
  — the *real* GT path for novel-view validation. Synthetic complements; it
  doesn't replace.

## Philosophical tension (read this before adding anything)

The strongest argument *against* synthetic: it lets the model cheat. A
synthetic clip has perfect calibration, perfect motion, no rolling shutter, no
real-world lens noise. If we train on it as if it were `video <=> video`
ground truth, we teach the model to expect those properties.

The strongest argument *for* synthetic: it's the only way to hold a scene
fixed and vary the camera. Real video can't do that. So synthetic's role is
**diagnostic and architectural**, not training-signal.

When in doubt, ask: "would this synthetic clip be a legitimate `video <=>
video` ground truth?" If yes, you're probably violating the contract. If no
(it's a probe, an augmentation, or a bullet-time render), it's fine.
