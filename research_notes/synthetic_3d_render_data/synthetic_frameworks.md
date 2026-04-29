# Synthetic Data Frameworks

Tools, not asset libraries. These produce training data at scale by consuming
3D assets and emitting rendered clips with ground truth.

## Kubric (Google) — too synthetic, avoid

**What it is**: Blender-based synthetic data framework. Scripted scene
generation, physics, domain randomization, ground-truth flow / depth /
segmentation / 3D bounding boxes.

**The aesthetic problem**: Kubric outputs are famous for *looking* synthetic.
Floating geometric primitives on flat-textured ground planes, simple
materials, sterile lighting. It's a 2015-physics-demo look.

**Why it has that aesthetic**: Kubric ships with a small set of CC-licensed
shapes (cubes, cylinders, KLEVR primitives), and most published Kubric
benchmarks (MOVi-A through MOVi-F) deliberately use simplified scenes for
benchmark consistency. The framework *can* load complex `.blend` files but
the published examples don't, and the community converged on the simple
look.

**When it's fine**: optical flow benchmarks, segmentation pretraining,
synthetic data for pure structural tasks where photorealism doesn't matter.

**When it's wrong**: anything where the target distribution is real-world
video. A model that learned "the world looks like Kubric" will not generalize
to action-cam footage of skateboarding.

**Verdict for DynaWorld and Nova**: don't use as-is. The aesthetic gap is
exactly the kind of distribution shift that breaks `video <=> video` training.

## BlenderProc (DLR) — Kubric's better cousin

**What it is**: Same idea as Kubric — scriptable Blender pipeline with ground
truth — but **designed to load arbitrary `.blend` files** as scene roots.

**Why it's better**: when you point BlenderProc at a Blender Foundation movie
(or a Polyhaven scene, or a custom asset), the rendered output inherits the
*real* materials, lighting, and animation of the source. You get Kubric's
automation without Kubric's aesthetic.

**Capabilities**:
- Ground-truth depth, normals, instance segmentation, semantic segmentation,
  optical flow, 3D bounding boxes, camera intrinsics/extrinsics.
- Domain randomization: lights, materials, camera poses, object positions.
- Cycles or Eevee backend.
- Replicable Python API; configs are runnable scripts.

**Typical use**: load a `.blend` of a real scene, randomize lighting + camera
across N renders, emit clips with GT.

**Recommended path** for DynaWorld synthetic supplements:
1. Use v2's Blender Foundation scenes as input.
2. Wrap with BlenderProc for randomization + GT export.
3. Result: thousands of clips per scene, photoreal materials, ground-truth
   camera path.

**Effort estimate**: ~1 week to integrate with existing v2 pipeline.

## InfiniGen (Princeton) — procedural natural-world

**What it is**: Procedural Blender pipeline that *generates entire scenes
from scratch* — terrain, plants, rocks, water, animals — using rule-based
geometry. No asset library needed.

**Variants**: InfiniGen (outdoor / natural) and InfiniGen Indoors.

**Strength**: infinite unique scenes, no asset licensing concerns, full GT.
Material quality is genuinely good for the "natural world" domain.

**Limitation**: it generates *natural* scenes. Forests, deserts, oceans,
caves, indoor rooms. It does NOT generate humans, vehicles, sports
environments, urban action scenes. If your domain is action-cam footage of
skateboarding, InfiniGen is the wrong source.

**Verdict for DynaWorld**: useful for nature-domain augmentation. Wrong tool
if the target is human action.

## Hypersim (Apple) — already-rendered, no pipeline needed

**What it is**: 77,000 photorealistic indoor frames, V-Ray-rendered, with
per-pixel ground truth (depth, normals, instance/semantic segmentation,
material). Just a downloadable dataset, not a generator.

**Strength**: unmatched material quality among free synthetic indoor
datasets. The V-Ray render quality is genuinely close to architectural
visualization standards.

**Limitation**: indoor only, sterile interior architectural scenes. No people,
no motion (these are stills, not video). Sterile-architectural-render look.

**Verdict**: useful as a static-scene reference distribution. Not a video
source.

## ThreeDWorld / TDW — physics-rich Unity

**What it is**: Unity-based, focused on physical simulation (cloth, fluids,
multi-agent). Used in cognitive-science / embodied-AI research.

**Verdict**: niche. Skip unless physics interaction is the explicit goal.

## Decision matrix

| Tool | Scene-domain fit | Aesthetic | Effort | Verdict for DynaWorld |
|---|---|---|---|---|
| Kubric | weak (sterile primitives) | bad | low | avoid |
| BlenderProc | strong (any .blend) | good (inherits source) | medium | **recommended** |
| InfiniGen | strong for nature, none for human | good | medium | nature-domain only |
| Hypersim | strong for static indoor | excellent | none (just download) | reference only, not video |
| TDW | strong for physics, weak otherwise | medium | high | skip |

## Recommended stack for DynaWorld synthetic probes

**For small-set / high-quality probes** (camera-leakage tests, novel-view
unit tests):
- v2 Blender pipeline + Blender Foundation scenes. No framework needed.

**For large-set / domain-randomized augmentation**:
- BlenderProc on top of v2's Blender Foundation scenes.
- Polyhaven HDRIs for lighting variation.
- Mixamo characters/animations for motion variation.

**Avoid**:
- Kubric primitives (aesthetic mismatch).
- Re-rendering Hypersim-style indoor stills (no motion).
- Building a new framework from scratch (BlenderProc already exists).

## The Kubric trap

A common mistake is to reach for Kubric because "it's the standard synthetic
data tool." It became standard for benchmarks where the aesthetic doesn't
matter (optical flow), and the community has not corrected the perception
that it's the right tool for general-purpose video data.

For a project where the model needs to generalize to real-world video,
Kubric's aesthetic *is* the failure mode. Don't fall into it.
