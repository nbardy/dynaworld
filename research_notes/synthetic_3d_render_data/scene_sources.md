# Scene Sources — Where Rich Animated 3D Content Actually Lives

Tiered by quality-per-effort for our v2 Blender pipeline. "Quality" here means
*material richness, motion realism, lighting variety* — the things Kubric's
floating-cube aesthetic gets wrong.

## Tier 1: Blender Foundation open movies (best fit)

Free CC-BY, fully rigged, fully animated, fully Blender-native. Drop straight
into `data/v2/bullet-time/assets/` and the existing pipeline opens them.

| Title | Year | What's in it | Stub in v2? |
|---|---|---|---|
| Spring | 2019 | character + creature + dramatic terrain | ✅ `spring.zip` present, **unextracted** |
| Agent 327 | 2017 | hero character, detailed barbershop interior | ✅ `agent327/` empty |
| Sintel | 2010 | rigged character + dragon, research-favorite | ✅ `sintel/` empty |
| Cosmos Laundromat | 2015 | landscape + hero animation | ✅ `cosmos/` empty |
| Big Buck Bunny | 2008 | 4 rigged characters, comedy | ✅ `bbb/` empty |
| Sprite Fright | 2021 | forest, 5 rigged kids + monster | ❌ not staged |
| Charge | 2022 | open-world demo, vehicles, fire VFX | ❌ not staged |
| Coffee Run / Hero / Settlers | various | smaller, fast load | ❌ not staged |

Source: <https://studio.blender.org/projects/>

**Why this tier wins**: these are the only free assets where you get a full
shot — character + environment + lighting + camera + rigged animation —
designed to be opened and rendered. Nothing else (CC0 or paid) bundles all of
those at production quality.

**Limitation**: stylized, not photoreal. If the goal is "model needs to learn
real-world video distribution," these are the wrong source. If the goal is
"diverse, dynamic, rich-material clips for camera-consistency probes,"
they're ideal.

## Tier 2: Unreal free AAA (richest data, but new pipeline)

Only relevant if a UE pipeline is started. Caveat: exporting UE assets to
Blender is genuinely painful — most attempts hit material/shader translation
issues.

- **City Sample (Matrix Awakens)** — 500 GB, LA-scale city + MetaHumans + AI traffic.
  Single richest free 3D dataset that exists.
- **Paragon characters** — ~20 AAA MOBA heroes, rigged, free, originally ~$15k of content.
- **MetaHumans** — photoreal humans, face rigs, procedurally generated.
- **Quixel Megascans** — 16,000+ scanned real-world assets, free for UE use.

All consolidated on **Fab** (fab.com).

**Verdict**: only justifies the pipeline cost if photoreal humans /
city-scale crowds are a hard requirement that nothing else covers.

## Tier 3: Unity sample projects (middle ground)

Documented in `~/git/nova/data/v3/docs/sample_projects.md`. Setup status:
**none installed**, Unity itself is missing.

| Project | Size | Render Pipeline | Risk | Why it's interesting |
|---|---|---|---|---|
| Boss Room | 8 GB | URP | low | 4 rigged heroes, combat, dungeon |
| Megacity Metro | 15 GB | URP + DOTS | high | 4 km² city, 150+ AI agents — but DOTS breaks |
| The Heretic | 5 GB | HDRP | medium | digital humans, facial rigs |
| Boat Attack | 6 GB | URP | low | water/vehicles, outdoor |
| FPS Sample | 18 GB | HDRP | high | mocap data, but Unity 2018→2022 upgrade required |

**Verdict**: Boss Room is the safe entry. Megacity is the prize but
historically the most fragile. Skip FPS Sample unless someone has time to
debug a multi-year Unity version migration.

## Tier 4: Compose-your-own sources

When the goal is *variety* — thousands of unique scenes, not ten polished
ones:

- **Polyhaven** (polyhaven.com) — CC0 HDRIs, PBR materials, models. Highest
  free quality. **Best HDRI source by far.**
- **Mixamo** (adobe.com/mixamo) — 2,500+ free character animations with
  auto-rigging. **Best path to action / sports motion** if CMU MoCap stays
  dead.
- **BlenderKit / Blend Swap** — community Blender, mixed quality.
- **Sketchfab** (CC-filtered) — huge variety, quality is a coin flip.
- **Kenney.nl** — CC0, low-poly, vast volume. Useful for domain randomization
  (think: backgrounds, props, environment fillers).

**The composition workflow**: Polyhaven HDRI + Mixamo character + Mixamo
animation + Polyhaven prop = unique synthetic clip in ~10 minutes per scene.
This is what Kubric should have been but wasn't.

## Tier 5: Built-for-ML synthetic frameworks

These are tools, not asset libraries — they consume Tier 1–4 assets and emit
training data at scale. Covered in detail in `synthetic_frameworks.md`.

- **BlenderProc** (DLR) — scriptable on top of any `.blend` file.
- **InfiniGen** (Princeton) — procedural Blender natural-world scenes.
- **Kubric** (Google) — too synthetic. Avoid.
- **Hypersim** (Apple) — already-rendered 77K photoreal indoor frames.

## MultiCamVideo Dataset (KwaiVGI)

- **Source**: <https://huggingface.co/datasets/KwaiVGI/MultiCamVideo-Dataset>
- **What**: synchronized multi-camera videos rendered in **Unreal Engine 5**.
- **Why it fits here**: synthetic-rendered multi-camera training set. Directly
  relevant as a synthetic supplement to AIST / DeepView / Neural3D / ViVo.
- **Status**: not yet locally intaken.

## Practical recommendations

**For DynaWorld novel-camera probes** (small set, high quality):
1. Extract `spring.zip` (already in v2/assets).
2. Add 2–3 more Blender Foundation movies (Sprite Fright, Charge, Cosmos).
3. Render multi-camera rigs with v2's existing pipeline.
4. Total effort: <1 day. Total scenes: ~5 production-quality.

**For domain-randomized augmentation** (large set, varied):
1. BlenderProc on top of Tier 1 scenes.
2. Polyhaven HDRIs for lighting variation.
3. Mixamo characters/animations for motion variation.
4. Effort: ~1 week to wire up. Output: thousands of clips.

**For human-motion specifically** (not "scene render," but related):
- See `human_motion_and_action_datasets.md`. BEDLAM is usually a better
  starting point than re-rendering humans yourself.
