# Blender Sintel synthetic render pipeline — chronological session notes

Date: 2026-04-28. Goal: build a synthetic-data ingest source for the dynaworld
trainer that renders Blender Open Movies (Sintel) from novel camera trajectories,
with `c2w` + `K` JSON sidecars in OpenCV convention.

## Why Sintel

Earlier sessions narrowed the open-movie pick to "fully CG, has people, free
to obtain, modern enough":

- Spring, Sprite Fright, Charge, Cosmos Laundromat — modern, photoreal-ish, but
  Blender Studio-subscription gated for production .blend files.
- Tears of Steel — free production .blends, but actors are live-action plates;
  novel-camera rendering only works on the CG VFX, not the people.
- Big Buck Bunny / Caminandes / Agent 327 — cartoon shading, useful only as
  diagnostic probes.
- Hero / Coffee Run / Wing It / Project Gold — 2D grease-pencil / NPR. Useless
  for a real-video model.
- **Sintel** — fully CG, has people (Sintel + Shaman + dragon), free
  production tree on archive.org as the original 4-DVD set. Picked it.

## Asset acquisition

- Original Durian SVN at `download.blender.org/durian/svn/` is permanently 404.
- Public mirror is the **archive.org `sintel-dvd` torrent** (29.4 GB, perpetual
  webseed, CC-BY 3.0). Direct curl works for individual ISOs.
- `Sintel_DATA.iso` (8 GB) contains most of the production tree at `pro/`.
- `Sintel_PAL.iso` (8 GB) contains the rest of the shot files plus the
  rendered film as DVD VIDEO_TS (PAL 720×576) — useful as the SSIM reference.
- The DVD set's `pro/scenes/` covers ~27 scenes total. **The opening fight
  (`01_snowbandits`) only ships with layout-stage gray-box blends**; its
  lighting was never published. `02_shaman` has full per-shot lighting blends
  for shots a–k.

## Modern Blender (5.1) cannot render Sintel

Both EEVEE and Cycles produced pink-blob garbage. Sintel's materials were
authored for Blender Internal renderer (deprecated 2.79b → 2.80). Modern
Blender's auto-conversion paths on file-open replace unsupported BI shader
nodes with default Principled BSDF approximations — characters render as
flat pink, environments lose all texture detail. The depsgraph also reports
"7 dependency cycles" on materials like `MAwood`, `MALT_StaffGrip` that
indicate broken auto-converted shader graphs. Linking the production lights
into the modern scene fixed the dark/silhouetted output but didn't fix the
clay-pink material problem.

## Pivot: Blender 2.79b

Last version of Blender with Blender Internal renderer. Downloaded the
official x86_64 build (`download.blender.org/release/Blender2.79/`,
~141 MB), runs on Apple Silicon via Rosetta 2 with ~30% overhead. Required
`xattr -dr com.apple.quarantine` on the .app to bypass macOS Sequoia
gatekeeper for the unsigned 2018 binary.

Blender 2.79b ships Python 3.5 (no `from __future__ import annotations`,
no parameterized type aliases, no `@dataclass`). Made `camera_export.py`
3.5-compatible so it loads in both 2.79b and modern Python.

## Headless render quirks discovered the hard way

In rough order of pain:

1. **Image format enum locked to FFMPEG.** Sintel-era .blend files have
   `scene.render.image_settings.file_format = 'FFMPEG'` (or AVI_JPEG in
   2.79b) and the runtime enum rejects all other values, even though
   `bl_rna.properties["file_format"].enum_items` reports all 16 formats.
   Direct assignment `= 'PNG'` raises `enum "PNG" not found in ('FFMPEG')`.
   Workaround: render to memory, then save via the Image object's own
   `file_format` (which isn't scene-locked):
   ```python
   bpy.ops.render.render(write_still=False)
   src = bpy.data.images["Render Result"]
   img = src.copy()              # second quirk: render result is invalidated after first save
   img.file_format = "PNG"
   img.filepath_raw = path
   img.save()
   bpy.data.images.remove(img)
   ```

2. **scene.layers ships with only L01 enabled.** The camera (`02_A`) and 47
   of the scene's main objects live on L09. `blender -b file.blend -a` would
   render basically empty content. The production team must have had a
   render-farm prelude. Set `scene.layers = [True]*20` headlessly.

3. **resolution_percentage ships at 25 (production preview).** Multiplied
   downscale by 4× silently. Force to 100 to get the resolution you asked
   for.

4. **Timeline-marker camera binding.** The .blend has `timeline_markers`
   that bind `marker.camera` at specific frames; on `frame_set(f)`, the
   active camera is rewritten from the marker. So setting `scene.camera = obj`
   in a script gets overridden at render time. Workaround: iterate
   `scene.timeline_markers` and set `m.camera = None` for all, then assign.

5. **Multi-camera-per-script bug in 2.79b.** Even after clearing markers
   and setting `scene.camera` per iteration, all 12 renders in the loop
   produced byte-identical output (12 of 12 same hash). The fix that
   actually worked was bash-looping separate Blender invocations:
   ~30 sec setup × 12 cams = 6 min total but reliable.

6. **Lights live in linked-library group instances.** `bpy.data.lamps`
   only exposes the 6 Lamp datablocks loaded directly at file-open time;
   the bulk of the production lighting (38 lamps from
   `env_snow/shamanhut.blend`) come in via group instances and aren't in
   `bpy.data.lamps`. Scaling lamp energy to dim the scene only catches the
   6 visible ones — the dominant lights stay at full intensity.

7. **Raytraced AO is the perf cliff.** Across 3.3M verts × 4.9M faces × 17
   lamps, "Occlusion preprocessing" alone takes 15+ minutes per render
   without finishing on this Mac (Apple Silicon Rosetta running x86_64
   Blender Internal). Disabled via:
   ```python
   scene.render.use_raytrace = False
   world.light_settings.use_ambient_occlusion = False
   world.light_settings.use_indirect_light = False
   for rl in scene.render.layers:
       if rl.name != "1 RenderLayer":
           rl.use = False
   ```
   Cuts render time from "never finishes" to ~30 sec/frame at downscale 4.
   Trade-off: severe overexposure (no AO/indirect light to soften the
   direct lamp contribution).

## Per-shot .blend files: each one renders ONE shot

The 02_shaman scene ships 11 per-shot .blend files (`02.a` through `02.k`)
plus 11 corresponding `_comp.blend` compositing files. Each shot file has
its OWN scene state — its own frame range, its own active camera, its own
intended composition. **Don't use `02.a.blend` to render shot 02_G's
camera at some arbitrary frame.** Open `02.g.blend` instead. The frame
numbering in each shot file is local; `02.a` covers frames 10-157, `02.g`
covers frames 1-112, and they're describing different ~5-second slices of
the Shaman scene's edit timeline.

This is what initially produced the wrong-looking renders: we'd been
opening `02.a.blend`, switching to camera 02_G via Python, and rendering
at frame 80 — which gave us shot A's animation state through camera G's
(possibly default-position) viewpoint. Garbage in, garbage out.

## Reverse-engineering the production timeline

`02.a.blend` has 23 timeline markers binding cameras to frames. Decoded:

| Marker frame | Camera | Lens |
|---|---|---|
| 0 → 314 | 02_A | 25mm — overhead-on-table close-up |
| 314 → 367 | 02_C_E | 28mm |
| 367 → 412 | 02_D | 35mm — wall ornament insert |
| 412 → 485 | 02_C_E | 28mm |
| 485 → 560 | 02_F | 50mm — vertical wall strips |
| 560 → 626 | 02_G | 18mm — **wide establishing shot of hut** |
| 626 → 672 | 02_E (data viewer.004) | 50mm |
| 672 → 744 | 02_H_K | 50mm |
| 744 → 841 | 02_I | 35mm — Sintel hair close-up |
| 841 → 905 | 02_J_L | 50mm |
| 905 → 1127 | 02_H_K | 50mm |
| 1127+ | 02_J_L | 50mm |

Total ~47 sec of edit timeline distributed across 11 per-shot .blend files.

## Film matching + SSIM

Reference frames extracted from `/Volumes/Sintel_PAL/VIDEO_TS/VTS_01_1.VOB`
(the main feature, 15.7 min, MJPEG / 720×576).

The Shaman scene starts in the film around t=120s. Wide establishing
shot (matching 02_G) runs film t=120-126s. SSIM helper written in pure
numpy + cv2 (`src/dataset_pipeline/blender_synthetic/ssim.py`).

Best result so far: `02.g.blend` at frame 50, downscale 4, lamps 0.4×,
AO/raytrace disabled → SSIM 0.25 luma vs film t=126s. Composition is
correct (Sintel left, Shaman right, eye-level wide of round hut) but
SSIM is gated by overexposure — the central area where the fire glows
in the film is white-blown-out in our render because the central fire
is a particle/procedural effect we haven't enabled, and direct lamps
dominate without AO/indirect to soften them.

## State of code

- `src/dataset_pipeline/blender_synthetic/camera_export.py` — Blender → OpenCV
  c2w + pixel-K math. Pure numpy, no deps. Python 3.5/3.10+ compatible.
- `src/dataset_pipeline/blender_synthetic/render_scene.py` — modern Blender
  (5.x) entry point.
- `src/dataset_pipeline/blender_synthetic/render_scene_279.py` — Blender 2.79b
  entry point.
- `src/dataset_pipeline/blender_synthetic/verify_projection.py` — 4 sanity
  checks for c2w/K math, all passing to machine precision.
- `src/dataset_pipeline/blender_synthetic/ssim.py` — Wang/Bovik SSIM in pure
  numpy + cv2 (11×11 Gaussian, σ=1.5, K1=0.01, K2=0.03).
- `/tmp/render_native.py` — minimal "let Blender's own settings render"
  variant for shot-faithful output.
- `/tmp/render_one_cam.py` — single-camera-with-marker-clear variant for
  contact-sheet workflows (bash-loop one Blender invocation per camera).

## What's NOT done

- **`trajectories.py`** — programmatic novel-view cameras. This was the
  whole point of the synthetic pipeline and we haven't started it.
- `ingest.py` + JSONC config + shell driver matching `youtube_ingest.py`'s
  style.
- Exposure fix (next). Two paths: aggressive lamp scaling (~5-10 min
  iteration; likely SSIM 0.4-0.5) or re-enable AO + indirect light at
  low samples (~30-60 min per frame; likely SSIM 0.55-0.7).
- Hunting the lamps inside linked group instances (the dominant light
  source not currently affected by `LAMP_SCALE`).

## Decisions still open

- Blender 2.79b for the entire pipeline vs 2.79b only for Sintel-style
  Internal-renderer assets. Modern productions (Spring, Sprite Fright,
  Charge) use Cycles natively — they'd render correctly in modern Blender.
- Whether to commit to one full cinematic-quality 148-frame render of
  shot a (would take 1.5-2 hours per shot at 30-60 sec/frame including
  AO).
- Whether to use the PAL DVD's 720×576 reference vs downloading the
  1080p MKV from `download.blender.org/durian/movies/` for visual
  side-by-side (SSIM normalizes resolution either way).
