# Dataset Rehydrate Runbook

This repo is set up so local raw data can be partial. The checked-in source of
truth is the config/script layer; generated raw media, extracted media,
inventories, previews, and clip sets stay ignored.

Use this status command before or after deleting local data:

```bash
./src/dataset_scripts/local_data_status.sh
```

## Default Local Smoke Set

Tiny local training/test baseline, scene-distinct but not GT novel-view:

```bash
./src/dataset_scripts/youtube_scene_distinct_30_seed.sh
./src/dataset_scripts/youtube_scene_distinct_30_256_seed.sh build-clips
```

Source index:

- `src/dataset_configs/youtube_scene_distinct_30_64_4fps_16f.jsonc`
- `src/dataset_configs/youtube_scene_distinct_30_256_4fps_16f.jsonc`
- `data/youtube_scene_distinct/candidates/*.jsonl`

The scripts skip existing raw MP4s by default. Use `--overwrite-raw` only when
you intentionally want to redownload short YouTube source sections. The 256px
script defaults to `build-clips` so it can re-materialize higher-resolution PNG
clips from already-local segments without touching the network.

Network-free validation and intermediate cleanup (both 64px and 256px share
the same raw + segments tree):

```bash
uv run python src/dataset_pipeline/youtube_ingest.py validate-local \
  --config src/dataset_configs/youtube_scene_distinct_30_64_4fps_16f.jsonc
uv run python src/dataset_pipeline/youtube_ingest.py validate-local \
  --config src/dataset_configs/youtube_scene_distinct_30_256_4fps_16f.jsonc
./src/dataset_scripts/cleanup_youtube_scene_distinct.sh                 # DRY RUN
./src/dataset_scripts/cleanup_youtube_scene_distinct.sh --execute       # actually delete
```

`validate-local` parses `candidates/*.jsonl`, confirms the referenced raw and
segment mp4s exist on disk, and ffprobes a 3-file sample from `raw/` (override
with `--sample-probes N`). `cleanup_youtube_scene_distinct.sh` defaults to
deleting only `clip_sets/<dataset_name>/`, `logs/`, and `raw/*.part` partials.
Pass `--include-segments` to also delete `segments/*.mp4` (recoverable from
raw via the segment stage). Pass `--include-raw` to also delete raw mp4s --
those require yt-dlp + network to recover.

## Curated YouTube Spans

Hand-labeled qualitative/special-effect spans:

```bash
./src/dataset_scripts/youtube_curated_spans_seed.sh
```

Source index:

- `src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc`
- `data/youtube_curated_spans/candidates/curated_spans.jsonl`
- `/Users/nicholasbardy/git/gsplats_browser/corrective_splat_trainer/manual_list.jsonl`

These are not full videos. They are exact local MP4 spans and tiny optional clip
sets.

Network-free validation and intermediate cleanup:

```bash
uv run python src/dataset_pipeline/youtube_curated_spans.py validate-local \
  --config src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc
./src/dataset_scripts/cleanup_youtube_curated_spans.sh                  # DRY RUN
./src/dataset_scripts/cleanup_youtube_curated_spans.sh --execute        # actually delete
```

`validate-local` walks the materialized `curated_spans.jsonl`, confirms each
referenced span has a matching raw mp4, and ffprobes a 3-file sample.
`cleanup_youtube_curated_spans.sh` defaults to deleting only
`clip_sets/<dataset_name>/`, `logs/`, and `raw/*.part`. Raw curated spans are
left in place because re-running yt-dlp against the original YouTube URLs is
the only way to rebuild them.

## YouTube High Camera Motion

Mined seed for high translation/rotation camera footage. Same `youtube_ingest`
entrypoint as scene-distinct but with its own data root and search/segment
tuning:

```bash
./src/dataset_scripts/youtube_motion_seed.sh
```

Source index:

- `src/dataset_configs/youtube_high_camera_motion_seed.jsonc`
- `data/youtube_motion/candidates/*.jsonl` (created on first run)

Network-free validation and intermediate cleanup:

```bash
uv run python src/dataset_pipeline/youtube_ingest.py validate-local \
  --config src/dataset_configs/youtube_high_camera_motion_seed.jsonc
./src/dataset_scripts/cleanup_youtube_motion.sh                         # DRY RUN
./src/dataset_scripts/cleanup_youtube_motion.sh --execute               # actually delete
```

Same defaults as scene-distinct cleanup: only `clip_sets/`, `logs/`, and
partial-download files are removed by default. Raw mp4s require yt-dlp +
network to rebuild, so they are gated behind `--include-raw`.

## DeepView Video

Raw GT multi-camera light-field videos:

```bash
./src/dataset_scripts/deepview_video_seed.sh list-scenes
./src/dataset_scripts/deepview_video_seed.sh all
```

Source index:

- `src/dataset_configs/deepview_video_seed.jsonc`

That config contains all 15 official scene URLs and measured archive sizes. The
default local seed downloads only `03_Dog` and `15_Branches`. Use:

```bash
./src/dataset_scripts/deepview_video_seed.sh all --all-scenes
```

only when there is enough disk; the full configured archive set is about 65.5
GB compressed.

Once each scene is extracted (i.e. its `extracted/<scene>/models.json` exists),
the matching `raw/*.zip` is no longer needed for training and can be reclaimed:

```bash
# Dry-run (~1.5 GB on the default 03_Dog + 15_Branches seed):
./src/dataset_scripts/cleanup_deepview_zips.sh

# Actually delete:
./src/dataset_scripts/cleanup_deepview_zips.sh --execute
```

The cleanup-zips stage is also reachable through the seed script
(`./src/dataset_scripts/deepview_video_seed.sh cleanup-zips [--execute]`). It
skips any zip whose extracted scene is missing or incomplete and is never run
as part of `all`.

## Neural 3D Video

Raw GT multi-camera dynamic-NeRF scenes:

```bash
./src/dataset_scripts/download_neural_3d_video_seed.sh
```

Source index:

- `src/dataset_configs/neural_3d_video_seed.jsonc`
- GitHub release metadata fetched from `facebookresearch/Neural_3D_Video`

The current local seed is `coffee_martini.zip`. Additional release scenes are
tracked as validation TODOs in the config:

- `cook_spinach.zip`
- `cut_roasted_beef.zip`
- `flame_salmon_1_split.*`
- `flame_steak.zip`
- `sear_steak.zip`

To pull all Neural 3D Video release assets when disk allows:

```bash
uv run python src/dataset_pipeline/neural_3d_video.py download --config src/dataset_configs/neural_3d_video_seed.jsonc --all-assets
uv run python src/dataset_pipeline/neural_3d_video.py extract --config src/dataset_configs/neural_3d_video_seed.jsonc
uv run python src/dataset_pipeline/neural_3d_video.py inspect --config src/dataset_configs/neural_3d_video_seed.jsonc
```

Once each scene is extracted (i.e. its `extracted/<scene>/<scene>/poses_bounds.npy`
exists), the corresponding `raw/*.zip` is no longer needed for training and can
be reclaimed. The cleanup stage is opt-in and dry-run by default:

```bash
# Dry-run: print exactly which zips would be deleted, plus total reclaimable bytes.
uv run python src/dataset_pipeline/neural_3d_video.py cleanup-zips \
    --config src/dataset_configs/neural_3d_video_seed.jsonc

# Actually delete (non-interactive wrapper, passes --execute):
./src/dataset_scripts/cleanup_neural_3d_video_zips.sh
```

The cleanup never touches a zip whose extracted scene is missing, and it
never auto-deletes -- you must pass `--execute` (or use the wrapper) to
actually unlink. Re-extraction requires re-downloading from the GitHub release.

## AIST Dance DB

Selected GT fixed-camera dance videos are downloaded by the multi-camera val
builder:

```bash
./src/dataset_scripts/multicam_val_v1_seed.sh download-aist
```

Source index:

- `src/dataset_configs/multicam_val_v1_128_4fps_16f.jsonc`
- official AIST refined 10Mbps CSV URL in that config

The generated selected-video manifest is ignored and can be rebuilt from the
config plus the official CSV.

## ViVo

**ViVo is the exception**: upstream access is gated by an MS Form, not a
public release URL. **Redownloading raw ViVo data is not automatable** — it
requires manual access through the project flow.

**Access links (manual):**

- Project site: <https://vivo-bvicr.github.io/>
- MS Form (request access): <https://forms.office.com/e/gtKpYriSMJ>
- Temporary dataset folder (per-recipient Drive link):
  <https://drive.google.com/drive/folders/1uG4JB2GDWrIRMqmbI6NCP2kA0jUDAbvp?usp=sharing>
- Processing code: <https://github.com/azzarelli/ViVo-DataProcessing>

### What's recoverable vs. NOT

Under `data/external/vivo/`:

- **NOT recoverable without MS-Form access** (PROTECTED — never auto-deleted):
  - `raw/` — original zip/tar bundles before extraction
  - `extracted/<scene>/{train,test}/<cam>/*.jpg(.meta.json)` — raw colour frames
  - `extracted/<scene>/calibration.json` — rig calibration (and
    `rotation_correction.json` when present)
- **Recoverable from `extracted/` (safe to delete and rebuild):**
  - `rgb_mp4/<scene>/{train,test}/<cam>.mp4` — rebuilt by `compact-rgb`
  - `metadata/rgb_mp4_frames/`, `metadata/rgb_mp4_manifest.jsonl` — rebuilt by
    `compact-rgb`
  - `metadata/scene_inventory.json` — rebuilt by `inspect`
  - `logs/` — ffmpeg/inspect logs

### Validate the local bundle (CPU-only)

```bash
uv run python /tmp/audit_vivo_bundle.py
```

Reports per-area disk usage, scene presence, calibration status, per-camera
frame counts, and an mp4 sample dimension. Works even if no compacted MP4s
exist yet.

### Local redo from already-downloaded raw ViVo scene

```bash
uv run python src/dataset_pipeline/vivo.py inspect --config src/dataset_configs/vivo_seed.jsonc
uv run python src/dataset_pipeline/vivo.py compact-rgb --config src/dataset_configs/vivo_seed.jsonc --scene athlete_rows
```

### Reclaim derived staging space (safe; never touches `extracted/` or `raw/`)

Dry-run preview:

```bash
uv run python src/dataset_pipeline/vivo.py cleanup-staging --config src/dataset_configs/vivo_seed.jsonc
```

Actually delete derived artifacts:

```bash
./src/dataset_scripts/cleanup_vivo_staging.sh
```

Source index:

- `src/dataset_configs/vivo_seed.jsonc`
- local generated `data/external/vivo/metadata/rgb_mp4_manifest.jsonl` after
  compaction

**If `extracted/` is also deleted, the only path back is the MS Form +
per-recipient Drive link.** The cleanup script never deletes it.

## Ex4DGS Checkpoints

Pretrained checkpoint/model bundles, not raw GT videos:

```bash
./src/dataset_scripts/ex4dgs_pretrained_val_seed.sh
```

Source index:

- `src/dataset_configs/ex4dgs_pretrained_val_seed.jsonc`
- GitHub release metadata fetched from `juno181/Ex4DGS`

These can be redownloaded independently, but they do not replace raw
source/target validation videos.

Network-free validation that the extracted bundles are loadable, plus disk
reclaim of the raw zips:

```bash
uv run python src/dataset_pipeline/ex4dgs_pretrained.py validate-local \
  --config src/dataset_configs/ex4dgs_pretrained_val_seed.jsonc

# Dry-run (~318 MB on default 3-bundle seed):
./src/dataset_scripts/cleanup_ex4dgs_zips.sh

# Actually delete:
./src/dataset_scripts/cleanup_ex4dgs_zips.sh --execute
```

`validate-local` checks that each bundle has a non-empty `cameras.json` plus
`mean_metrics.json` plus at least one `.ply` point cloud (no torch model load).
`cleanup-zips` only removes a zip whose extracted bundle passes that same
populated-check; otherwise it skips with a reason.

## Blender Synthetic / Sintel

Hand-rendered synthetic-render bundle plus its upstream sources. There is no
download stage: the production tree comes from the Sintel DVD ISO on
archive.org and Blender 2.79b from the Blender release server.

Source index:

- `data/blender_synthetic/sintel/_iso/Sintel_DATA.iso` (~7.5 GB)
  -- the ONLY free public source for Sintel production blends; the original
  Durian SVN at `download.blender.org/durian/svn/` is dead. NEVER auto-deleted.
  Re-acquire from <https://archive.org/download/sintel-dvd/>.
- `data/blender_synthetic/sintel/_iso/Sintel_PAL.iso` (~7.4 GB) -- PAL DVD
  ISO; also on archive.org. Not the production-blend source. Reclaimable on
  opt-in.
- `data/blender_synthetic/sintel/_iso/Sintel.2010.1080p.mkv` (~1.1 GB) --
  finished movie file. Reclaimable.
- `data/blender_synthetic/sintel/renders/02_a_*` -- locally produced scratch
  renders of the `02_shaman` scene. Reclaimable on opt-in (re-renderable).
- `data/blender_synthetic/_blender_2_79b/blender-2.79b-macOS.zip` -- Blender
  2.79b installer; re-fetchable from
  <https://download.blender.org/release/Blender2.79/>. Reclaimable.
- `data/blender_synthetic/_blender_2_79b/blender-2.79b-macOS-10.6/` --
  extracted Blender app. Reclaimable on opt-in (re-extractable from the zip).

Inspect, validate, and reclaim:

```bash
uv run python src/dataset_pipeline/blender_synthetic_inventory.py inspect
uv run python src/dataset_pipeline/blender_synthetic_inventory.py validate-local

# Dry-run shows the auto-reclaimable + opt-in sets, and the PRESERVE entry for
# Sintel_DATA.iso so it is always visible.
./src/dataset_scripts/cleanup_blender_synthetic.sh                          # ~1.3 GB auto
./src/dataset_scripts/cleanup_blender_synthetic.sh --include-protected      # ~9.7 GB total
./src/dataset_scripts/cleanup_blender_synthetic.sh --include-protected --execute
```

`validate-local` errors out if `Sintel_DATA.iso` is missing or truncated --
that is the load-bearing asset to keep. The cleanup script is hard-wired to
never touch the DATA ISO; it must be deleted by hand if you really want to.

## Multi-Camera Validation Set

Build the current tiny GT novel-camera validation set after required source
datasets are present:

```bash
./src/dataset_scripts/deepview_video_seed.sh all
./src/dataset_scripts/multicam_val_v1_seed.sh all
```

`multicam_val_v1_seed.sh all` runs every multicam stage in order:
`download-aist`, `download-aist-cameras`, `build`, `inspect`. The
`download-aist-cameras` stage pulls the AIST++ camera-parameter zip
(`cameras.zip`) and writes a per-sequence camera inventory; running just that
stage:

```bash
./src/dataset_scripts/multicam_val_v1_seed.sh download-aist-cameras
```

Source index:

- `src/dataset_configs/multicam_val_v1_128_4fps_16f.jsonc`

The output under `data/multicam_val/clip_sets/` is generated and ignored. It is
safe to delete and rebuild as long as the selected source videos/inventories are
present or re-downloadable.
