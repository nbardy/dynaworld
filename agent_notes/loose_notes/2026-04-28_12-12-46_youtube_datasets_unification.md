# YouTube datasets: validate-local + cleanup-intermediates unification

Date: 2026-04-28
Topic: bringing the three YouTube training-video pipelines (curated spans,
scene-distinct 64/256, high-camera-motion) under a uniform offline-validate +
intermediate-cleanup contract so the local data tree is reproducible from raw
mp4s + checked-in configs alone.

## What changed

1. `src/dataset_pipeline/youtube_curated_spans.py`
   - Added `validate-local` stage. Materializes spans (no network), checks
     each `clip_id` has a matching mp4 in `raw/`, ignores `*.part` partials
     when matching (the existing `existing_download` glob would otherwise
     return `<clip_id>.mp4.part`). Ffprobes a sample of N raw mp4s
     (`--sample-probes`, default 3) and emits a JSON summary plus warnings.
   - Added `cleanup-intermediates` stage. Default DRY-RUN. Lists reclaimable
     bytes for `clip_sets/<dataset_name>/`, `logs/`, and `raw/*.part`
     partials. `--execute` actually deletes. `--include-raw` is a separate
     opt-in for the source-of-truth raw spans.

2. `src/dataset_pipeline/youtube_ingest.py` (shared by scene-distinct AND
   high-camera-motion seeds)
   - Same `validate-local` and `cleanup-intermediates` stages, adapted to the
     ingest schema:
     - validate cross-checks `downloads.jsonl` (raw) and
       `segments_manifest.jsonl` (segments) against the on-disk tree
     - cleanup defaults to `clip_sets/<dataset_name>/` + `logs/` + raw
       partials; `--include-segments` is a separate opt-in for `segments/`,
       and `--include-raw` is the strongest opt-in.

3. New cleanup wrapper scripts (all non-interactive, default dry-run):
   - `src/dataset_scripts/cleanup_youtube_curated_spans.sh`
   - `src/dataset_scripts/cleanup_youtube_motion.sh`
   - `src/dataset_scripts/cleanup_youtube_scene_distinct.sh` (loops over both
     the 64px and 256px configs since they share `raw/` and `segments/` and
     differ only in their `clip_sets/<dataset_name>/` subtree).

4. `data/REHYDRATE.md`
   - Extended "Default Local Smoke Set" (scene-distinct) and "Curated YouTube
     Spans" with the new validate/cleanup commands.
   - Added a new "YouTube High Camera Motion" section.
   - Did not touch the Neural 3D / ViVo / DeepView / Ex4DGS / multicam_val
     sections.

## Source-of-truth vs intermediates contract

For every YouTube pipeline:

```
SOURCE OF TRUTH  (never auto-deleted)
  candidates/*.jsonl     -- materialized span / search / download manifests
  raw/*.mp4              -- yt-dlp output (requires network to recover)
  segments/*.mp4         -- only youtube_ingest pipelines; recoverable from
                            raw via the segment stage but kept by default

INTERMEDIATES  (deletable by default with --execute)
  clip_sets/<dataset_name>/   -- PNG frame dumps + per-clip summary.json,
                                 fully derived from raw/segments + config
  logs/                       -- yt-dlp stdout/stderr captures
  raw/*.part, raw/*.ytdl, raw/*.tmp  -- partial yt-dlp downloads
```

Cleanup tiers (each level is a strictly larger superset of the previous):
- default: `clip_sets/<name>/ + logs/ + raw partials`
- `--include-segments` (ingest only): + `segments/*.mp4`
- `--include-raw`: + `raw/*.mp4`

The escalation is intentional: raw spans cost a yt-dlp re-run and the YouTube
URL still has to be live; segments are CPU-only re-encoding from raw; clips
are pure PNG-dump from segments/raw.

## Disk-size audit (today)

```
data/youtube_curated_spans/
  candidates/    29443 B    0.03 MB     (source of truth)
  clip_sets/   2160575 B    2.06 MB
  logs/         728990 B    0.70 MB
  raw/        55816000 B   53.23 MB     (source of truth, includes 1 .part)

data/youtube_motion/
  candidates/        0 B    0.00 MB     (empty -- never seeded)
  clip_sets/         0 B    0.00 MB
  logs/              0 B    0.00 MB
  raw/               0 B    0.00 MB
  segments/          0 B    0.00 MB

data/youtube_scene_distinct/
  candidates/    61259 B    0.06 MB     (source of truth)
  clip_sets/  46081097 B   43.95 MB     (64f + 256f combined)
  logs/         409801 B    0.39 MB
  raw/        47000521 B   44.82 MB     (source of truth)
  segments/   28304291 B   26.99 MB     (recoverable from raw)
```

Reclaimable today (default cleanup, no `--include-raw`/`--include-segments`):

| dataset                          | reclaimable        |
|----------------------------------|--------------------|
| curated_spans                    | 2,889,565 B (2.76 MB) |
| scene_distinct (64f config)      | 4,283,058 B (4.08 MB) |
| scene_distinct (256f config)     | 42,617,641 B (40.64 MB) |
| youtube_motion                   | 0 B (empty data root)  |

## Validate-local audit

- `youtube_curated_spans` (config: 64f): 27 expected spans, 27 materialized,
  26 raw present, 1 missing (`ODmhPsgqGgQ_seg_000_s01354000_e01362000` --
  download was interrupted; only the `.part` survives). 3-file ffprobe sample
  on `mkVYpzyJvG8_seg_{000,001,002}_*.mp4` reports `h264 1280x720 ~5-19 s`.
- `youtube_scene_distinct_30_64`: 45 downloads recorded, 45 raw present, 0
  missing; 41 segments recorded, 41 segments present. ffprobe sample on
  `Tovxu_sm9BA / Bq4rmeIvJbs / 1aedKShR1rA` reports `h264 640x360 ~8 s`.
- `youtube_high_camera_motion`: empty data tree -- validator returns zero
  counts and no warnings (correct, no `candidates/` jsonl yet).

## Bug fixed along the way

`existing_download(raw_dir, clip_id)` globs `clip_id.*` and sorts. When a
yt-dlp run is interrupted the `.part` sibling sorts after `.mp4`, so the
helper would return the partial. The validator now filters
`PARTIAL_DOWNLOAD_SUFFIXES` after the lookup. This is documented inline at
the call site (and the partial is still surfaced separately as a warning so
the operator can clean it up). Worth porting the same filter into the
download stage's `existing_download` short-circuit later -- right now an
interrupted previous run could trick the resume logic into thinking the file
is already there.

## Style / convention follow-through

- All stages registered through the existing argparse subcommand pattern.
  No new env-var fanout; configs stay in JSONC.
- Shell wrappers do nothing the python entrypoint does not already do; they
  just bind the right `--config` and forward `$@`. The scene-distinct wrapper
  iterates both configs since they share `raw/` + `segments/`.
- ffprobe is required only for `validate-local`; the failure message names
  the missing tool. No silent fallback if ffmpeg isn't installed.
- All new code routes through the existing `Paths` dataclass (`resolve_paths`)
  so the config remains the single source of truth for the on-disk layout.

## What still defers

- Adding `validate-local` to the regular `seed.sh` "all" pipelines as a final
  post-step. Right now you have to call it explicitly. Postponed because the
  seed scripts are network-on flows and validate is offline-only -- worth
  keeping them separate until we wire a real CI smoke.
- `download` resume logic in both pipelines should drop `.part`/`.ytdl`/
  `.tmp` from `existing_download` matches before deciding "already present".
  Same one-line filter as in validate. Followup ticket.
- The 256f scene-distinct config still hardcodes the same `search` block as
  the 64f config; if that ever drifts the two cleanup script invocations
  could see different `dataset_name`s but the same raw set, which is already
  the case today and works fine. Just noting.
