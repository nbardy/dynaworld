# YouTube Curated Span Follow-Up

The user asked about an older file/folder with manually annotated YouTube time
spans, separate from the newer Dynaworld YouTube ingest.

Found the older annotation file in the parent repo:

```text
/Users/nicholasbardy/git/gsplats_browser/corrective_splat_trainer/manual_list.jsonl
```

It contains user-curated YouTube URLs with either `segments` entries carrying
`start_seconds` / `end_seconds`, or `whole_video: true` for shorts. It is not in
`dynaworld/data/youtube_*`; it predates the newer search-based ingest work.

Added a Dynaworld curated-span ingest that imports that old JSONL and appends a
new Matrix bullet-time test clip:

```text
https://www.youtube.com/watch?v=8DajVKAkL50
0:37 => 0:46
title: matrix bullettime
split: test
```

New files:

```text
src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc
src/dataset_pipeline/youtube_curated_spans.py
src/dataset_scripts/youtube_curated_spans_seed.sh
```

The pipeline stages are:

```bash
./src/dataset_scripts/youtube_curated_spans_seed.sh materialize
./src/dataset_scripts/youtube_curated_spans_seed.sh download
./src/dataset_scripts/youtube_curated_spans_seed.sh build-clips
```

The exact downloaded span MP4s go under:

```text
data/youtube_curated_spans/raw/
```

The optional frame dataset goes under:

```text
data/youtube_curated_spans/clip_sets/youtube_curated_spans_64_4fps_16f/
```

Run results:

- `materialize`: wrote 27 curated spans (`26` train from the old parent file,
  `1` Matrix test span).
- `download`: wrote 26 exact-span MP4s; 1 old train record failed.
- Failed record: `ODmhPsgqGgQ` at `22:34` defaulted to `22:42`, but `yt-dlp`
  currently resolves that video to about 41 seconds, so ffmpeg cannot cut the
  annotated range.
- Matrix output:

```text
data/youtube_curated_spans/raw/matrix_bullettime.mp4
```

`ffprobe` reports a roughly 9.0 second 636x360 MP4, matching the requested
`0:37 => 0:46` range.

- `build-clips`: wrote 19 usable 64px/4fps/16-frame clips (`18` train, `1`
  test). The very short 2-3 second spans downloaded correctly as MP4s but are
  too short for 16 frames at 4fps, so the frame builder skipped them.

Follow-up correction after inline review:

- Updated the parent hand-labeled source
  `/Users/nicholasbardy/git/gsplats_browser/corrective_splat_trainer/manual_list.jsonl`.
- Corrected `mkVYpzyJvG8` segment 0 to `1:51 => 2:10`, corresponding to
  `0:36 => 0:55` inside the original local `1:15 => 2:10` clip. An
  intermediate attempt incorrectly treated `0:35 => 0:55` as absolute YouTube
  time; the final manifest uses the absolute source times.
- Corrected `hlaZbH_OFBU` segment 4 from `7:48 => 7:54` to `7:49 => 7:54`,
  matching the user's "start at 0:01 not 0:00" review note for the local clip.
- Raised the curated raw download cap from 360p to 720p, then regenerated
  `curated_spans.jsonl`, `downloads.jsonl`, and the 64px/4fps/16f clip set.
  The same stale `ODmhPsgqGgQ` source still fails.

Download infra follow-up:

- The curated-span and scene-distinct wrapper scripts no longer force raw-media
  overwrite by default.
- Default runs now reuse existing local raw MP4s when the clip/video ID already
  exists. Use `--overwrite`, `--overwrite-raw`, or `OVERWRITE_RAW=1` when a real
  redownload is intended.
- Derived 64px clip sets still overwrite by default because those are cheap
  generated artifacts and otherwise `build_clip_dataset.py` refuses to replace
  an existing output directory. Use `--no-overwrite-clips` to keep them.
- Added local-file reuse to `youtube_ingest.py`; the curated-span pipeline
  already had the skip path, but the wrapper was previously bypassing it by
  always passing `--overwrite`.
- Loader smoke: the default scene-distinct manifest loads a train and test
  sequence as `(16, 3, 64, 64)` tensors at 4fps, with `20` train and `10` test
  manifest entries.
