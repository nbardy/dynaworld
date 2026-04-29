# YouTube Multiview/Comms Playlist Discovery

Date: 2026-04-29 23:54 +0700

User found this playlist as a possible real-world multiview source:

```text
https://www.youtube.com/playlist?list=PLU1686G8dBiTpsGq2LInnTkfSMrM0AJ2D
```

## Metadata Scrape

Command run from the Dynaworld root:

```bash
uv run --with yt-dlp yt-dlp --flat-playlist --dump-single-json "https://www.youtube.com/playlist?list=PLU1686G8dBiTpsGq2LInnTkfSMrM0AJ2D"
uv run --with yt-dlp yt-dlp --flat-playlist --print "%(playlist_index)03d\t%(id)s\t%(duration_string)s\t%(title)s\t%(uploader)s\t%(webpage_url)s" "https://www.youtube.com/playlist?list=PLU1686G8dBiTpsGq2LInnTkfSMrM0AJ2D"
```

Observed facts:

- Playlist title: `Multiview/Coms`.
- Playlist description: `Recordings of video directors calling shows.`
- Owner/channel: `Benji York`.
- Playlist count: 45 entries.
- Public/resolvable entries in the flat scrape: 36.
- Private/deleted entries in the flat scrape: 9.
- Public entries under 10 minutes: 13.
- YouTube/yt-dlp warning: 9 unavailable videos are hidden.
- No video payloads were downloaded in this pass.
- Local disk at scrape time: about 47 GiB free, so full-video scraping should be avoided.

Persisted metadata artifacts:

```text
data/youtube_multiview_comms/candidates/playlist_summary.json
data/youtube_multiview_comms/candidates/playlist_entries.jsonl
```

## Why This Is Interesting

These are not calibrated multiview datasets like DeepView, ViVo, Neural 3D
Video, or AIST. They are mostly production monitor-wall / director-comms
recordings: one YouTube video contains several camera feeds visible at the
same time in a tiled multiview layout.

That still makes them useful. A single monitor-wall video can be split into
roughly synchronized tile streams:

```text
download short source span -> detect/crop tile rectangles -> emit camera_0000,
camera_0001, ... streams for the same timeframe
```

This creates real-world pseudo-multicam supervision. It is not metric NVS
validation because we do not have intrinsics/extrinsics, but it can provide
cross-view pressure for an implicit-camera or camera-ID-conditioned training
path.

## Main Pros

- Real events, real cameras, real operator choices, real dynamic foreground.
- Multiple viewpoints are visible simultaneously in one video, so tile crops
  share a clock by construction.
- Many clips are stage, worship, sports, or broadcast-control scenes with
  humans and strong lighting changes.
- Good fit for a cheap pretraining/regularization lane where exact camera
  calibration is not required.

## Main Cons

- Not a calibrated held-out-camera benchmark.
- Tile crops are compressed twice: original camera feed into production system,
  then monitor-wall/composite into YouTube.
- Layouts vary by video and may change during a video.
- Tiles may include program/preview feeds, labels, borders, black regions,
  tally overlays, UI chrome, or non-camera feeds.
- Camera latency inside the production system may not be perfectly uniform.
- Some views are not useful novel views of the same 3D scene, such as graphics,
  slides, audience shots, or control-room shots.
- Full videos are often long, so whole-video download is the wrong first move.

## Suggested Pipeline

Add a separate lane rather than folding this into the calibrated multicam
validation path:

```text
src/dataset_pipeline/youtube_multiview_wall.py
src/dataset_configs/youtube_multiview_comms_seed.jsonc
src/dataset_scripts/youtube_multiview_comms_seed.sh
```

Stages:

1. `scrape-playlist`: store flat playlist metadata and skip private/deleted
   entries.
2. `download-spans`: download short 8-12 second sections only, preferably at
   720p or 1080p because tiles need enough pixels after cropping.
3. `probe-frames`: save a few contact sheets per source span for manual or
   automatic tile-layout review.
4. `detect-grid`: find stable tile rectangles using black borders, repeated
   grid lines, and per-tile motion/brightness filters.
5. `crop-tiles`: write one MP4 per accepted tile/camera for the same time span.
6. `build-manifest`: emit pseudo-multicam records with unknown calibration:

```json
{
  "sample_id": "youtube_multiview_comms_iaRmO3v9KEg_s060_e072",
  "dataset": "youtube_multiview_wall",
  "modality": "pseudo_multicam_uncalibrated",
  "source_url": "https://www.youtube.com/watch?v=iaRmO3v9KEg",
  "frame_count": 16,
  "fps": 4.0,
  "camera_pose_source": "unknown_tile_camera",
  "camera_ids": ["tile_r0_c0", "tile_r0_c1", "tile_r1_c0", "tile_r1_c1"],
  "heldout_policy": "tile_heldout_for_implicit_camera_only"
}
```

Do not report these as calibrated held-out-camera PSNR/SSIM baselines. If we
use a tile as heldout, the metric is only a pseudo-view metric under unknown
camera geometry.

## First Seed Candidates

Prefer shorter public videos first to limit storage and reduce layout drift:

```text
001 iaRmO3v9KEg 7:56  Kari Jobe Forever Live Multiview | 1DayHouston
002 yq5rqxTkZ2Y 6:06  Live Production Multiview - Worthy by Elevation Worship at CreationFest
006 vBKatqx98R4 6:03  Live Multiview with Director | Athey Music "Trust"
016 UElXy9ndJ0w 2:19  King of my Heart - Broadcast Multiview - Bethel Church
024 pe5ssLQpkQA 8:21  Heaven Come: LA 2019 | Ain't No Grave Multiview
025 p8yCCvBBP2M 2:51  WorshipU After Hours Moment - Multiview
027 UZwyENkffPQ 6:07  Festive Overture for Symphony Band - Multiview Director Preview
028 gfjWjkTP4p8 5:40  Inside the Super Bowl 50 Halftime Show Control Room
029 RxFg9YIDH0o 5:40  Basic TV Newscast - Control Room Multiview
030 DP2QOmN57iU 3:29  Grease Live Control Room Split Screen
041 to0P2ZAVdy0 6:05  Never Lost - Amplify Online Multi View Live Switching
042 LF9x9q9UeEs 7:04  JOM Monitor Wall Director's Track 2012_04_28A
```

## Recommended Next Step

Do a tiny manual seed before automation:

1. Download 3 short 8-12 second sections from three sources above.
2. Generate first/middle/last contact sheets.
3. Manually choose tile rectangles for the best one.
4. Crop tile MP4s into a pseudo-multicam manifest.
5. Run a loader-only smoke that checks synchronized frame counts and tile sizes.

Only after one seed works should we automate grid detection across the rest of
the playlist.
