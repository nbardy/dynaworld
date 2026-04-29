# DomeDB and BrettZone Multicam Source Notes

Date: 2026-04-30 00:14 +0700

User pointed at two more sources:

```text
http://domedb.perception.cs.cmu.edu/
https://brettzone.nhrl.io/brettZone/fightReviewSync.php?gameID=W-14&tournamentID=nhrl_dec25_30lb
```

The request was also to mark "multicam practice" as an important scalable
source category, especially for YouTube-like monitor-wall / multiview material.

## Category Decision

Add and preserve this third lane:

```text
scalable pseudo-multicam practice data
```

It is separate from:

- monocular YouTube scene-diverse smoke data; and
- calibrated GT held-out-camera validation data.

The category includes monitor-wall videos, director-comms recordings, live
production multiviews, and event archives where the same real-world moment is
visible from several camera feeds but camera intrinsics/extrinsics are unknown.

Use it for:

- implicit-camera practice;
- camera-ID-conditioned training;
- cross-view pressure;
- tile-heldout or camera-heldout qualitative checks;
- robustness pretraining on real dynamic events.

Do not use it for final held-out-camera PSNR/SSIM unless calibration and timing
are recovered well enough to make the target camera metric meaningful.

## CMU Panoptic Studio / DomeDB

Site:

```text
http://domedb.perception.cs.cmu.edu/
```

Local verification command:

```bash
curl -L --max-time 20 -s "http://domedb.perception.cs.cmu.edu/" | sed -n "1,220p"
```

Facts observed on the homepage:

- Title: `CMU Panoptic Dataset`.
- Hardware: 480 VGA camera views, 30+ HD views, 10 RGB-D sensors.
- Synchronization: hardware-based sync.
- Calibration: listed explicitly.
- Scene labels: multiple people, socially interacting groups, 3D body pose, 3D
  facial landmarks, transcripts, speaker ID.
- Dataset size: 65 sequences, 5.5 hours, about 1.5 million 3D skeletons.
- License: research-only; no commercial use; no redistribution of the dataset
  or modified versions without organizer permission.
- Related PtCloudDB note: 10 Kinects with 41 RGB videos and 6+ hours of data.
- Sep. 2016 release note: 480 VGA videos, 31 HD videos, 3D body pose, and
  calibration data available.

Dynaworld routing:

- This is a curated external calibrated dataset candidate, not a scrape target.
- It belongs closer to DeepView / ViVo / Neural 3D Video / AIST than to the
  pseudo-multicam practice lane.
- First pass should inspect one sequence and answer:
  - how downloads are indexed;
  - whether HD or VGA video is the right first adapter target;
  - calibration file format and axis convention;
  - synchronized frame window availability for 16 frames at 4fps;
  - whether the license is compatible with local research-only experiments.

## BrettZone / NHRL Robot Fights

Example fight:

```text
https://brettzone.nhrl.io/brettZone/fightReviewSync.php?gameID=W-14&tournamentID=nhrl_dec25_30lb
```

Observed page facts:

- Page title: `KaZaA Lite vs Lil Lash - Multi-Camera`.
- Tournament: `2025 World Championships 30lb`.
- Match duration: 3:00.
- The UI exposes camera names including Program Feed, Cage 1 Overhead High,
  Cage 1 Blue High, Cage 1 Ceiling High, Cage 1 NE High, Cage 1 Red High,
  Cage 1 SW High, MobileCam 1, MobileCam 3, MobileCam 4, SkyCam A, SkyCam B.
- The page embeds `window.MATCH_DATA.recordings` with per-camera MP4 URLs:
  `proxy720`, `proxy360`, `proxy72`, and native `s3path`.
- The visible camera selector counted 12 selectable angles in this example,
  while the stats API reported `cameraCount: 13`; inspect this mismatch before
  assuming the page count and API count always match.

Stats/API surfaces verified:

```bash
curl -L -s "https://brettzone.nhrl.io/brettZone/api.php/stats/fight/nhrl_dec25_30lb/W-14"
curl -L -s "https://brettzone.nhrl.io/brettZone/api.php/stats/tournament/nhrl_dec25_30lb"
curl -L -s "https://brettzone.nhrl.io/brettZone/api.php/tournaments" | jq ".tournaments | length"
curl -L -s "https://brettzone.nhrl.io/brettZone/backend/getTournamentMatchesDataTables.php?tournamentID=nhrl_dec25_30lb&draw=1&start=0&length=100"
```

Observed API facts:

- BrettZone stats docs say no auth is required and responses are JSON.
- Stats docs list base URL `/api.php/stats`.
- Stats docs say original tournament/fight/player/video endpoints remain at
  `/api.php`.
- `/api.php/tournaments` returned 133 tournament records at probe time.
- `nhrl_dec25_30lb` stats: 28 players, 63 total fights, 48 completed fights,
  average completed duration 94.4 seconds.
- Example fight `W-14`: 180 seconds, final, judges decision, `cameraCount: 13`.
- Tournament DataTables endpoint returned 41 rows for `nhrl_dec25_30lb` in the
  web table query, with rows linking directly to `fightReviewSync.php`.

Dynaworld routing:

- BrettZone is a very strong scalable pseudo-multicam practice source.
- It is better than tiled YouTube monitor-wall video because individual camera
  MP4 URLs are already exposed; no tile-cropping step is required.
- It is still not calibrated NVS validation by default: camera intrinsics,
  extrinsics, lens model, and exact sync metadata are not present in the
  verified API/page surfaces.
- It is excellent for dynamics: short matches, high motion, hard occlusion,
  sparks/fire/debris, small fast objects, stable cage cameras, and multiple
  tournaments/weight classes.

Suggested future pipeline:

```text
src/dataset_pipeline/brettzone_nhrl.py
src/dataset_configs/brettzone_nhrl_seed.jsonc
src/dataset_scripts/brettzone_nhrl_seed.sh
data/brettzone_nhrl/
```

Stages:

1. `list-tournaments`: call `/api.php/tournaments`, filter public tournaments.
2. `list-fights`: query `backend/getTournamentMatchesDataTables.php` per
   tournament, selecting fights with enough cameras and valid duration.
3. `inspect-fight`: fetch `fightReviewSync.php`, parse
   `window.MATCH_DATA.recordings`, and write camera URL metadata.
4. `download-spans`: download short synchronized spans from selected camera
   MP4s, not whole tournaments.
5. `build-pseudo-multicam`: emit records with unknown camera pose source:

```json
{
  "sample_id": "brettzone_nhrl_dec25_30lb_W-14",
  "dataset": "brettzone_nhrl",
  "modality": "pseudo_multicam_uncalibrated",
  "source_url": "https://brettzone.nhrl.io/brettZone/fightReviewSync.php?gameID=W-14&tournamentID=nhrl_dec25_30lb",
  "duration_seconds": 180,
  "camera_pose_source": "unknown_brettzone_arena_camera",
  "camera_ids": ["Program-Feed", "Cage-1-Overhead-High", "Cage-1-Blue-High"],
  "heldout_policy": "camera_feed_heldout_for_practice_only"
}
```

The first smoke should use one fight, three cameras, 16 frames at 4fps, and no
W&B/training. Just prove synchronized decode, manifest shape, and frame-size
normalization.

## TVMCE / Virtual Film Studio Multi-Camera Editing

Project page:

```text
https://virtualfilmstudio.github.io/projects/multicam/
```

Paper:

```text
https://arxiv.org/abs/2210.08737
```

Data link exposed on project page:

```text
https://drive.google.com/drive/folders/1V-ZKbwJgUD5rv3lI0dzCSAspLjm1mMMb?usp=sharing
```

Verified project-page facts:

- Paper/project title: `Temporal and Contextual Transformer for Multi-Camera
  Editing of TV Shows`.
- Venue: ECCVW 2022.
- Dataset name: TV shows MultiCamera Editing, abbreviated TVMCE.
- Scenarios: concerts, sports games, gala shows, and contests.
- Each scenario contains 6 synchronized tracks recorded by different cameras.
- Dataset scale: 88 hours of raw videos contributing to 14 hours of edited
  videos.
- Scenario mix includes 39% gala shows and 14% sports.
- Most shots last 0 to 8 seconds; some long shots last more than 32 seconds.

Dynaworld routing:

- This is a strong pseudo-multicam practice / editing-supervision source.
- It is not calibrated novel-view validation by default. The project page
  advertises synchronized camera tracks and edited-video supervision, not
  camera intrinsics/extrinsics.
- It is particularly useful for view-selection, transition timing, temporal
  context, and cross-view conditioning objectives because the edited program
  stream gives a human camera-choice label.
- Treat it as closer to BrettZone and YouTube production multiview than to
  DeepView/DomeDB/ViVo unless a calibration file is found inside the Drive
  data.

Suggested future root:

```text
data/external/tvmce/
```

Suggested first pass:

1. Inspect the Drive folder structure without downloading the full dataset.
2. Identify whether the six synchronized tracks are separate files with a
   shared timeline and whether the edited output is aligned by timestamp or
   shot metadata.
3. Check license/terms before local mirroring.
4. Download one small event if terms allow, then emit a practice manifest with:

```json
{
  "sample_id": "tvmce_<event_id>_<window_id>",
  "dataset": "tvmce",
  "modality": "pseudo_multicam_editing",
  "camera_pose_source": "unknown_tv_production_camera",
  "camera_ids": ["camera_00", "camera_01", "camera_02", "camera_03", "camera_04", "camera_05"],
  "supervision": ["synchronized_camera_frames", "edited_program_view_id"]
}
```
