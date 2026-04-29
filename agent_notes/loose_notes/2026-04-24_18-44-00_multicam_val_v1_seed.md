# Multi-Camera Val V1 Seed

The user asked to pull AIST validation clips but not make the validation set
only dancing people. Built a tiny paired source/target multi-camera validation
seed with three sources:

- AIST Dance DB refined 10Mbps `gBR_sBM_d04_mBR0_ch01`, `c01 -> c05` and
  `c01 -> c09`.
- Neural 3D Video `coffee_martini`, `cam00 -> cam10` and `cam00 -> cam20`.
- ViVo `athlete_rows`, `000404613112 -> 000497113112` and
  `000404613112 -> 000516213112`.

New files:

```text
src/dataset_configs/multicam_val_v1_128_4fps_16f.jsonc
src/dataset_pipeline/multicam_val.py
src/dataset_scripts/multicam_val_v1_seed.sh
src/train/multicam_val_data.py
```

Output root:

```text
data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/
```

The output has six validation pairs. Each pair has source frames, target frames,
a side-by-side preview MP4, a sample summary JSON, and a shared manifest. The
paired loader `multicam_val_data.py` smoke-loaded every record as
`(16, 3, 128, 128)` source and target tensors.

Follow-up: the first preview MP4s looked choppy because they were rendered at
the same 4fps as the metric tensors. Split preview playback from metric
sampling: validation frames remain 4fps, while side-by-side preview MP4s now
render at 30fps.

Follow-up 2: raw validation data should stay native. Rebuilt the validation
sample set with `materialize_metric_frames=false`; the manifest now points at
native MP4s plus synchronized time windows, and `multicam_val_data.py` samples
the `(16, 3, 128, 128)` source/target tensors from native videos at load time.
The generated clip directories now contain per-sample summaries rather than
downsampled frame PNGs.

AIST raw videos were downloaded through the official refined 10Mbps CSV:

```text
data/external/aist_dance_db/raw/refined_10M_sBM/
```

ViVo required timestamp alignment: local train/test camera MP4s have different
first capture timestamps, so the validation builder computes the overlap window
from metadata instead of assuming frame 0 in each camera is synchronized.
