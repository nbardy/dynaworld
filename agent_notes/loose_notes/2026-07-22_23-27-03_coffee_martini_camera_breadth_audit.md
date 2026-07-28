# Coffee Martini camera-breadth audit

## Scope and sources

This is a read-only audit of the local Coffee Martini data under the canonical
multicamera contract. It does not change a manifest, loader, browser bundle, or
generated asset. Sources inspected:

- `research_notes/data_contract.md`
- `src/train/multicam_video_data.py`
- `src/train/paper_training_types.py`
- `src/train/paper_training_protocol.py`
- `src/train/export_dynaworld_browser_bundle.py`
- `src/dataset_configs/neural3d_coffee_martini_train2_holdout1_full_300f_manifest.jsonl`
- `src/dataset_configs/neural3d_coffee_martini_paper_triplets_full_300f_manifest.jsonl`
- local MP4s and `poses_bounds.npy` under
  `data/external/neural_3d_video/extracted/coffee_martini/coffee_martini/`
- the current browser `coffee_martini_multicam` JSON and PNG atlases

## Exact local breadth

The local scene has **18 calibrated camera videos**:

`cam00`, `cam01`, `cam02`, `cam04`, `cam05`, `cam06`, `cam07`, `cam08`,
`cam09`, `cam10`, `cam11`, `cam12`, `cam13`, `cam14`, `cam16`, `cam18`,
`cam19`, and `cam20`.

The upstream numbering is non-contiguous: `cam03`, `cam15`, and `cam17` are
absent. This is expected and is documented by the Neural3D adapter. The local
`poses_bounds.npy` has shape `(18, 17)`, exactly matching the 18 MP4s. The
loader maps calibration rows to lexicographically sorted MP4 names.

## Synchronized availability

All 18 MP4s report the same stream facts:

- 2704 x 2028 pixels
- 30/1 fps
- 300 frames
- stream and container start time 0.0 seconds
- stream and container duration 10.0 seconds
- time base 1/15360

Therefore every camera has the full canonical frame-index interval `0..299`,
and `(camera, frame_index)` is available across all 18 views. This satisfies
the repo's synchronized Neural3D contract. Container metadata alone cannot
independently prove sub-frame capture phase, so physical synchronization is
inherited from the Neural3D release contract rather than re-measured locally.

## Currently declared split

The canonical full-temporal row remains:

- train: `cam04`, `cam09`
- heldout validation only: `cam06`
- anchor/condition: `cam04`
- 300 frames at 30 fps from time zero
- split label: `train2_holdout1`

The browser bundle faithfully serializes only this declared split, sampled at
8 frame indices. It does not establish a separate camera/split contract. The
two additional paper-triplet rows (`cam13`/`cam18` -> `cam00` and
`cam02`/`cam07` -> `cam12`) are separate breadth evaluations, not additions to
the active train set.

## Is all-minus-one legitimate?

**Yes, as a new canonical experiment; no, as an undeclared browser override.**

The loader permits any non-empty, duplicate-free train list disjoint from a
non-empty heldout list. The paper contract permits arbitrary train-camera
count, and the current paper adapter specifically supports exactly one
heldout camera. Consequently a `train17_holdout1` row using every local camera
except (for example) `cam06` is structurally legitimate.

To remain contract-correct it must be introduced as a new manifest row and
protocol, with `cam06` excluded from optimization and initialization. The
browser exporter may then serialize that row. It must not replace or be
reported as the existing `train2_holdout1` baseline: 17 surrounding views make
the single heldout view a much easier interpolation test. Retain the train2
split for comparison and label all-minus-one as a camera-breadth ablation.

## Recommended cameras per optimizer step

Start with **K=4 distinct train cameras at one shared timestamp per optimizer
step**, sampled without replacement and rotated coverage-exactly across the 17
train views. Under the existing paper sampler this corresponds to
`frames_per_step=4`, `same_time_count=4`, and `local_time_count=0` for the first
clean breadth baseline. One train17 epoch is 5,100 camera-frame samples, or
1,275 full K=4 batches.

Why K=4:

- it doubles same-time geometric constraints relative to the current two-view
  protocol without making every step scale with all 17 cameras;
- it samples enough separated baselines to reduce two-view ambiguity;
- camera coverage comes from rotation across steps, not an expensive 17-view
  optimizer step;
- it fits the current four-target-frame budget, enabling a meaningful first
  throughput comparison.

After that baseline, a temporal-mix ablation can use six target samples per
step: four same-time cameras, one nearby-time sample, and one globally sampled
camera-time pair. Do not call that K=6 cameras; its camera breadth remains K=4.

## Data and bundle implications

Measured local storage:

- all 18 source MP4s: 1,186,186,520 bytes (1,131.24 MiB)
- the current `cam04`/`cam09`/`cam06` MP4 subset: 192,390,566 bytes
- current browser JSON plus three 8-frame 96x72 PNG atlases: 366,476 bytes
- current three PNG atlases alone: 275,416 bytes

Linear image-payload projections from the current 8-frame bundle:

- 18 cameras x 8 frames at 96x72: about 1.65 MB of compressed PNG atlases
- 18 cameras x 300 frames at 96x72: about 62.0 MB compressed, content-dependent
- decoded RGBA for 18 x 8 x 96 x 72: 3,981,312 bytes
- decoded RGBA for 18 x 300 x 96 x 72: 149,299,200 bytes (142.38 MiB)

The current exporter concatenates every camera's frames horizontally into one
PNG. At 300 frames and 96-pixel width that becomes a 28,800-pixel-wide texture,
which exceeds portable WebGPU texture-dimension limits. A full-temporal browser
bundle would therefore need chunked atlases, texture arrays, or streamed decode;
simply exporting the current atlas format is not viable.

The Python loader is also eager across every selected camera. Approximate frame
tensor storage for all 18 views x 300 frames in float32 RGB is:

| Decode size | Frame tensor storage |
| --- | ---: |
| 96x72 | 0.42 GiB |
| 128x96 | 0.74 GiB |
| 256x192 | 2.97 GiB |
| 512x384 | 11.87 GiB |
| native 2704x2028 | 330.94 GiB |

At 512x384, the 17 train-camera tensors alone are about 11.21 GiB before
heldout frames, ray grids, model state, gradients, or optimizer state. Thus
all-minus-one is valid semantically but should use lazy/on-demand K-camera,
K-frame decode and bounded caching. Expanding the eager browser/Python bundle
first would spend memory and download bandwidth without increasing per-step
supervision.

## Recommendation

Keep the current train2/holdout1 browser demo as the comparable prototype.
Create a separate canonical train17/holdout1 breadth row only when the runtime
can sample **K=4** cameras lazily. Use all 17 train cameras over an epoch, not
all 17 in every optimizer step, and keep `cam06` validation-only throughout
data loading, initialization, training, and metric computation.
