# Human Motion & Action Datasets

Not strictly "scene renders" but adjacent — these are the realistic
alternatives to re-rendering humans yourself. Most are either pre-rendered
(BEDLAM, AGORA, SURREAL) or real-captured (AIST++, Hi4D, FineGym).

For DynaWorld, these matter because action-cam content usually has a human
subject, and the model needs to handle articulated motion + cloth dynamics +
self-occlusion. Pure scene-render datasets miss this.

## Human-motion-with-GT (rendered or hybrid)

### BEDLAM (Max Planck, 2023) — top recommendation
- **What**: 10K videos of clothed humans, rendered in **Unreal Engine**.
- **GT**: SMPL-X body + camera + depth + per-frame.
- **Why it's the standout**: Unreal-rendered means *not* Kubric-sterile. Cloth
  simulation, varied skin/hair, realistic environments. As close to "real
  video distribution" as synthetic gets for humans.
- **Use for DynaWorld**: data augmentation when the model needs articulated
  human motion. Don't use as primary `video <=> video` GT (still synthetic),
  but it's the best free option if synthetic human motion is needed.
- **Caveat**: still synthetic, still has Unreal rendering artifacts (skin
  shading, motion smoothness). Use alongside real video, not as a substitute.

### AGORA — 14K photoreal human images, pose/shape labels
- Static images, not video. Useful as reference distribution.

### SURREAL (older, still cited) — synthetic humans on real backgrounds
- Pre-BEDLAM gold standard. Lower quality but still in some pipelines.

### Hi4D — interacting humans, scanned + rigged
- Two-person interaction is rare in datasets. Useful for physical-contact
  scenarios.

## Real-captured multi-camera (DynaWorld already uses)

These overlap with `../DATASET_V1.md` §AIST/DeepView/Neural3D/ViVo. Listed
here for completeness — synthetic supplements, doesn't replace.

### AIST++ — extension of AIST Dance DB
- Same source videos as AIST Dance DB, plus 3D pose ground truth from SMPL.
- Already in DynaWorld's `multicam_val_v1` set.

### MoVi — annotated everyday motion
- ~9k motions, multi-view real video + IMU + mocap + SMPL.
- Smaller than BEDLAM but real.

### AMASS — unified mocap aggregator
- Combines CMU MoCap, KIT, ACCAD, others into a single SMPL-format archive.
- The right replacement for the dead CMU MoCap URLs that v1 was trying to
  reach.
- ~11K motions. Free for research.

## Sports / action specifically

For action-cam content (Nova's domain):

### SkatingVerse — figure skating video
- Hand-curated, high-quality skating clips.

### FineGym — fine-grained gymnastics
- Hierarchical action labels (event → set → element). Real gymnastics video.

### SportsMoT — multi-object tracking in sports
- Multi-person tracking with ID consistency. Real broadcast video.

### AIST++ (dance) and AIST Dance DB (multi-camera)
- Already covered. Closest large-scale multi-camera GT for human action.

### Something-Something V2
- Generic action video. Less curated for sports specifically but huge.

## Comparison table

| Dataset | Modality | Source | GT | Best for |
|---|---|---|---|---|
| BEDLAM | rendered video | Unreal | SMPL-X + cam + depth | synthetic human motion w/ realism |
| AGORA | rendered images | mixed | SMPL pose/shape | static pose reference |
| SURREAL | rendered video | older | SMPL | pre-BEDLAM legacy |
| Hi4D | scanned + rendered | studio | per-scan | two-person interaction |
| AMASS | mocap | aggregated | SMPL | source motion (replace CMU) |
| AIST++ / AIST DB | real video + mocap | studio multi-cam | 3D pose + cams | multi-camera GT for dance |
| MoVi | real multi-view | studio | mocap + SMPL | smaller real-multi-view |
| SkatingVerse | real video | curated | labels | skating |
| FineGym | real video | broadcast | hierarchical labels | gymnastics |
| SportsMoT | real video | broadcast | tracking | sports tracking |
| SS-V2 | real video | crowdsourced | action labels | generic action |

## Recommended use within DynaWorld

**Primary GT** (matches `DATASET_V1.md` contract):
- AIST Dance DB, DeepView Video, Neural 3D Video, ViVo. **Real multi-camera,
  no synthetic.** Already in `multicam_val_v1`.

**Augmentation / pretraining pressure** (synthetic, used carefully):
- BEDLAM for articulated human motion variety.
- AMASS for mocap source if v1 retargeting ever gets unblocked.
- BlenderProc-rendered Blender Foundation scenes for non-human variety.

**Don't use as `video <=> video` GT**:
- BEDLAM, AGORA, SURREAL — these are synthetic. The contract bans synthetic
  GT. Use them for probes and pretraining, not as the loss-of-record source.

## For Nova specifically (separate from DynaWorld)

Nova's bullet-time / deblur / stabilization features are an *application*,
not a world-model research project. The same contract doesn't apply. Nova
can train on synthetic data freely.

For Nova:
- BEDLAM is probably the single best free human-motion training set.
- Mixamo + Polyhaven + v2 Blender pipeline for compose-your-own.
- AIST Dance DB videos for multi-camera reference.
