# Static datasets unification: DeepView + Ex4DGS + Blender/Sintel + multicam_val pass

Date: 2026-04-28
Topic: bringing the four "static" upstream datasets I own (DeepView Video raw,
Ex4DGS pretrained checkpoints, the Blender/Sintel synthetic-render bundle, and
the multicam_val orchestrator that consumes DeepView/N3D/AIST/ViVo) onto the
same `download / extract / inspect / validate-local / cleanup-zips` contract
that the YouTube and Neural 3D Video pipelines already use, and writing the
disk-reclaim story down so future agents can act on it without re-deriving the
"safe vs. NOT safe to delete" math.

## Why this work was needed

Disk on this laptop is finite and three of the four datasets had grown raw
artifacts that were strictly redundant with the extracted media:

- DeepView raw: `03_Dog.zip` (1.3 GB) + `15_Branches.zip` (171 MB) ~= 1.5 GB
  redundant with `extracted/<scene>/`.
- Ex4DGS raw: 3 zips totalling ~318 MB redundant with the `extracted/` bundle
  trees.
- Blender/Sintel: 17 GB, but most of it is `Sintel_DATA.iso` which is
  precious -- per agent memory the original Durian SVN is dead and the
  archive.org DVD set is the only free public source. Auto-deleting it would
  be unrecoverable. The other 9.7 GB IS reclaimable but most of it is
  opt-in-only (PAL ISO, extracted Blender app, scratch renders).

The Neural 3D Video agent already established the precedent
(`cleanup-zips` stage on the pipeline + non-interactive shell wrapper +
default dry-run + opt-in `--execute`). I copied that contract onto DeepView
and Ex4DGS verbatim, and adapted it for Blender/Sintel where the safe set is
not "everything that has been extracted" but a hard-coded list of
re-fetchable artifacts.

## What changed

1. `src/dataset_pipeline/deepview_video.py`
   - Added `cleanup-zips` stage. A zip is reclaimable iff its corresponding
     `extracted/<scene>/` tree has a populated `models.json` (the canonical
     DeepView per-camera calibration file, written last by the official
     archive layout). Counting raw files would let us reclaim a zip whose
     extraction crashed mid-way; checking `models.json` is the strongest
     single-signal we have without re-running the inventory pass.
   - Wired `--execute` flag. Default is dry-run.
   - The new stage is NOT included in `all` -- destructive stages must be
     explicitly invoked.

2. `src/dataset_pipeline/ex4dgs_pretrained.py`
   - Added `cleanup-zips` and `validate-local` stages.
   - `cleanup-zips` reclaim rule: extracted bundle has both `cameras.json` and
     `mean_metrics.json`. This matches what the existing `inspect` stage
     already exercises so we never delete a bundle the inspect pass would
     flag as broken.
   - `validate-local` checks each configured asset has those metadata files
     and at least one `.ply` point cloud. Deliberately does NOT load weights
     into a torch model; this is the "did the extraction actually finish" gate,
     not "do the weights compile".

3. `src/dataset_pipeline/blender_synthetic_inventory.py` (new; named to avoid
   colliding with the existing `src/dataset_pipeline/blender_synthetic/`
   render-scripts package)
   - Three stages: `inspect`, `validate-local`, `cleanup`.
   - The reclaim list is a **hard-coded** `reclaim_candidates(root)` table
     with two flags per entry: present-on-disk, and `safe_to_auto_delete`.
   - `safe_to_auto_delete=True` means `--execute` will delete it. This is
     limited to:
     * `__MACOSX/` cruft
     * `blender-2.79b-macOS.zip` (re-fetchable from download.blender.org)
     * `Sintel.2010.1080p.mkv` (finished movie, not used for rendering)
   - `safe_to_auto_delete=False` means even with `--execute` we skip it
     unless the user also passes `--include-protected`. This covers:
     * extracted Blender app directory (re-extractable from the zip)
     * `sintel/renders/` (locally rendered scratch outputs)
     * `Sintel_PAL.iso` (also on archive.org but not the prod-blend source)
   - `Sintel_DATA.iso` is in `critical_assets()` -- it is NEVER deletable
     through this script. `validate-local` errors out if it's missing or
     truncated (<1 GB indicates partial download, which is worse than
     missing). To delete it the user must `rm` it by hand, deliberately.
   - Reason chose this design over a generic "everything extracted is
     reclaimable" rule: there's no implicit relationship between the ISO and
     a typed extraction tree the way there is for DeepView/Ex4DGS. The ISO
     just sits there as the upstream blob; the production .blend tree on the
     ISO is mounted ad-hoc when rendering. So the cleanup contract has to be
     explicit-list, not implicit-by-extraction-state.

4. New shell scripts (all non-interactive, default dry-run):
   - `src/dataset_scripts/cleanup_deepview_zips.sh`
   - `src/dataset_scripts/cleanup_ex4dgs_zips.sh`
   - `src/dataset_scripts/cleanup_blender_synthetic.sh`
   Each forwards `--execute`, none touch other datasets.

5. `src/dataset_scripts/deepview_video_seed.sh`,
   `src/dataset_scripts/ex4dgs_pretrained_val_seed.sh`
   - Extended argv parsing to accept `--execute` and the new stages (so users
     can call `./src/dataset_scripts/deepview_video_seed.sh cleanup-zips
     --execute` without going through the dedicated cleanup script).

6. `data/REHYDRATE.md`
   - Added cleanup blocks for DeepView and Ex4DGS, sized in MB.
   - Added a brand-new `## Blender Synthetic / Sintel` section documenting
     the upstream-source map, the auto-reclaimable vs. opt-in vs. preserved
     sets, and the validate-local + inspect commands.
   - Updated `## Multi-Camera Validation Set` to mention that
     `multicam_val_v1_seed.sh all` now runs `download-aist-cameras` between
     `download-aist` and `build`.
   - Did NOT touch the AIST, Neural 3D, ViVo, or YouTube sections (other
     agents' scope).

## multicam_val orchestration check

Verified that `./src/dataset_scripts/multicam_val_v1_seed.sh inspect` runs
clean with the existing local data (8 validation pairs across AIST, DeepView,
Neural 3D, ViVo). The new `download-aist-cameras` stage from the AIST agent's
work is reachable through the orchestration script's positional-stage
forwarding without any shell changes -- I verified by running `--help` and
confirming `download-aist-cameras` appears in the choices list. Did not run
`all` (would re-download AIST + AIST cameras over the network) but the wiring
is mechanically correct.

## Disk-size audit

| dir | bytes | MB | reclaimable |
|---|---|---|---|
| `data/external/deepview_video/raw` | 1,578,034,932 | 1578.0 | YES (all 1.5 GB; matches extracted) |
| `data/external/deepview_video/extracted` | 1,577,905,766 | 1577.9 | NO (this is the working set) |
| `data/external/ex4dgs_pretrained` | 715,099,478 | 715.0 | 318 MB of 715 (the raw zips) |
| `data/blender_synthetic/_blender_2_79b` | 585,919,057 | 585.9 | 148 MB auto + 437 MB opt-in |
| `data/blender_synthetic/sintel` | 17,156,274,222 | 17156.2 | 1180 MB auto + 7976 MB opt-in; 8001 MB preserved (DATA ISO) |

## Cleanup dry-run outputs (verbatim totals)

```
DeepView:    TOTAL reclaimable: 1,578,034,932 bytes (1578.0 MB) across 2 archives
Ex4DGS:      TOTAL reclaimable: 317,869,692   bytes (317.9 MB)  across 3 archives
Blender:     TOTAL auto-reclaimable: 1,328,363,835 bytes (1328.4 MB) across 3 entries
Blender (--include-protected): 9,741,623,935 bytes (9741.6 MB)
                               (preserves Sintel_DATA.iso = 8000.6 MB)
```

## Validator outputs

```
ex4dgs validate-local:
  ok  coffee_martini  cameras=5400  ply_files=3  bytes=138473458
  ok  Birthday        cameras=800   ply_files=3  bytes=170964838
  ok  Fabien          cameras=800   ply_files=3  bytes=87765852
  validate-local: 3 bundles look loadable.

blender_synthetic validate-local:
  ok  data/blender_synthetic/sintel/_iso/Sintel_DATA.iso  8000569344 bytes
  ok  data/blender_synthetic/_blender_2_79b/blender-2.79b-macOS-10.6/blender.app
  validate-local: critical assets present.

deepview inspect (existing pass, re-run for sanity):
  03_Dog:      videos=41 models=41 matched=41 projection=['fisheye'] median_duration=5.01s
  15_Branches: videos=45 models=45 matched=45 projection=['fisheye'] median_duration=0.33s
```

## What still defers

- I did not actually run any cleanup with `--execute`; that is the user's
  call. Dry-run output is in this note for review.
- No DeepView camera-adapter changes. The `multicam_video_data.py` adapter
  (which I read but did not modify) loads DeepView records via `models.json`
  / `video_path` / `radial_distortion` straight off the inventory; nothing in
  this pass changes those columns.
- I did not touch any AIST-specific code in `multicam_val.py` or the data
  adapter. The orchestration verification was read-only.
- `blender_synthetic_inventory.py` has no `download` stage. That's intentional --
  archive.org throttles ISO fetches and Blender Studio is subscription-walled,
  so any "download" stage would be misleading. The bundle is rehydrated by
  hand: see the source-index map in `data/REHYDRATE.md`.
