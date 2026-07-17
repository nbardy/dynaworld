# WorldFoam framebitmask i32 offsets and prep bottleneck

Context: continued the WorldFoam Gate4 framebitmask fork after the site8/site24
selector gates had promoted framebitmask over regular factorized. The goal was
to see whether the same idea survives the larger render64/site24 matched-STAR
style smoke instead of only tiny render16 selector ladders.

What changed:

- `owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid`
  now keeps both `track_change_offsets_i32` and `change_offsets_i32` for the
  framebitmask variant. The small render16 path could use int16 offsets, but
  render64/site24 overflowed them:
  - 4f failed on `change_offsets_i16` with max `55,797`.
  - 8f failed on `track_change_offsets_i16` with max `51,154`.
- The regular factorized/framegroup16 path stays on int16 metadata. The wider
  offsets are scoped to framebitmask, where larger track/change prefix counts
  are real data, not a stale-buffer bug.
- Rebuilt `world_foam_lane2_fused_slab_v0` and reran the focused owner-run
  parity/storage tests:

  ```bash
  rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
    research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_framebitmask_removes_dense_frame_table \
    research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays -v
  ```

  Result: both tests passed.

Artifacts:

- Clean-enough 2/4f render64/site24 step timing:
  `research_experiments/world_foam_lane2/results/2026-05-19_worldfoam_framebitmask_render64_site24_i32offsets_2_4_steps8_warm4.json`
  - 2f: total `2.338ms`, backward `2.062ms`, selected segments `34,545`,
    schema bytes `618,842`, track offset max `8,192`, change offset max `9,744`.
  - 4f: total `2.849ms`, backward `2.562ms`, selected segments `81,570`,
    schema bytes `830,876`, track offset max `23,803`, change offset max
    `55,797`.
  - 2x frames cost `1.218x` total and `1.242x` backward.
- Contended 8f correctness smoke:
  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_framebitmask_render64_site24_i32_offsets_8f_contended_correctness.json`
  - status `ok`; total `4.232ms`, backward `3.679ms`, selected segments
    `174,029`, schema bytes `1,198,494`, track offset max `51,154`,
    change offset max `134,340`.
- Contended 16f correctness smoke:
  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_framebitmask_render64_site24_i32_offsets_16f_contended_correctness.json`
  - status `ok`; total `1642.094ms`, backward `1517.562ms`, selected segments
    `352,171`, schema bytes `1,645,290`, track offset max `85,612`,
    change offset max `228,745`.
  - The process was interrupted just after the JSON was written, so the shell
    returned `130`, but the artifact loads cleanly and reports `status=ok`.

Interpretation:

- Correctness now reaches 16f at render64/site24 for the framebitmask path.
- The GPU/VJP step is sublinear in the clean 2->4 evidence and functionally
  unblocked through 8/16f, but the 8/16 timings are contaminated and should not
  be promoted as clean speed.
- The practical blocker shifted to CPU tape prep. The 16f artifact spent
  `221.47s` building endpoint-record sequences, `455.77s` building segment
  tapes, and `137.85s` compacting baseline tapes for train before the single
  GPU step. The shader cannot look competitive end-to-end while the owner-run
  tape is rebuilt this way in Python/slow-owner-run mode.

Next:

1. Do a clean idle 2/4/8/16 render64/site24 ladder only after the machine is
   quiet; do not cite the contended 8/16 timings as speed.
2. Move/cached-build the owner-run tape prep before spending another shader fork
   on local replay micro-optimizations.
3. Re-run the matched STAR UVT comparison once prep is not dominating wall time,
   because STAR's cleanliness came from a first-class time-tubed representation
   rather than repeatedly rebuilding per-frame endpoint rows.
