# Paper lane process isolation after the MPS memory incident

The 2026-07-22 fixed-512 matrix attempt was killed by the operator after the
workstation entered severe unified-memory compression and swap pressure. No
MPS work was launched in this follow-up session. All validation described
below was CPU-only source parsing, dry-run construction, or unit testing.

## Change

- `multicam_heldout_compare.py` can now run only `world_tubes` or only
  `dynamic_3dgs`; each lane writes its normal report and exits.
- The unified runner launches those two lanes into separate child directories,
  reuses a completed child report after interruption, rejects protocol/seed/
  camera metadata drift, and forms the legacy combined report contract only
  after both lane reports exist.
- WorldFoam moved behind `run_worldfoam_paper_lane.py`, so the parent no longer
  imports or runs its trainer in-process and its allocator dies with the child.
- The selected device is propagated into WorldFoam instead of silently keeping
  the base config's `mps` device.
- Both STAR paper children and the WorldFoam child independently fail closed
  on local MPS unless the guarded parent propagates explicit authorization.
- The old incident-calibrated combined eager estimate is intentionally retained
  as a conservative launch guard. Process isolation is not treated as evidence
  that the full 512 workload is safe.

## Verification

- Python parsing succeeded for the unified runner, WorldFoam child, STAR
  comparison entrypoint, and focused tests.
- `tests/test_unified_paper_ablation.py` plus
  `tests/test_unified_paper_matrix.py`: `19 passed`.
- A unified-runner dry run produced three isolated child commands and expected
  lane-report paths without initializing MPS or loading video frames.

## Remaining boundary

Do not run another publication-scale local MPS row. The next safe engineering
step is streaming targets/rays/evaluation (or moving execution to a larger
machine). The three already complete progressive Coffee Martini seeds remain
valid; the interrupted fixed row does not.

## Existing-evidence packaging

After the isolation commit, the matrix runner gained a no-execution
`--aggregate-existing` mode. It accepts only complete `run_summary.json`
artifacts with clean superproject/STAR provenance and matching seed, protocol,
backward policy, and WorldFoam initializer. Running it on the interrupted
submission directory accepted the three progressive seeds, rejected the four
missing controls by absence, and produced 9 validated lane rows plus
JSON/CSV/Markdown/LaTeX/SVG outputs. Those accepted aggregates were appended
to `BASELINES.md` and inserted into the manuscript as an explicitly partial
public table; no MPS process was launched.
