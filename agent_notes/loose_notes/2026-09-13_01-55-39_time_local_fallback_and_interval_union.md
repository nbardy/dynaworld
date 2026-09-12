# Time-local fallback passes the saved-world correctness budget

The preceding user-facing clarification recovered the successful source fits,
but added no new execution evidence. This continuation resumed the pending
compiler defect, completed guarded native checks and saved-world runs, and
committed the STAR change as `1a1fc3e`. It advances a demonstrated blocker;
it does not complete the broad overnight objective or the public paper sweep.

## Cause and correction

The old marker attached a fallback reason to an entire cell interval when any
one sampled time was ambiguous. In the unchanged saved F4 atlas, 188 of 280
fallback tile samples had no near-depth pair at that sample. These are sampled
observations, not a continuous-time visibility certificate.

The marker now records reasons per sample and splits cells into contiguous
runs with equal fallback status/reasons. Existing inherited fallback remains
in force over its original interval. The depth epsilon stays 1e-6; no support,
opacity threshold, depth semantics, or acceptance limit was weakened.

Splitting initially exposed another capacity issue: repeated intervals for the
same trace occupied separate native slots. The pending change alone required
up to 206 fast slots and overflowed 25 tiles at capacity 128. The packer now
unions touching or overlapping intervals for each tile/slab and trace-table
id. Actual gaps and different chart ids remain separate. Native selection
already composites each active trace id once using live depth/id order, so
this union preserves its active set and blend semantics. No shader/ABI or
native build was needed.

On the saved F4 coefficients, native spatial bins now have maxima 63 (all
cells) and 120 (fast cells), with zero overflow at capacity 128. Removing
fallback creates holes in the fast trace intervals, explaining why the fast
maximum exceeds the all-cell maximum. Exact active-set equality and absence
of duplicate active ids were checked for all 768 tile samples in each route,
including empty tiles. There are 4,575 all-cell packed entries and 6,844 fast
packed entries. Stored topology still has 15,085 cell entries and 2,007 cells;
its complexity counters intentionally describe retained topology, not the
coalesced native slot count.

## Real saved-world results

Both runs strict-load the same September 11 checkpoint, with file SHA
`8ca44d076cf17e6aed274f2986a5cb033a7220fa2a1c14a41ca1e589345949f9`
and world SHA
`ccf3d00cfe5463a8b14b2dccfb36ee6e1b2695fdb392cb29e08a7e779e50c64a`.
They use 256 tubes, cam06, 96x128, the same full four-frame physical interval,
peak-splat alpha, capacity128, and the original numerical/fallback limits.
No optimizer step or model modification occurred.

| Selected frames | Before fallback | Now fallback | Max RGB error | Global world VJP error | Max parameter VJP error |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0,1,2,3] | 38.8889% | 12.7778% | 6.8545341e-7 | 7.9251665e-7 | 7.7983507e-7 |
| [0,2,3] | 38.1481% | 17.0370% | 6.8545341e-7 | 9.9963957e-7 | 1.0969077e-6 |

Both local report rows have `accepted=true` and all eight row checks pass.
F3 also passes the independent non-unit selected-time slicing validator:
chunked vs single-frame RGB is exact, global VJP error 5.0660789e-7, maximum
parameter-group error 5.8558217e-7, and every world hash stays unchanged.
The F4 report intentionally does not run this separate proof again.

Existing retained-storage validators accepted the actual binary atlas files.
F4 is 1,040,795 bytes (32,768 tensor payload + 1,008,027 topology/container),
versus 981,498 bytes before this change: localizing fallback increases retained
cell topology even though native packing consumes fewer slots. F3 is 816,117
bytes. Do not call this a serialized-storage reduction.

## Validation and retained artifacts

- CPU-only focused depth/order, correctness, UVT-producer, and binning suite:
  72 passed, 32 skipped (MPS explicitly masked).
- Guarded actual Metal depth/order suite: 14 passed, including interval gaps,
  RGB and coefficient/opacity/color VJPs, centered source depth, slicing,
  inherited fallback, and cached live updates.
- Other native packer consumers: 21 passed, 25 deselected, including interval,
  rolling/exposure, quadratic, and optimizer-step paths.
- All three guarded GPU processes completed. Maximum sampled process tree plus
  launcher RSS was 1,884,192,768 bytes; zero new swap and no resource guard trip.
  The existing 3-GiB RSS, 2-GiB MPS, host reserve, disk, and 600-second limits
  were unchanged. No application termination or network upload occurred.

Results, launch scripts, source hashes, reports, atlas binaries, tests, and
resource receipts live in
`outputs/benchmarks/2026-09-13_time_local_fallback/`.
`verified_comparison.json` binds the two reports to unchanged source hashes,
checkpoint identity, original acceptance limits, retained storage, and resource
receipts. Its checks call the existing storage and selected-time validators;
no new acceptance verifier or schema was introduced. The preceding diagnostic
with the overflowing pre-union result remains under
`outputs/benchmarks/2026-09-13_fallback_distribution/`.
W&B is omitted only for these checkpoint-only mechanical correctness checks;
there was no training or controlled performance benchmark in this chunk.

The older split-window packing test now lowers fitted charts into distinct
coefficient-table ids before asserting their intervals remain separate. The
new regression additionally requires exact sampled membership across temporal
slab sizes 1,2,4, rejects duplicate active ids, and preserves actual gaps.

## Remaining work

These are local dirty-worktree mechanical rows, not the complete clean-source
fixed-interval F={4,8,16,32,64,128,full} sweep, warmed/repeated timing evidence,
or public quality matrix. Paper counts stay 0/7 contexts and 0/21 lane rows;
BASELINES is unchanged. Old successful source fits and the newer 19.69/14.96
dB training/heldout controls remain separate evidence.

The observed F4 diagnostic has 7.48 s compilation, 5.74 s compiled forward and
5.11 s backward, versus 0.059/0.042 s replay. These are single diagnostic
measurements without benchmark warmups/repeats, but clearly do not establish a
speed win. Code inspection identifies a concrete next cost: mixed fallback
calls `render_reference_with_fallback()` on the whole atlas, then copies only
flagged tiles. A sparse fallback must include every contributing trace in each
flagged tile/sample, including unflagged cells, and preserve centered depth,
source-id ties, and world VJPs. Simply rendering the subset of flagged cells
would omit occluders/contributors and be wrong. Profile and restrict this work
before claiming that a lower fallback fraction translates into runtime gains.
