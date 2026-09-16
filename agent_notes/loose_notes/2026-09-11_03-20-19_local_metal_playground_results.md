# Local Metal playground: actual runs and a negative compiler check

The user cleared Hearthstone and authorized cleaning GPU contention, running
the ablations, and shrinking the baseline for roughly 8 GB RAM / 16 GB disk.
After the earlier Chrome cleanup, GPU utilization was zero. Existing desktop
swap remained around 7.5 GiB; the full publication gate required <=2 GiB swap
and >=10 GiB available, so even a tiny job could not satisfy an 8-GB-host goal.

Added an explicit small protocol to the existing runner rather than changing
the full paper thresholds. The local contract uses 3 GiB for the process tree
plus observer, caps the Torch MPS allocator at 2 GiB, reserves 2 GiB for the OS,
and checks available memory, new swap growth (256 MiB), output size (2 GiB),
and installed dependencies/data plus reserved outputs (16 GiB). RSS is polled
at 250 ms, host/output state at 1 s; these are sampled measurements, not a
physical-memory proof. Descendants and launcher-owned W&B processes are charged.
Final W&B file growth is checked before emitting the completed summary.

Two real runtime defects appeared despite CPU preflight success:

- STAR's TORCH_LIBRARY-only `_C` has no `PyInit__C`. Provenance now loads that
  operator library with `torch.ops.load_library`, while unrelated import
  failures still propagate. No shader or binary rebuild was necessary.
- WorldFoam finished training but its residency validator indexed
  `init_video_chunk_frames` in the unnormalized base config. It now consumes
  the retained resolved config after checking its runner-owned binding.

All three two-step Metal smokes completed. The subsequent real 80-step run uses
32 frames, corrected LLFF v2 calibration, train cam04/cam09, heldout cam06,
128-to-256 primitives, 48x64 then 96x128 images, and the same 160 target frames /
1,597,440 target pixels in every lane. No heldout images train the models.

| Lane | Heldout PSNR | SSIM | L1 | Sampled tree/launcher RSS | Offline W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| World Tubes | 6.504495 | 0.014809 | 0.417118 | 1.56 GiB | paec86cd509ed7 |
| Dynamic 3DGS | 5.961154 | 0.007357 | 0.456643 | 1.25 GiB | pac019dcd46dc7 |
| WorldFoam | 6.643607 | 0.020026 | 0.400195 | 1.75 GiB | pf0ec2d4e6f9de |

Largest reported driver allocation was 1,158,152,192 bytes (WorldFoam); every
lane's swap-growth receipt is zero. Selected installed runtime/code/whole-scene
paths occupy 5,074,723,791 bytes (~4.73 GiB); final run outputs including offline
W&B are ~11.2 MB. This was tested on the user's 24-GiB Mac under smaller caps,
not on physical 8-GiB hardware. Old research outputs, unrelated data, Git
history, and download caches are excluded from the installation measurement.

World Tubes and dynamic 3DGS report train wall 66.57 / 66.98 s. WorldFoam reports
71.16 s accumulated updates but 118.45 s on the broader wall timer that includes
final evaluation; do not silently treat these scopes as identical kernel timing.
The World Tubes preview is mostly dark with sparse blobs: completing the small
run does not establish good reconstruction. Parameter counts and initialization
models also differ despite matched sample budgets.

The frozen learned-world F=4 smoke is **negative** under the unchanged gates:
max image error 4.69178e-4 (limit 1e-5), global normalized VJP error 5.27662e-4,
max per-parameter error 3.82485e-3 (limits 1e-5), fallback fraction 0.38889
(limit 0.20). Gradient cosine is 0.99999972 and mean image error is tiny, but
those do not replace the explicit failed checks. Cold compiled forward was
12.606 s including 11.500 s compilation, versus replay 0.07339 s; compiled
backward 12.641 s versus replay 0.02779 s. This is one un-warmed time count, not
a fitted complexity exponent or accepted sublinear result. Diagnose parity and
fallback behavior before launching a larger compiler sweep.

Artifacts: `outputs/benchmarks/2026-09-11_local_playground/` contains the small
run and its sibling smoke, raw comparisons, checkpoints, previews, memory
receipts, summaries, and W&B identities. Reproduction is documented in the
paper `REPRODUCE.md`. The final focused gate passed 65 tests. Source remains
dirty; publication acceptance counts and `BASELINES.md` standings are unchanged.
Automatic approval review rejected online W&B upload for lack of explicit
approval to export media/configuration. Runs remained offline with logging on;
no approval workaround or online synchronization was attempted.
