# Title preference, gauge scope, and local Metal target

The user prefers retaining “sublinear” in the paper title and confirms that
the intended training backend is local Metal. Treat this as authorization for
local Metal training within the existing host-resource gates; do not ask for
the same backend permission again. It does not waive those gates or authorize
remote compute.

A suitable working title direction is “World Tubes: Sublinear Temporal
Overhead in Differentiable Gaussian Rendering.” The abstract must define the
overhead as reusable world-side geometry/metadata/adjoint preparation, separate
from the unavoidable dense pixel/output work. The final title remains
conditional on demonstrating the claimed scaling for the actual executed
path. No canonical manuscript title was changed in this clarification turn.

Useful mathematics remains camera-relative coordinates, changes of ray
parameter with the physical Jacobian, Gaussian marginalization, reuse domains,
and the compiler VJP. Full gauge-theory terminology is not necessary to
implement these constructions, and the reviewed Pro results establish no
speed or simplification advantage from introducing connections or curvature.

The four previously executed CPU scripts were downloaded float64 NumPy/SciPy
equation and finite-difference diagnostics. They were not training, did not
replace the Metal backend, and did not produce performance evidence. The
new Pro primitives are not implemented Metal training backends.

## One current resource probe

Read the current canonical Paper-A runner and called its existing
`live_resource_snapshot()` and `require_live_resources()` once with no Torch
import or GPU initialization. Results on 2026-09-10:

- available memory: 6,653,820,928 bytes (about 6.20 GiB), below 10 GiB;
- swap: unavailable (`CalledProcessError` from the system probe), not an
  observed large swap allocation; the runner correctly uses a rejection
  sentinel for unknown pressure;
- load per logical CPU: 0.4034, below the 0.75 threshold;
- free disk: 31,793,164,288 bytes, above this runner's 8 GiB threshold.

The existing guard rejected full local MPS paper execution for available
memory and unknown swap. No training, build, repeated resource polling,
new verifier, or runtime evidence was produced. The known source/correctness
requirements remain separate from host readiness.

Volta's read-only route check located the existing
`coffee_martini_protocol_smoke_2step.jsonc` protocol for the frozen-world
executor: two real optimizer steps, 128 primitives at 48x64 followed by 256
at 96x128, then frozen replay/compiled comparison on MPS. The static route
uses `project_world_tube_sequence`, so the audited dynamic-first-order temporal
envelope reset should not block this static smoke. The frozen comparison also
does not exercise the optimizer's stale support-cache reuse pattern. These
route distinctions prevent treating every earlier audit defect as a blocker
for every training path. No smoke was launched because the host gate failed.

## Follow-up: user requested freeing the machine and running

The user subsequently asked to free memory/GPU and run the ablations. Process
inspection found no obvious World Tubes training process. Hearthstone was live
(PID 58086), and the GPU driver's last-submitting PID matched it. Chrome had
many resident processes. Reading `vm.swapusage` with approved sandbox escalation
succeeded: 12,384.44 MiB used (about 12.09 GiB), versus the runner's 2 GiB cap.
The earlier unknown-swap blocker is now resolved into observed high swap.

Requested specific confirmation before quitting Hearthstone and Chrome because
doing so could interrupt a live match or unsaved browser forms. General local
Metal authorization remains in force. No application was terminated while
that answer was pending.

Volta verified the existing sibling `dynaworld-paper-freeze` checkout is clean
at main `d5b0db5` and STAR `6c99452`, with the native CPython-3.11 extension,
runtime, smoke protocol, and Coffee assets present. It can run the small
four-frame smoke after the host clears. Its eager loader and missing recent
RSS/streaming protections exclude it from the full 300-frame experiment.
Current-source paper runs additionally need selective preservation in a clean
source snapshot; unrelated browser/PowerFoam edits must remain untouched.

The user deferred closing Hearthstone/Chrome until later; do not interpret the
cleanup authorization as a request to close them now or schedule an automatic
run. A subsequent read-only GPU-client inspection found Chrome, WindowServer,
ChatGPT/Codex, Claude, Slack, QuickTime, video decode/capture services, game
launchers, and other desktop apps registered. A driver snapshot showed 70%
overall utilization and Hearthstone as the last submitting PID; this does not
attribute that entire percentage to Hearthstone. No Python training client
appeared in the GPU-client list. Registered clients are not evidence of heavy
active computation. No applications were closed and no training was started.
