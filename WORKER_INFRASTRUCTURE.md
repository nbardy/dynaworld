# Nonblocking Browser Trainer Integration

This is prototype infrastructure around the existing browser WGSL trainer. It does not define a new
model, renderer lane, calibration format, or Python trainer contract.

## Run With Cross-Origin Isolation

From the repository root:

```bash
python3 web/dynaworld_browser_trainer/serve_isolated.py --port 8080
```

The server adds COOP/COEP headers, making `crossOriginIsolated` and `SharedArrayBuffer` available on
supporting browsers. Without those headers, the client automatically falls back to throttled status
messages. Optimization still runs in the dedicated worker.

## Integration API

```js
import { createNonblockingTrainer } from "./nonblockingTrainerClient.js";

const client = createNonblockingTrainer();
await client.init({
	dataset,
	canvas: document.querySelector("#renderCanvas"),
	trainerOptions: { splatCount: 768 },
	trainOptions: { learningRate: 1.25, samplesPerStep: 96, camerasPerStep: 4 },
	renderOptions: { viewIndices: [0, 9, 6] },
	schedule: { burstSteps: 8, metricEvery: 256, validationEvery: 2048, renderFps: 20 },
});

client.start();
const status = client.getStatus(); // Atomic SAB read when isolated; message snapshot otherwise.
client.setTrainOptions({ learningRate: 1.0 });
client.setRenderOptions({ time: 0.5, renderMode: 0 });
client.resize(1200, 300);
client.requestValidation({ gridSize: 12 });
client.pause();
client.dispose();
```

Listen for `ready`, `metrics`, `validation`, `status`, `capability`, and `error` events. Event payloads
are in `event.detail`. The `ready` payload reports three capabilities explicitly:

- `sharedStatus`: atomic status is active because the page is cross-origin isolated.
- `offscreenRender`: the canvas transferred successfully; rendering is worker-owned.
- `validationWorker`: copied parameter snapshots are evaluated in a separate CPU worker.

If `offscreenRender` is false, the optimizer remains worker-owned but this slice does not attempt to
share its `GPUDevice` or live parameter buffers with a main-thread renderer. The UI should report the
capability and omit live result rendering until a copied render-snapshot fallback is added.

## Scheduling Contract

`trainStep()` is never awaited. Metric and parameter readbacks are launched as single-flight promises,
and the pump continues submitting work. Validation receives an owned CPU copy of a point-in-time
parameter buffer and cannot stall the optimizer worker. The displayed throughput comes from a
nonblocking `GPUQueue.onSubmittedWorkDone()` probe, so it measures completed GPU work rather than CPU
command-enqueue speed.

The SharedArrayBuffer uses a small atomic seqlock defined in `workerProtocol.js`. It publishes trainer
state, submitted step, completed steps/s, latest loss/PSNR/SSIM, pending flags, and the steps associated
with the latest metric and validation snapshots. It also publishes the active camera count, rotation
offset, and total train-camera count.

## Camera Sampling Contract

When a dataset has more than four train cameras, the worker defaults to `camerasPerStep: 4`. Step `s`
uses a circular window beginning at `(s * K) % trainViewCount`, so every declared train camera is
covered. Camera roles, rather than array position, define membership; heldout cameras cannot enter a
training window. `K >= trainViewCount` retains the original all-camera sampling path for legacy
contiguous datasets.

Motion and static rays keep their existing per-ray frame sampling. Focused samples are grouped into
per-camera ranges once at initialization, so each ray selects directly from its assigned active camera
without rejection scans. When that camera has no sample in the requested focus class, the ray remains
a uniform sample from the same active camera. This is rotating K-camera membership, not a same-time
camera group. Implementing synchronized K-camera patches would require a distinct batch and loss contract.

Rendering accepts `renderOptions.viewIndices`. Without an explicit list, role-aware datasets render a
representative first train camera, a middle train camera, and `heldoutViewIndex`. The historical first
three camera fallback is used only when roles and a heldout index do not define that triptych.
