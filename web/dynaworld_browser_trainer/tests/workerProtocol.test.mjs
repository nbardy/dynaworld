import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
	assertProtocolMessage, createSharedStatusBuffer, protocolMessage, publishSharedStatus, readSharedStatus,
	StatusFlag, TrainerState, WorkerCommand, WORKER_PROTOCOL_VERSION,
} from "../workerProtocol.js";

test("shared status publishes a coherent atomic snapshot", () => {
	const scope = { crossOriginIsolated: true, SharedArrayBuffer };
	const buffer = createSharedStatusBuffer(scope);
	publishSharedStatus(new Int32Array(buffer), {
		state: TrainerState.RUNNING,
		step: 1234,
		stepsPerSecond: 731.25,
		loss: 0.0025,
		psnr: 26.02,
		ssim: 0.8125,
		flags: StatusFlag.SHARED_MEMORY | StatusFlag.METRICS_PENDING,
		lastMetricStep: 1024,
		lastValidationStep: 768,
		camerasPerStep: 4,
		cameraRotationStart: 8,
		trainViewCount: 17,
	});
	const status = readSharedStatus(buffer);
	assert.equal(status.version, WORKER_PROTOCOL_VERSION);
	assert.equal(status.state, TrainerState.RUNNING);
	assert.equal(status.step, 1234);
	assert.equal(status.stepsPerSecond, 731.25);
	assert.ok(Math.abs(status.loss - 0.0025) < 1e-7);
	assert.equal(status.lastValidationStep, 768);
	assert.equal(status.camerasPerStep, 4);
	assert.equal(status.cameraRotationStart, 8);
	assert.equal(status.trainViewCount, 17);
});

test("shared status is explicitly unavailable without cross-origin isolation", () => {
	assert.equal(createSharedStatusBuffer({ crossOriginIsolated: false, SharedArrayBuffer }), null);
});

test("protocol rejects mismatched versions", () => {
	assert.equal(assertProtocolMessage(protocolMessage(WorkerCommand.START)).type, WorkerCommand.START);
	assert.throws(() => assertProtocolMessage({ version: 999, type: WorkerCommand.START }), /unsupported/);
});

test("optimizer pump never awaits train steps, metrics, or validation", async () => {
	const source = await readFile(new URL("../trainingWorker.js", import.meta.url), "utf8");
	assert.doesNotMatch(source, /await\s+trainer\.(?:trainStep|readLoss|readParams)/);
	assert.match(source, /new Worker\(new URL\("\.\/validationWorker\.js"/);
	assert.match(source, /setTimeout\(\(\) => pump\(token\), delay\)/);
	assert.match(source,
		/message\.schedule\?\.maxQueuedSteps\s*\?\?\s*backendDescriptor\.defaultSchedule\.maxQueuedSteps/);
	assert.match(source, /queuedSteps\s*>=\s*maxQueuedSteps/);
	assert.match(source, /Math\.min\(burstSteps,\s*maxQueuedSteps\s*-\s*queuedSteps\)/);
});

test("live preview can be disabled without pausing optimization", async () => {
	const [workerSource, appSource] = await Promise.all([
		readFile(new URL("../trainingWorker.js", import.meta.url), "utf8"),
		readFile(new URL("../app.js", import.meta.url), "utf8"),
	]);
	assert.match(workerSource, /if\s*\(!renderOptions\.enabled/);
	assert.match(appSource, /enabled:\s*controls\.live\.checked/);
});

test("SPA exposes packed-FP16 checkpoints and sends the selected precision to the worker", async () => {
	const [htmlSource, appSource] = await Promise.all([
		readFile(new URL("../index.html", import.meta.url), "utf8"),
		readFile(new URL("../app.js", import.meta.url), "utf8"),
	]);
	assert.match(htmlSource, /id="checkpointPrecisionSelect"/);
	assert.ok(htmlSource.indexOf('value="packed-f16"') < htmlSource.indexOf('value="f32"'));
	assert.match(appSource, /checkpointPrecision:\s*controls\.precision\.value/);
	assert.match(appSource, /controls\.precision\.disabled\s*=\s*sampledBackendSelected\(\)/);
	assert.match(appSource, /sampledBackendSelected\(\)\s*\?\s*2048\s*:\s*4096/);
});
