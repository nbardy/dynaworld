import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
	combineDatasetSharingTelemetry,
	datasetSharingCapability,
	hydrateDatasetSharedViews,
	prepareDatasetForWorkerSharing,
} from "../datasetSharing.js";
import {
	BACKGROUND_BANK_FORMAT_RGBA32_FLOAT,
	FRAME_BANK_FORMAT_RGBA8,
} from "../dataset.js";

function makeDataset() {
	const width = 2;
	const height = 1;
	const frameCount = 2;
	const viewCount = 2;
	const frameValuesPerView = width * height * frameCount * 4;
	const backgroundValuesPerView = width * height * 4;
	const frames = new Float32Array(frameValuesPerView * viewCount);
	const backgrounds = new Float32Array(backgroundValuesPerView * viewCount);
	for (let index = 0; index < frames.length; index += 1) frames[index] = index / 13;
	for (let index = 0; index < backgrounds.length; index += 1) backgrounds[index] = index / 7;
	const cameras = Array.from({ length: viewCount }, (_, view) => ({
		name: `cam0${view}`,
		role: view === 0 ? "train" : "heldout",
		intrinsics: new Float32Array([1, 1, 0.5, 0.5]),
		worldToCamera: new Float32Array([1, 0, 0, view, 0, 1, 0, 0, 0, 0, 1, 1]),
	}));
	const dataset = {
		width,
		height,
		frameCount,
		viewCount,
		trainViewCount: 1,
		heldoutViewIndex: 1,
		frames,
		backgrounds,
		background: backgrounds.subarray(0, backgroundValuesPerView),
		seedPoints: new Float32Array([0, 0, 1, 1, 0, 0]),
		motionSamples: new Uint32Array([1, 2]),
		staticSamples: new Uint32Array([3, 4]),
		cameras,
		comparisonViewIndices: [0, 1],
	};
	dataset.viewDatasets = cameras.map((camera, view) => ({
		label: camera.name,
		viewIndex: view,
		width,
		height,
		frameCount,
		frames: frames.subarray(view * frameValuesPerView, (view + 1) * frameValuesPerView),
		background: backgrounds.subarray(
			view * backgroundValuesPerView,
			(view + 1) * backgroundValuesPerView,
		),
	}));
	dataset.previewViews = dataset.comparisonViewIndices.map((view) => dataset.viewDatasets[view]);
	return dataset;
}

const sharedScope = { crossOriginIsolated: true, SharedArrayBuffer };

test("dataset sharing requires both cross-origin isolation and SharedArrayBuffer", () => {
	assert.deepEqual(datasetSharingCapability(sharedScope), { available: true, reason: null });
	assert.equal(datasetSharingCapability({
		crossOriginIsolated: false,
		SharedArrayBuffer,
	}).available, false);
	assert.equal(datasetSharingCapability({
		crossOriginIsolated: true,
		SharedArrayBuffer: undefined,
	}).available, false);
});

test("decoded Float32 target banks share one backing while aliases keep their contract", () => {
	const dataset = makeDataset();
	const expectedFrames = Array.from(dataset.frames);
	const expectedBackgrounds = Array.from(dataset.backgrounds);
	const originalFrames = dataset.frames;
	const originalBackgrounds = dataset.backgrounds;
	const originalSeedPoints = dataset.seedPoints;
	const prepared = prepareDatasetForWorkerSharing(dataset, sharedScope);

	assert.equal(prepared.dataset, dataset);
	assert.ok(dataset.frames instanceof Float32Array);
	assert.ok(dataset.backgrounds instanceof Float32Array);
	assert.ok(dataset.frames.buffer instanceof SharedArrayBuffer);
	assert.ok(dataset.backgrounds.buffer instanceof SharedArrayBuffer);
	assert.notEqual(dataset.frames, originalFrames);
	assert.notEqual(dataset.backgrounds, originalBackgrounds);
	assert.deepEqual(Array.from(dataset.frames), expectedFrames);
	assert.deepEqual(Array.from(dataset.backgrounds), expectedBackgrounds);
	assert.equal(dataset.background.buffer, dataset.backgrounds.buffer);
	assert.equal(dataset.viewDatasets[1].frames.buffer, dataset.frames.buffer);
	assert.equal(dataset.viewDatasets[1].frames.byteOffset, originalFrames.byteLength / 2);
	assert.equal(dataset.viewDatasets[1].background.buffer, dataset.backgrounds.buffer);
	assert.equal(dataset.previewViews[0], dataset.viewDatasets[0]);
	assert.equal(dataset.seedPoints, originalSeedPoints);
	assert.ok(dataset.seedPoints.buffer instanceof ArrayBuffer);
	assert.ok(dataset.motionSamples.buffer instanceof ArrayBuffer);
	assert.equal(prepared.telemetry.mode, "shared-array-buffer");
	assert.equal(prepared.telemetry.readOnlyBytes, dataset.frames.byteLength + dataset.backgrounds.byteLength);
	assert.equal(prepared.telemetry.estimatedBytesAvoided, prepared.telemetry.readOnlyBytes * 2);
	const sharedFrames = dataset.frames;
	const sharedBackgrounds = dataset.backgrounds;
	prepareDatasetForWorkerSharing(dataset, sharedScope);
	assert.equal(dataset.frames, sharedFrames);
	assert.equal(dataset.backgrounds, sharedBackgrounds);
});

test("compact frames and FP32 backgrounds share mixed typed roots without losing metadata", () => {
	const dataset = makeDataset();
	const compact = Uint8Array.from(dataset.frames, (value, index) =>
		index % 4 === 3 ? 127 : Math.round(Math.min(1, value) * 255));
	dataset.frames = compact;
	dataset.frameBank = { format: FRAME_BANK_FORMAT_RGBA8, data: compact };
	dataset.backgroundBank = {
		format: BACKGROUND_BANK_FORMAT_RGBA32_FLOAT,
		data: dataset.backgrounds,
	};
	const prepared = prepareDatasetForWorkerSharing(dataset, sharedScope);
	assert.ok(dataset.frames instanceof Uint8Array);
	assert.ok(dataset.frames.buffer instanceof SharedArrayBuffer);
	assert.ok(dataset.backgrounds instanceof Float32Array);
	assert.ok(dataset.backgrounds.buffer instanceof SharedArrayBuffer);
	assert.equal(dataset.frameBank.data, dataset.frames);
	assert.equal(dataset.backgroundBank.data, dataset.backgrounds);
	assert.equal(dataset.viewDatasets[1].frameBank.format, FRAME_BANK_FORMAT_RGBA8);
	assert.equal(dataset.viewDatasets[1].frameBank.data.buffer, dataset.frames.buffer);
	assert.equal(dataset.viewDatasets[1].backgroundBank.data.buffer, dataset.backgrounds.buffer);
	assert.equal(prepared.telemetry.frameBankFormat, FRAME_BANK_FORMAT_RGBA8);
	assert.equal(prepared.telemetry.frameBankBytes, dataset.frames.byteLength);
	assert.equal(prepared.telemetry.backgroundBankFormat, BACKGROUND_BANK_FORMAT_RGBA32_FLOAT);
	assert.equal(
		prepared.telemetry.readOnlyBytes,
		dataset.frames.byteLength + dataset.backgrounds.byteLength,
	);

	const worker = hydrateDatasetSharedViews(structuredClone(dataset), sharedScope).dataset;
	assert.ok(worker.frames instanceof Uint8Array);
	worker.frames[0] = 211;
	assert.equal(dataset.frames[0], 211);
	assert.equal(worker.frameBank.data, worker.frames);
	assert.equal(worker.viewDatasets[0].frames, worker.viewDatasets[0].frameBank.data);
});

test("structured-cloned workers observe shared targets but keep mutable state private", () => {
	const main = prepareDatasetForWorkerSharing(makeDataset(), sharedScope).dataset;
	const initialParams = new Float32Array([1, 2, 3, 4]);
	const trainingMessage = structuredClone({ dataset: main, initialParams });
	const training = hydrateDatasetSharedViews(trainingMessage.dataset, sharedScope).dataset;
	const validationMessage = structuredClone({
		dataset: training,
		initialParams: trainingMessage.initialParams,
	});
	const validation = hydrateDatasetSharedViews(validationMessage.dataset, sharedScope).dataset;

	training.frames[0] = 0.875;
	assert.equal(main.frames[0], 0.875);
	assert.equal(validation.frames[0], 0.875);
	training.backgrounds[0] = 0.625;
	assert.equal(main.backgrounds[0], 0.625);
	assert.equal(validation.backgrounds[0], 0.625);

	training.seedPoints[0] = 99;
	training.cameras[0].worldToCamera[0] = 88;
	training.motionSamples[0] = 77;
	trainingMessage.initialParams[0] = 66;
	assert.notEqual(main.seedPoints[0], 99);
	assert.notEqual(validation.seedPoints[0], 99);
	assert.notEqual(main.cameras[0].worldToCamera[0], 88);
	assert.notEqual(validation.cameras[0].worldToCamera[0], 88);
	assert.notEqual(main.motionSamples[0], 77);
	assert.notEqual(validation.motionSamples[0], 77);
	assert.notEqual(initialParams[0], 66);
	assert.notEqual(validationMessage.initialParams[0], 66);
});

test("fallback preserves the current structured-clone behavior", () => {
	const dataset = makeDataset();
	const originalFrames = dataset.frames;
	const prepared = prepareDatasetForWorkerSharing(dataset, {
		crossOriginIsolated: false,
		SharedArrayBuffer,
	});
	assert.equal(dataset.frames, originalFrames);
	assert.ok(dataset.frames.buffer instanceof ArrayBuffer);
	assert.equal(prepared.telemetry.mode, "structured-clone");
	assert.equal(prepared.telemetry.estimatedCopiesAvoided, 0);
	assert.match(prepared.telemetry.reason, /Cross-origin isolation/);

	const worker = structuredClone(dataset);
	worker.frames[0] = 123;
	assert.notEqual(dataset.frames[0], 123);
});

test("combined telemetry only claims sharing when all three contexts confirm it", () => {
	const main = prepareDatasetForWorkerSharing(makeDataset(), sharedScope).telemetry;
	const workerDataset = prepareDatasetForWorkerSharing(makeDataset(), sharedScope).dataset;
	const training = hydrateDatasetSharedViews(structuredClone(workerDataset), sharedScope).telemetry;
	const validation = hydrateDatasetSharedViews(structuredClone(workerDataset), sharedScope).telemetry;
	const shared = combineDatasetSharingTelemetry(main, training, validation);
	assert.equal(shared.mode, "shared-array-buffer");
	assert.deepEqual(shared.contexts, {
		main: "shared-array-buffer",
		training: "shared-array-buffer",
		validation: "shared-array-buffer",
	});
	assert.equal(shared.estimatedBytesAvoided, shared.readOnlyBytes * 2);

	const fallback = combineDatasetSharingTelemetry(
		main,
		training,
		{ ...validation, mode: "structured-clone", sharedBytes: 0 },
	);
	assert.equal(fallback.mode, "structured-clone");
	assert.equal(fallback.estimatedBytesAvoided, 0);
});

test("main, training, and validation worker boundaries all use the sharing adapter", async () => {
	const [clientSource, trainingSource, validationSource] = await Promise.all([
		readFile(new URL("../nonblockingTrainerClient.js", import.meta.url), "utf8"),
		readFile(new URL("../trainingWorker.js", import.meta.url), "utf8"),
		readFile(new URL("../validationWorker.js", import.meta.url), "utf8"),
	]);
	assert.match(clientSource, /prepareDatasetForWorkerSharing\(dataset\)/);
	assert.match(clientSource, /datasetSharing:\s*sharedDataset\.telemetry/);
	assert.match(trainingSource, /hydrateDatasetSharedViews\(message\.dataset\)/);
	assert.match(trainingSource, /await initializeValidationWorker\(trainer\.dataset,\s*trainer\.initialParams\)/);
	assert.match(trainingSource, /combineDatasetSharingTelemetry/);
	assert.match(validationSource, /hydrateDatasetSharedViews\(data\.dataset\)/);
	assert.match(validationSource, /initialParams\s*=\s*new Float32Array\(data\.initialParams\)/);
	assert.match(validationSource, /previousParams\s*=\s*initialParams\.slice\(\)/);
});
