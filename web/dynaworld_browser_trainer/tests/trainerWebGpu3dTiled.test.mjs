import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
	DynamicSplatWebGpu3dTiledTrainer,
	DEFAULT_CHECKPOINT_PRECISION,
	DEFAULT_MAX_TILE_CAPACITY,
	DEFAULT_STATIC_WARMUP_STEPS,
	densityDispatchesForStep,
	ellipseIntersectsRect,
	fullFramePairForStep,
	MAX_SCALE_ASPECT_RATIO,
	MAX_WORKGROUPS_PER_DIMENSION,
	opacityAwarePixelBounds,
	packedTrainingBackgroundForStep,
	ROTATION_LR_FROM_MOTION,
	resolveCheckpointLayout,
	resolveCheckpointPrecision,
	resolvePairDispatch,
	resolveSsimRadius,
	resolveStaticWarmupSteps,
	resolveTileCapacity,
	resolveTiledCapacity,
	SCALE_LR_FROM_COLOR,
	trainingBackgroundForStep,
	trainingPairForStep,
	windowedL1DssimCpu,
} from "../trainerWebGpu3dTiled.js";
import { computeMultiviewSamples, normalizedMotionLossWeights } from "../dataset.js";
import { canonicalGaussianSsim } from "../snapshotMetrics.js";
import { FILTER_SIGMA_PIXELS, MAX_SPLAT_COLOR } from "../trainerWebGpu3d.js";

const source = readFileSync(new URL("../trainerWebGpu3dTiled.js", import.meta.url), "utf8");

function assertClose(actual, expected, tolerance = 1e-7) {
	const scale = Math.max(1, Math.abs(actual), Math.abs(expected));
	assert.ok(Math.abs(actual - expected) <= tolerance * scale,
		`expected ${actual} to be within ${tolerance} relative of ${expected}`);
}

test("tiled capacity reserves growth while respecting explicit bounds", () => {
	assert.equal(resolveTiledCapacity(8), 24);
	assert.equal(resolveTiledCapacity(768), 2304);
	assert.equal(resolveTiledCapacity(768, 512), 768);
	assert.equal(resolveTiledCapacity(768, 1000.9), 1000);
	assert.equal(resolveTiledCapacity(1024, 4096), 4096);
	assert.equal(resolveTiledCapacity(2048, 8192), 4096);
	assert.throws(() => resolveTiledCapacity(7), /at least 8/);
	assert.throws(() => resolveTiledCapacity(8.5), /integer/);
});

test("tile capacity covers every splat so no valid tile can overflow", () => {
	assert.equal(DEFAULT_MAX_TILE_CAPACITY, 4096);
	assert.equal(resolveTileCapacity(768), 1024);
	assert.equal(resolveTileCapacity(1536), 2048);
	assert.equal(resolveTileCapacity(4096), 4096);
	assert.equal(resolveTileCapacity(4096, 4096), 4096);
	assert.throws(() => resolveTileCapacity(4096, 2048), /cover all 4096 splats/);
	assert.throws(() => resolveTileCapacity(4096, 4097), /at most 4096/);
});

test("active-pair indirect dispatch spans two dimensions within WebGPU limits", () => {
	assert.equal(MAX_WORKGROUPS_PER_DIMENSION, 65535);
	assert.deepEqual(resolvePairDispatch(0), { x: 0, y: 1, z: 1 });
	assert.deepEqual(resolvePairDispatch(65535), { x: 65535, y: 1, z: 1 });
	assert.deepEqual(resolvePairDispatch(65536), { x: 65535, y: 2, z: 1 });
	assert.deepEqual(resolvePairDispatch(432 * 4096), { x: 65535, y: 28, z: 1 });
	assert.throws(() => resolvePairDispatch(-1), /non-negative/);
});

test("checkpoint stride expands only when the raster would exceed the binding limit", () => {
	const storageLimit = 128 * 1024 * 1024;
	assert.deepEqual(resolveCheckpointLayout(96 * 72, 2048, storageLimit), {
		stride: 16,
		blocksPerTile: 128,
		byteLength: 14_155_776,
	});
	assert.deepEqual(resolveCheckpointLayout(384 * 288, 2048, storageLimit), {
		stride: 32,
		blocksPerTile: 64,
		byteLength: 113_246_208,
	});
	assert.deepEqual(resolveCheckpointLayout(384 * 288, 4096, storageLimit), {
		stride: 64,
		blocksPerTile: 64,
		byteLength: 113_246_208,
	});
	assert.deepEqual(resolveCheckpointLayout(384 * 288, 2048, storageLimit, 8), {
		stride: 16,
		blocksPerTile: 128,
		byteLength: 113_246_208,
	});
});

test("checkpoint precision is explicit and packed FP16 does not require FP16 arithmetic", () => {
	assert.equal(DEFAULT_CHECKPOINT_PRECISION, "packed-f16");
	assert.equal(resolveCheckpointPrecision(), "packed-f16");
	assert.equal(resolveCheckpointPrecision("packed-f16"), "packed-f16");
	assert.throws(() => resolveCheckpointPrecision("f8"), /f32.*packed-f16/);
	assert.match(source, /pack2x16float/);
	assert.match(source, /unpack2x16float/);
	assert.doesNotMatch(source, /enable f16/);
});

test("train backgrounds are deterministic, step-varying RGB values", () => {
	const first = trainingBackgroundForStep(0);
	const repeated = trainingBackgroundForStep(0);
	const next = trainingBackgroundForStep(1);
	assert.deepEqual(first, repeated);
	assert.notDeepEqual(first, next);
	assert.ok([...first, ...next].every((value) => value >= 0 && value < 1));
	assert.equal(packedTrainingBackgroundForStep(0, false), 0);
	assert.equal(packedTrainingBackgroundForStep(0) >>> 31, 1);
	assert.throws(() => trainingBackgroundForStep(-1), /non-negative/);
	assert.throws(() => trainingBackgroundForStep(1.5), /safe integer/);
});

test("target paging uploads exactly one selected Float32 frame and reuses the resident page", () => {
	const trainer = Object.create(DynamicSplatWebGpu3dTiledTrainer.prototype);
	const writes = [];
	trainer.dataset = {
		width: 2,
		height: 1,
		frameCount: 2,
		frames: Float32Array.from({ length: 2 * 2 * 2 * 4 }, (_, index) => index + 0.25),
		backgrounds: Float32Array.from({ length: 2 * 2 * 4 }, (_, index) => 100 + index),
	};
	trainer.device = { queue: { writeBuffer: (...args) => writes.push(args) } };
	trainer.targetPageKey = null;
	const target = { label: "target-page" };
	assert.equal(trainer.uploadTargetPage(target, 1, 1), 6);
	assert.equal(writes.length, 1);
	assert.equal(writes[0][0], target);
	assert.equal(writes[0][1], 0);
	assert.deepEqual(Array.from(writes[0][2]), Array.from(trainer.dataset.frames.subarray(24, 32)));
	assert.equal(trainer.uploadTargetPage(target, 1, 1), 6);
	assert.equal(writes.length, 1);
	assert.equal(trainer.uploadTargetPage(target, 0, 1), 2);
	assert.equal(writes.length, 2);
	assert.deepEqual(Array.from(writes[1][2]), Array.from(trainer.dataset.frames.subarray(8, 16)));
	assert.equal(trainer.uploadTargetPage(target, 1, 0, { staticWarmup: true }), 2);
	assert.equal(writes.length, 3);
	assert.deepEqual(Array.from(writes[2][2]), Array.from(trainer.dataset.backgrounds.subarray(8, 16)));
	assert.equal(trainer.uploadTargetPage(target, 1, 1, { staticWarmup: true }), 2);
	assert.equal(writes.length, 3);
});

test("SSIM radius accepts the benchmark range and preserves the 11x11 default", () => {
	assert.equal(resolveSsimRadius(), 5);
	assert.equal(resolveSsimRadius(0), 0);
	assert.equal(resolveSsimRadius(15), 15);
	assert.throws(() => resolveSsimRadius(-1), /0 through 15/);
	assert.throws(() => resolveSsimRadius(5.5), /integer/);
});

test("density schedule fills only reserved slots and preserves active topology afterward", () => {
	assert.equal(densityDispatchesForStep(4096, 4096, 512), 0);
	assert.equal(densityDispatchesForStep(4096, 4096, 119808), 0);
	assert.equal(densityDispatchesForStep(1536, 3072, 599), 0);
	assert.equal(densityDispatchesForStep(1536, 3072, 600), 4);
	assert.equal(densityDispatchesForStep(1536, 3072, 10100), 4);
	assert.equal(densityDispatchesForStep(1536, 3072, 10200), 0);
	assert.equal(densityDispatchesForStep(1536, 3072, 10240), 0);
	assert.equal(densityDispatchesForStep(1536, 3072, 119808), 0);
	assert.equal(densityDispatchesForStep(1536, 3072, 120320), 0);
	assert.equal(densityDispatchesForStep(1536, 4096, 16500), 4);
	assert.equal(densityDispatchesForStep(1536, 4096, 16896), 0);
});

test("full-frame schedule shuffles and visits every camera/time pair before cycling", () => {
	const trainViews = [2, 5, 9];
	const cycle = Array.from({ length: 6 }, (_, step) => fullFramePairForStep(trainViews, 2, step));
	assert.equal(new Set(cycle.map(({ viewIndex, frameIndex }) => `${viewIndex}:${frameIndex}`)).size, 6);
	assert.notEqual(cycle[0].frameIndex, cycle[1].frameIndex);
	assert.deepEqual(fullFramePairForStep(trainViews, 2, 6), cycle[0]);
	assert.deepEqual(fullFramePairForStep(trainViews, 2, -4), cycle[0]);
	assert.throws(() => fullFramePairForStep([], 2, 0), /train view/);
});

test("static warmup rotates only train-camera means before restarting the dynamic pair cycle", () => {
	const trainViews = [2, 5, 9];
	assert.equal(DEFAULT_STATIC_WARMUP_STEPS, 2048);
	assert.equal(resolveStaticWarmupSteps(), 0);
	assert.equal(resolveStaticWarmupSteps(2048), 2048);
	assert.throws(() => resolveStaticWarmupSteps(-1), /0 through 1000000/);
	assert.throws(() => resolveStaticWarmupSteps(1.5), /integer/);
	const warmup = Array.from({ length: 3 }, (_, step) =>
		trainingPairForStep(trainViews, 8, step, 3));
	assert.equal(new Set(warmup.map(({ viewIndex }) => viewIndex)).size, 3);
	assert.ok(warmup.every(({ frameIndex, staticWarmup }) => frameIndex === 3 && staticWarmup));
	assert.deepEqual(
		trainingPairForStep(trainViews, 8, 3, 3),
		{ ...fullFramePairForStep(trainViews, 8, 0), staticWarmup: false },
	);
});

test("warmup freezes temporal gates and display filtering follows display resolution", () => {
	assert.match(source, /cfg\.staticWarmup!=0u\)\{return 0\.5;/);
	assert.match(source, /select\(temporal_gate\(p,t,cfg\.temporalSigma\),1\.0,cfg\.staticWarmup!=0u\)/);
	assert.match(source, /let tc=select\(t\*2\.0-1\.0,0\.0,staticWarmup\)/);
	assert.match(source, /let gradStaticMix=select\(/);
	const displaySource = readFileSync(new URL("../trainerWebGpu3d.js", import.meta.url), "utf8");
	assert.match(displaySource,
		/filterVariance = pow\(\$\{FILTER_SIGMA_PIXELS\} \/ max\(1\.0, cfg\.height\), 2\.0\)/);
	assert.match(displaySource, /rawAlpha < 0\.00392156863/);
	assert.doesNotMatch(displaySource,
		/filterVariance = pow\(\$\{FILTER_SIGMA_PIXELS\} \/ max\(1\.0, cfg\.targetHeight\)/);
});

test("opacity-aware pixel bounds shrink support and clip to the image", () => {
	const projection = {
		valid: true,
		center: [0.5, 0.5],
		covariance: [1 / 64, 0, 1 / 64],
	};
	assert.deepEqual(opacityAwarePixelBounds(projection, 1, 8, 8), {
		minX: 1, maxX: 7, minY: 1, maxY: 7, qLimit: 9,
	});
	const lowOpacity = opacityAwarePixelBounds(projection, 0.2, 8, 8, 0.1);
	assert.deepEqual(
		{ minX: lowOpacity.minX, maxX: lowOpacity.maxX, minY: lowOpacity.minY, maxY: lowOpacity.maxY },
		{ minX: 2, maxX: 6, minY: 2, maxY: 6 },
	);
	assertClose(lowOpacity.qLimit, 2 * Math.log(2));
	assert.equal(opacityAwarePixelBounds(projection, 0.1, 8, 8, 0.1), null);
	assert.equal(opacityAwarePixelBounds({ ...projection, valid: false }, 1, 8, 8), null);
});

test("ellipse/rectangle test handles containment, edge intersection, and separation", () => {
	const conic = [1, 0, 1];
	assert.equal(ellipseIntersectsRect([0, 0], conic, 1,
		{ minX: -0.2, minY: -0.2, maxX: 0.2, maxY: 0.2 }), true);
	assert.equal(ellipseIntersectsRect([0, 0], conic, 1,
		{ minX: 0.8, minY: -0.1, maxX: 1.2, maxY: 0.1 }), true);
	assert.equal(ellipseIntersectsRect([0, 0], conic, 1,
		{ minX: 1.1, minY: -0.1, maxX: 1.3, maxY: 0.1 }), false);
	assert.equal(ellipseIntersectsRect([0, 0], [4, 0, 1], 1,
		{ minX: 0.45, minY: -0.1, maxX: 0.55, maxY: 0.1 }), true);
	assert.equal(ellipseIntersectsRect([0, 0], [4, 0, 1], 1,
		{ minX: 0.6, minY: -0.1, maxX: 0.8, maxY: 0.1 }), false);
});

test("windowed L1 plus SSIM is zero with zero gradient for identical images", () => {
	const image = Float64Array.from([
		0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7,
		0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8,
	]);
	const result = windowedL1DssimCpu(image, image, 3, 2, { radius: 1 });
	assertClose(result.loss, 0, 1e-12);
	assertClose(result.l1, 0, 1e-12);
	assertClose(result.dssim, 0, 1e-12);
	for (const gradient of result.gradient) assertClose(gradient, 0, 1e-12);
});

test("windowed L1 plus SSIM analytic gradient matches finite differences", () => {
	const width = 3;
	const height = 2;
	const length = width * height * 3;
	const target = Float64Array.from(
		{ length },
		(_, index) => 0.08 + 0.8 * ((index * 7) % length) / (length - 1),
	);
	const prediction = Float64Array.from(
		target,
		(value, index) => value + (index % 2 ? -0.025 : 0.03),
	);
	const analytic = windowedL1DssimCpu(prediction, target, width, height, { radius: 1 });
	const epsilon = 1e-5;
	for (let index = 0; index < length; index += 1) {
		const plus = Float64Array.from(prediction);
		const minus = Float64Array.from(prediction);
		plus[index] += epsilon;
		minus[index] -= epsilon;
		const finiteDifference = (
			windowedL1DssimCpu(plus, target, width, height, { radius: 1 }).loss
			- windowedL1DssimCpu(minus, target, width, height, { radius: 1 }).loss
		) / (2 * epsilon);
		assertClose(analytic.gradient[index], finiteDifference, 1e-6);
	}
});

test("motion weights emphasize residuals, preserve mean scale, and keep the image gradient exact", () => {
	const weights = normalizedMotionLossWeights([0, 0.00035, 0.001, 0.004, 0.02]);
	assertClose(weights.reduce((sum, value) => sum + value, 0) / weights.length, 1, 1e-7);
	assert.ok(weights[0] < weights[2]);
	assert.ok(weights[2] < weights[3]);
	assertClose(weights[3], weights[4], 1e-7);

	const width = 3;
	const height = 2;
	const target = Float64Array.from(
		{ length: width * height * 3 },
		(_, index) => 0.1 + 0.7 * ((index * 5) % 17) / 16,
	);
	const prediction = Float64Array.from(
		target,
		(value, index) => value + (index % 2 ? -0.02 : 0.03),
	);
	const pixelWeights = Float64Array.from([0.4, 0.7, 1.1, 1.4, 0.9, 1.5]);
	const analytic = windowedL1DssimCpu(prediction, target, width, height, {
		radius: 1,
		pixelWeights,
	});
	const epsilon = 1e-5;
	for (const index of [0, 5, 11, prediction.length - 1]) {
		const plus = Float64Array.from(prediction);
		const minus = Float64Array.from(prediction);
		plus[index] += epsilon;
		minus[index] -= epsilon;
		const finiteDifference = (
			windowedL1DssimCpu(plus, target, width, height, {
				radius: 1,
				pixelWeights,
				computeGradient: false,
			}).loss
			- windowedL1DssimCpu(minus, target, width, height, {
				radius: 1,
				pixelWeights,
				computeGradient: false,
			}).loss
		) / (2 * epsilon);
		assertClose(analytic.gradient[index], finiteDifference, 1e-6);
	}
});

test("calibrated train frames store normalized motion weights in otherwise-unused alpha", () => {
	const frames = new Float32Array(2 * 2 * 4);
	for (let pixel = 0; pixel < 4; pixel += 1) frames[pixel * 4 + 3] = 1;
	frames[(2 + 0) * 4] = 1;
	frames[(2 + 0) * 4 + 1] = 1;
	frames[(2 + 0) * 4 + 2] = 1;
	const backgrounds = new Float32Array(2 * 4);
	computeMultiviewSamples(frames, backgrounds, 2, 1, 2, 1);
	assertClose((frames[3] + frames[7]) / 2, 1, 1e-7);
	assertClose((frames[11] + frames[15]) / 2, 1, 1e-7);
	assert.ok(frames[11] > frames[15]);
});

test("default 11x11 training SSIM matches the Gaussian validation metric", () => {
	const width = 12;
	const height = 11;
	const target = Float64Array.from(
		{ length: width * height * 3 },
		(_, index) => ((index * 17) % 101) / 100,
	);
	const prediction = Float64Array.from(
		target,
		(value, index) => Math.min(1, Math.max(0, value + (index % 5 === 0 ? 0.03 : -0.01))),
	);
	const training = windowedL1DssimCpu(prediction, target, width, height);
	const validationSsim = canonicalGaussianSsim(prediction, target, width, height);

	assertClose(1 - training.dssim, validationSsim, 1e-10);
	assert.match(source, /case 0: \{ return 0\.2660117149; \}/);
	assert.match(source, /fn reflected_weight/);
});

test("default Gaussian SSIM image gradient matches finite differences", () => {
	const width = 12;
	const height = 11;
	const length = width * height * 3;
	const target = Float64Array.from(
		{ length },
		(_, index) => 0.05 + 0.9 * ((index * 13) % 97) / 96,
	);
	const prediction = Float64Array.from(
		target,
		(value, index) => value + (index % 3 === 0 ? 0.02 : -0.015),
	);
	const analytic = windowedL1DssimCpu(prediction, target, width, height);
	const epsilon = 1e-5;
	for (const index of [0, 1, 35, 117, 229, length - 1]) {
		const plus = Float64Array.from(prediction);
		const minus = Float64Array.from(prediction);
		plus[index] += epsilon;
		minus[index] -= epsilon;
		const finiteDifference = (
			windowedL1DssimCpu(plus, target, width, height, { computeGradient: false }).loss
			- windowedL1DssimCpu(minus, target, width, height, { computeGradient: false }).loss
		) / (2 * epsilon);
		assertClose(analytic.gradient[index], finiteDifference, 2e-6);
	}
});

test("tiled trainer source preserves the full-frame shared-backward contract", () => {
	const trainStep = DynamicSplatWebGpu3dTiledTrainer.prototype.trainStep.toString();
	assert.match(trainStep, /trainingPairForStep/);
	assert.match(trainStep, /this\.lastSampleCount\s*=\s*this\.pixelCount/);
	assert.match(trainStep, /this\.tilesX,\s*this\.tilesY/);
	assert.match(trainStep, /dispatchWorkgroupsIndirect/);
	assert.match(source, /fn\s+project_and_bin/);
	assert.match(source, /fn\s+sort_tiles/);
	assert.match(source, /depthKeys/);
	assert.match(source, /workgroupUniformLoad\(&tileSortCount\)/);
	assert.match(source, /span>=max\(count,1u\)/);
	assert.match(source, /width<=sortCount/);
	assert.doesNotMatch(source, /width<=cfg\.tileCapacity/);
	assert.match(source, /fn\s+raster_forward/);
	assert.match(source, /training_background\(cfg\.trainingBackgroundPacked\)/);
	assert.match(source, /color\+transmittance\*background/);
	assert.match(source, /rendered\[pixel\]\.xyz-before-transmittance\*alpha/);
	assert.match(source, /fn\s+ssim_stats/);
	assert.match(source, /fn\s+ssim_gradient/);
	assert.match(source, /cfg\.motionWeighting!=0u/);
	assert.match(source, /fn\s+pair_backward/);
	assert.match(source, /fn\s+reduce_update/);
	assert.match(source, /atomicCompareExchangeWeak/);
	assert.match(source, /wid\.y\*\$\{MAX_WORKGROUPS_PER_DIMENSION\}u\+wid\.x/);
	assert.match(source, /f32\(stopRanks\[pixel\]\)/);
	assert.match(source, /rank<u32\(pixelGrad\[pixel\]\.w\)/);
	assert.doesNotMatch(source, /bitcast<f32>\(stopRanks|bitcast<u32>\(pixelGrad\[pixel\]\.w/);
	assert.match(trainStep, /uploadTargetPage/);
	assert.match(trainStep, /const targetOffset = 0/);
	assert.ok(trainStep.indexOf("uploadTargetPage") < trainStep.indexOf("writeBuffer(this.buffers.tiledConfig"));
	assert.ok(trainStep.indexOf("writeBuffer(this.buffers.tiledConfig") < trainStep.indexOf("queue.submit"));
	assert.equal(SCALE_LR_FROM_COLOR, 0.30);
	assert.equal(ROTATION_LR_FROM_MOTION, 1.25);
	assert.equal(MAX_SCALE_ASPECT_RATIO, 6);
	assert.equal(FILTER_SIGMA_PIXELS, 0.3);
	assert.equal(MAX_SPLAT_COLOR, 1);
	assert.match(source, /pairData/);
	assert.match(source, /gradientAtoms/);
	assert.doesNotMatch(source, /gaussianPairSlots|pairGradients:array/);
	assert.doesNotMatch(trainStep, /samplesPerStep|sampleIndices/);
});

test("SPA defaults to train-only random backgrounds and exposes the control", () => {
	const app = readFileSync(new URL("../app.js", import.meta.url), "utf8");
	const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
	const worker = readFileSync(new URL("../trainingWorker.js", import.meta.url), "utf8");
	assert.match(html, /id="randomBackgroundToggle"[^>]*checked/);
	assert.match(app,
		/randomBackground:\s*!sampledBackendSelected\(\)\s*&&\s*controls\.randomBackground\.checked/);
	assert.match(worker, /randomBackground:\s*false/);
	assert.match(html, /validation and preview remain black/);
});
