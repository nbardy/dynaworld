import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
	activePrefixDispatchSizes,
	activeSplatCountForStep,
	cameraSceneRadius,
	DynamicSplatWebGpu3dTiledTrainer,
	DEFAULT_BROWSER_GROWTH_CAPACITY,
	DEFAULT_CHECKPOINT_PRECISION,
	DEFAULT_MAX_TILE_CAPACITY,
	DEFAULT_STATIC_WARMUP_STEPS,
	densityDispatchesForStep,
	ellipseIntersectsRect,
	fullFramePairForStep,
	MAX_SCALE_ASPECT_RATIO,
	MAX_WORKGROUPS_PER_DIMENSION,
	opacityAwarePixelBounds,
	packDepthSplatKey,
	packedTrainingBackgroundForStep,
	PIXEL_GS_DEPTH_GAMMA,
	ROTATION_LR_FROM_MOTION,
	resolveCheckpointLayout,
	resolveCheckpointPrecision,
	resolveCheckpointStride,
	resolvePairDispatch,
	resolvePixelDepthGamma,
	resolveSsimRadius,
	resolveStaticWarmupSteps,
	resolveTileCapacity,
	resolveTiledBackwardGranularity,
	resolveTiledCapacity,
	resolveTiledCheckpointOrder,
	resolveTiledProjectionLayout,
	resolveTiledProjectionVjpPrecision,
	resolveTiledSsimLayout,
	resolveTiledTileSize,
	SCALE_LR_FROM_POSITION,
	summarizeCycleMetrics,
	TILED_BACKWARD_GRANULARITIES,
	TILED_CHECKPOINT_ORDERS,
	TILED_PROJECTION_LAYOUTS,
	TILED_PROJECTION_VJP_PRECISIONS,
	TILED_SSIM_LAYOUTS,
	TILED_DEPTH_KEY_MASK,
	TILED_SPLAT_ID_BITS,
	TILED_SPLAT_ID_MASK,
	telemetryAliasPeriod,
	trainingBackgroundForStep,
	trainingPairForStep,
	unpackDepthSplatId,
	windowedL1DssimCpu,
} from "../trainerWebGpu3dTiled.js";
import {
	DynamicSplatWebGpu3dTiledFastTrainer,
	resolveFastTileCapacity,
} from "../trainerWebGpu3dTiledFast.js";
import { computeMultiviewSamples, normalizedMotionLossWeights } from "../dataset.js";
import { canonicalGaussianSsim } from "../snapshotMetrics.js";
import {
	FILTER_SIGMA_PIXELS,
	MAX_SPLAT_COLOR,
	sampledOrderCacheEntries,
} from "../trainerWebGpu3d.js";

const source = readFileSync(new URL("../trainerWebGpu3dTiled.js", import.meta.url), "utf8");
const fastSource = readFileSync(new URL("../trainerWebGpu3dTiledFast.js", import.meta.url), "utf8");

function assertClose(actual, expected, tolerance = 1e-7) {
	const scale = Math.max(1, Math.abs(actual), Math.abs(expected));
	assert.ok(Math.abs(actual - expected) <= tolerance * scale,
		`expected ${actual} to be within ${tolerance} relative of ${expected}`);
}

test("tiled capacity reserves growth while respecting explicit bounds", () => {
	assert.equal(DEFAULT_BROWSER_GROWTH_CAPACITY, 8192);
	assert.equal(resolveTiledCapacity(8), 24);
	assert.equal(resolveTiledCapacity(768), 2304);
	assert.equal(resolveTiledCapacity(768, 512), 768);
	assert.equal(resolveTiledCapacity(768, 1000.9), 1000);
	assert.equal(resolveTiledCapacity(1024, 4096), 4096);
	assert.equal(resolveTiledCapacity(2048, 8192), 8192);
	assert.equal(resolveTiledCapacity(4096), 8192);
	assert.equal(resolveTiledCapacity(8192), 8192);
	assert.equal(resolveTiledCapacity(4096, 32768), 32768);
	assert.equal(resolveTiledCapacity(16384), 16384);
	assert.equal(resolveTiledCapacity(32768), 32768);
	assert.throws(() => resolveTiledCapacity(7), /8 through 32768/);
	assert.throws(() => resolveTiledCapacity(8.5), /integer/);
	assert.throws(() => resolveTiledCapacity(32769), /8 through 32768/);
});

test("32K models retain a portable bounded tile-local sort", () => {
	assert.equal(DEFAULT_MAX_TILE_CAPACITY, 4096);
	assert.equal(resolveTileCapacity(8), 16);
	assert.equal(resolveTileCapacity(768), 1024);
	assert.equal(resolveTileCapacity(1536), 2048);
	assert.equal(resolveTileCapacity(4096), 4096);
	assert.equal(resolveTileCapacity(8192), 4096);
	assert.equal(resolveTileCapacity(32768), 4096);
	assert.equal(resolveTileCapacity(4096, 4096), 4096);
	assert.equal(resolveTileCapacity(8192, 2048), 2048);
	assert.throws(() => resolveTileCapacity(4096, 4097), /8 through 4096/);
	assert.throws(() => resolveTileCapacity(32769), /8 through 32768/);
});

test("fast tiled defaults track measured occupancy without hiding explicit benchmark controls", () => {
	assert.equal(resolveFastTileCapacity(4096, 8192), 1024);
	assert.equal(resolveFastTileCapacity(8192, 16384), 2048);
	assert.equal(resolveFastTileCapacity(4096, 24576), 4096);
	assert.equal(resolveFastTileCapacity(4096, 32768), 4096);
	assert.equal(resolveFastTileCapacity(4096, 8192, 2048), 2048);
	assert.match(fastSource,
		/backwardGranularity:\s*TILED_BACKWARD_GRANULARITIES\.CHECKPOINT_BLOCK/);
	assert.doesNotMatch(fastSource, /checkpointOrder:/);
	assert.match(fastSource, /checkpointStride:\s*DEFAULT_CHECKPOINT_STRIDE/);
	assert.match(fastSource, /tileSize:\s*8/);
	assert.match(fastSource,
		/projectionLayout:\s*TILED_PROJECTION_LAYOUTS\.SPLIT_COMPACT/);
	assert.match(fastSource, /ssimLayout:\s*TILED_SSIM_LAYOUTS\.SEPARABLE/);
	assert.match(fastSource, /\.\.\.options,[\s\S]*backwardMode:\s*TILED_BACKWARD_MODES\.STAGED_PROJECT_3D/);
	assert.equal(
		Object.getPrototypeOf(DynamicSplatWebGpu3dTiledFastTrainer.prototype).constructor.name,
		DynamicSplatWebGpu3dTiledTrainer.name,
	);
});

test("kernel fork controls reject unsupported checkpoint, tile, and replay layouts", () => {
	assert.equal(resolveCheckpointStride(8), 8);
	assert.equal(resolveCheckpointStride(16), 16);
	assert.equal(resolveCheckpointStride(32), 32);
	assert.throws(() => resolveCheckpointStride(12), /power of two/);
	assert.equal(resolveTiledTileSize(8), 8);
	assert.equal(resolveTiledTileSize(16), 16);
	assert.throws(() => resolveTiledTileSize(32), /8, 16/);
	assert.equal(resolveTiledBackwardGranularity("checkpoint-block"),
		TILED_BACKWARD_GRANULARITIES.CHECKPOINT_BLOCK);
	assert.throws(() => resolveTiledBackwardGranularity("pixel"), /pair, checkpoint-block/);
	assert.equal(resolveTiledCheckpointOrder("block-major"), TILED_CHECKPOINT_ORDERS.BLOCK_MAJOR);
	assert.equal(resolveTiledCheckpointOrder(), TILED_CHECKPOINT_ORDERS.PIXEL_MAJOR);
	assert.throws(() => resolveTiledCheckpointOrder("row-major"), /pixel-major, block-major/);
	assert.equal(resolveTiledProjectionLayout("split-compact"),
		TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT);
	assert.equal(resolveTiledProjectionLayout(), TILED_PROJECTION_LAYOUTS.MONOLITHIC);
	assert.throws(() => resolveTiledProjectionLayout("array-of-structs"),
		/monolithic, split-compact/);
	assert.equal(resolveTiledProjectionVjpPrecision("packed-f16"),
		TILED_PROJECTION_VJP_PRECISIONS.PACKED_F16);
	assert.equal(resolveTiledProjectionVjpPrecision(), TILED_PROJECTION_VJP_PRECISIONS.F32);
	assert.throws(() => resolveTiledProjectionVjpPrecision("f8"), /f32, packed-f16/);
	assert.equal(resolveTiledSsimLayout("separable"), TILED_SSIM_LAYOUTS.SEPARABLE);
	assert.equal(resolveTiledSsimLayout(), TILED_SSIM_LAYOUTS.NAIVE_2D);
	assert.throws(() => resolveTiledSsimLayout("box"), /naive-2d, separable/);
	assert.match(source, /block\*cfg\.pixelCount\+pixel/);
	assert.match(source, /pixel\*cfg\.blocksPerTile\+block/);
	assert.match(source, /checkpoint_index\(pixel,rank\/cfg\.checkpointStride\)/);
});

test("kernel lab preserves matched direct and staged variants with timestamped phase output", () => {
	const benchmark = readFileSync(new URL("../benchmarkTiledKernels.js", import.meta.url), "utf8");
	const benchmarkHtml = readFileSync(
		new URL("../benchmarkTiledKernels.html", import.meta.url),
		"utf8",
	);
	assert.match(benchmark, /id:\s*"direct-3d"/);
	assert.match(benchmark, /id:\s*"staged-project3d"/);
	assert.match(benchmark, /id:\s*"staged-split-f32"/);
	assert.match(benchmark, /id:\s*"staged-split-packed-f16"/);
	assert.match(benchmark, /context\.trainer\.profileGpuStep/);
	assert.match(benchmark, /round % 2 === 0/);
	assert.match(benchmark, /candidateThroughputSpeedup/);
	assert.match(benchmark, /stagedThroughputSpeedup/);
	assert.match(benchmark, /allocatedByteDelta/);
	assert.match(benchmark, /projectionLayout/);
	assert.match(benchmark, /projectionVjpPrecision/);
	assert.match(benchmarkHtml, /id="kernelProjectionVjpPrecision"/);
	assert.match(source, /TILED_GPU_PHASES/);
	assert.match(source, /createQuerySet/);
	assert.match(source, /timestampWrites/);
	assert.match(source, /gpuSpanMs/);
	assert.match(source, /phaseContract/);
	assert.match(source, /maintenanceIncluded:\s*false/);
	assert.match(benchmark, /gpuSpanMedianMs/);
	assert.match(benchmark, /summarizeRoundStability/);
	assert.match(benchmark, /validForPromotion/);
	assert.match(benchmark, /maxRoundCv/);
	assert.match(benchmark, /computeSamples:\s*false/);
	assert.match(benchmark, /frameBankFormat:\s*options\.frameBank/);
	assert.match(benchmarkHtml, /id="kernelFrameBank"/);
	assert.match(
		benchmarkHtml,
		/id="kernelMaxRoundCv"[^>]*min="0\.001"[^>]*step="0\.001"[^>]*value="0\.100"/,
	);
});

test("staged temporal VJP differentiates the same static-mixed gate as forward", () => {
	assert.match(source, /let timeWeight=select\(mix\(dynamicGate,1\.0,staticMix\),1\.0,staticWarmup\)/);
	assert.match(source, /let dynamicCore=\(1\.0-temporalFloor\)\*temporalKernel/);
	assert.doesNotMatch(source,
		/let timeWeight=select\(dynamicGate,1\.0,staticWarmup\)/);
});

test("Pixel-GS floater guard scales only the non-cancelling density statistic", () => {
	assert.equal(PIXEL_GS_DEPTH_GAMMA, 0.37);
	assert.equal(resolvePixelDepthGamma(), 0.37);
	assert.throws(() => resolvePixelDepthGamma(0), /positive finite/);
	const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
	const left = identity.slice(); left[3] = 1;
	const right = identity.slice(); right[3] = -1;
	assertClose(cameraSceneRadius([
		{ worldToCamera: left }, { worldToCamera: right },
	]), 1.1);
	assert.match(source, /fn density_gradient_scale\(cameraDepth:f32,cfg:TiledConfig\)/);
	assert.match(source,
		/let densitySignal=length\(barMu\)\*density_gradient_scale\(cp\.z,cfg\)/);
	assert.match(source,
		/vec4<f32>\(gradLogScale,densitySignal\),gradRotation/);
	assert.doesNotMatch(source, /worldGrad\s*\*\s*density_gradient_scale/);
});

test("split projection layout keeps the raster hot record at 32 bytes", () => {
	assert.match(source, /const RASTER_PROJECTION_BYTES = 2 \* 16/);
	assert.match(source, /const PROJECTION_VJP_BYTES = 5 \* 16/);
	assert.match(source, /const PACKED_PROJECTION_VJP_BYTES = 3 \* 16/);
	assert.match(source, /struct RasterProjection \{[\s\S]*screenConic0[\s\S]*conicDepthAlpha/);
	assert.match(source,
		/struct CompactProjectionVjp \{[\s\S]*cameraPointValid[\s\S]*jacobianSparse[\s\S]*basisVariance2/);
	assert.match(source,
		/struct PackedCompactProjectionVjp \{[\s\S]*cameraPointValid[\s\S]*packed0[\s\S]*packed1/);
	assert.match(source, /vec4<f32>\(j0\.x,j0\.z,j1\.y,j1\.z\)/);
	assert.match(source, /let storedVariances=variances\/varianceScale/);
	assert.match(source,
		/let variances=normalizedVariances[\s\S]*cfg\.geometryScale\*cfg\.geometryScale/);
	assert.match(source, /atomicAdd\(&counters\[8\],1u\)/);
	assert.match(source, /cameras\[cfg\.viewIndex\]/);
	assert.match(source, /rasterProjections\[index\]=raster;projectionVjps\[index\]=vjp/);
	assert.match(source, /projectionVjp \?\? this\.buffers\.projections/);
});

test("separable SSIM keeps the exact two-dimensional objective and adjoint contract", () => {
	assert.match(source, /fn ssim_horizontal/);
	assert.match(source, /fn ssim_vertical/);
	assert.match(source, /fn ssim_gradient_horizontal/);
	assert.match(source, /fn ssim_gradient_vertical/);
	assert.match(source, /constant\+targetCoefficient\*targetColor\+predictionCoefficient\*prediction/);
	assert.match(source, /this\.ssimLayout === TILED_SSIM_LAYOUTS\.SEPARABLE/);
});

test("depth sort keys preserve every 15-bit 32K splat ID", () => {
	assert.equal(TILED_SPLAT_ID_BITS, 15);
	assert.equal(TILED_SPLAT_ID_MASK, 0x7fff);
	assert.equal(TILED_DEPTH_KEY_MASK, 0xffff8000);
	for (const id of [0, 4095, 8192, 16384, 32767]) {
		assert.equal(unpackDepthSplatId(packDepthSplatKey(0x42f6a123, id)), id);
	}
	assert.throws(() => packDepthSplatKey(0x42f6a123, 32768), /splatId/);
	assert.match(source, /const depthMask = .*TILED_DEPTH_KEY_MASK/);
	assert.match(source, /const idMask = .*TILED_SPLAT_ID_MASK/);
	assert.match(source, /depthBits&\$\{depthMask\}/);
	assert.match(source, /depthKeys\[index\]&\$\{idMask\}/);
});

test("tiled inheritance does not reserve the sampled view-time depth-order cache", () => {
	const entries = sampledOrderCacheEntries(17, 16, 30000);
	assert.equal(entries, 8_160_000);
	assert.equal(entries * Uint32Array.BYTES_PER_ELEMENT, 32_640_000);
	assert.equal(sampledOrderCacheEntries(17, 16, 30000, false), 0);
	assert.match(
		readFileSync(new URL("../trainerWebGpu3d.js", import.meta.url), "utf8"),
		/!this\.skipSampleGradientAllocation/,
	);
});

test("telemetry cadence math explains the former 256-step camera-time alias", () => {
	assert.deepEqual(telemetryAliasPeriod(17 * 16, 256), {
		sampledPhases: 17,
		repeatSamples: 17,
		repeatSteps: 4352,
	});
	assert.deepEqual(telemetryAliasPeriod(17 * 16, 257), {
		sampledPhases: 272,
		repeatSamples: 272,
		repeatSteps: 69904,
	});
	assert.throws(() => telemetryAliasPeriod(272, 0), /positive/);
});

test("GPU cycle telemetry averages each recent camera-time objective exactly once", () => {
	const records = new Float32Array(4 * 4);
	for (let step = 4; step <= 7; step += 1) {
		const base = (step % 4) * 4;
		records.set([step, step + 0.1, step + 0.2, step + 1], base);
	}
	assert.deepEqual(summarizeCycleMetrics(records, 7, 4), {
		loss: 5.5,
		l1: 5.599999904632568,
		dssim: 5.699999809265137,
		samples: 4,
		complete: true,
		oldestStep: 4,
		newestStep: 7,
	});
	const partial = summarizeCycleMetrics(records, 7, 4, 6);
	assert.equal(partial.samples, 2);
	assert.equal(partial.complete, false);
	assert.equal(partial.loss, 6.5);
	assert.throws(() => summarizeCycleMetrics([], 7, 4), /Cycle metrics/);
	assert.match(
		source,
		/cycleMetrics\[cfg\.step%min\(cfg\.cycleMetricCount,arrayLength\(&cycleMetrics\)\)\]/,
	);
	assert.match(source, /copyBufferToBuffer\(\s*this\.buffers\.cycleMetrics/);
});

test("tiled telemetry retains loss fields plus FP16 saturation and overflow high-water state", () => {
	const benchmarkSource = readFileSync(
		new URL("../benchmarkTrainerBackends.js", import.meta.url),
		"utf8",
	);
	assert.match(source, /const TILED_METRICS_BYTES = 5 \* 16/);
	assert.match(source, /tiledMetrics:\s*makeBuffer\(TILED_METRICS_BYTES\)/);
	assert.match(source,
		/copyBufferToBuffer\([\s\S]*this\.buffers\.tiledMetrics,[\s\S]*TILED_METRICS_BYTES/);
	assert.match(source, /atomicAdd\(&counters\[4\],1u\)/);
	assert.match(source, /atomicMax\(&counters\[5\],count\)/);
	assert.match(source, /tileOverflowTotal:\s*values\[12\]/);
	assert.match(source, /maxTileOccupancyEver:\s*values\[13\]/);
	assert.match(source, /f32\(cfg\.step\),f32\(cfg\.activeSplatCount\)/);
	assert.match(source, /activeUpdateSplats:\s*values\[15\]/);
	assert.match(source, /dormantUpdateSplats:\s*values\[9\]\s*-\s*values\[15\]/);
	assert.match(source, /projectionVjpHalfSaturations:\s*values\[16\]/);
	assert.match(source, /projectionVjpHalfSaturationsTotal:\s*values\[17\]/);
	assert.match(source, /atomicAdd\(&counters\[9\],1u\)/);
	assert.match(source, /initialActiveUpdateSlots:\s*this\.activeSplatCount/);
	assert.match(source, /updateSlotCapacity:\s*this\.splatCount/);
	assert.match(source, /dormantSlotSparseUpdate:\s*true/);
	assert.match(source, /activeUpdateSlots:\s*profiledActiveSplats/);
	assert.match(source, /capacityUpdateSlots:\s*this\.splatCount/);
	assert.match(benchmarkSource, /tileOverflowTotal\s*\?\?\s*currentOverflow/);
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

test("compact target paging uploads bytes and schedules one GPU page decode", () => {
	const trainer = Object.create(DynamicSplatWebGpu3dTiledTrainer.prototype);
	const writes = [];
	trainer.dataset = {
		width: 2,
		height: 1,
		frameCount: 1,
		frames: Uint8Array.from([
			1, 2, 3, 127, 4, 5, 6, 254,
			7, 8, 9, 0, 10, 11, 12, 64,
		]),
		backgrounds: Float32Array.from({ length: 16 }, (_, index) => index + 0.5),
	};
	trainer.device = { queue: { writeBuffer: (...args) => writes.push(args) } };
	trainer.buffers = { targetPacked: { label: "packed-target-page" } };
	trainer.compactTargetFrames = true;
	trainer.targetDecodePending = false;
	trainer.targetPageKey = null;
	const target = { label: "float-target-page" };

	assert.equal(trainer.uploadTargetPage(target, 1, 0), 2);
	assert.equal(writes.length, 1);
	assert.equal(writes[0][0], trainer.buffers.targetPacked);
	assert.deepEqual(Array.from(writes[0][2]), Array.from(trainer.dataset.frames.subarray(8, 16)));
	assert.equal(trainer.targetDecodePending, true);

	assert.equal(trainer.uploadTargetPage(target, 1, 0, { staticWarmup: true }), 2);
	assert.equal(writes.length, 2);
	assert.equal(writes[1][0], target);
	assert.deepEqual(
		Array.from(writes[1][2]),
		Array.from(trainer.dataset.backgrounds.subarray(8, 16)),
	);
	assert.equal(trainer.targetDecodePending, false);
	assert.match(source, /unpack4x8unorm\(packed\)/);
	assert.match(source, /f32\(\(packed>>24u\)&0xffu\)\/127\.0/);
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
	assert.equal(densityDispatchesForStep(4096, 8192, 600), 4);
	assert.equal(densityDispatchesForStep(4096, 8192, 26100), 4);
	assert.equal(densityDispatchesForStep(4096, 8192, 26200), 0);
});

test("active prefix advances only after completed density events and clips the final split", () => {
	assert.equal(activeSplatCountForStep(4096, 8192, 0), 4096);
	assert.equal(activeSplatCountForStep(4096, 8192, 599), 4096);
	assert.equal(activeSplatCountForStep(4096, 8192, 600), 4112);
	assert.equal(activeSplatCountForStep(4096, 8192, 699), 4112);
	assert.equal(activeSplatCountForStep(4096, 8192, 700), 4128);
	assert.equal(activeSplatCountForStep(4096, 8192, 26100), 8192);
	assert.equal(activeSplatCountForStep(4096, 8192, 26200), 8192);
	assert.equal(activeSplatCountForStep(9, 14, 599), 9);
	assert.equal(activeSplatCountForStep(9, 14, 600), 14);
	assert.equal(densityDispatchesForStep(9, 14, 600), 2);
	assert.equal(densityDispatchesForStep(9, 14, 700), 0);
	assert.throws(() => activeSplatCountForStep(10, 9, 600), /capacity/);
	assert.throws(() => activeSplatCountForStep(9, 14, -1), /non-negative/);
});

test("clear and Adam dispatches scale with active prefix rather than reserved capacity", () => {
	assert.deepEqual(activePrefixDispatchSizes(4096, 8192, 108, 12), {
		activeUpdateSlots: 4096,
		capacitySlots: 8192,
		dormantUpdateSlots: 4096,
		gradientClearSlots: 49152,
		clearWorkgroups: 768,
		updateWorkgroups: 64,
	});
	assert.deepEqual(activePrefixDispatchSizes(4112, 8192, 108, 12), {
		activeUpdateSlots: 4112,
		capacitySlots: 8192,
		dormantUpdateSlots: 4080,
		gradientClearSlots: 49344,
		clearWorkgroups: 771,
		updateWorkgroups: 65,
	});
	assert.deepEqual(activePrefixDispatchSizes(4096, 8192, 108, 24), {
		activeUpdateSlots: 4096,
		capacitySlots: 8192,
		dormantUpdateSlots: 4096,
		gradientClearSlots: 98304,
		clearWorkgroups: 1536,
		updateWorkgroups: 64,
	});
	assert.equal(activePrefixDispatchSizes(8192, 8192, 108, 12).updateWorkgroups, 128);
	assert.throws(() => activePrefixDispatchSizes(8193, 8192, 108, 12), /capacity/);
});

test("tiled density activates contiguous tail slots with explicitly clean optimizer state", () => {
	const trainStep = DynamicSplatWebGpu3dTiledTrainer.prototype.trainStep.toString();
	assert.match(source, /activeSplatCount:u32/);
	assert.match(source,
		/gid\.x<cfg\.activeSplatCount\*\$\{gradientFloats\}u/);
	assert.match(source, /if\(i>=cfg\.activeSplatCount\)\{return;\}/);
	assert.match(source, /let adamStep=f32\(cfg\.step\+1u\)/);
	assert.doesNotMatch(source, /lastUpdatedStep|visibilitySparse/);
	assert.match(source, /let childIndex=activeCount\+slot/);
	assert.match(source, /for\(var i=0u;i<activeCount;i\+\+\)/);
	assert.match(source, /firstMoment\[childIndex\]=zero_splat\(\)/);
	assert.match(source, /secondMoment\[childIndex\]=zero_splat\(\)/);
	assert.match(source, /splatStats\[childIndex\]=vec4<f32>\(0\.0\)/);
	assert.match(source, /atomicStore\(&gradientAtoms\[gradientBase\+component\],0u\)/);
	assert.match(source, /atomicStore\(&counters\[7\],activeCount\+splitCount\)/);
	assert.match(trainStep, /activePrefixDispatchSizes/);
	assert.match(trainStep, /activeDispatch\.clearWorkgroups/);
	assert.match(trainStep, /activeDispatch\.updateWorkgroups/);
	assert.match(trainStep, /this\.tiledPipelines\.density/);
	assert.doesNotMatch(trainStep, /this\.pipelines\.maintenance/);
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
	assert.equal(SCALE_LR_FROM_POSITION, 9 / 7);
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

test("SPA exposes Pixel-GS density scaling as a reset-time ablation", () => {
	const app = readFileSync(new URL("../app.js", import.meta.url), "utf8");
	const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
	assert.match(html, /id="pixelDepthScalingToggle"[^>]*checked/);
	assert.match(app,
		/pixelDepthScaling:\s*controls\.pixelDepthScaling\.checked\s*&&\s*!sampledBackendSelected\(\)/);
	assert.match(app,
		/controls\.pixelDepthScaling\.addEventListener\("change",\s*\(\)\s*=>\s*\{\s*void resetTrainer\(\)/);
});
