import assert from "node:assert/strict";
import test from "node:test";
import {
	estimateDatasetResidentBytes,
	estimateTiledBenchmarkResources,
	estimateTiledTrainerBuffers,
	MIB,
	PORTABLE_STORAGE_BUFFER_LIMIT,
} from "../benchmarkResourcePlan.js";

const metadata = Object.freeze({
	width: 96,
	height: 72,
	viewCount: 18,
	trainViewCount: 17,
	frameCount: 16,
});

const baseOptions = Object.freeze({
	experiment: "backward",
	variant: "both",
	scale: 1,
	capacity: 30000,
	tileSize: 8,
	tileCapacity: 4096,
	checkpointPrecision: "packed-f16",
	checkpointStride: 16,
	projectionLayout: "split-compact",
	projectionVjpPrecision: "f32",
	ssimLayout: "separable",
});

test("resource estimator tracks the saved 30K/96 allocations plus sparse-prefix config", () => {
	const report = estimateTiledBenchmarkResources(metadata, baseOptions);
	assert.deepEqual(
		report.variants.map(({ id, allocatedBytes }) => ({ id, allocatedBytes })),
		[
			{ id: "direct-3d", allocatedBytes: 43_591_636 },
			{ id: "staged-project3d", allocatedBytes: 39_751_636 },
		],
	);
});

test("30K/384 packed checkpoints fit the portable 128 MiB binding floor", () => {
	const report = estimateTiledBenchmarkResources(metadata, {
		...baseOptions,
		scale: 4,
	});
	assert.deepEqual(report.raster, [384, 288]);
	assert.equal(report.valid, true);
	for (const variant of report.variants) {
		assert.deepEqual(variant.checkpoint, {
			stride: 32,
			blocksPerTile: 128,
			byteLength: 113_246_208,
		});
		assert.ok(variant.largestBinding.byteLength <= PORTABLE_STORAGE_BUFFER_LIMIT);
	}
	assert.ok(report.minimumAvailableMemoryBytes > 1.5 * 1024 ** 3);
	assert.ok(report.minimumAvailableMemoryBytes < 2.5 * 1024 ** 3);
});

test("RGBA8 sharing cuts 384 frames by four while backgrounds remain FP32", () => {
	const floatBank = estimateDatasetResidentBytes({
		sourceWidth: 96,
		sourceHeight: 72,
		width: 384,
		height: 288,
		viewCount: 18,
		frameCount: 16,
		channelBytes: 4,
	});
	const byteBank = estimateDatasetResidentBytes({
		sourceWidth: 96,
		sourceHeight: 72,
		width: 384,
		height: 288,
		viewCount: 18,
		frameCount: 16,
		channelBytes: 1,
	});
	assert.equal(floatBank.scaledBytes / MIB, 516.375);
	assert.equal(byteBank.scaledBytes / MIB, 151.875);
	assert.equal((floatBank.scaledBytes - byteBank.scaledBytes) / MIB, 364.5);
	assert.equal(byteBank.decodedAtlasBytes / MIB, 0.421875);
});

test("compact target planning includes the packed GPU page and lower host bank", () => {
	const compact = estimateTiledBenchmarkResources(metadata, {
		...baseOptions,
		variant: "candidate",
		scale: 4,
		projectionVjpPrecision: "packed-f16",
	}, {
		datasetChannelBytes: 1,
	});
	assert.equal(compact.variants[0].bufferBytes.packedTargetPage, 384 * 288 * 4);
	assert.equal(compact.dataset.scaledBytes / MIB, 151.875);
	assert.equal(compact.dataset.channelBytes, 1);
});

test("packed projection VJP storage removes 32 bytes per capacity splat", () => {
	const f32 = estimateTiledBenchmarkResources(metadata, {
		...baseOptions,
		variant: "candidate",
	});
	const packed = estimateTiledBenchmarkResources(metadata, {
		...baseOptions,
		variant: "candidate",
		projectionVjpPrecision: "packed-f16",
	});
	assert.equal(f32.variants[0].projectionVjpPrecision, "f32");
	assert.equal(packed.variants[0].projectionVjpPrecision, "packed-f16");
	assert.equal(
		f32.variants[0].allocatedBytes - packed.variants[0].allocatedBytes,
		baseOptions.capacity * 32,
	);
});

test("single-trainer estimates fail when no checkpoint layout fits", () => {
	assert.throws(() => estimateTiledTrainerBuffers({
		width: 16384,
		height: 16384,
		viewCount: 18,
		trainViewCount: 17,
		frameCount: 16,
		capacity: 30000,
		tileSize: 8,
		tileCapacity: 4096,
		checkpointPrecision: "packed-f16",
		checkpointStride: 16,
		projectionLayout: "split-compact",
		ssimLayout: "separable",
		backwardMode: "staged-project3d",
		storageBufferLimit: PORTABLE_STORAGE_BUFFER_LIMIT,
	}), /Checkpoint storage cannot fit/);
});
