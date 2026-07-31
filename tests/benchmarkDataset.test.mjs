import assert from "node:assert/strict";
import test from "node:test";
import {
	resizeDatasetForBenchmark,
	resizePackedRgbaNearest,
} from "../benchmarkDataset.js";
import {
	FRAME_BANK_FORMAT_RGBA8,
	FRAME_WEIGHT_BYTE_SCALE,
} from "../dataset.js";

test("nearest RGBA scaling preserves image boundaries and channels", () => {
	const source = Float32Array.from([
		1, 0, 0, 1,
		0, 1, 0, 1,
	]);
	const resized = resizePackedRgbaNearest(source, 2, 1, 1, 2);
	assert.deepEqual([...resized], [
		1, 0, 0, 1, 1, 0, 0, 1,
		0, 1, 0, 1, 0, 1, 0, 1,
		1, 0, 0, 1, 1, 0, 0, 1,
		0, 1, 0, 1, 0, 1, 0, 1,
	]);
});

test("benchmark dataset scaling rebuilds view slices and packed sample indices", () => {
	const frame = Float32Array.from([
		0.8, 0.1, 0.1, 1,
		0.2, 0.2, 0.2, 1,
	]);
	const background = Float32Array.from([
		0.1, 0.1, 0.1, 1,
		0.2, 0.2, 0.2, 1,
	]);
	const dataset = {
		name: "fixture",
		width: 2,
		height: 1,
		frameCount: 1,
		viewCount: 1,
		trainViewCount: 1,
		frames: frame,
		backgrounds: background,
		cameras: [{ name: "cam00", role: "train" }],
		comparisonViewIndices: [0],
	};
	const resized = resizeDatasetForBenchmark(dataset, 2);
	assert.equal(resized.width, 4);
	assert.equal(resized.height, 2);
	assert.deepEqual(resized.benchmarkSourceRaster, [2, 1]);
	assert.equal(resized.frames.length, 4 * 2 * 4);
	assert.equal(resized.viewDatasets[0].frames.length, resized.frames.length);
	assert.equal(resized.previewViews[0], resized.viewDatasets[0]);
	assert.ok(resized.motionSamples.length >= 1);
	for (const packed of resized.motionSamples) assert.ok(packed < resized.width * resized.height);
});

test("full-frame kernel benchmarks can skip unused sampled-ray preprocessing", () => {
	const dataset = {
		name: "fixture",
		width: 1,
		height: 1,
		frameCount: 1,
		viewCount: 1,
		trainViewCount: 1,
		frames: Float32Array.from([0.8, 0.1, 0.1, 1]),
		backgrounds: Float32Array.from([0.1, 0.1, 0.1, 1]),
		background: Float32Array.from([0.1, 0.1, 0.1, 1]),
		cameras: [{ name: "cam00", role: "train" }],
		comparisonViewIndices: [0],
	};
	const resized = resizeDatasetForBenchmark(
		dataset,
		2,
		{ computeSamples: false },
	);
	assert.equal(resized.frames.length, 2 * 2 * 4);
	assert.equal(resized.motionSamples.length, 0);
	assert.equal(resized.staticSamples.length, 0);
	assert.deepEqual([...resized.frames.filter((_value, index) => index % 4 === 3)], [1, 1, 1, 1]);
});

test("benchmark resizing preserves compact frame construction, format, and exact RGB bytes", () => {
	const frames = Uint8Array.from([
		1, 2, 3, FRAME_WEIGHT_BYTE_SCALE,
		4, 5, 6, FRAME_WEIGHT_BYTE_SCALE,
	]);
	const backgrounds = Float32Array.from([
		1 / 255, 2 / 255, 3 / 255, 1,
		4 / 255, 5 / 255, 6 / 255, 1,
	]);
	const dataset = {
		name: "compact fixture",
		width: 2,
		height: 1,
		frameCount: 1,
		viewCount: 1,
		trainViewCount: 1,
		frames,
		frameBank: { format: FRAME_BANK_FORMAT_RGBA8, data: frames },
		backgrounds,
		backgroundBank: { format: "rgba32float/v1", data: backgrounds },
		background: backgrounds,
		cameras: [{ name: "cam00", role: "train" }],
		comparisonViewIndices: [0],
	};
	const resized = resizeDatasetForBenchmark(dataset, 2, { computeSamples: false });
	assert.ok(resized.frames instanceof Uint8Array);
	assert.equal(resized.frameBank.format, FRAME_BANK_FORMAT_RGBA8);
	assert.equal(resized.frameBank.data, resized.frames);
	assert.deepEqual([...resized.frames], [
		1, 2, 3, 127, 1, 2, 3, 127, 4, 5, 6, 127, 4, 5, 6, 127,
		1, 2, 3, 127, 1, 2, 3, 127, 4, 5, 6, 127, 4, 5, 6, 127,
	]);
	assert.equal(resized.viewDatasets[0].frameBank.data.buffer, resized.frames.buffer);
});
