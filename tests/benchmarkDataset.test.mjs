import assert from "node:assert/strict";
import test from "node:test";
import {
	resizeDatasetForBenchmark,
	resizePackedRgbaNearest,
} from "../benchmarkDataset.js";

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
