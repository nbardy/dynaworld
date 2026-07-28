import assert from "node:assert/strict";
import test from "node:test";
import {
	packSamplesByCamera, resolveCamerasPerStep, resolveRenderViewIndices, resolveTrainViewIndices,
	rotatingTrainViewBatch,
} from "../trainerWebGpu3d.js";

function interleavedDataset() {
	return {
		trainViewCount: 17,
		heldoutViewIndex: 6,
		cameras: Array.from({ length: 18 }, (_, index) => ({
			name: `cam${index}`,
			role: index === 6 ? "heldout" : "train",
		})),
	};
}

test("defaults to four rotating cameras and covers all 17 train views", () => {
	const trainViews = resolveTrainViewIndices(interleavedDataset());
	assert.equal(resolveCamerasPerStep(trainViews.length), 4);
	const batches = Array.from({ length: 5 }, (_, step) => rotatingTrainViewBatch(trainViews, step));
	assert.deepEqual(batches.map(({ indices }) => indices), [
		[0, 1, 2, 3],
		[4, 5, 7, 8],
		[9, 10, 11, 12],
		[13, 14, 15, 16],
		[17, 0, 1, 2],
	]);
	assert.deepEqual([...new Set(batches.flatMap(({ indices }) => indices))].sort((a, b) => a - b), trainViews);
});

test("heldout camera is excluded from every rotating membership set", () => {
	const dataset = interleavedDataset();
	const trainViews = resolveTrainViewIndices(dataset);
	for (let step = 0; step < 100; step += 1) {
		assert.equal(rotatingTrainViewBatch(trainViews, step, 4).indices.includes(dataset.heldoutViewIndex), false);
	}
});

test("focused sample ranges retain camera membership without rejection scans", () => {
	const dataset = { width: 2, height: 1, frameCount: 2, cameras: [{}, {}, {}],
		motionSamples: new Uint32Array([0, 9, 4, 11]), staticSamples: new Uint32Array([3, 8, 7]) };
	const { indices, ranges } = packSamplesByCamera(dataset);
	assert.deepEqual([...indices], [0, 3, 4, 7, 9, 11, 8]);
	assert.deepEqual([...ranges], [0, 1, 1, 1, 2, 1, 3, 1, 4, 2, 6, 1]);
});

test("K at or above the train count preserves the full ordered membership", () => {
	const trainViews = [0, 1, 2];
	assert.deepEqual(rotatingTrainViewBatch(trainViews, 37, 99), { start: 0, indices: trainViews });
	assert.equal(resolveCamerasPerStep(3), 3);
});

test("render defaults to representative train views plus the true heldout index", () => {
	const dataset = interleavedDataset();
	assert.deepEqual(resolveRenderViewIndices(dataset), [0, 9, 6]);
	assert.deepEqual(resolveRenderViewIndices(dataset, [3, 12, 6]), [3, 12, 6]);
});

test("legacy datasets without roles retain first-N train and first-three render fallbacks", () => {
	const dataset = { trainViewCount: 2, cameras: [{}, {}, {}], heldoutViewIndex: -1 };
	assert.deepEqual(resolveTrainViewIndices(dataset), [0, 1]);
	assert.deepEqual(resolveRenderViewIndices(dataset), [0, 1, 2]);
});
