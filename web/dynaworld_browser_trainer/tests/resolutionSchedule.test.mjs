import assert from "node:assert/strict";
import test from "node:test";
import {
	PROGRESSIVE_RESOLUTION_SWITCH_STEP,
	RESOLUTION_MODE_PROGRESSIVE,
	assertResolutionContinuationCompatible,
	initialResolutionPreset,
	resolutionStageForStep,
} from "../resolutionSchedule.js";

function dataset(width, height) {
	return {
		width,
		height,
		frameCount: 16,
		viewCount: 3,
		trainViewCount: 2,
		trainViewIndices: [0, 1],
		heldoutViewIndex: 2,
		frameIndices: [0, 1, 2],
		seedPointCount: 1,
		seedPoints: new Float32Array([1, 2, 3, 0.2, 0.3, 0.4]),
		datasetContract: {
			pose_source: "canonical",
			anchor_camera: "cam00",
			coordinate_convention: "opencv",
		},
		cameras: ["cam00", "cam01", "cam02"].map((name, index) => ({
			name,
			role: index === 2 ? "heldout" : "train",
			intrinsics: new Float32Array([1, 1, 0.5, 0.5]),
			worldToCamera: new Float32Array([1, 0, 0, index, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
		})),
	};
}

test("progressive resolution keeps the coarse stage through step 8191", () => {
	assert.equal(PROGRESSIVE_RESOLUTION_SWITCH_STEP, 8192);
	assert.equal(initialResolutionPreset(RESOLUTION_MODE_PROGRESSIVE), "96x72");
	assert.equal(resolutionStageForStep(RESOLUTION_MODE_PROGRESSIVE, 8191).preset, "96x72");
	assert.equal(resolutionStageForStep(RESOLUTION_MODE_PROGRESSIVE, 8192).preset, "384x288");
	assert.deepEqual(resolutionStageForStep("384x288", 100), {
		preset: "384x288", progressive: false, transitionStep: null,
	});
});

test("continuation accepts only the same calibrated world at 4x resolution", () => {
	const coarse = dataset(96, 72);
	const fine = dataset(384, 288);
	assert.equal(assertResolutionContinuationCompatible(coarse, fine), true);
	fine.cameras[1].worldToCamera[3] += 0.01;
	assert.throws(() => assertResolutionContinuationCompatible(coarse, fine), /camera mismatch/);
});

test("continuation rejects mismatched topology and dimensions", () => {
	const coarse = dataset(96, 72);
	assert.throws(() => assertResolutionContinuationCompatible(coarse, dataset(192, 144)), /exactly 4x/);
	const fine = dataset(384, 288);
	fine.seedPoints[0] += 1;
	assert.throws(() => assertResolutionContinuationCompatible(coarse, fine), /seed geometry/);
	assert.throws(() => resolutionStageForStep(RESOLUTION_MODE_PROGRESSIVE, -1), /non-negative/);
});
