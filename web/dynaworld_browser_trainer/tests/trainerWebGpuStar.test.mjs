import assert from "node:assert/strict";
import test from "node:test";
import {
	AFFINE_STAR_BROWSER_CONTRACT,
	STAR_AFFINE_OPTIMIZED_COMPONENTS,
	affineStarLossAndGradients,
	compileCameraSpaceWorldTubes,
	createAffineQ,
	createTinyAffineStarFixture,
	finiteDifferenceAffineStar,
	renderAffineStarSample,
} from "../trainerWebGpuStar.js";

test("affine camera-space compiler produces a moving positive-definite UVT trace", () => {
	const state = compileCameraSpaceWorldTubes(
		{ fx: 20, fy: 24, cx: 8, cy: 7 },
		[{ position: [0.2, -0.1, 2], velocity: [0.04, 0.02, 0], sigmaPixels: [2, 3], sigmaTime: 4, color: [1, 0, 0], opacity: 0.7 }],
	);
	assert.equal(state[0], 10);
	assert.ok(Math.abs(state[1] - 5.8) < 1e-6);
	assert.ok(state[6] < 0);
	assert.ok(state[8] < 0);
	assert.equal(state[12], 2);
	const determinant2 = state[4] * state[7] - state[5] ** 2;
	const determinant3 = state[4] * (state[7] * state[9] - state[8] ** 2)
		- state[5] * (state[5] * state[9] - state[8] * state[6])
		+ state[6] * (state[5] * state[8] - state[7] * state[6]);
	assert.ok(state[4] > 0 && determinant2 > 0 && determinant3 > 0);
	assert.deepEqual(createAffineQ({ precisionU: 0.25, precisionV: 0.5, temporalPrecision: 0.1 }), [0.25, 0, -0, 0.5, -0, 0.1]);
});

test("stable conditional-depth order changes honest source-over color", () => {
	const { trueState } = createTinyAffineStarFixture();
	const sample = { x: 8.5, y: 8.5, t: 0 };
	const frontRed = renderAffineStarSample(trueState, sample);
	const swapped = trueState.slice();
	[swapped[12], swapped[32]] = [swapped[32], swapped[12]];
	const frontBlue = renderAffineStarSample(swapped, sample);
	assert.ok(frontRed[0] > frontBlue[0]);
	assert.ok(frontBlue[2] > frontRed[2]);
});

test("shared adjoint matches finite differences for every optimized parameter family", () => {
	const { initialState, samples } = createTinyAffineStarFixture();
	const analytic = affineStarLossAndGradients(initialState, samples).gradients;
	const numerical = finiteDifferenceAffineStar(initialState, samples, { epsilon: 1e-3 });
	let maxAbsoluteError = 0;
	let maxRelativeError = 0;
	for (let tube = 0; tube < initialState.length / 20; tube += 1) {
		for (const component of STAR_AFFINE_OPTIMIZED_COMPONENTS) {
			const index = tube * 20 + component;
			const absolute = Math.abs(analytic[index] - numerical[index]);
			const relative = absolute / Math.max(1e-5, Math.abs(analytic[index]), Math.abs(numerical[index]));
			maxAbsoluteError = Math.max(maxAbsoluteError, absolute);
			maxRelativeError = Math.max(maxRelativeError, relative);
		}
	}
	assert.ok(maxAbsoluteError < 2e-5, `max absolute error ${maxAbsoluteError}`);
	assert.ok(maxRelativeError < 2e-3, `max relative error ${maxRelativeError}`);
});

test("browser contract names the canonical lane and its deliberate omissions", () => {
	assert.match(AFFINE_STAR_BROWSER_CONTRACT.name, /affine STAR UVT \/ World Tubes/);
	assert.match(AFFINE_STAR_BROWSER_CONTRACT.visibility, /depth order/);
	assert.ok(AFFINE_STAR_BROWSER_CONTRACT.omissions.some((item) => item.includes("projective")));
	assert.ok(AFFINE_STAR_BROWSER_CONTRACT.matchedCostKnobs.includes("sampleCount"));
});
