import assert from "node:assert/strict";
import test from "node:test";
import { evaluateDynamicGsSample, makeDynamicGsState } from "../trainerWebGpuDynamicGs.js";

const camera = { intrinsics: [0.8, 0.8, 0.5, 0.5], worldToCamera: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
const dataset = { frameCount: 2, cameras: [{ ...camera, role: "train" }], seedPoints: [{ xyz: [0, 0, 2], rgb: [0.7, 0.3, 0.2] }, { xyz: [0.02, 0, 3], rgb: [0.2, 0.6, 0.8] }] };

test("uses independent per-frame state and ascending camera-depth order", () => {
	const state = makeDynamicGsState(dataset, { splatCount: 2 });
	assert.equal(state.length, 2 * 2 * 16);
	assert.deepEqual(evaluateDynamicGsSample({ state, splatCount: 2, frame: 0, camera, u: 0.5, v: 0.5 }).order, [0, 1]);
	state[2 * 16 + 12] += 1;
	assert.notDeepEqual(
		evaluateDynamicGsSample({ state, splatCount: 2, frame: 0, camera, u: 0.5, v: 0.5 }).color,
		evaluateDynamicGsSample({ state, splatCount: 2, frame: 1, camera, u: 0.5, v: 0.5 }).color,
	);
});

test("analytic backward matches central differences for every optimized channel", () => {
	const state = makeDynamicGsState(dataset, { splatCount: 2 }); const target = [0.15, 0.25, 0.35];
	const result = evaluateDynamicGsSample({ state, splatCount: 2, frame: 0, camera, u: 0.503, v: 0.498, target }); const epsilon = 1e-4;
	for (let splat = 0; splat < 2; splat += 1) for (let channel = 0; channel < 4; channel += 1) {
		const parameter = splat * 16 + (channel === 3 ? 3 : 12 + channel); const original = state[parameter];
		state[parameter] = original + epsilon; const plus = evaluateDynamicGsSample({ state, splatCount: 2, frame: 0, camera, u: 0.503, v: 0.498, target }).loss;
		state[parameter] = original - epsilon; const minus = evaluateDynamicGsSample({ state, splatCount: 2, frame: 0, camera, u: 0.503, v: 0.498, target }).loss; state[parameter] = original;
		const numeric = (plus - minus) / (2 * epsilon); const analytic = result.gradients[splat * 4 + channel];
		assert.ok(Math.abs(numeric - analytic) < 3e-4, `splat ${splat} channel ${channel}: numeric=${numeric}, analytic=${analytic}`);
	}
});

test("anisotropic rotation changes projected support", () => {
	const state = makeDynamicGsState(dataset, { splatCount: 1 });
	const horizontal = evaluateDynamicGsSample({ state, splatCount: 1, frame: 0, camera, u: 0.54, v: 0.5 }).color[0];
	state[8 + 2] = Math.SQRT1_2; state[8 + 3] = Math.SQRT1_2;
	const rotated = evaluateDynamicGsSample({ state, splatCount: 1, frame: 0, camera, u: 0.54, v: 0.5 }).color[0];
	assert.notEqual(horizontal, rotated);
});
