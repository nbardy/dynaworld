import test from "node:test";
import assert from "node:assert/strict";

import {
	TILED_PARITY_DIAGNOSTIC_COMPONENTS,
	TILED_PARITY_GRADIENT_FAMILIES,
	selectGradientChannels,
	summarizeForwardParity,
} from "../tiledParityHarness.js";

test("gradient selections cover trainable families and exclude diagnostic slots", () => {
	const gradients = new Float32Array(2 * 24);
	for (let index = 0; index < gradients.length; index += 1) gradients[index] = index / 100;
	gradients[11] = 1000;
	gradients[15] = 1000;
	const selected = selectGradientChannels(gradients, 2);

	assert.equal(selected.length, TILED_PARITY_GRADIENT_FAMILIES.length);
	assert.deepEqual(TILED_PARITY_DIAGNOSTIC_COMPONENTS, [11, 15]);
	assert.ok(selected.every(({ component }) =>
		!TILED_PARITY_DIAGNOSTIC_COMPONENTS.includes(component)));
	assert.deepEqual(selected.map(({ family }) => family),
		TILED_PARITY_GRADIENT_FAMILIES.map(({ name }) => name));
});

test("forward summary reports RGB and alpha errors separately", () => {
	const gpu = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
	const rgb = new Float32Array([0.1, 0.2, 0.3, 0.5, 0.6, 0.65]);
	const coverage = new Float32Array([0.4, 0.75]);
	const summary = summarizeForwardParity(gpu, rgb, coverage);

	assert.equal(summary.rgb.maxAbs, Math.abs(gpu[6] - rgb[5]));
	assert.equal(summary.alpha.maxAbs, Math.abs(gpu[7] - coverage[1]));
	assert.equal(summary.rgb.worstIndex, 6);
	assert.equal(summary.alpha.worstIndex, 7);
});
