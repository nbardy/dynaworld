import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
	TILED_PARITY_DIAGNOSTIC_COMPONENTS,
	TILED_PARITY_GRADIENT_FAMILIES,
	extractTiledParityTarget,
	makeTiledParityFixtureParameters,
	selectGradientChannels,
	selectTiledParityTrainingStep,
	summarizeForwardParity,
	summarizeGradientParity,
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

test("parity fixture makes spatial covariance and harmonic motion non-degenerate", () => {
	const initial = new Float32Array(2 * 24);
	for (let splatIndex = 0; splatIndex < 2; splatIndex += 1) {
		const base = splatIndex * 24;
		initial[base + 12] = Math.log(0.1);
		initial[base + 13] = Math.log(0.1);
		initial[base + 14] = Math.log(0.1);
		initial[base + 19] = 1;
	}
	const fixture = makeTiledParityFixtureParameters(initial);

	assert.notStrictEqual(fixture, initial);
	assert.deepEqual([...initial.subarray(8, 11)], [0, 0, 0]);
	for (let splatIndex = 0; splatIndex < 2; splatIndex += 1) {
		const base = splatIndex * 24;
		const scales = [12, 13, 14].map((component) => Math.exp(fixture[base + component]));
		const quaternion = [...fixture.subarray(base + 16, base + 20)];
		assert.ok(Math.max(...scales) / Math.min(...scales) > 2);
		assert.ok(Math.hypot(...quaternion.slice(0, 3)) > 0.1);
		assert.ok(Math.abs(Math.hypot(...quaternion) - 1) < 1e-6);
		assert.ok(Math.hypot(...fixture.subarray(base + 8, base + 11)) > 0);
	}
});

test("parity training pair excites linear and harmonic temporal bases", () => {
	const selected = selectTiledParityTrainingStep([0, 2, 4], 16);

	assert.ok(selected.step > 0);
	assert.ok(Math.abs(selected.linearBasis) >= 0.5);
	assert.ok(Math.abs(selected.harmonicBasis) >= 0.5);
});

test("parity pair selection can require a non-uniform motion-weight map", () => {
	const dataset = {
		width: 2,
		height: 1,
		frameCount: 16,
		frames: new Float32Array(2 * 16 * 2 * 4).fill(1),
	};
	const variedPair = selectTiledParityTrainingStep([0, 1], 16);
	const offset = (variedPair.viewIndex * dataset.frameCount + variedPair.frameIndex)
		* dataset.width * dataset.height * 4;
	dataset.frames[offset + 3] = 0.75;
	dataset.frames[offset + 7] = 1.25;

	const selected = selectTiledParityTrainingStep([0, 1], 16, dataset);
	assert.ok(selected.pixelWeightRange >= 0.5);
});

test("gradient parity requires every intended family to be active", () => {
	const checks = TILED_PARITY_GRADIENT_FAMILIES.map(({ name }) => ({
		family: name,
		active: true,
		pass: true,
	}));
	assert.equal(summarizeGradientParity(checks).pass, true);

	checks.find(({ family }) => family === "rotation").active = false;
	const summary = summarizeGradientParity(checks);
	assert.equal(summary.activeCount, TILED_PARITY_GRADIENT_FAMILIES.length - 1);
	assert.equal(summary.pass, false);
});

test("selected parity target preserves normalized alpha motion weights", () => {
	const dataset = {
		width: 2,
		height: 1,
		frameCount: 2,
		frames: new Float32Array([
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 1.6,
			0.8, 0.7, 0.6, 1.25,
			0.4, 0.3, 0.2, 0.75,
		]),
	};
	const target = extractTiledParityTarget(dataset, 0, 1);

	assert.deepEqual(target.rgb, new Float32Array([
		0.8, 0.7, 0.6,
		0.4, 0.3, 0.2,
	]));
	assert.deepEqual(target.pixelWeights, new Float32Array([1.25, 0.75]));
});

test("live parity explicitly exercises motion weighting instead of inheriting its default", () => {
	const source = readFileSync(new URL("../tiledParityHarness.js", import.meta.url), "utf8");
	assert.match(source, /ssimRadius:\s*PARITY_SSIM_RADIUS,\s*motionWeighting:\s*true/);
	assert.match(source, /maximumWeight - minimumWeight < 1e-3/);
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
