import assert from "node:assert/strict";
import test from "node:test";
import {
	BROWSER_ADAM_BETA1,
	BROWSER_ADAM_BETA2,
	BROWSER_ADAM_EPSILON,
	DENSITY_STAT_DECAY,
	LEARNING_RATE_DECAY_STEPS,
	browserLearningRates,
	learningRateMultipliers,
} from "../trainingSchedule.js";

test("browser optimizer defaults span a complete multicamera cycle", () => {
	assert.equal(BROWSER_ADAM_BETA1, 0.9);
	assert.equal(BROWSER_ADAM_BETA2, 0.999);
	assert.equal(BROWSER_ADAM_EPSILON, 1e-8);
	assert.equal(DENSITY_STAT_DECAY, 0.999);
});

test("learning-rate schedule decays geometry 100x and appearance 10x", () => {
	assert.equal(LEARNING_RATE_DECAY_STEPS, 120000);
	assert.deepEqual(learningRateMultipliers(0), { geometry: 1, appearance: 1, progress: 0 });
	const halfway = learningRateMultipliers(60000);
	assert.ok(Math.abs(halfway.geometry - 0.1) < 1e-12);
	assert.ok(Math.abs(halfway.appearance - Math.sqrt(0.1)) < 1e-12);
	assert.deepEqual(learningRateMultipliers(120000), {
		geometry: 0.01,
		appearance: 0.1,
		progress: 1,
	});
	assert.deepEqual(learningRateMultipliers(180000), learningRateMultipliers(120000));
	assert.deepEqual(learningRateMultipliers(180000, false), {
		geometry: 1,
		appearance: 1,
		progress: 0,
	});
	assert.throws(() => learningRateMultipliers(-1), /non-negative/);
});

test("resolved family rates preserve the former step-zero base values", () => {
	const rates = browserLearningRates(1.25, 0);
	assert.equal(rates.position, 1.25 * 0.00035);
	assert.equal(rates.motion, 1.25 * 0.0002);
	assert.equal(rates.color, 1.25 * 0.0015);
	assert.equal(rates.opacity, 1.25 * 0.0008);
	const final = browserLearningRates(1.25, 120000);
	assert.equal(final.position, rates.position * 0.01);
	assert.equal(final.color, rates.color * 0.1);
	assert.equal(browserLearningRates(0, 0).position, 0);
	assert.throws(() => browserLearningRates(-1, 0), /non-negative/);
});
