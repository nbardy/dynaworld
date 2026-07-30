import assert from "node:assert/strict";
import test from "node:test";
import { summarizeRoundStability } from "../benchmarkStatistics.js";

test("round stability compares throughput instead of unequal round durations", () => {
	const summary = summarizeRoundStability([
		{ round: 0, steps: 10, elapsedMs: 100, executionPosition: 0 },
		{ round: 1, steps: 20, elapsedMs: 200, executionPosition: 1 },
		{ round: 2, steps: 30, elapsedMs: 300, executionPosition: 0 },
		{ round: 3, steps: 40, elapsedMs: 400, executionPosition: 1 },
	], 0.01);
	assert.equal(summary.supported, true);
	assert.equal(summary.meanStepsPerSecond, 100);
	assert.equal(summary.coefficientOfVariation, 0);
	assert.equal(summary.stable, true);
	assert.equal(summary.executionPositionEffect.secondToFirstRatio, 1);
});

test("round stability rejects bursty measurements above the configured CV", () => {
	const summary = summarizeRoundStability([
		{ round: 0, steps: 10, elapsedMs: 100, executionPosition: 0 },
		{ round: 1, steps: 10, elapsedMs: 50, executionPosition: 1 },
	], 0.10);
	assert.equal(summary.supported, true);
	assert.ok(Math.abs(summary.coefficientOfVariation - 1 / 3) < 1e-12);
	assert.equal(summary.maxToMinRatio, 2);
	assert.equal(summary.stable, false);
	assert.ok(summary.executionPositionEffect.relativeDifference > 0.6);
});

test("one round is diagnostic rather than promotable stability evidence", () => {
	const summary = summarizeRoundStability([
		{ round: 0, steps: 10, elapsedMs: 100, executionPosition: 0 },
	]);
	assert.equal(summary.supported, false);
	assert.equal(summary.stable, false);
	assert.match(summary.reason, /At least two/);
});
