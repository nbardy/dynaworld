import assert from "node:assert/strict";
import test from "node:test";
import { summarizeBenchmarkPair } from "../benchmarkPairStatistics.js";

function report(order, {
	controlThroughput = 700,
	candidateThroughput = 1200,
	wallSpeedup = candidateThroughput / controlThroughput,
	gpuSpeedup = 1.7,
	promotable = true,
} = {}) {
	return {
		recordedAt: "2026-07-30T19:00:00.000Z",
		options: {
			experiment: "backward",
			variant: "both",
			splats: 8192,
			capacity: 8192,
			scale: 1,
			frameBank: "f32",
			warmup: 32,
			steps: 128,
			profiles: 5,
			checkpointPrecision: "packed-f16",
			checkpointStride: 16,
			checkpointOrder: "pixel-major",
			projectionLayout: "split-compact",
			projectionVjpPrecision: "f32",
			ssimLayout: "separable",
			sharePairPacket: false,
			backwardGranularity: "checkpoint-block",
			tileSize: 8,
			tileCapacity: 1024,
			maxRoundCv: 0.1,
		},
		dataset: {
			name: "fixture",
			width: 96,
			height: 72,
			viewCount: 18,
			trainViewCount: 17,
			frameCount: 16,
		},
		experiment: { id: "backward", order },
		results: [
			{ id: "direct-3d", stepsPerSecond: controlThroughput },
			{ id: "staged-project3d", stepsPerSecond: candidateThroughput },
		],
		comparison: {
			candidateThroughputSpeedup: wallSpeedup,
			candidateGpuTimeSpeedup: gpuSpeedup,
		},
		validity: { promotable },
		hostDiagnostics: {
			policy: "fail",
			thresholds: { maxRoundCv: 0.1 },
			postflightCooldownMs: 10000,
			preflight: {
				cpuBusySource: "top-second-sample",
				appleGpu: { deviceUtilizationPercent: 0 },
			},
			postflight: {
				cpuBusySource: "top-second-sample",
				appleGpu: { deviceUtilizationPercent: 1 },
			},
		},
	};
}

test("pair summary accepts matched reversed-start runs with low drift", () => {
	const summary = summarizeBenchmarkPair(
		report("control-first"),
		report("candidate-first", {
			controlThroughput: 710,
			candidateThroughput: 1210,
			wallSpeedup: 1210 / 710,
			gpuSpeedup: 1.72,
		}),
	);
	assert.equal(summary.validity.promotable, true);
	assert.ok(summary.drift.wallSpeedupRelative < 0.05);
	assert.ok(summary.drift.maxVariantThroughputRelative < 0.05);
});

test("pair summary rejects internally valid runs whose speedup does not reproduce", () => {
	const summary = summarizeBenchmarkPair(
		report("control-first", { wallSpeedup: 1.25 }),
		report("candidate-first", { wallSpeedup: 1.37 }),
	);
	assert.equal(summary.validity.promotable, false);
	assert.match(summary.validity.reasons.join(" "), /Wall-speedup drift/);
});

test("pair summary rejects duplicated order and workload drift", () => {
	const left = report("control-first");
	const right = report("control-first");
	right.options.tileCapacity = 4096;
	right.hostDiagnostics.postflightCooldownMs = 5000;
	right.hostDiagnostics.preflight.cpuBusySource = "os.cpus-delta";
	const summary = summarizeBenchmarkPair(left, right);
	assert.equal(summary.validity.promotable, false);
	assert.match(summary.validity.reasons.join(" "), /tileCapacity/);
	assert.match(summary.validity.reasons.join(" "), /postflight cooldown/);
	assert.match(summary.validity.reasons.join(" "), /CPU sampling method/);
	assert.match(summary.validity.reasons.join(" "), /control-first and one candidate-first/);
});
