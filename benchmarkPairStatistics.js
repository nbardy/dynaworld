const MATCHED_OPTION_KEYS = Object.freeze([
	"experiment",
	"variant",
	"splats",
	"capacity",
	"scale",
	"warmup",
	"steps",
	"profiles",
	"checkpointPrecision",
	"checkpointStride",
	"checkpointOrder",
	"projectionLayout",
	"ssimLayout",
	"sharePairPacket",
	"backwardGranularity",
	"tileSize",
	"tileCapacity",
	"maxRoundCv",
]);

function relativeDifference(left, right) {
	return Math.abs(left - right) / ((Math.abs(left) + Math.abs(right)) * 0.5);
}

function resultMap(report) {
	return new Map(report.results.map((result) => [result.id, result]));
}

function compareWorkload(left, right) {
	const mismatches = [];
	if (left.experiment.id !== right.experiment.id) {
		mismatches.push(`experiment.id: ${left.experiment.id} != ${right.experiment.id}`);
	}
	for (const key of MATCHED_OPTION_KEYS) {
		if (JSON.stringify(left.options[key]) !== JSON.stringify(right.options[key])) {
			mismatches.push(`options.${key}: ${left.options[key]} != ${right.options[key]}`);
		}
	}
	for (const key of ["name", "width", "height", "viewCount", "trainViewCount", "frameCount"]) {
		if (JSON.stringify(left.dataset[key]) !== JSON.stringify(right.dataset[key])) {
			mismatches.push(`dataset.${key}: ${left.dataset[key]} != ${right.dataset[key]}`);
		}
	}
	for (const [label, leftValue, rightValue] of [
		["host policy", left.hostDiagnostics?.policy, right.hostDiagnostics?.policy],
		["host thresholds", left.hostDiagnostics?.thresholds, right.hostDiagnostics?.thresholds],
		[
			"preflight CPU sampling method",
			left.hostDiagnostics?.preflight?.cpuBusySource,
			right.hostDiagnostics?.preflight?.cpuBusySource,
		],
		[
			"postflight CPU sampling method",
			left.hostDiagnostics?.postflight?.cpuBusySource,
			right.hostDiagnostics?.postflight?.cpuBusySource,
		],
		[
			"postflight cooldown",
			left.hostDiagnostics?.postflightCooldownMs,
			right.hostDiagnostics?.postflightCooldownMs,
		],
	]) {
		if (JSON.stringify(leftValue) !== JSON.stringify(rightValue)) {
			mismatches.push(`${label}: ${JSON.stringify(leftValue)} != ${JSON.stringify(rightValue)}`);
		}
	}
	const leftIds = [...resultMap(left).keys()].sort();
	const rightIds = [...resultMap(right).keys()].sort();
	if (JSON.stringify(leftIds) !== JSON.stringify(rightIds)) {
		mismatches.push(`result ids: ${leftIds.join(",")} != ${rightIds.join(",")}`);
	}
	return mismatches;
}

export function summarizeBenchmarkPair(left, right, {
	maxWallSpeedupDrift = 0.05,
	maxGpuSpeedupDrift = 0.10,
	maxVariantThroughputDrift = 0.05,
} = {}) {
	const reasons = compareWorkload(left, right);
	const orders = new Set([left.experiment.order, right.experiment.order]);
	if (
		orders.size !== 2
		|| !orders.has("control-first")
		|| !orders.has("candidate-first")
	) {
		reasons.push("Pair must contain one control-first and one candidate-first run.");
	}
	for (const [index, report] of [left, right].entries()) {
		if (report.validity?.promotable !== true) {
			reasons.push(`Run ${index + 1} is not individually promotable.`);
		}
		if (!report.comparison) reasons.push(`Run ${index + 1} has no matched comparison.`);
	}
	const leftResults = resultMap(left);
	const rightResults = resultMap(right);
	const variantThroughput = [...leftResults.keys()]
		.filter((id) => rightResults.has(id))
		.map((id) => {
			const leftThroughput = leftResults.get(id).stepsPerSecond;
			const rightThroughput = rightResults.get(id).stepsPerSecond;
			return {
				id,
				leftStepsPerSecond: leftThroughput,
				rightStepsPerSecond: rightThroughput,
				relativeDrift: relativeDifference(leftThroughput, rightThroughput),
			};
		});
	const wallSpeedups = [
		left.comparison?.candidateThroughputSpeedup,
		right.comparison?.candidateThroughputSpeedup,
	];
	const gpuSpeedups = [
		left.comparison?.candidateGpuTimeSpeedup,
		right.comparison?.candidateGpuTimeSpeedup,
	];
	const wallSpeedupDrift = wallSpeedups.every(Number.isFinite)
		? relativeDifference(...wallSpeedups) : Number.NaN;
	const gpuSpeedupDrift = gpuSpeedups.every(Number.isFinite)
		? relativeDifference(...gpuSpeedups) : Number.NaN;
	const maxObservedVariantThroughputDrift = variantThroughput.length
		? Math.max(...variantThroughput.map((variant) => variant.relativeDrift))
		: Number.NaN;
	if (!Number.isFinite(wallSpeedupDrift) || wallSpeedupDrift > maxWallSpeedupDrift) {
		reasons.push(
			`Wall-speedup drift ${wallSpeedupDrift} exceeds ${maxWallSpeedupDrift}.`,
		);
	}
	if (!Number.isFinite(gpuSpeedupDrift) || gpuSpeedupDrift > maxGpuSpeedupDrift) {
		reasons.push(
			`GPU-speedup drift ${gpuSpeedupDrift} exceeds ${maxGpuSpeedupDrift}.`,
		);
	}
	if (
		!Number.isFinite(maxObservedVariantThroughputDrift)
		|| maxObservedVariantThroughputDrift > maxVariantThroughputDrift
	) {
		reasons.push(
			`Variant throughput drift ${maxObservedVariantThroughputDrift} exceeds `
			+ `${maxVariantThroughputDrift}.`,
		);
	}
	return {
		schema: "dynaworld-browser-tiled-kernel-benchmark-pair/v1",
		recordedAt: new Date().toISOString(),
		experiment: left.experiment.id,
		workload: {
			splats: left.options.splats,
			capacity: left.options.capacity,
			raster: [left.dataset.width, left.dataset.height],
			measuredSteps: left.options.steps,
			tileSize: left.options.tileSize,
			tileCapacity: left.options.tileCapacity,
		},
		runs: [left, right].map((report) => ({
			recordedAt: report.recordedAt,
			order: report.experiment.order,
			promotable: report.validity?.promotable === true,
			preflightGpuUtilizationPercent:
				report.hostDiagnostics?.preflight?.appleGpu?.deviceUtilizationPercent,
			postflightGpuUtilizationPercent:
				report.hostDiagnostics?.postflight?.appleGpu?.deviceUtilizationPercent,
			candidateThroughputSpeedup:
				report.comparison?.candidateThroughputSpeedup,
			candidateGpuTimeSpeedup:
				report.comparison?.candidateGpuTimeSpeedup,
		})),
		drift: {
			wallSpeedupRelative: wallSpeedupDrift,
			gpuSpeedupRelative: gpuSpeedupDrift,
			maxVariantThroughputRelative: maxObservedVariantThroughputDrift,
			variantThroughput,
		},
		thresholds: {
			maxWallSpeedupDrift,
			maxGpuSpeedupDrift,
			maxVariantThroughputDrift,
		},
		validity: {
			promotable: reasons.length === 0,
			reasons,
		},
	};
}

export { MATCHED_OPTION_KEYS };
