function mean(values) {
	return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function populationStandardDeviation(values, valueMean) {
	const variance = values.reduce(
		(sum, value) => sum + (value - valueMean) ** 2,
		0,
	) / values.length;
	return Math.sqrt(variance);
}

function summarizeExecutionPositions(samples) {
	const byPosition = new Map();
	for (const sample of samples) {
		const values = byPosition.get(sample.executionPosition) ?? [];
		values.push(sample.stepsPerSecond);
		byPosition.set(sample.executionPosition, values);
	}
	const positions = [...byPosition.entries()]
		.sort(([left], [right]) => left - right)
		.map(([position, values]) => ({
			position,
			samples: values.length,
			meanStepsPerSecond: mean(values),
		}));
	if (positions.length !== 2) {
		return {
			positions,
			secondToFirstRatio: null,
			relativeDifference: null,
		};
	}
	const first = positions[0].meanStepsPerSecond;
	const second = positions[1].meanStepsPerSecond;
	return {
		positions,
		secondToFirstRatio: second / first,
		relativeDifference: Math.abs(second - first) / ((first + second) * 0.5),
	};
}

export function summarizeRoundStability(rounds, maxCoefficientOfVariation = 0.10) {
	const samples = rounds
		.filter((round) => (
			Number.isFinite(round.elapsedMs)
			&& round.elapsedMs > 0
			&& Number.isFinite(round.steps)
			&& round.steps > 0
		))
		.map((round) => ({
			round: round.round,
			steps: round.steps,
			elapsedMs: round.elapsedMs,
			executionPosition: round.executionPosition,
			stepsPerSecond: round.steps * 1000 / round.elapsedMs,
		}));
	if (samples.length < 2) {
		return {
			supported: false,
			reason: "At least two non-empty measurement rounds are required.",
			samples,
			maxAllowedCoefficientOfVariation: maxCoefficientOfVariation,
			stable: false,
			executionPositionEffect: summarizeExecutionPositions(samples),
		};
	}
	const throughputs = samples.map((sample) => sample.stepsPerSecond);
	const meanStepsPerSecond = mean(throughputs);
	const standardDeviationStepsPerSecond = populationStandardDeviation(
		throughputs,
		meanStepsPerSecond,
	);
	const coefficientOfVariation = standardDeviationStepsPerSecond / meanStepsPerSecond;
	const minimumStepsPerSecond = Math.min(...throughputs);
	const maximumStepsPerSecond = Math.max(...throughputs);
	return {
		supported: true,
		samples,
		meanStepsPerSecond,
		standardDeviationStepsPerSecond,
		coefficientOfVariation,
		minimumStepsPerSecond,
		maximumStepsPerSecond,
		maxToMinRatio: maximumStepsPerSecond / minimumStepsPerSecond,
		lastToFirstRatio: throughputs.at(-1) / throughputs[0],
		maxAllowedCoefficientOfVariation: maxCoefficientOfVariation,
		stable: coefficientOfVariation <= maxCoefficientOfVariation,
		executionPositionEffect: summarizeExecutionPositions(samples),
	};
}
