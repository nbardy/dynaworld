import { resizeDatasetForBenchmark } from "./benchmarkDataset.js?v=20260731-compactfp16-5";
import { summarizeRoundStability } from "./benchmarkStatistics.js?v=20260731-compactfp16-5";
import {
	FRAME_BANK_FORMAT_RGBA8,
	FRAME_BANK_FORMAT_RGBA32_FLOAT,
	loadPresetDataset,
} from "./dataset.js?v=20260731-compactfp16-5";
import {
	DynamicSplatWebGpu3dTiledTrainer,
	TILED_BACKWARD_GRANULARITIES,
	TILED_BACKWARD_MODES,
	TILED_CHECKPOINT_ORDERS,
	TILED_PROJECTION_LAYOUTS,
	TILED_PROJECTION_VJP_PRECISIONS,
	TILED_SSIM_LAYOUTS,
} from "./trainerWebGpu3dTiled.js?v=20260821-stablegs-ablation-1";
const EXPERIMENTS = Object.freeze({
	backward: Object.freeze([
		Object.freeze({
			id: "direct-3d",
			label: "Direct 3D VJP per tile pair",
			init: Object.freeze({
				backwardMode: TILED_BACKWARD_MODES.DIRECT_3D,
				backwardGranularity: TILED_BACKWARD_GRANULARITIES.PAIR,
				projectionLayout: TILED_PROJECTION_LAYOUTS.MONOLITHIC,
			}),
		}),
		Object.freeze({
			id: "staged-project3d",
			label: "Staged projected VJP per splat",
			init: Object.freeze({ backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D }),
		}),
	]),
	projection: Object.freeze([
		Object.freeze({
			id: "staged-monolithic",
			label: "Staged monolithic 192 B projection",
			init: Object.freeze({
				backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D,
				projectionLayout: TILED_PROJECTION_LAYOUTS.MONOLITHIC,
			}),
		}),
		Object.freeze({
			id: "staged-split-compact",
			label: "Staged split projection with selected VJP storage",
			init: Object.freeze({
				backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D,
				projectionLayout: TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT,
			}),
		}),
	]),
	precision: Object.freeze([
		Object.freeze({
			id: "staged-split-f32",
			label: "Staged split 32 B + 80 B FP32 VJP",
			init: Object.freeze({
				backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D,
				projectionLayout: TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT,
				projectionVjpPrecision: TILED_PROJECTION_VJP_PRECISIONS.F32,
			}),
		}),
		Object.freeze({
			id: "staged-split-packed-f16",
			label: "Staged split 32 B + 48 B packed-FP16 VJP",
			init: Object.freeze({
				backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D,
				projectionLayout: TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT,
				projectionVjpPrecision: TILED_PROJECTION_VJP_PRECISIONS.PACKED_F16,
			}),
		}),
	]),
	ssim: Object.freeze([
		Object.freeze({
			id: "staged-naive-2d",
			label: "Staged naive 2D SSIM",
			init: Object.freeze({
				backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D,
				ssimLayout: TILED_SSIM_LAYOUTS.NAIVE_2D,
			}),
		}),
		Object.freeze({
			id: "staged-separable",
			label: "Staged separable SSIM",
			init: Object.freeze({
				backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D,
				ssimLayout: TILED_SSIM_LAYOUTS.SEPARABLE,
			}),
		}),
	]),
	geometry: Object.freeze([
		Object.freeze({
			id: "fast-baseline",
			label: "Fast staged baseline",
			init: Object.freeze({ backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D }),
		}),
		Object.freeze({
			id: "stablegs-inspired",
			label: "Selected StableGS-inspired stack",
			init: Object.freeze({ backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D }),
		}),
	]),
});

const TRAIN_OPTIONS = Object.freeze({
	learningRate: 1.25,
	learningRateDecay: true,
	modelMode: 0,
	temporalSigma: 0.30,
	ssimRadius: 5,
	motionWeighting: false,
	randomBackground: false,
});

const form = document.querySelector("#kernelBenchmarkForm");
const status = document.querySelector("#kernelBenchmarkStatus");
const resultBody = document.querySelector("#kernelResults");
const phaseBody = document.querySelector("#phaseResults");
const jsonOutput = document.querySelector("#kernelBenchmarkJson");
const runButton = document.querySelector("#runKernelBenchmark");
const inputs = {
	experiment: document.querySelector("#kernelExperiment"),
	variant: document.querySelector("#kernelVariant"),
	order: document.querySelector("#kernelOrder"),
	splats: document.querySelector("#kernelSplats"),
	capacity: document.querySelector("#kernelCapacity"),
	scale: document.querySelector("#kernelRasterScale"),
	frameBank: document.querySelector("#kernelFrameBank"),
	warmup: document.querySelector("#kernelWarmup"),
	steps: document.querySelector("#kernelSteps"),
	profiles: document.querySelector("#kernelProfiles"),
	checkpointPrecision: document.querySelector("#kernelCheckpointPrecision"),
	checkpointStride: document.querySelector("#kernelCheckpointStride"),
	checkpointOrder: document.querySelector("#kernelCheckpointOrder"),
	projectionLayout: document.querySelector("#kernelProjectionLayout"),
	projectionVjpPrecision: document.querySelector("#kernelProjectionVjpPrecision"),
	ssimLayout: document.querySelector("#kernelSsimLayout"),
	pairPacket: document.querySelector("#kernelPairPacket"),
	backwardGranularity: document.querySelector("#kernelBackwardGranularity"),
	tileSize: document.querySelector("#kernelTileSize"),
	tileCapacity: document.querySelector("#kernelTileCapacity"),
	maxRoundCv: document.querySelector("#kernelMaxRoundCv"),
	pixelFilterMode: document.querySelector("#kernelPixelFilter"),
	opacityModel: document.querySelector("#kernelOpacityModel"),
	geometryColorWeight: document.querySelector("#kernelGeometryColorWeight"),
	crossViewDepth: document.querySelector("#kernelCrossViewDepth"),
	geometryConsistencyEvery: document.querySelector("#kernelGeometryEvery"),
	geometryDepthWeight: document.querySelector("#kernelGeometryDepthWeight"),
};

let running = false;

function integerValue(input, label) {
	const value = Number(input.value);
	if (!Number.isSafeInteger(value) || value < Number(input.min) || value > Number(input.max)) {
		throw new RangeError(`${label} must be an integer from ${input.min} through ${input.max}.`);
	}
	return value;
}

function numberValue(input, label) {
	const value = Number(input.value);
	if (!Number.isFinite(value) || value < Number(input.min) || value > Number(input.max)) {
		throw new RangeError(`${label} must be from ${input.min} through ${input.max}.`);
	}
	return value;
}

function readOptions() {
	const options = {
		experiment: inputs.experiment.value,
		variant: inputs.variant.value,
		order: inputs.order.value,
		splats: integerValue(inputs.splats, "Active splats"),
		capacity: integerValue(inputs.capacity, "Model capacity"),
		scale: integerValue(inputs.scale, "Raster scale"),
		frameBank: inputs.frameBank.value,
		warmup: integerValue(inputs.warmup, "Warmup steps"),
		steps: integerValue(inputs.steps, "Measured steps"),
		profiles: integerValue(inputs.profiles, "Profile samples"),
		checkpointPrecision: inputs.checkpointPrecision.value,
		checkpointStride: Number(inputs.checkpointStride.value),
		checkpointOrder: inputs.checkpointOrder.value,
		projectionLayout: inputs.projectionLayout.value,
		projectionVjpPrecision: inputs.projectionVjpPrecision.value,
		ssimLayout: inputs.ssimLayout.value,
		sharePairPacket: inputs.pairPacket.value === "shared",
		backwardGranularity: inputs.backwardGranularity.value,
		tileSize: Number(inputs.tileSize.value),
		tileCapacity: Number(inputs.tileCapacity.value),
		maxRoundCv: numberValue(inputs.maxRoundCv, "Maximum round CV"),
		pixelFilterMode: inputs.pixelFilterMode.value,
		opacityModel: inputs.opacityModel.value,
		geometryColorWeight: numberValue(inputs.geometryColorWeight, "Geometry color weight"),
		crossViewDepth: inputs.crossViewDepth.value === "true",
		geometryConsistencyEvery: integerValue(inputs.geometryConsistencyEvery, "Depth cadence"),
		geometryDepthWeight: numberValue(inputs.geometryDepthWeight, "Depth weight"),
	};
	if (options.capacity < options.splats) {
		throw new RangeError("Model capacity must be at least the active splat count.");
	}
	const variants = EXPERIMENTS[options.experiment];
	if (!variants) throw new RangeError("Unknown kernel experiment.");
	if (!["both", "control", "candidate"].includes(options.variant)
		&& !variants.some((variant) => variant.id === options.variant)) {
		throw new RangeError("Unknown kernel variant selection.");
	}
	if (!["control-first", "candidate-first"].includes(options.order)) {
		throw new RangeError("Unknown kernel execution order.");
	}
	if (!["f32", "rgba8"].includes(options.frameBank)) {
		throw new RangeError("Host frame bank must be f32 or rgba8.");
	}
	if (![8, 16, 32].includes(options.checkpointStride)) {
		throw new RangeError("Checkpoint stride must be 8, 16, or 32.");
	}
	if (!Object.values(TILED_CHECKPOINT_ORDERS).includes(options.checkpointOrder)) {
		throw new RangeError("Unknown checkpoint storage order.");
	}
	if (!Object.values(TILED_PROJECTION_LAYOUTS).includes(options.projectionLayout)) {
		throw new RangeError("Unknown projection packet layout.");
	}
	if (!Object.values(TILED_PROJECTION_VJP_PRECISIONS).includes(options.projectionVjpPrecision)) {
		throw new RangeError("Unknown projection VJP storage precision.");
	}
	if (!Object.values(TILED_SSIM_LAYOUTS).includes(options.ssimLayout)) {
		throw new RangeError("Unknown SSIM kernel layout.");
	}
	if (!Object.values(TILED_BACKWARD_GRANULARITIES).includes(options.backwardGranularity)) {
		throw new RangeError("Unknown backward granularity.");
	}
	if (![8, 16].includes(options.tileSize)) throw new RangeError("Tile size must be 8 or 16.");
	if (![256, 512, 1024, 2048, 4096].includes(options.tileCapacity)) {
		throw new RangeError("Tile capacity must be a supported power of two.");
	}
	if (!["legacy-floor", "mip-2d-compensated"].includes(options.pixelFilterMode)) {
		throw new RangeError("Unknown pixel filter mode.");
	}
	if (!["coupled", "dual"].includes(options.opacityModel)) {
		throw new RangeError("Unknown opacity model.");
	}
	return options;
}

function quantile(values, q) {
	if (!values.length) return Number.NaN;
	const sorted = [...values].sort((a, b) => a - b);
	const position = (sorted.length - 1) * q;
	const low = Math.floor(position); const high = Math.ceil(position);
	const fraction = position - low;
	return sorted[low] * (1 - fraction) + sorted[high] * fraction;
}

function summarizeProfiles(profiles) {
	const valid = profiles.filter((profile) => profile.supported);
	if (!valid.length) {
		return {
			supported: false,
			reason: profiles[0]?.reason ?? "No GPU timestamp samples.",
			gpuSpanMedianMs: Number.NaN,
			gpuSpanP95Ms: Number.NaN,
			phaseMedianMs: {},
			phaseP95Ms: {},
		};
	}
	const phaseNames = Object.keys(valid[0].phases);
	return {
		supported: true,
		samples: valid.length,
		totalMedianMs: quantile(valid.map((profile) => profile.totalMs), 0.5),
		totalP95Ms: quantile(valid.map((profile) => profile.totalMs), 0.95),
		gpuSpanMedianMs: quantile(valid.map((profile) => profile.gpuSpanMs), 0.5),
		gpuSpanP95Ms: quantile(valid.map((profile) => profile.gpuSpanMs), 0.95),
		phaseContract: valid[0].phaseContract,
		maintenanceIncluded: valid.every((profile) => profile.maintenanceIncluded),
		maintenanceDispatches: valid.map((profile) => profile.maintenanceDispatches),
		phaseMedianMs: Object.fromEntries(phaseNames.map((phase) => [
			phase,
			quantile(valid.map((profile) => profile.phases[phase]), 0.5),
		])),
		phaseP95Ms: Object.fromEntries(phaseNames.map((phase) => [
			phase,
			quantile(valid.map((profile) => profile.phases[phase]), 0.95),
		])),
	};
}

async function submitAndDrain(trainer, steps) {
	const startedAt = performance.now();
	for (let step = 0; step < steps; step += 1) trainer.trainStep(TRAIN_OPTIONS);
	await trainer.device.queue.onSubmittedWorkDone();
	return performance.now() - startedAt;
}

async function validateFirstStep(trainer) {
	if (!trainer.firstStepValidation) return;
	const error = await trainer.firstStepValidation;
	trainer.firstStepValidation = null;
	if (error) throw new Error(`First tiled submission failed: ${error.message}`);
}

function setState(message, state = "running") {
	status.textContent = message;
	status.dataset.state = state;
	document.documentElement.dataset.kernelBenchmarkState = state;
}

function number(value, digits = 2) {
	return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

async function initializeVariant(variant, dataset, options) {
	const trainer = new DynamicSplatWebGpu3dTiledTrainer(null);
	const selectedAblations = options.experiment === "geometry" && variant.id === "fast-baseline"
		? {
			pixelFilterMode: "legacy-floor", opacityModel: "coupled",
			geometryColorWeight: 0, crossViewDepth: false,
			geometryConsistencyEvery: options.geometryConsistencyEvery,
			geometryDepthWeight: options.geometryDepthWeight,
		} : {
			pixelFilterMode: options.pixelFilterMode,
			opacityModel: options.opacityModel,
			geometryColorWeight: options.geometryColorWeight,
			crossViewDepth: options.crossViewDepth,
			geometryConsistencyEvery: options.geometryConsistencyEvery,
			geometryDepthWeight: options.geometryDepthWeight,
		};
	await trainer.init(dataset, {
		splatCount: options.splats,
		growthCapacity: options.capacity,
		checkpointPrecision: options.checkpointPrecision,
		checkpointStride: options.checkpointStride,
		checkpointOrder: options.checkpointOrder,
		projectionLayout: options.projectionLayout,
		projectionVjpPrecision: options.projectionVjpPrecision,
		ssimLayout: options.ssimLayout,
		sharePairPacket: options.sharePairPacket,
		backwardGranularity: options.backwardGranularity,
		tileSize: options.tileSize,
		tileCapacity: options.tileCapacity,
		profileGpu: true,
		backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D,
		...selectedAblations,
		...variant.init,
	});
	await trainer.device.queue.onSubmittedWorkDone();
	await submitAndDrain(trainer, options.warmup);
	await validateFirstStep(trainer);
	return { variant, trainer, elapsedMs: 0, rounds: [], profiles: [] };
}

async function summarizeVariant(context, dataset, options) {
	const { variant, trainer, elapsedMs, rounds, profiles } = context;
	const loss = await trainer.readLoss();
	const roundStability = summarizeRoundStability(rounds, options.maxRoundCv);
	const overflow = trainer.lastLossBreakdown?.tileOverflowTotal ?? 0;
	const halfSaturations =
		trainer.lastLossBreakdown?.projectionVjpHalfSaturationsTotal ?? 0;
	const validityReasons = [];
	if (!Number.isFinite(loss)) validityReasons.push("The final loss is not finite.");
	if (overflow !== 0) validityReasons.push(`Tile overflow is ${overflow}, not zero.`);
	if (halfSaturations !== 0) {
		validityReasons.push(
			`Cumulative projection VJP FP16 saturation count is ${halfSaturations}, not zero.`,
		);
	}
	if (!roundStability.supported) validityReasons.push(roundStability.reason);
	else if (!roundStability.stable) {
		validityReasons.push(
			`Round throughput CV ${roundStability.coefficientOfVariation.toFixed(4)} exceeds `
			+ `${options.maxRoundCv}.`,
		);
	}
	return {
		id: variant.id,
		label: variant.label,
		adapter: trainer.adapterName,
		backwardMode: trainer.backwardMode,
		requestedProjectionVjpPrecision: options.projectionVjpPrecision,
		effectiveProjectionVjpPrecision: trainer.projectionVjpPrecision,
		raster: [dataset.width, dataset.height],
		requestedSplats: options.splats,
		capacity: trainer.splatCount,
		warmupSteps: options.warmup,
		measuredSteps: options.steps,
		measurementRounds: rounds.length,
		roundElapsedMs: rounds.map((round) => round.elapsedMs),
		rounds,
		roundStability,
		profileSteps: options.profiles,
		elapsedMs,
		stepsPerSecond: options.steps * 1000 / Math.max(elapsedMs, 0.001),
		loss,
		lossBreakdown: trainer.lastLossBreakdown ?? null,
		profile: summarizeProfiles(profiles),
		memoryPlan: trainer.memoryPlan,
		validity: {
			finiteLoss: Number.isFinite(loss),
			zeroTileOverflow: overflow === 0,
			zeroProjectionVjpHalfSaturations: halfSaturations === 0,
			stableRounds: roundStability.supported && roundStability.stable,
			valid: validityReasons.length === 0,
			reasons: validityReasons,
		},
	};
}

function renderResults(results, controlId, candidateId) {
	resultBody.replaceChildren();
	for (const result of results) {
		const row = document.createElement("tr");
		const cells = [
			result.label,
			number(result.stepsPerSecond),
			result.roundStability.supported
				? number(result.roundStability.coefficientOfVariation, 4) : "-",
			`${number(result.elapsedMs)} ms`,
			number(result.profile.gpuSpanMedianMs, 3),
			number(result.profile.gpuSpanP95Ms, 3),
			number(result.loss, 6),
			`${result.lossBreakdown?.tileOverflowTotal ?? 0}`,
			`${(result.memoryPlan.allocatedBytes / 1048576).toFixed(1)} MiB`,
		];
		for (const value of cells) {
			const cell = document.createElement("td");
			cell.textContent = value;
			row.append(cell);
		}
		resultBody.append(row);
	}
	phaseBody.replaceChildren();
	const controlResult = results.find((result) => result.id === controlId);
	const candidateResult = results.find((result) => result.id === candidateId);
	const phaseNames = Object.keys(results[0]?.profile.phaseMedianMs ?? {});
	for (const phase of phaseNames) {
		const row = document.createElement("tr");
		const control = controlResult?.profile.phaseMedianMs[phase];
		const candidate = candidateResult?.profile.phaseMedianMs[phase];
		for (const value of [
			phase,
			`${number(control, 3)} ms`,
			`${number(candidate, 3)} ms`,
			Number.isFinite(control) && Number.isFinite(candidate) && candidate > 0
				? `${number(control / candidate, 2)}x` : "-",
		]) {
			const cell = document.createElement("td");
			cell.textContent = value;
			row.append(cell);
		}
		phaseBody.append(row);
	}
}

async function runBenchmark(options) {
	setState("Loading calibrated benchmark targets");
	// Full-frame tiled training does not read sampled-ray banks, and this lab
	// keeps motion weighting disabled. Skip the million-candidate sample sort
	// so harness startup does not dominate or time out an isolated kernel run.
	const preset = await loadPresetDataset({
		computeSamples: false,
		frameBankFormat: options.frameBank === "rgba8"
			? FRAME_BANK_FORMAT_RGBA8 : FRAME_BANK_FORMAT_RGBA32_FLOAT,
	});
	const dataset = resizeDatasetForBenchmark(
		preset,
		options.scale,
		{ computeSamples: false },
	);
	const results = [];
	const experimentVariants = EXPERIMENTS[options.experiment];
	const [controlVariant, candidateVariant] = experimentVariants;
	let selectedVariants;
	if (options.variant === "both") selectedVariants = [...experimentVariants];
	else if (options.variant === "control") selectedVariants = [controlVariant];
	else if (options.variant === "candidate") selectedVariants = [candidateVariant];
	else selectedVariants = experimentVariants.filter((variant) => variant.id === options.variant);
	if (options.order === "candidate-first" && selectedVariants.length === 2) {
		selectedVariants.reverse();
	}
	const contexts = [];
	try {
		for (const variant of selectedVariants) {
			setState(`Initializing ${variant.label}`);
			contexts.push(await initializeVariant(variant, dataset, options));
		}
		const roundCount = Math.min(4, options.steps);
		const baseRoundSteps = Math.floor(options.steps / roundCount);
		const extraRoundSteps = options.steps % roundCount;
		for (let round = 0; round < roundCount; round += 1) {
			const roundSteps = baseRoundSteps + (round < extraRoundSteps ? 1 : 0);
			const roundOrder = round % 2 === 0 ? contexts : [...contexts].reverse();
			for (const [executionPosition, context] of roundOrder.entries()) {
				setState(
					`Measuring ${context.variant.label}, round ${round + 1}/${roundCount}`,
				);
				const elapsedMs = await submitAndDrain(context.trainer, roundSteps);
				context.elapsedMs += elapsedMs;
				context.rounds.push({
					round,
					steps: roundSteps,
					elapsedMs,
					executionPosition,
				});
			}
		}
		for (let sample = 0; sample < options.profiles; sample += 1) {
			const profileOrder = sample % 2 === 0 ? contexts : [...contexts].reverse();
			for (const context of profileOrder) {
				setState(`Profiling ${context.variant.label}`);
				context.profiles.push(await context.trainer.profileGpuStep(TRAIN_OPTIONS));
			}
		}
		for (const context of contexts) {
			results.push(await summarizeVariant(context, dataset, options));
		}
	} finally {
		for (const context of contexts) context.trainer.device?.destroy();
	}
	const control = results.find((result) => result.id === controlVariant.id);
	const candidate = results.find((result) => result.id === candidateVariant.id);
	const comparison = control && candidate ? {
		controlId: control.id,
		candidateId: candidate.id,
		candidateThroughputSpeedup: candidate.stepsPerSecond / control.stepsPerSecond,
		candidateGpuTimeSpeedup:
			control.profile.gpuSpanMedianMs / candidate.profile.gpuSpanMedianMs,
		lossDelta: candidate.loss - control.loss,
		allocatedByteDelta: candidate.memoryPlan.allocatedBytes - control.memoryPlan.allocatedBytes,
		validForPromotion: control.validity.valid && candidate.validity.valid,
		...(options.experiment === "backward" ? {
			stagedThroughputSpeedup: candidate.stepsPerSecond / control.stepsPerSecond,
			stagedGpuTimeSpeedup:
				control.profile.gpuSpanMedianMs / candidate.profile.gpuSpanMedianMs,
		} : {}),
	} : null;
	const invalidResults = results.filter((result) => !result.validity.valid);
	const report = {
		schema: "dynaworld-browser-tiled-kernel-benchmark/v3",
		recordedAt: new Date().toISOString(),
		options,
		dataset: {
			name: dataset.name,
			width: dataset.width,
			height: dataset.height,
			viewCount: dataset.viewCount,
			trainViewCount: dataset.trainViewCount,
			frameCount: dataset.frameCount,
		},
		results,
		experiment: {
			id: options.experiment,
			controlId: controlVariant.id,
			candidateId: candidateVariant.id,
			order: options.order,
		},
		comparison,
		validity: {
			correctnessAndStabilityPassed: invalidResults.length === 0,
			promotable: invalidResults.length === 0,
			reasons: invalidResults.flatMap((result) => (
				result.validity.reasons.map((reason) => `${result.id}: ${reason}`)
			)),
		},
	};
	renderResults(results, controlVariant.id, candidateVariant.id);
	jsonOutput.value = JSON.stringify(report, null, 2);
	globalThis.__tiledKernelBenchmarkResults = report;
	const summary = report.comparison
		? `candidate ${number(report.comparison.candidateThroughputSpeedup, 2)}x wall throughput, `
			+ `${number(report.comparison.candidateGpuTimeSpeedup, 2)}x timestamped GPU time`
		: `${results[0].label} ${number(results[0].stepsPerSecond)} steps/s`;
	const validitySuffix = report.validity.promotable
		? "measurement stable"
		: `diagnostic only: ${report.validity.reasons.join(" ")}`;
	setState(`Complete: ${summary}; ${validitySuffix}.`, "complete");
	return report;
}

function applyQueryOptions() {
	const query = new URLSearchParams(location.search);
	const mappings = {
		experiment: inputs.experiment,
		variant: inputs.variant,
		order: inputs.order,
		splats: inputs.splats,
		capacity: inputs.capacity,
		scale: inputs.scale,
		frameBank: inputs.frameBank,
		warmup: inputs.warmup,
		steps: inputs.steps,
		profiles: inputs.profiles,
		checkpoint: inputs.checkpointPrecision,
		stride: inputs.checkpointStride,
		checkpointOrder: inputs.checkpointOrder,
		projectionLayout: inputs.projectionLayout,
		projectionVjpPrecision: inputs.projectionVjpPrecision,
		ssimLayout: inputs.ssimLayout,
		pairPacket: inputs.pairPacket,
		granularity: inputs.backwardGranularity,
		tile: inputs.tileSize,
		tileCapacity: inputs.tileCapacity,
		maxRoundCv: inputs.maxRoundCv,
		pixelFilterMode: inputs.pixelFilterMode,
		opacityModel: inputs.opacityModel,
		geometryColorWeight: inputs.geometryColorWeight,
		crossViewDepth: inputs.crossViewDepth,
		geometryConsistencyEvery: inputs.geometryConsistencyEvery,
		geometryDepthWeight: inputs.geometryDepthWeight,
	};
	for (const [key, input] of Object.entries(mappings)) {
		if (query.has(key)) input.value = query.get(key);
	}
	return query.get("autorun") === "1";
}

form.addEventListener("submit", async (event) => {
	event.preventDefault();
	if (running) return;
	running = true; runButton.disabled = true;
	try {
		await runBenchmark(readOptions());
	} catch (error) {
		console.error(error);
		setState(error instanceof Error ? error.message : String(error), "failed");
	} finally {
		running = false; runButton.disabled = false;
	}
});

if (applyQueryOptions()) form.requestSubmit();
