import { loadCalibratedMulticamDataset } from "./dataset.js";
import { renderSnapshotFrame } from "./snapshotMetrics.js";
import { SPLAT_FLOATS } from "./trainerWebGpu3d.js";
import {
	DynamicSplatWebGpu3dTiledTrainer,
	windowedL1DssimCpu,
} from "./trainerWebGpu3dTiled.js";

export const TILED_PARITY_DIAGNOSTIC_COMPONENTS = Object.freeze([11, 15]);

export const TILED_PARITY_GRADIENT_FAMILIES = Object.freeze([
	Object.freeze({ name: "center", components: Object.freeze([0, 1, 2]) }),
	Object.freeze({ name: "staticMix", components: Object.freeze([3]) }),
	Object.freeze({ name: "velocity", components: Object.freeze([4, 5, 6]) }),
	Object.freeze({ name: "timeCenter", components: Object.freeze([7]) }),
	Object.freeze({ name: "harmonic", components: Object.freeze([8, 9, 10]) }),
	Object.freeze({ name: "logScale", components: Object.freeze([12, 13, 14]) }),
	Object.freeze({ name: "rotation", components: Object.freeze([16, 17, 18, 19]) }),
	Object.freeze({ name: "color", components: Object.freeze([20, 21, 22]) }),
	Object.freeze({ name: "opacity", components: Object.freeze([23]) }),
]);

const THRESHOLDS = Object.freeze({
	forwardRgbMaxAbs: 2e-4,
	forwardRgbRmse: 2e-5,
	forwardAlphaMaxAbs: 2e-4,
	objectiveAbs: 5e-5,
	gradientAbs: 5e-5,
	gradientRelative: 0.12,
	finiteDifferenceAbs: 5e-5,
	finiteDifferenceRelative: 0.08,
	gradientActivity: 1e-7,
	minimumActiveGradientFamilies: 6,
});
const PARITY_SSIM_RADIUS = 5;

function differenceSummary(left, right, stride, components) {
	if (left.length !== right.length) throw new RangeError("Parity arrays must have equal lengths.");
	let absoluteSum = 0;
	let squareSum = 0;
	let maximum = 0;
	let worstIndex = -1;
	let count = 0;
	for (let base = 0; base < left.length; base += stride) {
		for (const component of components) {
			const index = base + component;
			const error = Math.abs(left[index] - right[index]);
			absoluteSum += error;
			squareSum += error * error;
			count += 1;
			if (error > maximum) {
				maximum = error;
				worstIndex = index;
			}
		}
	}
	return {
		maxAbs: maximum,
		meanAbs: absoluteSum / count,
		rmse: Math.sqrt(squareSum / count),
		worstIndex,
	};
}

export function summarizeForwardParity(gpuRgba, cpuRgb, cpuCoverage) {
	if (gpuRgba.length % 4 !== 0
		|| cpuRgb.length !== gpuRgba.length / 4 * 3
		|| cpuCoverage.length !== gpuRgba.length / 4) {
		throw new RangeError("Forward parity inputs have incompatible packed image lengths.");
	}
	const cpuRgba = new Float32Array(gpuRgba.length);
	for (let pixel = 0; pixel < cpuCoverage.length; pixel += 1) {
		cpuRgba[pixel * 4] = cpuRgb[pixel * 3];
		cpuRgba[pixel * 4 + 1] = cpuRgb[pixel * 3 + 1];
		cpuRgba[pixel * 4 + 2] = cpuRgb[pixel * 3 + 2];
		cpuRgba[pixel * 4 + 3] = cpuCoverage[pixel];
	}
	return {
		rgb: differenceSummary(gpuRgba, cpuRgba, 4, [0, 1, 2]),
		alpha: differenceSummary(gpuRgba, cpuRgba, 4, [3]),
	};
}

export function selectGradientChannels(gradients, splatCount) {
	if (!Number.isSafeInteger(splatCount) || splatCount < 1
		|| gradients.length !== splatCount * SPLAT_FLOATS) {
		throw new RangeError("gradients must contain complete 24-float splats.");
	}
	return TILED_PARITY_GRADIENT_FAMILIES.map((family) => {
		let selected = null;
		for (let splatIndex = 0; splatIndex < splatCount; splatIndex += 1) {
			for (const component of family.components) {
				if (TILED_PARITY_DIAGNOSTIC_COMPONENTS.includes(component)) {
					throw new Error(`Gradient family ${family.name} includes diagnostic slot ${component}.`);
				}
				const parameterIndex = splatIndex * SPLAT_FLOATS + component;
				const gpuGradient = gradients[parameterIndex];
				if (!selected || Math.abs(gpuGradient) > Math.abs(selected.gpuGradient)) {
					selected = {
						family: family.name,
						splatIndex,
						component,
						parameterIndex,
						gpuGradient,
					};
				}
			}
		}
		return selected;
	});
}

function targetRgb(dataset, viewIndex, frameIndex) {
	const pixels = dataset.width * dataset.height;
	const source = (viewIndex * dataset.frameCount + frameIndex) * pixels * 4;
	const result = new Float32Array(pixels * 3);
	for (let pixel = 0; pixel < pixels; pixel += 1) {
		result[pixel * 3] = dataset.frames[source + pixel * 4];
		result[pixel * 3 + 1] = dataset.frames[source + pixel * 4 + 1];
		result[pixel * 3 + 2] = dataset.frames[source + pixel * 4 + 2];
	}
	return result;
}

function finiteDifferenceStep(component, geometryScale) {
	if (component <= 2 || (component >= 4 && component <= 6)
		|| (component >= 8 && component <= 10)) {
		return Math.max(2e-6, geometryScale * 0.0002);
	}
	if (component >= 20 && component <= 22) return 5e-4;
	return 1e-3;
}

function objective(dataset, params, target, viewIndex, frameIndex) {
	const rendered = renderSnapshotFrame(dataset, params, {
		viewIndex,
		frameIndex,
		splatCount: params.length / SPLAT_FLOATS,
		modelMode: 0,
		temporalSigma: 0.30,
	});
	return windowedL1DssimCpu(rendered.rgb, target, dataset.width, dataset.height, {
		radius: PARITY_SSIM_RADIUS,
		computeGradient: false,
	}).loss;
}

function centralDifference(dataset, params, target, viewIndex, frameIndex, parameterIndex, step) {
	const plus = params.slice();
	const minus = params.slice();
	plus[parameterIndex] = params[parameterIndex] + step;
	minus[parameterIndex] = params[parameterIndex] - step;
	const denominator = plus[parameterIndex] - minus[parameterIndex];
	return (objective(dataset, plus, target, viewIndex, frameIndex)
		- objective(dataset, minus, target, viewIndex, frameIndex)) / denominator;
}

function closeEnough(left, right, absoluteTolerance, relativeTolerance = 0) {
	return Math.abs(left - right) <= absoluteTolerance
		+ relativeTolerance * Math.max(Math.abs(left), Math.abs(right));
}

function checkGradient(dataset, initialParams, target, selection, viewIndex, frameIndex) {
	const epsilon = finiteDifferenceStep(selection.component, dataset.geometryScale);
	const finiteDifference = centralDifference(
		dataset, initialParams, target, viewIndex, frameIndex, selection.parameterIndex, epsilon,
	);
	const halfStepFiniteDifference = centralDifference(
		dataset, initialParams, target, viewIndex, frameIndex, selection.parameterIndex, epsilon / 2,
	);
	const finiteDifferenceStable = closeEnough(
		finiteDifference,
		halfStepFiniteDifference,
		THRESHOLDS.finiteDifferenceAbs,
		THRESHOLDS.finiteDifferenceRelative,
	);
	const active = Math.max(
		Math.abs(selection.gpuGradient),
		Math.abs(halfStepFiniteDifference),
	) >= THRESHOLDS.gradientActivity;
	const matchesGpu = closeEnough(
		selection.gpuGradient,
		halfStepFiniteDifference,
		THRESHOLDS.gradientAbs,
		THRESHOLDS.gradientRelative,
	);
	return {
		...selection,
		epsilon,
		finiteDifference,
		halfStepFiniteDifference,
		finiteDifferenceStable,
		active,
		absoluteError: Math.abs(selection.gpuGradient - halfStepFiniteDifference),
		relativeError: Math.abs(selection.gpuGradient - halfStepFiniteDifference)
			/ Math.max(Math.abs(selection.gpuGradient), Math.abs(halfStepFiniteDifference), 1e-12),
		pass: finiteDifferenceStable && matchesGpu,
	};
}

function objectiveParity(gpuMetrics, cpuMetrics) {
	const checks = [
		{ name: "loss", gpu: gpuMetrics[0], cpu: cpuMetrics.loss },
		{ name: "l1", gpu: gpuMetrics[1], cpu: cpuMetrics.l1 },
		{ name: "dssim", gpu: gpuMetrics[2], cpu: cpuMetrics.dssim },
	].map((entry) => ({
		...entry,
		absoluteError: Math.abs(entry.gpu - entry.cpu),
		pass: closeEnough(entry.gpu, entry.cpu, THRESHOLDS.objectiveAbs),
	}));
	return {
		tileOverflow: gpuMetrics[3],
		checks,
		pass: gpuMetrics[3] === 0 && checks.every((check) => check.pass),
	};
}

function failuresFor(report) {
	const failures = [];
	if (!report.forward.pass) failures.push("forward raster parity");
	if (!report.objective.pass) failures.push("training objective parity");
	if (!report.gradients.pass) failures.push("analytic gradient parity");
	return failures;
}

export class TiledParityError extends Error {
	constructor(report) {
		super(`Tiled WebGPU parity failed: ${report.failures.join(", ")}`);
		this.name = "TiledParityError";
		this.report = report;
	}
}

export async function runTiledParityHarness({
	throwOnFailure = true,
	onProgress = () => {},
} = {}) {
	if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser.");
	onProgress("Loading calibrated Coffee Martini preset");
	const sourceDataset = await loadCalibratedMulticamDataset();
	const trainer = new DynamicSplatWebGpu3dTiledTrainer(null);
	let report;
	try {
		onProgress("Initializing deterministic 8-splat FP32 trainer");
		await trainer.init(sourceDataset, {
			splatCount: 8,
			growthCapacity: 8,
			tileCapacity: 16,
			checkpointPrecision: "f32",
		});
		await trainer.device.queue.onSubmittedWorkDone();
		const initialParams = trainer.initialParams.slice();

		onProgress("Running one zero-learning-rate full-frame step");
		trainer.trainStep({
			learningRate: 0,
			modelMode: 0,
			temporalSigma: 0.30,
			ssimRadius: PARITY_SSIM_RADIUS,
		});
		const submissionError = await trainer.firstStepValidation;
		trainer.firstStepValidation = null;
		if (submissionError) throw new Error(`Tiled submission failed: ${submissionError.message}`);
		const debug = await trainer.readTiledStepDebugState();
		const dataset = trainer.dataset;
		const target = targetRgb(dataset, debug.viewIndex, debug.frameIndex);

		onProgress("Comparing GPU raster against CPU tile renderer");
		const cpuFrame = renderSnapshotFrame(dataset, initialParams, {
			viewIndex: debug.viewIndex,
			frameIndex: debug.frameIndex,
			splatCount: 8,
			modelMode: 0,
			temporalSigma: 0.30,
		});
		const forwardSummary = summarizeForwardParity(
			debug.renderedRgba,
			cpuFrame.rgb,
			cpuFrame.coverage,
		);
		const forward = {
			...forwardSummary,
			pass: forwardSummary.rgb.maxAbs <= THRESHOLDS.forwardRgbMaxAbs
				&& forwardSummary.rgb.rmse <= THRESHOLDS.forwardRgbRmse
				&& forwardSummary.alpha.maxAbs <= THRESHOLDS.forwardAlphaMaxAbs,
		};
		const cpuObjective = windowedL1DssimCpu(
			cpuFrame.rgb,
			target,
			dataset.width,
			dataset.height,
			{ radius: PARITY_SSIM_RADIUS, computeGradient: false },
		);

		onProgress("Checking selected analytic gradients with central differences");
		const gradientChecks = selectGradientChannels(debug.gradients, 8).map((selection) =>
			checkGradient(
				dataset,
				initialParams,
				target,
				selection,
				debug.viewIndex,
				debug.frameIndex,
			));
		const activeCount = gradientChecks.filter((check) => check.active).length;
		const gradients = {
			skippedDiagnosticComponents: [...TILED_PARITY_DIAGNOSTIC_COMPONENTS],
			selectedCount: gradientChecks.length,
			activeCount,
			checks: gradientChecks,
			pass: activeCount >= THRESHOLDS.minimumActiveGradientFamilies
				&& gradientChecks.every((check) => check.pass),
		};
		report = {
			contract: "dynaworld_tiled_webgpu_parity/v1",
			pass: false,
			dataset: {
				name: dataset.name,
				width: dataset.width,
				height: dataset.height,
				viewIndex: debug.viewIndex,
				camera: dataset.cameras[debug.viewIndex]?.name ?? null,
				frameIndex: debug.frameIndex,
			},
			trainer: {
				adapter: trainer.adapterName,
				splatCount: 8,
				growthCapacity: 8,
				tileCapacity: 16,
				checkpointPrecision: "f32",
				step: debug.step,
				ssimRadius: PARITY_SSIM_RADIUS,
				learningRate: 0,
			},
			thresholds: THRESHOLDS,
			forward,
			objective: objectiveParity(debug.metrics, cpuObjective),
			gradients,
			failures: [],
		};
		report.failures = failuresFor(report);
		report.pass = report.failures.length === 0;
	} finally {
		trainer.dispose();
		trainer.device?.destroy();
	}
	if (!report) throw new Error("Tiled parity run ended without a report.");
	if (!report.pass && throwOnFailure) throw new TiledParityError(report);
	return report;
}

function renderPage(report, error = null) {
	const status = document.querySelector("#parityStatus");
	const output = document.querySelector("#parityJson");
	const raster = document.querySelector("#rasterSummary");
	const gradients = document.querySelector("#gradientSummary");
	const objective = document.querySelector("#objectiveSummary");
	const pass = report?.pass === true;
	document.documentElement.dataset.parityState = pass ? "pass" : "fail";
	status.textContent = pass ? "PASS" : "FAIL";
	raster.textContent = report
		? `${report.forward.rgb.maxAbs.toExponential(2)} max RGB error`
		: "Unavailable";
	gradients.textContent = report
		? `${report.gradients.activeCount}/${report.gradients.selectedCount} active families`
		: "Unavailable";
	objective.textContent = report
		? `${report.objective.checks[0].absoluteError.toExponential(2)} loss error`
		: "Unavailable";
	output.textContent = JSON.stringify(report ?? {
		contract: "dynaworld_tiled_webgpu_parity/v1",
		pass: false,
		error: error instanceof Error ? error.message : String(error),
	}, null, 2);
}

async function main() {
	const detail = document.querySelector("#parityDetail");
	try {
		const report = await runTiledParityHarness({
			onProgress(message) {
				detail.textContent = message;
			},
		});
		globalThis.__tiledParityResult = report;
		detail.textContent = "GPU forward, objective, and selected gradients agree with CPU references.";
		renderPage(report);
	} catch (error) {
		const report = error instanceof TiledParityError ? error.report : null;
		globalThis.__tiledParityResult = report ?? { pass: false, error: String(error) };
		detail.textContent = error instanceof Error ? error.message : String(error);
		renderPage(report, error);
		setTimeout(() => {
			throw error;
		});
	}
}

if (typeof document !== "undefined") main();
