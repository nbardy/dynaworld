import { loadPresetDataset } from "./dataset.js?v=20260731-compactfp16-5";
import { resizeDatasetForBenchmark } from "./benchmarkDataset.js?v=20260731-compactfp16-5";
import {
	loadTrainerBackend,
	resolveTrainerBackend,
} from "./trainerBackendRegistry.js?v=20260731-compactfp16-5";

const BACKEND_IDS = Object.freeze(["tiled3d-fast", "tiled3d", "sampled3d"]);
const TRAIN_OPTIONS = Object.freeze({
	learningRate: 1.25,
	modelMode: 0,
	temporalSigma: 0.30,
	motionSampleRate: 0.90,
	staticSampleRate: 0.08,
	motionCoverageTarget: 0.52,
	camerasPerStep: 4,
});

const form = document.querySelector("#benchmarkForm");
const runButton = document.querySelector("#runBenchmark");
const statusOutput = document.querySelector("#benchmarkStatus");
const jsonOutput = document.querySelector("#benchmarkJson");
const resultsBody = document.querySelector("#resultsBody");
const inputs = {
	splatCount: document.querySelector("#splatCount"),
	tiledCapacity: document.querySelector("#tiledCapacity"),
	warmupSteps: document.querySelector("#warmupSteps"),
	measuredSteps: document.querySelector("#measuredSteps"),
	sampledPixels: document.querySelector("#sampledPixels"),
	rasterScale: document.querySelector("#rasterScale"),
	ssimWindow: document.querySelector("#ssimWindow"),
	checkpointPrecision: document.querySelector("#checkpointPrecision"),
};
const datasetFields = {
	name: document.querySelector("#datasetName"),
	raster: document.querySelector("#datasetRaster"),
	views: document.querySelector("#datasetViews"),
	frames: document.querySelector("#datasetFrames"),
};

let running = false;

function positiveInteger(input, label) {
	const value = Number(input.value);
	if (!Number.isSafeInteger(value) || value < Number(input.min) || value > Number(input.max)) {
		throw new RangeError(`${label} must be an integer from ${input.min} to ${input.max}.`);
	}
	return value;
}

function readOptions() {
	const options = {
		splatCount: positiveInteger(inputs.splatCount, "Requested splats"),
		tiledCapacity: positiveInteger(inputs.tiledCapacity, "Global model capacity"),
		warmupSteps: positiveInteger(inputs.warmupSteps, "Warmup steps"),
		measuredSteps: positiveInteger(inputs.measuredSteps, "Measured steps"),
		sampledPixels: positiveInteger(inputs.sampledPixels, "Sampled rays per step"),
		rasterScale: positiveInteger(inputs.rasterScale, "Raster scale"),
		ssimWindow: positiveInteger(inputs.ssimWindow, "SSIM window"),
		checkpointPrecision: inputs.checkpointPrecision.value,
	};
	if (options.tiledCapacity < options.splatCount) {
		throw new RangeError("Global model capacity must be at least the requested splat count.");
	}
	if (options.ssimWindow % 2 !== 1) {
		throw new RangeError("SSIM window must be odd.");
	}
	if (!["f32", "packed-f16"].includes(options.checkpointPrecision)) {
		throw new RangeError("Checkpoint precision must be FP32 or packed FP16.");
	}
	return options;
}

function formatNumber(value, maximumFractionDigits = 1) {
	return new Intl.NumberFormat(undefined, { maximumFractionDigits }).format(value);
}

function setStatus(message, state = "ready") {
	statusOutput.textContent = message;
	statusOutput.dataset.state = state;
	document.documentElement.dataset.benchmarkState = state;
}

function rowFor(backendId) {
	const row = resultsBody.querySelector(`tr[data-backend="${backendId}"]`);
	if (!row) throw new Error(`Missing result row for ${backendId}.`);
	return row;
}

function resetRows(options) {
	for (const backendId of BACKEND_IDS) {
		const descriptor = resolveTrainerBackend(backendId);
		const cells = rowFor(backendId).cells;
		cells[0].textContent = descriptor.label;
		cells[1].textContent = descriptor.objective;
		cells[2].textContent = "Pending";
		cells[3].textContent = formatNumber(options.splatCount, 0);
		cells[4].textContent = formatNumber(options.warmupSteps, 0);
		cells[5].textContent = formatNumber(options.measuredSteps, 0);
		for (let index = 6; index < cells.length; index += 1) cells[index].textContent = "-";
		cells[cells.length - 1].textContent = "Pending";
		cells[cells.length - 1].className = "pending";
	}
}

function showDataset(dataset) {
	datasetFields.name.textContent = dataset.name;
	datasetFields.raster.textContent = `${dataset.width} x ${dataset.height}`;
	datasetFields.views.textContent = `${dataset.trainViewCount} train / `
		+ `${dataset.viewCount - dataset.trainViewCount} held out`;
	datasetFields.frames.textContent = String(dataset.frameCount);
}

function assertCoffeeMartini(dataset) {
	const scene = dataset.datasetContract?.sample_id ?? "";
	if (!dataset.name?.includes("coffee_martini") || !scene.includes("coffee_martini")) {
		throw new Error(`Expected the calibrated Coffee Martini bundle, received "${dataset.name}".`);
	}
}

async function submitAndDrain(trainer, stepCount, trainOptions) {
	const initialStep = trainer.stepCount;
	const startedAt = performance.now();
	for (let step = 0; step < stepCount; step += 1) trainer.trainStep(trainOptions);
	await trainer.device.queue.onSubmittedWorkDone();
	const elapsedMs = performance.now() - startedAt;
	if (trainer.stepCount - initialStep !== stepCount) {
		throw new Error(`Submitted ${stepCount} steps but the trainer advanced `
			+ `${trainer.stepCount - initialStep}.`);
	}
	return elapsedMs;
}

async function checkFirstSubmission(trainer) {
	if (!trainer.firstStepValidation) return;
	const validationError = await trainer.firstStepValidation;
	trainer.firstStepValidation = null;
	if (validationError) {
		throw new Error(`First tiled training submission failed: ${validationError.message}`);
	}
}

function setRowRunning(backendId, message) {
	const cells = rowFor(backendId).cells;
	cells[cells.length - 1].textContent = message;
	cells[cells.length - 1].className = "running";
}

function setRowResult(result) {
	const cells = rowFor(result.backendId).cells;
	cells[2].textContent = result.workLabel;
	cells[3].textContent = result.capacity === result.requestedSplats
		? formatNumber(result.requestedSplats, 0)
		: `${formatNumber(result.requestedSplats, 0)} requested / ${formatNumber(result.capacity, 0)} capacity`;
	cells[4].textContent = formatNumber(result.warmupSteps, 0);
	cells[5].textContent = formatNumber(result.measuredSteps, 0);
	cells[6].textContent = `${formatNumber(result.elapsedMs, 2)} ms`;
	cells[7].textContent = formatNumber(result.stepsPerSecond, 2);
	cells[8].textContent = formatNumber(result.supervisedPixelsPerSecond, 0);
	cells[9].textContent = formatNumber(result.loss, 6);
	const currentOverflow = result.lossBreakdown?.tileOverflow;
	const totalOverflow = result.lossBreakdown?.tileOverflowTotal ?? currentOverflow;
	cells[10].textContent = currentOverflow == null
		? "-"
		: `${formatNumber(currentOverflow, 0)} now / ${formatNumber(totalOverflow, 0)} total`;
	cells[10].className = totalOverflow > 0 ? "failed" : "complete";
}

function setRowError(backendId, error) {
	const cells = rowFor(backendId).cells;
	cells[cells.length - 1].textContent = error instanceof Error ? error.message : String(error);
	cells[cells.length - 1].className = "failed";
}

async function benchmarkBackend(backendId, dataset, options) {
	const loaded = await loadTrainerBackend(backendId);
	const trainer = new loaded.Trainer(null);
	const trainOptions = {
		...TRAIN_OPTIONS,
		samplesPerStep: options.sampledPixels,
		ssimRadius: (options.ssimWindow - 1) / 2,
	};
	try {
		setRowRunning(backendId, "Initializing");
		const trainerOptions = { splatCount: options.splatCount };
		if (!loaded.descriptor.sampledControls) {
			trainerOptions.growthCapacity = options.tiledCapacity;
			trainerOptions.checkpointPrecision = options.checkpointPrecision;
		}
		await trainer.init(dataset, trainerOptions);
		await trainer.device.queue.onSubmittedWorkDone();

		setRowRunning(backendId, "Warming up");
		await submitAndDrain(trainer, options.warmupSteps, trainOptions);
		await checkFirstSubmission(trainer);

		setRowRunning(backendId, "Measuring");
		const elapsedMs = await submitAndDrain(trainer, options.measuredSteps, trainOptions);
		const loss = await trainer.readLoss(trainOptions);
		const supervisedPixelsPerStep = loaded.descriptor.sampledControls
			? options.sampledPixels
			: dataset.width * dataset.height;
		const stepsPerSecond = options.measuredSteps * 1000 / Math.max(elapsedMs, 0.001);
		return {
			backendId,
			label: loaded.descriptor.label,
			objective: loaded.descriptor.objective,
			trainingUnit: loaded.descriptor.trainingUnit,
			workLabel: loaded.descriptor.sampledControls
				? `${formatNumber(supervisedPixelsPerStep, 0)} sampled rays`
				: `${dataset.width} x ${dataset.height} full image (${formatNumber(supervisedPixelsPerStep, 0)} pixels)`,
			requestedSplats: options.splatCount,
			capacity: trainer.splatCount,
			warmupSteps: options.warmupSteps,
			measuredSteps: options.measuredSteps,
			elapsedMs,
			stepsPerSecond,
			supervisedPixelsPerStep,
			supervisedPixelsPerSecond: stepsPerSecond * supervisedPixelsPerStep,
			loss,
			lossBreakdown: trainer.lastLossBreakdown ?? null,
			adapter: trainer.adapterName,
			memoryPlan: trainer.memoryPlan ?? null,
		};
	} finally {
		trainer.device?.destroy();
	}
}

async function runBenchmark(options) {
	if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser.");
	resetRows(options);
	setStatus("Loading the calibrated Coffee Martini bundle...", "loading");
	const sourceDataset = await loadPresetDataset();
	assertCoffeeMartini(sourceDataset);
	const dataset = resizeDatasetForBenchmark(sourceDataset, options.rasterScale);
	showDataset(dataset);

	const results = [];
	for (const backendId of BACKEND_IDS) {
		const descriptor = resolveTrainerBackend(backendId);
		setStatus(`Running ${descriptor.label}: ${results.length + 1} of ${BACKEND_IDS.length}...`, "running");
		try {
			const result = await benchmarkBackend(backendId, dataset, options);
			results.push(result);
			setRowResult(result);
		} catch (error) {
			setRowError(backendId, error);
			results.push({
				backendId,
				label: descriptor.label,
				objective: descriptor.objective,
				error: error instanceof Error ? error.message : String(error),
			});
			const logFailure = error instanceof RangeError ? console.warn : console.error;
			logFailure(`${descriptor.label} benchmark failed.`, error);
		}
	}

	globalThis.__trainerBackendBenchmarkResults = {
		dataset: {
			name: dataset.name,
			width: dataset.width,
			height: dataset.height,
			frameCount: dataset.frameCount,
			trainViewCount: dataset.trainViewCount,
			heldoutViewIndex: dataset.heldoutViewIndex,
			sourceWidth: sourceDataset.width,
			sourceHeight: sourceDataset.height,
			rasterScale: options.rasterScale,
		},
		options,
		results,
		objectivesMatched: false,
	};
	jsonOutput.textContent = JSON.stringify(globalThis.__trainerBackendBenchmarkResults, null, 2);
	const completed = results.filter((result) => !result.error).length;
	if (completed === BACKEND_IDS.length) {
		setStatus("Benchmark complete. Every timed interval drained its GPU queue.", "complete");
	} else {
		setStatus(`Benchmark finished with ${BACKEND_IDS.length - completed} backend error(s).`, "failed");
	}
}

form.addEventListener("submit", async (event) => {
	event.preventDefault();
	if (running) return;
	running = true;
	runButton.disabled = true;
	for (const input of Object.values(inputs)) input.disabled = true;
	try {
		await runBenchmark(readOptions());
	} catch (error) {
		setStatus(error instanceof Error ? error.message : String(error), "failed");
		console.error("Backend benchmark failed.", error);
	} finally {
		running = false;
		runButton.disabled = false;
		for (const input of Object.values(inputs)) input.disabled = false;
	}
});
