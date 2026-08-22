import {
	resolveActiveSplatCount,
	resolveCamerasPerStep,
	resolveRenderViewIndices,
} from "./trainerWebGpu3d.js?v=20260803-fullfps-pixelgs-1";
import { loadTrainerBackend } from "./trainerBackendRegistry.js?v=20260821-stablegs-ablation-1";
import {
	assertProtocolMessage, protocolMessage, publishSharedStatus, StatusFlag, TrainerState,
	WorkerCommand, WorkerEvent, WORKER_PROTOCOL_VERSION,
} from "./workerProtocol.js";
import {
	combineDatasetSharingTelemetry,
	hydrateDatasetSharedViews,
	summarizeDatasetSharing,
} from "./datasetSharing.js";
import { assertResolutionContinuationCompatible } from "./resolutionSchedule.js";

const DEFAULT_TRAIN_OPTIONS = Object.freeze({
	learningRate: 1.25,
	learningRateDecay: true,
	samplesPerStep: 96,
	modelMode: 0,
	temporalSigma: 0.30,
	motionSampleRate: 0.90,
	staticSampleRate: 0.08,
	motionCoverageTarget: 0.52,
	motionWeighting: false,
	randomBackground: false,
	camerasPerStep: null,
});
let trainer = null;
let backendDescriptor = null;
let loadedBackend = null;
let trainerCanvas = null;
let trainerOptions = {};
let canonicalDataset = null;
let validationWorker = null;
let statusSlots = null;
let running = false;
let pumpToken = 0;
let trainOptions = { ...DEFAULT_TRAIN_OPTIONS };
let renderOptions = { enabled: true, time: 0.35, modelMode: 0, temporalSigma: 0.30, renderMode: 0,
	viewIndex: 0, viewIndices: null, previewCameras: null };
let burstSteps = 8;
let maxQueuedSteps = 32;
let metricEvery = 512;
let validationEvery = 0;
let renderFps = 15;
let metricsPending = false;
let validationPending = false;
let latestLoss = Number.NaN;
let latestPsnr = Number.NaN;
let latestSsim = Number.NaN;
let lastMetricStep = 0;
let lastValidationStep = 0;
let lastMetricRequestStep = 0;
let lastMetricRecycled = 0;
let lastValidationRequestStep = 0;
let lastRenderAt = 0;
let lastStatusMessageAt = 0;
let completedStep = 0;
let completedAt = performance.now();
let completionProbePending = false;
let stepsPerSecond = 0;
let flags = StatusFlag.VALIDATION_WORKER;
let stageTransitionPending = false;

function status(state = running ? TrainerState.RUNNING : TrainerState.PAUSED) {
	return { state, step: trainer?.stepCount ?? 0, stepsPerSecond, loss: latestLoss, psnr: latestPsnr,
		ssim: latestSsim, flags: flags | (metricsPending ? StatusFlag.METRICS_PENDING : 0)
			| (validationPending ? StatusFlag.VALIDATION_PENDING : 0),
		lastMetricStep, lastValidationStep,
		camerasPerStep: trainer?.lastCameraBatch?.length
			?? (backendDescriptor?.sampledControls ? trainOptions.camerasPerStep : 1) ?? 0,
		cameraRotationStart: trainer?.lastCameraBatchStart ?? 0,
		trainViewCount: trainer?.trainViewIndices?.length ?? 0 };
}

function publish(state, forceMessage = false) {
	const value = status(state);
	publishSharedStatus(statusSlots, value);
	const now = performance.now();
	if (!statusSlots && (forceMessage || now - lastStatusMessageAt >= 100)) {
		lastStatusMessageAt = now;
		self.postMessage(protocolMessage(WorkerEvent.STATUS, value));
	}
}

function reportError(error) {
	running = false;
	publish(TrainerState.FAILED, true);
	self.postMessage(protocolMessage(WorkerEvent.ERROR, {
		message: error?.message ?? String(error), stack: error?.stack,
	}));
}

function requestCompletionProbe(now, force = false) {
	if (completionProbePending || !trainer || (!force && now - completedAt < 250)) return;
	completionProbePending = true;
	const step = trainer.stepCount;
	trainer.device.queue.onSubmittedWorkDone().then(() => {
		const finishedAt = performance.now();
		stepsPerSecond = (step - completedStep) * 1000 / Math.max(1, finishedAt - completedAt);
		completedStep = step;
		completedAt = finishedAt;
	}).catch(reportError).finally(() => {
		completionProbePending = false;
		publish();
	});
}

function requestMetrics() {
	if (metricsPending || !trainer) return false;
	metricsPending = true;
	const step = trainer.stepCount;
	const totalRecycled = trainer.totalRecycled;
	lastMetricRequestStep = step;
	trainer.readLoss(trainOptions).then((loss) => {
		latestLoss = loss;
		lastMetricStep = step;
		const pairCycle = trainer.trainViewIndices.length * trainer.dataset.frameCount;
		const objectiveStep = Number.isFinite(trainer.lastLossBreakdown?.objectiveStep)
			? Math.round(trainer.lastLossBreakdown.objectiveStep)
			: step - 1;
		const fittedStep = Math.max(0, objectiveStep - (trainer.staticWarmupSteps ?? 0));
		const breakdown = {
			...(trainer.lastLossBreakdown ?? {}),
			pairCycle,
			cyclePhase: fittedStep % pairCycle,
			topologyOpsSinceMetric: totalRecycled - lastMetricRecycled,
		};
		lastMetricRecycled = totalRecycled;
		self.postMessage(protocolMessage(WorkerEvent.METRICS, {
			step, loss, breakdown, totalRecycled,
		}));
	}).catch(reportError).finally(() => {
		metricsPending = false;
		publish();
	});
	return true;
}

function requestValidation(options = {}) {
	if (validationPending || !trainer || !validationWorker) return false;
	validationPending = true;
	const step = trainer.stepCount;
	lastValidationRequestStep = step;
	// The queue copy and map complete asynchronously. Command submission continues in pump().
	trainer.readParams().then((params) => {
		validationWorker.postMessage({ version: WORKER_PROTOCOL_VERSION, type: "validate", step,
			options: { splatCount: resolveActiveSplatCount(trainer.splatCount, trainer.activeSplatCount),
				modelMode: trainOptions.modelMode,
				temporalSigma: trainOptions.temporalSigma,
				totalRecycled: trainer.totalRecycled,
				maxAspectRatio: backendDescriptor?.maxAspectRatio ?? 3,
				pixelFilterMode: trainer.pixelFilterMode ?? "legacy-floor",
				opacityModel: trainer.opacityModel ?? "coupled",
				materialOpacityBias: trainer.materialOpacityBias ?? Math.log(99),
				learningRateMultipliers: trainer.lastLearningRateMultipliers ?? null },
			params: params.buffer }, [params.buffer]);
	}).catch((error) => {
		validationPending = false;
		reportError(error);
	});
	return true;
}

function renderTrainer() {
	trainer.render(renderOptions.time, renderOptions.modelMode, renderOptions.temporalSigma,
		renderOptions.renderMode, renderOptions.viewIndex, renderOptions.viewIndices,
		renderOptions.previewCameras);
}

function render(now) {
	if (!renderOptions.enabled || !trainer?.context || renderFps <= 0
		|| now - lastRenderAt < 1000 / renderFps) return;
	renderTrainer();
	lastRenderAt = now;
}

function initializeValidationWorker(dataset, initialParams) {
	validationWorker = new Worker(new URL("./validationWorker.js?v=20260821-stablegs-ablation-1", import.meta.url),
		{ type: "module" });
	return new Promise((resolve, reject) => {
		let ready = false;
		validationWorker.onmessage = ({ data }) => {
			if (data?.type === "ready") {
				ready = true;
				resolve(data.datasetSharing);
			} else if (data?.type === "validation") {
				validationPending = false;
				lastValidationStep = data.step;
				latestPsnr = data.metrics.gridPsnr;
				latestSsim = data.metrics.gridSsim;
				self.postMessage(protocolMessage(WorkerEvent.VALIDATION, data));
				publish();
			} else if (data?.type === "error") {
				const error = new Error(data.message || "Validation worker failed.");
				if (!ready) {
					validationWorker.terminate();
					reject(error);
					return;
				}
				validationPending = false;
				self.postMessage(protocolMessage(WorkerEvent.ERROR, data));
				publish();
			}
		};
		validationWorker.onerror = (event) => {
			const error = new Error(event.message || "Validation worker failed.");
			if (!ready) {
				validationWorker.terminate();
				reject(error);
			} else {
				reportError(error);
			}
		};
		validationWorker.postMessage({
			version: WORKER_PROTOCOL_VERSION,
			type: "init",
			dataset,
			initialParams,
		});
	});
}

function schedulePump(token, delay = 0) {
	setTimeout(() => pump(token), delay);
}

function pump(token) {
	if (!running || token !== pumpToken || !trainer) return;
	try {
		const now = performance.now();
		const queuedSteps = trainer.stepCount - completedStep;
		if (queuedSteps >= maxQueuedSteps) {
			requestCompletionProbe(now, true);
			render(now); publish(TrainerState.RUNNING);
			schedulePump(token, 1);
			return;
		}
		const submitCount = Math.min(burstSteps, maxQueuedSteps - queuedSteps);
		for (let index = 0; index < submitCount; index += 1) trainer.trainStep(trainOptions);
		requestCompletionProbe(now, trainer.stepCount - completedStep >= maxQueuedSteps);
		render(now);
		if (metricEvery > 0 && trainer.stepCount - lastMetricRequestStep >= metricEvery) requestMetrics();
		if (validationEvery > 0 && trainer.stepCount - lastValidationRequestStep >= validationEvery) requestValidation();
		publish(TrainerState.RUNNING);
		schedulePump(token);
	} catch (error) {
		reportError(error);
	}
}

async function initialize(message) {
	statusSlots = message.statusBuffer ? new Int32Array(message.statusBuffer) : null;
	if (statusSlots) flags |= StatusFlag.SHARED_MEMORY;
	if (message.canvas) flags |= StatusFlag.OFFSCREEN_RENDER;
	trainOptions = { ...DEFAULT_TRAIN_OPTIONS, ...message.trainOptions };
	renderOptions = { ...renderOptions, ...message.renderOptions };
	loadedBackend = await loadTrainerBackend(message.trainerOptions?.backend);
	backendDescriptor = loadedBackend.descriptor;
	trainerCanvas = message.canvas ?? null;
	trainerOptions = { ...message.trainerOptions };
	burstSteps = Math.max(1, Math.min(64,
		message.schedule?.burstSteps ?? backendDescriptor.defaultSchedule.burstSteps));
	maxQueuedSteps = Math.max(1, Math.min(64,
		message.schedule?.maxQueuedSteps ?? backendDescriptor.defaultSchedule.maxQueuedSteps));
	metricEvery = Math.max(0,
		message.schedule?.metricEvery ?? backendDescriptor.defaultSchedule.metricEvery);
	validationEvery = Math.max(0, message.schedule?.validationEvery ?? validationEvery);
	renderFps = Math.max(0, Math.min(60, message.schedule?.renderFps ?? renderFps));
	trainer = new loadedBackend.Trainer(trainerCanvas);
	const incomingDataset = hydrateDatasetSharedViews(message.dataset);
	canonicalDataset = incomingDataset.dataset;
	await trainer.init(canonicalDataset, trainerOptions);
	if (backendDescriptor.sampledControls) {
		trainOptions.camerasPerStep = resolveCamerasPerStep(trainer.trainViewIndices.length,
			trainOptions.camerasPerStep);
	} else {
		trainOptions.camerasPerStep = 1;
	}
	renderOptions.viewIndices = resolveRenderViewIndices(trainer.dataset, renderOptions.viewIndices);
	const trainingDatasetSharing = summarizeDatasetSharing(trainer.dataset);
	const validationDatasetSharing = await initializeValidationWorker(trainer.dataset, trainer.initialParams);
	const datasetSharing = combineDatasetSharingTelemetry(
		message.datasetSharing ?? incomingDataset.telemetry,
		trainingDatasetSharing,
		validationDatasetSharing,
	);
	renderTrainer();
	lastRenderAt = performance.now();
	publish(TrainerState.READY, true);
	self.postMessage(protocolMessage(WorkerEvent.READY, {
		adapter: trainer.adapterName,
		backend: { ...backendDescriptor, initialSplats: trainer.initialSplatCount ?? trainer.splatCount,
			capacity: trainer.splatCount,
			memoryPlan: trainer.memoryPlan ?? null,
			trainingPixelsPerStep: backendDescriptor.sampledControls
				? trainOptions.samplesPerStep : trainer.dataset.width * trainer.dataset.height,
		},
		capabilities: { sharedStatus: Boolean(statusSlots), offscreenRender: Boolean(trainer.context),
			validationWorker: true, datasetSharing },
		cameraBatch: { camerasPerStep: trainOptions.camerasPerStep,
			trainViewCount: trainer.trainViewIndices.length, trainViewIndices: trainer.trainViewIndices,
			renderViewIndices: renderOptions.viewIndices, sameTimeGrouped: false },
	}));
}

async function switchDataset(message) {
	if (stageTransitionPending) throw new Error("A resolution-stage transition is already in progress.");
	stageTransitionPending = true;
	const wasRunning = running;
	const sourceTrainer = trainer;
	const sourceDataset = canonicalDataset;
	try {
		running = false;
		pumpToken += 1;
		publish(TrainerState.PAUSED, true);
		self.postMessage(protocolMessage(WorkerEvent.STAGE_STARTED, {
			step: sourceTrainer.stepCount,
			from: { width: sourceDataset.width, height: sourceDataset.height },
			to: { width: message.dataset.width, height: message.dataset.height },
			details: message.details ?? {},
		}));

		const incomingDataset = hydrateDatasetSharedViews(message.dataset);
		assertResolutionContinuationCompatible(sourceDataset, incomingDataset.dataset);
		validationWorker?.terminate();
		validationWorker = null;
		validationPending = false;

		// This is the only intentional training pause: drain already-submitted work,
		// then carry optimizer and topology state across resolution-dependent buffers.
		await sourceTrainer.device.queue.onSubmittedWorkDone();
		const continuation = await sourceTrainer.exportContinuationState();
		sourceTrainer.dispose();

		const nextTrainer = new loadedBackend.Trainer(trainerCanvas);
		await nextTrainer.init(incomingDataset.dataset, trainerOptions);
		await nextTrainer.restoreContinuationState(continuation);
		trainer = nextTrainer;
		canonicalDataset = incomingDataset.dataset;
		renderOptions.viewIndices = resolveRenderViewIndices(trainer.dataset, renderOptions.viewIndices);

		const trainingDatasetSharing = summarizeDatasetSharing(trainer.dataset);
		const validationDatasetSharing = await initializeValidationWorker(trainer.dataset, trainer.initialParams);
		const datasetSharing = combineDatasetSharingTelemetry(
			message.datasetSharing ?? incomingDataset.telemetry,
			trainingDatasetSharing,
			validationDatasetSharing,
		);
		completedStep = trainer.stepCount;
		completedAt = performance.now();
		stepsPerSecond = 0;
		lastMetricRequestStep = trainer.stepCount;
		lastValidationRequestStep = trainer.stepCount;
		renderTrainer();
		lastRenderAt = performance.now();
		self.postMessage(protocolMessage(WorkerEvent.STAGE_READY, {
			step: trainer.stepCount,
			width: trainer.dataset.width,
			height: trainer.dataset.height,
			backend: {
				id: backendDescriptor.id,
				label: backendDescriptor.label,
				initialSplats: trainer.initialSplatCount ?? trainer.splatCount,
				capacity: trainer.splatCount,
				memoryPlan: trainer.memoryPlan ?? null,
				trainingPixelsPerStep: backendDescriptor.sampledControls
					? trainOptions.samplesPerStep : trainer.dataset.width * trainer.dataset.height,
			},
			capabilities: { sharedStatus: Boolean(statusSlots), offscreenRender: Boolean(trainer.context),
				validationWorker: true, datasetSharing },
			details: message.details ?? {},
		}));
		if (wasRunning) {
			running = true;
			pumpToken += 1;
			schedulePump(pumpToken);
			publish(TrainerState.RUNNING, true);
		} else {
			publish(TrainerState.PAUSED, true);
		}
	} catch (error) {
		if (trainer === sourceTrainer) {
			trainer = sourceTrainer;
			canonicalDataset = sourceDataset;
		}
		throw error;
	} finally {
		stageTransitionPending = false;
	}
}

function switchTemporalPage(message) {
	if (stageTransitionPending) {
		throw new Error("Temporal paging cannot overlap a resolution-stage transition.");
	}
	const incoming = hydrateDatasetSharedViews(message.dataset);
	const nextDataset = trainer.replaceTemporalPage(incoming.dataset);
	canonicalDataset = incoming.dataset;
	renderOptions.viewIndices = resolveRenderViewIndices(nextDataset, renderOptions.viewIndices);
	validationWorker?.postMessage({
		version: WORKER_PROTOCOL_VERSION,
		type: "switch-dataset",
		dataset: nextDataset,
	});
	lastMetricRequestStep = trainer.stepCount;
	lastValidationRequestStep = trainer.stepCount;
	self.postMessage(protocolMessage(WorkerEvent.TEMPORAL_PAGE_READY, {
		step: trainer.stepCount,
		pageIndex: nextDataset.temporalPageIndex,
		frameCount: nextDataset.frameCount,
		frameIndices: nextDataset.frameIndices,
		details: message.details ?? {},
	}));
	publish(undefined, true);
}

self.onmessage = ({ data }) => {
	let message;
	try { message = assertProtocolMessage(data); } catch (error) { reportError(error); return; }
	if (message.type === WorkerCommand.INIT) {
		initialize(message).catch(reportError);
		return;
	}
	if (message.type === WorkerCommand.SWITCH_DATASET) {
		switchDataset(message).catch(reportError);
		return;
	}
	if (message.type === WorkerCommand.SWITCH_TEMPORAL_PAGE) {
		try { switchTemporalPage(message); } catch (error) { reportError(error); }
		return;
	}
	if (!trainer) return;
	switch (message.type) {
		case WorkerCommand.START:
			if (!running) {
				running = true; pumpToken += 1; completedAt = performance.now(); completedStep = trainer.stepCount;
				schedulePump(pumpToken); publish(TrainerState.RUNNING, true);
			}
			break;
		case WorkerCommand.PAUSE:
			running = false; pumpToken += 1; publish(TrainerState.PAUSED, true); break;
		case WorkerCommand.STEP:
			if (!running) {
				for (let index = 0; index < Math.max(1, message.count ?? 1); index += 1) trainer.trainStep(trainOptions);
				requestMetrics(); renderTrainer();
				lastRenderAt = performance.now(); publish(TrainerState.PAUSED, true);
			}
			break;
		case WorkerCommand.SET_TRAIN_OPTIONS:
			trainOptions = { ...trainOptions, ...message.options };
			trainOptions.camerasPerStep = backendDescriptor.sampledControls
				? resolveCamerasPerStep(trainer.trainViewIndices.length, trainOptions.camerasPerStep) : 1;
			break;
		case WorkerCommand.SET_RENDER_OPTIONS:
			renderOptions = { ...renderOptions, ...message.options };
			if (Object.hasOwn(message.options, "viewIndices")) {
				renderOptions.viewIndices = resolveRenderViewIndices(trainer.dataset, message.options.viewIndices);
			}
			if (!running) render(performance.now());
			break;
		case WorkerCommand.RESIZE:
			if (trainer.canvas) {
				trainer.canvas.width = Math.max(1, Math.floor(message.width));
				trainer.canvas.height = Math.max(1, Math.floor(message.height));
				renderTrainer();
				lastRenderAt = performance.now();
			}
			break;
		case WorkerCommand.REQUEST_METRICS:
			requestMetrics(); break;
		case WorkerCommand.REQUEST_VALIDATION:
			requestValidation(message.options); break;
		case WorkerCommand.DISPOSE:
			running = false; pumpToken += 1; validationWorker?.terminate(); trainer.dispose();
			publish(TrainerState.DISPOSED, true); self.postMessage(protocolMessage(WorkerEvent.DISPOSED)); self.close(); break;
		default:
			self.postMessage(protocolMessage(WorkerEvent.ERROR, { message: `Unknown command: ${message.type}` }));
	}
};
