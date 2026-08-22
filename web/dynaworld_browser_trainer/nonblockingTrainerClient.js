import {
	createSharedStatusBuffer, protocolMessage, readSharedStatus, TrainerState, WorkerCommand, WorkerEvent,
} from "./workerProtocol.js";
import { prepareDatasetForWorkerSharing } from "./datasetSharing.js";

function canTransferCanvas(canvas) {
	return Boolean(canvas && typeof canvas.transferControlToOffscreen === "function"
		&& typeof globalThis.OffscreenCanvas === "function");
}

export class NonblockingTrainerClient extends EventTarget {
	constructor({ workerUrl = new URL("./trainingWorker.js?v=20260821-stablegs-ablation-1", import.meta.url) } = {}) {
		super();
		this.worker = new Worker(workerUrl, { type: "module", name: "dynaworld-webgpu-trainer" });
		this.statusBuffer = createSharedStatusBuffer();
		this.lastMessageStatus = { state: TrainerState.BOOTING, step: 0, stepsPerSecond: 0 };
		this.capabilities = { sharedStatus: Boolean(this.statusBuffer), offscreenRender: false,
			validationWorker: false, datasetSharing: null };
		this.ready = new Promise((resolve, reject) => {
			this.resolveReady = resolve;
			this.rejectReady = reject;
		});
		this.pendingStage = null;
		this.pendingTemporalPage = null;
		this.worker.onmessage = ({ data }) => this.handleMessage(data);
		this.worker.onerror = (event) => {
			const error = new Error(event.message || "Training worker failed.");
			this.rejectReady?.(error);
			this.pendingStage?.reject(error);
			this.pendingStage = null;
			this.pendingTemporalPage?.reject(error);
			this.pendingTemporalPage = null;
			this.emit(WorkerEvent.ERROR, { message: error.message, error });
		};
	}

	async init({ dataset, canvas = null, trainerOptions = {}, trainOptions = {}, renderOptions = {}, schedule = {} }) {
		const sharedDataset = prepareDatasetForWorkerSharing(dataset);
		this.capabilities.datasetSharing = sharedDataset.telemetry;
		let offscreen = null;
		const transfer = [];
		if (canTransferCanvas(canvas)) {
			try {
				offscreen = canvas.transferControlToOffscreen();
				transfer.push(offscreen);
			} catch (error) {
				this.emit(WorkerEvent.CAPABILITY, { capability: "offscreenRender", available: false,
					reason: error?.message ?? String(error) });
			}
		} else if (canvas) {
			this.emit(WorkerEvent.CAPABILITY, { capability: "offscreenRender", available: false,
				reason: "OffscreenCanvas transfer is unavailable; optimization remains worker-owned." });
		}
		this.worker.postMessage(protocolMessage(WorkerCommand.INIT, {
			dataset: sharedDataset.dataset,
			datasetSharing: sharedDataset.telemetry,
			canvas: offscreen,
			statusBuffer: this.statusBuffer,
			trainerOptions,
			trainOptions,
			renderOptions,
			schedule,
		}), transfer);
		return this.ready;
	}

	handleMessage(message) {
		if (message?.type === WorkerEvent.STATUS) this.lastMessageStatus = message;
		if (message?.type === WorkerEvent.READY) {
			this.capabilities = message.capabilities;
			this.resolveReady?.(message);
			this.resolveReady = null;
			this.rejectReady = null;
		}
		if (message?.type === WorkerEvent.ERROR && this.rejectReady) {
			this.rejectReady(new Error(message.message));
			this.resolveReady = null;
			this.rejectReady = null;
		}
		if (message?.type === WorkerEvent.STAGE_READY && this.pendingStage) {
			this.pendingStage.resolve(message);
			this.pendingStage = null;
		}
		if (message?.type === WorkerEvent.TEMPORAL_PAGE_READY && this.pendingTemporalPage) {
			this.pendingTemporalPage.resolve(message);
			this.pendingTemporalPage = null;
		}
		if (message?.type === WorkerEvent.ERROR && this.pendingStage) {
			this.pendingStage.reject(new Error(message.message));
			this.pendingStage = null;
		}
		if (message?.type === WorkerEvent.ERROR && this.pendingTemporalPage) {
			this.pendingTemporalPage.reject(new Error(message.message));
			this.pendingTemporalPage = null;
		}
		this.emit(message?.type ?? "message", message);
	}

	emit(type, detail) {
		this.dispatchEvent(new CustomEvent(type, { detail }));
	}

	getStatus() {
		return readSharedStatus(this.statusBuffer) ?? this.lastMessageStatus;
	}

	command(type, payload = {}) {
		this.worker.postMessage(protocolMessage(type, payload));
	}

	start() { this.command(WorkerCommand.START); }
	pause() { this.command(WorkerCommand.PAUSE); }
	step(count = 1) { this.command(WorkerCommand.STEP, { count }); }
	setTrainOptions(options) { this.command(WorkerCommand.SET_TRAIN_OPTIONS, { options }); }
	setRenderOptions(options) { this.command(WorkerCommand.SET_RENDER_OPTIONS, { options }); }
	setCamerasPerStep(camerasPerStep) { this.setTrainOptions({ camerasPerStep }); }
	setRenderViewIndices(viewIndices) { this.setRenderOptions({ viewIndices }); }
	resize(width, height) { this.command(WorkerCommand.RESIZE, { width, height }); }
	requestMetrics() { this.command(WorkerCommand.REQUEST_METRICS); }
	requestValidation(options = {}) { this.command(WorkerCommand.REQUEST_VALIDATION, { options }); }
	switchDataset(dataset, details = {}) {
		if (this.pendingStage) throw new Error("A resolution-stage transition is already pending.");
		const sharedDataset = prepareDatasetForWorkerSharing(dataset);
		this.capabilities.datasetSharing = sharedDataset.telemetry;
		const transition = new Promise((resolve, reject) => {
			this.pendingStage = { resolve, reject };
		});
		try {
			this.command(WorkerCommand.SWITCH_DATASET, {
				dataset: sharedDataset.dataset,
				datasetSharing: sharedDataset.telemetry,
				details,
			});
		} catch (error) {
			this.pendingStage.reject(error);
			this.pendingStage = null;
		}
		return transition;
	}
	switchTemporalPage(dataset, details = {}) {
		if (this.pendingTemporalPage) throw new Error("A temporal-page transition is already pending.");
		const sharedDataset = prepareDatasetForWorkerSharing(dataset);
		this.capabilities.datasetSharing = sharedDataset.telemetry;
		const transition = new Promise((resolve, reject) => {
			this.pendingTemporalPage = { resolve, reject };
		});
		try {
			this.command(WorkerCommand.SWITCH_TEMPORAL_PAGE, {
				dataset: sharedDataset.dataset,
				datasetSharing: sharedDataset.telemetry,
				details,
			});
		} catch (error) {
			this.pendingTemporalPage.reject(error);
			this.pendingTemporalPage = null;
		}
		return transition;
	}
	dispose() { this.command(WorkerCommand.DISPOSE); }
}

export function createNonblockingTrainer(options) {
	return new NonblockingTrainerClient(options);
}
