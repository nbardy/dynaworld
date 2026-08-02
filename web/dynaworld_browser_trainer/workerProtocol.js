export const WORKER_PROTOCOL_VERSION = 3;

export const WorkerCommand = Object.freeze({
	INIT: "init",
	START: "start",
	PAUSE: "pause",
	STEP: "step",
	SET_TRAIN_OPTIONS: "set-train-options",
	SET_RENDER_OPTIONS: "set-render-options",
	RESIZE: "resize",
	REQUEST_METRICS: "request-metrics",
	REQUEST_VALIDATION: "request-validation",
	SWITCH_DATASET: "switch-dataset",
	SWITCH_TEMPORAL_PAGE: "switch-temporal-page",
	DISPOSE: "dispose",
});

export const WorkerEvent = Object.freeze({
	READY: "ready",
	STATUS: "status",
	METRICS: "metrics",
	VALIDATION: "validation",
	STAGE_STARTED: "stage-started",
	STAGE_READY: "stage-ready",
	TEMPORAL_PAGE_READY: "temporal-page-ready",
	CAPABILITY: "capability",
	ERROR: "error",
	DISPOSED: "disposed",
});

export const TrainerState = Object.freeze({
	BOOTING: 0,
	READY: 1,
	RUNNING: 2,
	PAUSED: 3,
	FAILED: 4,
	DISPOSED: 5,
});

export const StatusFlag = Object.freeze({
	SHARED_MEMORY: 1 << 0,
	OFFSCREEN_RENDER: 1 << 1,
	VALIDATION_WORKER: 1 << 2,
	METRICS_PENDING: 1 << 3,
	VALIDATION_PENDING: 1 << 4,
});

export const StatusSlot = Object.freeze({
	VERSION: 0,
	SEQUENCE: 1,
	STATE: 2,
	STEP: 3,
	STEPS_PER_SECOND: 4,
	LOSS: 5,
	PSNR: 6,
	SSIM: 7,
	FLAGS: 8,
	LAST_METRIC_STEP: 9,
	LAST_VALIDATION_STEP: 10,
	CAMERAS_PER_STEP: 11,
	CAMERA_ROTATION_START: 12,
	TRAIN_VIEW_COUNT: 13,
	COUNT: 16,
});

const encodeBuffer = new ArrayBuffer(4);
const encodeFloat = new Float32Array(encodeBuffer);
const encodeInt = new Int32Array(encodeBuffer);

function floatToBits(value) {
	encodeFloat[0] = Number(value);
	return encodeInt[0];
}

function bitsToFloat(value) {
	encodeInt[0] = value;
	return encodeFloat[0];
}

export function canUseSharedStatus(scope = globalThis) {
	return scope.crossOriginIsolated === true && typeof scope.SharedArrayBuffer === "function";
}

export function createSharedStatusBuffer(scope = globalThis) {
	if (!canUseSharedStatus(scope)) return null;
	const buffer = new scope.SharedArrayBuffer(StatusSlot.COUNT * Int32Array.BYTES_PER_ELEMENT);
	const slots = new Int32Array(buffer);
	Atomics.store(slots, StatusSlot.VERSION, WORKER_PROTOCOL_VERSION);
	Atomics.store(slots, StatusSlot.STATE, TrainerState.BOOTING);
	Atomics.store(slots, StatusSlot.LOSS, floatToBits(Number.NaN));
	Atomics.store(slots, StatusSlot.PSNR, floatToBits(Number.NaN));
	Atomics.store(slots, StatusSlot.SSIM, floatToBits(Number.NaN));
	return buffer;
}

export function publishSharedStatus(slots, status) {
	if (!slots) return;
	Atomics.add(slots, StatusSlot.SEQUENCE, 1);
	if (status.state != null) Atomics.store(slots, StatusSlot.STATE, status.state);
	if (status.step != null) Atomics.store(slots, StatusSlot.STEP, status.step);
	if (status.stepsPerSecond != null) {
		Atomics.store(slots, StatusSlot.STEPS_PER_SECOND, floatToBits(status.stepsPerSecond));
	}
	if (status.loss != null) Atomics.store(slots, StatusSlot.LOSS, floatToBits(status.loss));
	if (status.psnr != null) Atomics.store(slots, StatusSlot.PSNR, floatToBits(status.psnr));
	if (status.ssim != null) Atomics.store(slots, StatusSlot.SSIM, floatToBits(status.ssim));
	if (status.flags != null) Atomics.store(slots, StatusSlot.FLAGS, status.flags);
	if (status.lastMetricStep != null) Atomics.store(slots, StatusSlot.LAST_METRIC_STEP, status.lastMetricStep);
	if (status.lastValidationStep != null) {
		Atomics.store(slots, StatusSlot.LAST_VALIDATION_STEP, status.lastValidationStep);
	}
	if (status.camerasPerStep != null) Atomics.store(slots, StatusSlot.CAMERAS_PER_STEP, status.camerasPerStep);
	if (status.cameraRotationStart != null) {
		Atomics.store(slots, StatusSlot.CAMERA_ROTATION_START, status.cameraRotationStart);
	}
	if (status.trainViewCount != null) Atomics.store(slots, StatusSlot.TRAIN_VIEW_COUNT, status.trainViewCount);
	Atomics.add(slots, StatusSlot.SEQUENCE, 1);
}

export function readSharedStatus(buffer) {
	if (!buffer) return null;
	const slots = buffer instanceof Int32Array ? buffer : new Int32Array(buffer);
	for (let attempt = 0; attempt < 8; attempt += 1) {
		const before = Atomics.load(slots, StatusSlot.SEQUENCE);
		if (before & 1) continue;
		const status = {
			version: Atomics.load(slots, StatusSlot.VERSION),
			state: Atomics.load(slots, StatusSlot.STATE),
			step: Atomics.load(slots, StatusSlot.STEP),
			stepsPerSecond: bitsToFloat(Atomics.load(slots, StatusSlot.STEPS_PER_SECOND)),
			loss: bitsToFloat(Atomics.load(slots, StatusSlot.LOSS)),
			psnr: bitsToFloat(Atomics.load(slots, StatusSlot.PSNR)),
			ssim: bitsToFloat(Atomics.load(slots, StatusSlot.SSIM)),
			flags: Atomics.load(slots, StatusSlot.FLAGS),
			lastMetricStep: Atomics.load(slots, StatusSlot.LAST_METRIC_STEP),
			lastValidationStep: Atomics.load(slots, StatusSlot.LAST_VALIDATION_STEP),
			camerasPerStep: Atomics.load(slots, StatusSlot.CAMERAS_PER_STEP),
			cameraRotationStart: Atomics.load(slots, StatusSlot.CAMERA_ROTATION_START),
			trainViewCount: Atomics.load(slots, StatusSlot.TRAIN_VIEW_COUNT),
		};
		if (before === Atomics.load(slots, StatusSlot.SEQUENCE)) return status;
	}
	return null;
}

export function assertProtocolMessage(message) {
	if (!message || typeof message !== "object") throw new TypeError("Worker message must be an object.");
	if (message.version !== WORKER_PROTOCOL_VERSION) {
		throw new Error(`Worker protocol version ${message.version} is unsupported.`);
	}
	if (typeof message.type !== "string") throw new TypeError("Worker message type is required.");
	return message;
}

export function protocolMessage(type, payload = {}) {
	return { version: WORKER_PROTOCOL_VERSION, type, ...payload };
}
