export const DEFAULT_TRAINER_BACKEND = "tiled3d-fast";
// This only controls asynchronous UI readback cadence. The tiled kernel keeps
// a GPU-resident full camera/time-cycle mean, so burst quantization cannot
// alias the chart back onto a small recurring set of training pairs.
export const TILED_METRIC_INTERVAL = 256;

export const TRAINER_BACKENDS = Object.freeze({
	"tiled3d-fast": Object.freeze({
		id: "tiled3d-fast",
		label: "Fast tiled full-frame",
		parameterSchema: "dynamic-splat-24f/v1",
		representation: "trajectory-gated dynamic 3DGS",
		objective: "0.8 L1 + 0.2 (1 - SSIM)",
		trainingUnit: "full image",
		maxAspectRatio: 6,
		sampledControls: false,
		defaultSchedule: Object.freeze({
			burstSteps: 8, metricEvery: TILED_METRIC_INTERVAL, maxQueuedSteps: 32,
		}),
	}),
	tiled3d: Object.freeze({
		id: "tiled3d",
		label: "Direct tiled reference",
		parameterSchema: "dynamic-splat-24f/v1",
		representation: "trajectory-gated dynamic 3DGS",
		objective: "0.8 L1 + 0.2 (1 - SSIM)",
		trainingUnit: "full image",
		maxAspectRatio: 6,
		sampledControls: false,
		defaultSchedule: Object.freeze({
			burstSteps: 8, metricEvery: TILED_METRIC_INTERVAL, maxQueuedSteps: 32,
		}),
	}),
	sampled3d: Object.freeze({
		id: "sampled3d",
		label: "Sampled rays (control)",
		parameterSchema: "dynamic-splat-24f/v1",
		representation: "trajectory-gated dynamic 3DGS",
		objective: "sampled RGB MSE + support guards",
		trainingUnit: "sampled rays",
		maxAspectRatio: 4,
		sampledControls: true,
		defaultSchedule: Object.freeze({ burstSteps: 4, metricEvery: 256, maxQueuedSteps: 32 }),
	}),
});

export function resolveTrainerBackend(id = DEFAULT_TRAINER_BACKEND) {
	const backend = TRAINER_BACKENDS[id];
	if (!backend) {
		throw new RangeError(`Unknown browser trainer backend "${id}". `
			+ `Expected one of: ${Object.keys(TRAINER_BACKENDS).join(", ")}.`);
	}
	return backend;
}

export async function loadTrainerBackend(id = DEFAULT_TRAINER_BACKEND) {
	const descriptor = resolveTrainerBackend(id);
	if (descriptor.id === "tiled3d-fast") {
		const module = await import("./trainerWebGpu3dTiledFast.js?v=20260731-fasttiles6");
		return { descriptor, Trainer: module.DynamicSplatWebGpu3dTiledFastTrainer };
	}
	if (descriptor.id === "tiled3d") {
		const module = await import("./trainerWebGpu3dTiled.js?v=20260731-fasttiles6");
		return { descriptor, Trainer: module.DynamicSplatWebGpu3dTiledTrainer };
	}
	const module = await import("./trainerWebGpu3d.js?v=20260731-fasttiles6");
	return {
		descriptor,
		Trainer: module.DynamicSplatWebGpu3dTrainer,
	};
}
