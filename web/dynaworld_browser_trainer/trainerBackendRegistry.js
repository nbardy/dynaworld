export const DEFAULT_TRAINER_BACKEND = "tiled3d";

export const TRAINER_BACKENDS = Object.freeze({
	tiled3d: Object.freeze({
		id: "tiled3d",
		label: "Tiled full-frame",
		parameterSchema: "dynamic-splat-24f/v1",
		representation: "trajectory-gated dynamic 3DGS",
		objective: "0.8 L1 + 0.2 (1 - SSIM)",
		trainingUnit: "full image",
		maxAspectRatio: 6,
		sampledControls: false,
		defaultSchedule: Object.freeze({ burstSteps: 8, metricEvery: 256, maxQueuedSteps: 32 }),
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
	const module = descriptor.id === "tiled3d"
		? await import("./trainerWebGpu3dTiled.js")
		: await import("./trainerWebGpu3d.js");
	return {
		descriptor,
		Trainer: descriptor.id === "tiled3d"
			? module.DynamicSplatWebGpu3dTiledTrainer
			: module.DynamicSplatWebGpu3dTrainer,
	};
}
