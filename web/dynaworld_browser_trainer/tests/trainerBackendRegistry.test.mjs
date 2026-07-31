import assert from "node:assert/strict";
import test from "node:test";
import {
	DEFAULT_TRAINER_BACKEND,
	TRAINER_BACKENDS,
	loadTrainerBackend,
	resolveTrainerBackend,
} from "../trainerBackendRegistry.js";

test("the parity-validated fast tiled lane is the default browser trainer", () => {
	assert.equal(DEFAULT_TRAINER_BACKEND, "tiled3d-fast");
	assert.strictEqual(resolveTrainerBackend(), TRAINER_BACKENDS["tiled3d-fast"]);
	assert.deepEqual(TRAINER_BACKENDS["tiled3d-fast"], {
		id: "tiled3d-fast",
		label: "Fast tiled full-frame",
		parameterSchema: "dynamic-splat-24f/v1",
		representation: "trajectory-gated dynamic 3DGS",
		objective: "0.8 L1 + 0.2 (1 - SSIM)",
		trainingUnit: "full image",
		maxAspectRatio: 6,
		sampledControls: false,
		defaultSchedule: { burstSteps: 8, metricEvery: 512, maxQueuedSteps: 32 },
	});
});

test("the direct tiled VJP remains registered as a matched reference", () => {
	const direct = resolveTrainerBackend("tiled3d");
	assert.equal(direct.label, "Direct tiled reference");
	assert.equal(direct.objective, TRAINER_BACKENDS["tiled3d-fast"].objective);
	assert.equal(direct.trainingUnit, "full image");
	assert.equal(direct.sampledControls, false);
});

test("sampled backend remains a compatible control with its own queue schedule", () => {
	const sampled = resolveTrainerBackend("sampled3d");
	assert.deepEqual(sampled, {
		id: "sampled3d",
		label: "Sampled rays (control)",
		parameterSchema: "dynamic-splat-24f/v1",
		representation: "trajectory-gated dynamic 3DGS",
		objective: "sampled RGB MSE + support guards",
		trainingUnit: "sampled rays",
		maxAspectRatio: 4,
		sampledControls: true,
		defaultSchedule: { burstSteps: 4, metricEvery: 256, maxQueuedSteps: 32 },
	});
	assert.equal(sampled.parameterSchema, TRAINER_BACKENDS["tiled3d-fast"].parameterSchema);
	assert.notEqual(sampled.defaultSchedule.burstSteps,
		TRAINER_BACKENDS.tiled3d.defaultSchedule.burstSteps);
});

test("registry descriptors and schedules are immutable and reject unknown ids", () => {
	assert.equal(Object.isFrozen(TRAINER_BACKENDS), true);
	for (const descriptor of Object.values(TRAINER_BACKENDS)) {
		assert.equal(Object.isFrozen(descriptor), true);
		assert.equal(Object.isFrozen(descriptor.defaultSchedule), true);
		assert.ok(descriptor.defaultSchedule.burstSteps > 0);
		assert.ok(descriptor.defaultSchedule.maxQueuedSteps >= descriptor.defaultSchedule.burstSteps);
	}
	assert.throws(() => resolveTrainerBackend("world-tubes"), /Unknown browser trainer backend/);
});

test("backend loader resolves each descriptor to its concrete trainer", async () => {
	for (const id of Object.keys(TRAINER_BACKENDS)) {
		const loaded = await loadTrainerBackend(id);
		assert.strictEqual(loaded.descriptor, TRAINER_BACKENDS[id]);
		assert.equal(typeof loaded.Trainer, "function");
		if (id === "tiled3d-fast") {
			assert.match(loaded.Trainer.name, /TiledFastTrainer$/);
		} else {
			assert.match(loaded.Trainer.name, id === "tiled3d" ? /TiledTrainer$/ : /WebGpu3dTrainer$/);
		}
	}
});
