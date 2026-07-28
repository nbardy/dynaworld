import { loadCalibratedMulticamDataset } from "./dataset.js";
import { DynamicGsWebGpuTrainer, DYNAMIC_GS_LIMITS } from "./trainerWebGpuDynamicGs.js";

const output = document.getElementById("output");
document.getElementById("run").addEventListener("click", async () => {
	output.textContent = "Loading calibrated bundle..."; let trainer;
	try {
		const dataset = await loadCalibratedMulticamDataset(); const splatCount = 16; const samplesPerStep = 64; trainer = await DynamicGsWebGpuTrainer.create(dataset, { splatCount });
		trainer.device.pushErrorScope("validation");
		for (let index = 0; index < 20; index += 1) trainer.trainStep({ samplesPerStep }); await trainer.device.queue.onSubmittedWorkDone();
		const validationError = await trainer.device.popErrorScope(); if (validationError) throw validationError;
		const steps = 200; const start = performance.now(); for (let index = 0; index < steps; index += 1) trainer.trainStep({ samplesPerStep }); await trainer.device.queue.onSubmittedWorkDone(); const elapsedMs = performance.now() - start;
		const probe = dataset.motionSamples[0]; const probeBase = probe * 4;
		const result = { contract: "bounded_per_frame_dynamic_3dgs/v1", steps, elapsedMs, stepsPerSecond: steps * 1000 / elapsedMs, sampledPixelsPerSecond: steps * samplesPerStep * 1000 / elapsedMs, primitiveEvaluationsPerSecond: steps * samplesPerStep * splatCount * 1000 / elapsedMs, finalSampleLoss: await trainer.readLoss(), motionProbeRgb: Array.from(dataset.frames.subarray(probeBase, probeBase + 3)), frameCount: dataset.frameCount, trainCameraCount: dataset.cameras.filter((camera) => camera.role !== "heldout").length, splatCount, samplesPerStep, stateBytes: dataset.frameCount * splatCount * 16 * 4, limits: DYNAMIC_GS_LIMITS };
		output.textContent = JSON.stringify(result, null, 2);
	} catch (error) { output.textContent = error.stack ?? error.message ?? String(error); } finally { trainer?.dispose(); }
});
