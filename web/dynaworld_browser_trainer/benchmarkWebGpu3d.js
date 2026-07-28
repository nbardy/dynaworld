import { loadCalibratedMulticamDataset } from "./dataset.js";
import { DynamicSplatWebGpu3dTrainer } from "./trainerWebGpu3d.js";

const TRAIN_OPTIONS = {
	learningRate: 1.25,
	samplesPerStep: 96,
	modelMode: 0,
	temporalSigma: 0.30,
	motionSampleRate: 0.90,
	staticSampleRate: 0.08,
	motionCoverageTarget: 0.52,
};

function median(values) {
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[Math.floor(sorted.length / 2)];
}

async function synchronizedSteps(trainer, count) {
	const start = performance.now();
	for (let step = 0; step < count; step += 1) trainer.trainStep(TRAIN_OPTIONS);
	const loss = await trainer.readLoss(TRAIN_OPTIONS);
	return { elapsedMs: performance.now() - start, loss };
}

async function measuredValidation(trainer) {
	const frameGaps = [];
	let previousFrame = await new Promise((resolve) => requestAnimationFrame(resolve));
	let monitor = true;
	const monitorFrames = (now) => {
		frameGaps.push(now - previousFrame);
		previousFrame = now;
		if (monitor) requestAnimationFrame(monitorFrames);
	};
	requestAnimationFrame(monitorFrames);
	const start = performance.now();
	const metrics = await trainer.readValidationMetrics({ modelMode: 0, temporalSigma: 0.30, gridSize: 12 });
	const wallMs = performance.now() - start;
	monitor = false;
	await new Promise((resolve) => requestAnimationFrame(resolve));
	return {
		wallMs: Number(wallMs.toFixed(2)),
		maxAnimationFrameGapMs: Number(Math.max(...frameGaps).toFixed(2)),
		trainPsnr: metrics.gridPsnr,
		heldoutPsnr: metrics.heldoutPsnr,
	};
}

async function run() {
	const canvas = document.querySelector("#benchmarkCanvas");
	const trainer = new DynamicSplatWebGpu3dTrainer(canvas);
	const dataset = await loadCalibratedMulticamDataset();
	await trainer.init(dataset, { splatCount: 768 });
	await synchronizedSteps(trainer, 64);

	const isolationMs = [];
	for (let repeat = 0; repeat < 5; repeat += 1) {
		const sample = await synchronizedSteps(trainer, 256);
		isolationMs.push(sample.elapsedMs);
	}

	const frameGaps = [];
	let previousFrame = performance.now();
	let monitor = true;
	const monitorFrames = (now) => {
		frameGaps.push(now - previousFrame);
		previousFrame = now;
		if (monitor) requestAnimationFrame(monitorFrames);
	};
	requestAnimationFrame(monitorFrames);
	const usableStart = performance.now();
	let lastRender = usableStart;
	for (let step = 0; step < 256; step += 8) {
		for (let batchStep = 0; batchStep < 8; batchStep += 1) trainer.trainStep(TRAIN_OPTIONS);
		await new Promise((resolve) => setTimeout(resolve, 0));
		const now = performance.now();
		if (now - lastRender >= 1000 / 30) {
			trainer.render(0.35, 0, 0.30, 0, 0);
			lastRender = now;
		}
		if ((step + 8) % 64 === 0) await trainer.readLoss(TRAIN_OPTIONS);
	}
	const metrics = await trainer.readValidationMetrics({ modelMode: 0, temporalSigma: 0.30, gridSize: 12 });
	const sampledLoss = await trainer.readLoss(TRAIN_OPTIONS);
	const usableMs = performance.now() - usableStart;
	monitor = false;
	await new Promise((resolve) => requestAnimationFrame(resolve));
	const validationTiming = await measuredValidation(trainer);

	const medianIsolationMs = median(isolationMs);
	const result = {
		contract: { samplesPerStep: 96, splatCount: 768, validationGrid: 12 },
		adapter: trainer.adapterName,
		isolation: {
			repeatMs: isolationMs.map((value) => Number(value.toFixed(2))),
			medianStepsPerSecond: Number((256000 / medianIsolationMs).toFixed(1)),
		},
		usableDefault: {
			steps: 256,
			wallMsIncludingLossPreviewAndValidation: Number(usableMs.toFixed(2)),
			effectiveStepsPerSecond: Number((256000 / usableMs).toFixed(1)),
			maxAnimationFrameGapMs: Number(Math.max(...frameGaps).toFixed(2)),
			p95AnimationFrameGapMs: Number(frameGaps.sort((a, b) => a - b)[Math.floor(frameGaps.length * 0.95)].toFixed(2)),
			sampledLoss,
			validation: { trainPsnr: metrics.gridPsnr, heldoutPsnr: metrics.heldoutPsnr },
		},
		validationTiming,
	};
	document.querySelector("#result").textContent = JSON.stringify(result, null, 2);
	console.info("DynaWorld WebGPU 3D benchmark", result);
	trainer.dispose();
}

run().catch((error) => {
	document.querySelector("#result").textContent = error.stack ?? String(error);
	console.error(error);
});
