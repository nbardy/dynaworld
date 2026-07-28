import { loadCalibratedMulticamDataset } from "./dataset.js?v=20260722-baseline1";
import { DynamicSplatWebGpuTrainer } from "./trainerWebGpu.js?v=20260722-baseline1";

const TRAIN_OPTIONS = Object.freeze({
	learningRate: 1.25, samplesPerStep: 96, modelMode: 0, temporalSigma: 0.30,
	motionSampleRate: 0.90, staticSampleRate: 0.08, motionCoverageTarget: 0.52,
});

function sourceViewDataset(multicam, viewIndex = 0) {
	const source = multicam.viewDatasets[viewIndex];
	const valuesPerView = source.width * source.height * source.frameCount;
	const start = viewIndex * valuesPerView; const end = start + valuesPerView;
	const localize = (samples) => new Uint32Array(Array.from(samples)
		.filter((sample) => sample >= start && sample < end).map((sample) => sample - start));
	return { ...source, motionSamples: localize(multicam.motionSamples),
		staticSamples: localize(multicam.staticSamples) };
}

async function run() {
	const output = document.querySelector("#result");
	const trainer = new DynamicSplatWebGpuTrainer(document.querySelector("#baselineCanvas"));
	const dataset = sourceViewDataset(await loadCalibratedMulticamDataset());
	await trainer.init(dataset, { splatCount: 768 });
	const initial = await trainer.readValidationMetrics({ gridSize: 16 });
	const start = performance.now(); let loss = Number.NaN;
	for (let step = 0; step < 64; step += 1) {
		trainer.trainStep(TRAIN_OPTIONS);
		if ((step + 1) % 8 === 0) {
			loss = await trainer.readLoss();
			trainer.render(0.35, 0, 0.30, 0);
			output.textContent = `step ${step + 1}/64 · sampled loss ${loss.toFixed(6)}`;
			await new Promise((resolve) => setTimeout(resolve, 0));
		}
	}
	const final = await trainer.readValidationMetrics({ gridSize: 16 });
	trainer.render(0.35, 0, 0.30, 0);
	output.textContent = JSON.stringify({
		contract: "legacy single-source image-space baseline with source-view mean background",
		steps: 64, wallSeconds: (performance.now() - start) / 1000, sampledLoss: loss,
		initial: { psnr: initial.gridPsnr, ssim: initial.gridSsim },
		final: { psnr: final.gridPsnr, ssim: final.gridSsim },
	}, null, 2);
}

run().catch((error) => {
	document.querySelector("#result").textContent = error.stack ?? String(error);
	console.error(error);
});
