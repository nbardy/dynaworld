import {
	AFFINE_STAR_BROWSER_CONTRACT,
	AffineStarWebGpuTrainer,
	affineStarLossAndGradients,
	createTinyAffineStarFixture,
} from "./trainerWebGpuStar.js";

async function run() {
	const fixture = createTinyAffineStarFixture();
	const trainer = await AffineStarWebGpuTrainer.create(fixture.initialState, fixture.samples);
	const gradientCheck = await trainer.gradientCheck(fixture.samples);
	if (!gradientCheck.passed) throw new Error(`WebGPU affine STAR gradient check failed: ${JSON.stringify(gradientCheck)}`);
	const initialLoss = affineStarLossAndGradients(await trainer.readParams(), fixture.samples).loss;
	const warmup = 20;
	const measuredSteps = 200;
	for (let step = 0; step < warmup; step += 1) trainer.trainStep({ learningRate: 0.04 });
	await trainer.readParams();
	const startedAt = performance.now();
	for (let step = 0; step < measuredSteps; step += 1) trainer.trainStep({ learningRate: 0.04 });
	const finalState = await trainer.readParams();
	const elapsedMs = performance.now() - startedAt;
	const finalLoss = affineStarLossAndGradients(finalState, fixture.samples).loss;
	if (!(finalLoss < initialLoss)) throw new Error(`Affine STAR tiny fit did not converge: ${initialLoss} -> ${finalLoss}`);
	const report = {
		contract: AFFINE_STAR_BROWSER_CONTRACT,
		adapter: trainer.adapterName,
		fixture: { tubeCount: 2, sampleCount: fixture.samples.length, frameCount: fixture.frames, imageSize: [fixture.width, fixture.height] },
		gradientCheck,
		benchmark: { warmupSteps: warmup, measuredSteps, elapsedMs, stepsPerSecond: measuredSteps * 1000 / elapsedMs },
		convergence: { initialLoss, finalLoss, ratio: finalLoss / initialLoss },
	};
	document.querySelector("#result").textContent = JSON.stringify(report, null, 2);
	console.info("Affine STAR WebGPU benchmark", report);
	trainer.dispose();
}

run().catch((error) => {
	document.querySelector("#result").textContent = error.stack ?? String(error);
	console.error(error);
});
