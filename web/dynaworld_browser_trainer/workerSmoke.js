import { loadCalibratedMulticamDataset } from "./dataset.js";
import { createNonblockingTrainer } from "./nonblockingTrainerClient.js";

const output = document.querySelector("#result");
const results = { crossOriginIsolated, sharedStatus: false, offscreenRender: false, step: 0,
	loss: Number.NaN, trainPsnr: Number.NaN, heldoutPsnr: Number.NaN };

function show(state) {
	output.textContent = JSON.stringify({ state, ...results }, null, 2);
}

async function run() {
	const dataset = await loadCalibratedMulticamDataset();
	const client = createNonblockingTrainer();
	client.addEventListener("metrics", ({ detail }) => { results.loss = detail.loss; });
	const validation = new Promise((resolve, reject) => {
		client.addEventListener("validation", ({ detail }) => resolve(detail));
		client.addEventListener("error", ({ detail }) => reject(new Error(detail.message)), { once: true });
	});
	const ready = await client.init({ dataset, canvas: document.querySelector("#render"),
		trainerOptions: { splatCount: 96 }, schedule: { burstSteps: 2, metricEvery: 32, renderFps: 10 } });
	results.sharedStatus = ready.capabilities.sharedStatus;
	results.offscreenRender = ready.capabilities.offscreenRender;
	client.start();
	await new Promise((resolve, reject) => {
		const deadline = performance.now() + 20000;
		const poll = () => {
			const status = client.getStatus();
			results.step = status?.step ?? 0;
			results.loss = status?.loss ?? results.loss;
			show("training");
			if (results.step >= 96) { client.pause(); resolve(); return; }
			if (performance.now() > deadline) { reject(new Error("Timed out waiting for worker steps.")); return; }
			setTimeout(poll, 50);
		};
		poll();
	});
	client.requestValidation({ gridSize: 4 });
	const snapshot = await Promise.race([validation,
		new Promise((_, reject) => setTimeout(() => reject(new Error("Validation timed out.")), 20000))]);
	results.trainPsnr = snapshot.metrics.gridPsnr;
	results.heldoutPsnr = snapshot.metrics.heldoutPsnr;
	show("passed");
	setTimeout(() => client.dispose(), 30000);
}

run().catch((error) => {
	show(`failed: ${error?.stack ?? error}`);
	console.error(error);
});
