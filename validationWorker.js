import { SPLAT_FLOATS, resolveTrainViewIndices } from "./trainerWebGpu3d.js";
import {
	computeSnapshotMetrics,
	snapshotUpdateRatios,
	summarizeSplatParameters,
} from "./snapshotMetrics.js";
import { WORKER_PROTOCOL_VERSION } from "./workerProtocol.js";

let dataset = null;
let initialParams = null;
let previousParams = null;

function representativeTrainViews() {
	const trainViews = resolveTrainViewIndices(dataset);
	const preferred = (dataset.comparisonViewIndices ?? [])
		.filter((view, index, values) => trainViews.includes(view) && values.indexOf(view) === index);
	for (const view of trainViews) {
		if (!preferred.includes(view)) preferred.push(view);
	}
	return preferred.slice(0, Math.min(2, preferred.length));
}

function heldoutViews() {
	const heldout = dataset.cameras
		.map((camera, index) => camera.role === "heldout" ? index : -1)
		.filter((index) => index >= 0);
	if (!heldout.length && Number.isSafeInteger(dataset.heldoutViewIndex)
		&& dataset.heldoutViewIndex >= 0) heldout.push(dataset.heldoutViewIndex);
	if (!heldout.length) throw new Error("Full-image validation requires a heldout camera.");
	return heldout;
}

function meanAbsoluteDelta(before, after) {
	if (before?.length !== after.length) return Number.NaN;
	let delta = 0;
	for (let index = 0; index < after.length; index += 1) delta += Math.abs(after[index] - before[index]);
	return delta / after.length;
}

self.onmessage = ({ data }) => {
	if (data?.version !== WORKER_PROTOCOL_VERSION) return;
	if (data.type === "init") {
		dataset = data.dataset;
		initialParams = new Float32Array(data.initialParams);
		previousParams = initialParams.slice();
		self.postMessage({ version: WORKER_PROTOCOL_VERSION, type: "ready" });
		return;
	}
	if (data.type !== "validate" || !dataset) return;
	try {
		const startedAt = performance.now();
		const params = new Float32Array(data.params);
		const options = {
			splatCount: data.options?.splatCount,
			modelMode: data.options?.modelMode ?? 0,
			temporalSigma: data.options?.temporalSigma ?? 0.30,
		};
		const trainViewIndices = representativeTrainViews();
		const heldoutViewIndices = heldoutViews();
		const train = computeSnapshotMetrics(dataset, params, {
			...options,
			views: trainViewIndices,
			frames: "all",
		});
		const heldout = computeSnapshotMetrics(dataset, params, {
			...options,
			views: heldoutViewIndices,
			frames: "all",
		});
		const splatCount = options.splatCount ?? params.length / SPLAT_FLOATS;
		const metrics = {
			gridLoss: train.mse,
			gridMae: train.mae,
			gridPsnr: train.psnr,
			gridSsim: train.ssim,
			heldoutLoss: heldout.mse,
			heldoutMae: heldout.mae,
			heldoutPsnr: heldout.psnr,
			heldoutSsim: heldout.ssim,
			heldoutCoverage: heldout.coverage,
			motionLoss: Number.NaN,
			motionCoverage: train.coverage,
			staticCoverage: Number.NaN,
			...summarizeSplatParameters(params, {
				splatCount,
				temporalSigma: options.temporalSigma,
				frameCount: dataset.frameCount,
				maxAspectRatio: data.options?.maxAspectRatio ?? 3,
			}),
			parameterDelta: meanAbsoluteDelta(initialParams, params),
			parameterUpdateRatios: snapshotUpdateRatios(previousParams, params),
			learningRateMultipliers: data.options?.learningRateMultipliers ?? null,
			totalRecycled: data.options?.totalRecycled ?? 0,
			validationDurationMs: performance.now() - startedAt,
			validationContract: {
				allPixels: true,
				frames: dataset.frameCount,
				trainViews: trainViewIndices.map((view) => dataset.cameras[view].name),
				heldoutViews: heldoutViewIndices.map((view) => dataset.cameras[view].name),
				ssim: "channelwise_11x11_gaussian_sigma1.5_reflect",
			},
		};
		previousParams = params.slice();
		self.postMessage({ version: WORKER_PROTOCOL_VERSION, type: "validation", step: data.step, metrics });
	} catch (error) {
		self.postMessage({ version: WORKER_PROTOCOL_VERSION, type: "error", step: data.step,
			message: error?.message ?? String(error), stack: error?.stack });
	}
};
