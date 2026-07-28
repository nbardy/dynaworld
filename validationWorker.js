import { SPLAT_FLOATS, resolveTrainViewIndices } from "./trainerWebGpu3d.js";
import { computeSnapshotMetrics, snapshotUpdateRatios } from "./snapshotMetrics.js";
import { WORKER_PROTOCOL_VERSION } from "./workerProtocol.js";

let dataset = null;
let initialParams = null;
let previousParams = null;

function sigmoid(value) {
	return 1 / (1 + Math.exp(-value));
}

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

function parameterStats(params, splatCount) {
	let active = 0;
	let opacity = 0;
	let maxOpacity = 0;
	let radius = 0;
	let aspectRatio = 0;
	for (let index = 0; index < splatCount; index += 1) {
		const base = index * SPLAT_FLOATS;
		const alpha = sigmoid(params[base + 23]);
		const scales = [
			Math.exp(params[base + 12]),
			Math.exp(params[base + 13]),
			Math.exp(params[base + 14]),
		];
		opacity += alpha;
		maxOpacity = Math.max(maxOpacity, alpha);
		radius += Math.cbrt(scales[0] * scales[1] * scales[2]);
		aspectRatio += Math.max(...scales) / Math.max(1e-8, Math.min(...scales));
		if (alpha > 0.05) active += 1;
	}
	return {
		activeSplats: active,
		meanOpacity: opacity / splatCount,
		meanRadius: radius / splatCount,
		meanAspectRatio: aspectRatio / splatCount,
		motionMaxAlpha: maxOpacity,
	};
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
			...parameterStats(params, splatCount),
			parameterDelta: meanAbsoluteDelta(initialParams, params),
			parameterUpdateRatios: snapshotUpdateRatios(previousParams, params),
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
