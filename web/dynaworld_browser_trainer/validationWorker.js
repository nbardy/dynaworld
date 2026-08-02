import {
	SPLAT_FLOATS,
	resolveTrainViewIndices,
} from "./trainerWebGpu3d.js?v=20260803-fullfps-pixelgs-1";
import {
	computeSnapshotMetrics,
	snapshotUpdateRatios,
	summarizeSplatParameters,
} from "./snapshotMetrics.js?v=20260803-fullfps-pixelgs-1";
import { WORKER_PROTOCOL_VERSION } from "./workerProtocol.js?v=20260803-fullfps-pixelgs-1";
import { hydrateDatasetSharedViews } from "./datasetSharing.js";

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
		try {
			const hydrated = hydrateDatasetSharedViews(data.dataset);
			dataset = hydrated.dataset;
			initialParams = new Float32Array(data.initialParams);
			previousParams = initialParams.slice();
			self.postMessage({
				version: WORKER_PROTOCOL_VERSION,
				type: "ready",
				datasetSharing: hydrated.telemetry,
			});
		} catch (error) {
			self.postMessage({
				version: WORKER_PROTOCOL_VERSION,
				type: "error",
				message: error?.message ?? String(error),
				stack: error?.stack,
			});
		}
		return;
	}
	if (data.type === "switch-dataset") {
		try {
			dataset = hydrateDatasetSharedViews(data.dataset).dataset;
			self.postMessage({
				version: WORKER_PROTOCOL_VERSION,
				type: "temporal-page-ready",
				pageIndex: dataset.temporalPageIndex,
			});
		} catch (error) {
			self.postMessage({
				version: WORKER_PROTOCOL_VERSION,
				type: "error",
				message: error?.message ?? String(error),
				stack: error?.stack,
			});
		}
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
		// Keep the longitudinal metric on two representative cameras x all
		// times, then add one coherent center-time image from every train
		// camera. This exposes a weak camera without repeating a 17 x 16 CPU
		// sweep in the validation worker.
		const cameraSweepFrame = Math.floor((dataset.frameCount - 1) / 2);
		const cameraSweep = computeSnapshotMetrics(dataset, params, {
			...options,
			views: resolveTrainViewIndices(dataset),
			frames: [cameraSweepFrame],
		});
		const cameraPsnr = cameraSweep.snapshots
			.map(({ viewIndex, metrics }) => ({
				viewIndex,
				name: dataset.cameras[viewIndex].name,
				psnr: metrics.psnr,
				ssim: metrics.ssim,
			}))
			.sort((left, right) => left.psnr - right.psnr);
		const weakestCamera = cameraPsnr[0];
		const medianCamera = cameraPsnr[Math.floor((cameraPsnr.length - 1) / 2)];
		const strongestCamera = cameraPsnr.at(-1);
		const splatCount = options.splatCount ?? params.length / SPLAT_FLOATS;
		const activeValues = splatCount * SPLAT_FLOATS;
		const metrics = {
			gridLoss: train.mse,
			gridMae: train.mae,
			gridPsnr: train.psnr,
			gridSsim: train.ssim,
			gridDetailMae: train.detailMae,
			gridDetailErrorRatio: train.detailErrorRatio,
			gridLowPassPsnr: train.lowPassPsnr,
			heldoutLoss: heldout.mse,
			heldoutMae: heldout.mae,
			heldoutPsnr: heldout.psnr,
			heldoutSsim: heldout.ssim,
			heldoutDetailMae: heldout.detailMae,
			heldoutDetailErrorRatio: heldout.detailErrorRatio,
			heldoutLowPassPsnr: heldout.lowPassPsnr,
			heldoutCoverage: heldout.coverage,
			trainPrimitiveEvaluationsPerPixel: train.primitiveEvaluations / train.pixelCount,
			trainBinnedReferencesPerImage: train.binnedReferences / train.selectionCount,
			cameraSweepFrame,
			weakestTrainCamera: weakestCamera?.name ?? null,
			weakestTrainCameraPsnr: weakestCamera?.psnr ?? Number.NaN,
			medianTrainCameraPsnr: medianCamera?.psnr ?? Number.NaN,
			strongestTrainCameraPsnr: strongestCamera?.psnr ?? Number.NaN,
			weakestTrainCameraSsim: weakestCamera?.ssim ?? Number.NaN,
			motionLoss: Number.NaN,
			motionCoverage: train.coverage,
			staticCoverage: Number.NaN,
			...summarizeSplatParameters(params, {
				splatCount,
				temporalSigma: options.temporalSigma,
				frameCount: dataset.frameCount,
				maxAspectRatio: data.options?.maxAspectRatio ?? 3,
			}),
			parameterDelta: meanAbsoluteDelta(
				initialParams.subarray(0, activeValues),
				params.subarray(0, activeValues),
			),
			parameterUpdateRatios: snapshotUpdateRatios(
				previousParams.subarray(0, activeValues),
				params.subarray(0, activeValues),
			),
			learningRateMultipliers: data.options?.learningRateMultipliers ?? null,
			totalRecycled: data.options?.totalRecycled ?? 0,
			validationDurationMs: performance.now() - startedAt,
			validationContract: {
				allPixels: true,
				frames: dataset.frameCount,
				trainViews: trainViewIndices.map((view) => dataset.cameras[view].name),
				cameraSweepViews: cameraPsnr.map(({ name }) => name),
				cameraSweepFrame,
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
