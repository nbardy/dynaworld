import { decodeFrameRgb } from "./dataset.js?v=20260814-camera-stress-1";
import {
	cameraRigRadius,
	createOrbitCameraState,
	dollyOrbitCamera,
	orbitPreviewCamera,
	translateOrbitCamera,
} from "./orbitCamera.js?v=20260814-camera-stress-1";
import { renderSnapshotFrame } from "./snapshotMetrics.js?v=20260821-stablegs-ablation-1";

export const CAMERA_STRESS_DEFAULTS = Object.freeze({
	opticalZoom: 1.05,
	principalShift: 0.015,
	dollyFraction: 0.03,
	lateralFraction: 0.015,
	orbitRadians: Math.PI / 90,
	maxHeight: 48,
	largeFootprintFraction: 0.25,
	nearDepthGamma: 0.37,
});

function assertFiniteCamera(camera) {
	if (!camera?.worldToCamera || camera.worldToCamera.length !== 16
		|| !camera?.intrinsics || camera.intrinsics.length !== 4
		|| ![...camera.worldToCamera, ...camera.intrinsics].every(Number.isFinite)) {
		throw new TypeError("camera must provide finite 4x4 extrinsics and four intrinsics.");
	}
}

function cameraWithIntrinsics(camera, intrinsics) {
	assertFiniteCamera(camera);
	return { ...camera, worldToCamera: Array.from(camera.worldToCamera), intrinsics };
}

export function opticalCameraVariants(camera, {
	zoom = CAMERA_STRESS_DEFAULTS.opticalZoom,
	shift = CAMERA_STRESS_DEFAULTS.principalShift,
} = {}) {
	assertFiniteCamera(camera);
	if (!(zoom > 1) || !Number.isFinite(zoom)) {
		throw new RangeError("zoom must be finite and greater than one.");
	}
	if (!(shift > 0 && shift < 0.5) || !Number.isFinite(shift)) {
		throw new RangeError("shift must be a finite normalized image fraction in (0, 0.5).");
	}
	const [fx, fy, cx, cy] = camera.intrinsics;
	return [
		{ name: "zoom-in", camera: cameraWithIntrinsics(camera, [fx * zoom, fy * zoom, cx, cy]) },
		{ name: "zoom-out", camera: cameraWithIntrinsics(camera, [fx / zoom, fy / zoom, cx, cy]) },
		{ name: "shift-left", camera: cameraWithIntrinsics(camera, [fx, fy, cx - shift, cy]) },
		{ name: "shift-right", camera: cameraWithIntrinsics(camera, [fx, fy, cx + shift, cy]) },
		{ name: "shift-up", camera: cameraWithIntrinsics(camera, [fx, fy, cx, cy - shift]) },
		{ name: "shift-down", camera: cameraWithIntrinsics(camera, [fx, fy, cx, cy + shift]) },
	];
}

export function physicalCameraVariants(dataset, viewIndex, {
	dollyFraction = CAMERA_STRESS_DEFAULTS.dollyFraction,
	lateralFraction = CAMERA_STRESS_DEFAULTS.lateralFraction,
	orbitRadians = CAMERA_STRESS_DEFAULTS.orbitRadians,
} = {}) {
	if (![dollyFraction, lateralFraction, orbitRadians].every((value) =>
		Number.isFinite(value) && value > 0)) {
		throw new RangeError("Physical perturbations must be finite and positive.");
	}
	const state = createOrbitCameraState(dataset, viewIndex);
	return [
		{ name: "base", camera: orbitPreviewCamera(state) },
		{ name: "dolly-in", camera: orbitPreviewCamera(dollyOrbitCamera(state, 1 - dollyFraction)) },
		{ name: "dolly-out", camera: orbitPreviewCamera(dollyOrbitCamera(state, 1 + dollyFraction)) },
		{ name: "lateral-left", camera: orbitPreviewCamera(
			translateOrbitCamera(state, { rightFraction: -lateralFraction })), },
		{ name: "lateral-right", camera: orbitPreviewCamera(
			translateOrbitCamera(state, { rightFraction: lateralFraction })), },
		{ name: "orbit-left", camera: orbitPreviewCamera({ ...state, yaw: -orbitRadians }) },
		{ name: "orbit-right", camera: orbitPreviewCamera({ ...state, yaw: orbitRadians }) },
	];
}

function sampleBilinear(rgb, width, height, x, y, output, base) {
	const x0 = Math.floor(x);
	const y0 = Math.floor(y);
	const x1 = Math.min(width - 1, x0 + 1);
	const y1 = Math.min(height - 1, y0 + 1);
	const tx = x - x0;
	const ty = y - y0;
	for (let channel = 0; channel < 3; channel += 1) {
		const top = rgb[(y0 * width + x0) * 3 + channel] * (1 - tx)
			+ rgb[(y0 * width + x1) * 3 + channel] * tx;
		const bottom = rgb[(y1 * width + x0) * 3 + channel] * (1 - tx)
			+ rgb[(y1 * width + x1) * 3 + channel] * tx;
		output[base + channel] = top * (1 - ty) + bottom * ty;
	}
}

// Optical focal/principal-point perturbations preserve the camera center and
// orientation. Their target is therefore an exact crop/resample of the real
// captured image; physical dolly/orbit tests do not have this privilege.
export function warpTargetForIntrinsics(target, sourceWidth, sourceHeight, {
	baseIntrinsics,
	testIntrinsics,
	width,
	height,
}) {
	if (target?.length !== sourceWidth * sourceHeight * 3) {
		throw new RangeError("target length does not match its source dimensions.");
	}
	const [baseFx, baseFy, baseCx, baseCy] = baseIntrinsics;
	const [testFx, testFy, testCx, testCy] = testIntrinsics;
	if (![baseFx, baseFy, baseCx, baseCy, testFx, testFy, testCx, testCy]
		.every(Number.isFinite) || Math.min(baseFx, baseFy, testFx, testFy) <= 0) {
		throw new RangeError("Intrinsics must be finite with positive focal lengths.");
	}
	const rgb = new Float32Array(width * height * 3);
	const mask = new Uint8Array(width * height);
	let validPixels = 0;
	const boundaryEpsilon = 1e-7;
	for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
		const testU = (x + 0.5) / width;
		const testV = (y + 0.5) / height;
		const sourceU = baseCx + (testU - testCx) * baseFx / testFx;
		const sourceV = baseCy + (testV - testCy) * baseFy / testFy;
		const sourceX = sourceU * sourceWidth - 0.5;
		const sourceY = sourceV * sourceHeight - 0.5;
		if (sourceX < -boundaryEpsilon || sourceX > sourceWidth - 1 + boundaryEpsilon
			|| sourceY < -boundaryEpsilon || sourceY > sourceHeight - 1 + boundaryEpsilon) continue;
		const pixel = y * width + x;
		mask[pixel] = 1;
		validPixels += 1;
		sampleBilinear(target, sourceWidth, sourceHeight,
			Math.min(sourceWidth - 1, Math.max(0, sourceX)),
			Math.min(sourceHeight - 1, Math.max(0, sourceY)), rgb, pixel * 3);
	}
	return { rgb, mask, validPixels, validFraction: validPixels / (width * height) };
}

export function maskedRgbMetrics(prediction, target, mask) {
	if (prediction?.length !== target?.length || prediction.length !== mask?.length * 3) {
		throw new RangeError("prediction, target, and mask dimensions do not agree.");
	}
	let squaredError = 0;
	let absoluteError = 0;
	let values = 0;
	for (let pixel = 0; pixel < mask.length; pixel += 1) {
		if (!mask[pixel]) continue;
		for (let channel = 0; channel < 3; channel += 1) {
			const error = prediction[pixel * 3 + channel] - target[pixel * 3 + channel];
			squaredError += error * error;
			absoluteError += Math.abs(error);
			values += 1;
		}
	}
	if (!values) throw new Error("Camera stress target has no valid pixels.");
	const mse = squaredError / values;
	return {
		mse,
		mae: absoluteError / values,
		psnr: mse === 0 ? Number.POSITIVE_INFINITY : -10 * Math.log10(mse),
		validPixels: values / 3,
	};
}

function meanAbsoluteDifference(left, right) {
	let sum = 0;
	for (let index = 0; index < left.length; index += 1) sum += Math.abs(left[index] - right[index]);
	return sum / left.length;
}

function summarizeGeometry(rendered) {
	let coverage = 0;
	let near = 0;
	let large = 0;
	let normalizedDepthSpread = 0;
	const geometryCoverage = rendered.geometryCoverage ?? rendered.coverage;
	for (let pixel = 0; pixel < geometryCoverage.length; pixel += 1) {
		const weight = geometryCoverage[pixel];
		coverage += weight;
		near += rendered.nearCoverage[pixel];
		large += rendered.largeFootprintCoverage[pixel];
		if (weight > 1e-5 && rendered.depthMean[pixel] > 1e-5) {
			normalizedDepthSpread += weight * rendered.depthStd[pixel] / rendered.depthMean[pixel];
		}
	}
	return {
		coverage: coverage / geometryCoverage.length,
		nearContribution: near / Math.max(coverage, 1e-8),
		largeFootprintContribution: large / Math.max(coverage, 1e-8),
		normalizedDepthSpread: normalizedDepthSpread / Math.max(coverage, 1e-8),
		multiLayerRayFraction: rendered.multiLayerRayFraction ?? 0,
		secondLayerMass: rendered.meanSecondLayerMass ?? 0,
	};
}

function aggregateViews(perView, role) {
	const selected = perView.filter((item) => item.role === role);
	if (!selected.length) return null;
	const opticalWorst = selected.reduce((worst, item) =>
		item.optical.worstPsnr < worst.optical.worstPsnr ? item : worst);
	const poseWorst = (key) => selected.reduce((worst, item) =>
		item.pose[key] > worst.pose[key] ? item : worst);
	return {
		viewCount: selected.length,
		opticalWorstPsnr: opticalWorst.optical.worstPsnr,
		opticalPsnrDrop: Math.max(...selected.map((item) => item.optical.psnrDrop)),
		opticalWorstCamera: opticalWorst.camera,
		opticalWorstVariant: opticalWorst.optical.worstVariant,
		poseNearContribution: poseWorst("nearContribution").pose.nearContribution,
		poseLargeFootprintContribution: poseWorst("largeFootprintContribution")
			.pose.largeFootprintContribution,
		poseNormalizedDepthSpread: poseWorst("normalizedDepthSpread").pose.normalizedDepthSpread,
		poseMultiLayerRayFraction: poseWorst("multiLayerRayFraction")
			.pose.multiLayerRayFraction,
		poseSecondLayerMass: poseWorst("secondLayerMass").pose.secondLayerMass,
		poseCoverageDrift: poseWorst("coverageDrift").pose.coverageDrift,
	};
}

export function computeCameraStressMetrics(dataset, params, {
	viewIndices,
	frameIndex = Math.floor((dataset.frameCount - 1) / 2),
	splatCount,
	modelMode = 0,
	temporalSigma = 0.30,
	...settings
} = {}) {
	if (!Array.isArray(viewIndices) || !viewIndices.length) {
		throw new TypeError("viewIndices must contain at least one calibrated camera.");
	}
	const options = { ...CAMERA_STRESS_DEFAULTS, ...settings };
	const height = Math.min(dataset.height, options.maxHeight);
	const width = Math.max(1, Math.round(height * dataset.width / dataset.height));
	const nearDepthThreshold = options.nearDepthGamma * cameraRigRadius(dataset.cameras);
	const renderOptions = {
		frameIndex, width, height, splatCount, modelMode, temporalSigma,
		pixelFilterMode: options.pixelFilterMode ?? "legacy-floor",
		opacityModel: options.opacityModel ?? "coupled",
		materialOpacityBias: options.materialOpacityBias ?? 4.59511985013459,
		collectGeometryDiagnostics: true,
		nearDepthThreshold,
		largeFootprintFraction: options.largeFootprintFraction,
	};
	const perView = viewIndices.map((viewIndex) => {
		const baseCamera = dataset.cameras[viewIndex];
		assertFiniteCamera(baseCamera);
		const target = decodeFrameRgb(dataset, viewIndex, frameIndex);
		const baseTarget = warpTargetForIntrinsics(target, dataset.width, dataset.height, {
			baseIntrinsics: baseCamera.intrinsics,
			testIntrinsics: baseCamera.intrinsics,
			width,
			height,
		});
		const baseRender = renderSnapshotFrame(dataset, params, {
			...renderOptions, viewIndex, camera: baseCamera,
		});
		const baseMetrics = maskedRgbMetrics(baseRender.rgb, baseTarget.rgb, baseTarget.mask);
		const optical = opticalCameraVariants(baseCamera, {
			zoom: options.opticalZoom,
			shift: options.principalShift,
		}).map((variant) => {
			const rendered = renderSnapshotFrame(dataset, params, {
				...renderOptions, viewIndex, camera: variant.camera,
			});
			const warped = warpTargetForIntrinsics(target, dataset.width, dataset.height, {
				baseIntrinsics: baseCamera.intrinsics,
				testIntrinsics: variant.camera.intrinsics,
				width,
				height,
			});
			const warpedBase = warpTargetForIntrinsics(baseRender.rgb, width, height, {
				baseIntrinsics: baseCamera.intrinsics,
				testIntrinsics: variant.camera.intrinsics,
				width,
				height,
			});
			const metrics = maskedRgbMetrics(rendered.rgb, warped.rgb, warped.mask);
			const croppedBaseMetrics = maskedRgbMetrics(
				warpedBase.rgb, warped.rgb, warped.mask,
			);
			return {
				name: variant.name,
				validFraction: warped.validFraction,
				...metrics,
				croppedBasePsnr: croppedBaseMetrics.psnr,
				psnrDrop: croppedBaseMetrics.psnr - metrics.psnr,
			};
		});
		const worstOptical = optical.reduce((worst, item) => item.psnr < worst.psnr ? item : worst);
		const worstOpticalDrop = optical.reduce((worst, item) =>
			item.psnrDrop > worst.psnrDrop ? item : worst);
		const physical = physicalCameraVariants(dataset, viewIndex, options).map((variant) => {
			const rendered = variant.name === "base" ? baseRender : renderSnapshotFrame(dataset, params, {
				...renderOptions, viewIndex, camera: variant.camera,
			});
			return { name: variant.name, rendered, ...summarizeGeometry(rendered) };
		});
		const baseCoverage = physical[0].rendered.geometryCoverage
			?? physical[0].rendered.coverage;
		for (const pose of physical) {
			pose.coverageDrift = meanAbsoluteDifference(
				pose.rendered.geometryCoverage ?? pose.rendered.coverage, baseCoverage,
			);
			delete pose.rendered;
		}
		return {
			viewIndex,
			camera: baseCamera.name ?? `view${viewIndex}`,
			role: baseCamera.role === "heldout" ? "heldout" : "train",
			optical: {
				basePsnr: baseMetrics.psnr,
				worstPsnr: worstOptical.psnr,
				psnrDrop: worstOpticalDrop.psnrDrop,
				worstVariant: worstOptical.name,
				worstDropVariant: worstOpticalDrop.name,
				variants: optical,
			},
			pose: {
				nearContribution: Math.max(...physical.map((item) => item.nearContribution)),
				largeFootprintContribution: Math.max(...physical.map((item) =>
					item.largeFootprintContribution)),
				normalizedDepthSpread: Math.max(...physical.map((item) => item.normalizedDepthSpread)),
				multiLayerRayFraction: Math.max(...physical.map((item) =>
					item.multiLayerRayFraction)),
				secondLayerMass: Math.max(...physical.map((item) => item.secondLayerMass)),
				coverageDrift: Math.max(...physical.map((item) => item.coverageDrift)),
				variants: physical,
			},
		};
	});
	return {
		frameIndex,
		width,
		height,
		train: aggregateViews(perView, "train"),
		heldout: aggregateViews(perView, "heldout"),
		perView,
		contract: {
			opticalTarget: "real_frame_crop_resample",
			physicalPoseTarget: null,
			opticalZoom: options.opticalZoom,
			principalShift: options.principalShift,
			dollyFraction: options.dollyFraction,
			lateralFraction: options.lateralFraction,
			orbitRadians: options.orbitRadians,
			nearDepthThreshold,
			largeFootprintFraction: options.largeFootprintFraction,
			pixelFilterMode: renderOptions.pixelFilterMode,
			opacityModel: renderOptions.opacityModel,
			materialOpacityBias: renderOptions.materialOpacityBias,
			multimodalDepth: {
				kind: "diagnostic_only_separated_weighted_depth_modes",
				externalDepthPrior: false,
				trainingLoss: false,
			},
		},
	};
}
