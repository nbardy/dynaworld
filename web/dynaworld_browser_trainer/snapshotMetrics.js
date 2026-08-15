import {
	SPLAT_FLOATS,
	projectAnisotropicGaussianCpu,
	resolveTrainViewIndices,
} from "./trainerWebGpu3d.js";
import { decodeFrameRgb, frameTime01 } from "./dataset.js";

const DEFAULT_TILE_SIZE = 16;
const DEFAULT_ALPHA_THRESHOLD = 1 / 255;
const DEFAULT_TRANSMITTANCE_THRESHOLD = 1e-4;
const DEFAULT_TEMPORAL_SIGMA = 0.30;
const SSIM_RADIUS = 5;
const SSIM_SIGMA = 1.5;
const SSIM_C1 = 0.01 ** 2;
const SSIM_C2 = 0.03 ** 2;

export const SNAPSHOT_PARAMETER_FAMILIES = Object.freeze({
	center: Object.freeze([0, 1, 2]),
	staticMix: Object.freeze([3]),
	velocity: Object.freeze([4, 5, 6]),
	timeCenter: Object.freeze([7]),
	harmonic: Object.freeze([8, 9, 10]),
	logScale: Object.freeze([12, 13, 14]),
	rotation: Object.freeze([16, 17, 18, 19]),
	color: Object.freeze([20, 21, 22]),
	opacity: Object.freeze([23]),
});

function clamp(value, minimum, maximum) {
	return Math.min(maximum, Math.max(minimum, value));
}

function sigmoid(value) {
	return 1 / (1 + Math.exp(-value));
}

function quantile(values, probability) {
	if (values.length === 0) return Number.NaN;
	const sorted = [...values].sort((left, right) => left - right);
	const position = clamp(probability, 0, 1) * (sorted.length - 1);
	const lower = Math.floor(position);
	const fraction = position - lower;
	return sorted[lower] + (sorted[Math.min(sorted.length - 1, lower + 1)] - sorted[lower]) * fraction;
}

function ceilDiv(value, divisor) {
	return Math.floor((value + divisor - 1) / divisor);
}

function assertPositiveInteger(value, label) {
	if (!Number.isSafeInteger(value) || value < 1) {
		throw new RangeError(`${label} must be a positive safe integer.`);
	}
}

function assertDataset(dataset) {
	if (!dataset || typeof dataset !== "object") throw new TypeError("dataset must be an object.");
	assertPositiveInteger(dataset.width, "dataset.width");
	assertPositiveInteger(dataset.height, "dataset.height");
	assertPositiveInteger(dataset.frameCount, "dataset.frameCount");
	if (!Array.isArray(dataset.cameras) || dataset.cameras.length < 1) {
		throw new TypeError("dataset.cameras must contain at least one camera.");
	}
}

function resolveSplatCount(params, requested) {
	if (!ArrayBuffer.isView(params) && !Array.isArray(params)) {
		throw new TypeError("params must be an array or typed array.");
	}
	const count = requested ?? params.length / SPLAT_FLOATS;
	if (!Number.isSafeInteger(count) || count < 1 || count * SPLAT_FLOATS > params.length) {
		throw new RangeError(`splatCount must address complete ${SPLAT_FLOATS}-float splats.`);
	}
	return count;
}

function temporalGate(params, base, time, temporalSigma) {
	const sigma = clamp(temporalSigma, 0.12, 0.36);
	const floor = clamp(sigma * 0.30, 0.035, 0.12);
	const delta = time - clamp(params[base + 7], 0, 1);
	const dynamic = floor + (1 - floor) * Math.exp(-0.5 * delta * delta / (sigma * sigma));
	return dynamic * (1 - clamp(params[base + 3], 0, 1)) + clamp(params[base + 3], 0, 1);
}

export function summarizeSplatParameters(params, {
	splatCount: requestedSplatCount,
	temporalSigma = DEFAULT_TEMPORAL_SIGMA,
	frameCount = 1,
	maxAspectRatio = 3,
	alphaThreshold = DEFAULT_ALPHA_THRESHOLD,
} = {}) {
	const splatCount = resolveSplatCount(params, requestedSplatCount);
	assertPositiveInteger(frameCount, "frameCount");
	if (!(maxAspectRatio > 1) || !Number.isFinite(maxAspectRatio)) {
		throw new RangeError("maxAspectRatio must be finite and greater than one.");
	}
	if (!(alphaThreshold > 0 && alphaThreshold < 1)) {
		throw new RangeError("alphaThreshold must be between zero and one.");
	}
	let activeSplats = 0;
	let rasterDeadSplats = 0;
	let dynamicSplats = 0;
	let persistentSplats = 0;
	let opacitySum = 0;
	let radiusSum = 0;
	let maxPeakAlpha = 0;
	let edgeSupportSum = 0;
	const activeStaticMix = [];
	const activeAspectRatios = [];
	const activeVelocities = [];
	const activeHarmonics = [];
	const opacities = [];
	for (let index = 0; index < splatCount; index += 1) {
		const base = index * SPLAT_FLOATS;
		const opacity = sigmoid(params[base + 23]);
		const scales = [
			Math.exp(params[base + 12]),
			Math.exp(params[base + 13]),
			Math.exp(params[base + 14]),
		];
		const staticMix = clamp(params[base + 3], 0, 1);
		const aspectRatio = Math.max(...scales) / Math.max(1e-8, Math.min(...scales));
		const radius = Math.cbrt(scales[0] * scales[1] * scales[2]);
		let peakAlpha = 0;
		for (let frame = 0; frame < frameCount; frame += 1) {
			peakAlpha = Math.max(peakAlpha,
				opacity * temporalGate(params, base,
					frameCount <= 1 ? 0 : frame / (frameCount - 1), temporalSigma));
		}
		opacitySum += opacity;
		radiusSum += radius;
		opacities.push(opacity);
		maxPeakAlpha = Math.max(maxPeakAlpha, peakAlpha);
		if (peakAlpha <= alphaThreshold) rasterDeadSplats += 1;
		if (opacity <= 0.05) continue;
		activeSplats += 1;
		activeStaticMix.push(staticMix);
		activeAspectRatios.push(aspectRatio);
		activeVelocities.push(Math.hypot(params[base + 4], params[base + 5], params[base + 6]));
		activeHarmonics.push(Math.hypot(params[base + 8], params[base + 9], params[base + 10]));
		edgeSupportSum += 0.5 * (
			temporalGate(params, base, 0, temporalSigma)
			+ temporalGate(params, base, 1, temporalSigma)
		);
		if (staticMix < 0.5) dynamicSplats += 1;
		if (staticMix >= 0.9) persistentSplats += 1;
	}
	const aspectCapFraction = activeAspectRatios.length === 0 ? Number.NaN
		: activeAspectRatios.filter((value) => value >= maxAspectRatio * 0.98).length
			/ activeAspectRatios.length;
	return {
		activeSplats,
		rasterDeadSplats,
		dynamicSplats,
		persistentSplats,
		temporalAnalyzedSplats: activeSplats,
		meanOpacity: opacitySum / splatCount,
		opacityP10: quantile(opacities, 0.1),
		opacityP50: quantile(opacities, 0.5),
		opacityP90: quantile(opacities, 0.9),
		meanRadius: radiusSum / splatCount,
		meanAspectRatio: activeAspectRatios.length
			? activeAspectRatios.reduce((sum, value) => sum + value, 0) / activeAspectRatios.length
			: Number.NaN,
		aspectP90: quantile(activeAspectRatios, 0.9),
		aspectP99: quantile(activeAspectRatios, 0.99),
		aspectCapFraction,
		meanStaticMix: activeStaticMix.length
			? activeStaticMix.reduce((sum, value) => sum + value, 0) / activeStaticMix.length
			: Number.NaN,
		staticMixP10: quantile(activeStaticMix, 0.1),
		staticMixP50: quantile(activeStaticMix, 0.5),
		staticMixP90: quantile(activeStaticMix, 0.9),
		meanEdgeTemporalSupport: activeSplats ? edgeSupportSum / activeSplats : Number.NaN,
		velocityP90: quantile(activeVelocities, 0.9),
		harmonicP90: quantile(activeHarmonics, 0.9),
		motionMaxAlpha: maxPeakAlpha,
	};
}

function worldCenter(params, base, time, modelMode) {
	const centeredTime = time * 2 - 1;
	const wave = modelMode === 0 ? Math.sin(time * Math.PI * 2) : 0;
	return [
		params[base] + params[base + 4] * centeredTime + params[base + 8] * wave,
		params[base + 1] + params[base + 5] * centeredTime + params[base + 9] * wave,
		params[base + 2] + params[base + 6] * centeredTime + params[base + 10] * wave,
	];
}

function ellipseIntersectsRect(center, conic, qLimit, rectangle) {
	const [centerX, centerY] = center;
	const [a, b, c] = conic;
	const dx0 = rectangle.minX - centerX;
	const dx1 = rectangle.maxX - centerX;
	const dy0 = rectangle.minY - centerY;
	const dy1 = rectangle.maxY - centerY;
	if (centerX >= rectangle.minX && centerX <= rectangle.maxX
		&& centerY >= rectangle.minY && centerY <= rectangle.maxY) return true;
	const quadratic = (dx, dy) => a * dx * dx + 2 * b * dx * dy + c * dy * dy;
	let minimum = Math.min(
		quadratic(dx0, dy0), quadratic(dx0, dy1),
		quadratic(dx1, dy0), quadratic(dx1, dy1),
	);
	if (c > 1e-8) {
		minimum = Math.min(minimum,
			quadratic(dx0, clamp(-(b / c) * dx0, dy0, dy1)),
			quadratic(dx1, clamp(-(b / c) * dx1, dy0, dy1)));
	}
	if (a > 1e-8) {
		minimum = Math.min(minimum,
			quadratic(clamp(-(b / a) * dy0, dx0, dx1), dy0),
			quadratic(clamp(-(b / a) * dy1, dx0, dx1), dy1));
	}
	return minimum <= qLimit;
}

function opacityAwarePixelBounds(projection, peakAlpha, width, height, alphaThreshold) {
	if (!projection.valid || !(peakAlpha > alphaThreshold)) return null;
	const qLimit = Math.min(9, 2 * Math.log(peakAlpha / alphaThreshold));
	if (!(qLimit > 0)) return null;
	const centerX = projection.center[0] * height;
	const centerY = projection.center[1] * height;
	const radiusX = Math.sqrt(Math.max(0, qLimit * projection.covariance[0])) * height;
	const radiusY = Math.sqrt(Math.max(0, qLimit * projection.covariance[2])) * height;
	const minX = Math.max(0, Math.floor(centerX - radiusX - 0.5));
	const maxX = Math.min(width - 1, Math.ceil(centerX + radiusX - 0.5));
	const minY = Math.max(0, Math.floor(centerY - radiusY - 0.5));
	const maxY = Math.min(height - 1, Math.ceil(centerY + radiusY - 0.5));
	return minX <= maxX && minY <= maxY ? { minX, maxX, minY, maxY, qLimit } : null;
}

function projectFrame(dataset, params, {
	camera,
	viewIndex,
	frameIndex,
	splatCount,
	modelMode,
	temporalSigma,
	width,
	height,
}) {
	const time = frameTime01(dataset, frameIndex);
	const renderCamera = camera ?? dataset.cameras[viewIndex];
	const aspect = width / height;
	return Array.from({ length: splatCount }, (_, index) => {
		const base = index * SPLAT_FLOATS;
		const projection = projectAnisotropicGaussianCpu({
			center: worldCenter(params, base, time, modelMode),
			logScales: [params[base + 12], params[base + 13], params[base + 14]],
			quaternion: [params[base + 16], params[base + 17], params[base + 18], params[base + 19]],
			camera: renderCamera,
			aspect,
			height,
		});
		return {
			index,
			projection,
			peakAlpha: sigmoid(params[base + 23]) * temporalGate(params, base, time, temporalSigma),
			color: [params[base + 20], params[base + 21], params[base + 22]],
		};
	});
}

function binProjectedSplats(projected, width, height, tileSize, alphaThreshold) {
	const tilesX = ceilDiv(width, tileSize);
	const tilesY = ceilDiv(height, tileSize);
	const tiles = Array.from({ length: tilesX * tilesY }, () => []);
	for (const splat of projected) {
		const bounds = opacityAwarePixelBounds(
			splat.projection, splat.peakAlpha, width, height, alphaThreshold,
		);
		if (!bounds) continue;
		// The opacity-aware screen rectangle is deliberately used here: a
		// low-opacity large covariance should not be classified as a giant
		// floater when its visible support is actually compact.
		splat.screenAreaFraction = (bounds.maxX - bounds.minX + 1)
			* (bounds.maxY - bounds.minY + 1) / (width * height);
		const minTileX = Math.floor(bounds.minX / tileSize);
		const maxTileX = Math.floor(bounds.maxX / tileSize);
		const minTileY = Math.floor(bounds.minY / tileSize);
		const maxTileY = Math.floor(bounds.maxY / tileSize);
		for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
			for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
				const minPixelX = tileX * tileSize;
				const minPixelY = tileY * tileSize;
				const maxPixelX = Math.min(width - 1, (tileX + 1) * tileSize - 1);
				const maxPixelY = Math.min(height - 1, (tileY + 1) * tileSize - 1);
				const rectangle = {
					minX: (minPixelX + 0.5) / height,
					minY: (minPixelY + 0.5) / height,
					maxX: (maxPixelX + 0.5) / height,
					maxY: (maxPixelY + 0.5) / height,
				};
				if (ellipseIntersectsRect(
					splat.projection.center, splat.projection.conic, bounds.qLimit, rectangle,
				)) {
					tiles[tileY * tilesX + tileX].push(splat);
				}
			}
		}
	}
	for (const tile of tiles) {
		tile.sort((left, right) =>
			left.projection.cameraPoint[2] - right.projection.cameraPoint[2]
			|| left.index - right.index);
	}
	return { tiles, tilesX };
}

export function renderSnapshotFrame(dataset, params, {
	viewIndex = 0,
	frameIndex = 0,
	camera = null,
	width: requestedWidth = dataset.width,
	height: requestedHeight = dataset.height,
	splatCount: requestedSplatCount,
	tileSize = DEFAULT_TILE_SIZE,
	modelMode = 0,
	temporalSigma = DEFAULT_TEMPORAL_SIGMA,
	alphaThreshold = DEFAULT_ALPHA_THRESHOLD,
	transmittanceThreshold = DEFAULT_TRANSMITTANCE_THRESHOLD,
	collectGeometryDiagnostics = false,
	nearDepthThreshold = 0,
	largeFootprintFraction = 0.25,
} = {}) {
	assertDataset(dataset);
	const splatCount = resolveSplatCount(params, requestedSplatCount);
	assertPositiveInteger(tileSize, "tileSize");
	if (!Number.isSafeInteger(viewIndex) || viewIndex < 0 || viewIndex >= dataset.cameras.length) {
		throw new RangeError("viewIndex is outside dataset.cameras.");
	}
	if (!Number.isSafeInteger(frameIndex) || frameIndex < 0 || frameIndex >= dataset.frameCount) {
		throw new RangeError("frameIndex is outside dataset.frameCount.");
	}
	assertPositiveInteger(requestedWidth, "width");
	assertPositiveInteger(requestedHeight, "height");
	if (camera != null && (!camera.worldToCamera || !camera.intrinsics)) {
		throw new TypeError("camera must provide worldToCamera and intrinsics.");
	}
	if (!(alphaThreshold > 0 && alphaThreshold < 1)
		|| !(transmittanceThreshold > 0 && transmittanceThreshold < 1)) {
		throw new RangeError("Raster thresholds must be finite values between zero and one.");
	}
	if (!Number.isFinite(nearDepthThreshold) || nearDepthThreshold < 0) {
		throw new RangeError("nearDepthThreshold must be finite and nonnegative.");
	}
	if (!(largeFootprintFraction > 0 && largeFootprintFraction <= 1)) {
		throw new RangeError("largeFootprintFraction must be in (0, 1].");
	}
	const width = requestedWidth;
	const height = requestedHeight;
	const projected = projectFrame(dataset, params, {
		camera, viewIndex, frameIndex, splatCount, modelMode, temporalSigma, width, height,
	});
	const { tiles, tilesX } = binProjectedSplats(
		projected, width, height, tileSize, alphaThreshold,
	);
	const rgb = new Float32Array(width * height * 3);
	const coverage = new Float32Array(width * height);
	const depthMean = collectGeometryDiagnostics ? new Float32Array(width * height) : null;
	const depthStd = collectGeometryDiagnostics ? new Float32Array(width * height) : null;
	const nearCoverage = collectGeometryDiagnostics ? new Float32Array(width * height) : null;
	const largeFootprintCoverage = collectGeometryDiagnostics ? new Float32Array(width * height) : null;
	let primitiveEvaluations = 0;
	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			const pixel = y * width + x;
			const tile = tiles[Math.floor(y / tileSize) * tilesX + Math.floor(x / tileSize)];
			const pointX = (x + 0.5) / height;
			const pointY = (y + 0.5) / height;
			let red = 0;
			let green = 0;
			let blue = 0;
			let transmittance = 1;
			let depthWeight = 0;
			let depthFirstMoment = 0;
			let depthSecondMoment = 0;
			let nearContribution = 0;
			let largeFootprintContribution = 0;
			for (const splat of tile) {
				primitiveEvaluations += 1;
				const dx = pointX - splat.projection.center[0];
				const dy = pointY - splat.projection.center[1];
				const [a, b, c] = splat.projection.conic;
				const qform = a * dx * dx + 2 * b * dx * dy + c * dy * dy;
				if (!Number.isFinite(qform) || qform < 0 || qform > 9) continue;
				const rawAlpha = splat.peakAlpha * Math.exp(-0.5 * qform);
				const alpha = rawAlpha >= alphaThreshold ? Math.min(0.99, rawAlpha) : 0;
				const contribution = transmittance * alpha;
				red += contribution * splat.color[0];
				green += contribution * splat.color[1];
				blue += contribution * splat.color[2];
				if (collectGeometryDiagnostics && contribution > 0) {
					const depth = splat.projection.cameraPoint[2];
					depthWeight += contribution;
					depthFirstMoment += contribution * depth;
					depthSecondMoment += contribution * depth * depth;
					if (nearDepthThreshold > 0 && depth < nearDepthThreshold) {
						nearContribution += contribution;
					}
					if (splat.screenAreaFraction >= largeFootprintFraction) {
						largeFootprintContribution += contribution;
					}
				}
				transmittance *= 1 - alpha;
				if (transmittance < transmittanceThreshold) break;
			}
			const rgbBase = pixel * 3;
			rgb[rgbBase] = red;
			rgb[rgbBase + 1] = green;
			rgb[rgbBase + 2] = blue;
			coverage[pixel] = 1 - transmittance;
			if (collectGeometryDiagnostics && depthWeight > 0) {
				const mean = depthFirstMoment / depthWeight;
				depthMean[pixel] = mean;
				depthStd[pixel] = Math.sqrt(Math.max(0,
					depthSecondMoment / depthWeight - mean * mean));
				nearCoverage[pixel] = nearContribution;
				largeFootprintCoverage[pixel] = largeFootprintContribution;
			}
		}
	}
	return {
		width,
		height,
		viewIndex,
		frameIndex,
		rgb,
		coverage,
		...(collectGeometryDiagnostics ? {
			depthMean,
			depthStd,
			nearCoverage,
			largeFootprintCoverage,
		} : {}),
		primitiveEvaluations,
		binnedReferences: tiles.reduce((sum, tile) => sum + tile.length, 0),
	};
}

function reflectIndex(index, size) {
	if (size <= 1) return 0;
	let reflected = index;
	while (reflected < 0 || reflected >= size) {
		if (reflected < 0) reflected = -reflected;
		if (reflected >= size) reflected = 2 * size - 2 - reflected;
	}
	return reflected;
}

function gaussianKernel(radius = SSIM_RADIUS, sigma = SSIM_SIGMA) {
	const kernel = new Float64Array(radius * 2 + 1);
	let sum = 0;
	for (let offset = -radius; offset <= radius; offset += 1) {
		const value = Math.exp(-(offset * offset) / (2 * sigma * sigma));
		kernel[offset + radius] = value;
		sum += value;
	}
	for (let index = 0; index < kernel.length; index += 1) kernel[index] /= sum;
	return kernel;
}

const SSIM_KERNEL = gaussianKernel();

function assertRgbImages(prediction, target, width, height) {
	assertPositiveInteger(width, "width");
	assertPositiveInteger(height, "height");
	const expected = width * height * 3;
	if (prediction?.length !== expected || target?.length !== expected) {
		throw new RangeError(`RGB images must each contain exactly ${expected} values.`);
	}
}

export function canonicalGaussianSsim(prediction, target, width, height) {
	assertRgbImages(prediction, target, width, height);
	const pixels = width * height;
	const horizontal = Array.from({ length: 5 }, () => new Float64Array(pixels * 3));
	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			for (let channel = 0; channel < 3; channel += 1) {
				let meanX = 0;
				let meanY = 0;
				let squareX = 0;
				let squareY = 0;
				let product = 0;
				for (let offset = -SSIM_RADIUS; offset <= SSIM_RADIUS; offset += 1) {
					const sampleX = reflectIndex(x + offset, width);
					const packed = (y * width + sampleX) * 3 + channel;
					const weight = SSIM_KERNEL[offset + SSIM_RADIUS];
					const left = prediction[packed];
					const right = target[packed];
					meanX += weight * left;
					meanY += weight * right;
					squareX += weight * left * left;
					squareY += weight * right * right;
					product += weight * left * right;
				}
				const packed = (y * width + x) * 3 + channel;
				horizontal[0][packed] = meanX;
				horizontal[1][packed] = meanY;
				horizontal[2][packed] = squareX;
				horizontal[3][packed] = squareY;
				horizontal[4][packed] = product;
			}
		}
	}
	let sum = 0;
	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			for (let channel = 0; channel < 3; channel += 1) {
				let meanX = 0;
				let meanY = 0;
				let squareX = 0;
				let squareY = 0;
				let product = 0;
				for (let offset = -SSIM_RADIUS; offset <= SSIM_RADIUS; offset += 1) {
					const sampleY = reflectIndex(y + offset, height);
					const packed = (sampleY * width + x) * 3 + channel;
					const weight = SSIM_KERNEL[offset + SSIM_RADIUS];
					meanX += weight * horizontal[0][packed];
					meanY += weight * horizontal[1][packed];
					squareX += weight * horizontal[2][packed];
					squareY += weight * horizontal[3][packed];
					product += weight * horizontal[4][packed];
				}
				const varianceX = Math.max(0, squareX - meanX * meanX);
				const varianceY = Math.max(0, squareY - meanY * meanY);
				const covariance = product - meanX * meanY;
				sum += ((2 * meanX * meanY + SSIM_C1) * (2 * covariance + SSIM_C2))
					/ ((meanX * meanX + meanY * meanY + SSIM_C1)
						* (varianceX + varianceY + SSIM_C2));
			}
		}
	}
	return sum / (pixels * 3);
}

export function spatialDetailMetrics(prediction, target, width, height) {
	assertRgbImages(prediction, target, width, height);
	let detailError = 0;
	let targetDetail = 0;
	let detailValues = 0;
	const accumulateEdge = (left, right) => {
		for (let channel = 0; channel < 3; channel += 1) {
			const predictedGradient = prediction[right + channel] - prediction[left + channel];
			const targetGradient = target[right + channel] - target[left + channel];
			detailError += Math.abs(predictedGradient - targetGradient);
			targetDetail += Math.abs(targetGradient);
			detailValues += 1;
		}
	};
	for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
		const pixel = (y * width + x) * 3;
		if (x + 1 < width) accumulateEdge(pixel, pixel + 3);
		if (y + 1 < height) accumulateEdge(pixel, pixel + width * 3);
	}
	let lowPassSquaredError = 0;
	let lowPassValues = 0;
	for (let y = 0; y < height; y += 2) for (let x = 0; x < width; x += 2) {
		for (let channel = 0; channel < 3; channel += 1) {
			let predictedMean = 0;
			let targetMean = 0;
			let samples = 0;
			for (let offsetY = 0; offsetY < 2 && y + offsetY < height; offsetY += 1) {
				for (let offsetX = 0; offsetX < 2 && x + offsetX < width; offsetX += 1) {
					const packed = ((y + offsetY) * width + x + offsetX) * 3 + channel;
					predictedMean += prediction[packed];
					targetMean += target[packed];
					samples += 1;
				}
			}
			const error = predictedMean / samples - targetMean / samples;
			lowPassSquaredError += error * error;
			lowPassValues += 1;
		}
	}
	const detailMae = detailError / Math.max(1, detailValues);
	const targetDetailMae = targetDetail / Math.max(1, detailValues);
	const lowPassMse = lowPassSquaredError / Math.max(1, lowPassValues);
	return {
		detailMae,
		targetDetailMae,
		detailErrorRatio: detailMae / Math.max(targetDetailMae, 1e-8),
		lowPassMse,
		lowPassPsnr: lowPassMse === 0 ? Number.POSITIVE_INFINITY : -10 * Math.log10(lowPassMse),
	};
}

export function computeFullImageMetrics(prediction, target, width, height) {
	assertRgbImages(prediction, target, width, height);
	let squaredError = 0;
	let absoluteError = 0;
	for (let index = 0; index < prediction.length; index += 1) {
		const error = prediction[index] - target[index];
		squaredError += error * error;
		absoluteError += Math.abs(error);
	}
	const mse = squaredError / prediction.length;
	return {
		mse,
		mae: absoluteError / prediction.length,
		psnr: mse === 0 ? Number.POSITIVE_INFINITY : -10 * Math.log10(mse),
		ssim: canonicalGaussianSsim(prediction, target, width, height),
		...spatialDetailMetrics(prediction, target, width, height),
	};
}

function normalizeIndices(indices, count, label) {
	if (indices === "all") return Array.from({ length: count }, (_, index) => index);
	const requested = Number.isInteger(indices) ? [indices] : indices;
	if (!Array.isArray(requested) || requested.length < 1) {
		throw new TypeError(`${label} must be "all", an index, or a nonempty index array.`);
	}
	const unique = requested.filter((value, offset) => Number.isSafeInteger(value)
		&& value >= 0 && value < count && requested.indexOf(value) === offset);
	if (unique.length !== requested.length) throw new RangeError(`${label} contains an invalid or duplicate index.`);
	return unique;
}

export function resolveSnapshotSelections(dataset, {
	views = "all",
	frames = "all",
} = {}) {
	assertDataset(dataset);
	let viewIndices;
	if (views === "train") {
		viewIndices = resolveTrainViewIndices(dataset);
	} else if (views === "heldout") {
		viewIndices = dataset.cameras
			.map((camera, index) => camera.role === "heldout" ? index : -1)
			.filter((index) => index >= 0);
		if (viewIndices.length === 0 && Number.isSafeInteger(dataset.heldoutViewIndex)
			&& dataset.heldoutViewIndex >= 0) viewIndices = [dataset.heldoutViewIndex];
		if (viewIndices.length === 0) throw new Error("Dataset has no heldout camera.");
	} else {
		viewIndices = normalizeIndices(views, dataset.cameras.length, "views");
	}
	const frameIndices = normalizeIndices(frames, dataset.frameCount, "frames");
	return viewIndices.flatMap((viewIndex) =>
		frameIndices.map((frameIndex) => ({ viewIndex, frameIndex })));
}

function targetRgb(dataset, viewIndex, frameIndex) {
	return decodeFrameRgb(dataset, viewIndex, frameIndex);
}

export function computeSnapshotMetrics(dataset, params, {
	selections = null,
	views = "all",
	frames = "all",
	...renderOptions
} = {}) {
	const resolved = selections == null
		? resolveSnapshotSelections(dataset, { views, frames })
		: selections.map(({ viewIndex, frameIndex }) => ({ viewIndex, frameIndex }));
	if (resolved.length < 1) throw new Error("At least one snapshot selection is required.");
	let squaredError = 0;
	let absoluteError = 0;
	let rgbValues = 0;
	let coverage = 0;
	let coverageValues = 0;
	let ssim = 0;
	let detailMae = 0;
	let targetDetailMae = 0;
	let lowPassMse = 0;
	let primitiveEvaluations = 0;
	let binnedReferences = 0;
	const snapshots = resolved.map(({ viewIndex, frameIndex }) => {
		const rendered = renderSnapshotFrame(dataset, params, {
			...renderOptions, viewIndex, frameIndex,
		});
		const target = targetRgb(dataset, viewIndex, frameIndex);
		const metrics = computeFullImageMetrics(rendered.rgb, target, dataset.width, dataset.height);
		squaredError += metrics.mse * rendered.rgb.length;
		absoluteError += metrics.mae * rendered.rgb.length;
		rgbValues += rendered.rgb.length;
		for (const value of rendered.coverage) coverage += value;
		coverageValues += rendered.coverage.length;
		ssim += metrics.ssim;
		detailMae += metrics.detailMae;
		targetDetailMae += metrics.targetDetailMae;
		lowPassMse += metrics.lowPassMse;
		primitiveEvaluations += rendered.primitiveEvaluations;
		binnedReferences += rendered.binnedReferences;
		return { viewIndex, frameIndex, target, ...rendered, metrics };
	});
	const mse = squaredError / rgbValues;
	const meanDetailMae = detailMae / snapshots.length;
	const meanTargetDetailMae = targetDetailMae / snapshots.length;
	const meanLowPassMse = lowPassMse / snapshots.length;
	return {
		selectionCount: snapshots.length,
		pixelCount: coverageValues,
		mse,
		mae: absoluteError / rgbValues,
		psnr: mse === 0 ? Number.POSITIVE_INFINITY : -10 * Math.log10(mse),
		ssim: ssim / snapshots.length,
		coverage: coverage / coverageValues,
		detailMae: meanDetailMae,
		targetDetailMae: meanTargetDetailMae,
		detailErrorRatio: meanDetailMae / Math.max(meanTargetDetailMae, 1e-8),
		lowPassMse: meanLowPassMse,
		lowPassPsnr: meanLowPassMse === 0
			? Number.POSITIVE_INFINITY : -10 * Math.log10(meanLowPassMse),
		primitiveEvaluations,
		binnedReferences,
		snapshots,
	};
}

export function snapshotUpdateRatios(before, after, { epsilon = 1e-12 } = {}) {
	if ((!ArrayBuffer.isView(before) && !Array.isArray(before))
		|| (!ArrayBuffer.isView(after) && !Array.isArray(after))
		|| before.length !== after.length || before.length % SPLAT_FLOATS !== 0) {
		throw new RangeError(`Snapshots must have equal lengths divisible by ${SPLAT_FLOATS}.`);
	}
	if (!(epsilon > 0) || !Number.isFinite(epsilon)) {
		throw new RangeError("epsilon must be finite and positive.");
	}
	const splatCount = before.length / SPLAT_FLOATS;
	return Object.fromEntries(Object.entries(SNAPSHOT_PARAMETER_FAMILIES).map(([name, components]) => {
		let parameterSquares = 0;
		let updateSquares = 0;
		for (let splat = 0; splat < splatCount; splat += 1) {
			const base = splat * SPLAT_FLOATS;
			for (const component of components) {
				const value = before[base + component];
				const update = after[base + component] - value;
				parameterSquares += value * value;
				updateSquares += update * update;
			}
		}
		const valueCount = splatCount * components.length;
		const parameterRms = Math.sqrt(parameterSquares / valueCount);
		const updateRms = Math.sqrt(updateSquares / valueCount);
		return [name, {
			components: [...components],
			parameterRms,
			updateRms,
			ratio: updateRms / Math.max(parameterRms, epsilon),
		}];
	}));
}
