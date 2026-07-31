import {
	assertStorageBufferFits,
	DynamicSplatWebGpu3dTrainer,
	FILTER_SIGMA_PIXELS,
	MAX_BROWSER_RENDER_SPLATS,
	MAX_SPLAT_COLOR,
	SPLAT_FLOATS,
	makeInitialSplats,
	rgbaFloatFrameBytes,
} from "./trainerWebGpu3d.js?v=20260731-compactfp16-5";
import {
	FRAME_BANK_FORMAT_RGBA8,
	resolveFrameBank,
} from "./dataset.js?v=20260731-compactfp16-5";
import {
	BROWSER_ADAM_BETA1,
	BROWSER_ADAM_BETA2,
	BROWSER_ADAM_EPSILON,
	DENSITY_STAT_DECAY,
	browserLearningRates,
} from "./trainingSchedule.js?v=20260731-compactfp16-5";

const SPLAT_BYTES = SPLAT_FLOATS * 4;
export const DEFAULT_TILE_SIZE = 16;
export const TILED_TILE_SIZES = Object.freeze([8, 16]);
const MIN_CHECKPOINT_STRIDE = 8;
export const DEFAULT_CHECKPOINT_STRIDE = 16;
const MONOLITHIC_PROJECTION_BYTES = 12 * 16;
const RASTER_PROJECTION_BYTES = 2 * 16;
const PROJECTION_VJP_BYTES = 5 * 16;
const PACKED_PROJECTION_VJP_BYTES = 3 * 16;
const TILED_COUNTER_BYTES = 10 * Uint32Array.BYTES_PER_ELEMENT;
const TILED_METRICS_BYTES = 5 * 16;
const SSIM_STATS_BYTES = 5 * 16;
const DENSITY_START_STEP = 600;
const DENSITY_INTERVAL = 100;
const DENSITY_DISPATCHES = 4;
const DENSITY_SPLITS_PER_DISPATCH = 4;
const TILED_CONFIG_BYTES = 176;
const DIRECT_GRADIENT_FLOATS = SPLAT_FLOATS;
const PROJECTED_GRADIENT_FLOATS = 12;
const TILED_GPU_PHASES = Object.freeze([
	"targetDecode", "clear", "project", "sort", "finalize", "forward",
	"ssimStats", "ssimGradient", "metrics", "backward", "update",
]);
export const TILED_BACKWARD_MODES = Object.freeze({
	DIRECT_3D: "direct-3d",
	STAGED_PROJECT_3D: "staged-project3d",
});
export const TILED_BACKWARD_GRANULARITIES = Object.freeze({
	PAIR: "pair",
	CHECKPOINT_BLOCK: "checkpoint-block",
});
export const TILED_CHECKPOINT_ORDERS = Object.freeze({
	PIXEL_MAJOR: "pixel-major",
	BLOCK_MAJOR: "block-major",
});
export const TILED_PROJECTION_LAYOUTS = Object.freeze({
	MONOLITHIC: "monolithic",
	SPLIT_COMPACT: "split-compact",
});
export const TILED_PROJECTION_VJP_PRECISIONS = Object.freeze({
	F32: "f32",
	PACKED_F16: "packed-f16",
});
export const TILED_SSIM_LAYOUTS = Object.freeze({
	NAIVE_2D: "naive-2d",
	SEPARABLE: "separable",
});
export const DEFAULT_MAX_TILE_CAPACITY = 4096;
export const DEFAULT_BROWSER_GROWTH_CAPACITY = 8192;
export const DEFAULT_CHECKPOINT_PRECISION = "packed-f16";
export const DEFAULT_STATIC_WARMUP_STEPS = 2048;
export const MAX_WORKGROUPS_PER_DIMENSION = 65535;
export const SCALE_LR_FROM_COLOR = 0.30;
export const ROTATION_LR_FROM_MOTION = 1.25;
// This is an optimizer/performance trust region, not a roundness prior. A 6:1
// standard-deviation ratio still allows 36:1 covariance conditioning; larger
// needles increase tile pairs and were worse on heldout in the matched 12:1 run.
export const MAX_SCALE_ASPECT_RATIO = 6;
export const TILED_SPLAT_ID_BITS = Math.ceil(Math.log2(MAX_BROWSER_RENDER_SPLATS));
export const TILED_SPLAT_ID_MASK = (2 ** TILED_SPLAT_ID_BITS) - 1;
export const TILED_DEPTH_KEY_MASK = (~TILED_SPLAT_ID_MASK) >>> 0;

export function resolveTiledBackwardMode(value = TILED_BACKWARD_MODES.DIRECT_3D) {
	if (!Object.values(TILED_BACKWARD_MODES).includes(value)) {
		throw new RangeError(`backwardMode must be one of: ${Object.values(TILED_BACKWARD_MODES).join(", ")}.`);
	}
	return value;
}

export function resolveTiledBackwardGranularity(value = TILED_BACKWARD_GRANULARITIES.PAIR) {
	if (!Object.values(TILED_BACKWARD_GRANULARITIES).includes(value)) {
		throw new RangeError(`backwardGranularity must be one of: `
			+ `${Object.values(TILED_BACKWARD_GRANULARITIES).join(", ")}.`);
	}
	return value;
}

export function resolveTiledTileSize(value = DEFAULT_TILE_SIZE) {
	if (!TILED_TILE_SIZES.includes(value)) {
		throw new RangeError(`tileSize must be one of: ${TILED_TILE_SIZES.join(", ")}.`);
	}
	return value;
}

export function resolveTiledCheckpointOrder(value = TILED_CHECKPOINT_ORDERS.PIXEL_MAJOR) {
	if (!Object.values(TILED_CHECKPOINT_ORDERS).includes(value)) {
		throw new RangeError(`checkpointOrder must be one of: `
			+ `${Object.values(TILED_CHECKPOINT_ORDERS).join(", ")}.`);
	}
	return value;
}

export function resolveTiledProjectionLayout(
	value = TILED_PROJECTION_LAYOUTS.MONOLITHIC,
) {
	if (!Object.values(TILED_PROJECTION_LAYOUTS).includes(value)) {
		throw new RangeError(`projectionLayout must be one of: `
			+ `${Object.values(TILED_PROJECTION_LAYOUTS).join(", ")}.`);
	}
	return value;
}

export function resolveTiledProjectionVjpPrecision(
	value = TILED_PROJECTION_VJP_PRECISIONS.F32,
) {
	if (!Object.values(TILED_PROJECTION_VJP_PRECISIONS).includes(value)) {
		throw new RangeError(`projectionVjpPrecision must be one of: `
			+ `${Object.values(TILED_PROJECTION_VJP_PRECISIONS).join(", ")}.`);
	}
	return value;
}

export function resolveTiledSsimLayout(value = TILED_SSIM_LAYOUTS.NAIVE_2D) {
	if (!Object.values(TILED_SSIM_LAYOUTS).includes(value)) {
		throw new RangeError(`ssimLayout must be one of: `
			+ `${Object.values(TILED_SSIM_LAYOUTS).join(", ")}.`);
	}
	return value;
}

function hashU32(value) {
	let result = value >>> 0;
	result = Math.imul(result ^ (result >>> 16), 0x7feb352d);
	result = Math.imul(result ^ (result >>> 15), 0x846ca68b);
	return (result ^ (result >>> 16)) >>> 0;
}

export function packedTrainingBackgroundForStep(step, enabled = true) {
	if (!Number.isSafeInteger(step) || step < 0) {
		throw new RangeError("step must be a non-negative safe integer.");
	}
	if (!enabled) return 0;
	const seed = step >>> 0;
	const channel = (salt) => hashU32(seed ^ salt) & 0x3ff;
	return (0x80000000
		| channel(0x68bc21eb)
		| (channel(0x02e5be93) << 10)
		| (channel(0x967a889b) << 20)) >>> 0;
}

export function trainingBackgroundForStep(step) {
	const packed = packedTrainingBackgroundForStep(step);
	return [
		(packed & 0x3ff) / 1023,
		((packed >>> 10) & 0x3ff) / 1023,
		((packed >>> 20) & 0x3ff) / 1023,
	];
}

export function resolveSsimRadius(value = 5) {
	if (!Number.isSafeInteger(value) || value < 0 || value > 15) {
		throw new RangeError("ssimRadius must be an integer from 0 through 15.");
	}
	return value;
}

export function resolveCheckpointStride(value = DEFAULT_CHECKPOINT_STRIDE) {
	if (!Number.isSafeInteger(value) || value < MIN_CHECKPOINT_STRIDE
		|| value > DEFAULT_MAX_TILE_CAPACITY || (value & (value - 1)) !== 0) {
		throw new RangeError(`checkpointStride must be a power of two from `
			+ `${MIN_CHECKPOINT_STRIDE} through ${DEFAULT_MAX_TILE_CAPACITY}.`);
	}
	return value;
}

function ceilDiv(value, divisor) {
	return Math.floor((value + divisor - 1) / divisor);
}

function nextPowerOfTwo(value) {
	return 2 ** Math.ceil(Math.log2(Math.max(1, value)));
}

export function resolveTiledCapacity(initialSplats, requestedCapacity = null) {
	if (!Number.isSafeInteger(initialSplats) || initialSplats < 8
		|| initialSplats > MAX_BROWSER_RENDER_SPLATS) {
		throw new RangeError(`initialSplats must be an integer from 8 through ${MAX_BROWSER_RENDER_SPLATS}.`);
	}
	const capacity = requestedCapacity == null
		? Math.min(DEFAULT_BROWSER_GROWTH_CAPACITY, initialSplats * 3)
		: Number(requestedCapacity);
	return Math.min(MAX_BROWSER_RENDER_SPLATS, Math.max(initialSplats, Math.floor(capacity)));
}

function validateDensitySchedule(initialSplats, capacity, step) {
	if (!Number.isSafeInteger(initialSplats) || initialSplats < 1
		|| !Number.isSafeInteger(capacity) || capacity < initialSplats
		|| !Number.isSafeInteger(step) || step < 0) {
		throw new RangeError(
			"Density schedule inputs require non-negative integer steps and capacity >= initialSplats >= 1.",
		);
	}
}

export function activeSplatCountForStep(initialSplats, capacity, completedSteps) {
	validateDensitySchedule(initialSplats, capacity, completedSteps);
	if (completedSteps < DENSITY_START_STEP || initialSplats === capacity) {
		return initialSplats;
	}
	const completedEvents = Math.floor(
		(completedSteps - DENSITY_START_STEP) / DENSITY_INTERVAL,
	) + 1;
	return Math.min(
		capacity,
		initialSplats
			+ completedEvents * DENSITY_DISPATCHES * DENSITY_SPLITS_PER_DISPATCH,
	);
}

export function activePrefixDispatchSizes(
	activeSplats,
	capacity,
	tileCount,
	gradientFloats,
) {
	if (!Number.isSafeInteger(activeSplats) || activeSplats < 1
		|| !Number.isSafeInteger(capacity) || capacity < activeSplats
		|| !Number.isSafeInteger(tileCount) || tileCount < 1
		|| !Number.isSafeInteger(gradientFloats) || gradientFloats < 1) {
		throw new RangeError(
			"Active-prefix dispatch sizes require positive integers and capacity >= activeSplats.",
		);
	}
	const gradientClearSlots = activeSplats * gradientFloats;
	return {
		activeUpdateSlots: activeSplats,
		capacitySlots: capacity,
		dormantUpdateSlots: capacity - activeSplats,
		gradientClearSlots,
		clearWorkgroups: ceilDiv(Math.max(tileCount, gradientClearSlots), 64),
		updateWorkgroups: ceilDiv(activeSplats, 64),
	};
}

export function telemetryAliasPeriod(pairCount, metricInterval) {
	if (!Number.isSafeInteger(pairCount) || pairCount < 1
		|| !Number.isSafeInteger(metricInterval) || metricInterval < 1) {
		throw new RangeError("pairCount and metricInterval must be positive safe integers.");
	}
	let left = pairCount;
	let right = metricInterval;
	while (right !== 0) [left, right] = [right, left % right];
	const sampledPhases = pairCount / left;
	return {
		sampledPhases,
		repeatSamples: sampledPhases,
		repeatSteps: sampledPhases * metricInterval,
	};
}

export function summarizeCycleMetrics(records, objectiveStep, cycleLength, phaseStartStep = 0) {
	if (!(records instanceof Float32Array) || records.length < cycleLength * 4
		|| !Number.isSafeInteger(objectiveStep) || objectiveStep < 0
		|| !Number.isSafeInteger(cycleLength) || cycleLength < 1
		|| !Number.isSafeInteger(phaseStartStep) || phaseStartStep < 0) {
		throw new RangeError("Cycle metrics need packed vec4 records and non-negative integer steps.");
	}
	let loss = 0;
	let l1 = 0;
	let dssim = 0;
	let samples = 0;
	let oldestStep = objectiveStep;
	for (let index = 0; index < cycleLength; index += 1) {
		const base = index * 4;
		const stamp = Math.round(records[base + 3]);
		if (stamp < 1) continue;
		const sampleStep = stamp - 1;
		const age = objectiveStep - sampleStep;
		if (sampleStep < phaseStartStep || age < 0 || age >= cycleLength) continue;
		if (![records[base], records[base + 1], records[base + 2]].every(Number.isFinite)) continue;
		loss += records[base];
		l1 += records[base + 1];
		dssim += records[base + 2];
		samples += 1;
		oldestStep = Math.min(oldestStep, sampleStep);
	}
	return samples === 0 ? null : {
		loss: loss / samples,
		l1: l1 / samples,
		dssim: dssim / samples,
		samples,
		complete: samples === cycleLength,
		oldestStep,
		newestStep: objectiveStep,
	};
}

export function resolveTileCapacity(splatCount, requestedCapacity = null) {
	if (!Number.isSafeInteger(splatCount) || splatCount < 8
		|| splatCount > MAX_BROWSER_RENDER_SPLATS) {
		throw new RangeError(`splatCount must be an integer from 8 through ${MAX_BROWSER_RENDER_SPLATS}.`);
	}
	// Tile occupancy is normally much smaller than the global model. Keeping a
	// bounded tile-local sort avoids a non-portable 32 KiB+ workgroup allocation
	// at 8K; counters[1] reports any scene that violates this measured bound.
	const requested = requestedCapacity == null
		? Math.max(
			DEFAULT_CHECKPOINT_STRIDE,
			Math.min(nextPowerOfTwo(splatCount), DEFAULT_MAX_TILE_CAPACITY),
		)
		: Math.floor(Number(requestedCapacity));
	if (!Number.isSafeInteger(requested) || requested < 8 || requested > DEFAULT_MAX_TILE_CAPACITY) {
		throw new RangeError(`tileCapacity must be an integer from 8 through ${DEFAULT_MAX_TILE_CAPACITY}.`);
	}
	return nextPowerOfTwo(requested);
}

export function packDepthSplatKey(depthBits, splatId) {
	if (!Number.isSafeInteger(depthBits) || depthBits < 0 || depthBits > 0xffffffff
		|| !Number.isSafeInteger(splatId) || splatId < 0 || splatId >= MAX_BROWSER_RENDER_SPLATS) {
		throw new RangeError("Depth bits must be u32 and splatId must fit the browser splat ID field.");
	}
	return ((depthBits & TILED_DEPTH_KEY_MASK) | splatId) >>> 0;
}

export function unpackDepthSplatId(key) {
	return Number(key) & TILED_SPLAT_ID_MASK;
}

export function resolveCheckpointPrecision(value = DEFAULT_CHECKPOINT_PRECISION) {
	if (value !== "f32" && value !== "packed-f16") {
		throw new RangeError('checkpointPrecision must be "f32" or "packed-f16".');
	}
	return value;
}

export function resolveStaticWarmupSteps(value = 0) {
	const steps = Number(value);
	if (!Number.isSafeInteger(steps) || steps < 0 || steps > 1_000_000) {
		throw new RangeError("staticWarmupSteps must be an integer from 0 through 1000000.");
	}
	return steps;
}

export function resolvePairDispatch(pairCount) {
	if (!Number.isSafeInteger(pairCount) || pairCount < 0) {
		throw new RangeError("pairCount must be a non-negative safe integer.");
	}
	return {
		x: Math.min(pairCount, MAX_WORKGROUPS_PER_DIMENSION),
		y: Math.max(1, ceilDiv(pairCount, MAX_WORKGROUPS_PER_DIMENSION)),
		z: 1,
	};
}

export function resolveCheckpointLayout(
	pixelCount,
	tileCapacity,
	storageLimit,
	bytesPerCheckpoint = 16,
	minimumStride = DEFAULT_CHECKPOINT_STRIDE,
) {
	if (!Number.isSafeInteger(pixelCount) || pixelCount < 1
		|| !Number.isSafeInteger(tileCapacity) || tileCapacity < minimumStride
		|| !Number.isSafeInteger(storageLimit) || storageLimit < 16
		|| (bytesPerCheckpoint !== 8 && bytesPerCheckpoint !== 16)
		|| resolveCheckpointStride(minimumStride) !== minimumStride) {
		throw new RangeError("Checkpoint layout inputs must be positive safe integers.");
	}
	for (let stride = minimumStride; stride <= tileCapacity; stride *= 2) {
		const blocksPerTile = ceilDiv(tileCapacity, stride);
		const byteLength = pixelCount * blocksPerTile * bytesPerCheckpoint;
		if (Number.isSafeInteger(byteLength) && byteLength <= storageLimit) {
			return { stride, blocksPerTile, byteLength };
		}
	}
	throw new RangeError(`Even one checkpoint record per pixel exceeds the ${storageLimit}-byte storage limit.`);
}

export function densityDispatchesForStep(initialSplats, capacity, step) {
	validateDensitySchedule(initialSplats, capacity, step);
	if (step >= DENSITY_START_STEP
		&& (step - DENSITY_START_STEP) % DENSITY_INTERVAL === 0) {
		const before = activeSplatCountForStep(initialSplats, capacity, Math.max(0, step - 1));
		const after = activeSplatCountForStep(initialSplats, capacity, step);
		return ceilDiv(after - before, DENSITY_SPLITS_PER_DISPATCH);
	}
	// Once reserved capacity is full, keep the SfM scaffold stable. The former
	// perpetual recycling repeatedly erased useful seeds without a residual-
	// guided replacement test.
	return 0;
}

export function fullFramePairForStep(trainViewIndices, frameCount, step) {
	if (!Array.isArray(trainViewIndices) || trainViewIndices.length < 1) {
		throw new Error("At least one train view is required.");
	}
	const safeStep = Math.max(0, Math.floor(step));
	const pairsPerCycle = trainViewIndices.length * Math.max(1, frameCount);
	let stride = Math.max(1, Math.floor(pairsPerCycle * 0.618)) | 1;
	const gcd = (left, right) => {
		let a = left; let b = right;
		while (b !== 0) [a, b] = [b, a % b];
		return a;
	};
	while (gcd(stride, pairsPerCycle) !== 1) stride += 2;
	const pairIndex = (safeStep * stride) % pairsPerCycle;
	const viewSlot = pairIndex % trainViewIndices.length;
	const frameIndex = Math.floor(pairIndex / trainViewIndices.length);
	return { viewSlot, viewIndex: trainViewIndices[viewSlot], frameIndex };
}

export function trainingPairForStep(trainViewIndices, frameCount, step, staticWarmupSteps = 0) {
	const safeStep = Math.max(0, Math.floor(step));
	const warmupSteps = resolveStaticWarmupSteps(staticWarmupSteps);
	if (safeStep < warmupSteps) {
		const selected = fullFramePairForStep(trainViewIndices, 1, safeStep);
		return {
			...selected,
			frameIndex: Math.floor((Math.max(1, frameCount) - 1) / 2),
			staticWarmup: true,
		};
	}
	return {
		...fullFramePairForStep(trainViewIndices, frameCount, safeStep - warmupSteps),
		staticWarmup: false,
	};
}

export function opacityAwarePixelBounds(projection, peakAlpha, width, height, alphaThreshold = 1 / 255) {
	if (!projection?.valid || !(peakAlpha > alphaThreshold)) return null;
	const qLimit = Math.min(9, 2 * Math.log(peakAlpha / alphaThreshold));
	if (!(qLimit > 0)) return null;
	const centerX = projection.center[0] * height;
	const centerY = projection.center[1] * height;
	const radiusX = Math.sqrt(Math.max(0, qLimit * projection.covariance[0])) * height;
	const radiusY = Math.sqrt(Math.max(0, qLimit * projection.covariance[2])) * height;
	const minX = Math.max(0, Math.floor(centerX - radiusX));
	const maxX = Math.min(width - 1, Math.ceil(centerX + radiusX));
	const minY = Math.max(0, Math.floor(centerY - radiusY));
	const maxY = Math.min(height - 1, Math.ceil(centerY + radiusY));
	return minX <= maxX && minY <= maxY ? { minX, maxX, minY, maxY, qLimit } : null;
}

export function ellipseIntersectsRect(center, conic, qLimit, rectangle) {
	const [mx, my] = center; const [a, b, c] = conic;
	const { minX, minY, maxX, maxY } = rectangle;
	const dx0 = minX - mx; const dx1 = maxX - mx;
	const dy0 = minY - my; const dy1 = maxY - my;
	if (mx >= minX && mx <= maxX && my >= minY && my <= maxY) return true;
	const quadratic = (dx, dy) => a * dx * dx + 2 * b * dx * dy + c * dy * dy;
	let minimum = Math.min(
		quadratic(dx0, dy0), quadratic(dx0, dy1),
		quadratic(dx1, dy0), quadratic(dx1, dy1),
	);
	if (c > 1e-8) {
		minimum = Math.min(minimum,
			quadratic(dx0, Math.min(dy1, Math.max(dy0, -(b / c) * dx0))),
			quadratic(dx1, Math.min(dy1, Math.max(dy0, -(b / c) * dx1))));
	}
	if (a > 1e-8) {
		minimum = Math.min(minimum,
			quadratic(Math.min(dx1, Math.max(dx0, -(b / a) * dy0)), dy0),
			quadratic(Math.min(dx1, Math.max(dx0, -(b / a) * dy1)), dy1));
	}
	return minimum <= qLimit;
}

function reflectIndex(index, size) {
	if (size <= 1) return 0;
	let value = index;
	const maximum = size - 1;
	while (value < 0 || value > maximum) {
		if (value < 0) value = -value;
		if (value > maximum) value = 2 * maximum - value;
	}
	return value;
}

const SSIM_GAUSSIAN_11 = Object.freeze([
	0.0010283800844791101,
	0.0075987581352391850,
	0.036000772128430829,
	0.10936068950970002,
	0.21300553771125369,
	0.26601172486179436,
	0.21300553771125369,
	0.10936068950970002,
	0.036000772128430829,
	0.0075987581352391850,
	0.0010283800844791101,
]);

function ssimKernel1d(radius) {
	if (radius === 5) return SSIM_GAUSSIAN_11;
	const side = radius * 2 + 1;
	return Array.from({ length: side }, () => 1 / side);
}

function reflectedKernelWeight(center, pixel, size, radius, kernel) {
	const weightAt = (offset) => Math.abs(offset) <= radius ? kernel[offset + radius] : 0;
	let weight = weightAt(pixel - center);
	if (pixel > 0) weight += weightAt(-pixel - center);
	const maximum = size - 1;
	if (pixel < maximum) weight += weightAt(2 * maximum - pixel - center);
	return weight;
}

export function windowedL1DssimCpu(prediction, target, width, height, {
	l1Weight = 0.8,
	dssimWeight = 0.2,
	radius = 5,
	c1 = 0.0001,
	c2 = 0.0009,
	computeGradient = true,
	pixelWeights = null,
} = {}) {
	if (prediction.length !== target.length || prediction.length !== width * height * 3) {
		throw new RangeError("prediction and target must be packed RGB images.");
	}
	const pixels = width * height;
	if (pixelWeights && pixelWeights.length !== pixels) {
		throw new RangeError("pixelWeights must contain one value per image pixel.");
	}
	const kernel = ssimKernel1d(radius);
	const stats = Array.from({ length: pixels }, () => null);
	let l1 = 0;
	let ssimSum = 0;
	let weightSum = 0;
	for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
		const sums = Array.from({ length: 5 }, () => [0, 0, 0]);
		for (let oy = -radius; oy <= radius; oy += 1) {
			const sy = reflectIndex(y + oy, height);
			for (let ox = -radius; ox <= radius; ox += 1) {
				const sx = reflectIndex(x + ox, width);
				const base = (sy * width + sx) * 3;
				const weight = kernel[oy + radius] * kernel[ox + radius];
				for (let channel = 0; channel < 3; channel += 1) {
					const px = prediction[base + channel];
					const py = target[base + channel];
					sums[0][channel] += weight * px;
					sums[1][channel] += weight * py;
					sums[2][channel] += weight * px * px;
					sums[3][channel] += weight * py * py;
					sums[4][channel] += weight * px * py;
				}
			}
		}
		const muX = sums[0];
		const muY = sums[1];
		const varX = sums[2].map((value, channel) => value - muX[channel] ** 2);
		const varY = sums[3].map((value, channel) => value - muY[channel] ** 2);
		const cov = sums[4].map((value, channel) => value - muX[channel] * muY[channel]);
		stats[y * width + x] = { muX, muY, varX, varY, cov };
		const centerWeight = pixelWeights?.[y * width + x] ?? 1;
		weightSum += centerWeight;
		for (let channel = 0; channel < 3; channel += 1) {
			const numerator = (2 * muX[channel] * muY[channel] + c1) * (2 * cov[channel] + c2);
			const denominator = (muX[channel] ** 2 + muY[channel] ** 2 + c1)
				* (varX[channel] + varY[channel] + c2);
			ssimSum += centerWeight * numerator / Math.max(denominator, 1e-12);
			const base = (y * width + x) * 3 + channel;
			l1 += centerWeight * Math.abs(prediction[base] - target[base]);
		}
	}
	const objectiveDenominator = Math.max(1e-12, weightSum * 3);
	l1 /= objectiveDenominator;
	const dssim = 1 - ssimSum / objectiveDenominator;
	const loss = l1Weight * l1 + dssimWeight * dssim;
	if (!computeGradient) return { loss, l1, dssim, gradient: null };
	const gradient = new Float32Array(prediction.length);
	for (let py = 0; py < height; py += 1) for (let px = 0; px < width; px += 1) {
		const pixel = py * width + px;
		for (let channel = 0; channel < 3; channel += 1) {
			const packed = pixel * 3 + channel;
			const error = prediction[packed] - target[packed];
			let dssimGradient = 0;
			for (let cy = Math.max(0, py - radius); cy <= Math.min(height - 1, py + radius); cy += 1) {
				const yWeight = reflectedKernelWeight(cy, py, height, radius, kernel);
				for (let cx = Math.max(0, px - radius); cx <= Math.min(width - 1, px + radius); cx += 1) {
					const centerWeight = pixelWeights?.[cy * width + cx] ?? 1;
					const weight = centerWeight * yWeight
						* reflectedKernelWeight(cx, px, width, radius, kernel);
					if (weight === 0) continue;
					const center = stats[cy * width + cx];
					const mx = center.muX[channel]; const my = center.muY[channel];
					const vx = center.varX[channel]; const vy = center.varY[channel];
					const covariance = center.cov[channel];
					const a = 2 * mx * my + c1; const b = 2 * covariance + c2;
					const c = mx * mx + my * my + c1; const d = vx + vy + c2;
					const da = 2 * my * weight;
					const db = 2 * weight * (target[packed] - my);
					const dc = 2 * mx * weight;
					const dd = 2 * weight * (prediction[packed] - mx);
						const denominator = Math.max(c * d, 1e-12);
						dssimGradient -= (((da * b + a * db) * denominator)
							- (a * b) * (dc * d + c * dd)) / (denominator ** 2 * objectiveDenominator);
				}
			}
			const ownWeight = pixelWeights?.[pixel] ?? 1;
			gradient[packed] = l1Weight * ownWeight * Math.sign(error) / objectiveDenominator
				+ dssimWeight * dssimGradient;
		}
	}
	return { loss, l1, dssim, gradient };
}

function writeTiledConfig(buffer, values) {
	const view = new DataView(buffer);
	const u32 = (offset, value) => view.setUint32(offset, value, true);
	const f32 = (offset, value) => view.setFloat32(offset, value, true);
	u32(0, values.width); u32(4, values.height); u32(8, values.splatCount); u32(12, values.tileSize);
	u32(16, values.tilesX); u32(20, values.tilesY); u32(24, values.tileCapacity);
	u32(28, values.blocksPerTile);
	u32(32, values.viewIndex); u32(36, values.frameIndex); u32(40, values.step);
	u32(44, values.modelMode); u32(48, values.targetOffset); u32(52, values.pixelCount);
	u32(56, values.pairCapacity); u32(60, values.checkpointStride);
	f32(64, values.targetAspect); f32(68, values.temporalSigma); f32(72, values.alphaThreshold);
	f32(76, values.transmittanceThreshold); f32(80, values.lrPosition); f32(84, values.lrColor);
	f32(88, values.lrOpacity); f32(92, values.lrMotion); f32(96, values.geometryScale);
	f32(100, values.l1Weight); f32(104, values.dssimWeight); f32(108, values.statDecay);
	f32(112, BROWSER_ADAM_BETA1); f32(116, BROWSER_ADAM_BETA2);
	f32(120, BROWSER_ADAM_EPSILON);
	u32(124, packedTrainingBackgroundForStep(values.step, values.randomBackground));
	u32(128, values.ssimRadius); u32(132, values.frameCount);
	u32(136, values.staticWarmup ? 1 : 0); u32(140, values.motionWeighting ? 1 : 0);
	f32(144, 0.0001); f32(148, 0.0009);
	f32(152, 0.03 * values.geometryScale); f32(156, values.geometryScale);
	u32(160, values.activeSplatCount);
	u32(164, values.targetPacked ? 1 : 0); u32(168, 0); u32(172, 0);
}

const CONFIG_WGSL = `
	struct TiledConfig {
		width:u32, height:u32, splatCount:u32, tileSize:u32,
		tilesX:u32, tilesY:u32, tileCapacity:u32, blocksPerTile:u32,
		viewIndex:u32, frameIndex:u32, step:u32, modelMode:u32,
		targetOffset:u32, pixelCount:u32, pairCapacity:u32, checkpointStride:u32,
		targetAspect:f32, temporalSigma:f32, alphaThreshold:f32, transmittanceThreshold:f32,
		lrPosition:f32, lrColor:f32, lrOpacity:f32, lrMotion:f32,
		geometryScale:f32, l1Weight:f32, dssimWeight:f32, statDecay:f32,
		beta1:f32, beta2:f32, adamEpsilon:f32, trainingBackgroundPacked:u32,
		ssimRadius:u32, frameCount:u32, staticWarmup:u32, motionWeighting:u32,
		c1:f32, c2:f32, minScale:f32, maxScale:f32,
		activeSplatCount:u32, configPad0:u32, configPad1:u32, configPad2:u32,
	};
	struct Splat {
		centerStatic:vec4<f32>, velocityTime:vec4<f32>, harmonicPad:vec4<f32>,
		logScalePad:vec4<f32>, rotation:vec4<f32>, colorOpacity:vec4<f32>,
	};
	struct Projection {
		screenConic0:vec4<f32>, conicDepthAlpha:vec4<f32>, cameraPointValid:vec4<f32>,
		jacobian0:vec4<f32>, jacobian1:vec4<f32>,
		basis0:vec4<f32>, basis1:vec4<f32>, basis2:vec4<f32>,
		camera0:vec4<f32>, camera1:vec4<f32>, camera2:vec4<f32>, variancesPad:vec4<f32>,
	};
	struct RasterProjection {
		screenConic0:vec4<f32>, conicDepthAlpha:vec4<f32>,
	};
	struct ProjectionVjp {
		cameraPointValid:vec4<f32>, jacobian0:vec4<f32>, jacobian1:vec4<f32>,
		basis0:vec4<f32>, basis1:vec4<f32>, basis2:vec4<f32>,
		camera0:vec4<f32>, camera1:vec4<f32>, camera2:vec4<f32>, variancesPad:vec4<f32>,
	};
	struct CompactProjectionVjp {
		cameraPointValid:vec4<f32>, jacobianSparse:vec4<f32>,
		basisVariance0:vec4<f32>, basisVariance1:vec4<f32>, basisVariance2:vec4<f32>,
	};
	struct PackedCompactProjectionVjp {
		cameraPointValid:vec4<f32>, packed0:vec4<u32>, packed1:vec4<u32>,
	};
	struct Camera {
		row0:vec4<f32>, row1:vec4<f32>, row2:vec4<f32>, row3:vec4<f32>, intrinsics:vec4<f32>,
	};
	fn sigmoid(x:f32)->f32 { return 1.0/(1.0+exp(-x)); }
	fn safe_quaternion(raw:vec4<f32>)->vec4<f32> {
		let n2=dot(raw,raw); let normalized=raw*inverseSqrt(max(n2,1e-16));
		return select(vec4<f32>(0.0,0.0,0.0,1.0),normalized,n2>1e-16);
	}
	fn quaternion_matrix(raw:vec4<f32>)->mat3x3<f32> {
		let q=safe_quaternion(raw); let x=q.x; let y=q.y; let z=q.z; let w=q.w;
		return mat3x3<f32>(
			vec3<f32>(1.0-2.0*(y*y+z*z),2.0*(x*y+z*w),2.0*(x*z-y*w)),
			vec3<f32>(2.0*(x*y-z*w),1.0-2.0*(x*x+z*z),2.0*(y*z+x*w)),
			vec3<f32>(2.0*(x*z+y*w),2.0*(y*z-x*w),1.0-2.0*(x*x+y*y)));
	}
	fn outer3(a:vec3<f32>,b:vec3<f32>)->mat3x3<f32> {
		return mat3x3<f32>(a*b.x,a*b.y,a*b.z);
	}
	fn world_center(p:Splat,t:f32,modelMode:u32)->vec3<f32> {
		let tc=t*2.0-1.0; var center=p.centerStatic.xyz+p.velocityTime.xyz*tc;
		if(modelMode==0u){center+=p.harmonicPad.xyz*sin(t*6.28318530718);}
		return center;
	}
	fn temporal_gate(p:Splat,t:f32,sigmaValue:f32)->f32 {
		let sigma=clamp(sigmaValue,0.12,0.36);
		let floorValue=clamp(sigma*0.30,0.035,0.12);
		let dt=t-clamp(p.velocityTime.w,0.0,1.0);
		let dynamicGate=floorValue+(1.0-floorValue)*exp(-0.5*dt*dt/(sigma*sigma));
		return mix(dynamicGate,1.0,clamp(p.centerStatic.w,0.0,1.0));
	}
	fn frame_time(cfg:TiledConfig)->f32 {
		if(cfg.staticWarmup!=0u){return 0.5;}
		return select(0.0,f32(cfg.frameIndex)/f32(max(1u,cfg.frameCount-1u)),cfg.frameCount>1u);
	}
	fn training_background(packed:u32)->vec3<f32> {
		let rgb=vec3<f32>(
			f32(packed&0x3ffu),
			f32((packed>>10u)&0x3ffu),
			f32((packed>>20u)&0x3ffu))*(1.0/1023.0);
		return select(vec3<f32>(0.0),rgb,(packed&0x80000000u)!=0u);
	}
	fn pack_compact_projection_vjp(
		cameraPointValid:vec4<f32>,
		jacobianSparse:vec4<f32>,
		basisVariance0:vec4<f32>,
		basisVariance1:vec4<f32>,
		basisVariance2:vec4<f32>
	)->PackedCompactProjectionVjp {
		return PackedCompactProjectionVjp(
			cameraPointValid,
			vec4<u32>(
				pack2x16float(clamp(jacobianSparse.xy,vec2<f32>(-65504.0),vec2<f32>(65504.0))),
				pack2x16float(clamp(jacobianSparse.zw,vec2<f32>(-65504.0),vec2<f32>(65504.0))),
				pack2x16float(clamp(basisVariance0.xy,vec2<f32>(-65504.0),vec2<f32>(65504.0))),
				pack2x16float(clamp(basisVariance0.zw,vec2<f32>(-65504.0),vec2<f32>(65504.0)))),
			vec4<u32>(
				pack2x16float(clamp(basisVariance1.xy,vec2<f32>(-65504.0),vec2<f32>(65504.0))),
				pack2x16float(clamp(basisVariance1.zw,vec2<f32>(-65504.0),vec2<f32>(65504.0))),
				pack2x16float(clamp(basisVariance2.xy,vec2<f32>(-65504.0),vec2<f32>(65504.0))),
				pack2x16float(clamp(basisVariance2.zw,vec2<f32>(-65504.0),vec2<f32>(65504.0)))));
	}
	fn half_packable(value:vec4<f32>)->bool {
		// Comparisons also reject NaNs: both ordered bounds are false.
		return all(value>=vec4<f32>(-65504.0))&&all(value<=vec4<f32>(65504.0));
	}
`;

function targetDecodeWgsl() {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> packedTargets:array<u32>;
	@group(0) @binding(2) var<storage,read_write> floatTargets:array<vec4<f32>>;
	@compute @workgroup_size(256)
	fn decode_target_page(@builtin(global_invocation_id) gid:vec3<u32>){
		if(cfg.configPad0==0u||gid.x>=cfg.pixelCount){return;}
		let packed=packedTargets[gid.x];
		// RGBA8 is only the immutable dataset storage contract. Decode the selected
		// page before training so raster values and loss gradients stay continuous.
		let rgba=unpack4x8unorm(packed);
		floatTargets[gid.x]=vec4<f32>(
			rgba.xyz,
			f32((packed>>24u)&0xffu)/127.0);
	}`;
}

function projectWgsl(
	projectionLayout,
	projectionVjpPrecision = TILED_PROJECTION_VJP_PRECISIONS.F32,
) {
	const split = projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT;
	const packed = split
		&& projectionVjpPrecision === TILED_PROJECTION_VJP_PRECISIONS.PACKED_F16;
	const vjpType = packed
		? "PackedCompactProjectionVjp"
		: split ? "CompactProjectionVjp" : "ProjectionVjp";
	const compactVjp = (cameraPoint, jacobian, basis0, basis1, basis2) => packed
		? `pack_compact_projection_vjp(${cameraPoint},${jacobian},${basis0},${basis1},${basis2})`
		: `CompactProjectionVjp(${cameraPoint},${jacobian},${basis0},${basis1},${basis2})`;
	const storedVariances = packed ? "storedVariances" : "variances";
	const packedVjpPreflight = packed ? `
		// World-space variances can be subnormal in f16. Store them relative
		// to scene scale, then restore world units in the one-per-splat VJP.
		let varianceScale=max(cfg.geometryScale*cfg.geometryScale,1e-12);
		let storedVariances=variances/varianceScale;
		let packedJacobian=vec4<f32>(j0.x,j0.z,j1.y,j1.z);
		let packedBasisVariance0=vec4<f32>(basis[0],storedVariances.x);
		let packedBasisVariance1=vec4<f32>(basis[1],storedVariances.y);
		let packedBasisVariance2=vec4<f32>(basis[2],storedVariances.z);
		if(!half_packable(packedJacobian)
			||!half_packable(packedBasisVariance0)
			||!half_packable(packedBasisVariance1)
			||!half_packable(packedBasisVariance2)){
			atomicAdd(&counters[8],1u);
			atomicAdd(&counters[9],1u);
		}` : "";
	const zeroVjp = split
		? compactVjp(
			"vec4<f32>(0.0)",
			"vec4<f32>(0.0)",
			"vec4<f32>(0.0)",
			"vec4<f32>(0.0)",
			"vec4<f32>(0.0)",
		)
		: `ProjectionVjp(
			vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),
			vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),
			vec4<f32>(0.0),vec4<f32>(0.0))`;
	const cameraPointVjp = split
		? compactVjp(
			"vec4<f32>(cp,0.0)",
			"vec4<f32>(0.0)",
			"vec4<f32>(0.0)",
			"vec4<f32>(0.0)",
			"vec4<f32>(0.0)",
		)
		: `ProjectionVjp(
			vec4<f32>(cp,0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),
			vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),
			vec4<f32>(0.0),vec4<f32>(0.0))`;
	const validVjp = split
		? compactVjp(
			"vec4<f32>(cp,1.0)",
			"vec4<f32>(j0.x,j0.z,j1.y,j1.z)",
			`vec4<f32>(basis[0],${storedVariances}.x)`,
			`vec4<f32>(basis[1],${storedVariances}.y)`,
			`vec4<f32>(basis[2],${storedVariances}.z)`,
		)
		: `ProjectionVjp(vec4<f32>(cp,1.0),vec4<f32>(j0,0.0),vec4<f32>(j1,0.0),
			vec4<f32>(basis[0],0.0),vec4<f32>(basis[1],0.0),vec4<f32>(basis[2],0.0),
			vec4<f32>(cameraRotation[0],0.0),vec4<f32>(cameraRotation[1],0.0),
			vec4<f32>(cameraRotation[2],0.0),vec4<f32>(variances,0.0))`;
	const projectionBindings = split ? `
	@group(0) @binding(5) var<storage,read_write> rasterProjections:array<RasterProjection>;
	@group(0) @binding(6) var<storage,read_write> counters:array<atomic<u32>>;
	@group(0) @binding(7) var<storage,read_write> projectionVjps:array<${vjpType}>;
	fn write_projection(index:u32,raster:RasterProjection,vjp:${vjpType}){
		rasterProjections[index]=raster;projectionVjps[index]=vjp;
	}` : `
	@group(0) @binding(5) var<storage,read_write> projections:array<Projection>;
	@group(0) @binding(6) var<storage,read_write> counters:array<atomic<u32>>;
	fn write_projection(index:u32,raster:RasterProjection,vjp:ProjectionVjp){
		projections[index]=Projection(
			raster.screenConic0,raster.conicDepthAlpha,vjp.cameraPointValid,
			vjp.jacobian0,vjp.jacobian1,vjp.basis0,vjp.basis1,vjp.basis2,
			vjp.camera0,vjp.camera1,vjp.camera2,vjp.variancesPad);
	}`;
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> cameras:array<Camera>;
	@group(0) @binding(3) var<storage,read_write> tileCounts:array<atomic<u32>>;
	@group(0) @binding(4) var<storage,read_write> pairData:array<u32>;
	${projectionBindings}
	fn quadratic(d:vec2<f32>,q:vec3<f32>)->f32 {
		return q.x*d.x*d.x+2.0*q.y*d.x*d.y+q.z*d.y*d.y;
	}
	fn ellipse_intersects_rect(m:vec2<f32>,conic:vec3<f32>,tau:f32,
		minimum:vec2<f32>,maximum:vec2<f32>)->bool {
		let d0=minimum-m;let d1=maximum-m;
		if(all(m>=minimum)&&all(m<=maximum)){return true;}
		var qmin=min(min(quadratic(vec2<f32>(d0.x,d0.y),conic),
			quadratic(vec2<f32>(d0.x,d1.y),conic)),
			min(quadratic(vec2<f32>(d1.x,d0.y),conic),quadratic(vec2<f32>(d1.x,d1.y),conic)));
		if(conic.z>1e-8){
			qmin=min(qmin,quadratic(vec2<f32>(d0.x,clamp(-(conic.y/conic.z)*d0.x,d0.y,d1.y)),conic));
			qmin=min(qmin,quadratic(vec2<f32>(d1.x,clamp(-(conic.y/conic.z)*d1.x,d0.y,d1.y)),conic));
		}
		if(conic.x>1e-8){
			qmin=min(qmin,quadratic(vec2<f32>(clamp(-(conic.y/conic.x)*d0.y,d0.x,d1.x),d0.y),conic));
			qmin=min(qmin,quadratic(vec2<f32>(clamp(-(conic.y/conic.x)*d1.y,d0.x,d1.x),d1.y),conic));
		}
		return qmin<=tau;
	}
	@compute @workgroup_size(64)
	fn project_and_bin(@builtin(global_invocation_id) gid:vec3<u32>){
		let i=gid.x;if(i>=cfg.splatCount){return;}let p=params[i];
		let t=frame_time(cfg);
		let opacity=sigmoid(p.colorOpacity.w);
		let timeWeight=select(temporal_gate(p,t,cfg.temporalSigma),1.0,cfg.staticWarmup!=0u);
		let peak=opacity*timeWeight;
		var raster=RasterProjection(vec4<f32>(0.0),vec4<f32>(0.0));
		var vjp:${vjpType}=${zeroVjp};
		// Reserved topology slots carry opacity -12 and cannot contribute or
		// receive an image gradient. Reject them before camera covariance work;
		// this keeps a 32K reserve cheap while it is progressively populated.
		if(peak<=cfg.alphaThreshold){write_projection(i,raster,vjp);return;}
		let camera=cameras[cfg.viewIndex];
		let h=vec4<f32>(world_center(p,t,cfg.modelMode),1.0);
		let cp=vec3<f32>(dot(camera.row0,h),dot(camera.row1,h),dot(camera.row2,h));
		vjp=${cameraPointVjp};
		if(cp.z<=0.1){write_projection(i,raster,vjp);return;}
		let cameraRotation=mat3x3<f32>(
			vec3<f32>(camera.row0.x,camera.row1.x,camera.row2.x),
			vec3<f32>(camera.row0.y,camera.row1.y,camera.row2.y),
			vec3<f32>(camera.row0.z,camera.row1.z,camera.row2.z));
		let basis=cameraRotation*quaternion_matrix(p.rotation);
		let variances=exp(2.0*clamp(p.logScalePad.xyz,vec3<f32>(-16.0),vec3<f32>(4.0)));
		let sigmaCamera=variances.x*outer3(basis[0],basis[0])
			+variances.y*outer3(basis[1],basis[1])+variances.z*outer3(basis[2],basis[2]);
		let invZ=1.0/cp.z; let horizontalFocal=cfg.targetAspect*camera.intrinsics.x;
		let j0=vec3<f32>(horizontalFocal*invZ,0.0,-horizontalFocal*cp.x*invZ*invZ);
		let j1=vec3<f32>(0.0,camera.intrinsics.y*invZ,-camera.intrinsics.y*cp.y*invZ*invZ);
		// Conservative screen-space footprint floor. This is point-sampled
		// EWA-style filtering, not Mip-Splatting's compensated pixel filter.
		let filterVariance=pow(${FILTER_SIGMA_PIXELS}/max(1.0,f32(cfg.height)),2.0);
		let covariance=vec3<f32>(dot(j0,sigmaCamera*j0)+filterVariance,
			dot(j0,sigmaCamera*j1),dot(j1,sigmaCamera*j1)+filterVariance);
		let determinant=covariance.x*covariance.z-covariance.y*covariance.y;
		if(determinant<=1e-16){write_projection(i,raster,vjp);return;}
		let center=vec2<f32>(cfg.targetAspect*(camera.intrinsics.x*cp.x*invZ+camera.intrinsics.z),
			camera.intrinsics.y*cp.y*invZ+camera.intrinsics.w);
		let conic=vec3<f32>(covariance.z,-covariance.y,covariance.x)/determinant;
		raster=RasterProjection(
			vec4<f32>(center,conic.xy),vec4<f32>(conic.z,cp.z,opacity,timeWeight));
		${packedVjpPreflight}
		vjp=${validVjp};
		write_projection(i,raster,vjp);
		let qLimit=min(9.0,2.0*log(peak/cfg.alphaThreshold));
		let centerPx=vec2<f32>(center.x*f32(cfg.height),center.y*f32(cfg.height));
		let radiusPx=vec2<f32>(sqrt(max(0.0,qLimit*covariance.x)),
			sqrt(max(0.0,qLimit*covariance.z)))*f32(cfg.height);
		let minPixel=vec2<i32>(max(vec2<i32>(0),vec2<i32>(floor(centerPx-radiusPx-vec2<f32>(0.5)))));
		let maxPixel=min(vec2<i32>(i32(cfg.width)-1,i32(cfg.height)-1),
			vec2<i32>(ceil(centerPx+radiusPx-vec2<f32>(0.5))));
		if(any(minPixel>maxPixel)){return;}
		atomicAdd(&counters[3],1u);
		let minTile=vec2<u32>(minPixel)/cfg.tileSize; let maxTile=vec2<u32>(maxPixel)/cfg.tileSize;
		for(var ty=minTile.y;ty<=maxTile.y;ty++){
			for(var tx=minTile.x;tx<=maxTile.x;tx++){
				let pixelMin=vec2<f32>(f32(tx*cfg.tileSize)+0.5,f32(ty*cfg.tileSize)+0.5)/f32(cfg.height);
				let pixelMax=vec2<f32>(f32(min(cfg.width-1u,(tx+1u)*cfg.tileSize-1u))+0.5,
					f32(min(cfg.height-1u,(ty+1u)*cfg.tileSize-1u))+0.5)/f32(cfg.height);
				if(ellipse_intersects_rect(center,conic,qLimit,pixelMin,pixelMax)){
					let tile=ty*cfg.tilesX+tx; let slot=atomicAdd(&tileCounts[tile],1u);
					if(slot<cfg.tileCapacity){
						pairData[tile*cfg.tileCapacity+slot]=i;
					}else{
						atomicAdd(&counters[1],1u);
						atomicAdd(&counters[4],1u);
					}
				}
			}
		}
	}`;
}

function clearWgsl(gradientFloats) {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read_write> tileCounts:array<atomic<u32>>;
	@group(0) @binding(2) var<storage,read_write> counters:array<atomic<u32>>;
	@group(0) @binding(3) var<storage,read_write> indirectArgs:array<u32>;
	@group(0) @binding(4) var<storage,read_write> metrics:array<vec4<f32>>;
	@group(0) @binding(5) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	@compute @workgroup_size(64)
	fn clear_step(@builtin(global_invocation_id) gid:vec3<u32>){
		let tileCount=cfg.tilesX*cfg.tilesY;
		if(gid.x<tileCount){atomicStore(&tileCounts[gid.x],0u);}
		if(gid.x<cfg.activeSplatCount*${gradientFloats}u){atomicStore(&gradientAtoms[gid.x],0u);}
		if(gid.x==0u){
			atomicStore(&counters[0],0u);atomicStore(&counters[1],0u);
			atomicStore(&counters[2],0u);atomicStore(&counters[3],0u);
			atomicStore(&counters[6],0u);
			atomicStore(&counters[7],cfg.activeSplatCount);
			atomicStore(&counters[8],0u);
			indirectArgs[0]=0u;indirectArgs[1]=1u;indirectArgs[2]=1u;
			metrics[0]=vec4<f32>(0.0);metrics[1]=vec4<f32>(0.0);
			metrics[2]=vec4<f32>(0.0);metrics[3]=vec4<f32>(0.0);
			metrics[4]=vec4<f32>(0.0);
		}
	}`;
}

function sortWgsl(
	tileCapacity,
	backwardGranularity = TILED_BACKWARD_GRANULARITIES.PAIR,
	projectionLayout = TILED_PROJECTION_LAYOUTS.MONOLITHIC,
) {
	const checkpointBlocks = backwardGranularity
		=== TILED_BACKWARD_GRANULARITIES.CHECKPOINT_BLOCK;
	const projectionType = projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT
		? "RasterProjection" : "Projection";
	const depthMask = `0x${TILED_DEPTH_KEY_MASK.toString(16).padStart(8, "0")}u`;
	const idMask = `0x${TILED_SPLAT_ID_MASK.toString(16).padStart(8, "0")}u`;
	const compactSetup = checkpointBlocks ? `
			atomicAdd(&counters[0],count);
			compactCount=(count+cfg.checkpointStride-1u)/cfg.checkpointStride;
			compactBase=atomicAdd(&counters[6],compactCount);` : `
			compactCount=count;
			compactBase=atomicAdd(&counters[0],count);`;
	const compactSlot = checkpointBlocks
		? "tile*cfg.tileCapacity+index*cfg.checkpointStride"
		: "tile*cfg.tileCapacity+index";
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> projections:array<${projectionType}>;
	@group(0) @binding(2) var<storage,read_write> tileCounts:array<atomic<u32>>;
	@group(0) @binding(3) var<storage,read_write> pairData:array<u32>;
	@group(0) @binding(4) var<storage,read_write> counters:array<atomic<u32>>;
	var<workgroup> depthKeys:array<u32,${tileCapacity}>;
	var<workgroup> tileSortCount:u32;
	var<workgroup> compactBase:u32;
	var<workgroup> compactCount:u32;
	@compute @workgroup_size(256)
	fn sort_tiles(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
		let tile=wid.x;let tileCount=cfg.tilesX*cfg.tilesY;if(tile>=tileCount){return;}
		let count=min(atomicLoad(&tileCounts[tile]),cfg.tileCapacity);
		if(lid.x==0u){
			var span=1u;
			loop{
				if(span>=max(count,1u)){break;}
				span*=2u;
			}
			tileSortCount=span;
		}
		let sortCount=workgroupUniformLoad(&tileSortCount);
		for(var index=lid.x;index<sortCount;index+=256u){
			if(index<count){
				let id=pairData[tile*cfg.tileCapacity+index];
				let depthBits=bitcast<u32>(max(0.0,projections[id].conicDepthAlpha.y));
				depthKeys[index]=(depthBits&${depthMask})|(id&${idMask});
			}else{depthKeys[index]=0xffffffffu;}
		}
		workgroupBarrier();
		for(var width=2u;width<=sortCount;width*=2u){
			var stride=width/2u;
			loop{
				for(var index=lid.x;index<sortCount;index+=256u){
					let partner=index^stride;
					if(partner>index){
						let ascending=(index&width)==0u;
						let swap=select(depthKeys[index]<depthKeys[partner],
							depthKeys[index]>depthKeys[partner],ascending);
						if(swap){
							let key=depthKeys[index];depthKeys[index]=depthKeys[partner];depthKeys[partner]=key;
						}
					}
				}
				workgroupBarrier();if(stride==1u){break;}stride/=2u;
			}
		}
		for(var index=lid.x;index<count;index+=256u){
			let slot=tile*cfg.tileCapacity+index;let id=depthKeys[index]&${idMask};pairData[slot]=id;
		}
		if(lid.x==0u){
			atomicMax(&counters[2],count);
			atomicMax(&counters[5],count);
			${compactSetup}
		}
		workgroupBarrier();
		for(var index=lid.x;index<compactCount;index+=256u){
			pairData[cfg.pairCapacity+compactBase+index]=${compactSlot};
		}
	}`;
}

function finalizeWgsl(
	backwardGranularity = TILED_BACKWARD_GRANULARITIES.PAIR,
) {
	const counter = backwardGranularity === TILED_BACKWARD_GRANULARITIES.CHECKPOINT_BLOCK ? 6 : 0;
	return `
	@group(0) @binding(0) var<storage,read_write> counters:array<atomic<u32>>;
	@group(0) @binding(1) var<storage,read_write> indirectArgs:array<u32>;
	@compute @workgroup_size(1) fn finalize_pairs(){
		let dispatchCount=atomicLoad(&counters[${counter}]);
		indirectArgs[0]=min(dispatchCount,${MAX_WORKGROUPS_PER_DIMENSION}u);
		indirectArgs[1]=max(1u,(dispatchCount+${MAX_WORKGROUPS_PER_DIMENSION - 1}u)
			/${MAX_WORKGROUPS_PER_DIMENSION}u);
		indirectArgs[2]=1u;
	}
	`;
}

function checkpointIndexWgsl(order) {
	const expression = order === TILED_CHECKPOINT_ORDERS.BLOCK_MAJOR
		? "block*cfg.pixelCount+pixel"
		: "pixel*cfg.blocksPerTile+block";
	return `fn checkpoint_index(pixel:u32,block:u32)->u32{return ${expression};}`;
}

function checkpointForwardWgsl(precision, order) {
	return precision === "packed-f16" ? `
	@group(0) @binding(6) var<storage,read_write> checkpoints:array<vec2<u32>>;
	${checkpointIndexWgsl(order)}
	fn write_checkpoint(index:u32,state:vec4<f32>){
		checkpoints[index]=vec2<u32>(pack2x16float(state.xy),pack2x16float(state.zw));
	}` : `
	@group(0) @binding(6) var<storage,read_write> checkpoints:array<vec4<f32>>;
	${checkpointIndexWgsl(order)}
	fn write_checkpoint(index:u32,state:vec4<f32>){checkpoints[index]=state;}
	`;
}

function checkpointBackwardWgsl(precision, order) {
	return precision === "packed-f16" ? `
	@group(0) @binding(5) var<storage,read> checkpoints:array<vec2<u32>>;
	${checkpointIndexWgsl(order)}
	fn read_checkpoint(index:u32)->vec4<f32>{
		let packed=checkpoints[index];
		return vec4<f32>(unpack2x16float(packed.x),unpack2x16float(packed.y));
	}` : `
	@group(0) @binding(5) var<storage,read> checkpoints:array<vec4<f32>>;
	${checkpointIndexWgsl(order)}
	fn read_checkpoint(index:u32)->vec4<f32>{return checkpoints[index];}
	`;
}

function forwardWgsl(
	checkpointPrecision,
	checkpointOrder,
	tileSize = DEFAULT_TILE_SIZE,
	projectionLayout = TILED_PROJECTION_LAYOUTS.MONOLITHIC,
) {
	const projectionType = projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT
		? "RasterProjection" : "Projection";
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> projections:array<${projectionType}>;
	@group(0) @binding(3) var<storage,read_write> tileCounts:array<atomic<u32>>;
	@group(0) @binding(4) var<storage,read> pairData:array<u32>;
	@group(0) @binding(5) var<storage,read_write> rendered:array<vec4<f32>>;
	${checkpointForwardWgsl(checkpointPrecision, checkpointOrder)}
	@group(0) @binding(7) var<storage,read_write> stopRanks:array<u32>;
	fn alpha_at(proj:${projectionType},point:vec2<f32>)->f32{
		let d=point-proj.screenConic0.xy;
		let q=proj.screenConic0.z*d.x*d.x+2.0*proj.screenConic0.w*d.x*d.y
			+proj.conicDepthAlpha.x*d.y*d.y;
		if(q<0.0||q>9.0){return 0.0;}
		let raw=proj.conicDepthAlpha.z*proj.conicDepthAlpha.w*exp(-0.5*q);
		return select(0.0,min(0.99,raw),raw>=cfg.alphaThreshold);
	}
	@compute @workgroup_size(${tileSize},${tileSize})
	fn raster_forward(@builtin(global_invocation_id) gid:vec3<u32>){
		if(gid.x>=cfg.width||gid.y>=cfg.height){return;}
		let pixel=gid.y*cfg.width+gid.x;let tile=(gid.y/cfg.tileSize)*cfg.tilesX+(gid.x/cfg.tileSize);
		let count=min(atomicLoad(&tileCounts[tile]),cfg.tileCapacity);
		let point=vec2<f32>((f32(gid.x)+0.5)/f32(cfg.height),(f32(gid.y)+0.5)/f32(cfg.height));
		var color=vec3<f32>(0.0);var transmittance=1.0;var stop=count;
		// Depth-sorted source-over is the model's occlusion/transmittance law.
		// A softmax over contributors would normalize away this visibility state.
		for(var rank=0u;rank<count;rank++){
			if(rank%cfg.checkpointStride==0u){
				write_checkpoint(checkpoint_index(pixel,rank/cfg.checkpointStride),vec4<f32>(color,transmittance));
			}
			let id=pairData[tile*cfg.tileCapacity+rank];let alpha=alpha_at(projections[id],point);
			color+=transmittance*alpha*params[id].colorOpacity.xyz;transmittance*=1.0-alpha;
			if(transmittance<cfg.transmittanceThreshold){stop=rank+1u;break;}
		}
		// Randomizing only the train underlay breaks the color/opacity shortcut
		// without injecting a camera image. Alpha remains true splat coverage.
		let background=training_background(cfg.trainingBackgroundPacked);
		rendered[pixel]=vec4<f32>(color+transmittance*background,1.0-transmittance);
		stopRanks[pixel]=stop;
	}`;
}

function ssimStatsWgsl() {
	return `${CONFIG_WGSL}
	struct SsimStats { muX:vec4<f32>,muY:vec4<f32>,varX:vec4<f32>,varY:vec4<f32>,cov:vec4<f32> };
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> rendered:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read> targets:array<vec4<f32>>;
	@group(0) @binding(3) var<storage,read_write> stats:array<SsimStats>;
	@group(0) @binding(4) var<storage,read_write> pixelLoss:array<vec4<f32>>;
	fn loss_weight(targetPixel:vec4<f32>)->f32 {
		return select(1.0,targetPixel.w,cfg.motionWeighting!=0u);
	}
	fn reflect_index(index:i32,size:u32)->u32 {
		if(size<=1u){return 0u;}var resolved=index;let maximum=i32(size)-1;
		loop{
			if(resolved>=0&&resolved<=maximum){break;}
			if(resolved<0){resolved=-resolved;}
			if(resolved>maximum){resolved=2*maximum-resolved;}
		}
		return u32(resolved);
	}
	fn ssim_weight(offset:i32,radius:i32)->f32 {
		if(radius==0){return 1.0;}
		if(radius!=5){return 1.0/f32(radius*2+1);}
		switch abs(offset) {
			case 0: { return 0.2660117149; }
			case 1: { return 0.2130055428; }
			case 2: { return 0.1093606874; }
			case 3: { return 0.0360007733; }
			case 4: { return 0.0075987582; }
			case 5: { return 0.0010283801; }
			default: { return 0.0; }
		}
	}
	@compute @workgroup_size(64)
	fn ssim_stats(@builtin(global_invocation_id) gid:vec3<u32>){
		let pixel=gid.x;if(pixel>=cfg.pixelCount){return;}let x=i32(pixel%cfg.width);let y=i32(pixel/cfg.width);
		var mux=vec3<f32>(0.0);var muy=vec3<f32>(0.0);
		var ex2=vec3<f32>(0.0);var ey2=vec3<f32>(0.0);var exy=vec3<f32>(0.0);
		let radius=i32(cfg.ssimRadius);
		for(var oy=-radius;oy<=radius;oy++){
			let sy=reflect_index(y+oy,cfg.height);
			for(var ox=-radius;ox<=radius;ox++){
				let sx=reflect_index(x+ox,cfg.width);let sample=sy*cfg.width+sx;
				let px=rendered[sample].xyz;let py=targets[cfg.targetOffset+sample].xyz;
				let weight=ssim_weight(oy,radius)*ssim_weight(ox,radius);
				mux+=weight*px;muy+=weight*py;ex2+=weight*px*px;
				ey2+=weight*py*py;exy+=weight*px*py;
			}
		}
		let vx=ex2-mux*mux;
		let vy=ey2-muy*muy;let covariance=exy-mux*muy;
		stats[pixel]=SsimStats(vec4<f32>(mux,0.0),vec4<f32>(muy,0.0),vec4<f32>(vx,0.0),
			vec4<f32>(vy,0.0),vec4<f32>(covariance,0.0));
		let a=2.0*mux*muy+vec3<f32>(cfg.c1);let b=2.0*covariance+vec3<f32>(cfg.c2);
		let c=mux*mux+muy*muy+vec3<f32>(cfg.c1);let d=vx+vy+vec3<f32>(cfg.c2);
		let ssim=(a*b)/max(c*d,vec3<f32>(1e-12));
		let targetPixel=targets[cfg.targetOffset+pixel];
		let objectiveWeight=loss_weight(targetPixel);
		let err=rendered[pixel].xyz-targetPixel.xyz;
		pixelLoss[pixel]=vec4<f32>(objectiveWeight*(abs(err.x)+abs(err.y)+abs(err.z))/3.0,
			objectiveWeight*(1.0-(ssim.x+ssim.y+ssim.z)/3.0),rendered[pixel].w,0.0);
	}`;
}

function ssimGradientWgsl() {
	return `${CONFIG_WGSL}
	struct SsimStats { muX:vec4<f32>,muY:vec4<f32>,varX:vec4<f32>,varY:vec4<f32>,cov:vec4<f32> };
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> rendered:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read> targets:array<vec4<f32>>;
	@group(0) @binding(3) var<storage,read> stats:array<SsimStats>;
	@group(0) @binding(4) var<storage,read> stopRanks:array<u32>;
	@group(0) @binding(5) var<storage,read_write> pixelGrad:array<vec4<f32>>;
	fn loss_weight(targetPixel:vec4<f32>)->f32 {
		return select(1.0,targetPixel.w,cfg.motionWeighting!=0u);
	}
	fn ssim_weight(offset:i32,radius:i32)->f32 {
		if(radius==0){return 1.0;}
		if(radius!=5){return 1.0/f32(radius*2+1);}
		switch abs(offset) {
			case 0: { return 0.2660117149; }
			case 1: { return 0.2130055428; }
			case 2: { return 0.1093606874; }
			case 3: { return 0.0360007733; }
			case 4: { return 0.0075987582; }
			case 5: { return 0.0010283801; }
			default: { return 0.0; }
		}
	}
	fn reflected_weight(center:i32,pixel:i32,size:i32,radius:i32)->f32 {
		var weight=select(0.0,ssim_weight(pixel-center,radius),abs(center-pixel)<=radius);
		if(pixel>0&&abs(center+pixel)<=radius){weight+=ssim_weight(-pixel-center,radius);}
		let maximum=size-1;
		let rightOffset=2*maximum-pixel-center;
		if(pixel<maximum&&abs(rightOffset)<=radius){weight+=ssim_weight(rightOffset,radius);}
		return weight;
	}
	@compute @workgroup_size(64)
	fn ssim_gradient(@builtin(global_invocation_id) gid:vec3<u32>){
		let pixel=gid.x;if(pixel>=cfg.pixelCount){return;}
		let px=i32(pixel%cfg.width);let py=i32(pixel/cfg.width);
		let prediction=rendered[pixel].xyz;let targetColor=targets[cfg.targetOffset+pixel].xyz;
		var dssim=vec3<f32>(0.0);let radius=i32(cfg.ssimRadius);
		for(var cy=max(0,py-radius);cy<=min(i32(cfg.height)-1,py+radius);cy++){
			let yWeight=reflected_weight(cy,py,i32(cfg.height),radius);
			for(var cx=max(0,px-radius);cx<=min(i32(cfg.width)-1,px+radius);cx++){
				let center=u32(cy)*cfg.width+u32(cx);
				let weight=loss_weight(targets[cfg.targetOffset+center])
					*yWeight*reflected_weight(cx,px,i32(cfg.width),radius);
				if(weight==0.0){continue;}
				let s=stats[center];
				let mx=s.muX.xyz;let my=s.muY.xyz;let vx=s.varX.xyz;let vy=s.varY.xyz;let covariance=s.cov.xyz;
				let a=2.0*mx*my+vec3<f32>(cfg.c1);let b=2.0*covariance+vec3<f32>(cfg.c2);
				let c=mx*mx+my*my+vec3<f32>(cfg.c1);let d=vx+vy+vec3<f32>(cfg.c2);
				let da=2.0*my*weight;let db=2.0*weight*(targetColor-my);
				let dc=2.0*mx*weight;let dd=2.0*weight*(prediction-mx);
				let denominatorRaw=c*d;let denominator=max(denominatorRaw,vec3<f32>(1e-12));
				let dDenominator=select(dc*d+c*dd,vec3<f32>(0.0),denominatorRaw<vec3<f32>(1e-12));
				dssim-=(((da*b+a*db)*denominator)-(a*b)*dDenominator)
					/(denominator*denominator*f32(cfg.pixelCount)*3.0);
			}
		}
		let l1=loss_weight(targets[cfg.targetOffset+pixel])
			*sign(prediction-targetColor)/(f32(cfg.pixelCount)*3.0);
		pixelGrad[pixel]=vec4<f32>(cfg.l1Weight*l1+cfg.dssimWeight*dssim,
			f32(stopRanks[pixel]));
	}`;
}

function separableSsimHorizontalWgsl() {
	return `${CONFIG_WGSL}
	struct SsimStats { muX:vec4<f32>,muY:vec4<f32>,varX:vec4<f32>,varY:vec4<f32>,cov:vec4<f32> };
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> rendered:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read> targets:array<vec4<f32>>;
	@group(0) @binding(3) var<storage,read_write> scratch:array<SsimStats>;
	fn reflect_index(index:i32,size:u32)->u32 {
		if(size<=1u){return 0u;}var resolved=index;let maximum=i32(size)-1;
		loop{
			if(resolved>=0&&resolved<=maximum){break;}
			if(resolved<0){resolved=-resolved;}
			if(resolved>maximum){resolved=2*maximum-resolved;}
		}
		return u32(resolved);
	}
	fn ssim_weight(offset:i32,radius:i32)->f32 {
		if(radius==0){return 1.0;}
		if(radius!=5){return 1.0/f32(radius*2+1);}
		switch abs(offset) {
			case 0: { return 0.2660117149; }
			case 1: { return 0.2130055428; }
			case 2: { return 0.1093606874; }
			case 3: { return 0.0360007733; }
			case 4: { return 0.0075987582; }
			case 5: { return 0.0010283801; }
			default: { return 0.0; }
		}
	}
	@compute @workgroup_size(64)
	fn ssim_horizontal(@builtin(global_invocation_id) gid:vec3<u32>){
		let pixel=gid.x;if(pixel>=cfg.pixelCount){return;}
		let x=i32(pixel%cfg.width);let y=pixel/cfg.width;
		var mux=vec3<f32>(0.0);var muy=vec3<f32>(0.0);
		var ex2=vec3<f32>(0.0);var ey2=vec3<f32>(0.0);var exy=vec3<f32>(0.0);
		let radius=i32(cfg.ssimRadius);
		for(var ox=-radius;ox<=radius;ox++){
			let sx=reflect_index(x+ox,cfg.width);let sample=y*cfg.width+sx;
			let px=rendered[sample].xyz;let py=targets[cfg.targetOffset+sample].xyz;
			let weight=ssim_weight(ox,radius);
			mux+=weight*px;muy+=weight*py;ex2+=weight*px*px;
			ey2+=weight*py*py;exy+=weight*px*py;
		}
		scratch[pixel]=SsimStats(
			vec4<f32>(mux,0.0),vec4<f32>(muy,0.0),vec4<f32>(ex2,0.0),
			vec4<f32>(ey2,0.0),vec4<f32>(exy,0.0));
	}`;
}

function separableSsimVerticalWgsl() {
	return `${CONFIG_WGSL}
	struct SsimStats { muX:vec4<f32>,muY:vec4<f32>,varX:vec4<f32>,varY:vec4<f32>,cov:vec4<f32> };
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> rendered:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read> targets:array<vec4<f32>>;
	@group(0) @binding(3) var<storage,read> scratch:array<SsimStats>;
	@group(0) @binding(4) var<storage,read_write> stats:array<SsimStats>;
	@group(0) @binding(5) var<storage,read_write> pixelLoss:array<vec4<f32>>;
	fn loss_weight(targetPixel:vec4<f32>)->f32 {
		return select(1.0,targetPixel.w,cfg.motionWeighting!=0u);
	}
	fn reflect_index(index:i32,size:u32)->u32 {
		if(size<=1u){return 0u;}var resolved=index;let maximum=i32(size)-1;
		loop{
			if(resolved>=0&&resolved<=maximum){break;}
			if(resolved<0){resolved=-resolved;}
			if(resolved>maximum){resolved=2*maximum-resolved;}
		}
		return u32(resolved);
	}
	fn ssim_weight(offset:i32,radius:i32)->f32 {
		if(radius==0){return 1.0;}
		if(radius!=5){return 1.0/f32(radius*2+1);}
		switch abs(offset) {
			case 0: { return 0.2660117149; }
			case 1: { return 0.2130055428; }
			case 2: { return 0.1093606874; }
			case 3: { return 0.0360007733; }
			case 4: { return 0.0075987582; }
			case 5: { return 0.0010283801; }
			default: { return 0.0; }
		}
	}
	@compute @workgroup_size(64)
	fn ssim_vertical(@builtin(global_invocation_id) gid:vec3<u32>){
		let pixel=gid.x;if(pixel>=cfg.pixelCount){return;}
		let x=pixel%cfg.width;let y=i32(pixel/cfg.width);
		var mux=vec3<f32>(0.0);var muy=vec3<f32>(0.0);
		var ex2=vec3<f32>(0.0);var ey2=vec3<f32>(0.0);var exy=vec3<f32>(0.0);
		let radius=i32(cfg.ssimRadius);
		for(var oy=-radius;oy<=radius;oy++){
			let sy=reflect_index(y+oy,cfg.height);let sample=sy*cfg.width+x;
			let horizontal=scratch[sample];let weight=ssim_weight(oy,radius);
			mux+=weight*horizontal.muX.xyz;muy+=weight*horizontal.muY.xyz;
			ex2+=weight*horizontal.varX.xyz;ey2+=weight*horizontal.varY.xyz;
			exy+=weight*horizontal.cov.xyz;
		}
		let vx=ex2-mux*mux;let vy=ey2-muy*muy;let covariance=exy-mux*muy;
		stats[pixel]=SsimStats(
			vec4<f32>(mux,0.0),vec4<f32>(muy,0.0),vec4<f32>(vx,0.0),
			vec4<f32>(vy,0.0),vec4<f32>(covariance,0.0));
		let a=2.0*mux*muy+vec3<f32>(cfg.c1);
		let b=2.0*covariance+vec3<f32>(cfg.c2);
		let c=mux*mux+muy*muy+vec3<f32>(cfg.c1);
		let d=vx+vy+vec3<f32>(cfg.c2);
		let ssim=(a*b)/max(c*d,vec3<f32>(1e-12));
		let targetPixel=targets[cfg.targetOffset+pixel];
		let objectiveWeight=loss_weight(targetPixel);
		let err=rendered[pixel].xyz-targetPixel.xyz;
		pixelLoss[pixel]=vec4<f32>(
			objectiveWeight*(abs(err.x)+abs(err.y)+abs(err.z))/3.0,
			objectiveWeight*(1.0-(ssim.x+ssim.y+ssim.z)/3.0),
			rendered[pixel].w,0.0);
	}`;
}

function separableSsimGradientHorizontalWgsl() {
	return `${CONFIG_WGSL}
	struct SsimStats { muX:vec4<f32>,muY:vec4<f32>,varX:vec4<f32>,varY:vec4<f32>,cov:vec4<f32> };
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> targets:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read> stats:array<SsimStats>;
	@group(0) @binding(3) var<storage,read_write> scratch:array<SsimStats>;
	fn loss_weight(targetPixel:vec4<f32>)->f32 {
		return select(1.0,targetPixel.w,cfg.motionWeighting!=0u);
	}
	fn ssim_weight(offset:i32,radius:i32)->f32 {
		if(radius==0){return 1.0;}
		if(radius!=5){return 1.0/f32(radius*2+1);}
		switch abs(offset) {
			case 0: { return 0.2660117149; }
			case 1: { return 0.2130055428; }
			case 2: { return 0.1093606874; }
			case 3: { return 0.0360007733; }
			case 4: { return 0.0075987582; }
			case 5: { return 0.0010283801; }
			default: { return 0.0; }
		}
	}
	fn reflected_weight(center:i32,pixel:i32,size:i32,radius:i32)->f32 {
		var weight=select(0.0,ssim_weight(pixel-center,radius),abs(center-pixel)<=radius);
		if(pixel>0&&abs(center+pixel)<=radius){weight+=ssim_weight(-pixel-center,radius);}
		let maximum=size-1;let rightOffset=2*maximum-pixel-center;
		if(pixel<maximum&&abs(rightOffset)<=radius){weight+=ssim_weight(rightOffset,radius);}
		return weight;
	}
	@compute @workgroup_size(64)
	fn ssim_gradient_horizontal(@builtin(global_invocation_id) gid:vec3<u32>){
		let pixel=gid.x;if(pixel>=cfg.pixelCount){return;}
		let px=i32(pixel%cfg.width);let py=pixel/cfg.width;
		let radius=i32(cfg.ssimRadius);
		var constant=vec3<f32>(0.0);
		var targetCoefficient=vec3<f32>(0.0);
		var predictionCoefficient=vec3<f32>(0.0);
		for(var cx=max(0,px-radius);cx<=min(i32(cfg.width)-1,px+radius);cx++){
			let center=py*cfg.width+u32(cx);let s=stats[center];
			let mx=s.muX.xyz;let my=s.muY.xyz;let vx=s.varX.xyz;
			let vy=s.varY.xyz;let covariance=s.cov.xyz;
			let a=2.0*mx*my+vec3<f32>(cfg.c1);
			let b=2.0*covariance+vec3<f32>(cfg.c2);
			let c=mx*mx+my*my+vec3<f32>(cfg.c1);
			let d=vx+vy+vec3<f32>(cfg.c2);
			let numerator=a*b;let denominatorRaw=c*d;
			let denominator=max(denominatorRaw,vec3<f32>(1e-12));
			let denominatorGate=select(
				vec3<f32>(0.0),vec3<f32>(1.0),denominatorRaw>=vec3<f32>(1e-12));
			let centerScale=loss_weight(targets[cfg.targetOffset+center])
				/(denominator*denominator*f32(cfg.pixelCount)*3.0);
			let constantNumerator=2.0*my*b-2.0*a*my;
			let constantDenominator=denominatorGate*(2.0*mx*d-2.0*c*mx);
			let kernelWeight=reflected_weight(cx,px,i32(cfg.width),radius);
			constant+=kernelWeight*(-centerScale
				*(constantNumerator*denominator-numerator*constantDenominator));
			targetCoefficient+=kernelWeight*(-centerScale*(2.0*a*denominator));
			predictionCoefficient+=kernelWeight
				*(centerScale*numerator*denominatorGate*2.0*c);
		}
		scratch[pixel]=SsimStats(
			vec4<f32>(constant,0.0),vec4<f32>(targetCoefficient,0.0),
			vec4<f32>(predictionCoefficient,0.0),vec4<f32>(0.0),vec4<f32>(0.0));
	}`;
}

function separableSsimGradientVerticalWgsl() {
	return `${CONFIG_WGSL}
	struct SsimStats { muX:vec4<f32>,muY:vec4<f32>,varX:vec4<f32>,varY:vec4<f32>,cov:vec4<f32> };
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> rendered:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read> targets:array<vec4<f32>>;
	@group(0) @binding(3) var<storage,read> scratch:array<SsimStats>;
	@group(0) @binding(4) var<storage,read> stopRanks:array<u32>;
	@group(0) @binding(5) var<storage,read_write> pixelGrad:array<vec4<f32>>;
	fn loss_weight(targetPixel:vec4<f32>)->f32 {
		return select(1.0,targetPixel.w,cfg.motionWeighting!=0u);
	}
	fn ssim_weight(offset:i32,radius:i32)->f32 {
		if(radius==0){return 1.0;}
		if(radius!=5){return 1.0/f32(radius*2+1);}
		switch abs(offset) {
			case 0: { return 0.2660117149; }
			case 1: { return 0.2130055428; }
			case 2: { return 0.1093606874; }
			case 3: { return 0.0360007733; }
			case 4: { return 0.0075987582; }
			case 5: { return 0.0010283801; }
			default: { return 0.0; }
		}
	}
	fn reflected_weight(center:i32,pixel:i32,size:i32,radius:i32)->f32 {
		var weight=select(0.0,ssim_weight(pixel-center,radius),abs(center-pixel)<=radius);
		if(pixel>0&&abs(center+pixel)<=radius){weight+=ssim_weight(-pixel-center,radius);}
		let maximum=size-1;let rightOffset=2*maximum-pixel-center;
		if(pixel<maximum&&abs(rightOffset)<=radius){weight+=ssim_weight(rightOffset,radius);}
		return weight;
	}
	@compute @workgroup_size(64)
	fn ssim_gradient_vertical(@builtin(global_invocation_id) gid:vec3<u32>){
		let pixel=gid.x;if(pixel>=cfg.pixelCount){return;}
		let px=pixel%cfg.width;let py=i32(pixel/cfg.width);
		let radius=i32(cfg.ssimRadius);
		var constant=vec3<f32>(0.0);
		var targetCoefficient=vec3<f32>(0.0);
		var predictionCoefficient=vec3<f32>(0.0);
		for(var cy=max(0,py-radius);cy<=min(i32(cfg.height)-1,py+radius);cy++){
			let horizontal=scratch[u32(cy)*cfg.width+px];
			let weight=reflected_weight(cy,py,i32(cfg.height),radius);
			constant+=weight*horizontal.muX.xyz;
			targetCoefficient+=weight*horizontal.muY.xyz;
			predictionCoefficient+=weight*horizontal.varX.xyz;
		}
		let prediction=rendered[pixel].xyz;
		let targetColor=targets[cfg.targetOffset+pixel].xyz;
		let dssim=constant+targetCoefficient*targetColor+predictionCoefficient*prediction;
		let l1=loss_weight(targets[cfg.targetOffset+pixel])
			*sign(prediction-targetColor)/(f32(cfg.pixelCount)*3.0);
		pixelGrad[pixel]=vec4<f32>(
			cfg.l1Weight*l1+cfg.dssimWeight*dssim,f32(stopRanks[pixel]));
	}`;
}

function metricsWgsl() {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> pixelLoss:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read_write> counters:array<atomic<u32>>;
	@group(0) @binding(3) var<storage,read_write> metrics:array<vec4<f32>>;
	@group(0) @binding(4) var<storage,read> stopRanks:array<u32>;
	@group(0) @binding(5) var<storage,read_write> cycleMetrics:array<vec4<f32>>;
	var<workgroup> scratch:array<vec4<f32>,256>;
	var<workgroup> stopScratch:array<f32,256>;
	@compute @workgroup_size(256)
	fn reduce_metrics(@builtin(local_invocation_id) lid:vec3<u32>){
		var total=vec4<f32>(0.0);var stopTotal=0.0;
		for(var pixel=lid.x;pixel<cfg.pixelCount;pixel+=256u){
			total+=pixelLoss[pixel];stopTotal+=f32(stopRanks[pixel]);
		}
		scratch[lid.x]=total;stopScratch[lid.x]=stopTotal;workgroupBarrier();
		var stride=128u;loop{
			if(lid.x<stride){
				scratch[lid.x]+=scratch[lid.x+stride];
				stopScratch[lid.x]+=stopScratch[lid.x+stride];
			}
			workgroupBarrier();if(stride==1u){break;}stride/=2u;
		}
		if(lid.x==0u){
			let mean=scratch[0]/f32(cfg.pixelCount);
			let pairCount=f32(atomicLoad(&counters[0]));
			metrics[0]=vec4<f32>(
				cfg.l1Weight*mean.x+cfg.dssimWeight*mean.y,mean.x,mean.y,
				f32(atomicLoad(&counters[1])));
			metrics[1]=vec4<f32>(
				mean.z,pairCount,
				f32(atomicLoad(&counters[2])),stopScratch[0]/f32(cfg.pixelCount));
			metrics[2]=vec4<f32>(
				f32(atomicLoad(&counters[3])),f32(cfg.splatCount),
				f32(cfg.viewIndex),f32(cfg.frameIndex));
			metrics[3]=vec4<f32>(
				f32(atomicLoad(&counters[4])),f32(atomicLoad(&counters[5])),
				f32(cfg.step),f32(cfg.activeSplatCount));
			metrics[4]=vec4<f32>(
				f32(atomicLoad(&counters[8])),f32(atomicLoad(&counters[9])),0.0,0.0);
			cycleMetrics[cfg.step%arrayLength(&cycleMetrics)]=vec4<f32>(
				metrics[0].xyz,f32(cfg.step+1u));
		}
	}`;
}

function backwardWgsl(
	checkpointPrecision,
	checkpointOrder,
	tileSize = DEFAULT_TILE_SIZE,
) {
	const laneCount = tileSize * tileSize;
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> projections:array<Projection>;
	@group(0) @binding(3) var<storage,read> pairData:array<u32>;
	@group(0) @binding(4) var<storage,read> rendered:array<vec4<f32>>;
	${checkpointBackwardWgsl(checkpointPrecision, checkpointOrder)}
	@group(0) @binding(6) var<storage,read> pixelGrad:array<vec4<f32>>;
	@group(0) @binding(7) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	@group(0) @binding(8) var<storage,read_write> counters:array<atomic<u32>>;
	var<workgroup> gradientScratch:array<Splat,${laneCount}>;
	fn zero_splat()->Splat{return Splat(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));}
	fn add_splat(a:Splat,b:Splat)->Splat{return Splat(a.centerStatic+b.centerStatic,a.velocityTime+b.velocityTime,
		a.harmonicPad+b.harmonicPad,a.logScalePad+b.logScalePad,a.rotation+b.rotation,a.colorOpacity+b.colorOpacity);}
	fn atomic_add_f32(index:u32,value:f32){
		if(value==0.0){return;}
		var oldBits=atomicLoad(&gradientAtoms[index]);
		loop{
			let newBits=bitcast<u32>(bitcast<f32>(oldBits)+value);
			let result=atomicCompareExchangeWeak(&gradientAtoms[index],oldBits,newBits);
			if(result.exchanged){break;}
			oldBits=result.old_value;
		}
	}
	fn accumulate_splat(id:u32,gradient:Splat){
		let base=id*24u;
		for(var component=0u;component<4u;component++){
			atomic_add_f32(base+component,gradient.centerStatic[component]);
			atomic_add_f32(base+4u+component,gradient.velocityTime[component]);
			atomic_add_f32(base+8u+component,gradient.harmonicPad[component]);
			atomic_add_f32(base+12u+component,gradient.logScalePad[component]);
			atomic_add_f32(base+16u+component,gradient.rotation[component]);
			atomic_add_f32(base+20u+component,gradient.colorOpacity[component]);
		}
	}
	fn alpha_at(proj:Projection,point:vec2<f32>)->f32{
		if(proj.cameraPointValid.w<0.5){return 0.0;}let d=point-proj.screenConic0.xy;
		let q=proj.screenConic0.z*d.x*d.x+2.0*proj.screenConic0.w*d.x*d.y+proj.conicDepthAlpha.x*d.y*d.y;
		if(q<0.0||q>9.0){return 0.0;}
		let raw=proj.conicDepthAlpha.z*proj.conicDepthAlpha.w*exp(-0.5*q);
		return select(0.0,min(0.99,raw),raw>=cfg.alphaThreshold);
	}
	@compute @workgroup_size(${tileSize},${tileSize})
	fn pair_backward(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
		let lane=lid.y*${tileSize}u+lid.x;let pair=wid.y*${MAX_WORKGROUPS_PER_DIMENSION}u+wid.x;
		let pairValid=pair<atomicLoad(&counters[0]);var id=0u;var gradient=zero_splat();
		if(pairValid){
			let slot=pairData[cfg.pairCapacity+pair];let tile=slot/cfg.tileCapacity;
			let rank=slot%cfg.tileCapacity;id=pairData[slot];let tileX=tile%cfg.tilesX;let tileY=tile/cfg.tilesX;
			let x=tileX*cfg.tileSize+lid.x;let y=tileY*cfg.tileSize+lid.y;
			if(x<cfg.width&&y<cfg.height){
				let pixel=y*cfg.width+x;
				if(rank<u32(pixelGrad[pixel].w)){
					let point=vec2<f32>((f32(x)+0.5)/f32(cfg.height),(f32(y)+0.5)/f32(cfg.height));
					let block=rank/cfg.checkpointStride;
					let checkpoint=read_checkpoint(checkpoint_index(pixel,block));
					var before=checkpoint.xyz;var transmittance=checkpoint.w;
					for(var replay=block*cfg.checkpointStride;replay<rank;replay++){
						let prior=pairData[tile*cfg.tileCapacity+replay];let alpha=alpha_at(projections[prior],point);
						before+=transmittance*alpha*params[prior].colorOpacity.xyz;transmittance*=1.0-alpha;
					}
					let p=params[id];let proj=projections[id];let d=point-proj.screenConic0.xy;
					let qform=proj.screenConic0.z*d.x*d.x+2.0*proj.screenConic0.w*d.x*d.y+proj.conicDepthAlpha.x*d.y*d.y;
					if(qform>=0.0&&qform<=9.0&&transmittance>cfg.transmittanceThreshold){
						let gaussian=exp(-0.5*qform);
						let rawAlpha=proj.conicDepthAlpha.z*proj.conicDepthAlpha.w*gaussian;
						let alpha=select(0.0,min(0.99,rawAlpha),rawAlpha>=cfg.alphaThreshold);
						let denominator=transmittance*(1.0-alpha);
						// rendered.rgb already includes the train background, so
						// replay recovers deeper splats plus that same underlay.
						let behind=select(vec3<f32>(0.0),(rendered[pixel].xyz-before-transmittance*alpha*p.colorOpacity.xyz)
							/max(denominator,1e-8),denominator>1e-8);
						let imageGrad=pixelGrad[pixel].xyz;
						let alphaGrad=dot(imageGrad,transmittance*(p.colorOpacity.xyz-behind));
						let clampGate=select(0.0,1.0,rawAlpha<0.99&&rawAlpha>=cfg.alphaThreshold);
						let barQform=-0.5*alphaGrad*rawAlpha*clampGate;
						let conicDelta=vec2<f32>(proj.screenConic0.z*d.x+proj.screenConic0.w*d.y,
							proj.screenConic0.w*d.x+proj.conicDepthAlpha.x*d.y);
						let barMu=-2.0*barQform*conicDelta;
						let barC00=-barQform*conicDelta.x*conicDelta.x;
						let barC01=-barQform*conicDelta.x*conicDelta.y;
						let barC11=-barQform*conicDelta.y*conicDelta.y;
						let j0=proj.jacobian0.xyz;let j1=proj.jacobian1.xyz;
						let barSigma=barC00*outer3(j0,j0)+barC01*(outer3(j0,j1)+outer3(j1,j0))+barC11*outer3(j1,j1);
						let basis=mat3x3<f32>(proj.basis0.xyz,proj.basis1.xyz,proj.basis2.xyz);
						let variances=proj.variancesPad.xyz;
						let sigmaCamera=variances.x*outer3(basis[0],basis[0])+variances.y*outer3(basis[1],basis[1])
							+variances.z*outer3(basis[2],basis[2]);
						let sigmaJ0=sigmaCamera*j0;let sigmaJ1=sigmaCamera*j1;
						let barJ0=2.0*(barC00*sigmaJ0+barC01*sigmaJ1);
						let barJ1=2.0*(barC01*sigmaJ0+barC11*sigmaJ1);
						let cp=proj.cameraPointValid.xyz;let invZ=1.0/cp.z;
						let horizontalFocal=proj.jacobian0.x*cp.z;let verticalFocal=proj.jacobian1.y*cp.z;
						let cameraGrad=vec3<f32>(
							barMu.x*horizontalFocal*invZ-barJ0.z*horizontalFocal*invZ*invZ,
							barMu.y*verticalFocal*invZ-barJ1.z*verticalFocal*invZ*invZ,
							-barMu.x*horizontalFocal*cp.x*invZ*invZ-barMu.y*verticalFocal*cp.y*invZ*invZ
							-barJ0.x*horizontalFocal*invZ*invZ+barJ0.z*2.0*horizontalFocal*cp.x*invZ*invZ*invZ
							-barJ1.y*verticalFocal*invZ*invZ+barJ1.z*2.0*verticalFocal*cp.y*invZ*invZ*invZ);
						let cameraRotation=mat3x3<f32>(proj.camera0.xyz,proj.camera1.xyz,proj.camera2.xyz);
						let worldGrad=transpose(cameraRotation)*cameraGrad;
						var gradLogScale=vec3<f32>(0.0);
						for(var axis=0u;axis<3u;axis++){let column=basis[axis];
							gradLogScale[axis]=2.0*variances[axis]*dot(column,barSigma*column);}
						let barBasis=mat3x3<f32>(2.0*variances.x*(barSigma*basis[0]),
							2.0*variances.y*(barSigma*basis[1]),2.0*variances.z*(barSigma*basis[2]));
						let barRotation=transpose(cameraRotation)*barBasis;let q=safe_quaternion(p.rotation);
						let h00=barRotation[0].x;let h01=barRotation[1].x;let h02=barRotation[2].x;
						let h10=barRotation[0].y;let h11=barRotation[1].y;let h12=barRotation[2].y;
						let h20=barRotation[0].z;let h21=barRotation[1].z;let h22=barRotation[2].z;
						let normalizedQuatGrad=vec4<f32>(
							-4.0*q.x*(h11+h22)+2.0*q.y*(h01+h10)+2.0*q.z*(h02+h20)+2.0*q.w*(h21-h12),
							-4.0*q.y*(h00+h22)+2.0*q.x*(h01+h10)+2.0*q.z*(h12+h21)+2.0*q.w*(h02-h20),
							-4.0*q.z*(h00+h11)+2.0*q.x*(h02+h20)+2.0*q.y*(h12+h21)+2.0*q.w*(h10-h01),
							2.0*q.z*(h10-h01)+2.0*q.y*(h02-h20)+2.0*q.x*(h21-h12));
						let rawNorm2=dot(p.rotation,p.rotation);var gradRotation=vec4<f32>(0.0);
						if(rawNorm2>1e-16){gradRotation=(normalizedQuatGrad-q*dot(q,normalizedQuatGrad))*inverseSqrt(rawNorm2);}
						let staticWarmup=cfg.staticWarmup!=0u;let t=frame_time(cfg);
						let tc=select(t*2.0-1.0,0.0,staticWarmup);
						let wave=select(sin(t*6.28318530718),0.0,staticWarmup);
						let sigma=clamp(cfg.temporalSigma,0.12,0.36);let staticMix=clamp(p.centerStatic.w,0.0,1.0);
						let temporalFloor=clamp(sigma*0.30,0.035,0.12);let timeDelta=t-clamp(p.velocityTime.w,0.0,1.0);
						let dynamicGate=temporalFloor+(1.0-temporalFloor)*exp(-0.5*timeDelta*timeDelta/(sigma*sigma));
						let dynamicCore=max(0.0,(proj.conicDepthAlpha.w-staticMix-temporalFloor*(1.0-staticMix))
							/max(1e-6,1.0-staticMix));
						let gradTime=select(clampGate*alphaGrad*proj.conicDepthAlpha.z*gaussian
							*(1.0-staticMix)*dynamicCore*(t-p.velocityTime.w)/(sigma*sigma),0.0,staticWarmup);
						let gradStaticMix=select(
							clampGate*alphaGrad*proj.conicDepthAlpha.z*gaussian*(1.0-dynamicGate),
							0.0,staticWarmup);
						let gradOpacity=clampGate*alphaGrad*gaussian*proj.conicDepthAlpha.w
							*proj.conicDepthAlpha.z*(1.0-proj.conicDepthAlpha.z);
						let gradColor=imageGrad*transmittance*alpha;
						gradient=Splat(vec4<f32>(worldGrad,gradStaticMix),vec4<f32>(worldGrad*tc,gradTime),
							vec4<f32>(select(vec3<f32>(0.0),worldGrad*wave,cfg.modelMode==0u),
								alpha/f32(cfg.pixelCount)),vec4<f32>(gradLogScale,length(barMu)),gradRotation,
							vec4<f32>(gradColor,gradOpacity));
					}
				}
			}
		}
		gradientScratch[lane]=gradient;workgroupBarrier();
		var stride=${laneCount / 2}u;loop{
			if(lane<stride){gradientScratch[lane]=add_splat(gradientScratch[lane],gradientScratch[lane+stride]);}
			workgroupBarrier();if(stride==1u){break;}stride/=2u;
		}
		if(lane==0u&&pairValid){accumulate_splat(id,gradientScratch[0]);}
	}`;
}

function projectedGradientVjpWgsl(projectionType = "Projection") {
	const packed = projectionType === "PackedCompactProjectionVjp";
	const compact = packed || projectionType === "CompactProjectionVjp";
	const cameraArgument = compact ? ",camera:Camera" : "";
	const packedProjectionUnpack = `
			let sparse01=unpack2x16float(proj.packed0.x);
			let sparse23=unpack2x16float(proj.packed0.y);
			let basisVariance0=vec4<f32>(
				unpack2x16float(proj.packed0.z),
				unpack2x16float(proj.packed0.w));
			let basisVariance1=vec4<f32>(
				unpack2x16float(proj.packed1.x),
				unpack2x16float(proj.packed1.y));
			let basisVariance2=vec4<f32>(
				unpack2x16float(proj.packed1.z),
				unpack2x16float(proj.packed1.w));
			let sparse=vec4<f32>(sparse01,sparse23);
			let j0=vec3<f32>(sparse.x,0.0,sparse.y);
			let j1=vec3<f32>(0.0,sparse.z,sparse.w);
			let basis=mat3x3<f32>(
				basisVariance0.xyz,
				basisVariance1.xyz,
				basisVariance2.xyz);
			let normalizedVariances=vec3<f32>(
				basisVariance0.w,
				basisVariance1.w,
				basisVariance2.w);
			let variances=normalizedVariances
				*max(cfg.geometryScale*cfg.geometryScale,1e-12);`;
	const compactProjectionUnpack = `
			let sparse=proj.jacobianSparse;
			let j0=vec3<f32>(sparse.x,0.0,sparse.y);
			let j1=vec3<f32>(0.0,sparse.z,sparse.w);
			let basis=mat3x3<f32>(
				proj.basisVariance0.xyz,
				proj.basisVariance1.xyz,
				proj.basisVariance2.xyz);
			let variances=vec3<f32>(
				proj.basisVariance0.w,
				proj.basisVariance1.w,
				proj.basisVariance2.w);`;
	const sharedCameraUnpack = `
			// Camera rows are shared by every splat in the step. Rebuild the
			// column-major world-to-camera rotation here instead of storing the
			// same 48 bytes in every cold VJP packet.
			let cameraRotation=mat3x3<f32>(
				vec3<f32>(camera.row0.x,camera.row1.x,camera.row2.x),
				vec3<f32>(camera.row0.y,camera.row1.y,camera.row2.y),
				vec3<f32>(camera.row0.z,camera.row1.z,camera.row2.z));`;
	const projectionUnpack = compact ? `
			${packed ? packedProjectionUnpack : compactProjectionUnpack}
			${sharedCameraUnpack}`
		: `
			let j0=proj.jacobian0.xyz;let j1=proj.jacobian1.xyz;
			let basis=mat3x3<f32>(proj.basis0.xyz,proj.basis1.xyz,proj.basis2.xyz);
			let variances=proj.variancesPad.xyz;
			let cameraRotation=mat3x3<f32>(proj.camera0.xyz,proj.camera1.xyz,proj.camera2.xyz);`;
	return `
	struct ProjectedGradient{
		screen0:vec4<f32>,
		screen1:vec4<f32>,
		colorPad:vec4<f32>,
	};
	fn zero_projected_gradient()->ProjectedGradient{
		return ProjectedGradient(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));
	}
	fn add_projected_gradient(a:ProjectedGradient,b:ProjectedGradient)->ProjectedGradient{
		return ProjectedGradient(a.screen0+b.screen0,a.screen1+b.screen1,a.colorPad+b.colorPad);
	}
	// The raster pass accumulates derivatives of the projected mean, conic,
	// temporal peak, color, and diagnostics. Projection is fixed for a step,
	// so its VJP is linear: applying it once after the sum is mathematically
	// equivalent to applying it per pixel/pair, with much less repeated work.
	fn projected_to_splat_gradient(
		p:Splat,proj:${projectionType},g:ProjectedGradient${cameraArgument}
	)->Splat{
		let barMu=g.screen0.xy;
		let barC00=g.screen0.z;let barC01=g.screen0.w;let barC11=g.screen1.x;
		var worldGrad=vec3<f32>(0.0);var gradLogScale=vec3<f32>(0.0);
		var gradRotation=vec4<f32>(0.0);
		if(proj.cameraPointValid.w>=0.5){
			${projectionUnpack}
			let barSigma=barC00*outer3(j0,j0)
				+barC01*(outer3(j0,j1)+outer3(j1,j0))+barC11*outer3(j1,j1);
			let sigmaCamera=variances.x*outer3(basis[0],basis[0])
				+variances.y*outer3(basis[1],basis[1])
				+variances.z*outer3(basis[2],basis[2]);
			let sigmaJ0=sigmaCamera*j0;let sigmaJ1=sigmaCamera*j1;
			let barJ0=2.0*(barC00*sigmaJ0+barC01*sigmaJ1);
			let barJ1=2.0*(barC01*sigmaJ0+barC11*sigmaJ1);
			let cp=proj.cameraPointValid.xyz;let invZ=1.0/cp.z;
			let horizontalFocal=j0.x*cp.z;
			let verticalFocal=j1.y*cp.z;
			let cameraGrad=vec3<f32>(
				barMu.x*horizontalFocal*invZ-barJ0.z*horizontalFocal*invZ*invZ,
				barMu.y*verticalFocal*invZ-barJ1.z*verticalFocal*invZ*invZ,
				-barMu.x*horizontalFocal*cp.x*invZ*invZ
					-barMu.y*verticalFocal*cp.y*invZ*invZ
					-barJ0.x*horizontalFocal*invZ*invZ
					+barJ0.z*2.0*horizontalFocal*cp.x*invZ*invZ*invZ
					-barJ1.y*verticalFocal*invZ*invZ
					+barJ1.z*2.0*verticalFocal*cp.y*invZ*invZ*invZ);
			worldGrad=transpose(cameraRotation)*cameraGrad;
			for(var axis=0u;axis<3u;axis++){
				let column=basis[axis];
				gradLogScale[axis]=2.0*variances[axis]*dot(column,barSigma*column);
			}
			let barBasis=mat3x3<f32>(
				2.0*variances.x*(barSigma*basis[0]),
				2.0*variances.y*(barSigma*basis[1]),
				2.0*variances.z*(barSigma*basis[2]));
			let barRotation=transpose(cameraRotation)*barBasis;
			let q=safe_quaternion(p.rotation);
			let h00=barRotation[0].x;let h01=barRotation[1].x;let h02=barRotation[2].x;
			let h10=barRotation[0].y;let h11=barRotation[1].y;let h12=barRotation[2].y;
			let h20=barRotation[0].z;let h21=barRotation[1].z;let h22=barRotation[2].z;
			let normalizedQuatGrad=vec4<f32>(
				-4.0*q.x*(h11+h22)+2.0*q.y*(h01+h10)+2.0*q.z*(h02+h20)+2.0*q.w*(h21-h12),
				-4.0*q.y*(h00+h22)+2.0*q.x*(h01+h10)+2.0*q.z*(h12+h21)+2.0*q.w*(h02-h20),
				-4.0*q.z*(h00+h11)+2.0*q.x*(h02+h20)+2.0*q.y*(h12+h21)+2.0*q.w*(h10-h01),
				2.0*q.z*(h10-h01)+2.0*q.y*(h02-h20)+2.0*q.x*(h21-h12));
			let rawNorm2=dot(p.rotation,p.rotation);
			if(rawNorm2>1e-16){
				gradRotation=(normalizedQuatGrad-q*dot(q,normalizedQuatGrad))*inverseSqrt(rawNorm2);
			}
		}
		let staticWarmup=cfg.staticWarmup!=0u;let t=frame_time(cfg);
		let tc=select(t*2.0-1.0,0.0,staticWarmup);
		let wave=select(sin(t*6.28318530718),0.0,staticWarmup);
		let sigma=clamp(cfg.temporalSigma,0.12,0.36);
		let staticMix=clamp(p.centerStatic.w,0.0,1.0);
		let temporalFloor=clamp(sigma*0.30,0.035,0.12);
		let timeDelta=t-clamp(p.velocityTime.w,0.0,1.0);
		let dynamicGate=temporalFloor+(1.0-temporalFloor)
			*exp(-0.5*timeDelta*timeDelta/(sigma*sigma));
		let opacity=sigmoid(p.colorOpacity.w);
		let timeWeight=select(dynamicGate,1.0,staticWarmup);
		let dynamicCore=max(0.0,(timeWeight-staticMix
			-temporalFloor*(1.0-staticMix))/max(1e-6,1.0-staticMix));
		let gPeak=g.screen1.y;
		let gradTime=select(gPeak*opacity*(1.0-staticMix)*dynamicCore
			*(t-p.velocityTime.w)/(sigma*sigma),0.0,staticWarmup);
		let gradStaticMix=select(gPeak*opacity*(1.0-dynamicGate),0.0,staticWarmup);
		let gradOpacity=gPeak*timeWeight*opacity*(1.0-opacity);
		return Splat(
			vec4<f32>(worldGrad,gradStaticMix),
			vec4<f32>(worldGrad*tc,gradTime),
			vec4<f32>(select(vec3<f32>(0.0),worldGrad*wave,cfg.modelMode==0u),g.screen1.z),
			vec4<f32>(gradLogScale,g.screen1.w),
			gradRotation,
			vec4<f32>(g.colorPad.xyz,gradOpacity));
	}`;
}

function stagedBackwardWgsl(
	checkpointPrecision,
	checkpointOrder,
	sharePairPacket = false,
	tileSize = DEFAULT_TILE_SIZE,
	projectionLayout = TILED_PROJECTION_LAYOUTS.MONOLITHIC,
) {
	const laneCount = tileSize * tileSize;
	const projectionType = projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT
		? "RasterProjection" : "Projection";
	const sharedPairDeclarations = sharePairPacket ? `
	var<workgroup> sharedPairMeta:vec4<u32>;
	var<workgroup> sharedPairProjection:${projectionType};
	var<workgroup> sharedPairColorOpacity:vec4<f32>;` : "";
	const pairSetup = sharePairPacket ? `
		if(lane==0u&&pairValid){
			let sharedSlot=pairData[cfg.pairCapacity+pair];
			let sharedId=pairData[sharedSlot];
			sharedPairMeta=vec4<u32>(
				sharedSlot,
				sharedSlot/cfg.tileCapacity,
				sharedSlot%cfg.tileCapacity,
				sharedId);
			sharedPairProjection=projections[sharedId];
			sharedPairColorOpacity=params[sharedId].colorOpacity;
		}
		workgroupBarrier();
		if(pairValid){
			let slot=sharedPairMeta.x;let tile=sharedPairMeta.y;
			let rank=sharedPairMeta.z;id=sharedPairMeta.w;
			let proj=sharedPairProjection;
			let currentColorOpacity=sharedPairColorOpacity;` : `
		if(pairValid){
			let slot=pairData[cfg.pairCapacity+pair];let tile=slot/cfg.tileCapacity;
			let rank=slot%cfg.tileCapacity;id=pairData[slot];
			let proj=projections[id];
			let currentColorOpacity=params[id].colorOpacity;`;
	return `${CONFIG_WGSL}
	${projectedGradientVjpWgsl()}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> projections:array<${projectionType}>;
	@group(0) @binding(3) var<storage,read> pairData:array<u32>;
	@group(0) @binding(4) var<storage,read> rendered:array<vec4<f32>>;
	${checkpointBackwardWgsl(checkpointPrecision, checkpointOrder)}
	@group(0) @binding(6) var<storage,read> pixelGrad:array<vec4<f32>>;
	@group(0) @binding(7) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	@group(0) @binding(8) var<storage,read_write> counters:array<atomic<u32>>;
	var<workgroup> gradientScratch:array<ProjectedGradient,${laneCount}>;
	${sharedPairDeclarations}
	fn atomic_add_f32(index:u32,value:f32){
		if(value==0.0){return;}
		var oldBits=atomicLoad(&gradientAtoms[index]);
		loop{
			let newBits=bitcast<u32>(bitcast<f32>(oldBits)+value);
			let result=atomicCompareExchangeWeak(&gradientAtoms[index],oldBits,newBits);
			if(result.exchanged){break;}
			oldBits=result.old_value;
		}
	}
	fn accumulate_projected(id:u32,gradient:ProjectedGradient){
		let base=id*${PROJECTED_GRADIENT_FLOATS}u;
		for(var component=0u;component<4u;component++){
			atomic_add_f32(base+component,gradient.screen0[component]);
			atomic_add_f32(base+4u+component,gradient.screen1[component]);
			atomic_add_f32(base+8u+component,gradient.colorPad[component]);
		}
	}
	fn alpha_at(proj:${projectionType},point:vec2<f32>)->f32{
		let d=point-proj.screenConic0.xy;
		let q=proj.screenConic0.z*d.x*d.x+2.0*proj.screenConic0.w*d.x*d.y
			+proj.conicDepthAlpha.x*d.y*d.y;
		if(q<0.0||q>9.0){return 0.0;}
		let raw=proj.conicDepthAlpha.z*proj.conicDepthAlpha.w*exp(-0.5*q);
		return select(0.0,min(0.99,raw),raw>=cfg.alphaThreshold);
	}
	@compute @workgroup_size(${tileSize},${tileSize})
	fn pair_backward(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
		let lane=lid.y*${tileSize}u+lid.x;
		let pair=wid.y*${MAX_WORKGROUPS_PER_DIMENSION}u+wid.x;
		let pairValid=pair<atomicLoad(&counters[0]);
		var id=0u;var gradient=zero_projected_gradient();
		${pairSetup}
			let tileX=tile%cfg.tilesX;let tileY=tile/cfg.tilesX;
			let x=tileX*cfg.tileSize+lid.x;let y=tileY*cfg.tileSize+lid.y;
			if(x<cfg.width&&y<cfg.height){
				let pixel=y*cfg.width+x;
				if(rank<u32(pixelGrad[pixel].w)){
					let point=vec2<f32>((f32(x)+0.5)/f32(cfg.height),
						(f32(y)+0.5)/f32(cfg.height));
					let block=rank/cfg.checkpointStride;
					let checkpoint=read_checkpoint(checkpoint_index(pixel,block));
					var before=checkpoint.xyz;var transmittance=checkpoint.w;
					for(var replay=block*cfg.checkpointStride;replay<rank;replay++){
						let prior=pairData[tile*cfg.tileCapacity+replay];
						let alpha=alpha_at(projections[prior],point);
						before+=transmittance*alpha*params[prior].colorOpacity.xyz;
						transmittance*=1.0-alpha;
					}
					let d=point-proj.screenConic0.xy;
					let qform=proj.screenConic0.z*d.x*d.x
						+2.0*proj.screenConic0.w*d.x*d.y+proj.conicDepthAlpha.x*d.y*d.y;
					if(qform>=0.0&&qform<=9.0&&transmittance>cfg.transmittanceThreshold){
						let gaussian=exp(-0.5*qform);
						let rawAlpha=proj.conicDepthAlpha.z*proj.conicDepthAlpha.w*gaussian;
						let alpha=select(0.0,min(0.99,rawAlpha),rawAlpha>=cfg.alphaThreshold);
						let denominator=transmittance*(1.0-alpha);
						let behind=select(vec3<f32>(0.0),
							(rendered[pixel].xyz-before-transmittance*alpha*currentColorOpacity.xyz)
								/max(denominator,1e-8),
							denominator>1e-8);
						let imageGrad=pixelGrad[pixel].xyz;
						let alphaGrad=dot(imageGrad,transmittance*(currentColorOpacity.xyz-behind));
						let clampGate=select(0.0,1.0,
							rawAlpha<0.99&&rawAlpha>=cfg.alphaThreshold);
						let barQform=-0.5*alphaGrad*rawAlpha*clampGate;
						let conicDelta=vec2<f32>(
							proj.screenConic0.z*d.x+proj.screenConic0.w*d.y,
							proj.screenConic0.w*d.x+proj.conicDepthAlpha.x*d.y);
						let barMu=-2.0*barQform*conicDelta;
						let barC00=-barQform*conicDelta.x*conicDelta.x;
						let barC01=-barQform*conicDelta.x*conicDelta.y;
						let barC11=-barQform*conicDelta.y*conicDelta.y;
						gradient=ProjectedGradient(
							vec4<f32>(barMu,barC00,barC01),
							vec4<f32>(barC11,clampGate*alphaGrad*gaussian,
								alpha/f32(cfg.pixelCount),length(barMu)),
							vec4<f32>(imageGrad*transmittance*alpha,0.0));
					}
				}
			}
		}
		gradientScratch[lane]=gradient;workgroupBarrier();
		var stride=${laneCount / 2}u;loop{
			if(lane<stride){
				gradientScratch[lane]=add_projected_gradient(
					gradientScratch[lane],gradientScratch[lane+stride]);
			}
			workgroupBarrier();if(stride==1u){break;}stride/=2u;
		}
		if(lane==0u&&pairValid){accumulate_projected(id,gradientScratch[0]);}
	}`;
}

function checkpointBlockBackwardWgsl(
	checkpointPrecision,
	checkpointOrder,
	tileSize = DEFAULT_TILE_SIZE,
	projectionLayout = TILED_PROJECTION_LAYOUTS.MONOLITHIC,
) {
	const laneCount = tileSize * tileSize;
	const projectionType = projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT
		? "RasterProjection" : "Projection";
	return `${CONFIG_WGSL}
	${projectedGradientVjpWgsl()}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> projections:array<${projectionType}>;
	@group(0) @binding(3) var<storage,read> pairData:array<u32>;
	@group(0) @binding(4) var<storage,read> rendered:array<vec4<f32>>;
	${checkpointBackwardWgsl(checkpointPrecision, checkpointOrder)}
	@group(0) @binding(6) var<storage,read> pixelGrad:array<vec4<f32>>;
	@group(0) @binding(7) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	@group(0) @binding(9) var<storage,read_write> tileCounts:array<atomic<u32>>;
	var<workgroup> gradientScratch:array<ProjectedGradient,${laneCount}>;
	var<workgroup> blockMeta:vec4<u32>;
	var<workgroup> currentProjection:${projectionType};
	var<workgroup> currentColorOpacity:vec4<f32>;
	var<workgroup> currentId:u32;
	fn atomic_add_f32(index:u32,value:f32){
		if(value==0.0){return;}
		var oldBits=atomicLoad(&gradientAtoms[index]);
		loop{
			let newBits=bitcast<u32>(bitcast<f32>(oldBits)+value);
			let result=atomicCompareExchangeWeak(&gradientAtoms[index],oldBits,newBits);
			if(result.exchanged){break;}
			oldBits=result.old_value;
		}
	}
	fn accumulate_projected(id:u32,gradient:ProjectedGradient){
		let base=id*${PROJECTED_GRADIENT_FLOATS}u;
		for(var component=0u;component<4u;component++){
			atomic_add_f32(base+component,gradient.screen0[component]);
			atomic_add_f32(base+4u+component,gradient.screen1[component]);
			atomic_add_f32(base+8u+component,gradient.colorPad[component]);
		}
	}
	@compute @workgroup_size(${tileSize},${tileSize})
	fn block_backward(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
		let lane=lid.y*${tileSize}u+lid.x;
		let block=wid.y*${MAX_WORKGROUPS_PER_DIMENSION}u+wid.x;
		if(lane==0u){
			let startSlot=pairData[cfg.pairCapacity+block];
			let tile=startSlot/cfg.tileCapacity;
			blockMeta=vec4<u32>(
				startSlot,
				tile,
				startSlot%cfg.tileCapacity,
				min(atomicLoad(&tileCounts[tile]),cfg.tileCapacity));
		}
		workgroupBarrier();
		let tile=blockMeta.y;let startRank=blockMeta.z;let tileCount=blockMeta.w;
			let tileX=tile%cfg.tilesX;let tileY=tile/cfg.tilesX;
			let x=tileX*cfg.tileSize+lid.x;let y=tileY*cfg.tileSize+lid.y;
			let pixelValid=x<cfg.width&&y<cfg.height;
			var pixel=0u;var stopRank=0u;
			var point=vec2<f32>(0.0);var imageGrad=vec3<f32>(0.0);
			var renderedColor=vec3<f32>(0.0);
			var before=vec3<f32>(0.0);var transmittance=0.0;
			if(pixelValid){
				pixel=y*cfg.width+x;
				stopRank=u32(pixelGrad[pixel].w);
				point=vec2<f32>((f32(x)+0.5)/f32(cfg.height),
					(f32(y)+0.5)/f32(cfg.height));
				imageGrad=pixelGrad[pixel].xyz;
				renderedColor=rendered[pixel].xyz;
				if(startRank<stopRank){
					let checkpoint=read_checkpoint(
						checkpoint_index(pixel,startRank/cfg.checkpointStride));
					before=checkpoint.xyz;transmittance=checkpoint.w;
				}
			}
			// Owning a complete checkpoint block turns the old triangular
			// prefix replay into one source-over walk. The reduction and atomic
			// contract are unchanged; only redundant checkpoint/projection work
			// and dispatches disappear.
			for(var offset=0u;offset<cfg.checkpointStride;offset++){
				let rank=startRank+offset;let rankValid=rank<tileCount;
				if(lane==0u&&rankValid){
					currentId=pairData[tile*cfg.tileCapacity+rank];
					currentProjection=projections[currentId];
					currentColorOpacity=params[currentId].colorOpacity;
				}
				workgroupBarrier();
				var gradient=zero_projected_gradient();
				let contributes=rankValid&&pixelValid&&rank<stopRank;
				if(contributes){
					let d=point-currentProjection.screenConic0.xy;
					let qform=currentProjection.screenConic0.z*d.x*d.x
						+2.0*currentProjection.screenConic0.w*d.x*d.y
						+currentProjection.conicDepthAlpha.x*d.y*d.y;
					var alpha=0.0;
					if(qform>=0.0&&qform<=9.0&&transmittance>cfg.transmittanceThreshold){
						let gaussian=exp(-0.5*qform);
						let rawAlpha=currentProjection.conicDepthAlpha.z
							*currentProjection.conicDepthAlpha.w*gaussian;
						alpha=select(0.0,min(0.99,rawAlpha),rawAlpha>=cfg.alphaThreshold);
						let denominator=transmittance*(1.0-alpha);
						let behind=select(vec3<f32>(0.0),
							(renderedColor-before-transmittance*alpha*currentColorOpacity.xyz)
								/max(denominator,1e-8),
							denominator>1e-8);
						let alphaGrad=dot(
							imageGrad,
							transmittance*(currentColorOpacity.xyz-behind));
						let clampGate=select(0.0,1.0,
							rawAlpha<0.99&&rawAlpha>=cfg.alphaThreshold);
						let barQform=-0.5*alphaGrad*rawAlpha*clampGate;
						let conicDelta=vec2<f32>(
							currentProjection.screenConic0.z*d.x
								+currentProjection.screenConic0.w*d.y,
							currentProjection.screenConic0.w*d.x
								+currentProjection.conicDepthAlpha.x*d.y);
						let barMu=-2.0*barQform*conicDelta;
						gradient=ProjectedGradient(
							vec4<f32>(
								barMu,
								-barQform*conicDelta.x*conicDelta.x,
								-barQform*conicDelta.x*conicDelta.y),
							vec4<f32>(
								-barQform*conicDelta.y*conicDelta.y,
								clampGate*alphaGrad*gaussian,
								alpha/f32(cfg.pixelCount),
								length(barMu)),
							vec4<f32>(imageGrad*transmittance*alpha,0.0));
					}
					before+=transmittance*alpha*currentColorOpacity.xyz;
					transmittance*=1.0-alpha;
				}
				gradientScratch[lane]=gradient;workgroupBarrier();
				var stride=${laneCount / 2}u;loop{
					if(lane<stride){
						gradientScratch[lane]=add_projected_gradient(
							gradientScratch[lane],gradientScratch[lane+stride]);
					}
					workgroupBarrier();if(stride==1u){break;}stride/=2u;
				}
				if(lane==0u&&rankValid){
					accumulate_projected(currentId,gradientScratch[0]);
				}
				workgroupBarrier();
			}
	}`;
}

function updateWgsl(
	backwardMode = TILED_BACKWARD_MODES.DIRECT_3D,
	projectionLayout = TILED_PROJECTION_LAYOUTS.MONOLITHIC,
	projectionVjpPrecision = TILED_PROJECTION_VJP_PRECISIONS.F32,
) {
	const staged = backwardMode === TILED_BACKWARD_MODES.STAGED_PROJECT_3D;
	const compactProjection = projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT;
	const projectionType = compactProjection
		? projectionVjpPrecision === TILED_PROJECTION_VJP_PRECISIONS.PACKED_F16
			? "PackedCompactProjectionVjp" : "CompactProjectionVjp"
		: "Projection";
	const gradientLoader = staged ? `
	fn load_projected_gradient(id:u32)->ProjectedGradient{
		let base=id*${PROJECTED_GRADIENT_FLOATS}u;
		return ProjectedGradient(
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+1u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+2u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+3u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+4u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+5u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+6u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+7u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+8u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+9u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+10u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+11u])))
		);
	}` : `
	fn load_gradient(id:u32)->Splat{
		let base=id*${DIRECT_GRADIENT_FLOATS}u;
		return Splat(
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+1u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+2u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+3u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+4u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+5u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+6u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+7u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+8u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+9u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+10u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+11u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+12u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+13u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+14u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+15u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+16u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+17u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+18u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+19u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+20u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+21u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+22u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+23u])))
		);
	}`;
	return `${CONFIG_WGSL}
	${staged ? projectedGradientVjpWgsl(projectionType) : ""}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> paramsIn:array<Splat>;
	@group(0) @binding(2) var<storage,read_write> paramsOut:array<Splat>;
	@group(0) @binding(3) var<storage,read_write> firstMoment:array<Splat>;
	@group(0) @binding(4) var<storage,read_write> secondMoment:array<Splat>;
	@group(0) @binding(5) var<storage,read_write> splatStats:array<vec4<f32>>;
	@group(0) @binding(6) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	${staged ? `@group(0) @binding(7) var<storage,read> projections:array<${projectionType}>;` : ""}
	${staged && compactProjection
		? "@group(0) @binding(8) var<storage,read> cameras:array<Camera>;" : ""}
	${gradientLoader}
	@compute @workgroup_size(64)
	fn reduce_update(@builtin(global_invocation_id) gid:vec3<u32>){
		let i=gid.x;if(i>=cfg.activeSplatCount){return;}
		var gradient=${staged
		? `projected_to_splat_gradient(
			paramsIn[i],projections[i],load_projected_gradient(i)${compactProjection
				? ",cameras[cfg.viewIndex]" : ""}
		)`
		: "load_gradient(i)"};
		let meanAlpha=gradient.harmonicPad.w;gradient.harmonicPad.w=0.0;
		let screenGradient=gradient.logScalePad.w;gradient.logScalePad.w=0.0;
		var p=paramsIn[i];var m=firstMoment[i];var v=secondMoment[i];
		m.centerStatic=cfg.beta1*m.centerStatic+(1.0-cfg.beta1)*gradient.centerStatic;
		m.velocityTime=cfg.beta1*m.velocityTime+(1.0-cfg.beta1)*gradient.velocityTime;
		m.harmonicPad=cfg.beta1*m.harmonicPad+(1.0-cfg.beta1)*gradient.harmonicPad;
		m.logScalePad=cfg.beta1*m.logScalePad+(1.0-cfg.beta1)*gradient.logScalePad;
		m.rotation=cfg.beta1*m.rotation+(1.0-cfg.beta1)*gradient.rotation;
		m.colorOpacity=cfg.beta1*m.colorOpacity+(1.0-cfg.beta1)*gradient.colorOpacity;
		v.centerStatic=cfg.beta2*v.centerStatic+(1.0-cfg.beta2)*gradient.centerStatic*gradient.centerStatic;
		v.velocityTime=cfg.beta2*v.velocityTime+(1.0-cfg.beta2)*gradient.velocityTime*gradient.velocityTime;
		v.harmonicPad=cfg.beta2*v.harmonicPad+(1.0-cfg.beta2)*gradient.harmonicPad*gradient.harmonicPad;
		v.logScalePad=cfg.beta2*v.logScalePad+(1.0-cfg.beta2)*gradient.logScalePad*gradient.logScalePad;
		v.rotation=cfg.beta2*v.rotation+(1.0-cfg.beta2)*gradient.rotation*gradient.rotation;
		v.colorOpacity=cfg.beta2*v.colorOpacity+(1.0-cfg.beta2)*gradient.colorOpacity*gradient.colorOpacity;
		firstMoment[i]=m;secondMoment[i]=v;let adamStep=f32(cfg.step+1u);
		let mc=max(1e-6,1.0-pow(cfg.beta1,adamStep));let vc=max(1e-6,1.0-pow(cfg.beta2,adamStep));
		let posUpdate=(m.centerStatic/mc)/(sqrt(v.centerStatic/vc)+vec4<f32>(cfg.adamEpsilon));
		let velocityUpdate=(m.velocityTime/mc)/(sqrt(v.velocityTime/vc)+vec4<f32>(cfg.adamEpsilon));
		let harmonicUpdate=(m.harmonicPad/mc)/(sqrt(v.harmonicPad/vc)+vec4<f32>(cfg.adamEpsilon));
		let scaleUpdate=(m.logScalePad/mc)/(sqrt(v.logScalePad/vc)+vec4<f32>(cfg.adamEpsilon));
		let rotationUpdate=(m.rotation/mc)/(sqrt(v.rotation/vc)+vec4<f32>(cfg.adamEpsilon));
		let colorUpdate=(m.colorOpacity/mc)/(sqrt(v.colorOpacity/vc)+vec4<f32>(cfg.adamEpsilon));
		p.centerStatic=vec4<f32>(p.centerStatic.xyz-cfg.lrPosition*posUpdate.xyz,
			clamp(p.centerStatic.w-cfg.lrMotion*posUpdate.w,0.0,1.0));
		p.velocityTime=vec4<f32>(clamp(p.velocityTime.xyz-cfg.lrMotion*velocityUpdate.xyz,
			vec3<f32>(-2.0*cfg.geometryScale),vec3<f32>(2.0*cfg.geometryScale)),
			clamp(p.velocityTime.w-cfg.lrMotion*velocityUpdate.w,0.0,1.0));
		p.harmonicPad=vec4<f32>(clamp(p.harmonicPad.xyz-cfg.lrMotion*harmonicUpdate.xyz,
			vec3<f32>(-1.5*cfg.geometryScale),vec3<f32>(1.5*cfg.geometryScale)),p.harmonicPad.w);
		var nextLogScale=clamp(p.logScalePad.xyz-${SCALE_LR_FROM_COLOR}*cfg.lrColor*scaleUpdate.xyz,
			vec3<f32>(log(cfg.minScale)),vec3<f32>(log(cfg.maxScale)));
		let meanLog=(nextLogScale.x+nextLogScale.y+nextLogScale.z)/3.0;
		// Center the trust region in log scale so the ratio bound is symmetric
		// across axes and does not select a preferred world direction.
		let halfLogAspect=0.5*log(${MAX_SCALE_ASPECT_RATIO}.0);
		nextLogScale=clamp(nextLogScale,vec3<f32>(meanLog-halfLogAspect),vec3<f32>(meanLog+halfLogAspect));
		p.logScalePad=vec4<f32>(nextLogScale,p.logScalePad.w);
		let rotationTrial=p.rotation-${ROTATION_LR_FROM_MOTION}*cfg.lrMotion*rotationUpdate;
		let rotationNorm2=dot(rotationTrial,rotationTrial);
		p.rotation=select(vec4<f32>(0.0,0.0,0.0,1.0),rotationTrial*inverseSqrt(max(rotationNorm2,1e-16)),rotationNorm2>1e-16);
		p.colorOpacity=vec4<f32>(clamp(p.colorOpacity.xyz-cfg.lrColor*colorUpdate.xyz,
			vec3<f32>(0.0),vec3<f32>(${MAX_SPLAT_COLOR}.0)),
			clamp(p.colorOpacity.w-cfg.lrOpacity*colorUpdate.w,-12.0,3.0));
		paramsOut[i]=p;let observed=vec4<f32>(screenGradient,meanAlpha,
			abs(gradient.colorOpacity.w),length(gradient.velocityTime.xyz));
		splatStats[i]=cfg.statDecay*splatStats[i]+(1.0-cfg.statDecay)*observed;
	}`;
}

function densityWgsl(gradientFloats) {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read_write> params:array<Splat>;
	@group(0) @binding(2) var<storage,read_write> firstMoment:array<Splat>;
	@group(0) @binding(3) var<storage,read_write> secondMoment:array<Splat>;
	@group(0) @binding(4) var<storage,read_write> splatStats:array<vec4<f32>>;
	@group(0) @binding(5) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	@group(0) @binding(6) var<storage,read_write> counters:array<atomic<u32>>;
	fn zero_splat()->Splat{
		return Splat(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),
			vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));
	}
	@compute @workgroup_size(1)
	fn activate_prefix_splits(){
		let capacity=cfg.splatCount;
		let activeCount=atomicLoad(&counters[7]);
		if(activeCount>=capacity){return;}
		let splitCount=min(${DENSITY_SPLITS_PER_DISPATCH}u,capacity-activeCount);
		var parents:array<u32,${DENSITY_SPLITS_PER_DISPATCH}>;
		for(var slot=0u;slot<splitCount;slot++){
			var best=-1.0;var bestIndex=0xffffffffu;
			for(var i=0u;i<activeCount;i++){
				var used=false;
				for(var prior=0u;prior<slot;prior++){used=used||parents[prior]==i;}
				let score=splatStats[i].x+4.0*splatStats[i].y+splatStats[i].w;
				if(!used&&score>best){best=score;bestIndex=i;}
			}
			parents[slot]=bestIndex;
		}
		for(var slot=0u;slot<splitCount;slot++){
			let childIndex=activeCount+slot;
			let parentIndex=parents[slot];
			var parent=params[parentIndex];
			var child=parent;
			var axis=0u;
			if(parent.logScalePad.y>parent.logScalePad.x){axis=1u;}
			if(parent.logScalePad.z>parent.logScalePad[axis]){axis=2u;}
			let rotation=quaternion_matrix(parent.rotation);
			let offset=rotation[axis]*exp(parent.logScalePad[axis])*0.28;
			parent.centerStatic=vec4<f32>(
				parent.centerStatic.xyz-offset,max(0.0,parent.centerStatic.w-0.04));
			child.centerStatic=vec4<f32>(
				child.centerStatic.xyz+offset,max(0.0,child.centerStatic.w-0.04));
			let shrink=vec3<f32>(log(0.80));
			parent.logScalePad=vec4<f32>(parent.logScalePad.xyz+shrink,0.0);
			child.logScalePad=vec4<f32>(child.logScalePad.xyz+shrink,0.0);
			if(splatStats[parentIndex].w>splatStats[parentIndex].x){
				parent.velocityTime.w=clamp(parent.velocityTime.w-0.035,0.0,1.0);
				child.velocityTime.w=clamp(child.velocityTime.w+0.035,0.0,1.0);
			}
			let opacity=clamp(sigmoid(parent.colorOpacity.w),1e-4,0.999);
			let halfOpacity=clamp(1.0-sqrt(1.0-opacity),1e-4,0.999);
			let splitLogit=log(halfOpacity/(1.0-halfOpacity));
			parent.colorOpacity.w=splitLogit;
			child.colorOpacity.w=splitLogit;
			params[parentIndex]=parent;
			params[childIndex]=child;
			firstMoment[parentIndex]=zero_splat();
			secondMoment[parentIndex]=zero_splat();
			firstMoment[childIndex]=zero_splat();
			secondMoment[childIndex]=zero_splat();
			splatStats[parentIndex]=splatStats[parentIndex]*0.5;
			splatStats[childIndex]=vec4<f32>(0.0);
			let gradientBase=childIndex*${gradientFloats}u;
			for(var component=0u;component<${gradientFloats}u;component++){
				atomicStore(&gradientAtoms[gradientBase+component],0u);
			}
		}
		atomicStore(&counters[7],activeCount+splitCount);
	}`;
}

function stagedGradientDebugWgsl(
	projectionLayout = TILED_PROJECTION_LAYOUTS.MONOLITHIC,
	projectionVjpPrecision = TILED_PROJECTION_VJP_PRECISIONS.F32,
) {
	const compactProjection = projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT;
	const projectionType = compactProjection
		? projectionVjpPrecision === TILED_PROJECTION_VJP_PRECISIONS.PACKED_F16
			? "PackedCompactProjectionVjp" : "CompactProjectionVjp"
		: "Projection";
	return `${CONFIG_WGSL}
	${projectedGradientVjpWgsl(projectionType)}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> projections:array<${projectionType}>;
	@group(0) @binding(3) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	@group(0) @binding(4) var<storage,read_write> fullGradients:array<Splat>;
	${compactProjection ? "@group(0) @binding(5) var<storage,read> cameras:array<Camera>;" : ""}
	fn load_projected_gradient(id:u32)->ProjectedGradient{
		let base=id*${PROJECTED_GRADIENT_FLOATS}u;
		return ProjectedGradient(
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+1u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+2u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+3u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+4u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+5u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+6u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+7u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+8u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+9u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+10u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+11u])))
		);
	}
	@compute @workgroup_size(64)
	fn materialize_gradient(@builtin(global_invocation_id) gid:vec3<u32>){
		let id=gid.x;if(id>=cfg.splatCount){return;}
		fullGradients[id]=projected_to_splat_gradient(
			params[id],projections[id],load_projected_gradient(id)${compactProjection
				? ",cameras[cfg.viewIndex]" : ""});
	}`;
}

async function checkedModule(device, name, code) {
	const module = device.createShaderModule({ label: name, code });
	const info = await module.getCompilationInfo();
	const errors = info.messages.filter((message) => message.type === "error")
		.map((message) => `${name}:${message.lineNum}:${message.linePos} ${message.message}`);
	if (errors.length) throw new Error(`WGSL compilation failed:\n${errors.join("\n")}`);
	return module;
}

export class DynamicSplatWebGpu3dTiledTrainer extends DynamicSplatWebGpu3dTrainer {
	constructor(canvas) {
		super(canvas);
		this.initialSplatCount = 1536;
		this.skipSampleGradientAllocation = true;
		this.backwardMode = TILED_BACKWARD_MODES.DIRECT_3D;
		this.gradientFloats = DIRECT_GRADIENT_FLOATS;
		this.tiledConfigBytes = new ArrayBuffer(TILED_CONFIG_BYTES);
		this.activeSplatCount = this.initialSplatCount;
		this.projectionVjpPrecision = TILED_PROJECTION_VJP_PRECISIONS.F32;
		this.compactTargetFrames = false;
		this.targetDecodePending = false;
	}

	targetBufferByteLength(dataset) {
		return rgbaFloatFrameBytes(dataset);
	}

	uploadTargetPage(target, viewIndex, frameIndex, { staticWarmup = false } = {}) {
		const pixelCount = this.dataset.width * this.dataset.height;
		const sourcePixelOffset = staticWarmup
			? viewIndex * pixelCount
			: (viewIndex * this.dataset.frameCount + frameIndex) * pixelCount;
		const pageKey = staticWarmup ? `background:${viewIndex}` : `frame:${viewIndex}:${frameIndex}`;
		if (this.targetPageKey !== pageKey) {
			const sourceElementOffset = sourcePixelOffset * 4;
			const source = staticWarmup ? this.dataset.backgrounds : this.dataset.frames;
			const page = source.subarray(sourceElementOffset, sourceElementOffset + pixelCount * 4);
			if (!staticWarmup && this.compactTargetFrames) {
				if (!this.buffers?.targetPacked) {
					throw new Error("Compact target upload requires the packed target-page buffer.");
				}
				this.device.queue.writeBuffer(this.buffers.targetPacked, 0, page);
				this.targetDecodePending = true;
			} else {
				this.device.queue.writeBuffer(target, 0, page);
				this.targetDecodePending = false;
			}
			this.targetPageKey = pageKey;
		}
		return sourcePixelOffset;
	}

	initializeTargetBuffer(target) {
		this.targetPageKey = null;
		if (this.compactTargetFrames) return;
		this.uploadTargetPage(target, this.trainViewIndices[0], 0);
	}

	async init(dataset, {
		splatCount = 1536,
		growthCapacity = null,
		tileCapacity = null,
		checkpointPrecision = DEFAULT_CHECKPOINT_PRECISION,
		checkpointStride = DEFAULT_CHECKPOINT_STRIDE,
		checkpointOrder = TILED_CHECKPOINT_ORDERS.PIXEL_MAJOR,
		staticWarmupSteps = 0,
		backwardMode = TILED_BACKWARD_MODES.DIRECT_3D,
		backwardGranularity = TILED_BACKWARD_GRANULARITIES.PAIR,
		sharePairPacket = false,
		tileSize = DEFAULT_TILE_SIZE,
		projectionLayout = TILED_PROJECTION_LAYOUTS.MONOLITHIC,
		projectionVjpPrecision = TILED_PROJECTION_VJP_PRECISIONS.F32,
		ssimLayout = TILED_SSIM_LAYOUTS.NAIVE_2D,
		profileGpu = false,
	} = {}) {
		this.initialSplatCount = splatCount;
		this.requestedTileCapacity = tileCapacity;
		this.checkpointPrecision = resolveCheckpointPrecision(checkpointPrecision);
		this.requestedCheckpointStride = resolveCheckpointStride(checkpointStride);
		this.checkpointOrder = resolveTiledCheckpointOrder(checkpointOrder);
		this.staticWarmupSteps = resolveStaticWarmupSteps(staticWarmupSteps);
		this.backwardMode = resolveTiledBackwardMode(backwardMode);
		this.backwardGranularity = resolveTiledBackwardGranularity(backwardGranularity);
		this.tileSize = resolveTiledTileSize(tileSize);
		this.projectionLayout = resolveTiledProjectionLayout(projectionLayout);
		const requestedProjectionVjpPrecision =
			resolveTiledProjectionVjpPrecision(projectionVjpPrecision);
		this.projectionVjpPrecision = this.projectionLayout
			=== TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT
			? requestedProjectionVjpPrecision : TILED_PROJECTION_VJP_PRECISIONS.F32;
		this.ssimLayout = resolveTiledSsimLayout(ssimLayout);
		if (this.backwardGranularity === TILED_BACKWARD_GRANULARITIES.CHECKPOINT_BLOCK
			&& this.backwardMode !== TILED_BACKWARD_MODES.STAGED_PROJECT_3D) {
			throw new RangeError("checkpoint-block backward requires staged-project3d gradients.");
		}
		if (this.projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT
			&& this.backwardMode !== TILED_BACKWARD_MODES.STAGED_PROJECT_3D) {
			throw new RangeError("split-compact projections require staged-project3d gradients.");
		}
		this.gradientFloats = this.backwardMode === TILED_BACKWARD_MODES.STAGED_PROJECT_3D
			? PROJECTED_GRADIENT_FLOATS : DIRECT_GRADIENT_FLOATS;
		this.sharePairPacket = Boolean(sharePairPacket)
			&& this.backwardMode === TILED_BACKWARD_MODES.STAGED_PROJECT_3D
			&& this.backwardGranularity === TILED_BACKWARD_GRANULARITIES.PAIR;
		this.requestTimestampQueries = Boolean(profileGpu);
		this.compactTargetFrames = resolveFrameBank(dataset).format === FRAME_BANK_FORMAT_RGBA8;
		const capacity = resolveTiledCapacity(splatCount, growthCapacity);
		await super.init(dataset, { splatCount: capacity, requiredWorkgroupStorageSize: 24576 });
		const bindGroupError = await this.tiledBindGroupValidation;
		if (bindGroupError) {
			throw new Error(`Tiled WebGPU bind-group validation failed: ${bindGroupError.message}`);
		}
		this.adapterName = `${this.adapterName} · tiled full-frame · ${this.backwardMode}`
			+ ` · ${this.backwardGranularity}`
			+ ` · ${this.checkpointOrder} tape`
			+ ` · ${this.projectionLayout} projection`
			+ ` · ${this.ssimLayout} SSIM`
			+ (this.sharePairPacket ? " · shared pair packet" : "");
	}

	async createPipelines() {
		await super.createPipelines();
		this.tileCapacity = resolveTileCapacity(this.splatCount, this.requestedTileCapacity);
		const checkpointBlocks = this.backwardGranularity
			=== TILED_BACKWARD_GRANULARITIES.CHECKPOINT_BLOCK;
		const backwardSource = checkpointBlocks
			? checkpointBlockBackwardWgsl(
				this.checkpointPrecision,
				this.checkpointOrder,
				this.tileSize,
				this.projectionLayout,
			)
			: this.backwardMode === TILED_BACKWARD_MODES.STAGED_PROJECT_3D
				? stagedBackwardWgsl(
					this.checkpointPrecision,
					this.checkpointOrder,
					this.sharePairPacket,
					this.tileSize,
					this.projectionLayout,
				)
				: backwardWgsl(
					this.checkpointPrecision,
					this.checkpointOrder,
					this.tileSize,
				);
		const separableSsim = this.ssimLayout === TILED_SSIM_LAYOUTS.SEPARABLE;
		const moduleSources = {
			targetDecode: targetDecodeWgsl(),
			clear: clearWgsl(this.gradientFloats),
			project: projectWgsl(this.projectionLayout, this.projectionVjpPrecision),
			sort: sortWgsl(this.tileCapacity, this.backwardGranularity, this.projectionLayout),
			finalize: finalizeWgsl(this.backwardGranularity),
			forward: forwardWgsl(
				this.checkpointPrecision,
				this.checkpointOrder,
				this.tileSize,
				this.projectionLayout,
			),
			metrics: metricsWgsl(),
			backward: backwardSource,
			update: updateWgsl(
				this.backwardMode,
				this.projectionLayout,
				this.projectionVjpPrecision,
			),
			density: densityWgsl(this.gradientFloats),
			...(separableSsim ? {
				ssimHorizontal: separableSsimHorizontalWgsl(),
				ssimVertical: separableSsimVerticalWgsl(),
				ssimGradientHorizontal: separableSsimGradientHorizontalWgsl(),
				ssimGradientVertical: separableSsimGradientVerticalWgsl(),
			} : {
				ssimStats: ssimStatsWgsl(),
				ssimGradient: ssimGradientWgsl(),
			}),
		};
		const modules = Object.fromEntries(await Promise.all(
			Object.entries(moduleSources).map(async ([name, source]) => [
				name,
				await checkedModule(this.device, `tiled-${name}`, source),
			]),
		));
		const debugModule = this.backwardMode === TILED_BACKWARD_MODES.STAGED_PROJECT_3D
			? await checkedModule(this.device, "tiled-staged-gradient-debug",
				stagedGradientDebugWgsl(
					this.projectionLayout,
					this.projectionVjpPrecision,
				))
			: null;
		const pipeline = (module, entryPoint) => this.device.createComputePipeline({
			label: `tiled-${entryPoint}`, layout: "auto", compute: { module, entryPoint },
		});
		this.device.pushErrorScope("validation");
		this.tiledPipelines = {
			targetDecode: pipeline(modules.targetDecode, "decode_target_page"),
			clear: pipeline(modules.clear, "clear_step"),
			project: pipeline(modules.project, "project_and_bin"),
			sort: pipeline(modules.sort, "sort_tiles"),
			finalize: pipeline(modules.finalize, "finalize_pairs"),
			forward: pipeline(modules.forward, "raster_forward"),
			metrics: pipeline(modules.metrics, "reduce_metrics"),
			backward: pipeline(modules.backward,
				checkpointBlocks ? "block_backward" : "pair_backward"),
			update: pipeline(modules.update, "reduce_update"),
			density: pipeline(modules.density, "activate_prefix_splits"),
			gradientDebug: debugModule ? pipeline(debugModule, "materialize_gradient") : null,
			...(separableSsim ? {
				ssimHorizontal: pipeline(modules.ssimHorizontal, "ssim_horizontal"),
				ssimVertical: pipeline(modules.ssimVertical, "ssim_vertical"),
				ssimGradientHorizontal: pipeline(
					modules.ssimGradientHorizontal,
					"ssim_gradient_horizontal",
				),
				ssimGradientVertical: pipeline(
					modules.ssimGradientVertical,
					"ssim_gradient_vertical",
				),
			} : {
				ssimStats: pipeline(modules.ssimStats, "ssim_stats"),
				ssimGradient: pipeline(modules.ssimGradient, "ssim_gradient"),
			}),
		};
		const pipelineError = await this.device.popErrorScope();
		if (pipelineError) throw new Error(`Tiled WebGPU pipeline validation failed: ${pipelineError.message}`);
	}

	createBuffers() {
		super.createBuffers();
		const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
		const makeBuffer = (size, bufferUsage = usage) => this.device.createBuffer({
			size: Math.max(4, Math.ceil(size / 4) * 4), usage: bufferUsage,
		});
		const active = makeInitialSplats(this.dataset, this.initialSplatCount);
		this.activeSplatCount = this.initialSplatCount;
		const initial = new Float32Array(this.splatCount * SPLAT_FLOATS);
		initial.set(active);
		for (let i = this.initialSplatCount; i < this.splatCount; i += 1) {
			const source = (i % this.initialSplatCount) * SPLAT_FLOATS;
			const base = i * SPLAT_FLOATS;
			initial.set(active.subarray(source, source + SPLAT_FLOATS), base);
			initial[base + 23] = -12;
		}
		this.initialParams = initial.slice();
		for (const params of this.buffers.params) this.device.queue.writeBuffer(params, 0, initial);
		this.tilesX = ceilDiv(this.dataset.width, this.tileSize);
		this.tilesY = ceilDiv(this.dataset.height, this.tileSize);
		this.tileCount = this.tilesX * this.tilesY;
		this.pixelCount = this.dataset.width * this.dataset.height;
		this.pairCapacity = this.tileCount * this.tileCapacity;
		const checkpointLayout = resolveCheckpointLayout(
			this.pixelCount,
			this.tileCapacity,
			this.storageBufferLimit,
			this.checkpointPrecision === "packed-f16" ? 8 : 16,
			this.requestedCheckpointStride,
		);
		this.checkpointStride = checkpointLayout.stride;
		this.blocksPerTile = checkpointLayout.blocksPerTile;
		const tiledBufferBytes = {
			pairData: this.pairCapacity * 8,
			checkpoints: checkpointLayout.byteLength,
			gradientAccumulator: this.splatCount * this.gradientFloats * 4,
		};
		for (const [label, byteLength] of Object.entries(tiledBufferBytes)) {
			assertStorageBufferFits(`The tiled ${label} buffer`, byteLength, this.storageBufferLimit);
		}
		const splitProjection = this.projectionLayout
			=== TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT;
		const rasterProjectionBytes = this.splatCount * (splitProjection
			? RASTER_PROJECTION_BYTES : MONOLITHIC_PROJECTION_BYTES);
		const projectionVjpBytes = splitProjection
			? this.splatCount * (
				this.projectionVjpPrecision === TILED_PROJECTION_VJP_PRECISIONS.PACKED_F16
					? PACKED_PROJECTION_VJP_BYTES : PROJECTION_VJP_BYTES
			) : 0;
		const packedTargetPageBytes = this.compactTargetFrames ? this.pixelCount * 4 : 4;
		Object.assign(this.buffers, {
			tiledConfig: makeBuffer(TILED_CONFIG_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
			projections: makeBuffer(rasterProjectionBytes),
			tileCounts: makeBuffer(this.tileCount * 4),
			pairData: makeBuffer(tiledBufferBytes.pairData),
			counters: makeBuffer(TILED_COUNTER_BYTES),
			indirectArgs: makeBuffer(12, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT
				| GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC),
			renderedTrain: makeBuffer(this.pixelCount * 16),
			targetPacked: makeBuffer(packedTargetPageBytes),
			checkpoints: makeBuffer(tiledBufferBytes.checkpoints),
			stopRanks: makeBuffer(this.pixelCount * 4),
			ssimStats: makeBuffer(this.pixelCount * SSIM_STATS_BYTES),
			pixelLoss: makeBuffer(this.pixelCount * 16),
			pixelGrad: makeBuffer(this.pixelCount * 16),
			gradientAtoms: makeBuffer(tiledBufferBytes.gradientAccumulator),
			tiledMetrics: makeBuffer(TILED_METRICS_BYTES),
		});
		if (splitProjection) {
			this.buffers.projectionVjp = makeBuffer(projectionVjpBytes);
		}
		if (this.ssimLayout === TILED_SSIM_LAYOUTS.SEPARABLE) {
			this.buffers.ssimScratch = makeBuffer(this.pixelCount * SSIM_STATS_BYTES);
		}
		if (this.timestampQueryEnabled) {
			const timestampBytes = TILED_GPU_PHASES.length * 2 * 8;
			this.tiledTimestampQuerySet = this.device.createQuerySet({
				label: "tiled-phase-timestamps",
				type: "timestamp",
				count: TILED_GPU_PHASES.length * 2,
			});
			this.buffers.tiledTimestampResolve = makeBuffer(
				timestampBytes,
				GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
			);
			this.buffers.tiledTimestampReadback = makeBuffer(
				timestampBytes,
				GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			);
			this.tiledTimestampBytes = timestampBytes;
		}
		this.cycleMetricCount = this.trainViewIndices.length * this.dataset.frameCount;
		this.cycleMetricBytes = this.cycleMetricCount * 16;
		this.buffers.metricsReadback.destroy();
		this.buffers.metricsReadback = makeBuffer(TILED_METRICS_BYTES + this.cycleMetricBytes,
			GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
		this.buffers.cycleMetrics = makeBuffer(this.cycleMetricBytes);
		const size = (value) => Number(value?.size ?? 0);
		const total = (values) => values.reduce((sum, value) => sum + size(value), 0);
		const bufferBytes = Object.freeze({
			parameterPingPong: total(this.buffers.params),
			optimizerMoments: size(this.buffers.firstMoment) + size(this.buffers.secondMoment),
			densityStats: size(this.buffers.stats),
			gradientAccumulator: size(this.buffers.gradientAtoms),
			projections: size(this.buffers.projections) + size(this.buffers.projectionVjp),
			parameterReadback: size(this.buffers.paramsReadback),
			previewSort: total(this.buffers.renderOrder) + total(this.buffers.renderDepths),
			sampleIndices: size(this.buffers.sampleIndices),
			sampledWorkspace: size(this.buffers.sampleGradients) + size(this.buffers.sampleLosses),
			targetPage: size(this.buffers.target),
			packedTargetPage: size(this.buffers.targetPacked),
			cameraData: size(this.buffers.cameras) + size(this.buffers.trainViews)
				+ size(this.buffers.cameraSampleIndices) + size(this.buffers.cameraSampleRanges),
			rasterPairs: size(this.buffers.pairData) + size(this.buffers.tileCounts),
			transmittanceCheckpoints: size(this.buffers.checkpoints),
			fullImageWorkspace: size(this.buffers.renderedTrain) + size(this.buffers.stopRanks)
				+ size(this.buffers.ssimStats) + size(this.buffers.ssimScratch)
				+ size(this.buffers.pixelLoss) + size(this.buffers.pixelGrad),
			configAndTelemetry: size(this.buffers.trainConfig) + total(this.buffers.renderConfig)
				+ size(this.buffers.tiledConfig) + size(this.buffers.counters)
				+ size(this.buffers.indirectArgs) + size(this.buffers.tiledMetrics)
				+ size(this.buffers.cycleMetrics) + size(this.buffers.metricsReadback)
				+ size(this.buffers.tiledTimestampResolve) + size(this.buffers.tiledTimestampReadback),
			previewGeometry: size(this.buffers.quad),
		});
		const capacityScaledBytes = bufferBytes.parameterPingPong + bufferBytes.optimizerMoments
			+ bufferBytes.densityStats + bufferBytes.gradientAccumulator + bufferBytes.projections
			+ bufferBytes.parameterReadback + bufferBytes.previewSort;
		const allocatedBytes = Object.values(bufferBytes).reduce((sum, value) => sum + value, 0);
		this.memoryPlan = Object.freeze({
			targetPageBytes: this.targetBufferByteLength(this.dataset),
			packedTargetPageBytes,
			targetFrameBankFormat: resolveFrameBank(this.dataset).format,
			tileCapacity: this.tileCapacity,
			tileSize: this.tileSize,
			pairCapacity: this.pairCapacity,
			checkpointStride: this.checkpointStride,
			requestedCheckpointStride: this.requestedCheckpointStride,
			checkpointPrecision: this.checkpointPrecision,
			checkpointOrder: this.checkpointOrder,
			checkpointBytes: tiledBufferBytes.checkpoints,
			pairDataBytes: tiledBufferBytes.pairData,
			gradientAccumulatorBytes: tiledBufferBytes.gradientAccumulator,
			rasterProjectionBytes,
			projectionVjpBytes,
			projectionVjpPrecision: this.projectionVjpPrecision,
			projectionLayout: this.projectionLayout,
			ssimLayout: this.ssimLayout,
			ssimScratchBytes: size(this.buffers.ssimScratch),
			pairGradientBytes: 0,
			sampledDepthOrderCacheBytes: 0,
			capacityScaledBytes,
			bytesPerCapacitySplat: capacityScaledBytes / this.splatCount,
			allocatedBytes,
			bufferBytes,
			storageBufferLimit: this.storageBufferLimit,
			nativeShaderF16: this.supportsShaderF16,
			backwardMode: this.backwardMode,
			backwardGranularity: this.backwardGranularity,
			gradientRecordFloats: this.gradientFloats,
			timestampQuerySupported: this.supportsTimestampQuery,
			timestampQueryEnabled: this.timestampQueryEnabled,
			sharePairPacket: this.sharePairPacket,
			staticWarmupSteps: this.staticWarmupSteps,
			metricCycleSteps: this.cycleMetricCount,
			dormantSlotSparseUpdate: true,
			initialActiveUpdateSlots: this.activeSplatCount,
			updateSlotCapacity: this.splatCount,
			initialDormantUpdateSlots: this.splatCount - this.activeSplatCount,
		});
	}

	createBindGroups() {
		super.createBindGroups();
		this.device.pushErrorScope("validation");
		const group = (pipeline, entries, index = 0) => this.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(index), entries,
		});
		const buffer = (binding, value) => ({ binding, resource: { buffer: value } });
		const projectionVjp = this.buffers.projectionVjp ?? this.buffers.projections;
		const updateEntries = (paramsIn, paramsOut) => {
			const entries = [
				buffer(0, this.buffers.tiledConfig), buffer(1, paramsIn),
				buffer(2, paramsOut), buffer(3, this.buffers.firstMoment),
				buffer(4, this.buffers.secondMoment), buffer(5, this.buffers.stats),
				buffer(6, this.buffers.gradientAtoms),
			];
			if (this.backwardMode === TILED_BACKWARD_MODES.STAGED_PROJECT_3D) {
				entries.push(buffer(7, projectionVjp));
				if (this.projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT) {
					entries.push(buffer(8, this.buffers.cameras));
				}
			}
			return entries;
		};
		const projectEntries = (params) => {
			const entries = [
				buffer(0, this.buffers.tiledConfig), buffer(1, params),
				buffer(2, this.buffers.cameras), buffer(3, this.buffers.tileCounts),
				buffer(4, this.buffers.pairData), buffer(5, this.buffers.projections),
				buffer(6, this.buffers.counters),
			];
			if (this.projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT) {
				entries.push(buffer(7, projectionVjp));
			}
			return entries;
		};
		const backwardEntries = (params) => {
			const entries = [
				buffer(0, this.buffers.tiledConfig), buffer(1, params),
				buffer(2, this.buffers.projections), buffer(3, this.buffers.pairData),
				buffer(4, this.buffers.renderedTrain), buffer(5, this.buffers.checkpoints),
				buffer(6, this.buffers.pixelGrad), buffer(7, this.buffers.gradientAtoms),
			];
			if (this.backwardGranularity === TILED_BACKWARD_GRANULARITIES.CHECKPOINT_BLOCK) {
				entries.push(buffer(9, this.buffers.tileCounts));
			} else {
				entries.push(buffer(8, this.buffers.counters));
			}
			return entries;
		};
		const ssimBindGroups = this.ssimLayout === TILED_SSIM_LAYOUTS.SEPARABLE ? {
			ssimHorizontal: group(this.tiledPipelines.ssimHorizontal, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.renderedTrain),
				buffer(2, this.buffers.target), buffer(3, this.buffers.ssimScratch),
			]),
			ssimVertical: group(this.tiledPipelines.ssimVertical, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.renderedTrain),
				buffer(2, this.buffers.target), buffer(3, this.buffers.ssimScratch),
				buffer(4, this.buffers.ssimStats), buffer(5, this.buffers.pixelLoss),
			]),
			ssimGradientHorizontal: group(this.tiledPipelines.ssimGradientHorizontal, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.target),
				buffer(2, this.buffers.ssimStats), buffer(3, this.buffers.ssimScratch),
			]),
			ssimGradientVertical: group(this.tiledPipelines.ssimGradientVertical, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.renderedTrain),
				buffer(2, this.buffers.target), buffer(3, this.buffers.ssimScratch),
				buffer(4, this.buffers.stopRanks), buffer(5, this.buffers.pixelGrad),
			]),
		} : {
			ssimStats: group(this.tiledPipelines.ssimStats, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.renderedTrain),
				buffer(2, this.buffers.target), buffer(3, this.buffers.ssimStats),
				buffer(4, this.buffers.pixelLoss),
			]),
			ssimGradient: group(this.tiledPipelines.ssimGradient, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.renderedTrain),
				buffer(2, this.buffers.target), buffer(3, this.buffers.ssimStats),
				buffer(4, this.buffers.stopRanks), buffer(5, this.buffers.pixelGrad),
			]),
		};
		this.tiledBindGroups = {
			targetDecode: group(this.tiledPipelines.targetDecode, [
				buffer(0, this.buffers.tiledConfig),
				buffer(1, this.buffers.targetPacked),
				buffer(2, this.buffers.target),
			]),
			clear: group(this.tiledPipelines.clear, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.tileCounts),
				buffer(2, this.buffers.counters), buffer(3, this.buffers.indirectArgs),
				buffer(4, this.buffers.tiledMetrics), buffer(5, this.buffers.gradientAtoms),
			]),
			project: this.buffers.params.map((params) =>
				group(this.tiledPipelines.project, projectEntries(params))),
			sort: group(this.tiledPipelines.sort, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.projections),
				buffer(2, this.buffers.tileCounts), buffer(3, this.buffers.pairData),
				buffer(4, this.buffers.counters),
			]),
			finalize: group(this.tiledPipelines.finalize, [
				buffer(0, this.buffers.counters), buffer(1, this.buffers.indirectArgs),
			]),
			forward: this.buffers.params.map((params) => group(this.tiledPipelines.forward, [
				buffer(0, this.buffers.tiledConfig), buffer(1, params), buffer(2, this.buffers.projections),
				buffer(3, this.buffers.tileCounts), buffer(4, this.buffers.pairData),
				buffer(5, this.buffers.renderedTrain), buffer(6, this.buffers.checkpoints),
				buffer(7, this.buffers.stopRanks),
			])),
			...ssimBindGroups,
			metrics: group(this.tiledPipelines.metrics, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.pixelLoss),
				buffer(2, this.buffers.counters), buffer(3, this.buffers.tiledMetrics),
				buffer(4, this.buffers.stopRanks), buffer(5, this.buffers.cycleMetrics),
			]),
			backward: this.buffers.params.map((params) =>
				group(this.tiledPipelines.backward, backwardEntries(params))),
			update: [
				group(this.tiledPipelines.update,
					updateEntries(this.buffers.params[0], this.buffers.params[1])),
				group(this.tiledPipelines.update,
					updateEntries(this.buffers.params[1], this.buffers.params[0])),
			],
			density: this.buffers.params.map((params) => group(this.tiledPipelines.density, [
				buffer(0, this.buffers.tiledConfig), buffer(1, params),
				buffer(2, this.buffers.firstMoment), buffer(3, this.buffers.secondMoment),
				buffer(4, this.buffers.stats), buffer(5, this.buffers.gradientAtoms),
				buffer(6, this.buffers.counters),
			])),
		};
		this.tiledBindGroupValidation = this.device.popErrorScope();
	}

	beginTiledComputePass(encoder, phase = null) {
		if (!this.activeTimestampProfile || phase == null) return encoder.beginComputePass();
		const phaseIndex = TILED_GPU_PHASES.indexOf(phase);
		if (phaseIndex < 0) throw new Error(`Unknown tiled GPU phase "${phase}".`);
		return encoder.beginComputePass({
			timestampWrites: {
				querySet: this.tiledTimestampQuerySet,
				beginningOfPassWriteIndex: phaseIndex * 2,
				endOfPassWriteIndex: phaseIndex * 2 + 1,
			},
		});
	}

	encodePass(encoder, pipeline, bindGroup, x, y = 1, z = 1, phase = null) {
		const pass = this.beginTiledComputePass(encoder, phase);
		pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(x, y, z); pass.end();
	}

	trainStep({ learningRate = 1, learningRateDecay = false, modelMode = 0,
		temporalSigma = 0.30, ssimRadius = 5, motionWeighting = false,
		randomBackground = false } = {}) {
		const validateSubmission = this.stepCount === 0;
		const resolvedSsimRadius = resolveSsimRadius(ssimRadius);
		const rates = browserLearningRates(learningRate, this.stepCount, learningRateDecay);
		this.lastLearningRateMultipliers = {
			geometry: rates.geometry,
			appearance: rates.appearance,
			progress: rates.progress,
		};
		if (validateSubmission) this.device.pushErrorScope("validation");
		const selected = trainingPairForStep(
			this.trainViewIndices,
			this.dataset.frameCount,
			this.stepCount,
			this.staticWarmupSteps,
		);
		this.lastCameraBatch = [selected.viewIndex]; this.lastCameraBatchStart = selected.viewSlot;
		this.lastFrameIndex = selected.frameIndex;
		this.lastTrainingPhase = selected.staticWarmup ? "static_warmup" : "dynamic_fit";
		const expectedActiveSplats = activeSplatCountForStep(
			this.initialSplatCount,
			this.splatCount,
			this.stepCount,
		);
		if (this.activeSplatCount !== expectedActiveSplats) {
			throw new Error(
				`Tiled active-prefix schedule drifted: tracked ${this.activeSplatCount}, `
				+ `expected ${expectedActiveSplats} at step ${this.stepCount}.`,
			);
		}
		const activeDispatch = activePrefixDispatchSizes(
			this.activeSplatCount,
			this.splatCount,
			this.tileCount,
			this.gradientFloats,
		);
		this.lastActiveUpdateSplats = activeDispatch.activeUpdateSlots;
		this.lastUpdateWorkgroups = activeDispatch.updateWorkgroups;
		this.lastClearWorkgroups = activeDispatch.clearWorkgroups;
		this.lastTargetSourceOffset = this.uploadTargetPage(
			this.buffers.target,
			selected.viewIndex,
			selected.frameIndex,
			{ staticWarmup: selected.staticWarmup },
		);
		const targetOffset = 0;
		this.lastTargetOffset = targetOffset;
		writeTiledConfig(this.tiledConfigBytes, {
			width: this.dataset.width, height: this.dataset.height, splatCount: this.splatCount,
			tileSize: this.tileSize,
			tilesX: this.tilesX, tilesY: this.tilesY, tileCapacity: this.tileCapacity,
			blocksPerTile: this.blocksPerTile, viewIndex: selected.viewIndex, frameIndex: selected.frameIndex,
			step: this.stepCount, modelMode, targetOffset, pixelCount: this.pixelCount,
			pairCapacity: this.pairCapacity, targetAspect: this.dataset.width / this.dataset.height,
			temporalSigma, alphaThreshold: 1 / 255, transmittanceThreshold: 1e-4,
			lrPosition: rates.position, lrColor: rates.color,
			lrOpacity: rates.opacity, lrMotion: rates.motion,
			geometryScale: this.dataset.geometryScale, l1Weight: 0.8, dssimWeight: 0.2,
			statDecay: DENSITY_STAT_DECAY, ssimRadius: resolvedSsimRadius,
			frameCount: this.dataset.frameCount,
			staticWarmup: selected.staticWarmup,
			motionWeighting,
			randomBackground,
			checkpointStride: this.checkpointStride,
			activeSplatCount: this.activeSplatCount,
			targetPacked: this.targetDecodePending,
		});
		this.device.queue.writeBuffer(this.buffers.tiledConfig, 0, this.tiledConfigBytes);
		const encoder = this.device.createCommandEncoder();
		if (this.targetDecodePending || this.activeTimestampProfile) {
			this.encodePass(
				encoder,
				this.tiledPipelines.targetDecode,
				this.tiledBindGroups.targetDecode,
				this.targetDecodePending ? ceilDiv(this.pixelCount, 256) : 1,
				1,
				1,
				"targetDecode",
			);
		}
		this.targetDecodePending = false;
		this.encodePass(encoder, this.tiledPipelines.clear, this.tiledBindGroups.clear,
			activeDispatch.clearWorkgroups,
			1, 1, "clear");
		this.encodePass(encoder, this.tiledPipelines.project, this.tiledBindGroups.project[this.currentIndex],
			ceilDiv(this.splatCount, 64), 1, 1, "project");
		this.encodePass(encoder, this.tiledPipelines.sort, this.tiledBindGroups.sort,
			this.tileCount, 1, 1, "sort");
		this.encodePass(encoder, this.tiledPipelines.finalize, this.tiledBindGroups.finalize,
			1, 1, 1, "finalize");
		this.encodePass(encoder, this.tiledPipelines.forward, this.tiledBindGroups.forward[this.currentIndex],
			this.tilesX, this.tilesY, 1, "forward");
		const ssimWorkgroups = ceilDiv(this.pixelCount, 64);
		if (this.ssimLayout === TILED_SSIM_LAYOUTS.SEPARABLE) {
			const statsPass = this.beginTiledComputePass(encoder, "ssimStats");
			statsPass.setPipeline(this.tiledPipelines.ssimHorizontal);
			statsPass.setBindGroup(0, this.tiledBindGroups.ssimHorizontal);
			statsPass.dispatchWorkgroups(ssimWorkgroups);
			statsPass.setPipeline(this.tiledPipelines.ssimVertical);
			statsPass.setBindGroup(0, this.tiledBindGroups.ssimVertical);
			statsPass.dispatchWorkgroups(ssimWorkgroups);
			statsPass.end();
			const gradientPass = this.beginTiledComputePass(encoder, "ssimGradient");
			gradientPass.setPipeline(this.tiledPipelines.ssimGradientHorizontal);
			gradientPass.setBindGroup(0, this.tiledBindGroups.ssimGradientHorizontal);
			gradientPass.dispatchWorkgroups(ssimWorkgroups);
			gradientPass.setPipeline(this.tiledPipelines.ssimGradientVertical);
			gradientPass.setBindGroup(0, this.tiledBindGroups.ssimGradientVertical);
			gradientPass.dispatchWorkgroups(ssimWorkgroups);
			gradientPass.end();
		} else {
			this.encodePass(encoder, this.tiledPipelines.ssimStats, this.tiledBindGroups.ssimStats,
				ssimWorkgroups, 1, 1, "ssimStats");
			this.encodePass(
				encoder,
				this.tiledPipelines.ssimGradient,
				this.tiledBindGroups.ssimGradient,
				ssimWorkgroups,
				1,
				1,
				"ssimGradient",
			);
		}
		this.encodePass(encoder, this.tiledPipelines.metrics, this.tiledBindGroups.metrics,
			1, 1, 1, "metrics");
		const backward = this.beginTiledComputePass(encoder, "backward");
		backward.setPipeline(this.tiledPipelines.backward);
		backward.setBindGroup(0, this.tiledBindGroups.backward[this.currentIndex]);
		backward.dispatchWorkgroupsIndirect(this.buffers.indirectArgs, 0); backward.end();
		this.encodePass(encoder, this.tiledPipelines.update, this.tiledBindGroups.update[this.currentIndex],
			activeDispatch.updateWorkgroups, 1, 1, "update");
		const nextStep = this.stepCount + 1;
		const densityDispatches = densityDispatchesForStep(
			this.initialSplatCount, this.splatCount, nextStep);
		if (densityDispatches > 0) {
			// Pass boundaries make each four-way split observe the prefix grown
			// by the preceding split, including its initialized child state.
			for (let pass = 0; pass < densityDispatches; pass += 1) {
				const maintenance = encoder.beginComputePass();
				maintenance.setPipeline(this.tiledPipelines.density);
				maintenance.setBindGroup(0, this.tiledBindGroups.density[1 - this.currentIndex]);
				maintenance.dispatchWorkgroups(1);
				maintenance.end();
			}
		}
		const nextActiveSplats = activeSplatCountForStep(
			this.initialSplatCount,
			this.splatCount,
			nextStep,
		);
		const activatedSplats = nextActiveSplats - this.activeSplatCount;
		this.activeSplatCount = nextActiveSplats;
		this.totalRecycled += activatedSplats;
		if (this.activeTimestampProfile) {
			encoder.resolveQuerySet(
				this.tiledTimestampQuerySet,
				0,
				TILED_GPU_PHASES.length * 2,
				this.buffers.tiledTimestampResolve,
				0,
			);
			encoder.copyBufferToBuffer(
				this.buffers.tiledTimestampResolve,
				0,
				this.buffers.tiledTimestampReadback,
				0,
				this.tiledTimestampBytes,
			);
		}
		this.lastProjectionParamIndex = this.currentIndex;
		this.device.queue.submit([encoder.finish()]);
		if (validateSubmission) {
			this.firstStepValidation = this.device.popErrorScope();
		}
		this.lastSampleCount = this.pixelCount;
		this.currentIndex = 1 - this.currentIndex; this.stepCount = nextStep;
	}

	async profileGpuStep(trainOptions = {}) {
		if (!this.timestampQueryEnabled) {
			return {
				supported: false,
				reason: this.supportsTimestampQuery
					? "Timestamp queries were not requested at initialization."
					: "The WebGPU adapter does not expose timestamp-query.",
			};
		}
		if (this.activeTimestampProfile) throw new Error("A tiled GPU profile is already active.");
		const profiledStep = this.stepCount;
		const profiledActiveSplats = activeSplatCountForStep(
			this.initialSplatCount,
			this.splatCount,
			profiledStep,
		);
		const maintenanceDispatches = densityDispatchesForStep(
			this.initialSplatCount,
			this.splatCount,
			profiledStep + 1,
		);
		await this.device.queue.onSubmittedWorkDone();
		this.activeTimestampProfile = true;
		try {
			this.trainStep(trainOptions);
		} finally {
			this.activeTimestampProfile = false;
		}
		await this.device.queue.onSubmittedWorkDone();
		await this.buffers.tiledTimestampReadback.mapAsync(
			GPUMapMode.READ,
			0,
			this.tiledTimestampBytes,
		);
		const bytes = this.buffers.tiledTimestampReadback
			.getMappedRange(0, this.tiledTimestampBytes).slice(0);
		this.buffers.tiledTimestampReadback.unmap();
		const timestamps = new BigUint64Array(bytes);
		const phases = {};
		let totalMs = 0;
		for (let index = 0; index < TILED_GPU_PHASES.length; index += 1) {
			const elapsedMs = Number(timestamps[index * 2 + 1] - timestamps[index * 2]) / 1e6;
			phases[TILED_GPU_PHASES[index]] = elapsedMs;
			totalMs += elapsedMs;
		}
		const gpuSpanMs = Number(
			timestamps[timestamps.length - 1] - timestamps[0],
		) / 1e6;
		const staged = this.backwardMode === TILED_BACKWARD_MODES.STAGED_PROJECT_3D;
		return {
			supported: true,
			totalMs,
			gpuSpanMs,
			phases,
			phaseContract: {
				backward: staged
					? "raster derivative and projected-gradient accumulation"
					: "raster derivative plus repeated 3D projection/covariance VJP",
				update: staged
					? "one 3D projection/covariance VJP per splat plus Adam"
					: "Adam",
			},
			maintenanceDispatches,
			maintenanceIncluded: false,
			activeUpdateSlots: profiledActiveSplats,
			capacityUpdateSlots: this.splatCount,
			dormantUpdateSlots: this.splatCount - profiledActiveSplats,
			updateWorkgroups: ceilDiv(profiledActiveSplats, 64),
		};
	}

	async readLoss() {
		const submissionError = await this.firstStepValidation;
		this.firstStepValidation = null;
		if (submissionError) throw new Error(`Tiled full-frame submission failed: ${submissionError.message}`);
		this.device.pushErrorScope("validation");
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(
			this.buffers.tiledMetrics,
			0,
			this.buffers.metricsReadback,
			0,
			TILED_METRICS_BYTES,
		);
		encoder.copyBufferToBuffer(
			this.buffers.cycleMetrics,
			0,
			this.buffers.metricsReadback,
			TILED_METRICS_BYTES,
			this.cycleMetricBytes,
		);
		this.device.queue.submit([encoder.finish()]);
		const readbackError = await this.device.popErrorScope();
		if (readbackError) throw new Error(`Tiled metric readback failed: ${readbackError.message}`);
		const readbackBytes = TILED_METRICS_BYTES + this.cycleMetricBytes;
		await this.buffers.metricsReadback.mapAsync(GPUMapMode.READ, 0, readbackBytes);
		const bytes = this.buffers.metricsReadback.getMappedRange(0, readbackBytes).slice(0);
		this.buffers.metricsReadback.unmap();
		const values = new Float32Array(bytes, 0, TILED_METRICS_BYTES / 4);
		const cycleRecords = new Float32Array(
			bytes,
			TILED_METRICS_BYTES,
			this.cycleMetricCount * 4,
		);
		const objectiveStep = Math.round(values[14]);
		const phaseStartStep = objectiveStep < this.staticWarmupSteps ? 0 : this.staticWarmupSteps;
		const cycle = summarizeCycleMetrics(
			cycleRecords,
			objectiveStep,
			this.cycleMetricCount,
			phaseStartStep,
		);
		this.lastLossBreakdown = {
			loss: values[0], l1: values[1], dssim: values[2], tileOverflow: values[3],
			coverage: values[4],
			pairCount: values[5],
			maxTileOccupancy: values[6],
			meanStopRank: values[7],
			visibleSplats: values[8],
			capacitySplats: values[9],
			viewIndex: values[10],
			frameIndex: values[11],
			tileOverflowTotal: values[12],
			maxTileOccupancyEver: values[13],
			objectiveStep,
			activeUpdateSplats: values[15],
			dormantUpdateSplats: values[9] - values[15],
			projectionVjpHalfSaturations: values[16],
			projectionVjpHalfSaturationsTotal: values[17],
			cycleMeanLoss: cycle?.loss ?? values[0],
			cycleMeanL1: cycle?.l1 ?? values[1],
			cycleMeanDssim: cycle?.dssim ?? values[2],
			cycleSamples: cycle?.samples ?? 1,
			cycleComplete: cycle?.complete ?? false,
		};
		return values[0];
	}

	async readTiledStepDebugStateUnlocked() {
		const renderedBytes = this.pixelCount * 16;
		const targetBytes = this.pixelCount * 16;
		const gradientBytes = this.splatCount * SPLAT_BYTES;
		const targetOffset = renderedBytes;
		const gradientOffset = targetOffset + targetBytes;
		const metricsOffset = gradientOffset + gradientBytes;
		const staged = this.backwardMode === TILED_BACKWARD_MODES.STAGED_PROJECT_3D;
		const materializedGradients = staged ? this.device.createBuffer({
			label: "tiled-staged-full-gradients",
			size: gradientBytes,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		}) : null;
		const readback = this.device.createBuffer({
			label: "tiled-step-debug-readback",
			size: metricsOffset + TILED_METRICS_BYTES,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});
		try {
			const encoder = this.device.createCommandEncoder();
			if (staged) {
				const paramsIndex = this.lastProjectionParamIndex ?? this.currentIndex;
				const debugGroup = this.device.createBindGroup({
					layout: this.tiledPipelines.gradientDebug.getBindGroupLayout(0),
					entries: [
						{ binding: 0, resource: { buffer: this.buffers.tiledConfig } },
						{ binding: 1, resource: { buffer: this.buffers.params[paramsIndex] } },
						{ binding: 2, resource: {
							buffer: this.buffers.projectionVjp ?? this.buffers.projections,
						} },
						{ binding: 3, resource: { buffer: this.buffers.gradientAtoms } },
						{ binding: 4, resource: { buffer: materializedGradients } },
						...(this.projectionLayout === TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT
							? [{ binding: 5, resource: { buffer: this.buffers.cameras } }]
							: []),
					],
				});
				this.encodePass(
					encoder,
					this.tiledPipelines.gradientDebug,
					debugGroup,
					ceilDiv(this.splatCount, 64),
				);
			}
			encoder.copyBufferToBuffer(this.buffers.renderedTrain, 0, readback, 0, renderedBytes);
			encoder.copyBufferToBuffer(
				this.buffers.target,
				0,
				readback,
				targetOffset,
				targetBytes,
			);
			encoder.copyBufferToBuffer(
				materializedGradients ?? this.buffers.gradientAtoms,
				0,
				readback,
				gradientOffset,
				gradientBytes,
			);
			encoder.copyBufferToBuffer(
				this.buffers.tiledMetrics,
				0,
				readback,
				metricsOffset,
				TILED_METRICS_BYTES,
			);
			this.device.queue.submit([encoder.finish()]);
			await readback.mapAsync(GPUMapMode.READ);
			const bytes = readback.getMappedRange().slice(0);
			return {
				step: this.stepCount,
				viewIndex: this.lastCameraBatch?.[0] ?? null,
				frameIndex: this.lastFrameIndex ?? null,
				renderedRgba: new Float32Array(bytes, 0, this.pixelCount * 4).slice(),
				targetRgba: new Float32Array(
					bytes,
					targetOffset,
					this.pixelCount * 4,
				).slice(),
				gradients: new Float32Array(
					bytes,
					gradientOffset,
					this.splatCount * SPLAT_FLOATS,
				).slice(),
				metrics: new Float32Array(
					bytes,
					metricsOffset,
					TILED_METRICS_BYTES / 4,
				).slice(),
			};
		} finally {
			if (readback.mapState === "mapped") readback.unmap();
			readback.destroy();
			materializedGradients?.destroy();
		}
	}

	readTiledStepDebugState() {
		const read = this.readbackChain.then(() => this.readTiledStepDebugStateUnlocked());
		this.readbackChain = read.then(() => undefined, () => undefined);
		return read;
	}

	dispose() {
		this.tiledTimestampQuerySet?.destroy();
		this.tiledTimestampQuerySet = null;
		super.dispose();
	}
}
