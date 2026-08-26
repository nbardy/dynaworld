import {
	BROWSER_ADAM_BETA1,
	BROWSER_ADAM_BETA2,
	BROWSER_ADAM_EPSILON,
	DENSITY_STAT_DECAY,
	browserLearningRates,
} from "./trainingSchedule.js?v=20260803-fullfps-pixelgs-1";
import {
	FRAME_BANK_FORMAT_RGBA8,
	readFrameBankValue,
	resolveFrameBank,
} from "./dataset.js?v=20260803-fullfps-pixelgs-1";

export const SPLAT_FLOATS = 24;
export const CONTINUATION_STATE_SCHEMA = "dynaworld-browser-trainer-continuation/v1";
export const CONTINUATION_PARAMETER_SCHEMA = "dynamic-splat-24f/v1";
export const MAX_BROWSER_RENDER_SPLATS = 32768;
export const INITIAL_SPLAT_OPACITY = 0.1;
// Targets are bounded RGB. Matching that support prevents opacity from trading
// against an unphysical overbright color during optimization.
export const MAX_SPLAT_COLOR = 1;
// This is a conservative screen-space standard deviation, not a covariance.
// It adds 0.09 px^2 to each projected covariance axis: enough to keep subpixel
// conics finite without claiming the stronger filtering of Mip-Splatting.
export const FILTER_SIGMA_PIXELS = 0.3;
// The sampled fallback has a tighter trust region because every ray examines
// every splat. A 4:1 scale ratio already permits a 16:1 covariance condition.
export const MAX_SAMPLED_SCALE_ASPECT_RATIO = 4;
const SPLAT_BYTES = SPLAT_FLOATS * 4;
const MAX_SAMPLES_PER_STEP = 192;
const MAX_RENDER_VIEWS = 3;
const DENSITY_INTERVAL = 512;
const DENSITY_STOP_STEP = 16384;
const DENSITY_SPLITS_PER_PASS = 4;
const QUATERNION_EPSILON = 1e-8;
const CONIC_DETERMINANT_EPSILON = 1e-16;

function requireContinuationInteger(value, label, { minimum = 0, maximum = Number.MAX_SAFE_INTEGER } = {}) {
	if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
		throw new RangeError(`${label} must be an integer from ${minimum} through ${maximum}.`);
	}
}

function requireContinuationFloat32Array(value, length, label) {
	if (!(value instanceof Float32Array) || value.length !== length) {
		throw new TypeError(`${label} must be a Float32Array with ${length} values.`);
	}
	for (const entry of value) {
		if (!Number.isFinite(entry)) throw new RangeError(`${label} must contain only finite values.`);
	}
}

function continuationNumbersMatch(left, right) {
	return Number.isFinite(left) && Number.isFinite(right)
		&& Math.abs(left - right) <= 1e-7 * Math.max(1, Math.abs(left), Math.abs(right));
}

export function assertContinuationStateCompatible(state, expected) {
	if (!state || typeof state !== "object") throw new TypeError("Continuation state must be an object.");
	if (state.schema !== CONTINUATION_STATE_SCHEMA) {
		throw new Error(`Unsupported continuation-state schema: ${state.schema ?? "missing"}.`);
	}
	const contract = state.contract;
	if (!contract || typeof contract !== "object") throw new TypeError("Continuation state is missing its contract.");
	if (contract.parameterSchema !== CONTINUATION_PARAMETER_SCHEMA || contract.splatFloats !== SPLAT_FLOATS) {
		throw new Error("Continuation parameter schema does not match this trainer.");
	}
	requireContinuationInteger(contract.splatCount, "Continuation splatCount", { minimum: 1 });
	if (contract.splatCount !== expected.splatCount) {
		throw new Error(`Continuation splat capacity ${contract.splatCount} does not match trainer capacity ${expected.splatCount}.`);
	}
	if (!continuationNumbersMatch(contract.geometryScale, expected.geometryScale)) {
		throw new Error("Continuation geometry scale does not match the target dataset.");
	}
	for (const key of ["frameCount", "cameraCount"]) {
		requireContinuationInteger(contract[key], `Continuation ${key}`, { minimum: 1 });
		if (contract[key] !== expected[key]) throw new Error(`Continuation ${key} does not match the target dataset.`);
	}
	if (!Array.isArray(contract.trainViewIndices)
		|| contract.trainViewIndices.length !== expected.trainViewIndices.length
		|| contract.trainViewIndices.some((value, index) => value !== expected.trainViewIndices[index])) {
		throw new Error("Continuation training-camera split does not match the target dataset.");
	}
	requireContinuationInteger(state.stepCount, "Continuation stepCount");
	requireContinuationInteger(state.currentIndex, "Continuation currentIndex", { maximum: 1 });
	requireContinuationInteger(state.totalRecycled, "Continuation totalRecycled");
	const parameterValues = expected.splatCount * SPLAT_FLOATS;
	requireContinuationFloat32Array(state.params, parameterValues, "Continuation params");
	requireContinuationFloat32Array(state.firstMoment, parameterValues, "Continuation firstMoment");
	requireContinuationFloat32Array(state.secondMoment, parameterValues, "Continuation secondMoment");
	requireContinuationFloat32Array(state.densityStats, expected.splatCount * 4, "Continuation densityStats");
	requireContinuationFloat32Array(state.initialParams, parameterValues, "Continuation initialParams");
	return state;
}

export function sampleGradientBufferBytes(splatCount) {
	if (!Number.isSafeInteger(splatCount) || splatCount < 1) {
		throw new RangeError("splatCount must be a positive safe integer.");
	}
	const bytes = MAX_SAMPLES_PER_STEP * splatCount * SPLAT_BYTES;
	if (!Number.isSafeInteger(bytes)) throw new RangeError("splatCount requires an unsafe buffer size.");
	return bytes;
}

export function sampledOrderCacheEntries(trainViewCount, frameCount, splatCount, enabled = true) {
	for (const [label, value] of Object.entries({ trainViewCount, frameCount, splatCount })) {
		if (!Number.isSafeInteger(value) || value < 1) {
			throw new RangeError(`${label} must be a positive safe integer.`);
		}
	}
	if (!enabled) return 0;
	const entries = trainViewCount * frameCount * splatCount;
	if (!Number.isSafeInteger(entries)) throw new RangeError("The sampled depth-order cache is too large.");
	return entries;
}

export function assertStorageBufferFits(label, requiredBytes, storageLimit) {
	if (!Number.isSafeInteger(requiredBytes) || requiredBytes < 0
		|| !Number.isSafeInteger(storageLimit) || storageLimit < 1) {
		throw new RangeError("Storage-buffer sizes must be nonnegative safe integers and the limit must be positive.");
	}
	if (requiredBytes > storageLimit) {
		throw new RangeError(`${label} needs a ${requiredBytes}-byte storage buffer; `
			+ `this device supports ${storageLimit} bytes. Stream or page this data before increasing the raster.`);
	}
}

export function rgbaFloatFrameBytes(dataset) {
	const bytes = dataset.width * dataset.height * 4 * Float32Array.BYTES_PER_ELEMENT;
	if (!Number.isSafeInteger(bytes) || bytes < 16) {
		throw new RangeError("Dataset raster dimensions require an invalid RGBA32F frame size.");
	}
	return bytes;
}

function nextPowerOfTwo(value) {
	return 2 ** Math.ceil(Math.log2(Math.max(1, value)));
}

export function resolveTrainViewIndices(dataset) {
	const roleIndices = dataset.cameras
		.map((camera, index) => camera.role === "train" ? index : -1)
		.filter((index) => index >= 0);
	if (roleIndices.length > 0) return roleIndices;
	return Array.from({ length: Math.min(dataset.trainViewCount ?? dataset.cameras.length,
		dataset.cameras.length) }, (_, index) => index);
}

export function resolveCamerasPerStep(trainViewCount, requested) {
	if (trainViewCount < 1) throw new Error("At least one training camera is required.");
	const fallback = trainViewCount > 4 ? 4 : trainViewCount;
	return Math.min(trainViewCount, Math.max(1, Math.floor(requested ?? fallback)));
}

export function rotatingTrainViewBatch(trainViewIndices, step, requested) {
	const count = trainViewIndices.length;
	const camerasPerStep = resolveCamerasPerStep(count, requested);
	if (camerasPerStep >= count) return { start: 0, indices: trainViewIndices.slice() };
	const start = (Math.max(0, Math.floor(step)) * camerasPerStep) % count;
	return { start, indices: Array.from({ length: camerasPerStep },
		(_, offset) => trainViewIndices[(start + offset) % count]) };
}

export function resolveRenderViewIndices(dataset, requested = null) {
	if (Array.isArray(requested) && requested.length > 0) {
		const valid = requested.filter((index, offset) => Number.isInteger(index) && index >= 0
			&& index < dataset.cameras.length && requested.indexOf(index) === offset);
		if (valid.length > 0) return valid.slice(0, MAX_RENDER_VIEWS);
	}
	const train = resolveTrainViewIndices(dataset);
	const heldout = Number.isInteger(dataset.heldoutViewIndex) && dataset.heldoutViewIndex >= 0
		? dataset.heldoutViewIndex
		: dataset.cameras.findIndex((camera) => camera.role === "heldout");
	if (train.length >= 2 && heldout >= 0) {
		return [train[0], train[Math.floor(train.length / 2)], heldout];
	}
	return Array.from({ length: Math.min(MAX_RENDER_VIEWS, dataset.cameras.length) }, (_, index) => index);
}

export function resolveActiveSplatCount(capacity, active = null) {
	if (!Number.isSafeInteger(capacity) || capacity < 1) {
		throw new RangeError("Splat capacity must be a positive safe integer.");
	}
	if (active == null) return capacity;
	if (!Number.isSafeInteger(active) || active < 1 || active > capacity) {
		throw new RangeError("Active splat count must be a positive safe integer no larger than capacity.");
	}
	return active;
}

function sigmoid(value) {
	return 1 / (1 + Math.exp(-value));
}

function frameTime(frame, frameCount) {
	return frameCount <= 1 ? 0 : frame / (frameCount - 1);
}

export function normalizeQuaternionCpu(raw) {
	const norm = Math.hypot(raw[0], raw[1], raw[2], raw[3]);
	if (!Number.isFinite(norm) || norm < QUATERNION_EPSILON) return [0, 0, 0, 1];
	return [raw[0] / norm, raw[1] / norm, raw[2] / norm, raw[3] / norm];
}

export function quaternionMatrixCpu(raw) {
	const [x, y, z, w] = normalizeQuaternionCpu(raw);
	return [
		1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
		2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
		2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
	];
}

function mat3Multiply(left, right) {
	const result = new Array(9).fill(0);
	for (let row = 0; row < 3; row += 1) for (let col = 0; col < 3; col += 1) {
		for (let inner = 0; inner < 3; inner += 1) result[row * 3 + col] += left[row * 3 + inner] * right[inner * 3 + col];
	}
	return result;
}

function mat3Vector(matrix, vector) {
	return [
		matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
		matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
		matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
	];
}

function dot3(left, right) {
	return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

export function screenSpaceFilterVariance(height) {
	return (FILTER_SIGMA_PIXELS / Math.max(1, height)) ** 2;
}

export function projectAnisotropicGaussianCpu({ center, logScales, quaternion, camera, aspect, height }) {
	const matrix = camera.worldToCamera;
	const cameraPoint = [
		matrix[0] * center[0] + matrix[1] * center[1] + matrix[2] * center[2] + matrix[3],
		matrix[4] * center[0] + matrix[5] * center[1] + matrix[6] * center[2] + matrix[7],
		matrix[8] * center[0] + matrix[9] * center[1] + matrix[10] * center[2] + matrix[11],
	];
	if (!cameraPoint.every(Number.isFinite) || cameraPoint[2] <= 0.1) return { valid: false, cameraPoint };
	const rotationCamera = [matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9], matrix[10]];
	const rotationWorld = quaternionMatrixCpu(quaternion);
	const basis = mat3Multiply(rotationCamera, rotationWorld);
	const variances = logScales.map((value) => Math.exp(2 * Math.max(-16, Math.min(4, value))));
	const sigmaCamera = new Array(9).fill(0);
	for (let row = 0; row < 3; row += 1) for (let col = 0; col < 3; col += 1) {
		for (let axis = 0; axis < 3; axis += 1) {
			sigmaCamera[row * 3 + col] += basis[row * 3 + axis] * variances[axis] * basis[col * 3 + axis];
		}
	}
	const [cameraX, cameraY, cameraZ] = cameraPoint;
	const [fx, fy, cx, cy] = camera.intrinsics;
	const horizontalFocal = aspect * fx;
	const invZ = 1 / cameraZ;
	const jacobian = [horizontalFocal * invZ, 0, -horizontalFocal * cameraX * invZ * invZ,
		0, fy * invZ, -fy * cameraY * invZ * invZ];
	const row0 = jacobian.slice(0, 3); const row1 = jacobian.slice(3, 6);
	const sigmaRow0 = mat3Vector(sigmaCamera, row0); const sigmaRow1 = mat3Vector(sigmaCamera, row1);
	const filterVariance = screenSpaceFilterVariance(height);
	const covariance = [dot3(row0, sigmaRow0) + filterVariance, dot3(row0, sigmaRow1),
		dot3(row1, sigmaRow1) + filterVariance];
	const determinant = covariance[0] * covariance[2] - covariance[1] * covariance[1];
	if (!covariance.every(Number.isFinite) || !Number.isFinite(determinant) || determinant <= CONIC_DETERMINANT_EPSILON) {
		return { valid: false, cameraPoint, covariance, determinant };
	}
	return {
		valid: true,
		center: [aspect * (fx * cameraX * invZ + cx), fy * cameraY * invZ + cy],
		cameraPoint,
		covariance,
		conic: [covariance[2] / determinant, -covariance[1] / determinant, covariance[0] / determinant],
		determinant,
		jacobian,
		sigmaCamera,
		basis,
		variances,
		rotationCamera,
		quaternion: normalizeQuaternionCpu(quaternion),
		rawQuaternion: Array.from(quaternion),
	};
}

export function evaluateAnisotropicGaussianCpu(projection, sample, opacity = 1, timeWeight = 1) {
	if (!projection.valid) return { active: false, qform: Number.POSITIVE_INFINITY, gaussian: 0, alpha: 0 };
	const delta = [sample[0] - projection.center[0], sample[1] - projection.center[1]];
	const qform = projection.conic[0] * delta[0] * delta[0]
		+ 2 * projection.conic[1] * delta[0] * delta[1] + projection.conic[2] * delta[1] * delta[1];
	if (!Number.isFinite(qform) || qform < 0 || qform > 9) return { active: false, delta, qform, gaussian: 0, alpha: 0 };
	const gaussian = Math.exp(-0.5 * qform);
	return { active: true, delta, qform, gaussian, alpha: opacity * timeWeight * gaussian };
}

export function anisotropicGaussianVjpCpu({ projection, sample, opacity = 1, timeWeight = 1, barAlpha = 1 }) {
	const evaluation = evaluateAnisotropicGaussianCpu(projection, sample, opacity, timeWeight);
	const zero = { center: [0, 0, 0], logScales: [0, 0, 0], quaternion: [0, 0, 0, 0] };
	if (!evaluation.active) return { ...zero, evaluation };
	const [dx, dy] = evaluation.delta; const [k00, k01, k11] = projection.conic;
	const vx = k00 * dx + k01 * dy; const vy = k01 * dx + k11 * dy;
	const barQform = -0.5 * barAlpha * evaluation.alpha;
	const barMu = [-2 * barQform * vx, -2 * barQform * vy];
	const barC = [-barQform * vx * vx, -barQform * vx * vy, -barQform * vy * vy];
	const j0 = projection.jacobian.slice(0, 3); const j1 = projection.jacobian.slice(3, 6);
	const barSigma = new Array(9).fill(0);
	for (let row = 0; row < 3; row += 1) for (let col = 0; col < 3; col += 1) {
		barSigma[row * 3 + col] = barC[0] * j0[row] * j0[col]
			+ barC[1] * (j0[row] * j1[col] + j1[row] * j0[col]) + barC[2] * j1[row] * j1[col];
	}
	const sigmaJ0 = mat3Vector(projection.sigmaCamera, j0); const sigmaJ1 = mat3Vector(projection.sigmaCamera, j1);
	const barJ0 = sigmaJ0.map((value, axis) => 2 * (barC[0] * value + barC[1] * sigmaJ1[axis]));
	const barJ1 = sigmaJ0.map((value, axis) => 2 * (barC[1] * value + barC[2] * sigmaJ1[axis]));
	const [x, y, z] = projection.cameraPoint; const invZ = 1 / z;
	const horizontalFocal = projection.jacobian[0] * z; const verticalFocal = projection.jacobian[4] * z;
	const barCamera = [
		barMu[0] * horizontalFocal * invZ - barJ0[2] * horizontalFocal * invZ * invZ,
		barMu[1] * verticalFocal * invZ - barJ1[2] * verticalFocal * invZ * invZ,
		-barMu[0] * horizontalFocal * x * invZ * invZ - barMu[1] * verticalFocal * y * invZ * invZ
			-barJ0[0] * horizontalFocal * invZ * invZ + barJ0[2] * 2 * horizontalFocal * x * invZ ** 3
			-barJ1[1] * verticalFocal * invZ * invZ + barJ1[2] * 2 * verticalFocal * y * invZ ** 3,
	];
	const rc = projection.rotationCamera;
	const barCenter = [rc[0] * barCamera[0] + rc[3] * barCamera[1] + rc[6] * barCamera[2],
		rc[1] * barCamera[0] + rc[4] * barCamera[1] + rc[7] * barCamera[2],
		rc[2] * barCamera[0] + rc[5] * barCamera[1] + rc[8] * barCamera[2]];
	const barLogScales = [0, 0, 0]; const barBasis = new Array(9).fill(0);
	for (let axis = 0; axis < 3; axis += 1) {
		const column = [projection.basis[axis], projection.basis[3 + axis], projection.basis[6 + axis]];
		const transformed = mat3Vector(barSigma, column);
		barLogScales[axis] = 2 * projection.variances[axis] * dot3(column, transformed);
		for (let row = 0; row < 3; row += 1) barBasis[row * 3 + axis] = 2 * projection.variances[axis] * transformed[row];
	}
	const barRotation = new Array(9).fill(0);
	for (let row = 0; row < 3; row += 1) for (let col = 0; col < 3; col += 1) {
		barRotation[row * 3 + col] = rc[row] * barBasis[col] + rc[3 + row] * barBasis[3 + col]
			+ rc[6 + row] * barBasis[6 + col];
	}
	const [qx, qy, qz, qw] = projection.quaternion; const h = barRotation;
	const normalizedGradient = [
		-4 * qx * (h[4] + h[8]) + 2 * qy * (h[1] + h[3]) + 2 * qz * (h[2] + h[6]) + 2 * qw * (h[7] - h[5]),
		-4 * qy * (h[0] + h[8]) + 2 * qx * (h[1] + h[3]) + 2 * qz * (h[5] + h[7]) + 2 * qw * (h[2] - h[6]),
		-4 * qz * (h[0] + h[4]) + 2 * qx * (h[2] + h[6]) + 2 * qy * (h[5] + h[7]) + 2 * qw * (h[3] - h[1]),
		2 * qz * (h[3] - h[1]) + 2 * qy * (h[2] - h[6]) + 2 * qx * (h[7] - h[5]),
	];
	const rawNorm = Math.hypot(...projection.rawQuaternion);
	const radial = dot3(normalizedGradient, projection.quaternion) + normalizedGradient[3] * projection.quaternion[3];
	const barQuaternion = rawNorm < QUATERNION_EPSILON ? [0, 0, 0, 0]
		: normalizedGradient.map((value, index) => (value - radial * projection.quaternion[index]) / rawNorm);
	return { center: barCenter, logScales: barLogScales, quaternion: barQuaternion, evaluation };
}

export function resolveAnchorCameraIndex(dataset) {
	const anchorName = dataset.datasetContract?.anchor_camera;
	const anchorIndex = dataset.cameras.findIndex((camera) => camera.name === anchorName);
	return anchorIndex >= 0 ? anchorIndex : 0;
}

export function normalizeDatasetGeometry(dataset) {
	const anchor = dataset.cameras[resolveAnchorCameraIndex(dataset)].worldToCamera;
	const depths = [];
	for (let i = 0; i < dataset.seedPointCount; i += 1) {
		const base = i * 6;
		const depth = anchor[8] * dataset.seedPoints[base]
			+ anchor[9] * dataset.seedPoints[base + 1]
			+ anchor[10] * dataset.seedPoints[base + 2] + anchor[11];
		if (Number.isFinite(depth) && depth > 0) depths.push(depth);
	}
	depths.sort((a, b) => a - b);
	const medianDepth = depths.length > 0 ? depths[Math.floor((depths.length - 1) / 2)] : 1;
	const geometryScale = Math.min(1, Math.max(1e-4, 1 / Math.max(1e-4, medianDepth)));
	const seedPoints = dataset.seedPoints.slice();
	for (let i = 0; i < dataset.seedPointCount; i += 1) {
		const base = i * 6;
		seedPoints[base] *= geometryScale;
		seedPoints[base + 1] *= geometryScale;
		seedPoints[base + 2] *= geometryScale;
	}
	const cameras = dataset.cameras.map((camera) => {
		const worldToCamera = camera.worldToCamera.slice();
		worldToCamera[3] *= geometryScale;
		worldToCamera[7] *= geometryScale;
		worldToCamera[11] *= geometryScale;
		return { ...camera, worldToCamera };
	});
	return { ...dataset, seedPoints, cameras, geometryScale };
}

function symmetricEigenvectors3(matrix) {
	const values = [...matrix];
	const vectors = [1, 0, 0, 0, 1, 0, 0, 0, 1];
	for (let iteration = 0; iteration < 16; iteration += 1) {
		let p = 0; let q = 1; let largest = Math.abs(values[1]);
		for (const [row, col] of [[0, 2], [1, 2]]) {
			const magnitude = Math.abs(values[row * 3 + col]);
			if (magnitude > largest) {
				p = row; q = col; largest = magnitude;
			}
		}
		if (largest < 1e-12) break;
		const pp = values[p * 3 + p]; const qq = values[q * 3 + q];
		const angle = 0.5 * Math.atan2(2 * values[p * 3 + q], qq - pp);
		const cosine = Math.cos(angle); const sine = Math.sin(angle);
		for (let axis = 0; axis < 3; axis += 1) {
			if (axis === p || axis === q) continue;
			const ap = values[axis * 3 + p]; const aq = values[axis * 3 + q];
			values[axis * 3 + p] = cosine * ap - sine * aq;
			values[p * 3 + axis] = values[axis * 3 + p];
			values[axis * 3 + q] = sine * ap + cosine * aq;
			values[q * 3 + axis] = values[axis * 3 + q];
		}
		values[p * 3 + p] = cosine * cosine * pp - 2 * sine * cosine * values[p * 3 + q]
			+ sine * sine * qq;
		values[q * 3 + q] = sine * sine * pp + 2 * sine * cosine * values[p * 3 + q]
			+ cosine * cosine * qq;
		values[p * 3 + q] = 0; values[q * 3 + p] = 0;
		for (let row = 0; row < 3; row += 1) {
			const vp = vectors[row * 3 + p]; const vq = vectors[row * 3 + q];
			vectors[row * 3 + p] = cosine * vp - sine * vq;
			vectors[row * 3 + q] = sine * vp + cosine * vq;
		}
	}
	return [0, 1, 2]
		.map((axis) => ({
			value: Math.max(0, values[axis * 3 + axis]),
			vector: [vectors[axis], vectors[3 + axis], vectors[6 + axis]],
		}))
		.sort((left, right) => right.value - left.value);
}

function cross3(left, right) {
	return [
		left[1] * right[2] - left[2] * right[1],
		left[2] * right[0] - left[0] * right[2],
		left[0] * right[1] - left[1] * right[0],
	];
}

function normalized3(vector, fallback) {
	const norm = Math.hypot(...vector);
	return Number.isFinite(norm) && norm > 1e-10 ? vector.map((value) => value / norm) : fallback;
}

function matrixQuaternion(matrix) {
	const trace = matrix[0] + matrix[4] + matrix[8];
	let quaternion;
	if (trace > 0) {
		const scale = 2 * Math.sqrt(trace + 1);
		quaternion = [(matrix[7] - matrix[5]) / scale, (matrix[2] - matrix[6]) / scale,
			(matrix[3] - matrix[1]) / scale, 0.25 * scale];
	} else if (matrix[0] > matrix[4] && matrix[0] > matrix[8]) {
		const scale = 2 * Math.sqrt(1 + matrix[0] - matrix[4] - matrix[8]);
		quaternion = [0.25 * scale, (matrix[1] + matrix[3]) / scale,
			(matrix[2] + matrix[6]) / scale, (matrix[7] - matrix[5]) / scale];
	} else if (matrix[4] > matrix[8]) {
		const scale = 2 * Math.sqrt(1 + matrix[4] - matrix[0] - matrix[8]);
		quaternion = [(matrix[1] + matrix[3]) / scale, 0.25 * scale,
			(matrix[5] + matrix[7]) / scale, (matrix[2] - matrix[6]) / scale];
	} else {
		const scale = 2 * Math.sqrt(1 + matrix[8] - matrix[0] - matrix[4]);
		quaternion = [(matrix[2] + matrix[6]) / scale, (matrix[5] + matrix[7]) / scale,
			0.25 * scale, (matrix[3] - matrix[1]) / scale];
	}
	return normalizeQuaternionCpu(quaternion);
}

function localGaussianFrames(seeds, selectedSeeds, geometryScale, neighborCount = 8) {
	const minimumScale = 0.03 * geometryScale;
	const maximumScale = 0.60 * geometryScale;
	return selectedSeeds.map((seed, index) => {
		const source = seed * 6;
		const neighbors = [];
		for (let other = 0; other < selectedSeeds.length; other += 1) {
			if (other === index) continue;
			const candidate = selectedSeeds[other] * 6;
			const dx = seeds[source] - seeds[candidate];
			const dy = seeds[source + 1] - seeds[candidate + 1];
			const dz = seeds[source + 2] - seeds[candidate + 2];
			const distanceSquared = dx * dx + dy * dy + dz * dz;
			if (!Number.isFinite(distanceSquared)) continue;
			let insertAt = neighbors.length;
			while (insertAt > 0 && neighbors[insertAt - 1].distanceSquared > distanceSquared) insertAt -= 1;
			if (insertAt < neighborCount) {
				neighbors.splice(insertAt, 0, { source: candidate, distanceSquared });
				if (neighbors.length > neighborCount) neighbors.pop();
			}
		}
		if (neighbors.length < 3) {
			const radius = Math.min(maximumScale, Math.max(minimumScale,
				Math.sqrt(neighbors[0]?.distanceSquared ?? 0) * 0.75 || 0.30 * geometryScale));
			return { scales: [radius, radius, radius], quaternion: [0, 0, 0, 1] };
		}
		const local = [{ source }, ...neighbors];
		const mean = [0, 0, 0];
		for (const point of local) for (let axis = 0; axis < 3; axis += 1) {
			mean[axis] += seeds[point.source + axis] / local.length;
		}
		const covariance = new Array(9).fill(0);
		for (const point of local) {
			const delta = [seeds[point.source] - mean[0], seeds[point.source + 1] - mean[1],
				seeds[point.source + 2] - mean[2]];
			for (let row = 0; row < 3; row += 1) for (let col = 0; col < 3; col += 1) {
				covariance[row * 3 + col] += delta[row] * delta[col] / local.length;
			}
		}
		const eigen = symmetricEigenvectors3(covariance);
		const axis0 = normalized3(eigen[0].vector, [1, 0, 0]);
		const projectedAxis1 = eigen[1].vector.map((value, axis) =>
			value - axis0[axis] * dot3(eigen[1].vector, axis0));
		const axis1 = normalized3(projectedAxis1, [0, 1, 0]);
		const axis2 = normalized3(cross3(axis0, axis1), [0, 0, 1]);
		const rawScales = eigen.map(({ value }, axis) => Math.sqrt(value)
			* (axis === 2 ? 0.75 : 1.35));
		const scales = rawScales.map((value) => Math.min(maximumScale, Math.max(minimumScale, value)));
		const largest = Math.max(...scales);
		// Local PCA supplies orientation, but a sparse/noisy neighborhood can
		// produce a needle at initialization. Start within 3:1 and let verified
		// image gradients learn stronger anisotropy.
		for (let axis = 0; axis < 3; axis += 1) scales[axis] = Math.max(scales[axis], largest / 3);
		return {
			scales,
			quaternion: matrixQuaternion([
				axis0[0], axis1[0], axis2[0],
				axis0[1], axis1[1], axis2[1],
				axis0[2], axis1[2], axis2[2],
			]),
		};
	});
}

export function makeInitialSplats(dataset, splatCount) {
	const params = new Float32Array(splatCount * SPLAT_FLOATS);
	const seeds = dataset.seedPoints;
	const seedCount = Math.max(1, dataset.seedPointCount);
	const anchor = dataset.cameras[resolveAnchorCameraIndex(dataset)].worldToCamera;
	const selectedSeeds = Array.from({ length: splatCount }, (_, index) =>
		Math.min(seedCount - 1, Math.floor((index + 0.5) * seedCount / splatCount)));
	selectedSeeds.sort((left, right) => {
		const leftBase = left * 6; const rightBase = right * 6;
		const leftDepth = anchor[8] * seeds[leftBase] + anchor[9] * seeds[leftBase + 1]
			+ anchor[10] * seeds[leftBase + 2] + anchor[11];
		const rightDepth = anchor[8] * seeds[rightBase] + anchor[9] * seeds[rightBase + 1]
			+ anchor[10] * seeds[rightBase + 2] + anchor[11];
		return rightDepth - leftDepth;
	});
	const frames = localGaussianFrames(seeds, selectedSeeds, dataset.geometryScale);
	for (let i = 0; i < splatCount; i += 1) {
		const seed = selectedSeeds[i];
		const source = seed * 6;
		const base = i * SPLAT_FLOATS;
		params[base] = seeds[source];
		params[base + 1] = seeds[source + 1];
		params[base + 2] = seeds[source + 2];
		params[base + 3] = 0.92;
		params[base + 4] = 0;
		params[base + 5] = 0;
		params[base + 6] = 0;
		params[base + 7] = 0.5;
		params[base + 8] = 0;
		params[base + 9] = 0;
		params[base + 10] = 0;
		params[base + 11] = 0;
		params[base + 12] = Math.log(Math.max(1e-6, frames[i].scales[0]));
		params[base + 13] = Math.log(Math.max(1e-6, frames[i].scales[1]));
		params[base + 14] = Math.log(Math.max(1e-6, frames[i].scales[2]));
		params[base + 15] = 0;
		params.set(frames[i].quaternion, base + 16);
		params[base + 20] = seeds[source + 3];
		params[base + 21] = seeds[source + 4];
		params[base + 22] = seeds[source + 5];
		params[base + 23] = Math.log(INITIAL_SPLAT_OPACITY / (1 - INITIAL_SPLAT_OPACITY));
	}
	return params;
}

function packCameras(cameras) {
	const packed = new Float32Array(cameras.length * 20);
	for (let i = 0; i < cameras.length; i += 1) {
		packed.set(cameras[i].worldToCamera, i * 20);
		packed.set(cameras[i].intrinsics, i * 20 + 16);
	}
	return packed;
}

export function packPreviewCamera(camera, geometryScale = 1) {
	if (!camera?.worldToCamera || camera.worldToCamera.length !== 16
		|| !Array.from(camera.worldToCamera).every(Number.isFinite)) {
		throw new TypeError("Preview camera worldToCamera must contain 16 finite values.");
	}
	if (!camera?.intrinsics || camera.intrinsics.length !== 4
		|| !Array.from(camera.intrinsics).every(Number.isFinite)) {
		throw new TypeError("Preview camera intrinsics must contain four finite values.");
	}
	if (!Number.isFinite(geometryScale) || geometryScale <= 0) {
		throw new RangeError("Preview camera geometry scale must be finite and positive.");
	}
	const packed = new Float32Array(20);
	packed.set(camera.worldToCamera);
	packed[3] *= geometryScale;
	packed[7] *= geometryScale;
	packed[11] *= geometryScale;
	packed.set(camera.intrinsics, 16);
	return packed;
}

export function packSamplesByCamera(dataset) {
	const pixelsPerViewFrame = dataset.width * dataset.height * dataset.frameCount;
	const motion = Array.from({ length: dataset.cameras.length }, () => []);
	const statics = Array.from({ length: dataset.cameras.length }, () => []);
	for (const packed of dataset.motionSamples) {
		const view = Math.floor(packed / pixelsPerViewFrame);
		if (view < motion.length) motion[view].push(packed);
	}
	for (const packed of dataset.staticSamples) {
		const view = Math.floor(packed / pixelsPerViewFrame);
		if (view < statics.length) statics[view].push(packed);
	}
	const packed = [];
	const ranges = new Uint32Array(dataset.cameras.length * 4);
	for (let view = 0; view < dataset.cameras.length; view += 1) {
		const base = view * 4;
		ranges[base] = packed.length; ranges[base + 1] = motion[view].length;
		packed.push(...motion[view]);
		ranges[base + 2] = packed.length; ranges[base + 3] = statics[view].length;
		packed.push(...statics[view]);
	}
	return { indices: new Uint32Array(packed.length > 0 ? packed : [0]), ranges };
}

function writeTrainConfig(buffer, values) {
	const view = new DataView(buffer);
	view.setUint32(0, values.width, true);
	view.setUint32(4, values.height, true);
	view.setUint32(8, values.frameCount, true);
	view.setUint32(12, values.splatCount, true);
	view.setUint32(16, values.sampleCount, true);
	view.setUint32(20, values.step, true);
	view.setUint32(24, values.modelMode, true);
	view.setUint32(28, values.motionSampleCount, true);
	view.setFloat32(32, values.lrPosition, true);
	view.setFloat32(36, values.lrColor, true);
	view.setFloat32(40, values.lrOpacity, true);
	view.setFloat32(44, values.lrMotion, true);
	view.setFloat32(48, values.minRadius, true);
	view.setFloat32(52, values.maxRadius, true);
	view.setFloat32(56, values.temporalSigma, true);
	view.setFloat32(60, values.targetAspect, true);
	view.setUint32(64, Math.round(values.motionSampleRate * 1000), true);
	view.setFloat32(68, values.motionCoverageTarget, true);
	view.setFloat32(72, values.motionCoverageWeight, true);
	view.setFloat32(76, values.staticAlphaWeight, true);
	view.setFloat32(80, 0, true);
	view.setFloat32(84, 0.00045, true);
	view.setUint32(88, values.staticSampleCount, true);
	view.setUint32(92, Math.round(values.staticSampleRate * 1000), true);
	view.setFloat32(96, BROWSER_ADAM_BETA1, true);
	view.setFloat32(100, BROWSER_ADAM_BETA2, true);
	view.setFloat32(104, BROWSER_ADAM_EPSILON, true);
	view.setFloat32(108, DENSITY_STAT_DECAY, true);
	view.setFloat32(112, 0, true);
	view.setUint32(116, values.trainViewCount, true);
	view.setUint32(120, values.cameraCount, true);
	view.setFloat32(124, values.geometryScale, true);
	view.setUint32(128, values.camerasPerStep, true);
	view.setUint32(132, values.cameraRotationStart, true);
	view.setUint32(136, values.legacyAllCameraSampling ? 1 : 0, true);
}

function writeRenderConfig(buffer, values) {
	const view = new DataView(buffer);
	view.setFloat32(0, values.width, true);
	view.setFloat32(4, values.height, true);
	view.setFloat32(8, values.time, true);
	view.setFloat32(12, values.splatCount, true);
	view.setFloat32(16, 1, true);
	view.setFloat32(20, values.modelMode, true);
	view.setFloat32(24, values.targetAspect, true);
	view.setFloat32(28, values.temporalSigma, true);
	view.setFloat32(32, values.targetWidth, true);
	view.setFloat32(36, values.targetHeight, true);
	view.setFloat32(40, values.renderMode, true);
	view.setFloat32(44, values.viewIndex, true);
}

const ORDER_WGSL = `
	struct Splat { centerStatic: vec4<f32>, velocityTime: vec4<f32>, harmonicPad: vec4<f32>,
		logScalePad: vec4<f32>, rotation: vec4<f32>, colorOpacity: vec4<f32> };
	struct Camera { row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, row3: vec4<f32>, intrinsics: vec4<f32> };
	struct TrainConfig {
		width: u32, height: u32, frameCount: u32, splatCount: u32,
		sampleCount: u32, step: u32, modelMode: u32, motionSampleCount: u32,
		lrPosition: f32, lrColor: f32, lrOpacity: f32, lrMotion: f32,
		minRadius: f32, maxRadius: f32, temporalSigma: f32, targetAspect: f32,
		motionSamplePermil: u32, motionCoverageTarget: f32, motionCoverageWeight: f32,
		staticAlphaWeight: f32, opacityDecayWeight: f32, staticEnergyThreshold: f32,
		staticSampleCount: u32, staticSamplePermil: u32,
		beta1: f32, beta2: f32, adamEpsilon: f32, statDecay: f32, robustMix: f32,
		trainViewCount: u32, cameraCount: u32, geometryScale: f32,
		camerasPerStep: u32, cameraRotationStart: u32, legacyAllCameraSampling: u32,
	};
	@group(0) @binding(0) var<uniform> cfg: TrainConfig;
	@group(0) @binding(1) var<storage, read> params: array<Splat>;
	@group(0) @binding(2) var<storage, read> cameras: array<Camera>;
	@group(0) @binding(3) var<storage, read> trainViewIndices: array<u32>;
	@group(0) @binding(4) var<storage, read_write> orderOutput: array<u32>;
	var<workgroup> orderIds: array<u32, 2048>;
	var<workgroup> orderDepths: array<f32, 2048>;
	fn center(p: Splat, t: f32) -> vec3<f32> {
		let tc = t * 2.0 - 1.0; var result = p.centerStatic.xyz + p.velocityTime.xyz * tc;
		if (cfg.modelMode == 0u) { result = result + p.harmonicPad.xyz * sin(t * 6.28318530718); }
		return result;
	}
	@compute @workgroup_size(256)
	fn build_order(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
		let pairCount = cfg.camerasPerStep * cfg.frameCount; if (wid.x >= pairCount) { return; }
		let cameraSlot = wid.x / cfg.frameCount; let frame = wid.x % cfg.frameCount;
		let view = trainViewIndices[(cfg.cameraRotationStart + cameraSlot) % cfg.trainViewCount];
		let camera = cameras[view]; let t = select(0.0, f32(frame) / f32(cfg.frameCount - 1u), cfg.frameCount > 1u);
		for (var i = lid.x; i < 2048u; i = i + 256u) {
			orderIds[i] = i;
			if (i < cfg.splatCount) { orderDepths[i] = dot(camera.row2, vec4<f32>(center(params[i], t), 1.0)); }
			else { orderDepths[i] = -1e30; }
		}
		workgroupBarrier();
		for (var width = 2u; width <= 2048u; width = width * 2u) {
			var stride = width / 2u;
			loop {
				for (var i = lid.x; i < 2048u; i = i + 256u) {
					let partner = i ^ stride;
					if (partner > i) {
						let left = orderIds[i]; let right = orderIds[partner]; let descending = (i & width) == 0u;
						let swap = select(orderDepths[left] > orderDepths[right], orderDepths[left] < orderDepths[right], descending);
						if (swap) { orderIds[i] = right; orderIds[partner] = left; }
					}
				}
				workgroupBarrier(); if (stride == 1u) { break; } stride = stride / 2u;
			}
		}
		let outputBase = cfg.motionSampleCount + cfg.staticSampleCount + wid.x * cfg.splatCount;
		for (var rank = lid.x; rank < cfg.splatCount; rank = rank + 256u) { orderOutput[outputBase + rank] = orderIds[rank]; }
	}
`;

function trainWgsl(frameBankFormat) {
	const compactTargets = frameBankFormat === FRAME_BANK_FORMAT_RGBA8;
	const targetDeclaration = compactTargets
		? "@group(0) @binding(2) var<storage, read> targetFrames: array<u32>;"
		: "@group(0) @binding(2) var<storage, read> targetFrames: array<vec4<f32>>;";
	const targetRgb = compactTargets
		? "unpack4x8unorm(targetFrames[targetIndex]).xyz"
		: "targetFrames[targetIndex].xyz";
	return `
	struct Splat {
		centerStatic: vec4<f32>,
		velocityTime: vec4<f32>,
		harmonicPad: vec4<f32>,
		logScalePad: vec4<f32>,
		rotation: vec4<f32>,
		colorOpacity: vec4<f32>,
	};
	struct Camera {
		row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, row3: vec4<f32>,
		intrinsics: vec4<f32>,
	};
	struct TrainConfig {
		width: u32, height: u32, frameCount: u32, splatCount: u32,
		sampleCount: u32, step: u32, modelMode: u32, motionSampleCount: u32,
		lrPosition: f32, lrColor: f32, lrOpacity: f32, lrMotion: f32,
		minRadius: f32, maxRadius: f32, temporalSigma: f32, targetAspect: f32,
		motionSamplePermil: u32, motionCoverageTarget: f32, motionCoverageWeight: f32,
		staticAlphaWeight: f32, opacityDecayWeight: f32, staticEnergyThreshold: f32,
		staticSampleCount: u32, staticSamplePermil: u32,
		beta1: f32, beta2: f32, adamEpsilon: f32, statDecay: f32, robustMix: f32,
		trainViewCount: u32, cameraCount: u32, geometryScale: f32,
		camerasPerStep: u32, cameraRotationStart: u32, legacyAllCameraSampling: u32,
	};
	struct Projection {
		center: vec2<f32>, conic: vec3<f32>, covariance: vec3<f32>, cameraPoint: vec3<f32>,
		jacobian0: vec3<f32>, jacobian1: vec3<f32>, sigmaCamera: mat3x3<f32>, basis: mat3x3<f32>,
		variances: vec3<f32>, quaternion: vec4<f32>, valid: f32,
	};

	@group(0) @binding(0) var<uniform> cfg: TrainConfig;
	@group(0) @binding(1) var<storage, read> paramsIn: array<Splat>;
	${targetDeclaration}
	@group(0) @binding(3) var<storage, read> sampleIndices: array<u32>;
	@group(0) @binding(4) var<storage, read> cameras: array<Camera>;
	@group(0) @binding(5) var<storage, read_write> sampleGradients: array<Splat>;
	@group(0) @binding(6) var<storage, read_write> sampleLosses: array<f32>;
	@group(0) @binding(7) var<storage, read> trainViewIndices: array<u32>;
	@group(0) @binding(8) var<storage, read> cameraSampleIndices: array<u32>;
	@group(0) @binding(9) var<storage, read> cameraSampleRanges: array<vec4<u32>>;
	var<workgroup> fastTapeUnderAlpha: array<vec4<f32>, 768>;
	var<workgroup> fastTapeSuffix: array<f32, 768>;
	var<workgroup> samplePointTimeView: vec4<f32>;
	var<workgroup> sampleTargetCoverage: vec4<f32>;
	var<workgroup> samplePredMotion: vec4<f32>;

	fn hash_u32(v: u32) -> u32 {
		var x = v; x = ((x >> 16u) ^ x) * 0x7feb352du;
		x = ((x >> 15u) ^ x) * 0x846ca68bu; return (x >> 16u) ^ x;
	}
	fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
	fn outer3(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
		return mat3x3<f32>(a * b.x, a * b.y, a * b.z);
	}
	fn safe_quaternion(raw: vec4<f32>) -> vec4<f32> {
		let norm2 = dot(raw, raw);
		let normalized = raw * inverseSqrt(max(norm2, 1e-16));
		return select(vec4<f32>(0.0, 0.0, 0.0, 1.0), normalized, norm2 > 1e-16);
	}
	fn quaternion_matrix(raw: vec4<f32>) -> mat3x3<f32> {
		let q = safe_quaternion(raw); let x = q.x; let y = q.y; let z = q.z; let w = q.w;
		return mat3x3<f32>(
			vec3<f32>(1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y + z*w), 2.0 * (x*z - y*w)),
			vec3<f32>(2.0 * (x*y - z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z + x*w)),
			vec3<f32>(2.0 * (x*z + y*w), 2.0 * (y*z - x*w), 1.0 - 2.0 * (x*x + y*y)));
	}
	fn camera_rotation(camera: Camera) -> mat3x3<f32> {
		return mat3x3<f32>(vec3<f32>(camera.row0.x, camera.row1.x, camera.row2.x),
			vec3<f32>(camera.row0.y, camera.row1.y, camera.row2.y),
			vec3<f32>(camera.row0.z, camera.row1.z, camera.row2.z));
	}
	fn frame_time(frame: u32) -> f32 {
		if (cfg.frameCount <= 1u) { return 0.0; }
		return f32(frame) / f32(cfg.frameCount - 1u);
	}
	fn active_view(slot: u32) -> u32 {
		return trainViewIndices[(cfg.cameraRotationStart + slot) % cfg.trainViewCount];
	}
	fn world_center(p: Splat, t: f32) -> vec3<f32> {
		let tc = t * 2.0 - 1.0;
		var center = p.centerStatic.xyz + p.velocityTime.xyz * tc;
		if (cfg.modelMode == 0u) { center = center + p.harmonicPad.xyz * sin(t * 6.28318530718); }
		return center;
	}
	fn temporal_gate(p: Splat, t: f32) -> f32 {
		let sigma = clamp(cfg.temporalSigma, 0.12, 0.36);
		let floorValue = clamp(sigma * 0.30, 0.035, 0.12);
		let dt = t - clamp(p.velocityTime.w, 0.0, 1.0);
		let dynamicGate = floorValue + (1.0 - floorValue) * exp(-0.5 * dt * dt / (sigma * sigma));
		return mix(dynamicGate, 1.0, clamp(p.centerStatic.w, 0.0, 1.0));
	}
	fn camera_point(camera: Camera, point: vec3<f32>) -> vec3<f32> {
		let h = vec4<f32>(point, 1.0);
		return vec3<f32>(dot(camera.row0, h), dot(camera.row1, h), dot(camera.row2, h));
	}
	fn project(p: Splat, t: f32, cameraIndex: u32) -> Projection {
		let camera = cameras[cameraIndex];
		let cp = camera_point(camera, world_center(p, t));
		let zero3 = vec3<f32>(0.0); let zeroMatrix = mat3x3<f32>(zero3, zero3, zero3);
		if (cp.z <= 0.1) {
			return Projection(vec2<f32>(-10.0), zero3, zero3, cp, zero3, zero3,
				zeroMatrix, zeroMatrix, zero3, vec4<f32>(0.0, 0.0, 0.0, 1.0), 0.0);
		}
		let q = safe_quaternion(p.rotation); let worldRotation = quaternion_matrix(q);
		let cameraRotation = camera_rotation(camera); let basis = cameraRotation * worldRotation;
		let logScales = clamp(p.logScalePad.xyz, vec3<f32>(-16.0), vec3<f32>(4.0));
		let variances = exp(2.0 * logScales);
		let sigmaCamera = variances.x * outer3(basis[0], basis[0])
			+ variances.y * outer3(basis[1], basis[1]) + variances.z * outer3(basis[2], basis[2]);
		let invZ = 1.0 / cp.z; let horizontalFocal = cfg.targetAspect * camera.intrinsics.x;
		let j0 = vec3<f32>(horizontalFocal * invZ, 0.0, -horizontalFocal * cp.x * invZ * invZ);
		let j1 = vec3<f32>(0.0, camera.intrinsics.y * invZ,
			-camera.intrinsics.y * cp.y * invZ * invZ);
		// Minimal EWA-style pixel footprint. This samples at pixel centers; it
		// is not pixel-area integration or the determinant-corrected Mip filter.
		let filterVariance = pow(${FILTER_SIGMA_PIXELS} / max(1.0, f32(cfg.height)), 2.0);
		let covariance = vec3<f32>(dot(j0, sigmaCamera * j0) + filterVariance,
			dot(j0, sigmaCamera * j1), dot(j1, sigmaCamera * j1) + filterVariance);
		let determinant = covariance.x * covariance.z - covariance.y * covariance.y;
		if (determinant <= 1e-16) {
			return Projection(vec2<f32>(-10.0), zero3, covariance, cp, j0, j1,
				sigmaCamera, basis, variances, q, 0.0);
		}
		let center = vec2<f32>(cfg.targetAspect * (camera.intrinsics.x * cp.x * invZ + camera.intrinsics.z),
			camera.intrinsics.y * cp.y * invZ + camera.intrinsics.w);
		let conic = vec3<f32>(covariance.z, -covariance.y, covariance.x) / determinant;
		return Projection(center, conic, covariance, cp, j0, j1, sigmaCamera, basis, variances, q, 1.0);
	}
	@compute @workgroup_size(256)
	fn sample_gradients(@builtin(local_invocation_id) lid: vec3<u32>,
		@builtin(workgroup_id) wid: vec3<u32>) {
		let pixels = cfg.width * cfg.height; let norm = 1.0 / max(1.0, f32(cfg.sampleCount));
		let s = wid.x;
		if (s >= cfg.sampleCount) { return; }
		if (lid.x == 0u) {
				let seed = hash_u32(cfg.step * 747796405u + s * 2891336453u + 277803737u);
				var view = active_view(s % cfg.camerasPerStep);
				if (cfg.legacyAllCameraSampling != 0u) { view = seed % cfg.trainViewCount; }
				var frame = hash_u32(seed + 17u) % cfg.frameCount;
				var pixel = hash_u32(seed + 1013904223u) % pixels; var usedMotion = false;
				let bucket = hash_u32(seed + 1664525u) % 1000u;
				if (cfg.motionSampleCount > 0u && bucket < cfg.motionSamplePermil) {
					let range = cameraSampleRanges[view];
					if (cfg.legacyAllCameraSampling != 0u) {
						let packed = sampleIndices[hash_u32(seed + 22695477u) % cfg.motionSampleCount];
						pixel = packed % pixels; let viewFrame = packed / pixels;
						frame = viewFrame % cfg.frameCount; view = viewFrame / cfg.frameCount; usedMotion = true;
					} else if (range.y > 0u) {
						let packed = cameraSampleIndices[range.x + hash_u32(seed + 22695477u) % range.y];
						pixel = packed % pixels; let viewFrame = packed / pixels;
						frame = viewFrame % cfg.frameCount; usedMotion = true;
					}
				} else if (cfg.staticSampleCount > 0u
					&& bucket < cfg.motionSamplePermil + cfg.staticSamplePermil) {
					let range = cameraSampleRanges[view];
					if (cfg.legacyAllCameraSampling != 0u) {
						let packed = sampleIndices[cfg.motionSampleCount
							+ hash_u32(seed + 374761393u) % cfg.staticSampleCount];
						pixel = packed % pixels; let viewFrame = packed / pixels;
						frame = viewFrame % cfg.frameCount; view = viewFrame / cfg.frameCount;
					} else if (range.w > 0u) {
						let packed = cameraSampleIndices[range.z + hash_u32(seed + 374761393u) % range.w];
						pixel = packed % pixels; let viewFrame = packed / pixels;
						frame = viewFrame % cfg.frameCount;
					}
				}
				let x = pixel % cfg.width; let y = pixel / cfg.width;
				let point = vec2<f32>((f32(x) + 0.5) / f32(cfg.width), (f32(y) + 0.5) / f32(cfg.height));
				let t = frame_time(frame); let targetIndex = (view * cfg.frameCount + frame) * pixels + pixel;
				samplePointTimeView = vec4<f32>(point, t, f32(view));
				sampleTargetCoverage = vec4<f32>(${targetRgb}, 0.0);
				samplePredMotion = vec4<f32>(0.0, 0.0, 0.0, select(0.0, 1.0, usedMotion));
		}
		workgroupBarrier();
				let point = samplePointTimeView.xy; let t = samplePointTimeView.z;
				let view = u32(round(samplePointTimeView.w));
				for (var i = lid.x; i < cfg.splatCount; i = i + 256u) {
						let candidate = paramsIn[i]; let projection = project(candidate, t, view); var alpha = 0.0;
						if (projection.valid > 0.5) {
							let d = vec2<f32>(point.x * cfg.targetAspect, point.y) - projection.center;
							let qform = projection.conic.x * d.x * d.x + 2.0 * projection.conic.y * d.x * d.y
								+ projection.conic.z * d.y * d.y;
							if (qform >= 0.0 && qform <= 9.0) {
								alpha = sigmoid(candidate.colorOpacity.w) * exp(-0.5 * qform)
									* temporal_gate(candidate, t);
							}
						}
					if (cfg.splatCount <= 768u) {
						fastTapeUnderAlpha[i] = vec4<f32>(projection.cameraPoint.z, 0.0, 0.0, alpha);
						fastTapeSuffix[i] = 0.0;
						} else {
							sampleGradients[s * cfg.splatCount + i] = Splat(
								vec4<f32>(0.0, 0.0, 0.0, alpha), vec4<f32>(0.0, projection.cameraPoint.z, 0.0, 0.0), vec4<f32>(0.0),
								vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
						}
			}
		storageBarrier(); workgroupBarrier();
			if (lid.x == 0u) {
				let frame = u32(round(t * f32(max(1u, cfg.frameCount - 1u))));
				let pair = (s % cfg.camerasPerStep) * cfg.frameCount + frame;
				let orderBase = cfg.motionSampleCount + cfg.staticSampleCount + pair * cfg.splatCount;
				var accum = vec3<f32>(0.0); var transmittance = 1.0;
				for (var rank = 0u; rank < cfg.splatCount; rank = rank + 1u) {
					let j = sampleIndices[orderBase + rank];
					var alpha = 0.0;
					if (cfg.splatCount <= 768u) {
						alpha = fastTapeUnderAlpha[j].w;
						fastTapeUnderAlpha[j] = vec4<f32>(accum, alpha);
					} else {
						let tapeIndex = s * cfg.splatCount + j; var tape = sampleGradients[tapeIndex];
							alpha = tape.centerStatic.w; tape.centerStatic = vec4<f32>(accum, alpha);
						sampleGradients[tapeIndex] = tape;
					}
					transmittance = transmittance * (1.0 - alpha);
					accum = accum * (1.0 - alpha) + paramsIn[j].colorOpacity.xyz * alpha;
				}
				var suffix = 1.0; var reverse = cfg.splatCount;
				loop {
					if (reverse == 0u) { break; }
					reverse = reverse - 1u; let orderedIndex = sampleIndices[orderBase + reverse];
					if (cfg.splatCount <= 768u) {
						fastTapeSuffix[orderedIndex] = suffix;
						suffix = suffix * (1.0 - fastTapeUnderAlpha[orderedIndex].w);
					} else {
						let tapeIndex = s * cfg.splatCount + orderedIndex; var tape = sampleGradients[tapeIndex];
						tape.velocityTime.x = suffix; sampleGradients[tapeIndex] = tape;
							suffix = suffix * (1.0 - tape.centerStatic.w);
					}
				}
				sampleTargetCoverage = vec4<f32>(sampleTargetCoverage.xyz, 1.0 - transmittance);
				samplePredMotion = vec4<f32>(accum, samplePredMotion.w);
			}
		storageBarrier(); workgroupBarrier();
			let targetColor = sampleTargetCoverage.xyz; let coverage = sampleTargetCoverage.w;
			let prediction = samplePredMotion.xyz; let usedMotion = samplePredMotion.w > 0.5;
			let err = prediction - targetColor; let dLoss = err * (2.0 / 3.0) * norm;
			if (lid.x == 0u) { sampleLosses[s] = dot(err, err) / 3.0; }
			for (var i = lid.x; i < cfg.splatCount; i = i + 256u) {
					var taped = Splat(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
						vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
					if (cfg.splatCount <= 768u) {
						taped.centerStatic = fastTapeUnderAlpha[i]; taped.velocityTime.x = fastTapeSuffix[i];
					} else {
						taped = sampleGradients[s * cfg.splatCount + i];
					}
					let p = paramsIn[i]; let projection = project(p, t, view);
					var gradPosition = vec3<f32>(0.0); var gradLogScale = vec3<f32>(0.0);
					var gradVelocity = vec3<f32>(0.0); var gradTime = 0.0;
					var gradHarmonic = vec3<f32>(0.0); var gradColor = vec3<f32>(0.0);
					var gradRotation = vec4<f32>(0.0); var gradOpacity = 0.0; var gradStaticMix = 0.0; var meanAlpha = 0.0;
					if (projection.valid > 0.5) {
						let d = vec2<f32>(point.x * cfg.targetAspect, point.y) - projection.center;
						let qform = projection.conic.x * d.x * d.x + 2.0 * projection.conic.y * d.x * d.y
							+ projection.conic.z * d.y * d.y;
						if (qform >= 0.0 && qform <= 9.0) {
						let gaussian = exp(-0.5 * qform); let opacity = sigmoid(p.colorOpacity.w);
						let timeWeight = temporal_gate(p, t); let alphaWeight = opacity * gaussian * timeWeight;
							meanAlpha = alphaWeight * norm;
							var alphaGrad = dot(dLoss, (p.colorOpacity.xyz - taped.centerStatic.xyz)
								* taped.velocityTime.x);
							if (!usedMotion && cfg.staticAlphaWeight > 0.0) {
								alphaGrad = alphaGrad + 2.0 * cfg.staticAlphaWeight * taped.centerStatic.w * norm;
							}
							let coverageError = coverage - cfg.motionCoverageTarget;
							if (usedMotion && coverageError < 0.0) {
								alphaGrad = alphaGrad + 2.0 * cfg.motionCoverageWeight * coverageError
									* (1.0 - coverage) / max(1e-3, 1.0 - taped.centerStatic.w) * norm;
							}
							let barQform = -0.5 * alphaGrad * alphaWeight;
							let conicDelta = vec2<f32>(projection.conic.x * d.x + projection.conic.y * d.y,
								projection.conic.y * d.x + projection.conic.z * d.y);
							let barMu = -2.0 * barQform * conicDelta;
							let barC00 = -barQform * conicDelta.x * conicDelta.x;
							let barC01 = -barQform * conicDelta.x * conicDelta.y;
							let barC11 = -barQform * conicDelta.y * conicDelta.y;
							let j0 = projection.jacobian0; let j1 = projection.jacobian1;
							let barSigma = barC00 * outer3(j0, j0) + barC01 * (outer3(j0, j1) + outer3(j1, j0))
								+ barC11 * outer3(j1, j1);
							let sigmaJ0 = projection.sigmaCamera * j0; let sigmaJ1 = projection.sigmaCamera * j1;
							let barJ0 = 2.0 * (barC00 * sigmaJ0 + barC01 * sigmaJ1);
							let barJ1 = 2.0 * (barC01 * sigmaJ0 + barC11 * sigmaJ1);
							let camera = cameras[view]; let cp = projection.cameraPoint; let invZ = 1.0 / cp.z;
							let horizontalFocal = cfg.targetAspect * camera.intrinsics.x;
							let verticalFocal = camera.intrinsics.y;
							var cameraGrad = vec3<f32>(
								barMu.x * horizontalFocal * invZ - barJ0.z * horizontalFocal * invZ * invZ,
								barMu.y * verticalFocal * invZ - barJ1.z * verticalFocal * invZ * invZ,
								-barMu.x * horizontalFocal * cp.x * invZ * invZ
								-barMu.y * verticalFocal * cp.y * invZ * invZ
								-barJ0.x * horizontalFocal * invZ * invZ
								+barJ0.z * 2.0 * horizontalFocal * cp.x * invZ * invZ * invZ
								-barJ1.y * verticalFocal * invZ * invZ
								+barJ1.z * 2.0 * verticalFocal * cp.y * invZ * invZ * invZ);
							let worldGrad = vec3<f32>(
								dot(vec3<f32>(camera.row0.x, camera.row1.x, camera.row2.x), cameraGrad),
								dot(vec3<f32>(camera.row0.y, camera.row1.y, camera.row2.y), cameraGrad),
								dot(vec3<f32>(camera.row0.z, camera.row1.z, camera.row2.z), cameraGrad));
							let tc = t * 2.0 - 1.0; let wave = sin(t * 6.28318530718);
							gradPosition = worldGrad;
							gradVelocity = worldGrad * tc;
							if (cfg.modelMode == 0u) { gradHarmonic = worldGrad * wave; }
							for (var axis = 0u; axis < 3u; axis = axis + 1u) {
								let column = projection.basis[axis];
								gradLogScale[axis] = 2.0 * projection.variances[axis] * dot(column, barSigma * column);
							}
							let barBasis = mat3x3<f32>(
								2.0 * projection.variances.x * (barSigma * projection.basis[0]),
								2.0 * projection.variances.y * (barSigma * projection.basis[1]),
								2.0 * projection.variances.z * (barSigma * projection.basis[2]));
							let barRotation = transpose(camera_rotation(camera)) * barBasis;
							let q = projection.quaternion;
							let h00 = barRotation[0].x; let h01 = barRotation[1].x; let h02 = barRotation[2].x;
							let h10 = barRotation[0].y; let h11 = barRotation[1].y; let h12 = barRotation[2].y;
							let h20 = barRotation[0].z; let h21 = barRotation[1].z; let h22 = barRotation[2].z;
							let normalizedQuatGrad = vec4<f32>(
								-4.0*q.x*(h11+h22) + 2.0*q.y*(h01+h10) + 2.0*q.z*(h02+h20) + 2.0*q.w*(h21-h12),
								-4.0*q.y*(h00+h22) + 2.0*q.x*(h01+h10) + 2.0*q.z*(h12+h21) + 2.0*q.w*(h02-h20),
								-4.0*q.z*(h00+h11) + 2.0*q.x*(h02+h20) + 2.0*q.y*(h12+h21) + 2.0*q.w*(h10-h01),
								2.0*q.z*(h10-h01) + 2.0*q.y*(h02-h20) + 2.0*q.x*(h21-h12));
							let rawNorm2 = dot(p.rotation, p.rotation);
							if (rawNorm2 > 1e-16) {
								gradRotation = (normalizedQuatGrad - q * dot(q, normalizedQuatGrad)) * inverseSqrt(rawNorm2);
							}
							let sigma = clamp(cfg.temporalSigma, 0.12, 0.36);
							let staticMix = clamp(p.centerStatic.w, 0.0, 1.0);
						let temporalFloor = clamp(sigma * 0.30, 0.035, 0.12);
						let timeDelta = t - clamp(p.velocityTime.w, 0.0, 1.0);
						let dynamicGate = temporalFloor + (1.0 - temporalFloor)
							* exp(-0.5 * timeDelta * timeDelta / (sigma * sigma));
						let dynamicCore = max(0.0, (timeWeight - staticMix - temporalFloor * (1.0 - staticMix))
							/ max(1e-6, 1.0 - staticMix));
						gradTime = alphaGrad * opacity * gaussian
							* (1.0 - staticMix) * dynamicCore * (t - p.velocityTime.w) / (sigma * sigma);
						gradStaticMix = alphaGrad * opacity * gaussian * (1.0 - dynamicGate);
						gradColor = dLoss * alphaWeight * taped.velocityTime.x;
						gradOpacity = alphaGrad * gaussian * timeWeight * opacity * (1.0 - opacity);
						}
					}
				let gradient = Splat(vec4<f32>(gradPosition, gradStaticMix), vec4<f32>(gradVelocity, gradTime),
					vec4<f32>(gradHarmonic, meanAlpha), vec4<f32>(gradLogScale, 0.0), gradRotation,
					vec4<f32>(gradColor, gradOpacity));
				sampleGradients[s * cfg.splatCount + i] = gradient;
			}
	}
`;
}

const UPDATE_WGSL = `
	struct Splat { centerStatic: vec4<f32>, velocityTime: vec4<f32>, harmonicPad: vec4<f32>,
		logScalePad: vec4<f32>, rotation: vec4<f32>, colorOpacity: vec4<f32> };
	struct TrainConfig {
		width: u32, height: u32, frameCount: u32, splatCount: u32,
		sampleCount: u32, step: u32, modelMode: u32, motionSampleCount: u32,
		lrPosition: f32, lrColor: f32, lrOpacity: f32, lrMotion: f32,
		minRadius: f32, maxRadius: f32, temporalSigma: f32, targetAspect: f32,
		motionSamplePermil: u32, motionCoverageTarget: f32, motionCoverageWeight: f32,
		staticAlphaWeight: f32, opacityDecayWeight: f32, staticEnergyThreshold: f32,
		staticSampleCount: u32, staticSamplePermil: u32,
		beta1: f32, beta2: f32, adamEpsilon: f32, statDecay: f32, robustMix: f32,
		trainViewCount: u32, cameraCount: u32, geometryScale: f32,
		camerasPerStep: u32, cameraRotationStart: u32, legacyAllCameraSampling: u32,
	};
	@group(0) @binding(0) var<uniform> cfg: TrainConfig;
	@group(0) @binding(1) var<storage, read> paramsIn: array<Splat>;
	@group(0) @binding(2) var<storage, read_write> paramsOut: array<Splat>;
	@group(0) @binding(3) var<storage, read_write> firstMoment: array<Splat>;
	@group(0) @binding(4) var<storage, read_write> secondMoment: array<Splat>;
	@group(0) @binding(5) var<storage, read_write> splatStats: array<vec4<f32>>;
	@group(0) @binding(6) var<storage, read> sampleGradients: array<Splat>;
	@group(0) @binding(7) var<storage, read> sampleLosses: array<f32>;

	@compute @workgroup_size(256)
	fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
		let i = gid.x;
		if (i >= cfg.splatCount) { return; }
		var gradient = Splat(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
			vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
		var batchLoss = 0.0;
		for (var s = 0u; s < cfg.sampleCount; s = s + 1u) {
			let sample = sampleGradients[s * cfg.splatCount + i];
			gradient.centerStatic += sample.centerStatic;
			gradient.velocityTime += sample.velocityTime;
			gradient.harmonicPad += sample.harmonicPad;
			gradient.logScalePad += sample.logScalePad;
			gradient.rotation += sample.rotation;
			gradient.colorOpacity += sample.colorOpacity;
			if (i == 0u) { batchLoss += sampleLosses[s] / f32(cfg.sampleCount); }
		}
		let meanAlpha = gradient.harmonicPad.w;
		gradient.harmonicPad.w = 0.0;
		var p = paramsIn[i]; var m = firstMoment[i]; var v = secondMoment[i];
		m.centerStatic = cfg.beta1 * m.centerStatic + (1.0 - cfg.beta1) * gradient.centerStatic;
		m.velocityTime = cfg.beta1 * m.velocityTime + (1.0 - cfg.beta1) * gradient.velocityTime;
		m.harmonicPad = cfg.beta1 * m.harmonicPad + (1.0 - cfg.beta1) * gradient.harmonicPad;
		m.logScalePad = cfg.beta1 * m.logScalePad + (1.0 - cfg.beta1) * gradient.logScalePad;
		m.rotation = cfg.beta1 * m.rotation + (1.0 - cfg.beta1) * gradient.rotation;
		m.colorOpacity = cfg.beta1 * m.colorOpacity + (1.0 - cfg.beta1) * gradient.colorOpacity;
		v.centerStatic = cfg.beta2 * v.centerStatic + (1.0 - cfg.beta2) * gradient.centerStatic * gradient.centerStatic;
		v.velocityTime = cfg.beta2 * v.velocityTime + (1.0 - cfg.beta2) * gradient.velocityTime * gradient.velocityTime;
		v.harmonicPad = cfg.beta2 * v.harmonicPad + (1.0 - cfg.beta2) * gradient.harmonicPad * gradient.harmonicPad;
		v.logScalePad = cfg.beta2 * v.logScalePad + (1.0 - cfg.beta2) * gradient.logScalePad * gradient.logScalePad;
		v.rotation = cfg.beta2 * v.rotation + (1.0 - cfg.beta2) * gradient.rotation * gradient.rotation;
		v.colorOpacity = cfg.beta2 * v.colorOpacity + (1.0 - cfg.beta2) * gradient.colorOpacity * gradient.colorOpacity;
		firstMoment[i] = m; secondMoment[i] = v;
		let adamStep = f32(cfg.step + 1u); let mc = max(1e-6, 1.0 - pow(cfg.beta1, adamStep));
		let vc = max(1e-6, 1.0 - pow(cfg.beta2, adamStep));
		let posUpdate = (m.centerStatic / mc) / (sqrt(v.centerStatic / vc) + vec4<f32>(cfg.adamEpsilon));
		let velocityUpdate = (m.velocityTime / mc) / (sqrt(v.velocityTime / vc) + vec4<f32>(cfg.adamEpsilon));
		let harmonicUpdate = (m.harmonicPad / mc) / (sqrt(v.harmonicPad / vc) + vec4<f32>(cfg.adamEpsilon));
		let scaleUpdate = (m.logScalePad / mc) / (sqrt(v.logScalePad / vc) + vec4<f32>(cfg.adamEpsilon));
		let rotationUpdate = (m.rotation / mc) / (sqrt(v.rotation / vc) + vec4<f32>(cfg.adamEpsilon));
		let colorUpdate = (m.colorOpacity / mc) / (sqrt(v.colorOpacity / vc) + vec4<f32>(cfg.adamEpsilon));
		p.centerStatic = vec4<f32>(p.centerStatic.xyz - cfg.lrPosition * posUpdate.xyz,
			clamp(p.centerStatic.w - cfg.lrMotion * posUpdate.w, 0.0, 1.0));
		p.velocityTime = vec4<f32>(clamp(p.velocityTime.xyz - cfg.lrMotion * velocityUpdate.xyz,
			vec3<f32>(-2.0 * cfg.geometryScale), vec3<f32>(2.0 * cfg.geometryScale)),
			clamp(p.velocityTime.w - cfg.lrMotion * velocityUpdate.w, 0.0, 1.0));
		p.harmonicPad = vec4<f32>(clamp(p.harmonicPad.xyz - cfg.lrMotion * harmonicUpdate.xyz,
			vec3<f32>(-1.5 * cfg.geometryScale), vec3<f32>(1.5 * cfg.geometryScale)), p.harmonicPad.w);
		let minLogScale = log(max(1e-6, 0.03 * cfg.geometryScale));
		let maxLogScale = log(max(2e-6, cfg.geometryScale));
		var nextLogScale = clamp(p.logScalePad.xyz - 0.10 * cfg.lrPosition * scaleUpdate.xyz,
			vec3<f32>(minLogScale), vec3<f32>(maxLogScale));
		let meanLogScale = (nextLogScale.x + nextLogScale.y + nextLogScale.z) / 3.0;
		// Bound scale conditioning without forcing spheres. Long unconstrained
		// needles make the all-splat sampled fallback especially unstable.
		let halfLogAspectLimit = 0.5 * log(${MAX_SAMPLED_SCALE_ASPECT_RATIO}.0);
		nextLogScale = clamp(nextLogScale, vec3<f32>(meanLogScale - halfLogAspectLimit),
			vec3<f32>(meanLogScale + halfLogAspectLimit));
		p.logScalePad = vec4<f32>(nextLogScale, p.logScalePad.w);
		let rotationTrial = p.rotation - 0.25 * cfg.lrMotion * rotationUpdate;
		let rotationNorm2 = dot(rotationTrial, rotationTrial);
		p.rotation = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), rotationTrial * inverseSqrt(max(rotationNorm2, 1e-16)),
			rotationNorm2 > 1e-16);
		p.colorOpacity = vec4<f32>(clamp(p.colorOpacity.xyz - cfg.lrColor * colorUpdate.xyz,
			vec3<f32>(0.0), vec3<f32>(${MAX_SPLAT_COLOR}.0)),
			clamp(p.colorOpacity.w - cfg.lrOpacity * colorUpdate.w, -7.0, 3.0));
		paramsOut[i] = p;
		let observed = vec4<f32>(length(gradient.centerStatic.xyz), meanAlpha,
			abs(gradient.colorOpacity.w), length(gradient.velocityTime.xyz));
		let decayed = cfg.statDecay * splatStats[i] + (1.0 - cfg.statDecay) * observed;
		splatStats[i] = select(decayed, vec4<f32>(decayed.xyz, batchLoss), i == 0u);
	}
`;

const MAINTENANCE_WGSL = `
	struct Splat { centerStatic: vec4<f32>, velocityTime: vec4<f32>, harmonicPad: vec4<f32>,
		logScalePad: vec4<f32>, rotation: vec4<f32>, colorOpacity: vec4<f32> };
	@group(0) @binding(0) var<storage, read_write> params: array<Splat>;
	@group(0) @binding(1) var<storage, read_write> firstMoment: array<Splat>;
	@group(0) @binding(2) var<storage, read_write> secondMoment: array<Splat>;
	@group(0) @binding(3) var<storage, read_write> splatStats: array<vec4<f32>>;
	fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
	fn safe_quaternion(raw: vec4<f32>) -> vec4<f32> {
		let norm2 = dot(raw, raw); let normalized = raw * inverseSqrt(max(norm2, 1e-16));
		return select(vec4<f32>(0.0, 0.0, 0.0, 1.0), normalized, norm2 > 1e-16);
	}
	fn quaternion_matrix(raw: vec4<f32>) -> mat3x3<f32> {
		let q = safe_quaternion(raw); let x=q.x; let y=q.y; let z=q.z; let w=q.w;
		return mat3x3<f32>(
			vec3<f32>(1.0-2.0*(y*y+z*z), 2.0*(x*y+z*w), 2.0*(x*z-y*w)),
			vec3<f32>(2.0*(x*y-z*w), 1.0-2.0*(x*x+z*z), 2.0*(y*z+x*w)),
			vec3<f32>(2.0*(x*z+y*w), 2.0*(y*z-x*w), 1.0-2.0*(x*x+y*y)));
	}
	fn zero_splat() -> Splat {
		return Splat(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
			vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
	}
	@compute @workgroup_size(1)
	fn split_recycle() {
		let count = arrayLength(&params); if (count < 8u) { return; }
		var victims: array<u32, 4>; var parents: array<u32, 4>;
		for (var slot = 0u; slot < 4u; slot = slot + 1u) {
			var best = 1e30; var bestIndex = 0xffffffffu;
			for (var i = 0u; i < count; i = i + 1u) {
				var used = false; for (var prior = 0u; prior < slot; prior = prior + 1u) { used = used || victims[prior] == i; }
				let largestScale=max(params[i].logScalePad.x,max(params[i].logScalePad.y,params[i].logScalePad.z));
				let score = sigmoid(params[i].colorOpacity.w) + 12.0 * splatStats[i].y
					- 2.0 * exp(largestScale);
				if (!used && score < best) { best = score; bestIndex = i; }
			}
			victims[slot] = bestIndex;
		}
		for (var slot = 0u; slot < 4u; slot = slot + 1u) {
			var best = -1.0; var bestIndex = 0xffffffffu;
			for (var i = 0u; i < count; i = i + 1u) {
				var used = false;
				for (var prior = 0u; prior < 4u; prior = prior + 1u) { used = used || victims[prior] == i; }
				for (var prior = 0u; prior < slot; prior = prior + 1u) { used = used || parents[prior] == i; }
				let score = splatStats[i].x + 4.0 * splatStats[i].y + splatStats[i].w;
				if (!used && score > best) { best = score; bestIndex = i; }
			}
			parents[slot] = bestIndex;
		}
		for (var slot = 0u; slot < 4u; slot = slot + 1u) {
			let victim = victims[slot]; let parentIndex = parents[slot]; var parent = params[parentIndex]; var child = parent;
			var axis = 0u;
			if (parent.logScalePad.y > parent.logScalePad.x) { axis = 1u; }
			if (parent.logScalePad.z > parent.logScalePad[axis]) { axis = 2u; }
			let rotation = quaternion_matrix(parent.rotation); let offset = rotation[axis] * exp(parent.logScalePad[axis]) * 0.28;
			parent.centerStatic = vec4<f32>(parent.centerStatic.xyz - offset, max(0.0, parent.centerStatic.w - 0.04));
			child.centerStatic = vec4<f32>(child.centerStatic.xyz + offset, max(0.0, child.centerStatic.w - 0.04));
			let shrink = vec3<f32>(log(0.80)); parent.logScalePad = vec4<f32>(parent.logScalePad.xyz + shrink, 0.0);
			child.logScalePad = vec4<f32>(child.logScalePad.xyz + shrink, 0.0);
			if (splatStats[parentIndex].w > splatStats[parentIndex].x) {
				parent.velocityTime.w = clamp(parent.velocityTime.w - 0.035, 0.0, 1.0);
				child.velocityTime.w = clamp(child.velocityTime.w + 0.035, 0.0, 1.0);
			}
			let opacity = clamp(sigmoid(parent.colorOpacity.w), 1e-4, 0.999);
			let halfOpacity = clamp(1.0 - sqrt(1.0 - opacity), 1e-4, 0.999);
			let splitLogit = log(halfOpacity / (1.0 - halfOpacity));
			parent.colorOpacity.w = splitLogit; child.colorOpacity.w = splitLogit;
			params[parentIndex] = parent; params[victim] = child;
			firstMoment[parentIndex] = zero_splat(); secondMoment[parentIndex] = zero_splat();
			firstMoment[victim] = zero_splat(); secondMoment[victim] = zero_splat();
			splatStats[parentIndex] = splatStats[parentIndex] * 0.5; splatStats[victim] = vec4<f32>(0.0);
		}
	}
`;

const RENDER_SORT_WGSL = `
	struct Splat { centerStatic: vec4<f32>, velocityTime: vec4<f32>, harmonicPad: vec4<f32>,
		logScalePad: vec4<f32>, rotation: vec4<f32>, colorOpacity: vec4<f32> };
	struct Camera { row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, row3: vec4<f32>, intrinsics: vec4<f32> };
	struct RenderConfig { width: f32, height: f32, time: f32, splatCount: f32, pointScale: f32,
		modelMode: f32, targetAspect: f32, temporalSigma: f32, targetWidth: f32, targetHeight: f32,
		renderMode: f32, viewIndex: f32 };
	@group(0) @binding(0) var<uniform> cfg: RenderConfig;
	@group(0) @binding(1) var<storage, read> params: array<Splat>;
	@group(0) @binding(2) var<storage, read> cameras: array<Camera>;
	@group(0) @binding(3) var<storage, read_write> renderOrder: array<u32>;
	@group(0) @binding(4) var<storage, read_write> renderDepths: array<f32>;
	fn center(p: Splat) -> vec3<f32> {
		let tc = cfg.time * 2.0 - 1.0; var result = p.centerStatic.xyz + p.velocityTime.xyz * tc;
		if (cfg.modelMode < 0.5) { result = result + p.harmonicPad.xyz * sin(cfg.time * 6.28318530718); }
		return result;
	}
	@compute @workgroup_size(256)
	fn sort_render_order(@builtin(local_invocation_id) lid: vec3<u32>) {
		let count = u32(round(cfg.splatCount)); let capacity = arrayLength(&renderOrder);
		let camera = cameras[u32(round(cfg.viewIndex))];
		for (var i = lid.x; i < capacity; i = i + 256u) {
			renderOrder[i] = i;
			if (i < count) {
				let h = vec4<f32>(center(params[i]), 1.0); renderDepths[i] = dot(camera.row2, h);
			} else { renderDepths[i] = -1e30; }
		}
		storageBarrier(); workgroupBarrier();
		for (var width = 2u; width <= capacity; width = width * 2u) {
			var stride = width / 2u;
			loop {
				for (var i = lid.x; i < capacity; i = i + 256u) {
					let partner = i ^ stride;
					if (partner > i) {
						let left = renderOrder[i]; let right = renderOrder[partner];
						let descending = (i & width) == 0u;
						let swap = select(renderDepths[left] > renderDepths[right],
							renderDepths[left] < renderDepths[right], descending);
						if (swap) { renderOrder[i] = right; renderOrder[partner] = left; }
					}
				}
				storageBarrier(); workgroupBarrier();
				if (stride == 1u) { break; } stride = stride / 2u;
			}
		}
	}
`;

function renderWgsl({
	pixelFilterMode = "legacy-floor",
	opacityModel = "coupled",
	materialOpacityBias = Math.log(99),
} = {}) {
	const compensatedFilter = pixelFilterMode === "mip-2d-compensated";
	const dualOpacity = opacityModel === "dual";
	return `
	struct Splat { centerStatic: vec4<f32>, velocityTime: vec4<f32>, harmonicPad: vec4<f32>,
		logScalePad: vec4<f32>, rotation: vec4<f32>, colorOpacity: vec4<f32> };
	struct Camera { row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, row3: vec4<f32>, intrinsics: vec4<f32> };
	struct RenderConfig { width: f32, height: f32, time: f32, splatCount: f32, pointScale: f32,
		modelMode: f32, targetAspect: f32, temporalSigma: f32, targetWidth: f32, targetHeight: f32,
		renderMode: f32, viewIndex: f32 };
	struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) local: vec2<f32>,
		@location(1) color: vec3<f32>, @location(2) opacity: f32 };
	@group(0) @binding(0) var<uniform> cfg: RenderConfig;
	@group(0) @binding(1) var<storage, read> params: array<Splat>;
	@group(0) @binding(2) var<storage, read> cameras: array<Camera>;
	@group(0) @binding(3) var<storage, read> renderOrder: array<u32>;
	fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
	fn outer3(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> { return mat3x3<f32>(a*b.x, a*b.y, a*b.z); }
	fn safe_quaternion(raw: vec4<f32>) -> vec4<f32> {
		let norm2 = dot(raw, raw); let normalized = raw * inverseSqrt(max(norm2, 1e-16));
		return select(vec4<f32>(0.0, 0.0, 0.0, 1.0), normalized, norm2 > 1e-16);
	}
	fn quaternion_matrix(raw: vec4<f32>) -> mat3x3<f32> {
		let q = safe_quaternion(raw); let x=q.x; let y=q.y; let z=q.z; let w=q.w;
		return mat3x3<f32>(
			vec3<f32>(1.0-2.0*(y*y+z*z), 2.0*(x*y+z*w), 2.0*(x*z-y*w)),
			vec3<f32>(2.0*(x*y-z*w), 1.0-2.0*(x*x+z*z), 2.0*(y*z+x*w)),
			vec3<f32>(2.0*(x*z+y*w), 2.0*(y*z-x*w), 1.0-2.0*(x*x+y*y)));
	}
	fn camera_rotation(camera: Camera) -> mat3x3<f32> {
		return mat3x3<f32>(vec3<f32>(camera.row0.x,camera.row1.x,camera.row2.x),
			vec3<f32>(camera.row0.y,camera.row1.y,camera.row2.y),
			vec3<f32>(camera.row0.z,camera.row1.z,camera.row2.z));
	}
	fn center(p: Splat) -> vec3<f32> {
		let tc = cfg.time * 2.0 - 1.0; var result = p.centerStatic.xyz + p.velocityTime.xyz * tc;
		if (cfg.modelMode < 0.5) { result = result + p.harmonicPad.xyz * sin(cfg.time * 6.28318530718); }
		return result;
	}
	fn temporal_gate(p: Splat) -> f32 {
		let sigma = clamp(cfg.temporalSigma, 0.12, 0.36); let floorValue = clamp(sigma * 0.30, 0.035, 0.12);
		let dt = cfg.time - clamp(p.velocityTime.w, 0.0, 1.0);
		let dynamicGate = floorValue + (1.0 - floorValue) * exp(-0.5 * dt * dt / (sigma * sigma));
		return mix(dynamicGate, 1.0, clamp(p.centerStatic.w, 0.0, 1.0));
	}
	@vertex fn vs_main(@builtin(instance_index) iid: u32, @location(0) quad: vec2<f32>) -> VSOut {
		let p = params[renderOrder[iid]]; let camera = cameras[u32(round(cfg.viewIndex))]; let h = vec4<f32>(center(p), 1.0);
		let cp = vec3<f32>(dot(camera.row0, h), dot(camera.row1, h), dot(camera.row2, h));
		if (cp.z <= 0.1) { return VSOut(vec4<f32>(2.0, 2.0, 0.0, 1.0), quad, p.colorOpacity.xyz, 0.0); }
		let basis = camera_rotation(camera) * quaternion_matrix(p.rotation);
		let variances = exp(2.0 * clamp(p.logScalePad.xyz, vec3<f32>(-16.0), vec3<f32>(4.0)));
		let sigmaCamera = variances.x * outer3(basis[0],basis[0]) + variances.y * outer3(basis[1],basis[1])
			+ variances.z * outer3(basis[2],basis[2]);
		let invZ = 1.0 / cp.z; let horizontalFocal = cfg.targetAspect * camera.intrinsics.x;
		let j0 = vec3<f32>(horizontalFocal*invZ, 0.0, -horizontalFocal*cp.x*invZ*invZ);
		let j1 = vec3<f32>(0.0, camera.intrinsics.y*invZ, -camera.intrinsics.y*cp.y*invZ*invZ);
		// Use display height here so preview filtering matches its own pixel
		// footprint rather than the lower-resolution training raster.
		let filterVariance = pow(${FILTER_SIGMA_PIXELS} / max(1.0, cfg.height), 2.0);
		let unfilteredC00=dot(j0,sigmaCamera*j0);
		let c01=dot(j0,sigmaCamera*j1);let unfilteredC11=dot(j1,sigmaCamera*j1);
		let c00=unfilteredC00+filterVariance;let c11=unfilteredC11+filterVariance;
		let opacityCompensation=${compensatedFilter
			? "clamp(sqrt(max(unfilteredC00*unfilteredC11-c01*c01,0.0)/max(c00*c11-c01*c01,1e-16)),0.0,1.0)"
			: "1.0"};
		let l00 = sqrt(max(c00, 1e-12)); let l10 = c01 / l00;
		let l11 = sqrt(max(c11 - l10*l10, 1e-12));
		let offsetMetric = 3.0 * vec2<f32>(l00*quad.x, l10*quad.x + l11*quad.y);
		let projected = vec2<f32>(camera.intrinsics.x * cp.x * invZ + camera.intrinsics.z,
			camera.intrinsics.y * cp.y * invZ + camera.intrinsics.w);
		let ndc = vec2<f32>(projected.x * 2.0 - 1.0, 1.0 - projected.y * 2.0);
		let offset = vec2<f32>(2.0 * offsetMetric.x / cfg.targetAspect, -2.0 * offsetMetric.y);
		return VSOut(vec4<f32>(ndc + offset, 0.0, 1.0), quad * 3.0, p.colorOpacity.xyz,
			sigmoid(p.colorOpacity.w)*temporal_gate(p)*opacityCompensation
				*${dualOpacity ? `sigmoid(p.harmonicPad.w+${materialOpacityBias})` : "1.0"});
	}
	@fragment fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
		let qform = dot(input.local, input.local); if (qform > 9.0) { discard; }
		let rawAlpha = input.opacity * exp(-0.5 * qform);
		if (rawAlpha < 0.00392156863) { discard; }
		let alpha = min(0.99, rawAlpha);
		if (cfg.renderMode >= 1.5) { let a = clamp(alpha * 8.0, 0.0, 1.0); return vec4<f32>(0.1 * a, a, 0.65 * a, a); }
		if (cfg.renderMode >= 0.5) { let a = clamp(alpha * 8.0, 0.0, 1.0); return vec4<f32>(input.color * a, a); }
		return vec4<f32>(input.color * alpha, alpha);
	}
`;
}

function temporalGateCpu(params, base, time, sigma) {
	const floor = Math.min(0.12, Math.max(0.035, sigma * 0.30));
	const dt = time - params[base + 7];
	const dynamic = floor + (1 - floor) * Math.exp(-0.5 * dt * dt / (sigma * sigma));
	return dynamic * (1 - params[base + 3]) + params[base + 3];
}

function projectSplatAtTimeCpu(dataset, params, base, view, time, modelMode) {
	const tc = time * 2 - 1; const wave = modelMode === 0 ? Math.sin(time * Math.PI * 2) : 0;
	return projectAnisotropicGaussianCpu({
		center: [params[base] + params[base + 4] * tc + params[base + 8] * wave,
			params[base + 1] + params[base + 5] * tc + params[base + 9] * wave,
			params[base + 2] + params[base + 6] * tc + params[base + 10] * wave],
		logScales: [params[base + 12], params[base + 13], params[base + 14]],
		quaternion: [params[base + 16], params[base + 17], params[base + 18], params[base + 19]],
		camera: dataset.cameras[view], aspect: dataset.width / dataset.height, height: dataset.height,
	});
}

function projectFrameSplatsCpu(dataset, params, splatCount, view, frame, modelMode, temporalSigma) {
	const t = frameTime(frame, dataset.frameCount);
	return sortProjectedSplatsBackToFront(Array.from({ length: splatCount }, (_, i) => {
		const base = i * SPLAT_FLOATS;
		const projection = projectSplatAtTimeCpu(dataset, params, base, view, t, modelMode);
		return { index: i, projection, opacity: sigmoid(params[base + 23]),
			timeWeight: temporalGateCpu(params, base, t, temporalSigma),
			colorR: params[base + 20], colorG: params[base + 21], colorB: params[base + 22] };
	}));
}

export function sortProjectedSplatsBackToFront(splats) {
	return splats.sort((left, right) => {
		const leftDepth = left.projection.valid ? left.projection.cameraPoint[2] : Number.NEGATIVE_INFINITY;
		const rightDepth = right.projection.valid ? right.projection.cameraPoint[2] : Number.NEGATIVE_INFINITY;
		return rightDepth - leftDepth || (left.index ?? 0) - (right.index ?? 0);
	});
}

function evalModelCpu(dataset, projections, px, py) {
	let colorR = 0; let colorG = 0; let colorB = 0; let transmittance = 1;
	const sampleX = px * dataset.width / dataset.height;
	for (const splat of projections) {
		if (!splat.projection.valid) continue;
		const dx = sampleX - splat.projection.center[0]; const dy = py - splat.projection.center[1];
		const conic = splat.projection.conic;
		const qform = conic[0] * dx * dx + 2 * conic[1] * dx * dy + conic[2] * dy * dy;
		if (!Number.isFinite(qform) || qform < 0 || qform > 9) continue;
		const alpha = splat.opacity * splat.timeWeight * Math.exp(-0.5 * qform);
		colorR = colorR * (1 - alpha) + splat.colorR * alpha;
		colorG = colorG * (1 - alpha) + splat.colorG * alpha;
		colorB = colorB * (1 - alpha) + splat.colorB * alpha;
		transmittance *= 1 - alpha;
	}
	return { colorR, colorG, colorB, coverage: 1 - transmittance };
}

function computeViewMetrics(dataset, params, splatCount, views, modelMode, temporalSigma, gridSize) {
	let mse = 0; let mae = 0; let count = 0; let coverage = 0;
	let sx = 0; let sy = 0; let sx2 = 0; let sy2 = 0; let sxy = 0;
	const pixels = dataset.width * dataset.height;
	for (const view of views) for (let frame = 0; frame < dataset.frameCount; frame += 1) {
		const projections = projectFrameSplatsCpu(dataset, params, splatCount, view, frame,
			modelMode, temporalSigma);
		for (let gy = 0; gy < gridSize; gy += 1) {
			for (let gx = 0; gx < gridSize; gx += 1) {
				const x = Math.min(dataset.width - 1, Math.floor((gx + 0.5) * dataset.width / gridSize));
				const y = Math.min(dataset.height - 1, Math.floor((gy + 0.5) * dataset.height / gridSize));
				const result = evalModelCpu(dataset, projections,
					(x + 0.5) / dataset.width, (y + 0.5) / dataset.height);
				const base = ((view * dataset.frameCount + frame) * pixels + y * dataset.width + x) * 4;
				const targetR = readFrameBankValue(dataset, base);
				const targetG = readFrameBankValue(dataset, base + 1);
				const targetB = readFrameBankValue(dataset, base + 2);
				const errorR = result.colorR - targetR; const errorG = result.colorG - targetG;
				const errorB = result.colorB - targetB;
				mse += (errorR * errorR + errorG * errorG + errorB * errorB) / 3;
				mae += (Math.abs(errorR) + Math.abs(errorG) + Math.abs(errorB)) / 3;
				const pl = 0.2126 * result.colorR + 0.7152 * result.colorG + 0.0722 * result.colorB;
				const tl = 0.2126 * targetR + 0.7152 * targetG + 0.0722 * targetB;
				sx += pl; sy += tl; sx2 += pl * pl; sy2 += tl * tl; sxy += pl * tl;
				coverage += result.coverage; count += 1;
			}
		}
	}
	const meanX = sx / count; const meanY = sy / count;
	const vx = Math.max(0, sx2 / count - meanX * meanX); const vy = Math.max(0, sy2 / count - meanY * meanY);
	const cov = sxy / count - meanX * meanY; const c1 = 0.0001; const c2 = 0.0009;
	const loss = mse / count;
	return { loss, mae: mae / count, psnr: -10 * Math.log10(Math.max(1e-12, loss)),
		ssim: ((2 * meanX * meanY + c1) * (2 * cov + c2)) / ((meanX * meanX + meanY * meanY + c1) * (vx + vy + c2)),
		coverage: coverage / count };
}

export function computeSnapshotValidationMetrics(dataset, params, {
	splatCount = params.length / SPLAT_FLOATS,
	modelMode = 0,
	temporalSigma = 0.30,
	gridSize = 12,
} = {}) {
	const trainViews = resolveTrainViewIndices(dataset);
	const train = computeViewMetrics(dataset, params, splatCount, trainViews, modelMode, temporalSigma,
		Math.min(20, gridSize));
	const heldout = computeViewMetrics(dataset, params, splatCount, [dataset.heldoutViewIndex], modelMode,
		temporalSigma, Math.min(20, gridSize));
	let active = 0; let opacity = 0; let maxOpacity = 0; let radius = 0; let aspectRatio = 0;
	for (let index = 0; index < splatCount; index += 1) {
		const base = index * SPLAT_FLOATS; const value = sigmoid(params[base + 23]);
		opacity += value; maxOpacity = Math.max(maxOpacity, value);
		const scales = [Math.exp(params[base + 12]), Math.exp(params[base + 13]), Math.exp(params[base + 14])];
		radius += Math.cbrt(scales[0] * scales[1] * scales[2]);
		aspectRatio += Math.max(...scales) / Math.max(1e-8, Math.min(...scales));
		if (value > 0.05) active += 1;
	}
	return {
		gridLoss: train.loss, gridMae: train.mae, gridPsnr: train.psnr, gridSsim: train.ssim,
		motionLoss: Number.NaN, motionCoverage: train.coverage, staticCoverage: Number.NaN,
		motionMaxAlpha: maxOpacity, heldoutLoss: heldout.loss, heldoutMae: heldout.mae,
		heldoutPsnr: heldout.psnr, heldoutSsim: heldout.ssim, heldoutCoverage: heldout.coverage,
		activeSplats: active, meanOpacity: opacity / splatCount, meanRadius: radius / splatCount,
		meanAspectRatio: aspectRatio / splatCount,
	};
}

export class DynamicSplatWebGpu3dTrainer {
	constructor(canvas) {
		this.canvas = canvas; this.device = null; this.dataset = null; this.currentIndex = 0;
		this.stepCount = 0; this.splatCount = 768; this.totalRecycled = 0; this.readbackChain = Promise.resolve();
		this.requestTimestampQueries = false;
		this.supportsTimestampQuery = false;
		this.timestampQueryEnabled = false;
		this.configBytes = new ArrayBuffer(144);
		this.renderConfigBytes = Array.from({ length: MAX_RENDER_VIEWS }, () => new ArrayBuffer(48));
		this.pixelFilterMode = "legacy-floor";
		this.opacityModel = "coupled";
		this.materialOpacityBias = Math.log(99);
	}

	targetBufferByteLength(dataset) {
		return dataset.frames.byteLength;
	}

	initializeTargetBuffer(target) {
		this.device.queue.writeBuffer(target, 0, this.dataset.frames);
	}

	async init(dataset, { splatCount = 768, requiredWorkgroupStorageSize = 0 } = {}) {
		if (splatCount > MAX_BROWSER_RENDER_SPLATS) {
			throw new RangeError(`The browser render path supports at most ${MAX_BROWSER_RENDER_SPLATS} splats.`);
		}
		if (splatCount > 2048 && !this.skipSampleGradientAllocation) {
			throw new RangeError("The sampled-ray depth-order cache supports at most 2048 splats; "
				+ "use the tiled full-frame backend above that count.");
		}
		const sampleGradientBytes = this.skipSampleGradientAllocation
			? SPLAT_BYTES : sampleGradientBufferBytes(splatCount);
		if (!navigator.gpu) throw new Error("WebGPU unavailable in this browser.");
		const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error("WebGPU adapter unavailable.");
		this.adapterName = adapter.info?.description || adapter.info?.vendor || "WebGPU";
		this.supportsShaderF16 = adapter.features.has("shader-f16");
		this.supportsTimestampQuery = adapter.features.has("timestamp-query");
		const requiredFeatures = this.requestTimestampQueries && this.supportsTimestampQuery
			? ["timestamp-query"] : [];
		this.timestampQueryEnabled = requiredFeatures.length > 0;
		const requiredStorageBuffers = 9;
		if (adapter.limits.maxStorageBuffersPerShaderStage < requiredStorageBuffers) {
			throw new Error(`This trainer needs ${requiredStorageBuffers} storage buffers per shader stage; `
				+ `the adapter supports ${adapter.limits.maxStorageBuffersPerShaderStage}.`);
		}
		if (adapter.limits.maxComputeWorkgroupStorageSize < requiredWorkgroupStorageSize) {
			throw new Error(`This trainer needs ${requiredWorkgroupStorageSize} bytes of workgroup storage; `
				+ `the adapter supports ${adapter.limits.maxComputeWorkgroupStorageSize}.`);
		}
		const requiredLimits = { maxStorageBuffersPerShaderStage: requiredStorageBuffers };
		if (requiredWorkgroupStorageSize > 0) {
			requiredLimits.maxComputeWorkgroupStorageSize = requiredWorkgroupStorageSize;
		}
		this.device = await adapter.requestDevice({
			requiredFeatures,
			requiredLimits,
		});
		this.storageBufferLimit = Math.min(this.device.limits.maxStorageBufferBindingSize,
			this.device.limits.maxBufferSize);
		assertStorageBufferFits("The training target", this.targetBufferByteLength(dataset),
			this.storageBufferLimit);
		if (sampleGradientBytes > this.storageBufferLimit) {
			throw new RangeError(`splatCount ${splatCount} needs a ${sampleGradientBytes}-byte gradient buffer; `
				+ `this device supports ${this.storageBufferLimit} bytes.`);
		}
		this.context = this.canvas?.getContext("webgpu") ?? null;
		this.format = navigator.gpu.getPreferredCanvasFormat();
		this.context?.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
		this.dataset = normalizeDatasetGeometry(dataset); this.splatCount = splatCount;
		this.trainViewIndices = resolveTrainViewIndices(this.dataset);
		this.legacyContiguousTrainViews = this.trainViewIndices.every((view, index) => view === index);
		await this.createPipelines(); this.createBuffers(); this.createBindGroups();
	}

	async createPipelines() {
		const order = this.device.createShaderModule({ code: ORDER_WGSL });
		const gradients = this.device.createShaderModule({
			code: trainWgsl(resolveFrameBank(this.dataset).format),
		});
		const update = this.device.createShaderModule({ code: UPDATE_WGSL });
		const maintenance = this.device.createShaderModule({ code: MAINTENANCE_WGSL });
		const renderSort = this.device.createShaderModule({ code: RENDER_SORT_WGSL });
		const render = this.device.createShaderModule({ code: renderWgsl({
			pixelFilterMode: this.pixelFilterMode,
			opacityModel: this.opacityModel,
			materialOpacityBias: this.materialOpacityBias,
		}) });
		const modules = [["order", order], ["training", gradients], ["update", update], ["maintenance", maintenance],
			["render-sort", renderSort], ["render", render]];
		const diagnostics = await Promise.all(modules.map(async ([name, module]) => ({
			name, info: await module.getCompilationInfo(),
		})));
		const errors = diagnostics.flatMap(({ name, info }) => info.messages
			.filter((message) => message.type === "error")
			.map((message) => `${name}:${message.lineNum}:${message.linePos} ${message.message}`));
		if (errors.length) throw new Error(`WGSL compilation failed:\n${errors.join("\n")}`);
		this.device.pushErrorScope("validation");
		this.pipelines = {
			order: this.device.createComputePipeline({ layout: "auto", compute: { module: order, entryPoint: "build_order" } }),
			gradients: this.device.createComputePipeline({ layout: "auto",
				compute: { module: gradients, entryPoint: "sample_gradients" } }),
			update: this.device.createComputePipeline({ layout: "auto", compute: { module: update, entryPoint: "update" } }),
			maintenance: this.device.createComputePipeline({ layout: "auto",
				compute: { module: maintenance, entryPoint: "split_recycle" } }),
			renderSort: this.device.createComputePipeline({ layout: "auto",
				compute: { module: renderSort, entryPoint: "sort_render_order" } }),
			render: this.device.createRenderPipeline({ layout: "auto", vertex: { module: render, entryPoint: "vs_main",
				buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
				fragment: { module: render, entryPoint: "fs_main", targets: [{ format: this.format, blend: {
					color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
					alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" } } }] },
				primitive: { topology: "triangle-strip" } }),
		};
		const pipelineError = await this.device.popErrorScope();
		if (pipelineError) throw new Error(`WebGPU pipeline validation failed: ${pipelineError.message}`);
	}

	createBuffers() {
		const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
		const params = makeInitialSplats(this.dataset, this.splatCount); this.initialParams = params.slice();
		const makeBuffer = (size, bufferUsage = usage) => this.device.createBuffer({ size, usage: bufferUsage });
		const paramA = makeBuffer(params.byteLength); const paramB = makeBuffer(params.byteLength);
		this.device.queue.writeBuffer(paramA, 0, params); this.device.queue.writeBuffer(paramB, 0, params);
		const firstMoment = makeBuffer(params.byteLength); const secondMoment = makeBuffer(params.byteLength);
		this.device.queue.writeBuffer(firstMoment, 0, new Float32Array(params.length));
		this.device.queue.writeBuffer(secondMoment, 0, new Float32Array(params.length));
		const stats = makeBuffer(this.splatCount * 16); this.device.queue.writeBuffer(stats, 0, new Float32Array(this.splatCount * 4));
		const sampleGradients = makeBuffer(this.skipSampleGradientAllocation
			? SPLAT_BYTES : sampleGradientBufferBytes(this.splatCount));
		const sampleLosses = makeBuffer(MAX_SAMPLES_PER_STEP * 4);
		const target = makeBuffer(this.targetBufferByteLength(this.dataset),
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
		this.initializeTargetBuffer(target);
		const cameraData = packCameras(this.dataset.cameras); const cameras = makeBuffer(cameraData.byteLength,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST); this.device.queue.writeBuffer(cameras, 0, cameraData);
		// Rendering gets private camera slots so interactive novel views can
		// never mutate calibrated camera bytes used by training or validation.
		const renderCameraData = new Float32Array(cameraData.length + MAX_RENDER_VIEWS * 20);
		renderCameraData.set(cameraData);
		for (let panel = 0; panel < MAX_RENDER_VIEWS; panel += 1) {
			const sourceOffset = Math.min(panel, this.dataset.cameras.length - 1) * 20;
			renderCameraData.set(cameraData.subarray(sourceOffset, sourceOffset + 20), cameraData.length + panel * 20);
		}
		const renderCameras = makeBuffer(renderCameraData.byteLength,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.device.queue.writeBuffer(renderCameras, 0, renderCameraData);
		const trainViewData = new Uint32Array(this.trainViewIndices);
		const trainViews = makeBuffer(trainViewData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.device.queue.writeBuffer(trainViews, 0, trainViewData);
		const cameraSamples = packSamplesByCamera(this.dataset);
		const cameraSampleIndices = makeBuffer(cameraSamples.indices.byteLength,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		const cameraSampleRanges = makeBuffer(cameraSamples.ranges.byteLength,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.device.queue.writeBuffer(cameraSampleIndices, 0, cameraSamples.indices);
		this.device.queue.writeBuffer(cameraSampleRanges, 0, cameraSamples.ranges);
		const packedSampleCount = this.dataset.motionSamples.length + this.dataset.staticSamples.length;
		// The tiled backend owns depth ordering in its tile bins and never
		// dispatches the sampled-ray order kernel. Avoid reserving a second
		// view x time x splat order cache merely because it inherits preview
		// and maintenance pipelines from this class.
		const orderCount = sampledOrderCacheEntries(
			this.trainViewIndices.length,
			this.dataset.frameCount,
			this.splatCount,
			!this.skipSampleGradientAllocation,
		);
		const sampleData = new Uint32Array(Math.max(1, packedSampleCount + orderCount));
		sampleData.set(this.dataset.motionSamples, 0);
		sampleData.set(this.dataset.staticSamples, this.dataset.motionSamples.length);
		const sampleIndices = makeBuffer(sampleData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.device.queue.writeBuffer(sampleIndices, 0, sampleData);
		const quad = makeBuffer(32, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
		this.device.queue.writeBuffer(quad, 0, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]));
		const renderSortCapacity = nextPowerOfTwo(this.splatCount);
		const renderOrder = Array.from({ length: MAX_RENDER_VIEWS }, () => makeBuffer(renderSortCapacity * 4));
		const renderDepths = Array.from({ length: MAX_RENDER_VIEWS }, () => makeBuffer(renderSortCapacity * 4));
		this.buffers = { params: [paramA, paramB], firstMoment, secondMoment, stats, sampleGradients, sampleLosses,
			target, cameras, renderCameras, trainViews, sampleIndices, cameraSampleIndices, cameraSampleRanges,
				quad, renderOrder, renderDepths, trainConfig: makeBuffer(144, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
				renderConfig: Array.from({ length: MAX_RENDER_VIEWS },
					() => makeBuffer(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)),
				metricsReadback: makeBuffer(MAX_SAMPLES_PER_STEP * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
				paramsReadback: makeBuffer(params.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) };
	}

	createBindGroups() {
		const orderEntries = (input) => [
			{ binding: 0, resource: { buffer: this.buffers.trainConfig } },
			{ binding: 1, resource: { buffer: input } },
			{ binding: 2, resource: { buffer: this.buffers.cameras } },
			{ binding: 3, resource: { buffer: this.buffers.trainViews } },
			{ binding: 4, resource: { buffer: this.buffers.sampleIndices } },
		];
		const gradientEntries = (input) => [
			{ binding: 0, resource: { buffer: this.buffers.trainConfig } }, { binding: 1, resource: { buffer: input } },
			{ binding: 2, resource: { buffer: this.buffers.target } },
			{ binding: 3, resource: { buffer: this.buffers.sampleIndices } },
			{ binding: 4, resource: { buffer: this.buffers.cameras } },
			{ binding: 5, resource: { buffer: this.buffers.sampleGradients } },
			{ binding: 6, resource: { buffer: this.buffers.sampleLosses } },
			{ binding: 7, resource: { buffer: this.buffers.trainViews } },
			{ binding: 8, resource: { buffer: this.buffers.cameraSampleIndices } },
			{ binding: 9, resource: { buffer: this.buffers.cameraSampleRanges } },
		];
		const updateEntries = (input, output) => [
			{ binding: 0, resource: { buffer: this.buffers.trainConfig } }, { binding: 1, resource: { buffer: input } },
			{ binding: 2, resource: { buffer: output } }, { binding: 3, resource: { buffer: this.buffers.firstMoment } },
			{ binding: 4, resource: { buffer: this.buffers.secondMoment } },
			{ binding: 5, resource: { buffer: this.buffers.stats } },
			{ binding: 6, resource: { buffer: this.buffers.sampleGradients } },
			{ binding: 7, resource: { buffer: this.buffers.sampleLosses } },
		];
		const maintenanceEntries = (params) => [
			{ binding: 0, resource: { buffer: params } },
			{ binding: 1, resource: { buffer: this.buffers.firstMoment } },
			{ binding: 2, resource: { buffer: this.buffers.secondMoment } },
			{ binding: 3, resource: { buffer: this.buffers.stats } },
		];
		this.bindGroups = {
			order: this.buffers.params.map((params) => this.device.createBindGroup({
				layout: this.pipelines.order.getBindGroupLayout(0), entries: orderEntries(params) })),
			gradients: this.buffers.params.map((params) => this.device.createBindGroup({
				layout: this.pipelines.gradients.getBindGroupLayout(0), entries: gradientEntries(params) })),
			update: [this.device.createBindGroup({ layout: this.pipelines.update.getBindGroupLayout(0),
				entries: updateEntries(this.buffers.params[0], this.buffers.params[1]) }),
			this.device.createBindGroup({ layout: this.pipelines.update.getBindGroupLayout(0),
				entries: updateEntries(this.buffers.params[1], this.buffers.params[0]) })],
			maintenance: this.buffers.params.map((params) => this.device.createBindGroup({
				layout: this.pipelines.maintenance.getBindGroupLayout(0), entries: maintenanceEntries(params) })),
			renderSort: [[], []], render: [[], []],
		};
		for (let paramIndex = 0; paramIndex < this.buffers.params.length; paramIndex += 1) {
			for (let panel = 0; panel < this.buffers.renderConfig.length; panel += 1) {
				const renderConfig = this.buffers.renderConfig[panel];
				this.bindGroups.renderSort[paramIndex].push(this.device.createBindGroup({
					layout: this.pipelines.renderSort.getBindGroupLayout(0), entries: [
						{ binding: 0, resource: { buffer: renderConfig } },
						{ binding: 1, resource: { buffer: this.buffers.params[paramIndex] } },
						{ binding: 2, resource: { buffer: this.buffers.renderCameras } },
						{ binding: 3, resource: { buffer: this.buffers.renderOrder[panel] } },
						{ binding: 4, resource: { buffer: this.buffers.renderDepths[panel] } }] }));
				this.bindGroups.render[paramIndex].push(this.device.createBindGroup({
					layout: this.pipelines.render.getBindGroupLayout(0), entries: [
						{ binding: 0, resource: { buffer: renderConfig } },
						{ binding: 1, resource: { buffer: this.buffers.params[paramIndex] } },
						{ binding: 2, resource: { buffer: this.buffers.renderCameras } },
						{ binding: 3, resource: { buffer: this.buffers.renderOrder[panel] } }] }));
			}
		}
	}

	trainStep({ learningRate = 1, learningRateDecay = false, samplesPerStep = 96,
		modelMode = 0, temporalSigma = 0.30,
		motionSampleRate = 0.90, staticSampleRate = 0.08, motionCoverageTarget = 0.52,
		camerasPerStep = undefined } = {}) {
		samplesPerStep = Math.min(MAX_SAMPLES_PER_STEP, Math.max(1, samplesPerStep));
		const rates = browserLearningRates(learningRate, this.stepCount, learningRateDecay);
		this.lastLearningRateMultipliers = {
			geometry: rates.geometry,
			appearance: rates.appearance,
			progress: rates.progress,
		};
		const cameraBatch = rotatingTrainViewBatch(this.trainViewIndices, this.stepCount, camerasPerStep);
		this.lastCameraBatch = cameraBatch.indices;
		this.lastCameraBatchStart = cameraBatch.start;
		writeTrainConfig(this.configBytes, { width: this.dataset.width, height: this.dataset.height,
			frameCount: this.dataset.frameCount, splatCount: this.splatCount, sampleCount: samplesPerStep,
			step: this.stepCount, modelMode, motionSampleCount: this.dataset.motionSamples.length,
			staticSampleCount: this.dataset.staticSamples.length, lrPosition: rates.position,
			lrColor: rates.color, lrOpacity: rates.opacity, lrMotion: rates.motion,
			minRadius: 0.0015, maxRadius: 0.12, temporalSigma, targetAspect: this.dataset.width / this.dataset.height,
			motionSampleRate, motionCoverageTarget, motionCoverageWeight: 0.05, staticAlphaWeight: 0.08,
			staticSampleRate, trainViewCount: this.trainViewIndices.length, cameraCount: this.dataset.viewCount,
			geometryScale: this.dataset.geometryScale, camerasPerStep: cameraBatch.indices.length,
			cameraRotationStart: cameraBatch.start,
			legacyAllCameraSampling: this.legacyContiguousTrainViews
				&& cameraBatch.indices.length >= this.trainViewIndices.length });
		this.device.queue.writeBuffer(this.buffers.trainConfig, 0, this.configBytes);
		const encoder = this.device.createCommandEncoder();
		const orderPass = encoder.beginComputePass();
		orderPass.setPipeline(this.pipelines.order); orderPass.setBindGroup(0, this.bindGroups.order[this.currentIndex]);
		orderPass.dispatchWorkgroups(cameraBatch.indices.length * this.dataset.frameCount); orderPass.end();
		const gradientPass = encoder.beginComputePass();
		gradientPass.setPipeline(this.pipelines.gradients);
		gradientPass.setBindGroup(0, this.bindGroups.gradients[this.currentIndex]);
		gradientPass.dispatchWorkgroups(samplesPerStep); gradientPass.end();
		const updatePass = encoder.beginComputePass();
		updatePass.setPipeline(this.pipelines.update); updatePass.setBindGroup(0, this.bindGroups.update[this.currentIndex]);
		updatePass.dispatchWorkgroups(Math.ceil(this.splatCount / 256)); updatePass.end();
		const nextStep = this.stepCount + 1;
		if (nextStep >= DENSITY_INTERVAL && nextStep <= DENSITY_STOP_STEP && nextStep % DENSITY_INTERVAL === 0) {
			const maintenancePass = encoder.beginComputePass(); maintenancePass.setPipeline(this.pipelines.maintenance);
			maintenancePass.setBindGroup(0, this.bindGroups.maintenance[1 - this.currentIndex]);
			maintenancePass.dispatchWorkgroups(1); maintenancePass.end();
			this.totalRecycled += DENSITY_SPLITS_PER_PASS;
		}
		this.device.queue.submit([encoder.finish()]);
		this.lastSampleCount = samplesPerStep;
		this.currentIndex = 1 - this.currentIndex; this.stepCount += 1;
	}

	async readLoss({ modelMode = 0, temporalSigma = 0.30 } = {}) {
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.buffers.stats, 0, this.buffers.metricsReadback, 0, 16);
		this.device.queue.submit([encoder.finish()]);
		await this.buffers.metricsReadback.mapAsync(GPUMapMode.READ);
		const values = new Float32Array(this.buffers.metricsReadback.getMappedRange(0, 16).slice(0));
		this.buffers.metricsReadback.unmap();
		return values[3];
	}

	async readParamsUnlocked() {
		const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.buffers.params[this.currentIndex], 0,
			this.buffers.paramsReadback, 0, this.splatCount * SPLAT_BYTES); this.device.queue.submit([encoder.finish()]);
		await this.buffers.paramsReadback.mapAsync(GPUMapMode.READ);
		const params = new Float32Array(this.buffers.paramsReadback.getMappedRange().slice(0)); this.buffers.paramsReadback.unmap();
		return params;
	}

	readParams() {
		const read = this.readbackChain.then(() => this.readParamsUnlocked());
		this.readbackChain = read.then(() => undefined, () => undefined); return read;
	}

	continuationContract() {
		return {
			parameterSchema: CONTINUATION_PARAMETER_SCHEMA,
			splatFloats: SPLAT_FLOATS,
			splatCount: this.splatCount,
			geometryScale: this.dataset.geometryScale,
			frameCount: this.dataset.frameCount,
			cameraCount: this.dataset.cameras.length,
			trainViewIndices: Array.from(this.trainViewIndices),
		};
	}

	continuationBufferSpecs() {
		return [
			{ name: "params", buffer: this.buffers.params[this.currentIndex], Type: Float32Array,
				elementCount: this.splatCount * SPLAT_FLOATS },
			{ name: "firstMoment", buffer: this.buffers.firstMoment, Type: Float32Array,
				elementCount: this.splatCount * SPLAT_FLOATS },
			{ name: "secondMoment", buffer: this.buffers.secondMoment, Type: Float32Array,
				elementCount: this.splatCount * SPLAT_FLOATS },
			{ name: "densityStats", buffer: this.buffers.stats, Type: Float32Array,
				elementCount: this.splatCount * 4 },
		];
	}

	continuationMetadata() {
		return {
			contract: this.continuationContract(),
			initialParams: this.initialParams.slice(),
			stepCount: this.stepCount,
			currentIndex: this.currentIndex,
			totalRecycled: this.totalRecycled,
		};
	}

	continuationStateFromSnapshots(snapshots, metadata) {
		return { schema: CONTINUATION_STATE_SCHEMA, ...metadata,
			params: snapshots.params, firstMoment: snapshots.firstMoment,
			secondMoment: snapshots.secondMoment, densityStats: snapshots.densityStats };
	}

	async exportContinuationStateUnlocked() {
		if (!this.device || !this.buffers) throw new Error("Trainer must be initialized before exporting continuation state.");
		const specs = this.continuationBufferSpecs();
		const metadata = this.continuationMetadata();
		let byteLength = 0;
		for (const spec of specs) {
			spec.byteOffset = byteLength;
			spec.byteLength = spec.elementCount * spec.Type.BYTES_PER_ELEMENT;
			byteLength += spec.byteLength;
		}
		const readback = this.device.createBuffer({
			size: Math.max(4, byteLength),
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});
		let mapped = false;
		try {
			const encoder = this.device.createCommandEncoder();
			for (const spec of specs) {
				encoder.copyBufferToBuffer(spec.buffer, 0, readback, spec.byteOffset, spec.byteLength);
			}
			this.device.queue.submit([encoder.finish()]);
			await readback.mapAsync(GPUMapMode.READ, 0, byteLength);
			mapped = true;
			const bytes = readback.getMappedRange(0, byteLength);
			const snapshots = {};
			for (const spec of specs) {
				const copy = bytes.slice(spec.byteOffset, spec.byteOffset + spec.byteLength);
				snapshots[spec.name] = new spec.Type(copy);
			}
			return this.continuationStateFromSnapshots(snapshots, metadata);
		} finally {
			if (mapped) readback.unmap();
			readback.destroy();
		}
	}

	exportContinuationState() {
		const exportState = this.readbackChain.then(() => this.exportContinuationStateUnlocked());
		this.readbackChain = exportState.then(() => undefined, () => undefined);
		return exportState;
	}

	assertContinuationStateCompatible(state) {
		return assertContinuationStateCompatible(state, this.continuationContract());
	}

	continuationRestoreWrites(state) {
		return [
			{ buffer: this.buffers.params[0], data: state.params },
			{ buffer: this.buffers.params[1], data: state.params },
			{ buffer: this.buffers.firstMoment, data: state.firstMoment },
			{ buffer: this.buffers.secondMoment, data: state.secondMoment },
			{ buffer: this.buffers.stats, data: state.densityStats },
		];
	}

	applyContinuationMetadata(state) {
		this.initialParams = state.initialParams.slice();
		this.stepCount = state.stepCount;
		this.currentIndex = state.currentIndex;
		this.totalRecycled = state.totalRecycled;
	}

	async restoreContinuationStateUnlocked(state) {
		if (!this.device || !this.buffers) throw new Error("Trainer must be initialized before restoring continuation state.");
		this.assertContinuationStateCompatible(state);
		const hasErrorScope = typeof this.device.pushErrorScope === "function"
			&& typeof this.device.popErrorScope === "function";
		if (hasErrorScope) this.device.pushErrorScope("validation");
		let restoreError = null;
		try {
			for (const { buffer, data } of this.continuationRestoreWrites(state)) {
				this.device.queue.writeBuffer(buffer, 0, data);
			}
			await this.device.queue.onSubmittedWorkDone();
		} finally {
			if (hasErrorScope) restoreError = await this.device.popErrorScope();
		}
		if (restoreError) throw new Error(`Continuation-state restore failed: ${restoreError.message}`);
		this.applyContinuationMetadata(state);
		return this;
	}

	restoreContinuationState(state) {
		const restore = this.readbackChain.then(() => this.restoreContinuationStateUnlocked(state));
		this.readbackChain = restore.then(() => undefined, () => undefined);
		return restore;
	}

	async readValidationMetrics({ modelMode = 0, temporalSigma = 0.30, gridSize = 16 } = {}) {
		const params = await this.readParams();
		const splatCount = resolveActiveSplatCount(this.splatCount, this.activeSplatCount);
		const metrics = computeSnapshotValidationMetrics(this.dataset, params, {
			splatCount, modelMode, temporalSigma, gridSize,
		});
		const activeValues = splatCount * SPLAT_FLOATS;
		let delta = 0;
		for (let i = 0; i < activeValues; i += 1) delta += Math.abs(params[i] - this.initialParams[i]);
		return { ...metrics, parameterDelta: delta / activeValues, totalRecycled: this.totalRecycled };
	}

	async readPreviewErrorImage({ time = 0.35, modelMode = 0, temporalSigma = 0.30, viewIndex = null } = {}) {
		const params = await this.readParams(); const view = viewIndex ?? this.dataset.heldoutViewIndex;
		const splatCount = resolveActiveSplatCount(this.splatCount, this.activeSplatCount);
		const frame = Math.round(time * (this.dataset.frameCount - 1)); const pixels = this.dataset.width * this.dataset.height;
		const data = new Uint8ClampedArray(pixels * 4); let meanAbs = 0;
		for (let pixel = 0; pixel < pixels; pixel += 1) {
			const x = pixel % this.dataset.width; const y = Math.floor(pixel / this.dataset.width);
			const result = evalModelCpu(this.dataset, params, splatCount, view, frame,
				(x + 0.5) / this.dataset.width, (y + 0.5) / this.dataset.height, modelMode, temporalSigma);
			const base = ((view * this.dataset.frameCount + frame) * pixels + pixel) * 4;
			const e = Math.sqrt(((result.colorR - readFrameBankValue(this.dataset, base)) ** 2
				+ (result.colorG - readFrameBankValue(this.dataset, base + 1)) ** 2
				+ (result.colorB - readFrameBankValue(this.dataset, base + 2)) ** 2) / 3);
			meanAbs += e; const heat = Math.min(1, e * 3); const out = pixel * 4;
			data[out] = 255 * heat; data[out + 1] = 255 * Math.max(0, heat * 1.6 - 0.3);
			data[out + 2] = 255 * Math.max(0.05, 0.4 - heat * 0.3); data[out + 3] = 255;
		}
		return { frame, viewIndex: view, width: this.dataset.width, height: this.dataset.height, data,
			meanAbs: meanAbs / pixels, maxAbs: 1 };
	}

	maintainDensity() {
		if (!this.buffers || this.stepCount > DENSITY_STOP_STEP) return 0;
		const encoder = this.device.createCommandEncoder(); const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipelines.maintenance); pass.setBindGroup(0, this.bindGroups.maintenance[this.currentIndex]);
		pass.dispatchWorkgroups(1); pass.end(); this.device.queue.submit([encoder.finish()]);
		this.totalRecycled += DENSITY_SPLITS_PER_PASS; return DENSITY_SPLITS_PER_PASS;
	}

	resizeCanvas() {
		if (!this.canvas?.getBoundingClientRect) return;
		const dpr = Math.min(globalThis.devicePixelRatio || 1, 1.25); const rect = this.canvas.getBoundingClientRect();
		const width = Math.max(1, Math.floor(rect.width * dpr)); const height = Math.max(1, Math.floor(rect.height * dpr));
		if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
	}

	writePreviewCameras(cameras) {
		const previewCameras = cameras.slice(0, MAX_RENDER_VIEWS);
		const firstViewIndex = this.dataset.cameras.length;
		const packed = new Float32Array(previewCameras.length * 20);
		for (let panel = 0; panel < previewCameras.length; panel += 1) {
			packed.set(packPreviewCamera(previewCameras[panel], this.dataset.geometryScale ?? 1), panel * 20);
		}
		this.device.queue.writeBuffer(this.buffers.renderCameras,
			firstViewIndex * 20 * Float32Array.BYTES_PER_ELEMENT, packed);
		return previewCameras.map((_, panel) => firstViewIndex + panel);
	}

	render(time = 0.35, modelMode = 0, temporalSigma = 0.30, renderMode = 0, viewIndex = 0,
		viewIndices = null, previewCameras = null) {
		if (!this.buffers || !this.device || !this.context) return;
		this.resizeCanvas();
		const splatCount = resolveActiveSplatCount(this.splatCount, this.activeSplatCount);
		const resolvedViewIndices = Array.isArray(previewCameras) && previewCameras.length > 0
			? this.writePreviewCameras(previewCameras)
			: resolveRenderViewIndices(this.dataset, viewIndices);
		const renderViews = resolvedViewIndices.length;
		const panelWidth = this.canvas.width / renderViews;
		for (let panel = 0; panel < renderViews; panel += 1) {
			writeRenderConfig(this.renderConfigBytes[panel], { width: panelWidth, height: this.canvas.height,
				time, splatCount, modelMode, targetAspect: this.dataset.width / this.dataset.height,
				temporalSigma, targetWidth: this.dataset.width, targetHeight: this.dataset.height, renderMode,
				viewIndex: renderViews > 1 ? resolvedViewIndices[panel] : (resolvedViewIndices[0] ?? viewIndex) });
			this.device.queue.writeBuffer(this.buffers.renderConfig[panel], 0, this.renderConfigBytes[panel]);
		}
		const encoder = this.device.createCommandEncoder();
		for (let panel = 0; panel < renderViews; panel += 1) {
			const sortPass = encoder.beginComputePass(); sortPass.setPipeline(this.pipelines.renderSort);
			sortPass.setBindGroup(0, this.bindGroups.renderSort[this.currentIndex][panel]);
			sortPass.dispatchWorkgroups(1); sortPass.end();
		}
		const pass = encoder.beginRenderPass({ colorAttachments: [{
			view: this.context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 },
			loadOp: "clear", storeOp: "store" }] });
		pass.setPipeline(this.pipelines.render); pass.setVertexBuffer(0, this.buffers.quad);
		for (let panel = 0; panel < renderViews; panel += 1) {
			const left = Math.floor(panel * panelWidth);
			const right = panel === renderViews - 1 ? this.canvas.width : Math.floor((panel + 1) * panelWidth);
			pass.setViewport(left, 0, right - left, this.canvas.height, 0, 1);
			pass.setScissorRect(left, 0, right - left, this.canvas.height);
			pass.setBindGroup(0, this.bindGroups.render[this.currentIndex][panel]);
			pass.draw(4, splatCount);
		}
		pass.end(); this.device.queue.submit([encoder.finish()]);
	}

	dispose() {
		if (this.buffers) for (const buffer of Object.values(this.buffers)) {
			if (Array.isArray(buffer)) {
				for (const item of buffer) item?.destroy?.();
			} else {
				buffer?.destroy?.();
			}
		}
		this.context?.unconfigure(); this.buffers = null;
	}
}
