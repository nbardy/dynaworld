export const RESOLUTION_MODE_PROGRESSIVE = "progressive-96-384";
export const PROGRESSIVE_RESOLUTION_SWITCH_STEP = 8192;

export function initialResolutionPreset(mode) {
	return mode === RESOLUTION_MODE_PROGRESSIVE ? "96x72" : mode;
}

export function resolutionStageForStep(mode, step) {
	if (!Number.isSafeInteger(step) || step < 0) {
		throw new RangeError("Resolution-stage step must be a non-negative safe integer.");
	}
	if (mode !== RESOLUTION_MODE_PROGRESSIVE) {
		return { preset: mode, progressive: false, transitionStep: null };
	}
	return {
		preset: step < PROGRESSIVE_RESOLUTION_SWITCH_STEP ? "96x72" : "384x288",
		progressive: true,
		transitionStep: PROGRESSIVE_RESOLUTION_SWITCH_STEP,
	};
}

function sameNumbers(left, right, tolerance = 1e-6) {
	if (!left || !right || left.length !== right.length) return false;
	for (let index = 0; index < left.length; index += 1) {
		if (!Number.isFinite(left[index]) || !Number.isFinite(right[index])
			|| Math.abs(left[index] - right[index]) > tolerance) return false;
	}
	return true;
}

export function assertResolutionContinuationCompatible(source, target) {
	if (!source || !target) throw new TypeError("Both resolution datasets are required.");
	if (source.width * 4 !== target.width || source.height * 4 !== target.height) {
		throw new Error("Progressive resolution datasets must differ by exactly 4x in each image dimension.");
	}
	for (const key of ["frameCount", "viewCount", "trainViewCount", "heldoutViewIndex", "seedPointCount"]) {
		if (source[key] !== target[key]) {
			throw new Error(`Progressive resolution dataset mismatch: ${key}.`);
		}
	}
	if (!sameNumbers(source.trainViewIndices, target.trainViewIndices, 0)
		|| !sameNumbers(source.frameIndices, target.frameIndices, 0)
		|| !sameNumbers(source.seedPoints, target.seedPoints, 0)) {
		throw new Error("Progressive resolution datasets do not share splits, frames, and seed geometry.");
	}
	const sourceContract = source.datasetContract ?? {};
	const targetContract = target.datasetContract ?? {};
	for (const key of ["pose_source", "anchor_camera", "coordinate_convention"]) {
		if (sourceContract[key] !== targetContract[key]) {
			throw new Error(`Progressive resolution calibration mismatch: ${key}.`);
		}
	}
	if (source.cameras?.length !== target.cameras?.length) {
		throw new Error("Progressive resolution camera counts do not match.");
	}
	for (let index = 0; index < source.cameras.length; index += 1) {
		const left = source.cameras[index];
		const right = target.cameras[index];
		if (left.name !== right.name || left.role !== right.role
			|| !sameNumbers(left.intrinsics, right.intrinsics)
			|| !sameNumbers(left.worldToCamera, right.worldToCamera)) {
			throw new Error(`Progressive resolution camera mismatch at index ${index}.`);
		}
	}
	return true;
}
