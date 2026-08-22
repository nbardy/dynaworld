const EPSILON = 1e-8;
const ROTATE_RADIANS_PER_PIXEL = 0.006;
const PAN_FRACTION_PER_PIXEL = 0.0015;
const ZOOM_EXPONENT_PER_PIXEL = 0.0015;

function assertFiniteVector(value, length, label) {
	if (!value || value.length !== length || !Array.from(value).every(Number.isFinite)) {
		throw new TypeError(`${label} must contain ${length} finite values.`);
	}
}

function dot(left, right) {
	return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function cross(left, right) {
	return [
		left[1] * right[2] - left[2] * right[1],
		left[2] * right[0] - left[0] * right[2],
		left[0] * right[1] - left[1] * right[0],
	];
}

function normalize(value, label) {
	const length = Math.hypot(...value);
	if (!Number.isFinite(length) || length < EPSILON) throw new RangeError(`${label} has zero length.`);
	return value.map((component) => component / length);
}

function addScaled(target, axis, scale) {
	return target.map((value, index) => value + axis[index] * scale);
}

function median(values, fallback) {
	if (!values.length) return fallback;
	values.sort((left, right) => left - right);
	return values[Math.floor((values.length - 1) / 2)];
}

export function cameraCenterFromWorldToCamera(worldToCamera) {
	assertFiniteVector(worldToCamera, 16, "worldToCamera");
	const translation = [worldToCamera[3], worldToCamera[7], worldToCamera[11]];
	return [
		-dot([worldToCamera[0], worldToCamera[4], worldToCamera[8]], translation),
		-dot([worldToCamera[1], worldToCamera[5], worldToCamera[9]], translation),
		-dot([worldToCamera[2], worldToCamera[6], worldToCamera[10]], translation),
	];
}

export function cameraRigRadius(cameras) {
	if (!Array.isArray(cameras) || cameras.length === 0) {
		throw new TypeError("cameraRigRadius needs at least one calibrated camera.");
	}
	const centers = cameras.map((camera) =>
		cameraCenterFromWorldToCamera(camera?.worldToCamera));
	const mean = [0, 1, 2].map((axis) =>
		centers.reduce((sum, center) => sum + center[axis], 0) / centers.length);
	const radius = 1.1 * Math.max(...centers.map((center) => Math.hypot(
		center[0] - mean[0], center[1] - mean[1], center[2] - mean[2],
	)));
	// Pixel-GS's camera-radius definition collapses for a single camera.
	return radius > 1e-6 ? radius : 1;
}

export function lookAtOpenCv(eye, target, downHint = [0, 1, 0]) {
	assertFiniteVector(eye, 3, "eye");
	assertFiniteVector(target, 3, "target");
	assertFiniteVector(downHint, 3, "downHint");
	const forward = normalize(target.map((value, index) => value - eye[index]), "look direction");
	let right = cross(normalize(downHint, "downHint"), forward);
	if (Math.hypot(...right) < EPSILON) {
		right = cross(Math.abs(forward[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], forward);
	}
	right = normalize(right, "camera right axis");
	const down = normalize(cross(forward, right), "camera down axis");
	return [
		...right, -dot(right, eye),
		...down, -dot(down, eye),
		...forward, -dot(forward, eye),
		0, 0, 0, 1,
	];
}

export function createOrbitCameraState(dataset, viewIndex = 0) {
	const camera = dataset?.cameras?.[viewIndex];
	if (!camera) throw new RangeError(`No calibrated camera exists at view ${viewIndex}.`);
	assertFiniteVector(camera.worldToCamera, 16, "camera.worldToCamera");
	assertFiniteVector(camera.intrinsics, 4, "camera.intrinsics");
	const matrix = Array.from(camera.worldToCamera);
	const eye = cameraCenterFromWorldToCamera(matrix);
	const baseForward = normalize(matrix.slice(8, 11), "camera forward axis");
	const baseRight = normalize(matrix.slice(0, 3), "camera right axis");
	const baseDown = normalize(matrix.slice(4, 7), "camera down axis");
	const depths = [];
	for (let index = 0; index < (dataset.seedPointCount ?? 0); index += 1) {
		const base = index * 6;
		const depth = matrix[8] * dataset.seedPoints[base]
			+ matrix[9] * dataset.seedPoints[base + 1]
			+ matrix[10] * dataset.seedPoints[base + 2] + matrix[11];
		if (Number.isFinite(depth) && depth > EPSILON) depths.push(depth);
	}
	const distance = Math.max(EPSILON, median(depths, 1));
	return {
		target: addScaled(eye, baseForward, distance),
		baseRight,
		baseDown,
		baseForward,
		yaw: 0,
		pitch: 0,
		distance,
		minDistance: Math.max(EPSILON, distance * 0.04),
		maxDistance: distance * 24,
		intrinsics: Array.from(camera.intrinsics),
	};
}

function orbitEye(state) {
	const cosPitch = Math.cos(state.pitch);
	let eye = [...state.target];
	eye = addScaled(eye, state.baseRight, state.distance * Math.sin(state.yaw) * cosPitch);
	eye = addScaled(eye, state.baseDown, state.distance * Math.sin(state.pitch));
	eye = addScaled(eye, state.baseForward, -state.distance * Math.cos(state.yaw) * cosPitch);
	return eye;
}

export function orbitPreviewCamera(state) {
	return {
		worldToCamera: lookAtOpenCv(orbitEye(state), state.target, state.baseDown),
		intrinsics: [...state.intrinsics],
	};
}

export function rotateOrbitCamera(state, deltaX, deltaY) {
	const pitchLimit = Math.PI / 2 - 0.03;
	return {
		...state,
		yaw: state.yaw - deltaX * ROTATE_RADIANS_PER_PIXEL,
		pitch: Math.max(-pitchLimit, Math.min(pitchLimit,
			state.pitch + deltaY * ROTATE_RADIANS_PER_PIXEL)),
	};
}

export function panOrbitCamera(state, deltaX, deltaY) {
	const camera = orbitPreviewCamera(state).worldToCamera;
	let target = [...state.target];
	const scale = state.distance * PAN_FRACTION_PER_PIXEL;
	target = addScaled(target, camera.slice(0, 3), -deltaX * scale);
	target = addScaled(target, camera.slice(4, 7), -deltaY * scale);
	return { ...state, target };
}

export function zoomOrbitCamera(state, deltaY) {
	const distance = state.distance * Math.exp(deltaY * ZOOM_EXPONENT_PER_PIXEL);
	return { ...state, distance: Math.max(state.minDistance, Math.min(state.maxDistance, distance)) };
}

export function dollyOrbitCamera(state, distanceRatio) {
	if (!(distanceRatio > 0) || !Number.isFinite(distanceRatio)) {
		throw new RangeError("distanceRatio must be finite and positive.");
	}
	return {
		...state,
		distance: Math.max(state.minDistance, Math.min(state.maxDistance,
			state.distance * distanceRatio)),
	};
}

export function translateOrbitCamera(state, { rightFraction = 0, downFraction = 0 } = {}) {
	if (![rightFraction, downFraction].every(Number.isFinite)) {
		throw new TypeError("Camera translation fractions must be finite.");
	}
	const camera = orbitPreviewCamera(state).worldToCamera;
	let target = addScaled(state.target, camera.slice(0, 3), state.distance * rightFraction);
	target = addScaled(target, camera.slice(4, 7), state.distance * downFraction);
	return { ...state, target };
}
