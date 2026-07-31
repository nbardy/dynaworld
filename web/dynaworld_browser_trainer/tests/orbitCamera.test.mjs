import assert from "node:assert/strict";
import test from "node:test";

import {
	cameraCenterFromWorldToCamera,
	createOrbitCameraState,
	lookAtOpenCv,
	orbitPreviewCamera,
	panOrbitCamera,
	rotateOrbitCamera,
	zoomOrbitCamera,
} from "../orbitCamera.js";
import { packPreviewCamera } from "../trainerWebGpu3d.js";

function close(actual, expected, tolerance = 1e-6) {
	assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

function transform(matrix, point) {
	return [0, 1, 2].map((row) => matrix[row * 4] * point[0]
		+ matrix[row * 4 + 1] * point[1]
		+ matrix[row * 4 + 2] * point[2] + matrix[row * 4 + 3]);
}

function determinant(matrix) {
	return matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
		- matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
		+ matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8]);
}

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const fixture = {
	cameras: [{ worldToCamera: identity, intrinsics: [0.6, 0.8, 0.5, 0.5] }],
	seedPointCount: 3,
	seedPoints: new Float32Array([
		-1, 0, 2, 1, 0, 0,
		0, 0, 4, 0, 1, 0,
		1, 0, 6, 0, 0, 1,
	]),
};

test("orbit initialization exactly preserves the selected OpenCV camera", () => {
	const state = createOrbitCameraState(fixture, 0);
	assert.deepEqual(state.target, [0, 0, 4]);
	assert.equal(state.distance, 4);
	const preview = orbitPreviewCamera(state);
	preview.worldToCamera.forEach((value, index) => close(value, identity[index]));
	assert.deepEqual(cameraCenterFromWorldToCamera(preview.worldToCamera), [0, 0, 0]);
	const projectedTarget = transform(preview.worldToCamera, state.target);
	close(projectedTarget[0] / projectedTarget[2] * preview.intrinsics[0] + preview.intrinsics[2], 0.5);
	close(projectedTarget[1] / projectedTarget[2] * preview.intrinsics[1] + preview.intrinsics[3], 0.5);
});

test("orbit, pan, and zoom keep a finite proper camera with bounded distance", () => {
	const initial = createOrbitCameraState(fixture, 0);
	const rotated = rotateOrbitCamera(initial, 73, -41);
	const rotatedCamera = orbitPreviewCamera(rotated);
	close(determinant(rotatedCamera.worldToCamera), 1, 1e-5);
	rotatedCamera.worldToCamera.forEach((value) => assert.ok(Number.isFinite(value)));
	const eye = cameraCenterFromWorldToCamera(rotatedCamera.worldToCamera);
	close(Math.hypot(...eye.map((value, index) => value - rotated.target[index])), rotated.distance, 1e-5);
	assert.notDeepEqual(panOrbitCamera(rotated, 10, -8).target, rotated.target);
	assert.equal(zoomOrbitCamera(initial, -1e9).distance, initial.minDistance);
	assert.equal(zoomOrbitCamera(initial, 1e9).distance, initial.maxDistance);
});

test("preview camera packing scales translation without changing projection", () => {
	const worldToCamera = lookAtOpenCv([2, -1, -3], [0, 0, 4], [0, 1, 0]);
	const camera = { worldToCamera, intrinsics: [0.6, 0.8, 0.5, 0.5] };
	const scale = 0.071;
	const packed = packPreviewCamera(camera, scale);
	const point = [0.4, -0.2, 5.5];
	const raw = transform(worldToCamera, point);
	const normalized = transform(packed, point.map((value) => value * scale));
	close(raw[0] / raw[2], normalized[0] / normalized[2]);
	close(raw[1] / raw[2], normalized[1] / normalized[2]);
	assert.deepEqual(worldToCamera, camera.worldToCamera);
});
