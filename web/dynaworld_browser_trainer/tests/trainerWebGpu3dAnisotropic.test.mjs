import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
	SPLAT_FLOATS,
	INITIAL_SPLAT_OPACITY,
	anisotropicGaussianVjpCpu,
	evaluateAnisotropicGaussianCpu,
	makeInitialSplats,
	normalizeQuaternionCpu,
	normalizeDatasetGeometry,
	projectAnisotropicGaussianCpu,
	resolveAnchorCameraIndex,
	sortProjectedSplatsBackToFront,
} from "../trainerWebGpu3d.js";

const source = readFileSync(new URL("../trainerWebGpu3d.js", import.meta.url), "utf8");
const camera = {
	worldToCamera: new Float32Array([
		0.99, 0.01, 0.02, 0.03,
		-0.01, 0.98, 0.04, -0.02,
		-0.02, -0.04, 0.99, 0.1,
		0, 0, 0, 1,
	]),
	intrinsics: new Float32Array([0.54, 0.72, 0.5, 0.5]),
};
const baseState = {
	center: [0.1, -0.05, 2],
	logScales: [-2, -2.3, -2.5],
	quaternion: [0.1, 0.2, -0.1, 0.95],
};
const projectionOptions = { camera, aspect: 4 / 3, height: 72 };
const sample = [0.72, 0.49];

function alpha(state) {
	const projection = projectAnisotropicGaussianCpu({ ...state, ...projectionOptions });
	return evaluateAnisotropicGaussianCpu(projection, sample, 0.4, 0.8).alpha;
}

function centralDifference(state, key, index, epsilon = 1e-5) {
	const plus = { ...state, [key]: [...state[key]] };
	const minus = { ...state, [key]: [...state[key]] };
	plus[key][index] += epsilon;
	minus[key][index] -= epsilon;
	return (alpha(plus) - alpha(minus)) / (2 * epsilon);
}

function assertClose(actual, expected, tolerance = 2e-5) {
	const scale = Math.max(1, Math.abs(actual), Math.abs(expected));
	assert.ok(Math.abs(actual - expected) <= tolerance * scale,
		`expected ${actual} to be within ${tolerance} relative of ${expected}`);
}

test("24-float source contract reaches WGSL train, update, render, and hybrid tape", () => {
	assert.equal(SPLAT_FLOATS, 24);
	assert.ok((source.match(/logScalePad: vec4<f32>/g) ?? []).length >= 3);
	assert.ok((source.match(/rotation: vec4<f32>/g) ?? []).length >= 3);
	assert.match(source, /let conic = vec3<f32>\(covariance\.z, -covariance\.y, covariance\.x\) \/ determinant/);
	assert.match(source, /barJ0 = 2\.0 \* \(barC00 \* sigmaJ0 \+ barC01 \* sigmaJ1\)/);
	assert.match(source, /gradRotation = \(normalizedQuatGrad - q \* dot\(q, normalizedQuatGrad\)\)/);
	assert.match(source, /halfLogAspectLimit = log\(2\.0\)/);
	assert.match(source, /if \(qform > 9\.0\) \{ discard; \}/);
	assert.match(source, /cfg\.splatCount <= 768u/);
	assert.match(source, /sampleGradients\[s \* cfg\.splatCount \+ i\] = Splat/);
});

test("analytic anisotropic VJP matches finite differences", () => {
	const projection = projectAnisotropicGaussianCpu({ ...baseState, ...projectionOptions });
	assert.equal(projection.valid, true);
	const analytic = anisotropicGaussianVjpCpu({ projection, sample, opacity: 0.4, timeWeight: 0.8 });
	for (const [key, count] of [["center", 3], ["logScales", 3], ["quaternion", 4]]) {
		for (let index = 0; index < count; index += 1) {
			assertClose(analytic[key][index], centralDifference(baseState, key, index));
		}
	}
});

test("isotropic covariance is rotation invariant and quaternion sign invariant", () => {
	const isotropic = { ...baseState, logScales: [-2.1, -2.1, -2.1] };
	const identity = projectAnisotropicGaussianCpu({ ...isotropic, quaternion: [0, 0, 0, 1], ...projectionOptions });
	const rotated = projectAnisotropicGaussianCpu({ ...isotropic, quaternion: [0.3, -0.2, 0.4, 0.7], ...projectionOptions });
	for (let index = 0; index < 3; index += 1) assertClose(identity.covariance[index], rotated.covariance[index], 1e-6);
	const positive = projectAnisotropicGaussianCpu({ ...baseState, ...projectionOptions });
	const negative = projectAnisotropicGaussianCpu({ ...baseState,
		quaternion: baseState.quaternion.map((value) => -value), ...projectionOptions });
	for (let index = 0; index < 3; index += 1) assertClose(positive.covariance[index], negative.covariance[index], 1e-7);
});

test("one world covariance projects to camera-dependent conics", () => {
	const secondCamera = {
		worldToCamera: new Float32Array([
			0.94, 0, 0.342, -0.1,
			0, 1, 0, 0.02,
			-0.342, 0, 0.94, 0.2,
			0, 0, 0, 1,
		]),
		intrinsics: camera.intrinsics,
	};
	const first = projectAnisotropicGaussianCpu({ ...baseState, ...projectionOptions });
	const second = projectAnisotropicGaussianCpu({ ...baseState, ...projectionOptions, camera: secondCamera });
	assert.equal(first.valid && second.valid, true);
	assert.notDeepEqual(first.covariance.map((value) => value.toFixed(8)),
		second.covariance.map((value) => value.toFixed(8)));
});

test("quaternion normalization has a finite identity fallback", () => {
	assert.deepEqual(normalizeQuaternionCpu([0, 0, 0, 0]), [0, 0, 0, 1]);
	const normalized = normalizeQuaternionCpu([1, 2, 3, 4]);
	assertClose(Math.hypot(...normalized), 1, 1e-12);
});

test("geometry normalization follows the declared seed-coordinate anchor", () => {
	const dataset = {
		datasetContract: { anchor_camera: "anchor" },
		cameras: [
			{ name: "other", worldToCamera: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 90, 0, 0, 0, 1]) },
			{ name: "anchor", worldToCamera: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) },
		],
		seedPointCount: 3,
		seedPoints: new Float32Array([0, 0, 10, 1, 1, 1, 0, 0, 20, 1, 1, 1, 0, 0, 30, 1, 1, 1]),
	};
	assert.equal(resolveAnchorCameraIndex(dataset), 1);
	const normalized = normalizeDatasetGeometry(dataset);
	assertClose(normalized.geometryScale, 0.05);
	assertClose(normalized.seedPoints[2], 0.5);
	assertClose(normalized.cameras[0].worldToCamera[11], 4.5);
});

test("point-cloud initialization starts with bounded local anisotropy", () => {
	const points = [];
	for (let y = -1; y <= 1; y += 1) for (let x = -1; x <= 1; x += 1) {
		points.push(x * 0.12, y * 0.08, 2 + 0.002 * x * y, 0.4, 0.5, 0.6);
	}
	const dataset = {
		datasetContract: { anchor_camera: "anchor" },
		cameras: [{ name: "anchor", worldToCamera: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) }],
		seedPointCount: 9,
		seedPoints: new Float32Array(points),
		geometryScale: 1,
	};
	const params = makeInitialSplats(dataset, 9);
	let anisotropic = 0;
	for (let index = 0; index < 9; index += 1) {
		const base = index * SPLAT_FLOATS;
		const scales = [Math.exp(params[base + 12]), Math.exp(params[base + 13]), Math.exp(params[base + 14])];
		const aspect = Math.max(...scales) / Math.min(...scales);
		if (aspect > 1.2) anisotropic += 1;
		assert.ok(aspect <= 3.00001);
		assertClose(Math.hypot(...params.subarray(base + 16, base + 20)), 1, 1e-6);
		assertClose(1 / (1 + Math.exp(-params[base + 23])), INITIAL_SPLAT_OPACITY, 1e-6);
	}
	assert.ok(anisotropic >= 7);
});

test("camera-specific projected splats are composited back to front with a stable tie break", () => {
	const splats = [
		{ index: 2, projection: { valid: true, cameraPoint: [0, 0, 1] } },
		{ index: 1, projection: { valid: true, cameraPoint: [0, 0, 3] } },
		{ index: 0, projection: { valid: true, cameraPoint: [0, 0, 3] } },
		{ index: 3, projection: { valid: false, cameraPoint: [0, 0, 9] } },
	];
	assert.deepEqual(sortProjectedSplatsBackToFront(splats).map((splat) => splat.index), [0, 1, 2, 3]);
});

test("active WGSL has camera-time order caches, trainable static mix, and GPU split-recycle", () => {
	assert.match(source, /fn build_order/);
	assert.match(source, /pair \* cfg\.splatCount/);
	assert.match(source, /params\[renderOrder\[iid\]\]/);
	assert.match(source, /gradStaticMix = alphaGrad/);
	assert.match(source, /clamp\(p\.centerStatic\.w - cfg\.lrMotion \* posUpdate\.w/);
	assert.match(source, /fn split_recycle/);
	assert.match(source, /halfOpacity = clamp\(1\.0 - sqrt\(1\.0 - opacity\)/);
});
