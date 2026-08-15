import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
	computeCameraStressMetrics,
	maskedRgbMetrics,
	opticalCameraVariants,
	physicalCameraVariants,
	warpTargetForIntrinsics,
} from "../cameraStressMetrics.js";
import { cameraCenterFromWorldToCamera } from "../orbitCamera.js";
import { renderSnapshotFrame } from "../snapshotMetrics.js";
import { SPLAT_FLOATS } from "../trainerWebGpu3d.js";

const identity = [
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
];

function camera(name, role, translationX = 0) {
	const worldToCamera = identity.slice();
	worldToCamera[3] = translationX;
	return { name, role, intrinsics: [0.72, 0.72, 0.5, 0.5], worldToCamera };
}

function datasetFixture({ width = 20, height = 16 } = {}) {
	const cameras = [camera("cam00", "train"), camera("cam01", "heldout", -0.18)];
	return {
		width,
		height,
		frameCount: 1,
		viewCount: cameras.length,
		trainViewCount: 1,
		heldoutViewIndex: 1,
		cameras,
		frames: new Float32Array(width * height * cameras.length * 4),
		seedPointCount: 3,
		seedPoints: new Float32Array([
			-0.2, 0, 1.5, 1, 0, 0,
			0, 0, 2.0, 0, 1, 0,
			0.2, 0, 2.5, 0, 0, 1,
		]),
	};
}

function paramsFixture() {
	const params = new Float32Array(2 * SPLAT_FLOATS);
	const set = (index, { center, scales, color, opacity }) => {
		const base = index * SPLAT_FLOATS;
		params.set(center, base);
		params[base + 3] = 1;
		params[base + 7] = 0.5;
		params.set(scales.map(Math.log), base + 12);
		params.set([0, 0, 0, 1], base + 16);
		params.set(color, base + 20);
		params[base + 23] = Math.log(opacity / (1 - opacity));
	};
	set(0, { center: [0, 0, 0.35], scales: [0.20, 0.18, 0.14],
		color: [0.9, 0.2, 0.1], opacity: 0.45 });
	set(1, { center: [0.04, 0, 1.8], scales: [0.25, 0.18, 0.16],
		color: [0.1, 0.4, 0.9], opacity: 0.55 });
	return params;
}

function setTarget(dataset, viewIndex, rgb) {
	const pixels = dataset.width * dataset.height;
	const offset = viewIndex * pixels * 4;
	for (let pixel = 0; pixel < pixels; pixel += 1) {
		for (let channel = 0; channel < 3; channel += 1) {
			dataset.frames[offset + pixel * 4 + channel] = rgb[pixel * 3 + channel];
		}
		dataset.frames[offset + pixel * 4 + 3] = 1;
	}
}

test("optical perturbations preserve pose while physical perturbations move the camera", () => {
	const dataset = datasetFixture();
	const base = dataset.cameras[0];
	const optical = opticalCameraVariants(base);
	assert.equal(optical.length, 6);
	for (const variant of optical) {
		assert.deepEqual(variant.camera.worldToCamera, base.worldToCamera);
	}
	assert.equal(optical[0].camera.intrinsics[0], base.intrinsics[0] * 1.05);
	assert.equal(optical[2].camera.intrinsics[2], base.intrinsics[2] - 0.015);

	const physical = physicalCameraVariants(dataset, 0);
	assert.equal(physical.length, 7);
	physical[0].camera.worldToCamera.forEach((value, index) =>
		assert.ok(Math.abs(value - base.worldToCamera[index]) < 1e-12));
	assert.notDeepEqual(
		cameraCenterFromWorldToCamera(physical[1].camera.worldToCamera),
		cameraCenterFromWorldToCamera(base.worldToCamera),
	);
});

test("identity target warp is exact and shifted warp masks unavailable pixels", () => {
	const width = 8;
	const height = 6;
	const rgb = Float32Array.from({ length: width * height * 3 },
		(_, index) => ((index * 13) % 101) / 100);
	const intrinsics = [0.7, 0.7, 0.5, 0.5];
	const identityWarp = warpTargetForIntrinsics(rgb, width, height, {
		baseIntrinsics: intrinsics,
		testIntrinsics: intrinsics,
		width,
		height,
	});
	assert.deepEqual(identityWarp.rgb, rgb);
	assert.equal(identityWarp.validFraction, 1);
	assert.equal(maskedRgbMetrics(rgb, identityWarp.rgb, identityWarp.mask).mse, 0);

	const shifted = warpTargetForIntrinsics(rgb, width, height, {
		baseIntrinsics: intrinsics,
		testIntrinsics: [0.7, 0.7, 0.65, 0.5],
		width,
		height,
	});
	assert.ok(shifted.validFraction > 0.5 && shifted.validFraction < 1);
});

test("arbitrary-camera snapshots preserve calibrated parity and expose floater telemetry", () => {
	const dataset = datasetFixture();
	const params = paramsFixture();
	const calibrated = renderSnapshotFrame(dataset, params, {
		viewIndex: 0,
		collectGeometryDiagnostics: true,
		nearDepthThreshold: 0.5,
		largeFootprintFraction: 0.25,
	});
	const explicit = renderSnapshotFrame(dataset, params, {
		viewIndex: 0,
		camera: dataset.cameras[0],
		collectGeometryDiagnostics: true,
		nearDepthThreshold: 0.5,
		largeFootprintFraction: 0.25,
	});
	assert.deepEqual(explicit.rgb, calibrated.rgb);
	assert.deepEqual(explicit.coverage, calibrated.coverage);
	assert.ok(explicit.nearCoverage.some((value) => value > 0));
	assert.ok(explicit.largeFootprintCoverage.some((value) => value > 0));
	assert.ok(explicit.depthStd.some((value) => value > 0));

	const shiftedCamera = opticalCameraVariants(dataset.cameras[0])[2].camera;
	const shifted = renderSnapshotFrame(dataset, params, { viewIndex: 0, camera: shiftedCamera });
	assert.notDeepEqual(shifted.rgb, calibrated.rgb);
});

test("camera stress reports real optical targets separately from physical-pose risk", () => {
	const dataset = datasetFixture();
	const params = paramsFixture();
	for (let view = 0; view < dataset.cameras.length; view += 1) {
		setTarget(dataset, view, renderSnapshotFrame(dataset, params, { viewIndex: view }).rgb);
	}
	const stress = computeCameraStressMetrics(dataset, params, {
		viewIndices: [0, 1],
		maxHeight: 12,
		nearDepthGamma: 4,
	});
	assert.equal(stress.contract.opticalTarget, "real_frame_crop_resample");
	assert.equal(stress.contract.physicalPoseTarget, null);
	assert.ok(Number.isFinite(stress.train.opticalWorstPsnr));
	assert.ok(Number.isFinite(stress.heldout.opticalWorstPsnr));
	assert.ok(stress.train.poseNearContribution > 0);
	assert.ok(stress.train.poseLargeFootprintContribution > 0);
	assert.ok(stress.train.poseNormalizedDepthSpread > 0);
	assert.equal(stress.perView.length, 2);
	assert.equal(stress.perView[0].pose.variants.length, 7);
	assert.equal(stress.perView[0].optical.variants.length, 6);
});

test("camera stress is connected to asynchronous validation and the visible diagnostic grid", () => {
	const validationWorker = readFileSync(
		new URL("../validationWorker.js", import.meta.url), "utf8");
	const app = readFileSync(new URL("../app.js", import.meta.url), "utf8");
	const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
	assert.match(validationWorker, /computeCameraStressMetrics/);
	assert.match(validationWorker, /cameraStress: cameraStress\.contract/);
	assert.match(app, /metrics\.cameraStress\?\.train/);
	for (const id of [
		"cameraOpticalPsnrValue",
		"cameraNearAlphaValue",
		"cameraGiantAlphaValue",
		"cameraDepthSpreadValue",
	]) assert.match(html, new RegExp(`id="${id}"`));
});
