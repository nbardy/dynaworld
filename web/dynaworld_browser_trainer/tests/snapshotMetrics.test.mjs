import assert from "node:assert/strict";
import test from "node:test";
import {
	SNAPSHOT_PARAMETER_FAMILIES,
	computeFullImageMetrics,
	computeSnapshotMetrics,
	renderSnapshotFrame,
	resolveSnapshotSelections,
	snapshotUpdateRatios,
	summarizeSplatParameters,
} from "../snapshotMetrics.js";
import { SPLAT_FLOATS, projectAnisotropicGaussianCpu } from "../trainerWebGpu3d.js";

const identityCamera = {
	name: "cam00",
	role: "train",
	intrinsics: new Float32Array([0.72, 0.72, 0.5, 0.5]),
	worldToCamera: new Float32Array([
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	]),
};

function makeDataset({ width = 16, height = 16, frameCount = 1, cameras = [identityCamera] } = {}) {
	return {
		width,
		height,
		frameCount,
		viewCount: cameras.length,
		trainViewCount: cameras.filter(({ role }) => role === "train").length,
		heldoutViewIndex: cameras.findIndex(({ role }) => role === "heldout"),
		cameras,
		frames: new Float32Array(width * height * frameCount * cameras.length * 4),
	};
}

function makeParams(splats) {
	const params = new Float32Array(splats.length * SPLAT_FLOATS);
	for (let index = 0; index < splats.length; index += 1) {
		const base = index * SPLAT_FLOATS;
		const splat = splats[index];
		params.set(splat.center, base);
		params[base + 3] = splat.staticMix ?? 1;
		params.set(splat.velocity ?? [0, 0, 0], base + 4);
		params[base + 7] = splat.timeCenter ?? 0.5;
		params.set(splat.harmonic ?? [0, 0, 0], base + 8);
		params.set((splat.scales ?? [0.18, 0.15, 0.12]).map(Math.log), base + 12);
		params.set(splat.rotation ?? [0, 0, 0, 1], base + 16);
		params.set(splat.color, base + 20);
		const opacity = splat.opacity ?? 0.75;
		params[base + 23] = Math.log(opacity / (1 - opacity));
	}
	return params;
}

function setTarget(dataset, viewIndex, frameIndex, rgb) {
	const pixels = dataset.width * dataset.height;
	const offset = (viewIndex * dataset.frameCount + frameIndex) * pixels * 4;
	for (let pixel = 0; pixel < pixels; pixel += 1) {
		dataset.frames[offset + pixel * 4] = rgb[pixel * 3];
		dataset.frames[offset + pixel * 4 + 1] = rgb[pixel * 3 + 1];
		dataset.frames[offset + pixel * 4 + 2] = rgb[pixel * 3 + 2];
		dataset.frames[offset + pixel * 4 + 3] = 1;
	}
}

function bruteForceRender(dataset, params, {
	viewIndex = 0,
	frameIndex = 0,
	modelMode = 0,
	temporalSigma = 0.30,
	alphaThreshold = 1 / 255,
	transmittanceThreshold = 1e-4,
} = {}) {
	const time = dataset.frameCount <= 1 ? 0 : frameIndex / (dataset.frameCount - 1);
	const sigma = Math.min(0.36, Math.max(0.12, temporalSigma));
	const floor = Math.min(0.12, Math.max(0.035, sigma * 0.30));
	const projected = Array.from({ length: params.length / SPLAT_FLOATS }, (_, index) => {
		const base = index * SPLAT_FLOATS;
		const centeredTime = time * 2 - 1;
		const wave = modelMode === 0 ? Math.sin(time * Math.PI * 2) : 0;
		const center = [
			params[base] + params[base + 4] * centeredTime + params[base + 8] * wave,
			params[base + 1] + params[base + 5] * centeredTime + params[base + 9] * wave,
			params[base + 2] + params[base + 6] * centeredTime + params[base + 10] * wave,
		];
		const projection = projectAnisotropicGaussianCpu({
			center,
			logScales: [...params.subarray(base + 12, base + 15)],
			quaternion: [...params.subarray(base + 16, base + 20)],
			camera: dataset.cameras[viewIndex],
			aspect: dataset.width / dataset.height,
			height: dataset.height,
		});
		const opacity = 1 / (1 + Math.exp(-params[base + 23]));
		const delta = time - Math.min(1, Math.max(0, params[base + 7]));
		const dynamicGate = floor + (1 - floor) * Math.exp(-0.5 * delta * delta / (sigma * sigma));
		const staticMix = Math.min(1, Math.max(0, params[base + 3]));
		return {
			index,
			projection,
			peakAlpha: opacity * (dynamicGate * (1 - staticMix) + staticMix),
			color: [...params.subarray(base + 20, base + 23)],
		};
	}).filter(({ projection }) => projection.valid)
		.sort((left, right) => left.projection.cameraPoint[2] - right.projection.cameraPoint[2]
			|| left.index - right.index);
	const rgb = new Float32Array(dataset.width * dataset.height * 3);
	const coverage = new Float32Array(dataset.width * dataset.height);
	for (let y = 0; y < dataset.height; y += 1) for (let x = 0; x < dataset.width; x += 1) {
		const pixel = y * dataset.width + x;
		const point = [(x + 0.5) / dataset.height, (y + 0.5) / dataset.height];
		let transmittance = 1;
		for (const splat of projected) {
			const dx = point[0] - splat.projection.center[0];
			const dy = point[1] - splat.projection.center[1];
			const [a, b, c] = splat.projection.conic;
			const qform = a * dx * dx + 2 * b * dx * dy + c * dy * dy;
			if (!Number.isFinite(qform) || qform < 0 || qform > 9) continue;
			const rawAlpha = splat.peakAlpha * Math.exp(-0.5 * qform);
			const alpha = rawAlpha >= alphaThreshold ? Math.min(0.99, rawAlpha) : 0;
			for (let channel = 0; channel < 3; channel += 1) {
				rgb[pixel * 3 + channel] += transmittance * alpha * splat.color[channel];
			}
			transmittance *= 1 - alpha;
			if (transmittance < transmittanceThreshold) break;
		}
		coverage[pixel] = 1 - transmittance;
	}
	return { rgb, coverage };
}

function assertArrayClose(actual, expected, tolerance = 2e-7) {
	assert.equal(actual.length, expected.length);
	for (let index = 0; index < actual.length; index += 1) {
		assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
			`index ${index}: expected ${actual[index]} to be within ${tolerance} of ${expected[index]}`);
	}
}

test("identical full images have exact error metrics and unit Gaussian SSIM", () => {
	const image = Float32Array.from({ length: 13 * 12 * 3 },
		(_, index) => ((index * 17) % 101) / 100);
	const metrics = computeFullImageMetrics(image, image, 13, 12);
	assert.equal(metrics.mse, 0);
	assert.equal(metrics.mae, 0);
	assert.equal(metrics.psnr, Number.POSITIVE_INFINITY);
	assert.ok(Math.abs(metrics.ssim - 1) < 1e-12);
});

test("changed pixels degrade MSE, MAE, PSNR, and Gaussian SSIM", () => {
	const target = Float32Array.from({ length: 12 * 12 * 3 },
		(_, index) => ((index * 11) % 97) / 96);
	const prediction = Float32Array.from(target);
	for (let y = 3; y < 9; y += 1) for (let x = 4; x < 8; x += 1) {
		prediction[(y * 12 + x) * 3 + 1] = 1 - prediction[(y * 12 + x) * 3 + 1];
	}
	const metrics = computeFullImageMetrics(prediction, target, 12, 12);
	assert.ok(metrics.mse > 0);
	assert.ok(metrics.mae > 0);
	assert.ok(Number.isFinite(metrics.psnr));
	assert.ok(metrics.ssim < 1);
});

test("tile-binned snapshot rendering agrees with a tiny all-splats-per-pixel reference", () => {
	const dataset = makeDataset({ width: 9, height: 7, frameCount: 2 });
	const params = makeParams([
		{ center: [-0.06, 0.01, 1.8], velocity: [0.03, 0, 0], staticMix: 0.4,
			timeCenter: 0.2, scales: [0.24, 0.10, 0.14], color: [0.9, 0.1, 0.2], opacity: 0.82 },
		{ center: [0.08, -0.03, 2.6], harmonic: [0.02, 0.01, 0], staticMix: 0.7,
			scales: [0.13, 0.22, 0.12], color: [0.1, 0.7, 0.9], opacity: 0.67 },
		{ center: [2.5, 0, 2.1], scales: [0.08, 0.08, 0.08], color: [1, 1, 0], opacity: 0.9 },
	]);
	const tiled = renderSnapshotFrame(dataset, params, {
		frameIndex: 1, tileSize: 3, temporalSigma: 0.24,
	});
	const bruteForce = bruteForceRender(dataset, params, {
		frameIndex: 1, temporalSigma: 0.24,
	});
	assertArrayClose(tiled.rgb, bruteForce.rgb);
	assertArrayClose(tiled.coverage, bruteForce.coverage);
	assert.ok(tiled.primitiveEvaluations < dataset.width * dataset.height * (params.length / SPLAT_FLOATS));
});

test("train and heldout selections aggregate only their requested view/frame pixels", () => {
	const heldoutCamera = { ...identityCamera, name: "cam01", role: "heldout" };
	const dataset = makeDataset({
		width: 12, height: 12, frameCount: 2, cameras: [identityCamera, heldoutCamera],
	});
	const params = makeParams([
		{ center: [0, 0, 2], scales: [0.22, 0.18, 0.14], color: [0.8, 0.25, 0.1], opacity: 0.8 },
	]);
	for (let view = 0; view < 2; view += 1) for (let frame = 0; frame < 2; frame += 1) {
		const rendered = renderSnapshotFrame(dataset, params, { viewIndex: view, frameIndex: frame });
		setTarget(dataset, view, frame, rendered.rgb);
	}
	const heldoutOffset = (2 * 12 * 12) * 4;
	for (let pixel = 0; pixel < 12 * 12; pixel += 1) dataset.frames[heldoutOffset + pixel * 4] += 0.2;
	assert.deepEqual(resolveSnapshotSelections(dataset, { views: "train", frames: "all" }), [
		{ viewIndex: 0, frameIndex: 0 },
		{ viewIndex: 0, frameIndex: 1 },
	]);
	assert.deepEqual(resolveSnapshotSelections(dataset, { views: "heldout", frames: [0] }), [
		{ viewIndex: 1, frameIndex: 0 },
	]);
	const train = computeSnapshotMetrics(dataset, params, { views: "train", frames: "all" });
	const heldout = computeSnapshotMetrics(dataset, params, { views: "heldout", frames: [0] });
	assert.equal(train.selectionCount, 2);
	assert.equal(train.mse, 0);
	assert.equal(train.ssim, 1);
	assert.ok(heldout.mse > 0);
	assert.ok(heldout.ssim < 1);
});

test("snapshot update ratios report independent 24-float parameter families", () => {
	const before = new Float32Array(SPLAT_FLOATS * 2);
	const after = new Float32Array(SPLAT_FLOATS * 2);
	for (let splat = 0; splat < 2; splat += 1) {
		const base = splat * SPLAT_FLOATS;
		for (const components of Object.values(SNAPSHOT_PARAMETER_FAMILIES)) {
			for (const component of components) {
				before[base + component] = 2;
				after[base + component] = 2;
			}
		}
	}
	for (const component of SNAPSHOT_PARAMETER_FAMILIES.center) {
		after[component] = 3;
		after[SPLAT_FLOATS + component] = 3;
	}
	after[20] = 2.2;
	after[SPLAT_FLOATS + 20] = 2.2;
	const ratios = snapshotUpdateRatios(before, after);
	assert.ok(Math.abs(ratios.center.ratio - 0.5) < 1e-12);
	assert.ok(Math.abs(ratios.color.ratio - Math.sqrt((0.2 ** 2) / 3) / 2) < 1e-7);
	assert.equal(ratios.opacity.ratio, 0);
	assert.equal(ratios.rotation.ratio, 0);
	assert.deepEqual(ratios.logScale.components, [12, 13, 14]);
});

test("parameter summary exposes temporal persistence, dead slots, and aspect saturation", () => {
	const params = makeParams([
		{ center: [0, 0, 2], staticMix: 0.95, scales: [0.3, 0.1, 0.1],
			color: [0.8, 0.2, 0.1], opacity: 0.8 },
		{ center: [0, 0, 2], staticMix: 0.2, timeCenter: 0.5, scales: [0.1, 0.1, 0.1],
			velocity: [0.3, 0, 0], harmonic: [0, 0.2, 0], color: [0.2, 0.8, 0.1], opacity: 0.6 },
		{ center: [0, 0, 2], staticMix: 0.92, scales: [0.2, 0.1, 0.1],
			color: [0.1, 0.2, 0.8], opacity: 1e-5 },
	]);
	const summary = summarizeSplatParameters(params, {
		frameCount: 3,
		temporalSigma: 0.3,
		maxAspectRatio: 3,
	});
	assert.equal(summary.activeSplats, 2);
	assert.equal(summary.rasterDeadSplats, 1);
	assert.equal(summary.dynamicSplats, 1);
	assert.equal(summary.persistentSplats, 1);
	assert.equal(summary.temporalAnalyzedSplats, 2);
	assert.ok(Math.abs(summary.staticMixP50 - 0.575) < 1e-7);
	assert.ok(Math.abs(summary.opacityP50 - 0.6) < 1e-6);
	assert.ok(Math.abs(summary.aspectP90 - 2.8) < 1e-6);
	assert.equal(summary.aspectCapFraction, 0.5);
	assert.ok(summary.meanEdgeTemporalSupport > 0.5);
	assert.ok(summary.meanEdgeTemporalSupport < 1);
	assert.ok(Math.abs(summary.velocityP90 - 0.27) < 1e-6);
	assert.ok(Math.abs(summary.harmonicP90 - 0.18) < 1e-6);
	assert.throws(() => summarizeSplatParameters(params, { frameCount: 0 }), /positive/);
});
