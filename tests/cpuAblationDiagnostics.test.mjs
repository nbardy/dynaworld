import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { computeCameraStressMetrics } from "../cameraStressMetrics.js";
import {
	mip2dOpacityCompensation,
	renderSnapshotFrame,
	separatedWeightedDepthModes,
} from "../snapshotMetrics.js";
import { SPLAT_FLOATS, projectAnisotropicGaussianCpu } from "../trainerWebGpu3d.js";

const MATERIAL_BIAS = 4.59511985013459;
const camera = {
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

function datasetFixture({ width = 16, height = 12 } = {}) {
	return {
		width,
		height,
		frameCount: 1,
		viewCount: 1,
		trainViewCount: 1,
		heldoutViewIndex: -1,
		cameras: [camera],
		frames: new Float32Array(width * height * 4),
		seedPointCount: 2,
		seedPoints: new Float32Array([
			-0.1, 0, 1, 1, 0, 0,
			0.1, 0, 2, 0, 0, 1,
		]),
	};
}

function paramsFixture(splats) {
	const params = new Float32Array(splats.length * SPLAT_FLOATS);
	for (let index = 0; index < splats.length; index += 1) {
		const splat = splats[index];
		const base = index * SPLAT_FLOATS;
		params.set(splat.center, base);
		params[base + 3] = 1;
		params[base + 7] = 0.5;
		params[base + 11] = splat.materialLogit ?? 0;
		params.set((splat.scales ?? [0.16, 0.14, 0.12]).map(Math.log), base + 12);
		params.set([0, 0, 0, 1], base + 16);
		params.set(splat.color ?? [0.8, 0.2, 0.1], base + 20);
		const opacity = splat.opacity ?? 0.7;
		params[base + 23] = Math.log(opacity / (1 - opacity));
	}
	return params;
}

test("legacy coupled rendering ignores lane 11 and matches explicit defaults byte-for-byte", () => {
	const dataset = datasetFixture();
	const params = paramsFixture([{ center: [0, 0, 1.5], materialLogit: -50 }]);
	const implicit = renderSnapshotFrame(dataset, params);
	const explicit = renderSnapshotFrame(dataset, params, {
		pixelFilterMode: "legacy-floor",
		opacityModel: "coupled",
		materialOpacityBias: MATERIAL_BIAS,
	});
	assert.deepEqual(explicit.rgb, implicit.rgb);
	assert.deepEqual(explicit.coverage, implicit.coverage);

	const changedPadding = params.slice();
	changedPadding[11] = 50;
	const changed = renderSnapshotFrame(dataset, changedPadding);
	assert.deepEqual(changed.rgb, implicit.rgb);
	assert.deepEqual(changed.coverage, implicit.coverage);
});

test("dual opacity changes appearance while geometry diagnostics retain geometry opacity", () => {
	const dataset = datasetFixture();
	const opaqueMaterial = paramsFixture([
		{ center: [0, 0, 1.5], materialLogit: 20, opacity: 0.8 },
	]);
	const transparentMaterial = opaqueMaterial.slice();
	transparentMaterial[11] = -MATERIAL_BIAS - 4.59511985013459;
	const options = { opacityModel: "dual", collectGeometryDiagnostics: true };
	const opaque = renderSnapshotFrame(dataset, opaqueMaterial, options);
	const transparent = renderSnapshotFrame(dataset, transparentMaterial, options);
	assert.notDeepEqual(transparent.rgb, opaque.rgb);
	assert.notDeepEqual(transparent.coverage, opaque.coverage);
	assert.deepEqual(transparent.geometryCoverage, opaque.geometryCoverage);
	assert.deepEqual(transparent.depthMean, opaque.depthMean);
	assert.deepEqual(transparent.depthStd, opaque.depthStd);
});

test("2D Mip compensation is the clamped determinant ratio and reduces peak coverage", () => {
	const dataset = datasetFixture({ width: 8, height: 8 });
	const params = paramsFixture([{ center: [0, 0, 1.5], scales: [0.015, 0.012, 0.01] }]);
	const projection = projectAnisotropicGaussianCpu({
		center: [0, 0, 1.5],
		logScales: [0.015, 0.012, 0.01].map(Math.log),
		quaternion: [0, 0, 0, 1],
		camera,
		aspect: 1,
		height: 8,
	});
	const compensation = mip2dOpacityCompensation(projection, 8);
	assert.ok(compensation > 0 && compensation < 1);
	const legacy = renderSnapshotFrame(dataset, params);
	const compensated = renderSnapshotFrame(dataset, params, {
		pixelFilterMode: "mip-2d-compensated",
	});
	assert.ok(Math.max(...compensated.coverage) < Math.max(...legacy.coverage));
	assert.equal(mip2dOpacityCompensation({ valid: false }, 8), 0);
});

test("separated weighted depth modes report a diagnostic second layer without a prior", () => {
	assert.deepEqual(separatedWeightedDepthModes([
		{ depth: 1, weight: 0.45 },
		{ depth: 1.02, weight: 0.05 },
		{ depth: 2, weight: 0.5 },
	]), { multiLayer: true, secondLayerMass: 0.5, splitDepthRatio: 2 / 1.02 });
	assert.deepEqual(separatedWeightedDepthModes([
		{ depth: 1, weight: 0.5 },
		{ depth: 1.02, weight: 0.5 },
	]), { multiLayer: false, secondLayerMass: 0, splitDepthRatio: 1 });

	const dataset = datasetFixture();
	const params = paramsFixture([
		{ center: [0, 0, 1], opacity: 0.45 },
		{ center: [0, 0, 2], opacity: 0.8, color: [0.1, 0.2, 0.9] },
	]);
	const rendered = renderSnapshotFrame(dataset, params, {
		collectGeometryDiagnostics: true,
	});
	assert.ok(rendered.geometryCoveredRays > 0);
	assert.ok(rendered.multiLayerRayFraction > 0);
	assert.ok(rendered.meanSecondLayerMass > 0.1);
});

test("camera stress forwards ablations and labels multimodal depth diagnostic-only", () => {
	const dataset = datasetFixture();
	const params = paramsFixture([
		{ center: [0, 0, 1], opacity: 0.45, materialLogit: -MATERIAL_BIAS },
		{ center: [0, 0, 2], opacity: 0.8, materialLogit: -MATERIAL_BIAS },
	]);
	const target = renderSnapshotFrame(dataset, params).rgb;
	for (let pixel = 0; pixel < dataset.width * dataset.height; pixel += 1) {
		for (let channel = 0; channel < 3; channel += 1) {
			dataset.frames[pixel * 4 + channel] = target[pixel * 3 + channel];
		}
		dataset.frames[pixel * 4 + 3] = 1;
	}
	const stress = computeCameraStressMetrics(dataset, params, {
		viewIndices: [0],
		maxHeight: 12,
		pixelFilterMode: "mip-2d-compensated",
		opacityModel: "dual",
		materialOpacityBias: MATERIAL_BIAS,
	});
	assert.equal(stress.contract.pixelFilterMode, "mip-2d-compensated");
	assert.equal(stress.contract.opacityModel, "dual");
	assert.equal(stress.contract.multimodalDepth.trainingLoss, false);
	assert.equal(stress.contract.multimodalDepth.externalDepthPrior, false);
	assert.ok(Number.isFinite(stress.train.poseMultiLayerRayFraction));
});

test("validation worker forwards all CPU ablation options", () => {
	const source = readFileSync(new URL("../validationWorker.js", import.meta.url), "utf8");
	assert.match(source, /pixelFilterMode: data\.options\?\.pixelFilterMode/);
	assert.match(source, /opacityModel: data\.options\?\.opacityModel/);
	assert.match(source, /materialOpacityBias: data\.options\?\.materialOpacityBias/);
	assert.match(source, /trainMultiLayerRayFraction/);
	assert.match(source, /heldoutSecondLayerMass/);
});

test("invalid CPU ablation modes fail loudly", () => {
	const dataset = datasetFixture();
	const params = paramsFixture([{ center: [0, 0, 1.5] }]);
	assert.throws(() => renderSnapshotFrame(dataset, params, {
		pixelFilterMode: "mystery",
	}), /pixelFilterMode/);
	assert.throws(() => renderSnapshotFrame(dataset, params, {
		opacityModel: "triple",
	}), /opacityModel/);
});
