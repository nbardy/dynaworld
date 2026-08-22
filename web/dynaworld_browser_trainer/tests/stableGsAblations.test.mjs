import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
	DEFAULT_MATERIAL_OPACITY_BIAS,
	TILED_OPACITY_MODELS,
	TILED_PIXEL_FILTER_MODES,
	buildGeometryPairSchedule,
	mip2dOpacityCompensation,
	pairedReferenceViewForStep,
	resolveGeometryConsistencyEvery,
	resolveTiledOpacityModel,
	resolveTiledPixelFilterMode,
} from "../trainerWebGpu3dTiled.js";

const root = path.dirname(fileURLToPath(import.meta.url));
const tiledSource = fs.readFileSync(path.join(root, "../trainerWebGpu3dTiled.js"), "utf8");
const previewSource = fs.readFileSync(path.join(root, "../trainerWebGpu3d.js"), "utf8");

test("shader ablation modes are closed enums with baseline-compatible material init", () => {
	assert.equal(resolveTiledPixelFilterMode(), TILED_PIXEL_FILTER_MODES.LEGACY_FLOOR);
	assert.equal(resolveTiledOpacityModel(), TILED_OPACITY_MODELS.COUPLED);
	assert.ok(Math.abs(1 / (1 + Math.exp(-DEFAULT_MATERIAL_OPACITY_BIAS)) - 0.99) < 1e-12);
	assert.throws(() => resolveTiledPixelFilterMode("mip-ish"), /pixelFilterMode/);
	assert.throws(() => resolveTiledOpacityModel("softmax"), /opacityModel/);
	assert.throws(() => resolveGeometryConsistencyEvery(0), /1 through 1024/);
});

test("2D Mip compensation preserves broad splats and attenuates subpixel splats", () => {
	assert.equal(mip2dOpacityCompensation(4, 0, 9, 0), 1);
	const broad = mip2dOpacityCompensation(4, 0.5, 9, 0.01);
	const subpixel = mip2dOpacityCompensation(0.001, 0, 0.001, 0.01);
	assert.ok(broad > 0.99 && broad <= 1);
	assert.ok(subpixel > 0 && subpixel < 0.1);
	assert.equal(mip2dOpacityCompensation(1, 2, 1, 0.1), 0);
});

test("paired depth rotates reference cameras without touching heldout views", () => {
	const train = [0, 2, 4, 7];
	assert.deepEqual(
		Array.from({ length: 6 }, (_, event) => pairedReferenceViewForStep(train, 2, event)),
		[4, 7, 0, 4, 7, 0],
	);
	assert.throws(() => pairedReferenceViewForStep([2], 2, 0), /at least two train views/);
	assert.throws(() => pairedReferenceViewForStep(train, 3, 0), /train split/);
});

test("geometry pair schedule keeps train-only moderate-baseline seed overlap", () => {
	const worldToCamera = (degrees) => {
		const angle = degrees * Math.PI / 180;
		const cosine = Math.cos(angle); const sine = Math.sin(angle);
		return new Float32Array([
			cosine, 0, sine, 0,
			0, 1, 0, 0,
			-sine, 0, cosine, 0,
			0, 0, 0, 1,
		]);
	};
	const cameras = [0, 10, 30, 70].map((angle, index) => ({
		name: `cam${index}`,
		role: index === 3 ? "heldout" : "train",
		intrinsics: new Float32Array([0.5, 0.5, 0.5, 0.5]),
		worldToCamera: worldToCamera(angle),
	}));
	const points = new Float32Array(40 * 6);
	for (let point = 0; point < 40; point += 1) {
		points[point * 6] = (point % 5 - 2) * 0.01;
		points[point * 6 + 1] = (Math.floor(point / 5) % 5 - 2) * 0.01;
		points[point * 6 + 2] = 2;
	}
	const schedule = buildGeometryPairSchedule({
		cameras, seedPoints: points, seedPointCount: 40,
	}, [0, 1, 2]);
	assert.deepEqual(schedule.candidatesByView[0].map(({ viewIndex }) => viewIndex), [2]);
	assert.ok(schedule.candidatesByView[1].some(({ viewIndex }) => viewIndex === 2));
	assert.equal(schedule.candidatesByView.flat().some(({ viewIndex }) => viewIndex === 3), false);
	assert.equal(schedule.contract.featureTrackCoVisibility, false);
	assert.throws(() => buildGeometryPairSchedule(
		{ cameras, seedPoints: points, seedPointCount: 40 },
		[0, 1, 2],
		{ minCoVisibleFraction: 1.1 },
	), /between 0 and 1/);
	assert.throws(() => buildGeometryPairSchedule(
		{ cameras, seedPoints: points, seedPointCount: 40 },
		[0, 1, 2],
		{ minRotationDegrees: 61, maxRotationDegrees: 60 },
	), /rotation bounds/);
});

test("fast fused shader carries compensated-filter and dual-opacity adjoints", () => {
	assert.match(tiledSource, /determinantScale\*\(c11\/det-filteredC11\/filteredDet\)/);
	assert.match(tiledSource, /gradMaterial=.*g\.colorPad\.w\*materialFactor\*\(1\.0-materialFactor\)/s);
	assert.match(tiledSource, /geometryAlphaGrad=geometryColorAlphaGrad\s*\+geometryDepthGrad/);
	assert.match(tiledSource, /geometryCheckpoints:array<GeometryCheckpoint>/);
	assert.match(tiledSource, /let meanAlpha=projectedGradient\.screen1\.z/);
	assert.match(tiledSource, /harmonicUpdate\.w,-16\.0,8\.0/);
	assert.match(previewSource, /opacityCompensation=.*sqrt/s);
	assert.match(previewSource, /sigmoid\(p\.harmonicPad\.w\+/);
});

test("cross-view depth is periodic and queue-local rather than a readback loss", () => {
	assert.match(tiledSource, /this\.stepCount % this\.geometryConsistencyEvery === 0/);
	assert.match(tiledSource, /buildGeometryPairSchedule\(this\.dataset, this\.trainViewIndices\)/);
	assert.match(tiledSource, /this\.tiledPipelines\.referenceDepth/);
	assert.match(tiledSource, /this\.tiledPipelines\.depthConsistency/);
	const trainStep = tiledSource.slice(
		tiledSource.indexOf("\ttrainStep("),
		tiledSource.indexOf("\n\tasync profileGpuStep", tiledSource.indexOf("\ttrainStep(")),
	);
	assert.doesNotMatch(trainStep, /mapAsync|onSubmittedWorkDone/);
	assert.match(trainStep, /queue\.submit\(\[encoder\.finish\(\)\]\)/);
});
