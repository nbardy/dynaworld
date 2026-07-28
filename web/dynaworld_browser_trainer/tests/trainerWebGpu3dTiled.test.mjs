import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
	DynamicSplatWebGpu3dTiledTrainer,
	densityDispatchesForStep,
	ellipseIntersectsRect,
	fullFramePairForStep,
	MAX_SCALE_ASPECT_RATIO,
	opacityAwarePixelBounds,
	ROTATION_LR_FROM_MOTION,
	resolveSsimRadius,
	resolveTiledCapacity,
	SCALE_LR_FROM_COLOR,
	windowedL1DssimCpu,
} from "../trainerWebGpu3dTiled.js";

const source = readFileSync(new URL("../trainerWebGpu3dTiled.js", import.meta.url), "utf8");

function assertClose(actual, expected, tolerance = 1e-7) {
	const scale = Math.max(1, Math.abs(actual), Math.abs(expected));
	assert.ok(Math.abs(actual - expected) <= tolerance * scale,
		`expected ${actual} to be within ${tolerance} relative of ${expected}`);
}

test("tiled capacity reserves growth while respecting explicit bounds", () => {
	assert.equal(resolveTiledCapacity(8), 24);
	assert.equal(resolveTiledCapacity(768), 2304);
	assert.equal(resolveTiledCapacity(768, 512), 768);
	assert.equal(resolveTiledCapacity(768, 1000.9), 1000);
	assert.equal(resolveTiledCapacity(1024, 4096), 4096);
	assert.equal(resolveTiledCapacity(2048, 8192), 4096);
	assert.throws(() => resolveTiledCapacity(7), /at least 8/);
	assert.throws(() => resolveTiledCapacity(8.5), /integer/);
});

test("SSIM radius accepts the benchmark range and preserves the 11x11 default", () => {
	assert.equal(resolveSsimRadius(), 5);
	assert.equal(resolveSsimRadius(0), 0);
	assert.equal(resolveSsimRadius(15), 15);
	assert.throws(() => resolveSsimRadius(-1), /0 through 15/);
	assert.throws(() => resolveSsimRadius(5.5), /integer/);
});

test("density schedule fills reserved slots once, then recycles slowly", () => {
	assert.equal(densityDispatchesForStep(1536, 3072, 599), 0);
	assert.equal(densityDispatchesForStep(1536, 3072, 600), 4);
	assert.equal(densityDispatchesForStep(1536, 3072, 10100), 4);
	assert.equal(densityDispatchesForStep(1536, 3072, 10200), 0);
	assert.equal(densityDispatchesForStep(1536, 3072, 10500), 4);
	assert.equal(densityDispatchesForStep(1536, 3072, 60000), 4);
	assert.equal(densityDispatchesForStep(1536, 3072, 60500), 0);
	assert.equal(densityDispatchesForStep(1536, 4096, 16500), 4);
	assert.equal(densityDispatchesForStep(1536, 4096, 17000), 4);
});

test("full-frame schedule shuffles and visits every camera/time pair before cycling", () => {
	const trainViews = [2, 5, 9];
	const cycle = Array.from({ length: 6 }, (_, step) => fullFramePairForStep(trainViews, 2, step));
	assert.equal(new Set(cycle.map(({ viewIndex, frameIndex }) => `${viewIndex}:${frameIndex}`)).size, 6);
	assert.notEqual(cycle[0].frameIndex, cycle[1].frameIndex);
	assert.deepEqual(fullFramePairForStep(trainViews, 2, 6), cycle[0]);
	assert.deepEqual(fullFramePairForStep(trainViews, 2, -4), cycle[0]);
	assert.throws(() => fullFramePairForStep([], 2, 0), /train view/);
});

test("opacity-aware pixel bounds shrink support and clip to the image", () => {
	const projection = {
		valid: true,
		center: [0.5, 0.5],
		covariance: [1 / 64, 0, 1 / 64],
	};
	assert.deepEqual(opacityAwarePixelBounds(projection, 1, 8, 8), {
		minX: 1, maxX: 7, minY: 1, maxY: 7, qLimit: 9,
	});
	const lowOpacity = opacityAwarePixelBounds(projection, 0.2, 8, 8, 0.1);
	assert.deepEqual(
		{ minX: lowOpacity.minX, maxX: lowOpacity.maxX, minY: lowOpacity.minY, maxY: lowOpacity.maxY },
		{ minX: 2, maxX: 6, minY: 2, maxY: 6 },
	);
	assertClose(lowOpacity.qLimit, 2 * Math.log(2));
	assert.equal(opacityAwarePixelBounds(projection, 0.1, 8, 8, 0.1), null);
	assert.equal(opacityAwarePixelBounds({ ...projection, valid: false }, 1, 8, 8), null);
});

test("ellipse/rectangle test handles containment, edge intersection, and separation", () => {
	const conic = [1, 0, 1];
	assert.equal(ellipseIntersectsRect([0, 0], conic, 1,
		{ minX: -0.2, minY: -0.2, maxX: 0.2, maxY: 0.2 }), true);
	assert.equal(ellipseIntersectsRect([0, 0], conic, 1,
		{ minX: 0.8, minY: -0.1, maxX: 1.2, maxY: 0.1 }), true);
	assert.equal(ellipseIntersectsRect([0, 0], conic, 1,
		{ minX: 1.1, minY: -0.1, maxX: 1.3, maxY: 0.1 }), false);
	assert.equal(ellipseIntersectsRect([0, 0], [4, 0, 1], 1,
		{ minX: 0.45, minY: -0.1, maxX: 0.55, maxY: 0.1 }), true);
	assert.equal(ellipseIntersectsRect([0, 0], [4, 0, 1], 1,
		{ minX: 0.6, minY: -0.1, maxX: 0.8, maxY: 0.1 }), false);
});

test("windowed L1 plus SSIM is zero with zero gradient for identical images", () => {
	const image = Float64Array.from([
		0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7,
		0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8,
	]);
	const result = windowedL1DssimCpu(image, image, 3, 2, { radius: 1 });
	assertClose(result.loss, 0, 1e-12);
	assertClose(result.l1, 0, 1e-12);
	assertClose(result.dssim, 0, 1e-12);
	for (const gradient of result.gradient) assertClose(gradient, 0, 1e-12);
});

test("windowed L1 plus SSIM analytic gradient matches finite differences", () => {
	const width = 3;
	const height = 2;
	const length = width * height * 3;
	const target = Float64Array.from(
		{ length },
		(_, index) => 0.08 + 0.8 * ((index * 7) % length) / (length - 1),
	);
	const prediction = Float64Array.from(
		target,
		(value, index) => value + (index % 2 ? -0.025 : 0.03),
	);
	const analytic = windowedL1DssimCpu(prediction, target, width, height, { radius: 1 });
	const epsilon = 1e-5;
	for (let index = 0; index < length; index += 1) {
		const plus = Float64Array.from(prediction);
		const minus = Float64Array.from(prediction);
		plus[index] += epsilon;
		minus[index] -= epsilon;
		const finiteDifference = (
			windowedL1DssimCpu(plus, target, width, height, { radius: 1 }).loss
			- windowedL1DssimCpu(minus, target, width, height, { radius: 1 }).loss
		) / (2 * epsilon);
		assertClose(analytic.gradient[index], finiteDifference, 1e-6);
	}
});

test("tiled trainer source preserves the full-frame shared-backward contract", () => {
	const trainStep = DynamicSplatWebGpu3dTiledTrainer.prototype.trainStep.toString();
	assert.match(trainStep, /fullFramePairForStep/);
	assert.match(trainStep, /this\.lastSampleCount\s*=\s*this\.pixelCount/);
	assert.match(trainStep, /this\.tilesX,\s*this\.tilesY/);
	assert.match(trainStep, /dispatchWorkgroupsIndirect/);
	assert.match(source, /fn\s+project_and_bin/);
	assert.match(source, /fn\s+sort_tiles/);
	assert.match(source, /depthKeys/);
	assert.match(source, /workgroupUniformLoad\(&tileSortCount\)/);
	assert.match(source, /span>=max\(count,1u\)/);
	assert.match(source, /width<=sortCount/);
	assert.doesNotMatch(source, /width<=cfg\.tileCapacity/);
	assert.match(source, /fn\s+raster_forward/);
	assert.match(source, /fn\s+ssim_stats/);
	assert.match(source, /fn\s+ssim_gradient/);
	assert.match(source, /fn\s+pair_backward/);
	assert.match(source, /fn\s+reduce_update/);
	assert.equal(SCALE_LR_FROM_COLOR, 0.30);
	assert.equal(ROTATION_LR_FROM_MOTION, 1.25);
	assert.equal(MAX_SCALE_ASPECT_RATIO, 3);
	assert.match(source, /gaussianPairSlots/);
	assert.doesNotMatch(trainStep, /samplesPerStep|sampleIndices/);
});
