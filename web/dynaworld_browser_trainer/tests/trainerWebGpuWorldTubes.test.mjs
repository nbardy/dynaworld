import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { compileWorldTubeCpu } from "../worldTubesMath.js";
import { renderSnapshotFrame, snapshotUpdateRatios } from "../snapshotMetrics.js";
import { WorldTubesWebGpuTrainer } from "../trainerWebGpuWorldTubes.js";
import { CONTINUATION_STATE_SCHEMA } from "../trainerWebGpu3d.js";
const reference = JSON.parse(readFileSync(new URL("./fixtures/world_tubes_reference.json", import.meta.url)));

test("world compiler matches the canonical float64 SPD4 pushforward, including depth covariance", () => {
	for (let camera = 0; camera < 2; camera++) for (let atom = 0; atom < 2; atom++) {
		const trace = compileWorldTubeCpu(reference.params.slice(atom * 24, atom * 24 + 24),
			reference.cameras[camera], reference.width / reference.height, (0.3 / reference.height) ** 2);
		for (let k = 0; k < 20; k++) {
			const expected = reference.traces[(camera * 2 + atom) * 20 + k];
			assert.ok(Math.abs(trace[k] - expected) < 1e-9 * Math.max(1, Math.abs(expected)), `camera ${camera} atom ${atom} trace ${k}`);
		}
	}
});

test("validation uses the same UVT marginal, conditional depth and nonuniform global times", () => {
	const times = [0, 0.13, 0.61, 1];
	const dataset = { width: reference.width, height: reference.height, cameras: reference.cameras,
		frameCount: 4, frameTimesNormalized: times, frames: new Float32Array(24 * 18 * 4 * 2 * 4) };
	const params = Float32Array.from(reference.params);
	for (let viewIndex = 0; viewIndex < 2; viewIndex++) for (let frameIndex = 0; frameIndex < 4; frameIndex++) {
		const image = renderSnapshotFrame(dataset, params, { viewIndex, frameIndex, modelMode: 2 });
		for (let j = 0; j < 4; j++) {
			const index = (viewIndex * 4 + frameIndex) * 4 + j, sample = reference.samples[index];
			const pixel = Math.floor(sample[1] * dataset.height) * dataset.width + Math.floor(sample[0] * dataset.height);
			for (let c = 0; c < 3; c++) assert.ok(Math.abs(image.rgb[pixel * 3 + c] - reference.rgb[index * 3 + c]) < 1e-6);
		}
	}
});

test("training samples resident timestamps and excludes interleaved heldout cameras", () => {
	const trainer = new WorldTubesWebGpuTrainer(null);
	trainer.dataset = { width: 2, height: 2, frameCount: 3, frameTimesNormalized: [0.13, 0.61, 0.89],
		frames: new Float32Array(2 * 2 * 3 * 3 * 4) };
	trainer.trainViewIndices = [0, 2];
	const samples = trainer.trainingSamples({ samplesPerStep: 12, camerasPerStep: 2 });
	assert.deepEqual(samples.views, [0, 2]);
	for (let i = 0; i < 12; i++) assert.ok(Math.abs(samples.data[i * 8 + 2] - trainer.dataset.frameTimesNormalized[Math.floor(i / 2) % 3]) < 1e-7);
});

test("world temporal-width updates are not reported as static-mixture or harmonic updates", () => {
	const before=Float32Array.from(reference.params),after=before.slice();
	after[3]+=0.1;
	const ratios=snapshotUpdateRatios(before,after,{modelMode:2});
	assert.ok(ratios.temporalLogSigma.updateRms>0);
	assert.equal(ratios.staticMix,undefined);
	assert.equal(ratios.harmonic,undefined);
	assert.equal(ratios.materialOpacity,undefined);
});

test("public validation, error previews and continuation use the world-tube contract", async () => {
	const trainer = new WorldTubesWebGpuTrainer(null);
	const params = Float32Array.from(reference.params);
	trainer.dataset = { width: 24, height: 18, frameCount: 4, viewCount: 2, geometryScale: 1,
		frameTimesNormalized: [0, 0.13, 0.61, 1], cameras: reference.cameras,
		frames: new Float32Array(24 * 18 * 4 * 2 * 4) };
	trainer.splatCount = 2; trainer.trainViewIndices = [0, 1];
	trainer.readParams = async () => params;
	const metrics = await trainer.readValidationMetrics();
	assert.ok(Number.isFinite(metrics.gridLoss) && metrics.gridLoss > 0);
	const preview = await trainer.readPreviewErrorImage({ time: 0.6 });
	assert.equal(preview.frame, 2);
	assert.ok(preview.meanAbs > 0 && preview.data.some(v => v > 0));
	const state = { schema: CONTINUATION_STATE_SCHEMA, contract: trainer.continuationContract(),
		params, initialParams: params, firstMoment: new Float32Array(48), secondMoment: new Float32Array(48),
		densityStats: new Float32Array(8), stepCount: 7, currentIndex: 1, totalRecycled: 0 };
	assert.equal(trainer.assertContinuationStateCompatible(state), state);
	assert.throws(() => trainer.assertContinuationStateCompatible({ ...state,
		contract: { ...state.contract, parameterSchema: "world-tube-spd4-block-24f/v1" } }), /parameter schema/);
});

import { SPLAT_FLOATS } from "../trainerWebGpu3d.js";
import { temporalGate } from "../snapshotMetrics.js";
import {
	WORLD_TUBES_BROWSER_CONTRACT,
	makeWorldTubeInitialParams,
	worldTubeTemporalDensity,
} from "../trainerWebGpuWorldTubes.js";

test("World Tubes maps the shared browser record to a per-tube SPD4 temporal chart", () => {
	const source = new Float32Array(SPLAT_FLOATS * 2);
	source[3] = 0.92;
	source[7] = 0.5;
	source[8] = 0.4;
	source[SPLAT_FLOATS + 7] = 2;
	const params = makeWorldTubeInitialParams(source, { temporalSigma: 0.25 });
	assert.ok(Math.abs(params[3] - Math.log(0.25)) < 1e-6);
	assert.equal(params[7], 0.5);
	assert.equal(params[8], 0);
	assert.equal(params[SPLAT_FLOATS + 7], 1);
});

test("World Tubes temporal density is Gaussian and matches snapshot validation semantics", () => {
	const params = makeWorldTubeInitialParams(new Float32Array(SPLAT_FLOATS), { temporalSigma: 0.2 });
	params[7] = 0.4;
	assert.ok(Math.abs(worldTubeTemporalDensity(params, 0, 0.4) - 1) < 1e-12);
	const expected = Math.exp(-0.5);
	assert.ok(Math.abs(worldTubeTemporalDensity(params, 0, 0.6) - expected) < 1e-6);
	assert.ok(Math.abs(temporalGate(params, 0, 0.6, 0.35, 2) - expected) < 1e-6);
});

test("browser contract states the bounded compiler omissions", () => {
	assert.match(WORLD_TUBES_BROWSER_CONTRACT.training, /WGSL/);
	assert.match(WORLD_TUBES_BROWSER_CONTRACT.rendering, /UVT marginal.*conditional-depth/);
	assert.ok(WORLD_TUBES_BROWSER_CONTRACT.omissions.some((item) => item.includes("interval-atlas")));
});
