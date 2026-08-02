import assert from "node:assert/strict";
import test from "node:test";
import {
	assertContinuationStateCompatible,
	CONTINUATION_PARAMETER_SCHEMA,
	CONTINUATION_STATE_SCHEMA,
	DynamicSplatWebGpu3dTrainer,
	SPLAT_FLOATS,
} from "../trainerWebGpu3d.js";
import {
	assertTiledContinuationStateCompatible,
	DynamicSplatWebGpu3dTiledTrainer,
} from "../trainerWebGpu3dTiled.js";

globalThis.GPUBufferUsage ??= { COPY_DST: 1, MAP_READ: 2 };
globalThis.GPUMapMode ??= { READ: 1 };

function fakeBuffer(size, values = null) {
	const buffer = {
		size,
		bytes: new ArrayBuffer(size),
		async mapAsync() {},
		getMappedRange() { return this.bytes; },
		unmap() {},
		destroy() { this.destroyed = true; },
	};
	if (values) {
		new Uint8Array(buffer.bytes).set(
			new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
		);
	}
	return buffer;
}

function fakeDevice() {
	const queue = {
		writeCount: 0,
		writeBuffer(buffer, offset, data) {
			this.writeCount += 1;
			new Uint8Array(buffer.bytes).set(
				new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
				offset,
			);
		},
		submit(commandBuffers) {
			for (const operations of commandBuffers) {
				for (const operation of operations) {
					new Uint8Array(operation.destination.bytes, operation.destinationOffset, operation.size)
						.set(new Uint8Array(operation.source.bytes, operation.sourceOffset, operation.size));
				}
			}
		},
		async onSubmittedWorkDone() {},
	};
	return {
		queue,
		createBuffer({ size }) { return fakeBuffer(size); },
		createCommandEncoder() {
			const operations = [];
			return {
				copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
					operations.push({ source, sourceOffset, destination, destinationOffset, size });
				},
				finish() { return operations; },
			};
		},
		pushErrorScope() {},
		async popErrorScope() { return null; },
	};
}

function floatValues(length, offset) {
	return Float32Array.from({ length }, (_, index) => offset + index / 32);
}

function trainerContract(splatCount) {
	return {
		parameterSchema: CONTINUATION_PARAMETER_SCHEMA,
		splatFloats: SPLAT_FLOATS,
		splatCount,
		geometryScale: 2.5,
		frameCount: 16,
		cameraCount: 3,
		trainViewIndices: [0, 2],
	};
}

function continuationState(splatCount) {
	const parameterValues = splatCount * SPLAT_FLOATS;
	return {
		schema: CONTINUATION_STATE_SCHEMA,
		contract: trainerContract(splatCount),
		params: floatValues(parameterValues, 1),
		firstMoment: floatValues(parameterValues, 2),
		secondMoment: floatValues(parameterValues, 3),
		densityStats: floatValues(splatCount * 4, 4),
		initialParams: floatValues(parameterValues, 5),
		stepCount: 317,
		currentIndex: 1,
		totalRecycled: 19,
	};
}

function attachTrainerState(trainer, splatCount, source, device = fakeDevice()) {
	trainer.device = device;
	trainer.dataset = {
		geometryScale: source.contract.geometryScale,
		frameCount: source.contract.frameCount,
		cameras: Array.from({ length: source.contract.cameraCount }),
	};
	trainer.trainViewIndices = source.contract.trainViewIndices.slice();
	trainer.splatCount = splatCount;
	trainer.initialParams = source.initialParams.slice();
	trainer.stepCount = source.stepCount;
	trainer.currentIndex = source.currentIndex;
	trainer.totalRecycled = source.totalRecycled;
	trainer.buffers = {
		params: [
			fakeBuffer(source.params.byteLength, floatValues(source.params.length, -20)),
			fakeBuffer(source.params.byteLength, source.params),
		],
		firstMoment: fakeBuffer(source.firstMoment.byteLength, source.firstMoment),
		secondMoment: fakeBuffer(source.secondMoment.byteLength, source.secondMoment),
		stats: fakeBuffer(source.densityStats.byteLength, source.densityStats),
	};
	return trainer;
}

function bufferFloats(buffer) {
	return new Float32Array(buffer.bytes.slice(0));
}

test("continuation contract rejects incompatible or malformed bounded state", () => {
	const state = continuationState(3);
	assert.equal(assertContinuationStateCompatible(state, trainerContract(3)), state);
	assert.throws(
		() => assertContinuationStateCompatible(state, { ...trainerContract(4), splatCount: 4 }),
		/splat capacity/,
	);
	assert.throws(
		() => assertContinuationStateCompatible(state, {
			...trainerContract(3), trainViewIndices: [2, 0],
		}),
		/training-camera split/,
	);
	assert.throws(
		() => assertContinuationStateCompatible({ ...state, params: new Float32Array(1) }, trainerContract(3)),
		/Continuation params/,
	);
	const nonFinite = state.secondMoment.slice();
	nonFinite[4] = Number.NaN;
	assert.throws(
		() => assertContinuationStateCompatible({ ...state, secondMoment: nonFinite }, trainerContract(3)),
		/finite values/,
	);
});

test("base trainer exports current parameters and restores both ping-pong buffers", async () => {
	const source = continuationState(3);
	const exporter = attachTrainerState(new DynamicSplatWebGpu3dTrainer(null), 3, source);
	const exported = await exporter.exportContinuationState();
	assert.deepEqual(exported.params, source.params);
	assert.deepEqual(exported.firstMoment, source.firstMoment);
	assert.deepEqual(exported.secondMoment, source.secondMoment);
	assert.deepEqual(exported.densityStats, source.densityStats);
	assert.deepEqual(exported.initialParams, source.initialParams);
	assert.equal(exported.currentIndex, 1);

	const targetSeed = continuationState(3);
	const restorer = attachTrainerState(new DynamicSplatWebGpu3dTrainer(null), 3, targetSeed);
	restorer.dataset.width = 384;
	restorer.dataset.height = 288;
	await restorer.restoreContinuationState(exported);
	assert.deepEqual(bufferFloats(restorer.buffers.params[0]), source.params);
	assert.deepEqual(bufferFloats(restorer.buffers.params[1]), source.params);
	assert.deepEqual(bufferFloats(restorer.buffers.firstMoment), source.firstMoment);
	assert.deepEqual(bufferFloats(restorer.buffers.secondMoment), source.secondMoment);
	assert.deepEqual(bufferFloats(restorer.buffers.stats), source.densityStats);
	assert.deepEqual(restorer.initialParams, source.initialParams);
	assert.equal(restorer.stepCount, source.stepCount);
	assert.equal(restorer.currentIndex, source.currentIndex);
	assert.equal(restorer.totalRecycled, source.totalRecycled);
});

test("tiled trainer round trips growth state and only cumulative tile counters", async () => {
	const source = continuationState(16);
	source.stepCount = 600;
	source.initialSplatCount = 8;
	source.activeSplatCount = 16;
	source.totalRecycled = 8;
	const exporter = attachTrainerState(new DynamicSplatWebGpu3dTiledTrainer(null), 16, source);
	exporter.initialSplatCount = source.initialSplatCount;
	exporter.activeSplatCount = source.activeSplatCount;
	const counters = new Uint32Array([91, 92, 93, 94, 17, 311, 96, 16, 98, 4]);
	exporter.buffers.counters = fakeBuffer(counters.byteLength, counters);

	const exported = await exporter.exportContinuationState();
	assert.deepEqual(exported.cumulativeTileDiagnostics, {
		tileOverflowTotal: 17,
		maxTileOccupancyEver: 311,
		projectionVjpHalfSaturationsTotal: 4,
	});

	const restorer = attachTrainerState(new DynamicSplatWebGpu3dTiledTrainer(null), 16, source);
	restorer.buffers.counters = fakeBuffer(counters.byteLength, new Uint32Array(10).fill(0xffffffff));
	await restorer.restoreContinuationState(exported);
	assert.deepEqual(new Uint32Array(restorer.buffers.counters.bytes),
		new Uint32Array([0, 0, 0, 0, 17, 311, 0, 16, 0, 4]));
	assert.equal(restorer.initialSplatCount, 8);
	assert.equal(restorer.activeSplatCount, 16);
	assert.equal(restorer.totalRecycled, 8);
	assert.equal(restorer.lastProjectionParamIndex, source.currentIndex);
	assert.deepEqual(restorer.lastLossBreakdown, exported.cumulativeTileDiagnostics);
});

test("tiled restore rejects density-schedule drift before writing GPU state", async () => {
	const source = continuationState(16);
	source.stepCount = 599;
	source.initialSplatCount = 8;
	source.activeSplatCount = 16;
	const device = fakeDevice();
	const trainer = attachTrainerState(new DynamicSplatWebGpu3dTiledTrainer(null), 16, source, device);
	trainer.buffers.counters = fakeBuffer(10 * Uint32Array.BYTES_PER_ELEMENT);
	assert.throws(
		() => assertTiledContinuationStateCompatible(source, { splatCount: 16 }),
		/density schedule/,
	);
	await assert.rejects(() => trainer.restoreContinuationState(source), /density schedule/);
	assert.equal(device.queue.writeCount, 0);
});
