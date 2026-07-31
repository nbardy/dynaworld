import assert from "node:assert/strict";
import test from "node:test";
import {
	computeMultiviewSamples,
	decodeFrameRgb,
	drawTargetFrame,
	FRAME_BANK_FORMAT_RGBA8,
	FRAME_BANK_FORMAT_RGBA32_FLOAT,
	FRAME_WEIGHT_BYTE_SCALE,
	loadCalibratedMulticamDataset,
	normalizedMotionLossWeights,
	readFrameLossWeight,
	resolveFrameBank,
	validateCalibratedMulticamBundle,
	writeNormalizedFrameLossWeights,
} from "../dataset.js";

function referenceSamples(frames, backgrounds, width, height, frameCount, trainViewCount) {
	const pixels = width * height;
	const motion = [];
	const staticSamples = [];
	const weights = new Float32Array(frames.length / 4);
	for (let view = 0; view < trainViewCount; view += 1) {
		for (let frame = 0; frame < frameCount; frame += 1) {
			const energies = new Float32Array(pixels);
			for (let pixel = 0; pixel < pixels; pixel += 1) {
				const base = ((view * frameCount + frame) * pixels + pixel) * 4;
				const bgBase = (view * pixels + pixel) * 4;
				const dr = frames[base] - backgrounds[bgBase];
				const dg = frames[base + 1] - backgrounds[bgBase + 1];
				const db = frames[base + 2] - backgrounds[bgBase + 2];
				const energy = (dr * dr + dg * dg + db * db) / 3;
				energies[pixel] = energy;
				const packed = (view * frameCount + frame) * pixels + pixel;
				if (energy > 0.0006) motion.push({ packed, energy });
				else if (energy < 0.00035) staticSamples.push(packed);
			}
			weights.set(
				normalizedMotionLossWeights(energies),
				(view * frameCount + frame) * pixels,
			);
		}
	}
	motion.sort((left, right) => right.energy - left.energy);
	const maxSamples = 16384;
	const staticKept = staticSamples.length <= maxSamples
		? staticSamples
		: Array.from({ length: maxSamples }, (_, index) =>
			staticSamples[Math.floor((index + 0.5) * staticSamples.length / maxSamples)]);
	return {
		motionSamples: new Uint32Array(motion.slice(0, maxSamples).map(({ packed }) => packed)),
		staticSamples: new Uint32Array(staticKept),
		weights,
	};
}

test("compact RGB decoding is byte-exact and loss weights stay within half a quantization step", () => {
	const bytes = Uint8Array.from([
		0, 127, 255, 0,
		17, 33, 201, 254,
	]);
	const dataset = {
		width: 2,
		height: 1,
		frameCount: 1,
		viewCount: 1,
		frames: bytes,
		frameBank: { format: FRAME_BANK_FORMAT_RGBA8, data: bytes },
	};
	assert.deepEqual([...decodeFrameRgb(dataset, 0, 0)], [
		0, Math.fround(127 / 255), 1,
		Math.fround(17 / 255), Math.fround(33 / 255), Math.fround(201 / 255),
	]);
	assert.equal(readFrameLossWeight(dataset, 0), 0);
	assert.equal(readFrameLossWeight(dataset, 4), 2);

	for (let index = 0; index <= 1000; index += 1) {
		const weight = index * 2 / 1000;
		const encoded = Math.round(weight * FRAME_WEIGHT_BYTE_SCALE);
		assert.ok(Math.abs(encoded / FRAME_WEIGHT_BYTE_SCALE - weight) <= 1 / 254 + 1e-12);
	}
});

test("compact normalized weights preserve an exact mean-one byte sum", () => {
	const energies = Float32Array.from(
		{ length: 257 },
		(_, index) => ((index * 37) % 257) / 257 * 0.006,
	);
	const weights = normalizedMotionLossWeights(energies);
	const bytes = new Uint8Array(weights.length * 4);
	const bank = { format: FRAME_BANK_FORMAT_RGBA8, data: bytes };
	writeNormalizedFrameLossWeights(bank, 0, weights);

	let encodedSum = 0;
	let maximumError = 0;
	for (let pixel = 0; pixel < weights.length; pixel += 1) {
		const encoded = bytes[pixel * 4 + 3];
		encodedSum += encoded;
		maximumError = Math.max(
			maximumError,
			Math.abs(encoded / FRAME_WEIGHT_BYTE_SCALE - weights[pixel]),
		);
		assert.ok(encoded <= FRAME_WEIGHT_BYTE_SCALE * 2);
	}
	assert.equal(encodedSum, FRAME_WEIGHT_BYTE_SCALE * weights.length);
	assert.ok(maximumError <= 1 / FRAME_WEIGHT_BYTE_SCALE + 1e-7);
});

test("normalized frame-bank encoding rejects non-finite weights before mutation", () => {
	const bytes = new Uint8Array(8);
	const bank = { format: FRAME_BANK_FORMAT_RGBA8, data: bytes };
	assert.throws(
		() => writeNormalizedFrameLossWeights(bank, 0, Float32Array.of(1, Number.NaN)),
		/weights must be finite/,
	);
	assert.deepEqual(bytes, new Uint8Array(8));
});

test("target drawing copies compact RGB bytes directly and forces opaque display alpha", () => {
	const frames = Uint8Array.from([
		3, 17, 251, 0,
		99, 101, 103, 254,
	]);
	const dataset = {
		width: 2,
		height: 1,
		frameCount: 1,
		viewCount: 1,
		frames,
		frameBank: { format: FRAME_BANK_FORMAT_RGBA8, data: frames },
		background: new Float32Array(8),
	};
	let displayed = null;
	const context = {
		createImageData(width, height) {
			return { data: new Uint8ClampedArray(width * height * 4) };
		},
		putImageData(image) {
			displayed = image.data;
		},
	};
	const canvas = { width: 0, height: 0, getContext: () => context };
	assert.equal(drawTargetFrame(canvas, dataset, 0), 0);
	assert.deepEqual([...displayed], [3, 17, 251, 255, 99, 101, 103, 255]);
	assert.equal(canvas.width, 2);
	assert.equal(canvas.height, 1);
});

test("compact sampling preserves full-sort indices and bounded loss-weight error beyond 16K candidates", () => {
	const width = 40000;
	const height = 1;
	const frameCount = 1;
	const frames = new Float32Array(width * 4);
	const compact = new Uint8Array(width * 4);
	const backgrounds = new Float32Array(width * 4);
	for (let pixel = 0; pixel < width; pixel += 1) {
		const base = pixel * 4;
		const byte = pixel % 2 === 0 ? 180 + (pixel % 67) : 128;
		for (let channel = 0; channel < 3; channel += 1) {
			frames[base + channel] = Math.fround(byte / 255);
			compact[base + channel] = byte;
			backgrounds[base + channel] = Math.fround(128 / 255);
		}
		frames[base + 3] = 1;
		compact[base + 3] = FRAME_WEIGHT_BYTE_SCALE;
		backgrounds[base + 3] = 1;
	}
	const expected = referenceSamples(frames, backgrounds, width, height, frameCount, 1);
	const floatResult = computeMultiviewSamples(frames, backgrounds, width, height, frameCount, 1);
	const compactResult = computeMultiviewSamples(
		{ format: FRAME_BANK_FORMAT_RGBA8, data: compact },
		backgrounds,
		width,
		height,
		frameCount,
		1,
	);
	assert.deepEqual(floatResult.motionSamples, expected.motionSamples);
	assert.deepEqual(floatResult.staticSamples, expected.staticSamples);
	assert.deepEqual(compactResult.motionSamples, expected.motionSamples);
	assert.deepEqual(compactResult.staticSamples, expected.staticSamples);
	for (let pixel = 0; pixel < width; pixel += 1) {
		const compactWeight = readFrameLossWeight(
			{ format: FRAME_BANK_FORMAT_RGBA8, data: compact },
			pixel * 4,
		);
		assert.ok(Math.abs(compactWeight - expected.weights[pixel]) <= 1 / 254 + 1e-7);
		for (let channel = 0; channel < 3; channel += 1) {
			assert.equal(compact[pixel * 4 + channel], Math.round(frames[pixel * 4 + channel] * 255));
		}
	}
});

function installAtlasMocks() {
	const original = {
		fetch: globalThis.fetch,
		createImageBitmap: globalThis.createImageBitmap,
		document: globalThis.document,
		crossOriginIsolated: globalThis.crossOriginIsolated,
	};
	const width = 2;
	const height = 1;
	const frameCount = 2;
	const bundle = {
		version: "dynaworld_browser_multicam_dataset/v1",
		name: "compact fixture",
		decode_size: [width, height],
		frame_count: frameCount,
		frame_times_seconds: [0, 1],
		cameras: [
			{
				name: "cam00", role: "train", frame_atlas_url: "cam00.png",
				intrinsics: [1, 1, 0.5, 0.5],
				world_to_camera: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
			},
			{
				name: "cam01", role: "heldout", frame_atlas_url: "cam01.png",
				intrinsics: [1, 1, 0.5, 0.5],
				world_to_camera: [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
			},
		],
		seed_points_xyzrgb: [[0, 0, 1, 1, 0, 0]],
		seed_source: "fixture",
		dataset_contract: {
			anchor_camera: "cam00",
			pose_source: "neural_3d_llff_opencv_relative_pinhole_v2",
		},
		seed_coordinate_frame: "cam00_opencv",
		frame_indices: [0, 1],
	};
	const atlasBytes = new Map([
		["cam00.png", [
			Uint8ClampedArray.from([1, 2, 3, 9, 4, 5, 6, 8]),
			Uint8ClampedArray.from([7, 8, 9, 7, 10, 11, 12, 6]),
		]],
		["cam01.png", [
			Uint8ClampedArray.from([13, 14, 15, 5, 16, 17, 18, 4]),
			Uint8ClampedArray.from([19, 20, 21, 3, 22, 23, 24, 2]),
		]],
	]);
	globalThis.fetch = async (url) => {
		if (String(url).endsWith("coffee_martini_train17_holdout1.json")) {
			return { ok: true, json: async () => bundle };
		}
		return { ok: atlasBytes.has(String(url)), blob: async () => ({ url: String(url) }) };
	};
	globalThis.createImageBitmap = async ({ url }) => ({
		url,
		width: width * frameCount,
		height,
		close() {},
	});
	globalThis.document = {
		createElement(name) {
			assert.equal(name, "canvas");
			let bitmap = null;
			let frame = 0;
			return {
				width: 0,
				height: 0,
				getContext() {
					return {
						clearRect() {},
						drawImage(nextBitmap, sourceX) {
							bitmap = nextBitmap;
							frame = sourceX / width;
						},
						getImageData() {
							return { data: atlasBytes.get(bitmap.url)[frame] };
						},
					};
				},
			};
		},
	};
	Object.defineProperty(globalThis, "crossOriginIsolated", {
		value: false,
		configurable: true,
	});
	return () => {
		globalThis.fetch = original.fetch;
		globalThis.createImageBitmap = original.createImageBitmap;
		globalThis.document = original.document;
		if (original.crossOriginIsolated === undefined) {
			delete globalThis.crossOriginIsolated;
		} else {
			Object.defineProperty(globalThis, "crossOriginIsolated", {
				value: original.crossOriginIsolated,
				configurable: true,
			});
		}
	};
}

test("calibrated bundles reject the legacy LLFF camera-axis convention", () => {
	const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
	assert.throws(() => validateCalibratedMulticamBundle({
		version: "dynaworld_browser_multicam_dataset/v1",
		decode_size: [96, 72],
		seed_coordinate_frame: "cam00_opencv",
		dataset_contract: {
			anchor_camera: "cam00",
			pose_source: "neural_3d_llff_relative_pinhole",
		},
		cameras: [
			{ name: "cam00", role: "train", intrinsics: [1, 1, 0.5, 0.5], world_to_camera: identity },
			{ name: "cam01", role: "heldout", intrinsics: [1, 1, 0.5, 0.5], world_to_camera: identity },
		],
	}), /expected neural_3d_llff_opencv_relative_pinhole_v2/);
});

test("calibrated atlases decode sequentially into compact or explicit FP32 final banks", {
	concurrency: false,
}, async () => {
	const restore = installAtlasMocks();
	try {
		const compact = await loadCalibratedMulticamDataset({ computeSamples: false });
		assert.equal(compact.frameBank.format, FRAME_BANK_FORMAT_RGBA8);
		assert.ok(compact.frames instanceof Uint8Array);
		assert.deepEqual([...compact.frames.subarray(0, 8)], [1, 2, 3, 127, 4, 5, 6, 127]);
		assert.equal(compact.frames, compact.frameBank.data);
		assert.equal(compact.viewDatasets[1].frameBank.data.buffer, compact.frames.buffer);
		assert.equal(compact.backgroundBank.format, FRAME_BANK_FORMAT_RGBA32_FLOAT);

		const floats = await loadCalibratedMulticamDataset({
			computeSamples: false,
			frameBankFormat: FRAME_BANK_FORMAT_RGBA32_FLOAT,
		});
		assert.ok(floats.frames instanceof Float32Array);
		assert.deepEqual(decodeFrameRgb(compact, 1, 1), decodeFrameRgb(floats, 1, 1));
		assert.deepEqual(compact.backgrounds, floats.backgrounds);
		assert.equal(resolveFrameBank(floats).format, FRAME_BANK_FORMAT_RGBA32_FLOAT);
		assert.equal(compact.frames.byteLength * 4, floats.frames.byteLength);
	} finally {
		restore();
	}
});
