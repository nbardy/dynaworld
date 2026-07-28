import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
	assertStorageBufferFits,
	rgbaFloatFrameBytes,
	sampleGradientBufferBytes,
} from "../trainerWebGpu3d.js";

const source = readFileSync(new URL("../trainerWebGpu3d.js", import.meta.url), "utf8");

test("gradient storage scales beyond the former 768-splat ceiling", () => {
	assert.equal(sampleGradientBufferBytes(769), 192 * 769 * 96);
	assert.equal(sampleGradientBufferBytes(4096), 192 * 4096 * 96);
	assert.throws(() => sampleGradientBufferBytes(0), /positive safe integer/);
});

test("oversized dataset bindings fail before WebGPU bind-group construction", () => {
	assert.doesNotThrow(() => assertStorageBufferFits("target", 256, 256));
	assert.throws(
		() => assertStorageBufferFits("target", 257, 256),
		/target needs a 257-byte storage buffer.*supports 256 bytes.*Stream or page/s,
	);
});

test("one RGBA32F target page scales with one frame instead of the dataset", () => {
	const dataset = {
		width: 384,
		height: 288,
		frames: { byteLength: 384 * 288 * 4 * 18 * 16 * Float32Array.BYTES_PER_ELEMENT },
	};
	assert.equal(rgbaFloatFrameBytes(dataset), 1_769_472);
	assert.equal(dataset.frames.byteLength, 509_607_936);
});

test("training WGSL keeps the 768 fast tape and uses storage above it", () => {
	assert.doesNotMatch(source, /lane\s*<\s*3u/);
	assert.match(source, /fastTapeUnderAlpha: array<vec4<f32>, 768>/);
	assert.match(source, /cfg\.splatCount <= 768u/);
	assert.match(source, /for \(var i = lid\.x; i < cfg\.splatCount; i = i \+ 256u\)/);
	assert.match(source, /sampleGradients\[s \* cfg\.splatCount \+ i\] = Splat/);
	assert.ok((source.match(/storageBarrier\(\); workgroupBarrier\(\);/g) ?? []).length >= 2);
});

test("sampled trainer rejects counts beyond its fixed 2048-entry order cache", () => {
	assert.match(source, /splatCount > 2048 && !this\.skipSampleGradientAllocation/);
	assert.match(source, /sampled-ray depth-order cache supports at most 2048 splats/);
});
