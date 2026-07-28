import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { assertStorageBufferFits, sampleGradientBufferBytes } from "../trainerWebGpu3d.js";

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

test("training WGSL keeps the 768 fast tape and uses storage above it", () => {
	assert.doesNotMatch(source, /lane\s*<\s*3u/);
	assert.match(source, /fastTapeUnderAlpha: array<vec4<f32>, 768>/);
	assert.match(source, /cfg\.splatCount <= 768u/);
	assert.match(source, /for \(var i = lid\.x; i < cfg\.splatCount; i = i \+ 256u\)/);
	assert.match(source, /sampleGradients\[s \* cfg\.splatCount \+ i\] = Splat/);
	assert.ok((source.match(/storageBarrier\(\); workgroupBarrier\(\);/g) ?? []).length >= 2);
});
