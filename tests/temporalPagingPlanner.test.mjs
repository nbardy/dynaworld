import assert from "node:assert/strict";
import test from "node:test";
import {
	planTemporalPaging,
	selectNearestResidentSlot,
} from "../temporalPagingPlanner.js";

function coffeeMartiniPlan(overrides = {}) {
	return planTemporalPaging({
		nativeFrameIndices: Array.from({ length: 300 }, (_, index) => index),
		pageSize: 16,
		fps: 30,
		durationSeconds: 10,
		width: 384,
		height: 288,
		cameraCount: 18,
		...overrides,
	});
}

test("interleaved pages cover the timeline and leave only the final page short", () => {
	const first = coffeeMartiniPlan();
	const second = coffeeMartiniPlan();
	assert.deepEqual(first.pages, second.pages);
	assert.equal(first.pageCount, 19);
	assert.deepEqual(first.pages.map((page) => page.nativeFrameIndices.length), [
		16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
		16, 16, 16, 16, 16, 16, 16, 16, 12,
	]);
	for (const page of first.pages.slice(0, -1)) {
		assert.ok(page.nativeFrameIndices[0] < 20);
		assert.ok(page.nativeFrameIndices.at(-1) > 280);
	}
	assert.equal(first.pages.at(-1).nativeFrameIndices[0], 18);
	assert.equal(first.pages.at(-1).nativeFrameIndices.at(-1), 299);
	assert.deepEqual(
		first.pages.flatMap((page) => page.nativeFrameIndices).sort((a, b) => a - b),
		Array.from({ length: 300 }, (_, index) => index),
	);
});

test("times use native fps and normalize over observed frame centers", () => {
	const plan = coffeeMartiniPlan();
	const finalPage = plan.pages.at(-1);
	assert.equal(plan.lastFrameTimeSeconds, 299 / 30);
	assert.equal(finalPage.timeSeconds[0], 18 / 30);
	assert.ok(Math.abs(finalPage.normalizedTimes[0] - 18 / 299) < 1e-15);
	assert.equal(finalPage.timeSeconds.at(-1), 299 / 30);
	assert.equal(finalPage.normalizedTimes.at(-1), 1);
});

test("sparse native indices retain actual source timestamps", () => {
	const plan = coffeeMartiniPlan({
		nativeFrameIndices: [0, 20, 40],
		pageSize: 2,
		fps: 20,
		durationSeconds: 3,
	});
	assert.equal(plan.frameCount, 3);
	assert.deepEqual(
		plan.pages.flatMap((page) => page.nativeFrameIndices).sort((a, b) => a - b),
		[0, 20, 40],
	);
	assert.ok(plan.pages.some((page) => page.timeSeconds.includes(2)));
	assert.ok(plan.pages.some((page) => page.normalizedTimes.includes(1)));
});

test("memory separates the full corpus, resident RGBA8 pages, and FP32 backgrounds", () => {
	const { memory, pages } = coffeeMartiniPlan();
	assert.deepEqual(memory, {
		bytesPerRgba8Frame: 384 * 288 * 4,
		corpusRgba8Bytes: 384 * 288 * 4 * 18 * 300,
		rgba8PageBytes: 384 * 288 * 4 * 18 * 16,
		residentPageSlots: 2,
		rgba8DoubleBufferBytes: 384 * 288 * 4 * 18 * 16 * 2,
		backgroundBytes: 384 * 288 * 4 * 4 * 18,
		totalResidentBytes: 384 * 288 * 4 * 18 * 16 * 2 + 384 * 288 * 4 * 4 * 18,
	});
	assert.equal(pages.at(-1).rgba8Bytes, 384 * 288 * 4 * 18 * 12);
});

test("nearest preview selection uses resident timestamps and resolves ties earlier", () => {
	const page = {
		normalizedTimes: [0.05, 0.25, 0.8],
	};
	assert.equal(selectNearestResidentSlot(page, 0), 0);
	assert.equal(selectNearestResidentSlot(page, 1), 2);
	assert.equal(selectNearestResidentSlot(page, 0.24), 1);
	assert.equal(selectNearestResidentSlot(page, 0.525), 1);
});

test("a clip smaller than one page remains one timeline-ordered page", () => {
	const plan = coffeeMartiniPlan({
		nativeFrameIndices: [0, 1, 2],
		pageSize: 16,
		fps: 2,
		durationSeconds: 1,
	});
	assert.equal(plan.pageCount, 1);
	assert.deepEqual(plan.pages[0].nativeFrameIndices, [0, 1, 2]);
	assert.deepEqual(plan.pages[0].normalizedTimes, [0, 0.5, 1]);
	assert.equal(plan.memory.rgba8PageBytes, 384 * 288 * 4 * 18 * 3);
});

test("malformed paging and preview contracts fail before allocation", () => {
	assert.throws(() => planTemporalPaging(null), /must be an object/);
	for (const [key, value] of [
		["pageSize", 1.5],
		["fps", Number.NaN],
		["durationSeconds", 0],
		["width", -1],
		["height", 0],
		["cameraCount", 2 ** 53],
	]) {
		assert.throws(() => coffeeMartiniPlan({ [key]: value }), new RegExp(key));
	}
	assert.throws(() => coffeeMartiniPlan({ nativeFrameIndices: [] }), /must not be empty/);
	assert.throws(
		() => coffeeMartiniPlan({ nativeFrameIndices: [0, 2, 2] }),
		/strictly increasing/,
	);
	assert.throws(
		() => coffeeMartiniPlan({ fps: 30, durationSeconds: 9 }),
		/final native frame/,
	);
	assert.throws(
		() => coffeeMartiniPlan({ width: 2 ** 52, height: 2 }),
		/safe integer range/,
	);
	assert.throws(() => selectNearestResidentSlot(null, 0.5), /must be an object/);
	assert.throws(() => selectNearestResidentSlot({ normalizedTimes: [] }, 0.5), /non-empty/);
	assert.throws(
		() => selectNearestResidentSlot({ normalizedTimes: [0.2, 0.2] }, 0.2),
		/unique, increasing/,
	);
	assert.throws(
		() => selectNearestResidentSlot({ normalizedTimes: [0.2] }, 1.1),
		/within \[0, 1\]/,
	);
});
