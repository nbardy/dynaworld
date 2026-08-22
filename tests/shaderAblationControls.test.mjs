import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const app = readFileSync(new URL("../app.js", import.meta.url), "utf8");
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

const controls = [
	["pixelFilterSelect", "pixelFilter"],
	["opacityModelSelect", "opacityModel"],
	["geometryColorWeightInput", "geometryColorWeight"],
	["crossViewDepthToggle", "crossViewDepth"],
	["geometryConsistencyEverySelect", "geometryConsistencyEvery"],
	["geometryDepthWeightInput", "geometryDepthWeight"],
];

test("shader ablation controls preserve the exact baseline defaults", () => {
	assert.match(html, /<option value="legacy-floor" selected>/);
	assert.match(html, /<option value="mip-2d-compensated">/);
	assert.match(html, /<option value="coupled" selected>/);
	assert.match(html, /<option value="dual">/);
	assert.match(html, /id="geometryColorWeightInput"[^>]*value="0"/);
	assert.match(html, /id="crossViewDepthToggle"[^>]*type="checkbox"(?![^>]*checked)/);
	assert.match(html, /<option value="8" selected>Every 8 steps<\/option>/);
	assert.match(html, /id="geometryDepthWeightInput"[^>]*value="0\.05"/);
	assert.match(html, /StableGS-inspired/);
	assert.doesNotMatch(html, /full StableGS parity[^<]*[.!]?\s*<\/span>/i);
});

test("all shader controls use generic persistence and reset-time wiring", () => {
	for (const [id, name] of controls) {
		assert.ok(app.includes(`${name}: $("${id}")`), `${name} is registered`);
	}
	assert.match(app, /Object\.entries\(controls\)/);
	assert.match(app, /control\.addEventListener\("change", \(\) => \{ updateControlLabels\(\); void resetTrainer\(\); \}\)/);
});

test("worker initialization receives the exact typed trainer option contract", () => {
	assert.match(app, /pixelFilterMode: controls\.pixelFilter\.value/);
	assert.match(app, /opacityModel: controls\.opacityModel\.value/);
	assert.match(app, /geometryColorWeight: Number\(controls\.geometryColorWeight\.value\)/);
	assert.match(app, /crossViewDepth: controls\.crossViewDepth\.checked/);
	assert.match(app, /geometryConsistencyEvery: Number\.parseInt\(controls\.geometryConsistencyEvery\.value, 10\)/);
	assert.match(app, /geometryDepthWeight: Number\(controls\.geometryDepthWeight\.value\)/);
});

test("shader controls are fast-tiled-only and represented without a parity claim", () => {
	assert.match(app, /function fastTiledBackendSelected\(\)/);
	assert.match(app, /const shaderAblationsDisabled = !fastTiledBackendSelected\(\)/);
	for (const [, name] of controls) {
		assert.match(app, new RegExp(`controls\\.${name}`));
	}
	assert.match(app, /StableGS-inspired:/);
	assert.match(app, /baseline shader/);
	assert.match(app, /shaderAblationDescription\(\)/);
	assert.doesNotMatch(app, /full StableGS parity/i);
});
