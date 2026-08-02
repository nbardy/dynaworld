import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { ensureStaticHostIsolation } from "../crossOriginIsolation.js";

test("isolation bootstrap is inert when headers already isolated the page", async () => {
	assert.equal(await ensureStaticHostIsolation({ crossOriginIsolated: true }), false);
});

test("first static-host visit registers isolation and reloads once", async () => {
	let registered = null;
	let reloads = 0;
	const scope = {
		crossOriginIsolated: false,
		navigator: { serviceWorker: {
			controller: null,
			async register(url, options) {
				registered = { url, options };
				return { active: { state: "activated" } };
			},
		} },
		location: { reload() { reloads += 1; } },
	};
	assert.equal(await ensureStaticHostIsolation(scope), true);
	assert.deepEqual(registered, { url: "./coi-serviceworker.js", options: { scope: "./" } });
	assert.equal(reloads, 1);
});

test("hosted build carries service-worker isolation headers", async () => {
	const source = await readFile(new URL("../coi-serviceworker.js", import.meta.url), "utf8");
	assert.match(source, /Cross-Origin-Embedder-Policy", "require-corp"/);
	assert.match(source, /Cross-Origin-Opener-Policy", "same-origin"/);
});
