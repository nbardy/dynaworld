import assert from "node:assert/strict";
import test from "node:test";

import { DynamicSplatWebGpu3dTrainer } from "../trainerWebGpu3d.js";


test("trainer disposal tolerates optional null GPU buffers", () => {
	let destroyed = 0;
	let unconfigured = 0;
	const trainer = {
		buffers: {
			required: { destroy: () => { destroyed += 1; } },
			optional: null,
			pingPong: [{ destroy: () => { destroyed += 1; } }, null],
		},
		context: { unconfigure: () => { unconfigured += 1; } },
	};

	DynamicSplatWebGpu3dTrainer.prototype.dispose.call(trainer);

	assert.equal(destroyed, 2);
	assert.equal(unconfigured, 1);
	assert.equal(trainer.buffers, null);
});
