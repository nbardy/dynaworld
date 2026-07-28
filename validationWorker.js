import { computeSnapshotValidationMetrics } from "./trainerWebGpu3d.js";
import { WORKER_PROTOCOL_VERSION } from "./workerProtocol.js";

let dataset = null;
let initialParams = null;

self.onmessage = ({ data }) => {
	if (data?.version !== WORKER_PROTOCOL_VERSION) return;
	if (data.type === "init") {
		dataset = data.dataset;
		initialParams = data.initialParams;
		self.postMessage({ version: WORKER_PROTOCOL_VERSION, type: "ready" });
		return;
	}
	if (data.type !== "validate" || !dataset) return;
	try {
		const params = new Float32Array(data.params);
		const metrics = computeSnapshotValidationMetrics(dataset, params, data.options);
		let parameterDelta = 0;
		if (initialParams?.length === params.length) {
			for (let index = 0; index < params.length; index += 1) {
				parameterDelta += Math.abs(params[index] - initialParams[index]);
			}
			parameterDelta /= params.length;
		}
		metrics.parameterDelta = parameterDelta;
		metrics.totalRecycled = data.options?.totalRecycled ?? 0;
		self.postMessage({ version: WORKER_PROTOCOL_VERSION, type: "validation", step: data.step, metrics });
	} catch (error) {
		self.postMessage({ version: WORKER_PROTOCOL_VERSION, type: "error", step: data.step,
			message: error?.message ?? String(error), stack: error?.stack });
	}
};
