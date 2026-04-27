import { buildFrameInterleaved, loadBundleFromBaseUrl } from "./decoder.js";
import {
	assertWebGpuAvailable,
	createStaticGaussianWebGpuRenderer,
} from "./staticGaussianWebGpu.js";

const statusEl = document.getElementById("status");
const canvas = document.getElementById("renderCanvas");

function setStatus(message) {
	statusEl.textContent = message;
	console.log(`[headless] ${message}`);
}

function clamp(value, minValue, maxValue) {
	return Math.min(maxValue, Math.max(minValue, value));
}

function vec3Normalize(value) {
	const length = Math.hypot(value[0], value[1], value[2]) || 1;
	return [value[0] / length, value[1] / length, value[2] / length];
}

function vec3Cross(lhs, rhs) {
	return [
		lhs[1] * rhs[2] - lhs[2] * rhs[1],
		lhs[2] * rhs[0] - lhs[0] * rhs[2],
		lhs[0] * rhs[1] - lhs[1] * rhs[0],
	];
}

function vec3Sub(lhs, rhs) {
	return [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]];
}

function vec3Dot(lhs, rhs) {
	return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

function autoCameraFromBundle(bundle, width, height) {
	const bounds = bundle.manifest.bounds;
	const target = [...bounds.center];
	const dx = bounds.max[0] - bounds.min[0];
	const dy = bounds.max[1] - bounds.min[1];
	const dz = bounds.max[2] - bounds.min[2];
	const diagonal = Math.hypot(dx, dy, dz);
	const distance = Math.max(1.5, diagonal * 1.5);
	const yaw = 0.5;
	const pitch = 0.3;
	const cosPitch = Math.cos(pitch);
	const eye = [
		target[0] + distance * Math.sin(yaw) * cosPitch,
		target[1] + distance * Math.sin(pitch),
		target[2] + distance * Math.cos(yaw) * cosPitch,
	];
	const forwardThree = vec3Normalize(vec3Sub(target, eye));
	const backwardThree = [-forwardThree[0], -forwardThree[1], -forwardThree[2]];
	const upWorld = [0, 1, 0];
	const rightThree = vec3Normalize(vec3Cross(backwardThree, upWorld));
	const upThree = vec3Normalize(vec3Cross(rightThree, backwardThree));

	const threeViewRowMajor = [
		rightThree[0],
		rightThree[1],
		rightThree[2],
		-vec3Dot(rightThree, eye),
		upThree[0],
		upThree[1],
		upThree[2],
		-vec3Dot(upThree, eye),
		backwardThree[0],
		backwardThree[1],
		backwardThree[2],
		-vec3Dot(backwardThree, eye),
		0,
		0,
		0,
		1,
	];

	const particleW2cRowMajor = [
		threeViewRowMajor[0],
		threeViewRowMajor[1],
		threeViewRowMajor[2],
		threeViewRowMajor[3],
		-threeViewRowMajor[4],
		-threeViewRowMajor[5],
		-threeViewRowMajor[6],
		-threeViewRowMajor[7],
		-threeViewRowMajor[8],
		-threeViewRowMajor[9],
		-threeViewRowMajor[10],
		-threeViewRowMajor[11],
		0,
		0,
		0,
		1,
	];

	const fovDegrees = bundle.manifest.viewer_defaults.fov_degrees;
	const fy = height / (2 * Math.tan((fovDegrees * Math.PI) / 360));
	return {
		width,
		height,
		fx: fy,
		fy,
		cx: width * 0.5,
		cy: height * 0.5,
		near: bundle.manifest.viewer_defaults.near,
		far: Math.max(bundle.manifest.viewer_defaults.far, diagonal * 6),
		w2c: particleW2cRowMajor,
		rowMajor: true,
	};
}

async function loadCamera(cameraUrl, bundle, width, height) {
	if (!cameraUrl) {
		return autoCameraFromBundle(bundle, width, height);
	}
	const response = await fetch(cameraUrl);
	if (!response.ok) {
		throw new Error(`Failed to fetch camera JSON: HTTP ${response.status}`);
	}
	const camera = await response.json();
	camera.width = width;
	camera.height = height;
	camera.cx = width * 0.5;
	camera.cy = height * 0.5;
	return camera;
}

async function main() {
	try {
		const params = new URLSearchParams(window.location.search);
		const bundleBase = params.get("bundleBase");
		const timeValue = clamp(Number(params.get("time") ?? "0"), 0, 1);
		const width = Math.max(1, Number(params.get("width") ?? "1280"));
		const height = Math.max(1, Number(params.get("height") ?? "720"));
		const cameraUrl = params.get("cameraUrl");
		if (!bundleBase) {
			throw new Error("Missing bundleBase query parameter.");
		}

		canvas.width = width;
		canvas.height = height;
		setStatus("Loading bundle...");
		const bundle = await loadBundleFromBaseUrl(bundleBase);
		setStatus("Loading WebGPU...");
		await assertWebGpuAvailable();
		const renderer = createStaticGaussianWebGpuRenderer(canvas, setStatus);
		const frame = {
			interleaved: buildFrameInterleaved(bundle, timeValue),
			count: bundle.totalCount,
		};
		const camera = await loadCamera(cameraUrl, bundle, width, height);
		await renderer.loadFrameData(frame);
		const renderedCount = await renderer.renderFrameFromCamera(camera);
		await new Promise((resolve) =>
			requestAnimationFrame(() => requestAnimationFrame(resolve)),
		);
		window.__headlessRender = {
			ready: true,
			count: renderedCount,
			time: timeValue,
			width,
			height,
			bundleBase,
		};
		setStatus(`Rendered ${renderedCount} splats.`);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		window.__headlessRender = { ready: false, error: message };
		setStatus(message);
		console.error(error);
	}
}

void main();
