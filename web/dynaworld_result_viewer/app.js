import { buildFrameInterleaved, loadBundleFromFiles } from "./decoder.js";
import {
	assertWebGpuAvailable,
	createStaticGaussianWebGpuRenderer,
} from "./staticGaussianWebGpu.js";

const fileInput = document.getElementById("bundleInput");
const timeSlider = document.getElementById("timeSlider");
const timeLabel = document.getElementById("timeValue");
const autoplayToggle = document.getElementById("autoplayToggle");
const statusEl = document.getElementById("status");
const detailsEl = document.getElementById("details");
const noteEl = document.getElementById("notes");
const canvas = document.getElementById("renderCanvas");

let bundle = null;
let renderer = null;
let currentFrame = null;
let currentTime = Number(timeSlider.value);
let dirtyData = false;
let dirtyCamera = true;
let rendering = false;
let renderQueued = false;
let lastTimestampMs = 0;

const orbit = {
	target: [0, 0, 0],
	distance: 3,
	yaw: 0.5,
	pitch: 0.3,
	fovDegrees: 60,
	near: 0.01,
	far: 100,
};

function setStatus(message) {
	statusEl.textContent = message;
}

function resizeCanvasToDisplaySize() {
	const dpr = window.devicePixelRatio || 1;
	const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
	const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
	if (canvas.width !== width || canvas.height !== height) {
		canvas.width = width;
		canvas.height = height;
		dirtyCamera = true;
	}
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

function buildParticleCameraSpec() {
	resizeCanvasToDisplaySize();
	const cosPitch = Math.cos(orbit.pitch);
	const eye = [
		orbit.target[0] + orbit.distance * Math.sin(orbit.yaw) * cosPitch,
		orbit.target[1] + orbit.distance * Math.sin(orbit.pitch),
		orbit.target[2] + orbit.distance * Math.cos(orbit.yaw) * cosPitch,
	];
	const forwardThree = vec3Normalize(vec3Sub(orbit.target, eye));
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

	const width = Math.max(1, canvas.width);
	const height = Math.max(1, canvas.height);
	const fy = height / (2 * Math.tan((orbit.fovDegrees * Math.PI) / 360));
	const fx = fy;
	return {
		width,
		height,
		fx,
		fy,
		cx: width * 0.5,
		cy: height * 0.5,
		near: orbit.near,
		far: orbit.far,
		w2c: particleW2cRowMajor,
		rowMajor: true,
	};
}

function requestRender() {
	renderQueued = true;
	if (!rendering) {
		void renderLoop();
	}
}

async function drawFrame() {
	if (!bundle || !renderer) {
		return;
	}
	if (dirtyData || !currentFrame) {
		currentFrame = {
			interleaved: buildFrameInterleaved(bundle, currentTime),
			count: bundle.totalCount,
		};
		await renderer.loadFrameData(currentFrame);
		dirtyData = false;
	}
	if (dirtyCamera || currentFrame) {
		await renderer.renderFrameFromCamera(buildParticleCameraSpec());
		dirtyCamera = false;
	}
}

async function renderLoop(timestampMs = 0) {
	rendering = true;
	try {
		while (renderQueued || autoplayToggle.checked) {
			const shouldAdvance = autoplayToggle.checked && bundle;
			renderQueued = false;
			if (shouldAdvance) {
				const deltaSeconds = lastTimestampMs
					? Math.min(0.05, Math.max(0, (timestampMs - lastTimestampMs) / 1000))
					: 0;
				lastTimestampMs = timestampMs;
				currentTime = (currentTime + deltaSeconds * 0.15) % 1;
				timeSlider.value = currentTime.toFixed(4);
				timeLabel.textContent = currentTime.toFixed(3);
				dirtyData = true;
			}
			await drawFrame();
			if (autoplayToggle.checked) {
				await new Promise((resolve) =>
					requestAnimationFrame((nextTimestamp) => {
						timestampMs = nextTimestamp;
						resolve();
					}),
				);
				renderQueued = true;
			}
		}
	} catch (error) {
		setStatus(error instanceof Error ? error.message : String(error));
		console.error(error);
	} finally {
		rendering = false;
	}
}

function applyBundleDefaults(loadedBundle) {
	const bounds = loadedBundle.manifest.bounds;
	orbit.target = [...bounds.center];
	const dx = bounds.max[0] - bounds.min[0];
	const dy = bounds.max[1] - bounds.min[1];
	const dz = bounds.max[2] - bounds.min[2];
	const diagonal = Math.hypot(dx, dy, dz);
	orbit.distance = Math.max(1.5, diagonal * 1.5);
	orbit.fovDegrees = loadedBundle.manifest.viewer_defaults.fov_degrees;
	orbit.near = loadedBundle.manifest.viewer_defaults.near;
	orbit.far = Math.max(
		loadedBundle.manifest.viewer_defaults.far,
		orbit.distance * 6,
		diagonal * 6,
	);
}

async function handleBundleSelection(files) {
	setStatus("Loading bundle...");
	bundle = await loadBundleFromFiles(files);
	applyBundleDefaults(bundle);
	if (renderer) {
		renderer.dispose?.();
	}
	await assertWebGpuAvailable();
	renderer = createStaticGaussianWebGpuRenderer(canvas, setStatus);
	currentFrame = null;
	dirtyData = true;
	dirtyCamera = true;
	lastTimestampMs = 0;

	const counts = bundle.manifest.counts;
	detailsEl.textContent =
		`${counts.total_gaussians.toLocaleString()} splats ` +
		`(${counts.static_gaussians.toLocaleString()} static + ` +
		`${counts.dynamic_gaussians.toLocaleString()} dynamic)`;
	noteEl.textContent = bundle.manifest.model.notes.join(" ");
	timeLabel.textContent = currentTime.toFixed(3);
	setStatus("Bundle loaded.");
	requestRender();
}

fileInput.addEventListener("change", async (event) => {
	const { files } = event.target;
	if (!files || files.length === 0) {
		return;
	}
	try {
		await handleBundleSelection(files);
	} catch (error) {
		setStatus(error instanceof Error ? error.message : String(error));
		console.error(error);
	}
});

timeSlider.addEventListener("input", () => {
	currentTime = Number(timeSlider.value);
	timeLabel.textContent = currentTime.toFixed(3);
	dirtyData = true;
	requestRender();
});

autoplayToggle.addEventListener("change", () => {
	lastTimestampMs = 0;
	requestRender();
});

let dragging = false;
let lastPointerX = 0;
let lastPointerY = 0;

canvas.addEventListener("pointerdown", (event) => {
	dragging = true;
	lastPointerX = event.clientX;
	lastPointerY = event.clientY;
	canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
	if (!dragging) {
		return;
	}
	const dx = event.clientX - lastPointerX;
	const dy = event.clientY - lastPointerY;
	lastPointerX = event.clientX;
	lastPointerY = event.clientY;
	orbit.yaw -= dx * 0.01;
	orbit.pitch = clamp(orbit.pitch - dy * 0.01, -1.45, 1.45);
	dirtyCamera = true;
	requestRender();
});

canvas.addEventListener("pointerup", (event) => {
	dragging = false;
	canvas.releasePointerCapture(event.pointerId);
});

canvas.addEventListener("pointerleave", () => {
	dragging = false;
});

canvas.addEventListener(
	"wheel",
	(event) => {
		event.preventDefault();
		const scale = Math.exp(event.deltaY * 0.001);
		orbit.distance = clamp(orbit.distance * scale, 0.2, 500);
		dirtyCamera = true;
		requestRender();
	},
	{ passive: false },
);

new ResizeObserver(() => {
	dirtyCamera = true;
	requestRender();
}).observe(canvas);

setStatus("Select an exported Dynaworld bundle directory.");
