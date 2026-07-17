const DEFAULT_VIDEO_URL =
	"/data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/previews/neural3d_coffee_martini_cam00_to_cam10.mp4";

function hash01(seed) {
	let value = seed >>> 0;
	value ^= value << 13;
	value ^= value >>> 17;
	value ^= value << 5;
	return ((value >>> 0) % 100000) / 100000;
}

function addGaussian(frame, width, height, cx, cy, radius, color, gain = 1) {
	const r2 = radius * radius;
	const minX = Math.max(0, Math.floor((cx - radius * 3) * width));
	const maxX = Math.min(width - 1, Math.ceil((cx + radius * 3) * width));
	const minY = Math.max(0, Math.floor((cy - radius * 3) * height));
	const maxY = Math.min(height - 1, Math.ceil((cy + radius * 3) * height));
	for (let y = minY; y <= maxY; y += 1) {
		const py = (y + 0.5) / height;
		for (let x = minX; x <= maxX; x += 1) {
			const px = (x + 0.5) / width;
			const dx = px - cx;
			const dy = py - cy;
			const weight = Math.exp(-0.5 * (dx * dx + dy * dy) / r2) * gain;
			const idx = (y * width + x) * 4;
			frame[idx] += color[0] * weight;
			frame[idx + 1] += color[1] * weight;
			frame[idx + 2] += color[2] * weight;
			frame[idx + 3] = 1;
		}
	}
}

function clampFrames(frames) {
	for (let i = 0; i < frames.length; i += 4) {
		frames[i] = Math.min(1, Math.max(0, frames[i]));
		frames[i + 1] = Math.min(1, Math.max(0, frames[i + 1]));
		frames[i + 2] = Math.min(1, Math.max(0, frames[i + 2]));
		frames[i + 3] = 1;
	}
}

function computeMeanBackground(frames, width, height, frameCount) {
	const background = new Float32Array(width * height * 4);
	const invFrames = 1 / Math.max(1, frameCount);
	for (let f = 0; f < frameCount; f += 1) {
		const frameOffset = f * width * height * 4;
		for (let i = 0; i < width * height; i += 1) {
			background[i * 4] += frames[frameOffset + i * 4] * invFrames;
			background[i * 4 + 1] += frames[frameOffset + i * 4 + 1] * invFrames;
			background[i * 4 + 2] += frames[frameOffset + i * 4 + 2] * invFrames;
		}
	}
	for (let i = 0; i < width * height; i += 1) {
		background[i * 4 + 3] = 1;
	}
	return background;
}

function computeMotionSamples(frames, background, width, height, frameCount, maxSamples = 16384) {
	const scored = [];
	const pixelsPerFrame = width * height;
	for (let f = 0; f < frameCount; f += 1) {
		const frameOffset = f * pixelsPerFrame * 4;
		for (let pixel = 0; pixel < pixelsPerFrame; pixel += 1) {
			const base = frameOffset + pixel * 4;
			const bgBase = pixel * 4;
			const dr = frames[base] - background[bgBase];
			const dg = frames[base + 1] - background[bgBase + 1];
			const db = frames[base + 2] - background[bgBase + 2];
			const energy = (dr * dr + dg * dg + db * db) / 3;
			if (energy > 0.0006) {
				scored.push({ packed: f * pixelsPerFrame + pixel, energy });
			}
		}
	}
	scored.sort((a, b) => b.energy - a.energy);
	const kept = scored.slice(0, maxSamples);
	if (kept.length === 0) {
		return new Uint32Array(0);
	}
	return new Uint32Array(kept.map((item) => item.packed));
}

function computeStaticSamples(frames, background, width, height, frameCount, maxSamples = 16384) {
	const samples = [];
	const pixelsPerFrame = width * height;
	for (let f = 0; f < frameCount; f += 1) {
		const frameOffset = f * pixelsPerFrame * 4;
		for (let pixel = 0; pixel < pixelsPerFrame; pixel += 1) {
			const base = frameOffset + pixel * 4;
			const bgBase = pixel * 4;
			const dr = frames[base] - background[bgBase];
			const dg = frames[base + 1] - background[bgBase + 1];
			const db = frames[base + 2] - background[bgBase + 2];
			const energy = (dr * dr + dg * dg + db * db) / 3;
			if (energy < 0.00045) {
				samples.push(f * pixelsPerFrame + pixel);
			}
		}
	}
	if (samples.length <= maxSamples) {
		return new Uint32Array(samples);
	}
	const thinned = new Uint32Array(maxSamples);
	for (let i = 0; i < maxSamples; i += 1) {
		thinned[i] = samples[Math.floor((i + 0.5) * samples.length / maxSamples)];
	}
	return thinned;
}

export function createProceduralDnerfMini({
	width = 96,
	height = 96,
	frameCount = 8,
} = {}) {
	const frames = new Float32Array(width * height * frameCount * 4);
	const palette = [
		[0.95, 0.31, 0.23],
		[0.20, 0.73, 0.92],
		[0.95, 0.77, 0.28],
		[0.48, 0.86, 0.54],
	];

	for (let f = 0; f < frameCount; f += 1) {
		const t = frameCount === 1 ? 0 : f / (frameCount - 1);
		const frame = frames.subarray(f * width * height * 4, (f + 1) * width * height * 4);

		for (let y = 0; y < height; y += 1) {
			for (let x = 0; x < width; x += 1) {
				const px = x / Math.max(1, width - 1);
				const py = y / Math.max(1, height - 1);
				const vignette = 0.035 + 0.045 * (1 - Math.hypot(px - 0.5, py - 0.54));
				const idx = (y * width + x) * 4;
				frame[idx] = vignette * 0.55;
				frame[idx + 1] = vignette * 0.68;
				frame[idx + 2] = vignette;
				frame[idx + 3] = 1;
			}
		}

		for (let i = 0; i < 34; i += 1) {
			const seed = i + 17;
			const baseX = 0.18 + 0.64 * hash01(seed * 13);
			const baseY = 0.22 + 0.56 * hash01(seed * 31);
			const phase = hash01(seed * 47) * Math.PI * 2;
			const orbit = Math.sin(t * Math.PI * 2 + phase);
			const twist = Math.cos(t * Math.PI * 2 + phase * 0.7);
			const x = baseX + 0.035 * orbit + 0.018 * Math.sin(t * Math.PI * 4 + phase);
			const y = baseY + 0.048 * twist;
			const radius = 0.017 + 0.016 * hash01(seed * 71);
			addGaussian(frame, width, height, x, y, radius, palette[i % palette.length], 0.75);
		}

		for (let i = 0; i < 22; i += 1) {
			const angle = (i / 22) * Math.PI * 2;
			const wave = Math.sin(t * Math.PI * 2);
			const cx = 0.5 + Math.cos(angle + t * 1.3) * (0.18 + 0.035 * wave);
			const cy = 0.52 + Math.sin(angle * 1.7 - t * 2.1) * (0.16 - 0.025 * wave);
			const color = palette[(i + 1) % palette.length];
			addGaussian(frame, width, height, cx, cy, 0.019, color, 0.58);
		}
	}

	clampFrames(frames);
	const background = computeMeanBackground(frames, width, height, frameCount);
	return {
		name: "Synthetic D-NeRF mini",
		source: "procedural_blender_style",
		width,
		height,
		frameCount,
		frames,
		background,
		motionSamples: computeMotionSamples(frames, background, width, height, frameCount),
		staticSamples: computeStaticSamples(frames, background, width, height, frameCount),
	};
}

async function canFetch(url) {
	try {
		const response = await fetch(url, { method: "HEAD" });
		return response.ok;
	} catch {
		return false;
	}
}

function waitForVideoEvent(video, eventName) {
	return new Promise((resolve, reject) => {
		const cleanup = () => {
			video.removeEventListener(eventName, onEvent);
			video.removeEventListener("error", onError);
		};
		const onEvent = () => {
			cleanup();
			resolve();
		};
		const onError = () => {
			cleanup();
			reject(new Error("Video decode failed."));
		};
		video.addEventListener(eventName, onEvent, { once: true });
		video.addEventListener("error", onError, { once: true });
	});
}

async function seekVideo(video, time) {
	const promise = waitForVideoEvent(video, "seeked");
	video.currentTime = time;
	await promise;
}

function resolveVideoCrop(videoWidth, videoHeight, cropMode) {
	if (cropMode === "full" || videoWidth / Math.max(1, videoHeight) < 1.75) {
		return { x: 0, y: 0, width: videoWidth, height: videoHeight, label: "" };
	}
	const paneWidth = Math.floor(videoWidth / 2);
	if (cropMode === "target_view") {
		return {
			x: videoWidth - paneWidth,
			y: 0,
			width: paneWidth,
			height: videoHeight,
			label: "target-view crop",
		};
	}
	return {
		x: 0,
		y: 0,
		width: paneWidth,
		height: videoHeight,
		label: "source-view crop",
	};
}

export async function loadVideoFrameDataset({
	url = DEFAULT_VIDEO_URL,
	width = null,
	height = null,
	maxLongEdge = 128,
	frameCount = 8,
	cropMode = "source_view",
	name = "Neural3D coffee_martini preview",
} = {}) {
	if (!(await canFetch(url))) {
		throw new Error(`Dataset video unavailable: ${url}`);
	}

	const video = document.createElement("video");
	video.muted = true;
	video.playsInline = true;
	video.preload = "auto";
	video.crossOrigin = "anonymous";
	video.src = url;
	await waitForVideoEvent(video, "loadedmetadata");

	const videoWidth = Math.max(1, video.videoWidth || 1);
	const videoHeight = Math.max(1, video.videoHeight || 1);
	const crop = resolveVideoCrop(videoWidth, videoHeight, cropMode);
	if (width == null || height == null) {
		const aspect = crop.width / Math.max(1, crop.height);
		if (aspect >= 1) {
			width = maxLongEdge;
			height = Math.max(1, Math.round(maxLongEdge / aspect));
		} else {
			height = maxLongEdge;
			width = Math.max(1, Math.round(maxLongEdge * aspect));
		}
	}

	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d", { willReadFrequently: true });
	if (!ctx) {
		throw new Error("2D canvas unavailable for video frame preload.");
	}

	const frames = new Float32Array(width * height * frameCount * 4);
	const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : frameCount / 4;
	for (let f = 0; f < frameCount; f += 1) {
		const t = frameCount === 1 ? 0 : f / (frameCount - 1);
		await seekVideo(video, Math.min(duration * 0.92, duration * (0.04 + t * 0.88)));
		ctx.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, width, height);
		const image = ctx.getImageData(0, 0, width, height).data;
		const offset = f * width * height * 4;
		for (let i = 0; i < width * height; i += 1) {
			frames[offset + i * 4] = image[i * 4] / 255;
			frames[offset + i * 4 + 1] = image[i * 4 + 1] / 255;
			frames[offset + i * 4 + 2] = image[i * 4 + 2] / 255;
			frames[offset + i * 4 + 3] = 1;
		}
	}

	const background = computeMeanBackground(frames, width, height, frameCount);
	return {
		name: crop.label ? `${name} (${crop.label})` : name,
		source: url,
		cropMode,
		width,
		height,
		frameCount,
		frames,
		background,
		motionSamples: computeMotionSamples(frames, background, width, height, frameCount),
		staticSamples: computeStaticSamples(frames, background, width, height, frameCount),
	};
}

function makePreviewView(dataset, label) {
	return {
		label,
		width: dataset.width,
		height: dataset.height,
		frameCount: dataset.frameCount,
		frames: dataset.frames,
		background: dataset.background,
	};
}

export async function loadPresetDataset() {
	try {
		const source = await loadVideoFrameDataset();
		try {
			const target = await loadVideoFrameDataset({ cropMode: "target_view" });
			source.previewViews = [
				makePreviewView(source, "Source"),
				makePreviewView(target, "Target"),
			];
		} catch (error) {
			console.info("Target preview crop unavailable.", error);
			source.previewViews = [makePreviewView(source, "Source")];
		}
		return source;
	} catch (error) {
		console.info("Falling back to procedural dataset.", error);
		const procedural = createProceduralDnerfMini();
		procedural.previewViews = [makePreviewView(procedural, "Synthetic")];
		return procedural;
	}
}

export function drawTargetFrame(canvas, dataset, time01, { view = "rgb" } = {}) {
	const frame = Math.min(
		dataset.frameCount - 1,
		Math.max(0, Math.round(time01 * (dataset.frameCount - 1))),
	);
	const ctx = canvas.getContext("2d");
	if (!ctx) {
		return frame;
	}
	const image = ctx.createImageData(dataset.width, dataset.height);
	const offset = frame * dataset.width * dataset.height * 4;
	for (let i = 0; i < dataset.width * dataset.height; i += 1) {
		let r = dataset.frames[offset + i * 4];
		let g = dataset.frames[offset + i * 4 + 1];
		let b = dataset.frames[offset + i * 4 + 2];
		if (view === "motion_residual") {
			const dr = Math.abs(r - dataset.background[i * 4]);
			const dg = Math.abs(g - dataset.background[i * 4 + 1]);
			const db = Math.abs(b - dataset.background[i * 4 + 2]);
			const energy = Math.sqrt((dr * dr + dg * dg + db * db) / 3);
			const boost = Math.min(1, energy * 9);
			r = Math.min(1, dr * 7 + boost * 0.35);
			g = Math.min(1, dg * 7 + boost * 0.78);
			b = Math.min(1, db * 7 + boost);
		}
		image.data[i * 4] = Math.round(r * 255);
		image.data[i * 4 + 1] = Math.round(g * 255);
		image.data[i * 4 + 2] = Math.round(b * 255);
		image.data[i * 4 + 3] = 255;
	}
	canvas.width = dataset.width;
	canvas.height = dataset.height;
	ctx.putImageData(image, 0, 0);
	return frame;
}
