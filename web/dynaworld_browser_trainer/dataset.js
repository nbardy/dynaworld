const DEFAULT_VIDEO_URL =
	"/data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/previews/neural3d_coffee_martini_cam00_to_cam10.mp4";
const CALIBRATED_MULTICAM_URL = "./coffee_martini_train17_holdout1.json";
export const CALIBRATED_MULTICAM_POSE_SOURCE = "neural_3d_llff_opencv_relative_pinhole_v2";

export const FRAME_BANK_FORMAT_RGBA8 = "rgba8unorm-rgb+weight-u8x127/v1";
export const FRAME_BANK_FORMAT_RGBA32_FLOAT = "rgba32float/v1";
export const BACKGROUND_BANK_FORMAT_RGBA32_FLOAT = "rgba32float/v1";
export const FRAME_WEIGHT_BYTE_SCALE = 127;

function inferFrameBankFormat(data) {
	if (data instanceof Uint8Array) return FRAME_BANK_FORMAT_RGBA8;
	if (data instanceof Float32Array) return FRAME_BANK_FORMAT_RGBA32_FLOAT;
	throw new TypeError("Frame-bank data must be a Uint8Array or Float32Array.");
}

export function resolveFrameBank(source) {
	const data = source?.frameBank?.data ?? source?.frames ?? source?.data ?? source;
	const format = source?.frameBank?.format ?? source?.format ?? inferFrameBankFormat(data);
	if (!(data instanceof Uint8Array) && !(data instanceof Float32Array)) {
		throw new TypeError("Frame-bank data must be a Uint8Array or Float32Array.");
	}
	if (format === FRAME_BANK_FORMAT_RGBA8 && !(data instanceof Uint8Array)) {
		throw new TypeError(`${FRAME_BANK_FORMAT_RGBA8} requires Uint8Array data.`);
	}
	if (format === FRAME_BANK_FORMAT_RGBA32_FLOAT && !(data instanceof Float32Array)) {
		throw new TypeError(`${FRAME_BANK_FORMAT_RGBA32_FLOAT} requires Float32Array data.`);
	}
	if (format !== FRAME_BANK_FORMAT_RGBA8 && format !== FRAME_BANK_FORMAT_RGBA32_FLOAT) {
		throw new RangeError(`Unsupported frame-bank format: ${format}`);
	}
	return { format, data };
}

export function readFrameBankValue(source, index) {
	const bank = resolveFrameBank(source);
	if (!Number.isSafeInteger(index) || index < 0 || index >= bank.data.length) {
		throw new RangeError("Frame-bank index is out of range.");
	}
	return bank.format === FRAME_BANK_FORMAT_RGBA8
		? Math.fround(bank.data[index] / 255)
		: bank.data[index];
}

export function readFrameLossWeight(source, pixelBase) {
	const bank = resolveFrameBank(source);
	const index = pixelBase + 3;
	if (!Number.isSafeInteger(pixelBase) || pixelBase < 0 || index >= bank.data.length) {
		throw new RangeError("Frame-bank pixel offset is out of range.");
	}
	return bank.format === FRAME_BANK_FORMAT_RGBA8
		? bank.data[index] / FRAME_WEIGHT_BYTE_SCALE
		: bank.data[index];
}

function writeFrameLossWeight(bank, pixelBase, weight) {
	if (bank.format === FRAME_BANK_FORMAT_RGBA8) {
		const clamped = Math.min(2, Math.max(0, Number(weight)));
		bank.data[pixelBase + 3] = Math.round(clamped * FRAME_WEIGHT_BYTE_SCALE);
	} else {
		bank.data[pixelBase + 3] = Number(weight);
	}
}

export function writeNormalizedFrameLossWeights(source, frameBase, weights) {
	const bank = resolveFrameBank(source);
	if (!Number.isSafeInteger(frameBase) || frameBase < 0 || frameBase % 4 !== 0
		|| frameBase + weights.length * 4 > bank.data.length) {
		throw new RangeError("Normalized frame weights require a complete in-range RGBA frame.");
	}
	for (let pixel = 0; pixel < weights.length; pixel += 1) {
		if (!Number.isFinite(Number(weights[pixel]))) {
			throw new TypeError("Normalized frame weights must be finite.");
		}
	}
	if (bank.format !== FRAME_BANK_FORMAT_RGBA8) {
		for (let pixel = 0; pixel < weights.length; pixel += 1) {
			writeFrameLossWeight(bank, frameBase + pixel * 4, weights[pixel]);
		}
		return;
	}

	// The tiled objective divides by pixel count because normalized motion
	// weights have mean one. Independent rounding breaks that invariant. Start
	// from nearest bytes, then make the minimum-error one-byte corrections that
	// restore an exact per-frame sum of 127 * pixels.
	const scaled = new Float64Array(weights.length);
	let encodedSum = 0;
	for (let pixel = 0; pixel < weights.length; pixel += 1) {
		const value = Math.min(2, Math.max(0, Number(weights[pixel])))
			* FRAME_WEIGHT_BYTE_SCALE;
		scaled[pixel] = value;
		const encoded = Math.round(value);
		bank.data[frameBase + pixel * 4 + 3] = encoded;
		encodedSum += encoded;
	}
	const targetSum = FRAME_WEIGHT_BYTE_SCALE * weights.length;
	const delta = targetSum - encodedSum;
	if (delta === 0) return;
	const direction = Math.sign(delta);
	const candidates = [];
	for (let pixel = 0; pixel < weights.length; pixel += 1) {
		const index = frameBase + pixel * 4 + 3;
		const encoded = bank.data[index];
		if ((direction > 0 && encoded === FRAME_WEIGHT_BYTE_SCALE * 2)
			|| (direction < 0 && encoded === 0)) continue;
		const currentError = Math.abs(encoded - scaled[pixel]);
		const adjustedError = Math.abs(encoded + direction - scaled[pixel]);
		candidates.push({ pixel, cost: adjustedError - currentError });
	}
	candidates.sort((left, right) => left.cost - right.cost || left.pixel - right.pixel);
	if (Math.abs(delta) > candidates.length) {
		throw new RangeError("Compact normalized motion weights cannot preserve their mean.");
	}
	for (let index = 0; index < Math.abs(delta); index += 1) {
		const pixel = candidates[index].pixel;
		bank.data[frameBase + pixel * 4 + 3] += direction;
	}
}

export function decodeFrameRgb(dataset, viewIndex, frameIndex, target = null) {
	const pixels = Number(dataset.width) * Number(dataset.height);
	const viewCount = Number(dataset.viewCount ?? 1);
	if (!Number.isSafeInteger(viewIndex) || viewIndex < 0 || viewIndex >= viewCount
		|| !Number.isSafeInteger(frameIndex) || frameIndex < 0 || frameIndex >= dataset.frameCount) {
		throw new RangeError("Requested view/frame is outside the dataset.");
	}
	const bank = resolveFrameBank(dataset);
	const sourceOffset = (viewIndex * dataset.frameCount + frameIndex) * pixels * 4;
	if (sourceOffset + pixels * 4 > bank.data.length) {
		throw new RangeError("Frame bank does not contain the requested view/frame.");
	}
	const result = target ?? new Float32Array(pixels * 3);
	if (!(result instanceof Float32Array) || result.length !== pixels * 3) {
		throw new RangeError("Decoded RGB target must be a Float32Array sized to one frame.");
	}
	for (let pixel = 0; pixel < pixels; pixel += 1) {
		const sourceBase = sourceOffset + pixel * 4;
		const targetBase = pixel * 3;
		if (bank.format === FRAME_BANK_FORMAT_RGBA8) {
			result[targetBase] = bank.data[sourceBase] / 255;
			result[targetBase + 1] = bank.data[sourceBase + 1] / 255;
			result[targetBase + 2] = bank.data[sourceBase + 2] / 255;
		} else {
			result[targetBase] = bank.data[sourceBase];
			result[targetBase + 1] = bank.data[sourceBase + 1];
			result[targetBase + 2] = bank.data[sourceBase + 2];
		}
	}
	return result;
}

function attachPixelBanks(dataset, frameData, backgrounds, frameFormat = inferFrameBankFormat(frameData)) {
	dataset.frameBank = { format: frameFormat, data: frameData };
	dataset.backgroundBank = { format: BACKGROUND_BANK_FORMAT_RGBA32_FLOAT, data: backgrounds };
	// These aliases keep existing consumers working while new code can inspect
	// the format explicitly instead of guessing from the array constructor.
	dataset.frames = dataset.frameBank.data;
	dataset.backgrounds = dataset.backgroundBank.data;
	dataset.background = backgrounds.subarray(0, dataset.width * dataset.height * 4);
	return dataset;
}

function allocateFrameBank(length, format, scope = globalThis) {
	const Constructor = format === FRAME_BANK_FORMAT_RGBA8 ? Uint8Array : Float32Array;
	if (format !== FRAME_BANK_FORMAT_RGBA8 && format !== FRAME_BANK_FORMAT_RGBA32_FLOAT) {
		throw new RangeError(`Unsupported frame-bank format: ${format}`);
	}
	const bytes = length * Constructor.BYTES_PER_ELEMENT;
	const BufferConstructor = scope?.crossOriginIsolated === true
		&& typeof scope?.SharedArrayBuffer === "function"
		? scope.SharedArrayBuffer
		: ArrayBuffer;
	return new Constructor(new BufferConstructor(bytes));
}

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

function computeMeanBackground(frames, width, height, frameCount, sourceOffset = 0) {
	const bank = resolveFrameBank(frames);
	const background = new Float32Array(width * height * 4);
	const invFrames = 1 / Math.max(1, frameCount);
	for (let f = 0; f < frameCount; f += 1) {
		const frameOffset = sourceOffset + f * width * height * 4;
		for (let i = 0; i < width * height; i += 1) {
			const base = frameOffset + i * 4;
			const target = i * 4;
			if (bank.format === FRAME_BANK_FORMAT_RGBA8) {
				background[target] += Math.fround(bank.data[base] / 255) * invFrames;
				background[target + 1] += Math.fround(bank.data[base + 1] / 255) * invFrames;
				background[target + 2] += Math.fround(bank.data[base + 2] / 255) * invFrames;
			} else {
				background[target] += bank.data[base] * invFrames;
				background[target + 1] += bank.data[base + 1] * invFrames;
				background[target + 2] += bank.data[base + 2] * invFrames;
			}
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
	return attachPixelBanks({
		name: "Synthetic D-NeRF mini",
		source: "procedural_blender_style",
		width,
		height,
		frameCount,
		motionSamples: computeMotionSamples(frames, background, width, height, frameCount),
		staticSamples: computeStaticSamples(frames, background, width, height, frameCount),
	}, frames, background);
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
	return attachPixelBanks({
		name: crop.label ? `${name} (${crop.label})` : name,
		source: url,
		cropMode,
		width,
		height,
		frameCount,
		motionSamples: computeMotionSamples(frames, background, width, height, frameCount),
		staticSamples: computeStaticSamples(frames, background, width, height, frameCount),
	}, frames, background);
}

function makePreviewView(dataset, label) {
	return {
		label,
		width: dataset.width,
		height: dataset.height,
		frameCount: dataset.frameCount,
		frames: dataset.frames,
		frameBank: dataset.frameBank,
		background: dataset.background,
		backgroundBank: dataset.backgroundBank,
	};
}

function writeDecodedImage(targetBank, targetOffset, image) {
	if (targetBank.format === FRAME_BANK_FORMAT_RGBA8) {
		for (let pixel = 0; pixel < image.length / 4; pixel += 1) {
			const sourceBase = pixel * 4;
			const targetBase = targetOffset + sourceBase;
			targetBank.data[targetBase] = image[sourceBase];
			targetBank.data[targetBase + 1] = image[sourceBase + 1];
			targetBank.data[targetBase + 2] = image[sourceBase + 2];
			targetBank.data[targetBase + 3] = FRAME_WEIGHT_BYTE_SCALE;
		}
		return;
	}
	for (let pixel = 0; pixel < image.length / 4; pixel += 1) {
		const sourceBase = pixel * 4;
		const targetBase = targetOffset + sourceBase;
		targetBank.data[targetBase] = image[sourceBase] / 255;
		targetBank.data[targetBase + 1] = image[sourceBase + 1] / 255;
		targetBank.data[targetBase + 2] = image[sourceBase + 2] / 255;
		targetBank.data[targetBase + 3] = 1;
	}
}

async function decodeVideoFramesInto({
	url,
	width,
	height,
	frameTimesSeconds,
	target,
	targetOffset = 0,
}) {
	if (!(await canFetch(url))) {
		throw new Error(`Dataset video unavailable: ${url}`);
	}
	const targetBank = resolveFrameBank(target);
	const video = document.createElement("video");
	video.muted = true;
	video.preload = "auto";
	video.src = url;
	video.load();
	await waitForVideoEvent(video, "loadedmetadata");
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d", { willReadFrequently: true });
	if (!ctx) {
		throw new Error("2D canvas unavailable for multicamera preload.");
	}
	const frameCount = frameTimesSeconds.length;
	const availableDuration = Number.isFinite(video.duration) ? video.duration : frameTimesSeconds.at(-1);
	for (let frame = 0; frame < frameCount; frame += 1) {
		const time = Math.min(availableDuration * 0.999, frameTimesSeconds[frame]);
		await seekVideo(video, time);
		ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, width, height);
		const image = ctx.getImageData(0, 0, width, height).data;
		writeDecodedImage(targetBank, targetOffset + frame * width * height * 4, image);
	}
	video.removeAttribute("src");
	video.load();
}

async function decodeFrameAtlasInto({
	url,
	width,
	height,
	frameCount,
	target,
	targetOffset = 0,
}) {
	const targetBank = resolveFrameBank(target);
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error(`Frame atlas unavailable: ${url}`);
	}
	const bitmap = await createImageBitmap(await response.blob());
	if (bitmap.width !== width * frameCount || bitmap.height !== height) {
		throw new Error(`Frame atlas ${url} has ${bitmap.width}x${bitmap.height}; expected ${width * frameCount}x${height}.`);
	}
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d", { willReadFrequently: true });
	if (!ctx) {
		bitmap.close();
		throw new Error("2D canvas unavailable for frame-atlas decode.");
	}
	for (let frame = 0; frame < frameCount; frame += 1) {
		ctx.clearRect(0, 0, width, height);
		ctx.drawImage(bitmap, frame * width, 0, width, height, 0, 0, width, height);
		const image = ctx.getImageData(0, 0, width, height).data;
		writeDecodedImage(targetBank, targetOffset + frame * width * height * 4, image);
	}
	bitmap.close();
}

export const MOTION_LOSS_WEIGHT_MAX = 2;

export function normalizedMotionLossWeights(energies, maximumWeight = MOTION_LOSS_WEIGHT_MAX) {
	if (!energies || !Number.isFinite(maximumWeight) || maximumWeight < 1) {
		throw new RangeError("Motion-loss weights require finite energies and a maximum weight of at least one.");
	}
	const weights = new Float32Array(energies.length);
	let sum = 0;
	for (let index = 0; index < energies.length; index += 1) {
		const linear = Math.min(1, Math.max(0, (Number(energies[index]) - 0.00035) / (0.004 - 0.00035)));
		const score = linear * linear * (3 - 2 * linear);
		const weight = 1 + (maximumWeight - 1) * score;
		weights[index] = weight;
		sum += weight;
	}
	const mean = sum / Math.max(1, weights.length);
	for (let index = 0; index < weights.length; index += 1) weights[index] /= mean;
	return weights;
}

function motionCandidateIsWorse(left, right) {
	return left.energy < right.energy || (left.energy === right.energy && left.packed > right.packed);
}

function siftMotionCandidateUp(heap, index) {
	while (index > 0) {
		const parent = Math.floor((index - 1) / 2);
		if (!motionCandidateIsWorse(heap[index], heap[parent])) break;
		[heap[index], heap[parent]] = [heap[parent], heap[index]];
		index = parent;
	}
}

function siftMotionCandidateDown(heap, index) {
	for (;;) {
		const left = index * 2 + 1;
		if (left >= heap.length) return;
		const right = left + 1;
		let worse = left;
		if (right < heap.length && motionCandidateIsWorse(heap[right], heap[left])) worse = right;
		if (!motionCandidateIsWorse(heap[worse], heap[index])) return;
		[heap[index], heap[worse]] = [heap[worse], heap[index]];
		index = worse;
	}
}

function keepTopMotionCandidate(heap, packed, energy, limit) {
	if (heap.length < limit) {
		heap.push({ packed, energy });
		siftMotionCandidateUp(heap, heap.length - 1);
		return;
	}
	const worst = heap[0];
	if (worst.energy > energy || (worst.energy === energy && worst.packed < packed)) return;
	heap[0] = { packed, energy };
	siftMotionCandidateDown(heap, 0);
}

function frameRgbEnergy(frameBank, backgrounds, frameBase, backgroundBase) {
	let dr;
	let dg;
	let db;
	if (frameBank.format === FRAME_BANK_FORMAT_RGBA8) {
		dr = Math.fround(frameBank.data[frameBase] / 255) - backgrounds[backgroundBase];
		dg = Math.fround(frameBank.data[frameBase + 1] / 255) - backgrounds[backgroundBase + 1];
		db = Math.fround(frameBank.data[frameBase + 2] / 255) - backgrounds[backgroundBase + 2];
	} else {
		dr = frameBank.data[frameBase] - backgrounds[backgroundBase];
		dg = frameBank.data[frameBase + 1] - backgrounds[backgroundBase + 1];
		db = frameBank.data[frameBase + 2] - backgrounds[backgroundBase + 2];
	}
	return (dr * dr + dg * dg + db * db) / 3;
}

export function computeMultiviewSamples(frames, backgrounds, width, height, frameCount, trainViewCount) {
	const frameBank = resolveFrameBank(frames);
	if (!(backgrounds instanceof Float32Array)) {
		throw new TypeError("backgrounds must be a Float32Array.");
	}
	const pixels = width * height;
	const maxSamples = 16384;
	const motion = [];
	let staticCount = 0;
	for (let view = 0; view < trainViewCount; view += 1) {
		for (let frame = 0; frame < frameCount; frame += 1) {
			const energies = new Float32Array(pixels);
			for (let pixel = 0; pixel < pixels; pixel += 1) {
				const base = ((view * frameCount + frame) * pixels + pixel) * 4;
				const bgBase = (view * pixels + pixel) * 4;
				const energy = frameRgbEnergy(frameBank, backgrounds, base, bgBase);
				energies[pixel] = energy;
				const packed = (view * frameCount + frame) * pixels + pixel;
				if (energy > 0.0006) {
					keepTopMotionCandidate(motion, packed, energy, maxSamples);
				} else if (energy < 0.00035) {
					staticCount += 1;
				}
			}
			const weights = normalizedMotionLossWeights(energies);
			const frameBase = (view * frameCount + frame) * pixels * 4;
			writeNormalizedFrameLossWeights(frameBank, frameBase, weights);
		}
	}
	motion.sort((left, right) => right.energy - left.energy || left.packed - right.packed);
	const staticKept = new Uint32Array(Math.min(maxSamples, staticCount));
	let staticOrdinal = 0;
	let keptIndex = 0;
	for (let view = 0; view < trainViewCount && keptIndex < staticKept.length; view += 1) {
		for (let frame = 0; frame < frameCount && keptIndex < staticKept.length; frame += 1) {
			for (let pixel = 0; pixel < pixels && keptIndex < staticKept.length; pixel += 1) {
				const base = ((view * frameCount + frame) * pixels + pixel) * 4;
				const bgBase = (view * pixels + pixel) * 4;
				if (frameRgbEnergy(frameBank, backgrounds, base, bgBase) >= 0.00035) continue;
				const desiredOrdinal = staticCount <= maxSamples
					? keptIndex
					: Math.floor((keptIndex + 0.5) * staticCount / maxSamples);
				if (staticOrdinal === desiredOrdinal) {
					staticKept[keptIndex] = (view * frameCount + frame) * pixels + pixel;
					keptIndex += 1;
				}
				staticOrdinal += 1;
			}
		}
	}
	return {
		motionSamples: new Uint32Array(motion.map((item) => item.packed)),
		staticSamples: staticKept,
	};
}

export function validateCalibratedMulticamBundle(bundle) {
	if (bundle?.version !== "dynaworld_browser_multicam_dataset/v1") {
		throw new Error(`Unsupported calibrated browser bundle: ${bundle?.version ?? "missing version"}`);
	}
	const contract = bundle.dataset_contract;
	if (contract?.pose_source !== CALIBRATED_MULTICAM_POSE_SOURCE) {
		throw new Error(`Calibrated browser bundle pose source ${contract?.pose_source ?? "missing"}; `
			+ `expected ${CALIBRATED_MULTICAM_POSE_SOURCE}. Refresh or rebuild the dataset bundle.`);
	}
	if (!contract.anchor_camera || bundle.seed_coordinate_frame !== `${contract.anchor_camera}_opencv`) {
		throw new Error("Calibrated browser seeds and anchor-relative cameras must share the declared OpenCV frame.");
	}
	if (!Array.isArray(bundle.decode_size) || bundle.decode_size.length !== 2
		|| !bundle.decode_size.every((value) => Number.isSafeInteger(value) && value > 0)) {
		throw new Error("Calibrated browser decode_size must contain positive integer width and height.");
	}
	if (!Array.isArray(bundle.cameras) || bundle.cameras.length < 2) {
		throw new Error("Calibrated browser bundle must contain train and heldout cameras.");
	}
	for (const camera of bundle.cameras) {
		const matrix = camera.world_to_camera;
		if (!Array.isArray(camera.intrinsics) || camera.intrinsics.length !== 4
			|| !camera.intrinsics.every(Number.isFinite) || camera.intrinsics[0] <= 0
			|| camera.intrinsics[1] <= 0) {
			throw new Error(`Camera ${camera.name ?? "unnamed"} has invalid normalized pinhole intrinsics.`);
		}
		if (!Array.isArray(matrix) || matrix.length !== 4
			|| matrix.some((row) => !Array.isArray(row) || row.length !== 4 || !row.every(Number.isFinite))) {
			throw new Error(`Camera ${camera.name ?? "unnamed"} must provide a finite 4x4 world_to_camera matrix.`);
		}
		const homogeneousError = Math.max(
			Math.abs(matrix[3][0]), Math.abs(matrix[3][1]), Math.abs(matrix[3][2]),
			Math.abs(matrix[3][3] - 1),
		);
		let orthogonalityError = 0;
		for (let row = 0; row < 3; row += 1) {
			for (let other = 0; other < 3; other += 1) {
				let dot = 0;
				for (let column = 0; column < 3; column += 1) dot += matrix[row][column] * matrix[other][column];
				orthogonalityError = Math.max(orthogonalityError, Math.abs(dot - Number(row === other)));
			}
		}
		const determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
			- matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
			+ matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
		if (homogeneousError > 1e-4 || orthogonalityError > 1e-3 || Math.abs(determinant - 1) > 1e-3) {
			throw new Error(`Camera ${camera.name ?? "unnamed"} world_to_camera must be a rigid proper transform.`);
		}
	}
	const anchor = bundle.cameras.find((camera) => camera.name === contract.anchor_camera);
	if (!anchor || anchor.role !== "train") {
		throw new Error("The calibrated anchor camera must exist in the training split.");
	}
	const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
	const anchorMatrix = anchor.world_to_camera.flat();
	if (anchorMatrix.some((value, index) => Math.abs(value - identity[index]) > 1e-4)) {
		throw new Error("The calibrated world frame must be relative to the declared anchor camera.");
	}
	return bundle;
}

export async function loadCalibratedMulticamDataset({
	computeSamples = true,
	frameBankFormat = FRAME_BANK_FORMAT_RGBA8,
} = {}) {
	const response = await fetch(CALIBRATED_MULTICAM_URL);
	if (!response.ok) {
		throw new Error(`Calibrated browser bundle unavailable: ${response.status}`);
	}
	const bundle = validateCalibratedMulticamBundle(await response.json());
	const [width, height] = bundle.decode_size;
	const frameCount = bundle.frame_count;
	const valuesPerView = width * height * frameCount * 4;
	const frames = allocateFrameBank(valuesPerView * bundle.cameras.length, frameBankFormat);
	const backgrounds = new Float32Array(width * height * 4 * bundle.cameras.length);
	// Decode one atlas at a time into its final bank slice. In compact mode this
	// avoids both per-camera FP32 banks and a second concatenated FP32 bank.
	for (let view = 0; view < bundle.cameras.length; view += 1) {
		const camera = bundle.cameras[view];
		const decode = camera.frame_atlas_url ? decodeFrameAtlasInto : decodeVideoFramesInto;
		await decode({
			url: camera.frame_atlas_url ?? camera.video_url,
			width,
			height,
			frameCount,
			frameTimesSeconds: bundle.frame_times_seconds,
			target: { format: frameBankFormat, data: frames },
			targetOffset: view * valuesPerView,
		});
		backgrounds.set(
			computeMeanBackground(
				{ format: frameBankFormat, data: frames },
				width,
				height,
				frameCount,
				view * valuesPerView,
			),
			view * width * height * 4,
		);
	}
	const trainViewIndices = bundle.cameras
		.map((camera, index) => camera.role === "train" ? index : -1)
		.filter((index) => index >= 0);
	const heldoutViewIndex = bundle.cameras.findIndex((camera) => camera.role === "heldout");
	const trainViewCount = trainViewIndices.length;
	if (!trainViewIndices.every((view, index) => view === index) || heldoutViewIndex !== trainViewCount) {
		throw new Error("Browser trainer requires train cameras first and the heldout camera last.");
	}
	const samples = computeSamples
		? computeMultiviewSamples(
			frames,
			backgrounds,
			width,
			height,
			frameCount,
			trainViewCount,
		)
		: {
			motionSamples: new Uint32Array(0),
			staticSamples: new Uint32Array(0),
		};
	const cameras = bundle.cameras.map((camera) => ({
		...camera,
		intrinsics: new Float32Array(camera.intrinsics),
		worldToCamera: new Float32Array(camera.world_to_camera.flat()),
	}));
	const dataset = attachPixelBanks({
		name: bundle.name,
		source: CALIBRATED_MULTICAM_URL,
		width,
		height,
		frameCount,
		viewCount: cameras.length,
		trainViewCount,
		trainViewIndices,
		heldoutViewIndex,
		cameras,
		seedPoints: new Float32Array(bundle.seed_points_xyzrgb.flat()),
		seedPointCount: bundle.seed_points_xyzrgb.length,
		seedSource: bundle.seed_source,
		seedProvenance: bundle.seed_provenance ?? {
			method: "legacy_external_unverified",
			source_report: null,
			source_path: bundle.seed_source,
			input_cameras: [],
			train_only_verified: false,
		},
		datasetContract: bundle.dataset_contract,
		frameIndices: bundle.frame_indices,
		...samples,
	}, frames, backgrounds, frameBankFormat);
	dataset.viewDatasets = cameras.map((camera, view) => ({
		label: `${camera.name} ${camera.role}`,
		width,
		height,
		frameCount,
		frames: frames.subarray(view * valuesPerView, (view + 1) * valuesPerView),
		frameBank: {
			format: frameBankFormat,
			data: frames.subarray(view * valuesPerView, (view + 1) * valuesPerView),
		},
		background: backgrounds.subarray(view * width * height * 4, (view + 1) * width * height * 4),
		backgroundBank: {
			format: BACKGROUND_BANK_FORMAT_RGBA32_FLOAT,
			data: backgrounds.subarray(
				view * width * height * 4,
				(view + 1) * width * height * 4,
			),
		},
		viewIndex: view,
	}));
	const anchorName = bundle.dataset_contract?.anchor_camera;
	const trainA = Math.max(0, cameras.findIndex((camera) => camera.name === anchorName));
	const preferredTrainB = cameras.findIndex((camera) => camera.name === "cam09" && camera.role === "train");
	const trainB = preferredTrainB >= 0 ? preferredTrainB : trainViewIndices.find((view) => view !== trainA) ?? trainA;
	dataset.comparisonViewIndices = [trainA, trainB, heldoutViewIndex];
	dataset.previewViews = dataset.comparisonViewIndices.map((view) => dataset.viewDatasets[view]);
	return dataset;
}

export async function loadPresetDataset({
	allowLegacyFallback = false,
	computeSamples = true,
	// Both GPU trainers decode compact targets at their binding boundary. Keep
	// the much larger all-camera frame bank byte-packed on the host.
	frameBankFormat = FRAME_BANK_FORMAT_RGBA8,
} = {}) {
	try {
		return await loadCalibratedMulticamDataset({ computeSamples, frameBankFormat });
	} catch (error) {
		if (!allowLegacyFallback) {
			throw error;
		}
		console.info("Calibrated multicamera bundle unavailable; legacy fallback was explicitly enabled.", error);
	}
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
	const frameBank = resolveFrameBank(dataset);
	for (let i = 0; i < dataset.width * dataset.height; i += 1) {
		const base = offset + i * 4;
		let r;
		let g;
		let b;
		if (frameBank.format === FRAME_BANK_FORMAT_RGBA8) {
			r = frameBank.data[base] / 255;
			g = frameBank.data[base + 1] / 255;
			b = frameBank.data[base + 2] / 255;
		} else {
			r = frameBank.data[base];
			g = frameBank.data[base + 1];
			b = frameBank.data[base + 2];
		}
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
