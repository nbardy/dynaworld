import { computeMultiviewSamples } from "./dataset.js";

function positiveInteger(value, label) {
	if (!Number.isSafeInteger(value) || value < 1) {
		throw new RangeError(`${label} must be a positive integer.`);
	}
	return value;
}

export function resizePackedRgbaNearest(source, sourceWidth, sourceHeight, imageCount, scale) {
	positiveInteger(sourceWidth, "sourceWidth");
	positiveInteger(sourceHeight, "sourceHeight");
	positiveInteger(imageCount, "imageCount");
	positiveInteger(scale, "scale");
	const sourcePixels = sourceWidth * sourceHeight;
	if (source.length !== sourcePixels * imageCount * 4) {
		throw new RangeError("source must contain imageCount packed RGBA images.");
	}
	if (scale === 1) return Float32Array.from(source);
	const width = sourceWidth * scale;
	const height = sourceHeight * scale;
	const targetPixels = width * height;
	const resized = new Float32Array(targetPixels * imageCount * 4);
	for (let image = 0; image < imageCount; image += 1) {
		const sourceImage = image * sourcePixels * 4;
		const targetImage = image * targetPixels * 4;
		for (let y = 0; y < height; y += 1) {
			const sourceY = Math.floor(y / scale);
			for (let x = 0; x < width; x += 1) {
				const sourceX = Math.floor(x / scale);
				const sourceBase = sourceImage + (sourceY * sourceWidth + sourceX) * 4;
				const targetBase = targetImage + (y * width + x) * 4;
				resized.set(source.subarray(sourceBase, sourceBase + 4), targetBase);
			}
		}
	}
	return resized;
}

export function resizeDatasetForBenchmark(dataset, scale, { computeSamples = true } = {}) {
	positiveInteger(scale, "scale");
	if (scale === 1) return dataset;
	const width = dataset.width * scale;
	const height = dataset.height * scale;
	const frames = resizePackedRgbaNearest(
		dataset.frames,
		dataset.width,
		dataset.height,
		dataset.viewCount * dataset.frameCount,
		scale,
	);
	const backgrounds = resizePackedRgbaNearest(
		dataset.backgrounds,
		dataset.width,
		dataset.height,
		dataset.viewCount,
		scale,
	);
	const samples = computeSamples
		? computeMultiviewSamples(
			frames,
			backgrounds,
			width,
			height,
			dataset.frameCount,
			dataset.trainViewCount,
		)
		: {
			motionSamples: new Uint32Array(0),
			staticSamples: new Uint32Array(0),
		};
	const valuesPerView = width * height * dataset.frameCount * 4;
	const backgroundValuesPerView = width * height * 4;
	const resized = {
		...dataset,
		name: `${dataset.name} (${scale}x raster benchmark)`,
		width,
		height,
		frames,
		backgrounds,
		background: backgrounds.subarray(0, backgroundValuesPerView),
		benchmarkSourceRaster: [dataset.width, dataset.height],
		benchmarkRasterScale: scale,
		...samples,
	};
	resized.viewDatasets = dataset.cameras.map((camera, view) => ({
		label: `${camera.name} ${camera.role}`,
		width,
		height,
		frameCount: dataset.frameCount,
		frames: frames.subarray(view * valuesPerView, (view + 1) * valuesPerView),
		background: backgrounds.subarray(
			view * backgroundValuesPerView,
			(view + 1) * backgroundValuesPerView,
		),
		viewIndex: view,
	}));
	resized.previewViews = resized.comparisonViewIndices.map((view) => resized.viewDatasets[view]);
	return resized;
}
