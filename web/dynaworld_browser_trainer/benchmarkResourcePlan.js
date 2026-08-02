const MIB = 1024 ** 2;
const SPLAT_BYTES = 24 * Float32Array.BYTES_PER_ELEMENT;
const MAX_RENDER_VIEWS = 3;
const MAX_SAMPLES_PER_STEP = 192;
const SSIM_STATS_BYTES = 5 * 16;
const TILED_CONFIG_BYTES = 192;
const PORTABLE_STORAGE_BUFFER_LIMIT = 128 * MIB;

function positiveInteger(value, label) {
	if (!Number.isSafeInteger(value) || value < 1) {
		throw new RangeError(`${label} must be a positive safe integer.`);
	}
	return value;
}

function ceilDiv(value, divisor) {
	return Math.floor((value + divisor - 1) / divisor);
}

function nextPowerOfTwo(value) {
	return 2 ** Math.ceil(Math.log2(Math.max(1, value)));
}

function resolveCheckpointBytes(
	pixelCount,
	tileCapacity,
	bytesPerCheckpoint,
	minimumStride,
	storageBufferLimit,
) {
	for (let stride = minimumStride; stride <= tileCapacity; stride *= 2) {
		const blocksPerTile = ceilDiv(tileCapacity, stride);
		const byteLength = pixelCount * blocksPerTile * bytesPerCheckpoint;
		if (Number.isSafeInteger(byteLength) && byteLength <= storageBufferLimit) {
			return { stride, blocksPerTile, byteLength };
		}
	}
	throw new RangeError(
		`Checkpoint storage cannot fit within ${storageBufferLimit} bytes.`,
	);
}

function variantDefinitions(options) {
	const base = {
		checkpointPrecision: options.checkpointPrecision,
		checkpointStride: options.checkpointStride,
		projectionLayout: options.projectionLayout,
		projectionVjpPrecision: options.projectionVjpPrecision ?? "f32",
		ssimLayout: options.ssimLayout,
	};
	const variants = {
		backward: [
			{
				id: "direct-3d",
				...base,
				backwardMode: "direct-3d",
				projectionLayout: "monolithic",
			},
			{
				id: "staged-project3d",
				...base,
				backwardMode: "staged-project3d",
			},
		],
		projection: [
			{
				id: "staged-monolithic",
				...base,
				backwardMode: "staged-project3d",
				projectionLayout: "monolithic",
			},
			{
				id: "staged-split-compact",
				...base,
				backwardMode: "staged-project3d",
				projectionLayout: "split-compact",
			},
		],
		precision: [
			{
				id: "staged-split-f32",
				...base,
				backwardMode: "staged-project3d",
				projectionLayout: "split-compact",
				projectionVjpPrecision: "f32",
			},
			{
				id: "staged-split-packed-f16",
				...base,
				backwardMode: "staged-project3d",
				projectionLayout: "split-compact",
				projectionVjpPrecision: "packed-f16",
			},
		],
		ssim: [
			{
				id: "staged-naive-2d",
				...base,
				backwardMode: "staged-project3d",
				ssimLayout: "naive-2d",
			},
			{
				id: "staged-separable",
				...base,
				backwardMode: "staged-project3d",
				ssimLayout: "separable",
			},
		],
	}[options.experiment];
	if (!variants) throw new RangeError(`Unknown experiment: ${options.experiment}.`);
	if (options.variant === "both") return variants;
	if (options.variant === "control") return [variants[0]];
	if (options.variant === "candidate") return [variants[1]];
	const selected = variants.filter((variant) => variant.id === options.variant);
	if (!selected.length) throw new RangeError(`Unknown benchmark variant: ${options.variant}.`);
	return selected;
}

export function estimateTiledTrainerBuffers({
	width,
	height,
	viewCount,
	trainViewCount,
	frameCount,
	capacity,
	tileSize,
	tileCapacity,
	checkpointPrecision,
	checkpointStride,
	projectionLayout,
	projectionVjpPrecision = "f32",
	ssimLayout,
	backwardMode,
	compactTargetFrames = false,
	storageBufferLimit = PORTABLE_STORAGE_BUFFER_LIMIT,
}) {
	for (const [label, value] of Object.entries({
		width,
		height,
		viewCount,
		trainViewCount,
		frameCount,
		capacity,
		tileSize,
		tileCapacity,
		checkpointStride,
		storageBufferLimit,
	})) positiveInteger(value, label);
	if (!["f32", "packed-f16"].includes(projectionVjpPrecision)) {
		throw new RangeError("projectionVjpPrecision must be f32 or packed-f16.");
	}
	const pixelCount = width * height;
	const tileCount = ceilDiv(width, tileSize) * ceilDiv(height, tileSize);
	const pairCapacity = tileCount * tileCapacity;
	const bytesPerCheckpoint = checkpointPrecision === "packed-f16" ? 8 : 16;
	const checkpoint = resolveCheckpointBytes(
		pixelCount,
		tileCapacity,
		bytesPerCheckpoint,
		checkpointStride,
		storageBufferLimit,
	);
	const staged = backwardMode === "staged-project3d";
	const splitProjection = projectionLayout === "split-compact";
	const gradientFloats = staged ? 12 : 24;
	const projectionVjpBytesPerSplat = projectionVjpPrecision === "packed-f16" ? 48 : 80;
	const projectionBytesPerSplat = splitProjection ? 32 + projectionVjpBytesPerSplat : 192;
	const rasterProjectionBytes = capacity * (splitProjection ? 32 : 192);
	const projectionVjpBytes = splitProjection ? capacity * projectionVjpBytesPerSplat : 0;
	const cycleMetricBytes = trainViewCount * frameCount * 16;
	const timestampBytes = 11 * 2 * BigUint64Array.BYTES_PER_ELEMENT;
	const bufferBytes = {
		parameterPingPong: capacity * SPLAT_BYTES * 2,
		optimizerMoments: capacity * SPLAT_BYTES * 2,
		densityStats: capacity * 16,
		gradientAccumulator: capacity * gradientFloats * 4,
		projections: capacity * projectionBytesPerSplat,
		parameterReadback: capacity * SPLAT_BYTES,
		previewSort: nextPowerOfTwo(capacity) * 4 * MAX_RENDER_VIEWS * 2,
		sampleIndices: 4,
		sampledWorkspace: SPLAT_BYTES + MAX_SAMPLES_PER_STEP * 4,
		targetPage: pixelCount * 4 * Float32Array.BYTES_PER_ELEMENT,
		packedTargetPage: compactTargetFrames ? pixelCount * 4 : 4,
		cameraData: (viewCount * 2 + MAX_RENDER_VIEWS) * 20 * 4 + trainViewCount * 4 + 4 + viewCount * 4 * 4,
		rasterPairs: pairCapacity * 8 + tileCount * 4,
		transmittanceCheckpoints: checkpoint.byteLength,
		fullImageWorkspace: pixelCount * (
			16
			+ 4
			+ SSIM_STATS_BYTES
			+ (ssimLayout === "separable" ? SSIM_STATS_BYTES : 0)
			+ 16
			+ 16
		),
		configAndTelemetry: 144 + MAX_RENDER_VIEWS * 48 + TILED_CONFIG_BYTES + 40 + 12 + 80
			+ cycleMetricBytes + (80 + cycleMetricBytes) + timestampBytes * 2,
		previewGeometry: 32,
	};
	const allocatedBytes = Object.values(bufferBytes).reduce((sum, value) => sum + value, 0);
	const bindingBytes = {
		gradientAccumulator: bufferBytes.gradientAccumulator,
		rasterProjections: rasterProjectionBytes,
		projectionVjp: projectionVjpBytes,
		targetPage: bufferBytes.targetPage,
		pairData: pairCapacity * 8,
		transmittanceCheckpoints: bufferBytes.transmittanceCheckpoints,
		fullImageRecord: Math.max(
			pixelCount * SSIM_STATS_BYTES,
			ssimLayout === "separable" ? pixelCount * SSIM_STATS_BYTES : 0,
		),
	};
	const largestBinding = Object.entries(bindingBytes)
		.sort((left, right) => right[1] - left[1])[0];
	return {
		allocatedBytes,
		bufferBytes,
		bindingBytes,
		largestBinding: { label: largestBinding[0], byteLength: largestBinding[1] },
		checkpoint,
		pixelCount,
		tileCount,
		pairCapacity,
		projectionVjpPrecision: splitProjection ? projectionVjpPrecision : "f32",
	};
}

export function estimateDatasetResidentBytes({
	sourceWidth,
	sourceHeight,
	width,
	height,
	viewCount,
	frameCount,
	channelBytes = Float32Array.BYTES_PER_ELEMENT,
}) {
	for (const [label, value] of Object.entries({
		sourceWidth,
		sourceHeight,
		width,
		height,
		viewCount,
		frameCount,
		channelBytes,
	})) positiveInteger(value, label);
	const frameBytes = (rasterWidth, rasterHeight, bytesPerChannel) => (
		viewCount * frameCount * rasterWidth * rasterHeight * 4 * bytesPerChannel
	);
	const backgroundBytes = (rasterWidth, rasterHeight) => (
		viewCount * rasterWidth * rasterHeight * 4 * Float32Array.BYTES_PER_ELEMENT
	);
	const sourceFrameBytes = frameBytes(sourceWidth, sourceHeight, channelBytes);
	const sourceBackgroundBytes = backgroundBytes(sourceWidth, sourceHeight);
	const sourceResidentBytes = sourceFrameBytes + sourceBackgroundBytes;
	const scaledBytes = width === sourceWidth && height === sourceHeight
		? 0
		: frameBytes(width, height, channelBytes) + backgroundBytes(width, height);
	// Atlas decode is sequential, so one camera atlas is the transient peak.
	const decodedAtlasBytes = sourceWidth * frameCount * sourceHeight * 4;
	return {
		sourceFrameBytes,
		sourceBackgroundBytes,
		sourceResidentBytes,
		scaledBytes,
		decodedAtlasBytes,
		totalBytes: sourceResidentBytes + scaledBytes + decodedAtlasBytes,
		channelBytes,
	};
}

export function estimateTiledBenchmarkResources(metadata, options, {
	storageBufferLimit = PORTABLE_STORAGE_BUFFER_LIMIT,
	datasetChannelBytes = Float32Array.BYTES_PER_ELEMENT,
	fixedBrowserHeadroomBytes = 512 * MIB,
	workingSetMultiplier = 1.5,
} = {}) {
	const width = positiveInteger(metadata.width, "metadata.width") * options.scale;
	const height = positiveInteger(metadata.height, "metadata.height") * options.scale;
	const common = {
		width,
		height,
		viewCount: metadata.viewCount,
		trainViewCount: metadata.trainViewCount,
		frameCount: metadata.frameCount,
		capacity: options.capacity,
		tileSize: options.tileSize,
		tileCapacity: options.tileCapacity,
		compactTargetFrames: datasetChannelBytes === Uint8Array.BYTES_PER_ELEMENT,
		storageBufferLimit,
	};
	const variants = variantDefinitions(options).map((variant) => ({
		id: variant.id,
		...estimateTiledTrainerBuffers({ ...common, ...variant }),
	}));
	const dataset = estimateDatasetResidentBytes({
		sourceWidth: metadata.width,
		sourceHeight: metadata.height,
		width,
		height,
		viewCount: metadata.viewCount,
		frameCount: metadata.frameCount,
		channelBytes: datasetChannelBytes,
	});
	const gpuBufferBytes = variants.reduce((sum, variant) => sum + variant.allocatedBytes, 0);
	const retainedInitialParamsBytes = variants.length * options.capacity * SPLAT_BYTES;
	const knownWorkingSetBytes = dataset.totalBytes + gpuBufferBytes + retainedInitialParamsBytes;
	const minimumAvailableMemoryBytes = Math.ceil(
		fixedBrowserHeadroomBytes + knownWorkingSetBytes * workingSetMultiplier,
	);
	const invalidBindings = variants
		.filter((variant) => variant.largestBinding.byteLength > storageBufferLimit)
		.map((variant) => (
			`${variant.id} ${variant.largestBinding.label} needs `
			+ `${variant.largestBinding.byteLength} bytes`
		));
	return {
		schema: "dynaworld-browser-tiled-resource-plan/v1",
		raster: [width, height],
		sourceRaster: [metadata.width, metadata.height],
		dataset,
		variants,
		gpuBufferBytes,
		retainedInitialParamsBytes,
		knownWorkingSetBytes,
		fixedBrowserHeadroomBytes,
		workingSetMultiplier,
		minimumAvailableMemoryBytes,
		portableStorageBufferLimit: storageBufferLimit,
		valid: invalidBindings.length === 0,
		reasons: invalidBindings,
		limitations: [
			"GPU bytes are WebGPU buffer allocations; Apple unified memory is shared with the host.",
			"Decoded image and browser-runtime peaks are estimated, not driver allocation telemetry.",
			"The browser still validates every binding against the selected adapter before training.",
		],
	};
}

export { MIB, PORTABLE_STORAGE_BUFFER_LIMIT };
