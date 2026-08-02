const RGBA_CHANNELS = 4;
const RGBA8_BYTES_PER_CHANNEL = Uint8Array.BYTES_PER_ELEMENT;
const BACKGROUND_BYTES_PER_CHANNEL = Float32Array.BYTES_PER_ELEMENT;
const RESIDENT_PAGE_SLOTS = 2;

function positiveInteger(value, label) {
	if (!Number.isSafeInteger(value) || value < 1) {
		throw new RangeError(`${label} must be a positive safe integer.`);
	}
	return value;
}

function positiveFinite(value, label) {
	if (!Number.isFinite(value) || value <= 0) {
		throw new RangeError(`${label} must be finite and positive.`);
	}
	return value;
}

function validateNativeFrameIndices(value) {
	if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
		throw new TypeError("nativeFrameIndices must be an Array or typed array.");
	}
	const indices = Array.from(value);
	if (indices.length === 0) {
		throw new RangeError("nativeFrameIndices must not be empty.");
	}
	for (let index = 0; index < indices.length; index += 1) {
		if (!Number.isSafeInteger(indices[index]) || indices[index] < 0) {
			throw new RangeError(
				"nativeFrameIndices must contain non-negative safe integers.",
			);
		}
		if (index > 0 && indices[index] <= indices[index - 1]) {
			throw new RangeError(
				"nativeFrameIndices must be strictly increasing and unique.",
			);
		}
	}
	return indices;
}

function safeProduct(label, ...values) {
	let product = 1;
	for (const value of values) {
		if (product > Number.MAX_SAFE_INTEGER / value) {
			throw new RangeError(`${label} exceeds JavaScript's safe integer range.`);
		}
		product *= value;
	}
	return product;
}

function safeSum(label, ...values) {
	let sum = 0;
	for (const value of values) {
		if (sum > Number.MAX_SAFE_INTEGER - value) {
			throw new RangeError(`${label} exceeds JavaScript's safe integer range.`);
		}
		sum += value;
	}
	return sum;
}

function extraStrata(stratumCount, extraCount) {
	if (extraCount === 0) return new Set();
	if (extraCount === 1) return new Set([Math.floor((stratumCount - 1) / 2)]);
	return new Set(Array.from(
		{ length: extraCount },
		(_, index) => Math.round(index * (stratumCount - 1) / (extraCount - 1)),
	));
}


function interleavedNativePages(frameIndices, pageSize) {
	const nativeFrameCount = frameIndices.length;
	const stratumCount = Math.min(nativeFrameCount, pageSize);
	const completeRounds = Math.floor(nativeFrameCount / stratumCount);
	const remainder = nativeFrameCount % stratumCount;
	const largerStrata = extraStrata(stratumCount, remainder);
	const strata = [];
	let firstNativeFrame = 0;

	for (let stratum = 0; stratum < stratumCount; stratum += 1) {
		const size = completeRounds + Number(largerStrata.has(stratum));
		strata.push({ firstNativeFrame, size });
		firstNativeFrame += size;
	}

	const pages = [];
	for (let round = 0; round < completeRounds + Number(remainder > 0); round += 1) {
		const nativeFrameIndices = [];
		for (const stratum of strata) {
			if (round < stratum.size) {
				nativeFrameIndices.push(frameIndices[stratum.firstNativeFrame + round]);
			}
		}
		pages.push(nativeFrameIndices);
	}
	return pages;
}

/**
 * Plans two-slot host paging for every native frame in a fixed-rate clip.
 * RGBA8 pages are double-buffered; one camera-local RGBA32F background bank
 * remains resident independently of temporal pages.
 */
export function planTemporalPaging(contract) {
	if (contract == null || typeof contract !== "object" || Array.isArray(contract)) {
		throw new TypeError("Temporal paging contract must be an object.");
	}
	const frameIndices = validateNativeFrameIndices(contract.nativeFrameIndices);
	const frameCount = frameIndices.length;
	const pageSize = positiveInteger(contract.pageSize, "pageSize");
	const fps = positiveFinite(contract.fps, "fps");
	const durationSeconds = positiveFinite(contract.durationSeconds, "durationSeconds");
	const width = positiveInteger(contract.width, "width");
	const height = positiveInteger(contract.height, "height");
	const cameraCount = positiveInteger(contract.cameraCount, "cameraCount");
	const firstFrameTimeSeconds = frameIndices[0] / fps;
	const lastFrameTimeSeconds = frameIndices.at(-1) / fps;
	const observedTimeSpanSeconds = lastFrameTimeSeconds - firstFrameTimeSeconds;
	if (lastFrameTimeSeconds > durationSeconds + Number.EPSILON * Math.max(1, durationSeconds)) {
		throw new RangeError(
			"durationSeconds must include the timestamp of the final native frame.",
		);
	}

	const bytesPerRgba8Frame = safeProduct(
		"bytesPerRgba8Frame",
		width,
		height,
		RGBA_CHANNELS,
		RGBA8_BYTES_PER_CHANNEL,
	);
	const bytesPerCameraBackground = safeProduct(
		"bytesPerCameraBackground",
		width,
		height,
		RGBA_CHANNELS,
		BACKGROUND_BYTES_PER_CHANNEL,
	);
	const residentFramesPerPage = Math.min(pageSize, frameCount);
	const rgba8PageBytes = safeProduct(
		"rgba8PageBytes",
		bytesPerRgba8Frame,
		cameraCount,
		residentFramesPerPage,
	);
	const rgba8DoubleBufferBytes = safeProduct(
		"rgba8DoubleBufferBytes",
		rgba8PageBytes,
		RESIDENT_PAGE_SLOTS,
	);
	const backgroundBytes = safeProduct(
		"backgroundBytes",
		bytesPerCameraBackground,
		cameraCount,
	);
	const totalResidentBytes = safeSum(
		"totalResidentBytes",
		rgba8DoubleBufferBytes,
		backgroundBytes,
	);

	const pages = interleavedNativePages(frameIndices, pageSize).map(
		(nativeFrameIndices, pageIndex) => {
			const timeSeconds = nativeFrameIndices.map((frame) => frame / fps);
			return {
				pageIndex,
				nativeFrameIndices,
				timeSeconds,
				// Model time spans observed frame centers. Clip duration includes the
				// final frame's display interval and would otherwise leave t=1 unseen.
				normalizedTimes: timeSeconds.map((time) => observedTimeSpanSeconds > 0
					? (time - firstFrameTimeSeconds) / observedTimeSpanSeconds : 0),
				rgba8Bytes: safeProduct(
					`pages[${pageIndex}].rgba8Bytes`,
					bytesPerRgba8Frame,
					cameraCount,
					nativeFrameIndices.length,
				),
			};
		},
	);

	return {
		frameCount,
		pageSize,
		pageCount: pages.length,
		fps,
		durationSeconds,
		firstFrameTimeSeconds,
		lastFrameTimeSeconds,
		observedTimeSpanSeconds,
		pages,
		memory: {
			bytesPerRgba8Frame,
			corpusRgba8Bytes: safeProduct(
				"corpusRgba8Bytes",
				bytesPerRgba8Frame,
				cameraCount,
				frameCount,
			),
			rgba8PageBytes,
			residentPageSlots: RESIDENT_PAGE_SLOTS,
			rgba8DoubleBufferBytes,
			backgroundBytes,
			totalResidentBytes,
		},
	};
}

/** Returns the page-local slot closest to a normalized preview time. */
export function selectNearestResidentSlot(page, normalizedTime) {
	if (page == null || typeof page !== "object" || Array.isArray(page)) {
		throw new TypeError("Resident page must be an object.");
	}
	if (!Number.isFinite(normalizedTime) || normalizedTime < 0 || normalizedTime > 1) {
		throw new RangeError("normalizedTime must be finite and within [0, 1].");
	}
	const times = page.normalizedTimes;
	if (!Array.isArray(times) || times.length === 0) {
		throw new TypeError("Resident page normalizedTimes must be a non-empty array.");
	}
	for (let index = 0; index < times.length; index += 1) {
		if (!Number.isFinite(times[index]) || times[index] < 0 || times[index] > 1
			|| (index > 0 && times[index] <= times[index - 1])) {
			throw new RangeError(
				"Resident page normalizedTimes must be finite, unique, increasing, and within [0, 1].",
			);
		}
	}

	let low = 0;
	let high = times.length;
	while (low < high) {
		const middle = Math.floor((low + high) / 2);
		if (times[middle] < normalizedTime) low = middle + 1;
		else high = middle;
	}
	if (low === 0) return 0;
	if (low === times.length) return times.length - 1;
	return normalizedTime - times[low - 1] <= times[low] - normalizedTime
		? low - 1
		: low;
}
