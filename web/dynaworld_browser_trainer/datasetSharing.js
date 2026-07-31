const SHARED_MODE = "shared-array-buffer";
const CLONED_MODE = "structured-clone";

function requireFloat32Array(value, label) {
	if (!(value instanceof Float32Array)) {
		throw new TypeError(`${label} must be a Float32Array.`);
	}
	return value;
}

function sharedArrayBufferConstructor(scope = globalThis) {
	return typeof scope?.SharedArrayBuffer === "function" ? scope.SharedArrayBuffer : null;
}

function isSharedBuffer(buffer, scope = globalThis) {
	const SharedBuffer = sharedArrayBufferConstructor(scope);
	return Boolean(SharedBuffer && buffer instanceof SharedBuffer);
}

function readOnlyRoots(dataset) {
	const roots = [requireFloat32Array(dataset?.frames, "dataset.frames")];
	if (dataset.backgrounds != null) {
		roots.push(requireFloat32Array(dataset.backgrounds, "dataset.backgrounds"));
	} else if (dataset.background != null) {
		roots.push(requireFloat32Array(dataset.background, "dataset.background"));
	}
	return roots.filter((view, index, values) =>
		values.findIndex((candidate) => candidate.buffer === view.buffer) === index);
}

function copyToShared(view, SharedBuffer) {
	if (view.buffer instanceof SharedBuffer) return view;
	const shared = new view.constructor(new SharedBuffer(view.byteLength));
	shared.set(view);
	return shared;
}

function rebuildDatasetAliases(dataset) {
	const pixels = Number(dataset.width) * Number(dataset.height);
	const frameValuesPerView = pixels * Number(dataset.frameCount) * 4;
	const backgroundValuesPerView = pixels * 4;
	if (!Number.isSafeInteger(pixels) || pixels < 1
		|| !Number.isSafeInteger(frameValuesPerView) || frameValuesPerView < 1) {
		throw new RangeError("Dataset dimensions must describe a non-empty frame bank.");
	}
	if (dataset.backgrounds) {
		dataset.background = dataset.backgrounds.subarray(0, backgroundValuesPerView);
	}
	if (Array.isArray(dataset.viewDatasets)) {
		for (let index = 0; index < dataset.viewDatasets.length; index += 1) {
			const viewDataset = dataset.viewDatasets[index];
			const view = Number.isSafeInteger(viewDataset.viewIndex) ? viewDataset.viewIndex : index;
			viewDataset.frames = dataset.frames.subarray(
				view * frameValuesPerView,
				(view + 1) * frameValuesPerView,
			);
			if (dataset.backgrounds) {
				viewDataset.background = dataset.backgrounds.subarray(
					view * backgroundValuesPerView,
					(view + 1) * backgroundValuesPerView,
				);
			} else if (view === 0 && dataset.background) {
				viewDataset.background = dataset.background;
			}
		}
	}
	if (Array.isArray(dataset.comparisonViewIndices) && Array.isArray(dataset.viewDatasets)) {
		dataset.previewViews = dataset.comparisonViewIndices
			.map((view) => dataset.viewDatasets[view])
			.filter(Boolean);
	}
	return dataset;
}

export function datasetSharingCapability(scope = globalThis) {
	if (scope?.crossOriginIsolated !== true) {
		return {
			available: false,
			reason: "Cross-origin isolation is unavailable; decoded targets use structured cloning.",
		};
	}
	if (!sharedArrayBufferConstructor(scope)) {
		return {
			available: false,
			reason: "SharedArrayBuffer is unavailable; decoded targets use structured cloning.",
		};
	}
	return { available: true, reason: null };
}

export function summarizeDatasetSharing(dataset, scope = globalThis) {
	const roots = readOnlyRoots(dataset);
	const readOnlyBytes = roots.reduce((sum, view) => sum + view.byteLength, 0);
	const sharedRoots = roots.filter((view) => isSharedBuffer(view.buffer, scope));
	const sharedBytes = sharedRoots.reduce((sum, view) => sum + view.byteLength, 0);
	return {
		mode: sharedRoots.length === roots.length ? SHARED_MODE : CLONED_MODE,
		readOnlyBytes,
		sharedBytes,
		sharedBufferCount: sharedRoots.length,
	};
}

export function hydrateDatasetSharedViews(dataset, scope = globalThis) {
	rebuildDatasetAliases(dataset);
	return {
		dataset,
		telemetry: summarizeDatasetSharing(dataset, scope),
	};
}

export function prepareDatasetForWorkerSharing(dataset, scope = globalThis) {
	const capability = datasetSharingCapability(scope);
	if (!capability.available) {
		const hydrated = hydrateDatasetSharedViews(dataset, scope);
		return {
			...hydrated,
			telemetry: {
				...hydrated.telemetry,
				available: false,
				estimatedCopiesAvoided: 0,
				estimatedBytesAvoided: 0,
				reason: capability.reason,
			},
		};
	}

	const SharedBuffer = sharedArrayBufferConstructor(scope);
	// SharedArrayBuffer cannot enforce read-only access. Admit only the decoded
	// target banks; optimizer, seed, camera, and sampling arrays remain private.
	dataset.frames = copyToShared(requireFloat32Array(dataset.frames, "dataset.frames"), SharedBuffer);
	if (dataset.backgrounds != null) {
		dataset.backgrounds = copyToShared(
			requireFloat32Array(dataset.backgrounds, "dataset.backgrounds"),
			SharedBuffer,
		);
	} else if (dataset.background != null) {
		dataset.background = copyToShared(
			requireFloat32Array(dataset.background, "dataset.background"),
			SharedBuffer,
		);
	}
	const hydrated = hydrateDatasetSharedViews(dataset, scope);
	return {
		...hydrated,
		telemetry: {
			...hydrated.telemetry,
			available: true,
			estimatedCopiesAvoided: 2,
			estimatedBytesAvoided: hydrated.telemetry.readOnlyBytes * 2,
			reason: null,
		},
	};
}

export function combineDatasetSharingTelemetry(main, training, validation) {
	const contexts = {
		main: main?.mode ?? "unknown",
		training: training?.mode ?? "unknown",
		validation: validation?.mode ?? "unknown",
	};
	const sharedEverywhere = main?.available === true
		&& Object.values(contexts).every((mode) => mode === SHARED_MODE);
	const readOnlyBytes = Math.max(
		Number(main?.readOnlyBytes ?? 0),
		Number(training?.readOnlyBytes ?? 0),
		Number(validation?.readOnlyBytes ?? 0),
	);
	return {
		available: main?.available === true,
		mode: sharedEverywhere ? SHARED_MODE : CLONED_MODE,
		readOnlyBytes,
		sharedBytes: sharedEverywhere ? readOnlyBytes : 0,
		estimatedCopiesAvoided: sharedEverywhere ? 2 : 0,
		estimatedBytesAvoided: sharedEverywhere ? readOnlyBytes * 2 : 0,
		contexts,
		reason: sharedEverywhere ? null
			: main?.reason ?? "At least one worker received a private decoded-target copy.",
	};
}
