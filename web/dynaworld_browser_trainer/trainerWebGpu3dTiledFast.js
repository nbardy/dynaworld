import {
	DEFAULT_CHECKPOINT_STRIDE,
	DynamicSplatWebGpu3dTiledTrainer,
	resolveTiledCapacity,
	TILED_BACKWARD_GRANULARITIES,
	TILED_BACKWARD_MODES,
	TILED_PROJECTION_LAYOUTS,
	TILED_SSIM_LAYOUTS,
} from "./trainerWebGpu3dTiled.js?v=20260814-camera-stress-1";

export function resolveFastTileCapacity(
	initialSplats,
	growthCapacity = null,
	requestedTileCapacity = null,
) {
	if (requestedTileCapacity != null) return Number(requestedTileCapacity);
	const capacity = resolveTiledCapacity(initialSplats, growthCapacity);
	if (capacity <= 8192) return 1024;
	if (capacity <= 16384) return 2048;
	return 4096;
}

// Keep the complete tiled data/objective contract while selecting the measured
// browser-native kernel defaults. Explicit options still let the benchmark lab
// falsify each choice without creating another trainer hierarchy.
export class DynamicSplatWebGpu3dTiledFastTrainer extends DynamicSplatWebGpu3dTiledTrainer {
	async init(dataset, options = {}) {
		const initialSplats = options.splatCount ?? this.initialSplatCount;
		return super.init(dataset, {
			backwardGranularity: TILED_BACKWARD_GRANULARITIES.CHECKPOINT_BLOCK,
			checkpointStride: DEFAULT_CHECKPOINT_STRIDE,
			tileSize: 8,
			tileCapacity: resolveFastTileCapacity(
				initialSplats,
				options.growthCapacity,
				options.tileCapacity,
			),
			...options,
			backwardMode: TILED_BACKWARD_MODES.STAGED_PROJECT_3D,
			projectionLayout: TILED_PROJECTION_LAYOUTS.SPLIT_COMPACT,
			ssimLayout: TILED_SSIM_LAYOUTS.SEPARABLE,
		});
	}
}
