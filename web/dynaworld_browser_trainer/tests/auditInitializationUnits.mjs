// Diagnostic: changing reconstruction units must not change projected geometry.
// Run from the repository root; this uses no GPU or target/heldout RGB.
import { readFileSync, writeFileSync } from "node:fs";
import {
	makeInitialSplats, normalizeDatasetGeometry, projectAnisotropicGaussianCpu,
} from "../trainerWebGpu3d.js";

const root = new URL("../", import.meta.url);
const median = (values) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
const scenes = ["deep3d_mask_eval_jump_train9_holdout1", "coffee_martini_train17_holdout1"];
const rows = [];
for (const scene of scenes) {
	const bundle = JSON.parse(readFileSync(new URL(`${scene}.json`, root), "utf8"));
	let reference;
	for (const sourceUnitMultiplier of [1, 0.1, 10]) {
		const dataset = normalizeDatasetGeometry({
			datasetContract: bundle.dataset_contract,
			seedPointCount: bundle.seed_points_xyzrgb.length,
			seedPoints: Float32Array.from(bundle.seed_points_xyzrgb.flatMap((point) =>
				point.map((value, axis) => axis < 3 ? value * sourceUnitMultiplier : value))),
			cameras: bundle.cameras.map((camera) => ({
				name: camera.name, role: camera.role, intrinsics: camera.intrinsics,
				worldToCamera: Float32Array.from(camera.world_to_camera.flat().map((value, index) =>
					[3, 7, 11].includes(index) ? value * sourceUnitMultiplier : value)),
			})),
		});
		const params = makeInitialSplats(dataset, 4096);
		const camera = dataset.cameras.find((item) => item.name === bundle.dataset_contract.anchor_camera);
		const radii = []; const aspects = []; const transmittance = new Float64Array(96 * 72).fill(1);
		let allAxesAtCap = 0; let maxNormalizedCenterError = 0;
		const alpha = 0.1; const threshold = 1 / 255;
		const supportLimit = -2 * Math.log(threshold / alpha);
		for (let index = 0; index < 4096; index += 1) {
			const base = index * 24;
			const scales = [12, 13, 14].map((axis) => Math.exp(params[base + axis]));
			if (scales.every((value) => Math.abs(value / (0.6 * dataset.trainingSceneScale) - 1) < 1e-5)) allAxesAtCap += 1;
			aspects.push(Math.max(...scales) / Math.min(...scales));
			if (reference) for (let axis = 0; axis < 3; axis += 1) {
				maxNormalizedCenterError = Math.max(maxNormalizedCenterError,
					Math.abs(params[base + axis] - reference[base + axis]));
			}
			const projection = projectAnisotropicGaussianCpu({
				center: Array.from(params.subarray(base, base + 3)),
				logScales: Array.from(params.subarray(base + 12, base + 15)),
				quaternion: Array.from(params.subarray(base + 16, base + 20)),
				camera, aspect: 4 / 3, height: 72,
			});
			if (!projection.valid) continue;
			radii.push(Math.sqrt((projection.covariance[0] + projection.covariance[2]) / 2) * 72);
			const [cx, cy] = projection.center.map((value) => value * 72);
			const rx = Math.sqrt(supportLimit * projection.covariance[0]) * 72;
			const ry = Math.sqrt(supportLimit * projection.covariance[2]) * 72;
			for (let y = Math.max(0, Math.floor(cy - ry)); y <= Math.min(71, Math.ceil(cy + ry)); y += 1) {
				for (let x = Math.max(0, Math.floor(cx - rx)); x <= Math.min(95, Math.ceil(cx + rx)); x += 1) {
					const dx = (x + 0.5 - cx) / 72; const dy = (y + 0.5 - cy) / 72;
					const [a, b, c] = projection.conic;
					const q = a * dx * dx + 2 * b * dx * dy + c * dy * dy;
					if (q <= supportLimit) transmittance[y * 96 + x] *= 1 - alpha * Math.exp(-0.5 * q);
				}
			}
		}
		rows.push({ scene, sourceUnitMultiplier, geometryScale: dataset.geometryScale, trainingSceneScale: dataset.trainingSceneScale,
			maxNormalizedCenterError, allAxesAtCap, medianAspect: median(aspects),
			medianProjectedSigmaPixels: median(radii),
			meanComposedAlpha: transmittance.reduce((sum, value) => sum + 1 - value, 0) / transmittance.length,
			pixelFractionAlphaAboveHalf: transmittance.filter((value) => value < 0.5).length / transmittance.length,
			tiledMaxScale: dataset.trainingSceneScale, tiledMaxVelocityComponent: 2 * dataset.trainingSceneScale });
		reference ??= params;
	}
}
const report = {
	kind: "initialization-source-unit-diagnostic", width: 96, height: 72, splats: 4096,
	assumptions: "Anchor camera; CPU projected conics; alpha=0.1; no temporal gate or Mip opacity compensation. Alpha is order independent. No RGB quality or GPU timing claim.",
	rows,
};
const output = `${JSON.stringify(report, null, 2)}\n`;
if (process.argv[2]) writeFileSync(process.argv[2], output);
process.stdout.write(output);
