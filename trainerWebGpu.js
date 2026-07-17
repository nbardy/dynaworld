const SPLAT_FLOATS = 12;
const SPLAT_BYTES = SPLAT_FLOATS * 4;

function createSeededRandom(seed = 17) {
	let state = seed >>> 0;
	return () => {
		state = (1664525 * state + 1013904223) >>> 0;
		return state / 0x100000000;
	};
}

function sampleDatasetColor(dataset, frame, x, y) {
	const px = Math.min(dataset.width - 1, Math.max(0, Math.floor(x * dataset.width)));
	const py = Math.min(dataset.height - 1, Math.max(0, Math.floor(y * dataset.height)));
	const idx = ((frame * dataset.height + py) * dataset.width + px) * 4;
	return [
		dataset.frames[idx],
		dataset.frames[idx + 1],
		dataset.frames[idx + 2],
	];
}

function averageDatasetColor(dataset, x, y) {
	const color = [0, 0, 0];
	for (let frame = 0; frame < dataset.frameCount; frame += 1) {
		const sampled = sampleDatasetColor(dataset, frame, x, y);
		color[0] += sampled[0];
		color[1] += sampled[1];
		color[2] += sampled[2];
	}
	const invFrames = 1 / Math.max(1, dataset.frameCount);
	return [color[0] * invFrames, color[1] * invFrames, color[2] * invFrames];
}

function residualVectorAt(dataset, frame, pixel) {
	const frameBase = (frame * dataset.width * dataset.height + pixel) * 4;
	const bgBase = pixel * 4;
	const dr = dataset.frames[frameBase] - dataset.background[bgBase];
	const dg = dataset.frames[frameBase + 1] - dataset.background[bgBase + 1];
	const db = dataset.frames[frameBase + 2] - dataset.background[bgBase + 2];
	return [dr, dg, db, (dr * dr + dg * dg + db * db) / 3];
}

function motionEnergyAt(dataset, frame, pixel) {
	return residualVectorAt(dataset, frame, pixel)[3];
}

function computeFrameMotionVelocities(dataset) {
	const frameCount = dataset.frameCount;
	const velocities = Array.from({ length: frameCount }, () => [0, 0]);
	if (frameCount <= 1) {
		return velocities;
	}
	const pixelsPerFrame = dataset.width * dataset.height;
	const centroids = [];
	for (let frame = 0; frame < frameCount; frame += 1) {
		let sum = 0;
		let sumX = 0;
		let sumY = 0;
		for (let pixel = 0; pixel < pixelsPerFrame; pixel += 1) {
			const energy = motionEnergyAt(dataset, frame, pixel);
			if (energy <= 0.0006) {
				continue;
			}
			const x = pixel % dataset.width;
			const y = Math.floor(pixel / dataset.width);
			sum += energy;
			sumX += ((x + 0.5) / dataset.width) * energy;
			sumY += ((y + 0.5) / dataset.height) * energy;
		}
		centroids.push(sum > 1e-8 ? [sumX / sum, sumY / sum] : null);
	}
	for (let frame = 0; frame < frameCount; frame += 1) {
		const current = centroids[frame];
		if (!current) {
			continue;
		}
		let previous = null;
		let next = null;
		let previousFrame = frame;
		let nextFrame = frame;
		for (let f = frame - 1; f >= 0; f -= 1) {
			if (centroids[f]) {
				previous = centroids[f];
				previousFrame = f;
				break;
			}
		}
		for (let f = frame + 1; f < frameCount; f += 1) {
			if (centroids[f]) {
				next = centroids[f];
				nextFrame = f;
				break;
			}
		}
		if (previous && next) {
			const dt = frameTime(nextFrame, frameCount) - frameTime(previousFrame, frameCount);
			velocities[frame] = dt > 1e-6 ? [(next[0] - previous[0]) / dt, (next[1] - previous[1]) / dt] : [0, 0];
		} else if (next) {
			const dt = frameTime(nextFrame, frameCount) - frameTime(frame, frameCount);
			velocities[frame] = dt > 1e-6 ? [(next[0] - current[0]) / dt, (next[1] - current[1]) / dt] : [0, 0];
		} else if (previous) {
			const dt = frameTime(frame, frameCount) - frameTime(previousFrame, frameCount);
			velocities[frame] = dt > 1e-6 ? [(current[0] - previous[0]) / dt, (current[1] - previous[1]) / dt] : [0, 0];
		}
	}
	return velocities;
}

function estimateLocalMotionVelocity(dataset, frame, pixel, fallbackVelocity) {
	if (dataset.frameCount <= 1 || pixel < 0) {
		return fallbackVelocity;
	}
	const currentResidual = residualVectorAt(dataset, frame, pixel);
	if (currentResidual[3] <= 0.0006) {
		return fallbackVelocity;
	}
	const currentX = pixel % dataset.width;
	const currentY = Math.floor(pixel / dataset.width);
	const searchRadius = 7;
	const matchFrame = (targetFrame) => {
		let bestPixel = -1;
		let bestScore = Number.POSITIVE_INFINITY;
		const minY = Math.max(0, currentY - searchRadius);
		const maxY = Math.min(dataset.height - 1, currentY + searchRadius);
		const minX = Math.max(0, currentX - searchRadius);
		const maxX = Math.min(dataset.width - 1, currentX + searchRadius);
		for (let y = minY; y <= maxY; y += 1) {
			for (let x = minX; x <= maxX; x += 1) {
				const candidatePixel = y * dataset.width + x;
				const candidateResidual = residualVectorAt(dataset, targetFrame, candidatePixel);
				if (candidateResidual[3] <= 0.00045) {
					continue;
				}
				const dr = candidateResidual[0] - currentResidual[0];
				const dg = candidateResidual[1] - currentResidual[1];
				const db = candidateResidual[2] - currentResidual[2];
				const colorCost = (dr * dr + dg * dg + db * db) / 3;
				const dx = x - currentX;
				const dy = y - currentY;
				const spatialCost = (dx * dx + dy * dy) / (searchRadius * searchRadius);
				const energyReward = Math.min(candidateResidual[3], currentResidual[3]);
				const score = colorCost + 0.018 * spatialCost - 0.025 * energyReward;
				if (score < bestScore) {
					bestScore = score;
					bestPixel = candidatePixel;
				}
			}
		}
		return bestPixel >= 0 ? bestPixel : null;
	};
	let previous = null;
	let next = null;
	let previousFrame = frame;
	let nextFrame = frame;
	for (let f = frame - 1; f >= 0; f -= 1) {
		previous = matchFrame(f);
		if (previous != null) {
			previousFrame = f;
			break;
		}
	}
	for (let f = frame + 1; f < dataset.frameCount; f += 1) {
		next = matchFrame(f);
		if (next != null) {
			nextFrame = f;
			break;
		}
	}
	const pixelToPoint = (matchedPixel) => [
		((matchedPixel % dataset.width) + 0.5) / dataset.width,
		(Math.floor(matchedPixel / dataset.width) + 0.5) / dataset.height,
	];
	const currentPoint = pixelToPoint(pixel);
	let localVelocity = null;
	if (previous != null && next != null) {
		const previousPoint = pixelToPoint(previous);
		const nextPoint = pixelToPoint(next);
		const dt = frameTime(nextFrame, dataset.frameCount) - frameTime(previousFrame, dataset.frameCount);
		localVelocity = dt > 1e-6 ? [(nextPoint[0] - previousPoint[0]) / dt, (nextPoint[1] - previousPoint[1]) / dt] : null;
	} else if (next != null) {
		const nextPoint = pixelToPoint(next);
		const dt = frameTime(nextFrame, dataset.frameCount) - frameTime(frame, dataset.frameCount);
		localVelocity = dt > 1e-6 ? [(nextPoint[0] - currentPoint[0]) / dt, (nextPoint[1] - currentPoint[1]) / dt] : null;
	} else if (previous != null) {
		const previousPoint = pixelToPoint(previous);
		const dt = frameTime(frame, dataset.frameCount) - frameTime(previousFrame, dataset.frameCount);
		localVelocity = dt > 1e-6 ? [(currentPoint[0] - previousPoint[0]) / dt, (currentPoint[1] - previousPoint[1]) / dt] : null;
	}
	if (!localVelocity) {
		return fallbackVelocity;
	}
	return [
		localVelocity[0] * 0.75 + fallbackVelocity[0] * 0.25,
		localVelocity[1] * 0.75 + fallbackVelocity[1] * 0.25,
	];
}

function sigmoid(x) {
	return 1 / (1 + Math.exp(-x));
}

function frameTime(frame, frameCount) {
	return frameCount <= 1 ? 0 : frame / (frameCount - 1);
}

function temporalGateCpu(timeCenter, t, temporalSigma) {
	const sigma = Math.min(0.36, Math.max(0.12, temporalSigma));
	const floor = Math.min(0.12, Math.max(0.035, sigma * 0.30));
	const dt = t - Math.min(1, Math.max(0, timeCenter));
	const gate = Math.exp(-0.5 * dt * dt / (sigma * sigma));
	return floor + (1 - floor) * gate;
}

function splatCenterCpu(params, base, t, modelMode) {
	const tc = t * 2 - 1;
	const x = params[base] + params[base + 4] * tc;
	const y = params[base + 1] + params[base + 5] * tc;
	if (modelMode !== 0) {
		return [x, y];
	}
	const angle = t * Math.PI * 2;
	return [
		x + params[base + 6] * Math.sin(angle),
		y + params[base + 7] * Math.cos(angle),
	];
}

function evalModelCpu(dataset, params, splatCount, px, py, frame, modelMode, temporalSigma) {
	const pixel = frame.pixel;
	let r = dataset.background[pixel * 4];
	let g = dataset.background[pixel * 4 + 1];
	let b = dataset.background[pixel * 4 + 2];
	const aspect = Math.max(0.25, dataset.width / Math.max(1, dataset.height));
	const t = frame.time;
	for (let j = 0; j < splatCount; j += 1) {
		const base = j * SPLAT_FLOATS;
		const [cx, cy] = splatCenterCpu(params, base, t, modelMode);
		const radius = Math.min(0.09, Math.max(0.009, params[base + 3]));
		const dx = (px - cx) * aspect;
		const dy = py - cy;
		const dist2 = dx * dx + dy * dy;
		const r2 = radius * radius;
		if (dist2 > 9 * r2) {
			continue;
		}
		const gaussian = Math.exp(-0.5 * dist2 / r2);
		const alpha = sigmoid(params[base + 11]) * gaussian * temporalGateCpu(params[base + 2], t, temporalSigma);
		r = r * (1 - alpha) + params[base + 8] * alpha;
		g = g * (1 - alpha) + params[base + 9] * alpha;
		b = b * (1 - alpha) + params[base + 10] * alpha;
	}
	return [r, g, b];
}

function evalDynamicCoverageCpu(dataset, params, splatCount, px, py, frame, modelMode, temporalSigma) {
	const aspect = Math.max(0.25, dataset.width / Math.max(1, dataset.height));
	const t = frame.time;
	let transmittance = 1;
	let maxAlpha = 0;
	for (let j = 0; j < splatCount; j += 1) {
		const base = j * SPLAT_FLOATS;
		const [cx, cy] = splatCenterCpu(params, base, t, modelMode);
		const radius = Math.min(0.09, Math.max(0.009, params[base + 3]));
		const dx = (px - cx) * aspect;
		const dy = py - cy;
		const dist2 = dx * dx + dy * dy;
		const r2 = radius * radius;
		if (dist2 > 9 * r2) {
			continue;
		}
		const gaussian = Math.exp(-0.5 * dist2 / r2);
		const alpha = sigmoid(params[base + 11]) * gaussian * temporalGateCpu(params[base + 2], t, temporalSigma);
		maxAlpha = Math.max(maxAlpha, alpha);
		transmittance *= 1 - alpha;
	}
	return {
		coverage: 1 - transmittance,
		maxAlpha,
	};
}

function computeSplatHealth(params, splatCount) {
	let opacitySum = 0;
	let radiusSum = 0;
	let motionSum = 0;
	let activeSplats = 0;
	for (let i = 0; i < splatCount; i += 1) {
		const base = i * SPLAT_FLOATS;
		const opacity = sigmoid(params[base + 11]);
		const radius = params[base + 3];
		const vx = params[base + 4];
		const vy = params[base + 5];
		opacitySum += opacity;
		radiusSum += radius;
		motionSum += Math.hypot(vx, vy);
		if (opacity > 0.05) {
			activeSplats += 1;
		}
	}
	const inv = 1 / Math.max(1, splatCount);
	return {
		activeSplats,
		meanOpacity: opacitySum * inv,
		meanRadius: radiusSum * inv,
		meanMotion: motionSum * inv,
	};
}

function computeGridValidationMetrics(dataset, params, splatCount, { modelMode = 0, temporalSigma = 0.30, gridSize = 32 } = {}) {
	const gridX = Math.min(dataset.width, Math.max(1, Math.round(gridSize)));
	const gridY = Math.min(dataset.height, Math.max(1, Math.round(gridSize * dataset.height / Math.max(1, dataset.width))));
	let loss = 0;
	let absError = 0;
	let gridWeightedMotionLoss = 0;
	let motionWeight = 0;
	let predLumaSum = 0;
	let targetLumaSum = 0;
	let predLumaSqSum = 0;
	let targetLumaSqSum = 0;
	let lumaProductSum = 0;
	let staticCoverageSum = 0;
	let staticCoverageCount = 0;
	let count = 0;
	for (let f = 0; f < dataset.frameCount; f += 1) {
		const frame = {
			index: f,
			time: frameTime(f, dataset.frameCount),
			pixel: 0,
		};
		for (let gy = 0; gy < gridY; gy += 1) {
			const y = Math.min(dataset.height - 1, Math.max(0, Math.floor((gy + 0.5) * dataset.height / gridY)));
			const py = (y + 0.5) / dataset.height;
			for (let gx = 0; gx < gridX; gx += 1) {
				const x = Math.min(dataset.width - 1, Math.max(0, Math.floor((gx + 0.5) * dataset.width / gridX)));
				const px = (x + 0.5) / dataset.width;
				frame.pixel = y * dataset.width + x;
				const [r, g, b] = evalModelCpu(dataset, params, splatCount, px, py, frame, modelMode, temporalSigma);
				const targetBase = ((f * dataset.height + y) * dataset.width + x) * 4;
				const dr = r - dataset.frames[targetBase];
				const dg = g - dataset.frames[targetBase + 1];
				const db = b - dataset.frames[targetBase + 2];
				const mse = (dr * dr + dg * dg + db * db) / 3;
				absError += (Math.abs(dr) + Math.abs(dg) + Math.abs(db)) / 3;
				const predLuma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
				const targetLuma = 0.2126 * dataset.frames[targetBase]
					+ 0.7152 * dataset.frames[targetBase + 1]
					+ 0.0722 * dataset.frames[targetBase + 2];
				predLumaSum += predLuma;
				targetLumaSum += targetLuma;
				predLumaSqSum += predLuma * predLuma;
				targetLumaSqSum += targetLuma * targetLuma;
				lumaProductSum += predLuma * targetLuma;
				const bgBase = frame.pixel * 4;
				const tr = dataset.frames[targetBase] - dataset.background[bgBase];
				const tg = dataset.frames[targetBase + 1] - dataset.background[bgBase + 1];
				const tb = dataset.frames[targetBase + 2] - dataset.background[bgBase + 2];
				const motionEnergy = (tr * tr + tg * tg + tb * tb) / 3;
				if (motionEnergy < 0.00045 && ((f + gx + gy) % 4) === 0) {
					const coverage = evalDynamicCoverageCpu(dataset, params, splatCount, px, py, frame, modelMode, temporalSigma);
					staticCoverageSum += coverage.coverage;
					staticCoverageCount += 1;
				}
				loss += mse;
				gridWeightedMotionLoss += mse * motionEnergy;
				motionWeight += motionEnergy;
				count += 1;
			}
		}
	}
	let motionSampleLoss = 0;
	let motionCoverageSum = 0;
	let motionCoverage50 = 0;
	let motionMaxAlphaSum = 0;
	const motionSamples = dataset.motionSamples ?? new Uint32Array(0);
	const motionCount = Math.min(motionSamples.length, 4096);
	for (let i = 0; i < motionCount; i += 1) {
		const packed = motionSamples[Math.floor((i + 0.5) * motionSamples.length / motionCount)];
		const pixel = packed % (dataset.width * dataset.height);
		const frameIndex = Math.min(dataset.frameCount - 1, Math.floor(packed / (dataset.width * dataset.height)));
		const x = pixel % dataset.width;
		const y = Math.floor(pixel / dataset.width);
		const px = (x + 0.5) / dataset.width;
		const py = (y + 0.5) / dataset.height;
		const frame = {
			index: frameIndex,
			time: frameTime(frameIndex, dataset.frameCount),
			pixel,
		};
		const [r, g, b] = evalModelCpu(dataset, params, splatCount, px, py, frame, modelMode, temporalSigma);
		const targetBase = (frameIndex * dataset.width * dataset.height + pixel) * 4;
		const dr = r - dataset.frames[targetBase];
		const dg = g - dataset.frames[targetBase + 1];
		const db = b - dataset.frames[targetBase + 2];
		motionSampleLoss += (dr * dr + dg * dg + db * db) / 3;
		const coverage = evalDynamicCoverageCpu(dataset, params, splatCount, px, py, frame, modelMode, temporalSigma);
		motionCoverageSum += coverage.coverage;
		motionMaxAlphaSum += coverage.maxAlpha;
		if (coverage.coverage >= 0.50) {
			motionCoverage50 += 1;
		}
	}
	const gridLoss = loss / Math.max(1, count);
	const gridMae = absError / Math.max(1, count);
	const gridPsnr = gridLoss > 1e-12 ? -10 * Math.log10(gridLoss) : 99;
	const invCount = 1 / Math.max(1, count);
	const predMean = predLumaSum * invCount;
	const targetMean = targetLumaSum * invCount;
	const predVar = Math.max(0, predLumaSqSum * invCount - predMean * predMean);
	const targetVar = Math.max(0, targetLumaSqSum * invCount - targetMean * targetMean);
	const covariance = lumaProductSum * invCount - predMean * targetMean;
	const c1 = 0.01 * 0.01;
	const c2 = 0.03 * 0.03;
	const gridSsim = ((2 * predMean * targetMean + c1) * (2 * covariance + c2))
		/ ((predMean * predMean + targetMean * targetMean + c1) * (predVar + targetVar + c2));
	const weightedMotionLoss = motionWeight > 1e-8 ? gridWeightedMotionLoss / motionWeight : gridLoss;
	const health = computeSplatHealth(params, splatCount);
	return {
		gridLoss,
		gridMae,
		gridPsnr,
		gridSsim: Math.max(-1, Math.min(1, gridSsim)),
		motionLoss: motionCount > 0 ? motionSampleLoss / motionCount : weightedMotionLoss,
		weightedMotionLoss,
		motionWeight: motionWeight / Math.max(1, count),
		motionSampleCount: motionCount,
			motionCoverage: motionCount > 0 ? motionCoverageSum / motionCount : 0,
			motionCoverage50: motionCount > 0 ? motionCoverage50 / motionCount : 0,
			motionMaxAlpha: motionCount > 0 ? motionMaxAlphaSum / motionCount : 0,
			staticCoverage: staticCoverageCount > 0 ? staticCoverageSum / staticCoverageCount : 0,
			...health,
		};
	}

function makeInitialSplats(dataset, splatCount) {
	const random = createSeededRandom(1701);
	const params = new Float32Array(splatCount * SPLAT_FLOATS);
	const aspect = Math.max(0.25, Math.min(4, dataset.width / Math.max(1, dataset.height)));
	const gridY = Math.max(1, Math.ceil(Math.sqrt(splatCount / aspect)));
	const gridX = Math.max(1, Math.ceil(splatCount / gridY));
	const baseRadius = Math.min(0.055, Math.max(0.014, 0.42 / gridY));
	const motionSamples = dataset.motionSamples ?? new Uint32Array(0);
	const motionSeedCount = Math.min(motionSamples.length, Math.floor(splatCount * 0.48));
	const motionSeedStart = splatCount - motionSeedCount;
	const pixelsPerFrame = dataset.width * dataset.height;
	const frameVelocities = computeFrameMotionVelocities(dataset);
	for (let i = 0; i < splatCount; i += 1) {
		const gx = i % gridX;
		const gy = Math.floor(i / gridX);
		const jitterX = (random() - 0.5) * 0.35 / gridX;
		const jitterY = (random() - 0.5) * 0.35 / gridY;
		let x = (gx + 0.5) / gridX + jitterX;
		let y = (gy + 0.5) / gridY + jitterY;
		const temporalBin = dataset.frameCount <= 1 ? 0 : i % dataset.frameCount;
		let timeCenter = dataset.frameCount <= 1
			? 0
			: Math.min(1, Math.max(0, (temporalBin + 0.5 + (random() - 0.5) * 0.34) / dataset.frameCount));
		let frame = Math.min(
			dataset.frameCount - 1,
			Math.max(0, Math.round(timeCenter * (dataset.frameCount - 1))),
		);
		let radius = baseRadius * (0.86 + random() * 0.34);
		let opacityLogit = -3.2 + random() * 0.4;
		let motionSeedPixel = -1;
		if (i >= motionSeedStart && motionSeedCount > 0) {
			const motionIndex = i - motionSeedStart;
			const sampleIndex = Math.min(
				motionSamples.length - 1,
				Math.floor(((motionIndex + 0.35 * random()) / motionSeedCount) * motionSamples.length),
			);
			const packed = motionSamples[sampleIndex];
			frame = Math.min(dataset.frameCount - 1, Math.floor(packed / pixelsPerFrame));
			motionSeedPixel = packed % pixelsPerFrame;
			const px = motionSeedPixel % dataset.width;
			const py = Math.floor(motionSeedPixel / dataset.width);
			x = Math.min(0.995, Math.max(0.005, (px + 0.5 + (random() - 0.5) * 0.45) / dataset.width));
			y = Math.min(0.995, Math.max(0.005, (py + 0.5 + (random() - 0.5) * 0.45) / dataset.height));
			timeCenter = frameTime(frame, dataset.frameCount);
			if (dataset.frameCount > 1) {
				timeCenter = Math.min(1, Math.max(0, timeCenter + (random() - 0.5) * 0.10 / (dataset.frameCount - 1)));
			}
			radius = baseRadius * (0.74 + random() * 0.34);
			opacityLogit = -2.10 + random() * 0.35;
		}
		const localColor = sampleDatasetColor(dataset, frame, x, y);
		const averageColor = averageDatasetColor(dataset, x, y);
		let linearMotionX = (random() - 0.5) * 0.025;
		let linearMotionY = (random() - 0.5) * 0.025;
		let harmonicMotionX = (random() - 0.5) * 0.015;
		let harmonicMotionY = (random() - 0.5) * 0.015;
		if (i >= motionSeedStart && motionSeedCount > 0) {
			const velocity = estimateLocalMotionVelocity(dataset, frame, motionSeedPixel, frameVelocities[frame] ?? [0, 0]);
			linearMotionX = Math.min(0.10, Math.max(-0.10, velocity[0] * 0.5)) + (random() - 0.5) * 0.008;
			linearMotionY = Math.min(0.10, Math.max(-0.10, velocity[1] * 0.5)) + (random() - 0.5) * 0.008;
			harmonicMotionX *= 0.5;
			harmonicMotionY *= 0.5;
		}
		const tc = timeCenter * 2 - 1;
		const angle = timeCenter * Math.PI * 2;
		const baseX = x - linearMotionX * tc - harmonicMotionX * Math.sin(angle);
		const baseY = y - linearMotionY * tc - harmonicMotionY * Math.cos(angle);
		const base = i * SPLAT_FLOATS;
		params[base] = Math.min(0.98, Math.max(0.02, baseX));
		params[base + 1] = Math.min(0.98, Math.max(0.02, baseY));
		params[base + 2] = timeCenter;
		params[base + 3] = radius;
		params[base + 4] = linearMotionX;
		params[base + 5] = linearMotionY;
		params[base + 6] = harmonicMotionX;
		params[base + 7] = harmonicMotionY;
		params[base + 8] = Math.min(1.35, localColor[0] * 0.78 + averageColor[0] * 0.24 + random() * 0.02);
		params[base + 9] = Math.min(1.35, localColor[1] * 0.78 + averageColor[1] * 0.24 + random() * 0.02);
		params[base + 10] = Math.min(1.35, localColor[2] * 0.78 + averageColor[2] * 0.24 + random() * 0.02);
		params[base + 11] = opacityLogit;
	}
	return params;
}

function writeTrainConfig(buffer, values) {
	const view = new DataView(buffer);
	const staticSampleRate = values.staticSampleCount > 0
		? Math.min(0.20, Math.max(0, values.staticSampleRate ?? 0.08))
		: 0;
	const motionSampleRate = Math.min(
		1 - staticSampleRate,
		Math.max(0, values.motionSampleRate ?? 0.95),
	);
	view.setUint32(0, values.width, true);
	view.setUint32(4, values.height, true);
	view.setUint32(8, values.frameCount, true);
	view.setUint32(12, values.splatCount, true);
	view.setUint32(16, values.sampleCount, true);
	view.setUint32(20, values.step, true);
	view.setUint32(24, values.modelMode ?? 0, true);
	view.setUint32(28, values.motionSampleCount ?? 0, true);
	view.setFloat32(32, values.lrPos, true);
	view.setFloat32(36, values.lrColor, true);
	view.setFloat32(40, values.lrOpacity, true);
	view.setFloat32(44, values.lrMotion, true);
	view.setFloat32(48, values.minRadius, true);
	view.setFloat32(52, values.maxRadius, true);
	view.setFloat32(56, values.temporalSigma, true);
	view.setFloat32(60, values.targetAspect, true);
	view.setUint32(64, Math.round(motionSampleRate * 1000), true);
	view.setFloat32(68, values.motionCoverageTarget ?? 0.44, true);
	view.setFloat32(72, values.motionCoverageWeight ?? 0.08, true);
	view.setFloat32(76, values.staticAlphaWeight ?? 4.0, true);
	view.setFloat32(80, values.opacityDecayWeight ?? 0.025, true);
	view.setFloat32(84, values.staticEnergyThreshold ?? 0.00045, true);
	view.setUint32(88, values.staticSampleCount ?? 0, true);
	view.setUint32(92, Math.round(staticSampleRate * 1000), true);
	view.setFloat32(96, values.beta1 ?? 0.9, true);
	view.setFloat32(100, values.beta2 ?? 0.99, true);
	view.setFloat32(104, values.adamEpsilon ?? 1e-6, true);
	view.setFloat32(108, values.statDecay ?? 0.95, true);
	view.setFloat32(112, values.robustMix ?? 0.20, true);
}

function writeRenderConfig(buffer, values) {
	const view = new DataView(buffer);
	view.setFloat32(0, values.width, true);
	view.setFloat32(4, values.height, true);
	view.setFloat32(8, values.time, true);
	view.setFloat32(12, values.splatCount, true);
	view.setFloat32(16, values.pointScale, true);
	view.setFloat32(20, values.modelMode ?? 0, true);
	view.setFloat32(24, values.targetAspect, true);
	view.setFloat32(28, values.temporalSigma, true);
	view.setFloat32(32, values.targetWidth, true);
	view.setFloat32(36, values.targetHeight, true);
	view.setFloat32(40, values.renderMode ?? 0, true);
	view.setFloat32(44, 0, true);
}

const TRAIN_WGSL = `
	struct Splat {
		posRadius: vec4<f32>,
		motion: vec4<f32>,
		colorOpacity: vec4<f32>,
	};

	struct TrainConfig {
		width: u32,
		height: u32,
		frameCount: u32,
		splatCount: u32,
		sampleCount: u32,
		step: u32,
		modelMode: u32,
		motionSampleCount: u32,
		lrPos: f32,
		lrColor: f32,
		lrOpacity: f32,
		lrMotion: f32,
		minRadius: f32,
		maxRadius: f32,
		temporalSigma: f32,
		targetAspect: f32,
		motionSamplePermil: u32,
		motionCoverageTarget: f32,
		motionCoverageWeight: f32,
		staticAlphaWeight: f32,
		opacityDecayWeight: f32,
		staticEnergyThreshold: f32,
		staticSampleCount: u32,
		staticSamplePermil: u32,
		beta1: f32,
		beta2: f32,
		adamEpsilon: f32,
		statDecay: f32,
		robustMix: f32,
	};

	@group(0) @binding(0) var<uniform> cfg: TrainConfig;
	@group(0) @binding(1) var<storage, read> paramsIn: array<Splat>;
	@group(0) @binding(2) var<storage, read_write> paramsOut: array<Splat>;
	@group(0) @binding(3) var<storage, read> targetFrames: array<vec4<f32>>;
	@group(0) @binding(4) var<storage, read_write> metrics: array<f32>;
	@group(0) @binding(5) var<storage, read> staticBackground: array<vec4<f32>>;
	@group(0) @binding(6) var<storage, read> motionSamples: array<u32>;
	@group(0) @binding(7) var<storage, read> staticSamples: array<u32>;
	@group(0) @binding(8) var<storage, read_write> firstMoment: array<Splat>;
	@group(0) @binding(9) var<storage, read_write> secondMoment: array<Splat>;
	@group(0) @binding(10) var<storage, read_write> splatStats: array<vec4<f32>>;

	fn hash_u32(v: u32) -> u32 {
		var x = v;
		x = ((x >> 16u) ^ x) * 0x7feb352du;
		x = ((x >> 15u) ^ x) * 0x846ca68bu;
		x = (x >> 16u) ^ x;
		return x;
	}

	fn sigmoid(x: f32) -> f32 {
		return 1.0 / (1.0 + exp(-x));
	}

	fn frame_time(frame: u32) -> f32 {
		if (cfg.frameCount <= 1u) {
			return 0.0;
		}
		return f32(frame) / f32(cfg.frameCount - 1u);
	}

	fn tube_center(p: Splat, t: f32) -> vec2<f32> {
		let tc = t * 2.0 - 1.0;
		if (cfg.modelMode != 0u) {
			return p.posRadius.xy + p.motion.xy * tc;
		}
		let wave = sin(t * 6.28318530718);
		let orbit = cos(t * 6.28318530718);
		return p.posRadius.xy + p.motion.xy * tc + vec2<f32>(p.motion.z * wave, p.motion.w * orbit);
	}

	fn metric_delta(point: vec2<f32>, center: vec2<f32>) -> vec2<f32> {
		let aspect = max(0.25, cfg.targetAspect);
		let d = point - center;
		return vec2<f32>(d.x * aspect, d.y);
	}

		fn temporal_gate(p: Splat, t: f32) -> f32 {
			let sigma = clamp(cfg.temporalSigma, 0.12, 0.36);
			let floor = clamp(sigma * 0.30, 0.035, 0.12);
			let dt = t - clamp(p.posRadius.z, 0.0, 1.0);
			let gate = exp(-0.5 * dt * dt / (sigma * sigma));
			return floor + (1.0 - floor) * gate;
	}

	struct CompositeEval {
		pred: vec3<f32>,
		under: vec3<f32>,
		suffixTransmittance: f32,
		currentAlpha: f32,
		coverage: f32,
	};

	fn eval_model(pixel: u32, px: f32, py: f32, t: f32, current: u32) -> CompositeEval {
		let bg = staticBackground[pixel].xyz;
		var accum = bg;
		var under = bg;
		var suffixTransmittance = 1.0;
		var transmittance = 1.0;
		var currentAlpha = 0.0;
		for (var j = 0u; j < cfg.splatCount; j = j + 1u) {
			let p = paramsIn[j];
			if (j == current) {
				under = accum;
			}
			let c = tube_center(p, t);
			let radius = clamp(p.posRadius.w, cfg.minRadius, cfg.maxRadius);
			let d = metric_delta(vec2<f32>(px, py), c);
			let dist2 = dot(d, d);
			let r2 = radius * radius;
			var alpha = 0.0;
			if (dist2 <= 9.0 * r2) {
				let g = exp(-0.5 * dist2 / r2);
				let opacity = sigmoid(p.colorOpacity.w);
				alpha = opacity * g * temporal_gate(p, t);
			}
			if (j == current) {
				currentAlpha = alpha;
			}
			if (j > current) {
				suffixTransmittance = suffixTransmittance * (1.0 - alpha);
			}
			transmittance = transmittance * (1.0 - alpha);
			accum = accum * (1.0 - alpha) + p.colorOpacity.xyz * alpha;
		}
		return CompositeEval(accum, under, suffixTransmittance, currentAlpha, 1.0 - transmittance);
	}

	@compute @workgroup_size(1)
	fn train(@builtin(global_invocation_id) gid: vec3<u32>) {
		let i = gid.x;
		if (i >= cfg.splatCount) {
			return;
		}

		var p = paramsIn[i];
		var gradColor = vec3<f32>(0.0);
		var gradOpacity = 0.0;
		var gradCenter = vec2<f32>(0.0);
		var gradMotion = vec4<f32>(0.0);
		var gradRadius = 0.0;
		var gradTimeCenter = 0.0;
		var meanAlpha = 0.0;
		var loss = 0.0;
		let sampleNorm = 1.0 / max(1.0, f32(cfg.sampleCount));

		for (var s = 0u; s < cfg.sampleCount; s = s + 1u) {
			let seed = hash_u32(cfg.step * 747796405u + s * 2891336453u + 277803737u);
			var frame = seed % cfg.frameCount;
			let pixelSeed = hash_u32(seed + 1013904223u);
			var pixel = pixelSeed % (cfg.width * cfg.height);
			var usedMotionSample = false;
			let sampleBucket = hash_u32(seed + 1664525u) % 1000u;
			if (cfg.motionSampleCount > 0u && sampleBucket < cfg.motionSamplePermil) {
				let packed = motionSamples[hash_u32(seed + 22695477u) % cfg.motionSampleCount];
				frame = packed / (cfg.width * cfg.height);
				pixel = packed % (cfg.width * cfg.height);
				usedMotionSample = true;
			} else if (cfg.staticSampleCount > 0u && sampleBucket < cfg.motionSamplePermil + cfg.staticSamplePermil) {
				let packed = staticSamples[hash_u32(seed + 374761393u) % cfg.staticSampleCount];
				frame = packed / (cfg.width * cfg.height);
				pixel = packed % (cfg.width * cfg.height);
			}
			let x = pixel % cfg.width;
			let y = pixel / cfg.width;
			let px = (f32(x) + 0.5) / f32(cfg.width);
			let py = (f32(y) + 0.5) / f32(cfg.height);
			let t = frame_time(frame);
				let targetRgb = targetFrames[frame * cfg.width * cfg.height + pixel].xyz;
				let bg = staticBackground[pixel].xyz;
				let targetResidual = targetRgb - bg;
				let targetMotionEnergy = dot(targetResidual, targetResidual) / 3.0;
				let eval = eval_model(pixel, px, py, t, i);
			let pred = eval.pred;
			let err = pred - targetRgb;
			let robustDenom = max(vec3<f32>(1e-4), abs(err));
			let l1Grad = err / robustDenom / 3.0;
			let mseGrad = err * (2.0 / 3.0);
			let dLossDPred = mix(mseGrad, l1Grad, cfg.robustMix) * sampleNorm;

			if (i == 0u) {
				loss = loss + dot(err, err) / 3.0;
			}

			let center = tube_center(p, t);
			let radius = clamp(p.posRadius.w, cfg.minRadius, cfg.maxRadius);
			let rawD = vec2<f32>(px, py) - center;
			let d = metric_delta(vec2<f32>(px, py), center);
			let r2 = radius * radius;
			let dist2 = dot(d, d);
			if (dist2 > 9.0 * r2) {
				continue;
			}
			let g = exp(-0.5 * dist2 / r2);
			let opacity = sigmoid(p.colorOpacity.w);
			let timeWeight = temporal_gate(p, t);
			let alphaWeight = opacity * g * timeWeight;
			meanAlpha = meanAlpha + alphaWeight * sampleNorm;
			let color = p.colorOpacity.xyz;
			let colorDot = dot(dLossDPred, (color - eval.under) * eval.suffixTransmittance);
			var alphaGrad = colorDot;
			if (cfg.staticAlphaWeight > 0.0 && targetMotionEnergy < cfg.staticEnergyThreshold) {
				let lowMotionWeight = 1.0 - smoothstep(cfg.staticEnergyThreshold * 0.25, cfg.staticEnergyThreshold, targetMotionEnergy);
				alphaGrad = alphaGrad + 2.0 * cfg.staticAlphaWeight * lowMotionWeight * eval.currentAlpha * sampleNorm;
			}
			let coverageError = eval.coverage - cfg.motionCoverageTarget;
			if (usedMotionSample && cfg.motionCoverageWeight > 0.0 && coverageError < 0.0) {
				let dCoverageDAlpha = (1.0 - eval.coverage) / max(1e-3, 1.0 - eval.currentAlpha);
				alphaGrad = alphaGrad + 2.0 * cfg.motionCoverageWeight * coverageError * dCoverageDAlpha * sampleNorm;
			}
			let gradCommon = alphaGrad * opacity * g * timeWeight;
			let centerGrad = gradCommon * vec2<f32>(rawD.x * cfg.targetAspect * cfg.targetAspect, rawD.y) / r2;
			let tc = t * 2.0 - 1.0;
			let wave = sin(t * 6.28318530718);
			let orbit = cos(t * 6.28318530718);
			let timeCenter = clamp(p.posRadius.z, 0.0, 1.0);
			let temporalSigma = clamp(cfg.temporalSigma, 0.12, 0.36);
			let temporalFloor = clamp(temporalSigma * 0.30, 0.035, 0.12);
			let temporalRange = 1.0 - temporalFloor;
			let temporalCore = max(0.0, (timeWeight - temporalFloor) / temporalRange);
			let sampleGradTimeCenter = alphaGrad * opacity * g * temporalRange * temporalCore * (t - timeCenter) / (temporalSigma * temporalSigma);

			gradColor = gradColor + dLossDPred * alphaWeight * eval.suffixTransmittance;
			gradOpacity = gradOpacity + alphaGrad * g * timeWeight * opacity * (1.0 - opacity);
			gradCenter = gradCenter + centerGrad;
			if (cfg.modelMode == 0u) {
				gradMotion = gradMotion + vec4<f32>(
					centerGrad.x * tc,
					centerGrad.y * tc,
					centerGrad.x * wave,
					centerGrad.y * orbit
				);
			} else {
				gradMotion = gradMotion + vec4<f32>(centerGrad.x * tc, centerGrad.y * tc, 0.0, 0.0);
			}
			gradRadius = gradRadius + gradCommon * dist2 / max(1e-5, radius * radius * radius);
			gradTimeCenter = gradTimeCenter + sampleGradTimeCenter;
		}

		let opacityForDecay = sigmoid(p.colorOpacity.w);
		gradOpacity = gradOpacity + cfg.opacityDecayWeight * opacityForDecay * (1.0 - opacityForDecay);
		let gradient = Splat(
			vec4<f32>(gradCenter, gradTimeCenter * 0.22, gradRadius * 0.12),
			gradMotion,
			vec4<f32>(gradColor, gradOpacity)
		);
		var m = firstMoment[i];
		var v = secondMoment[i];
		m.posRadius = cfg.beta1 * m.posRadius + (1.0 - cfg.beta1) * gradient.posRadius;
		m.motion = cfg.beta1 * m.motion + (1.0 - cfg.beta1) * gradient.motion;
		m.colorOpacity = cfg.beta1 * m.colorOpacity + (1.0 - cfg.beta1) * gradient.colorOpacity;
		v.posRadius = cfg.beta2 * v.posRadius + (1.0 - cfg.beta2) * gradient.posRadius * gradient.posRadius;
		v.motion = cfg.beta2 * v.motion + (1.0 - cfg.beta2) * gradient.motion * gradient.motion;
		v.colorOpacity = cfg.beta2 * v.colorOpacity + (1.0 - cfg.beta2) * gradient.colorOpacity * gradient.colorOpacity;
		firstMoment[i] = m;
		secondMoment[i] = v;
		let adamStep = f32(cfg.step + 1u);
		let mCorrection = max(1e-6, 1.0 - pow(cfg.beta1, adamStep));
		let vCorrection = max(1e-6, 1.0 - pow(cfg.beta2, adamStep));
		let mHat = Splat(m.posRadius / mCorrection, m.motion / mCorrection, m.colorOpacity / mCorrection);
		let vHat = Splat(v.posRadius / vCorrection, v.motion / vCorrection, v.colorOpacity / vCorrection);
		let posUpdate = mHat.posRadius / (sqrt(vHat.posRadius) + vec4<f32>(cfg.adamEpsilon));
		let motionUpdate = mHat.motion / (sqrt(vHat.motion) + vec4<f32>(cfg.adamEpsilon));
		let colorUpdate = mHat.colorOpacity / (sqrt(vHat.colorOpacity) + vec4<f32>(cfg.adamEpsilon));
		let nextColor = clamp(p.colorOpacity.xyz - cfg.lrColor * colorUpdate.xyz, vec3<f32>(0.0), vec3<f32>(1.6));
		p.colorOpacity.x = nextColor.x;
		p.colorOpacity.y = nextColor.y;
		p.colorOpacity.z = nextColor.z;
		p.colorOpacity.w = clamp(p.colorOpacity.w - cfg.lrOpacity * colorUpdate.w, -7.0, 2.6);
		let nextPos = clamp(p.posRadius.xy - cfg.lrPos * posUpdate.xy, vec2<f32>(-0.25), vec2<f32>(1.25));
		p.posRadius.x = nextPos.x;
		p.posRadius.y = nextPos.y;
		p.motion = clamp(p.motion - cfg.lrMotion * motionUpdate, vec4<f32>(-0.42), vec4<f32>(0.42));
		p.posRadius.z = clamp(p.posRadius.z - cfg.lrMotion * posUpdate.z, 0.0, 1.0);
		p.posRadius.w = clamp(p.posRadius.w - cfg.lrPos * posUpdate.w, cfg.minRadius, cfg.maxRadius);
		paramsOut[i] = p;
		let oldStats = splatStats[i];
		let observedStats = vec4<f32>(length(gradCenter), meanAlpha, abs(gradOpacity), length(gradMotion));
		splatStats[i] = cfg.statDecay * oldStats + (1.0 - cfg.statDecay) * observedStats;

		if (i == 0u) {
			metrics[0] = loss * sampleNorm;
		}
	}
`;

const RENDER_WGSL = `
	struct Splat {
		posRadius: vec4<f32>,
		motion: vec4<f32>,
		colorOpacity: vec4<f32>,
	};

		struct RenderConfig {
		width: f32,
		height: f32,
		time: f32,
		splatCount: f32,
		pointScale: f32,
		modelMode: f32,
		targetAspect: f32,
		temporalSigma: f32,
			targetWidth: f32,
			targetHeight: f32,
			renderMode: f32,
			pad1: f32,
		};

	struct VSOut {
		@builtin(position) pos: vec4<f32>,
		@location(0) local: vec2<f32>,
		@location(1) color: vec3<f32>,
		@location(2) opacity: f32,
	};

		@group(0) @binding(0) var<uniform> cfg: RenderConfig;
		@group(0) @binding(1) var<storage, read> params: array<Splat>;

	fn sigmoid(x: f32) -> f32 {
		return 1.0 / (1.0 + exp(-x));
	}

	fn tube_center(p: Splat, t: f32) -> vec2<f32> {
		let tc = t * 2.0 - 1.0;
		if (cfg.modelMode >= 0.5) {
			return p.posRadius.xy + p.motion.xy * tc;
		}
		let wave = sin(t * 6.28318530718);
		let orbit = cos(t * 6.28318530718);
		return p.posRadius.xy + p.motion.xy * tc + vec2<f32>(p.motion.z * wave, p.motion.w * orbit);
	}

		fn temporal_gate(p: Splat, t: f32) -> f32 {
			let sigma = clamp(cfg.temporalSigma, 0.12, 0.36);
			let floor = clamp(sigma * 0.30, 0.035, 0.12);
			let dt = t - clamp(p.posRadius.z, 0.0, 1.0);
			let gate = exp(-0.5 * dt * dt / (sigma * sigma));
			return floor + (1.0 - floor) * gate;
	}

		fn fit_scale() -> vec2<f32> {
		let canvasAspect = cfg.width / max(1.0, cfg.height);
		let targetAspect = max(0.25, cfg.targetAspect);
		if (canvasAspect > targetAspect) {
			return vec2<f32>(targetAspect / canvasAspect, 1.0);
		}
			return vec2<f32>(1.0, canvasAspect / targetAspect);
		}

		@vertex
		fn vs_main(@builtin(instance_index) iid: u32, @location(0) quad: vec2<f32>) -> VSOut {
		if (f32(iid) >= cfg.splatCount) {
			return VSOut(vec4<f32>(0.0), vec2<f32>(0.0), vec3<f32>(0.0), 0.0);
		}
		let p = params[iid];
		let center = tube_center(p, cfg.time);
		let radius = clamp(p.posRadius.w * cfg.pointScale, 0.002, 0.2);
		let targetAspect = max(0.25, cfg.targetAspect);
		let scale = fit_scale();
		let ndcCenter = vec2<f32>((center.x * 2.0 - 1.0) * scale.x, (1.0 - center.y * 2.0) * scale.y);
		let ndcOffset = vec2<f32>(
			quad.x * radius * 6.0 * scale.x / targetAspect,
			-quad.y * radius * 6.0 * scale.y
		);
		return VSOut(
			vec4<f32>(ndcCenter + ndcOffset, 0.0, 1.0),
			quad * 3.0,
			p.colorOpacity.xyz,
			sigmoid(p.colorOpacity.w) * temporal_gate(p, cfg.time)
		);
	}

	@fragment
			fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
				let q = dot(input.local, input.local);
				let alpha = clamp(input.opacity * exp(-0.5 * q), 0.0, 1.0);
				if (cfg.renderMode >= 1.5) {
					let visAlpha = clamp(alpha * 12.0, 0.0, 1.0);
					let heat = vec3<f32>(
						0.08 + 0.22 * visAlpha,
						0.22 + 0.78 * visAlpha,
						0.32 + 0.68 * visAlpha
					);
					return vec4<f32>(heat * visAlpha, visAlpha);
				}
				if (cfg.renderMode >= 0.5) {
					let residual = min(vec3<f32>(1.0), input.color * 1.8 + vec3<f32>(0.08, 0.18, 0.28));
					let visAlpha = clamp(alpha * 12.0, 0.0, 1.0);
					return vec4<f32>(residual * visAlpha, visAlpha);
				}
			return vec4<f32>(input.color * alpha, alpha);
		}
	`;

const BACKGROUND_WGSL = `
		struct RenderConfig {
		width: f32,
		height: f32,
		time: f32,
		splatCount: f32,
		pointScale: f32,
		modelMode: f32,
		targetAspect: f32,
		temporalSigma: f32,
			targetWidth: f32,
			targetHeight: f32,
			renderMode: f32,
			pad1: f32,
		};

	struct VSOut {
		@builtin(position) pos: vec4<f32>,
	};

	@group(0) @binding(0) var<uniform> cfg: RenderConfig;
	@group(0) @binding(1) var<storage, read> staticBackground: array<vec4<f32>>;

	fn fit_scale() -> vec2<f32> {
		let canvasAspect = cfg.width / max(1.0, cfg.height);
		let targetAspect = max(0.25, cfg.targetAspect);
		if (canvasAspect > targetAspect) {
			return vec2<f32>(targetAspect / canvasAspect, 1.0);
		}
		return vec2<f32>(1.0, canvasAspect / targetAspect);
	}

	@vertex
	fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
		let x = f32((vertexIndex << 1u) & 2u);
		let y = f32(vertexIndex & 2u);
		return VSOut(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0));
	}

	@fragment
	fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
		let scale = fit_scale();
		let ndc = vec2<f32>(
			pos.x / max(1.0, cfg.width) * 2.0 - 1.0,
			1.0 - pos.y / max(1.0, cfg.height) * 2.0
		);
		if (abs(ndc.x) > scale.x || abs(ndc.y) > scale.y) {
			return vec4<f32>(0.014, 0.017, 0.022, 1.0);
		}
		let uv = vec2<f32>(
			clamp((ndc.x / scale.x + 1.0) * 0.5, 0.0, 0.99999),
			clamp((1.0 - ndc.y / scale.y) * 0.5, 0.0, 0.99999)
		);
		let targetWidth = max(1u, u32(round(cfg.targetWidth)));
		let targetHeight = max(1u, u32(round(cfg.targetHeight)));
		let x = min(targetWidth - 1u, u32(uv.x * f32(targetWidth)));
		let y = min(targetHeight - 1u, u32(uv.y * f32(targetHeight)));
			let rgb = staticBackground[y * targetWidth + x].xyz;
			if (cfg.renderMode >= 0.5) {
				return vec4<f32>(0.0, 0.0, 0.0, 1.0);
			}
			return vec4<f32>(rgb, 1.0);
		}
	`;

export class DynamicSplatWebGpuTrainer {
	constructor(canvas) {
		this.canvas = canvas;
		this.dataset = null;
		this.device = null;
		this.context = null;
		this.format = null;
		this.adapterName = "WebGPU";
		this.splatCount = 768;
		this.stepCount = 0;
		this.currentIndex = 0;
		this.totalRecycled = 0;
		this.lastRecycleCount = 0;
		this.readbackChain = Promise.resolve();
		this.buffers = null;
		this.pipelines = null;
		this.bindGroups = null;
			this.configBytes = new ArrayBuffer(128);
		this.renderConfigBytes = new ArrayBuffer(48);
	}

	async init(dataset, { splatCount = 768 } = {}) {
		if (!navigator.gpu) {
			throw new Error("WebGPU unavailable in this browser.");
		}
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) {
			throw new Error("WebGPU adapter unavailable.");
		}
		this.adapterName = adapter.info?.description || adapter.info?.vendor || "WebGPU";
		this.device = await adapter.requestDevice();
		this.context = this.canvas.getContext("webgpu");
		if (!this.context) {
			throw new Error("WebGPU canvas context unavailable.");
		}
		this.format = navigator.gpu.getPreferredCanvasFormat();
		this.context.configure({
			device: this.device,
			format: this.format,
			alphaMode: "opaque",
		});

		this.dataset = dataset;
		this.splatCount = splatCount;
		this.stepCount = 0;
		this.currentIndex = 0;
		this.createPipelines();
		this.createBuffers();
		this.createBindGroups();
	}

	dispose() {
		if (this.buffers) {
			for (const buffer of Object.values(this.buffers)) {
				if (Array.isArray(buffer)) {
					for (const child of buffer) {
						child.destroy?.();
					}
				} else {
					buffer.destroy?.();
				}
			}
		}
		this.context?.unconfigure();
		this.buffers = null;
		this.bindGroups = null;
	}

	createPipelines() {
		const device = this.device;
		const trainModule = device.createShaderModule({ code: TRAIN_WGSL });
		const renderModule = device.createShaderModule({ code: RENDER_WGSL });
		const backgroundModule = device.createShaderModule({ code: BACKGROUND_WGSL });
		this.pipelines = {
			train: device.createComputePipeline({
				layout: "auto",
				compute: { module: trainModule, entryPoint: "train" },
			}),
			background: device.createRenderPipeline({
				layout: "auto",
				vertex: {
					module: backgroundModule,
					entryPoint: "vs_main",
				},
				fragment: {
					module: backgroundModule,
					entryPoint: "fs_main",
					targets: [{ format: this.format }],
				},
				primitive: { topology: "triangle-strip" },
			}),
			render: device.createRenderPipeline({
				layout: "auto",
				vertex: {
					module: renderModule,
					entryPoint: "vs_main",
					buffers: [
						{
							arrayStride: 8,
							attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
						},
					],
				},
				fragment: {
					module: renderModule,
					entryPoint: "fs_main",
					targets: [
						{
							format: this.format,
							blend: {
								color: {
									srcFactor: "one",
									dstFactor: "one-minus-src-alpha",
									operation: "add",
								},
								alpha: {
									srcFactor: "one",
									dstFactor: "one-minus-src-alpha",
									operation: "add",
								},
							},
						},
					],
				},
				primitive: { topology: "triangle-strip" },
			}),
		};
	}

	createBuffers() {
		const device = this.device;
		const params = makeInitialSplats(this.dataset, this.splatCount);
		this.initialParams = params.slice();
		const bufferUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
		const paramA = device.createBuffer({ size: params.byteLength, usage: bufferUsage });
		const paramB = device.createBuffer({ size: params.byteLength, usage: bufferUsage });
		device.queue.writeBuffer(paramA, 0, params);
		device.queue.writeBuffer(paramB, 0, params);
		const optimizerBytes = params.byteLength;
		const firstMoment = device.createBuffer({ size: optimizerBytes, usage: bufferUsage });
		const secondMoment = device.createBuffer({ size: optimizerBytes, usage: bufferUsage });
		device.queue.writeBuffer(firstMoment, 0, new Float32Array(params.length));
		device.queue.writeBuffer(secondMoment, 0, new Float32Array(params.length));
		const statsBytes = this.splatCount * 16;
		const splatStats = device.createBuffer({ size: statsBytes, usage: bufferUsage });
		device.queue.writeBuffer(splatStats, 0, new Float32Array(this.splatCount * 4));

		const targetBuffer = device.createBuffer({
			size: this.dataset.frames.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(targetBuffer, 0, this.dataset.frames);

		const backgroundBuffer = device.createBuffer({
			size: this.dataset.background.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(backgroundBuffer, 0, this.dataset.background);

		const motionSamples = this.dataset.motionSamples?.length
			? this.dataset.motionSamples
			: new Uint32Array([0]);
		const motionSampleBuffer = device.createBuffer({
			size: Math.max(4, motionSamples.byteLength),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(motionSampleBuffer, 0, motionSamples);

		const staticSamples = this.dataset.staticSamples?.length
			? this.dataset.staticSamples
			: new Uint32Array([0]);
		const staticSampleBuffer = device.createBuffer({
			size: Math.max(4, staticSamples.byteLength),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(staticSampleBuffer, 0, staticSamples);

		const quadBuffer = device.createBuffer({
			size: 32,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(
			quadBuffer,
			0,
			new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
		);

		this.buffers = {
			params: [paramA, paramB],
			firstMoment,
			secondMoment,
			splatStats,
			target: targetBuffer,
			background: backgroundBuffer,
			motionSamples: motionSampleBuffer,
			staticSamples: staticSampleBuffer,
			paramsReadback: device.createBuffer({
				size: params.byteLength,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			}),
			statsReadback: device.createBuffer({
				size: statsBytes,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			}),
			trainConfig: device.createBuffer({
				size: 128,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			}),
			renderConfig: device.createBuffer({
				size: 48,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			}),
			metrics: device.createBuffer({
				size: 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			}),
			metricsReadback: device.createBuffer({
				size: 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			}),
			quad: quadBuffer,
		};
	}

	createBindGroups() {
		const device = this.device;
		this.bindGroups = {
			train: [
				device.createBindGroup({
					layout: this.pipelines.train.getBindGroupLayout(0),
					entries: [
						{ binding: 0, resource: { buffer: this.buffers.trainConfig } },
						{ binding: 1, resource: { buffer: this.buffers.params[0] } },
						{ binding: 2, resource: { buffer: this.buffers.params[1] } },
						{ binding: 3, resource: { buffer: this.buffers.target } },
						{ binding: 4, resource: { buffer: this.buffers.metrics } },
						{ binding: 5, resource: { buffer: this.buffers.background } },
						{ binding: 6, resource: { buffer: this.buffers.motionSamples } },
						{ binding: 7, resource: { buffer: this.buffers.staticSamples } },
						{ binding: 8, resource: { buffer: this.buffers.firstMoment } },
						{ binding: 9, resource: { buffer: this.buffers.secondMoment } },
						{ binding: 10, resource: { buffer: this.buffers.splatStats } },
					],
				}),
				device.createBindGroup({
					layout: this.pipelines.train.getBindGroupLayout(0),
					entries: [
						{ binding: 0, resource: { buffer: this.buffers.trainConfig } },
						{ binding: 1, resource: { buffer: this.buffers.params[1] } },
						{ binding: 2, resource: { buffer: this.buffers.params[0] } },
						{ binding: 3, resource: { buffer: this.buffers.target } },
						{ binding: 4, resource: { buffer: this.buffers.metrics } },
						{ binding: 5, resource: { buffer: this.buffers.background } },
						{ binding: 6, resource: { buffer: this.buffers.motionSamples } },
						{ binding: 7, resource: { buffer: this.buffers.staticSamples } },
						{ binding: 8, resource: { buffer: this.buffers.firstMoment } },
						{ binding: 9, resource: { buffer: this.buffers.secondMoment } },
						{ binding: 10, resource: { buffer: this.buffers.splatStats } },
					],
				}),
			],
			background: device.createBindGroup({
				layout: this.pipelines.background.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.buffers.renderConfig } },
					{ binding: 1, resource: { buffer: this.buffers.background } },
				],
			}),
			render: [
				device.createBindGroup({
					layout: this.pipelines.render.getBindGroupLayout(0),
					entries: [
							{ binding: 0, resource: { buffer: this.buffers.renderConfig } },
							{ binding: 1, resource: { buffer: this.buffers.params[0] } },
						],
					}),
				device.createBindGroup({
					layout: this.pipelines.render.getBindGroupLayout(0),
					entries: [
							{ binding: 0, resource: { buffer: this.buffers.renderConfig } },
							{ binding: 1, resource: { buffer: this.buffers.params[1] } },
						],
					}),
			],
		};
	}

	trainStep({
		learningRate = 1.0,
		samplesPerStep = 96,
		modelMode = 0,
		temporalSigma = 0.30,
		motionSampleRate = 0.95,
		staticSampleRate = 0.08,
		motionCoverageTarget = 0.52,
	} = {}) {
		const lr = Number(learningRate);
		writeTrainConfig(this.configBytes, {
			width: this.dataset.width,
			height: this.dataset.height,
			frameCount: this.dataset.frameCount,
			splatCount: this.splatCount,
			sampleCount: Number(samplesPerStep),
			step: this.stepCount,
			modelMode,
			motionSampleCount: this.dataset.motionSamples?.length ?? 0,
			staticSampleCount: this.dataset.staticSamples?.length ?? 0,
			lrPos: lr * 0.00006,
			lrColor: lr * 0.00030,
			lrOpacity: lr * 0.00020,
			lrMotion: lr * 0.00004,
			minRadius: 0.009,
			maxRadius: 0.09,
			temporalSigma,
			targetAspect: this.dataset.width / Math.max(1, this.dataset.height),
			motionSampleRate,
			motionCoverageTarget,
			motionCoverageWeight: 0.08,
			staticAlphaWeight: 4.0,
			opacityDecayWeight: 0.0,
			beta1: 0.9,
			beta2: 0.99,
			adamEpsilon: 1e-6,
			statDecay: 0.95,
			robustMix: 0.0,
			staticEnergyThreshold: 0.00045,
			staticSampleRate,
		});
		this.device.queue.writeBuffer(this.buffers.trainConfig, 0, this.configBytes);

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipelines.train);
		pass.setBindGroup(0, this.bindGroups.train[this.currentIndex]);
		pass.dispatchWorkgroups(this.splatCount);
		pass.end();
		this.device.queue.submit([encoder.finish()]);
		this.currentIndex = 1 - this.currentIndex;
		this.stepCount += 1;
	}

	async readLoss() {
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.buffers.metrics, 0, this.buffers.metricsReadback, 0, 4);
		this.device.queue.submit([encoder.finish()]);
		await this.buffers.metricsReadback.mapAsync(GPUMapMode.READ);
		const value = new Float32Array(this.buffers.metricsReadback.getMappedRange().slice(0))[0];
		this.buffers.metricsReadback.unmap();
		return value;
	}

	async readParamsUnlocked() {
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(
			this.buffers.params[this.currentIndex],
			0,
			this.buffers.paramsReadback,
			0,
			this.splatCount * SPLAT_BYTES,
		);
		this.device.queue.submit([encoder.finish()]);
		await this.buffers.paramsReadback.mapAsync(GPUMapMode.READ);
		const mapped = this.buffers.paramsReadback.getMappedRange();
		const params = new Float32Array(mapped.slice(0));
		this.buffers.paramsReadback.unmap();
		return params;
	}

	readParams() {
		const read = this.readbackChain.then(() => this.readParamsUnlocked());
		this.readbackChain = read.then(() => undefined, () => undefined);
		return read;
	}

	async readSplatStats() {
		const byteLength = this.splatCount * 16;
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.buffers.splatStats, 0, this.buffers.statsReadback, 0, byteLength);
		this.device.queue.submit([encoder.finish()]);
		await this.buffers.statsReadback.mapAsync(GPUMapMode.READ);
		const stats = new Float32Array(this.buffers.statsReadback.getMappedRange().slice(0));
		this.buffers.statsReadback.unmap();
		return stats;
	}

	async maintainDensity({ modelMode = 0, temporalSigma = 0.30, maxRecycles = 8 } = {}) {
		if (this.stepCount < 256 || this.stepCount % 256 !== 0) {
			return 0;
		}
		const [params, stats] = await Promise.all([this.readParams(), this.readSplatStats()]);
		const replaceable = [];
		for (let i = 0; i < this.splatCount; i += 1) {
			const base = i * SPLAT_FLOATS;
			const opacity = sigmoid(params[base + 11]);
			const contribution = stats[i * 4 + 1];
			const gradient = stats[i * 4];
			if (opacity < 0.085 || contribution < 0.00035) {
				replaceable.push({ i, score: opacity * 0.70 + contribution * 20 + gradient * 0.02 });
			}
		}
		replaceable.sort((a, b) => a.score - b.score);
		const recycleCount = Math.min(maxRecycles, replaceable.length);
		if (recycleCount === 0) {
			this.lastRecycleCount = 0;
			return 0;
		}

		const motionSamples = this.dataset.motionSamples ?? new Uint32Array(0);
		if (motionSamples.length === 0) {
			return 0;
		}
		const candidateCount = Math.min(768, motionSamples.length);
		const residuals = [];
		const pixelsPerFrame = this.dataset.width * this.dataset.height;
		for (let c = 0; c < candidateCount; c += 1) {
			const sampleIndex = Math.min(motionSamples.length - 1, Math.floor((c + 0.5) * motionSamples.length / candidateCount));
			const packed = motionSamples[sampleIndex];
			const pixel = packed % pixelsPerFrame;
			const frameIndex = Math.min(this.dataset.frameCount - 1, Math.floor(packed / pixelsPerFrame));
			const x = pixel % this.dataset.width;
			const y = Math.floor(pixel / this.dataset.width);
			const frame = { index: frameIndex, time: frameTime(frameIndex, this.dataset.frameCount), pixel };
			const pred = evalModelCpu(
				this.dataset,
				params,
				this.splatCount,
				(x + 0.5) / this.dataset.width,
				(y + 0.5) / this.dataset.height,
				frame,
				modelMode,
				temporalSigma,
			);
			const targetBase = (frameIndex * pixelsPerFrame + pixel) * 4;
			const dr = pred[0] - this.dataset.frames[targetBase];
			const dg = pred[1] - this.dataset.frames[targetBase + 1];
			const db = pred[2] - this.dataset.frames[targetBase + 2];
			residuals.push({ packed, error: (dr * dr + dg * dg + db * db) / 3 });
		}
		residuals.sort((a, b) => b.error - a.error);
		const frameVelocities = computeFrameMotionVelocities(this.dataset);
		for (let slot = 0; slot < recycleCount; slot += 1) {
			const index = replaceable[slot].i;
			const packed = residuals[Math.min(residuals.length - 1, slot * 3 % residuals.length)].packed;
			const frame = Math.min(this.dataset.frameCount - 1, Math.floor(packed / pixelsPerFrame));
			const pixel = packed % pixelsPerFrame;
			const x = ((pixel % this.dataset.width) + 0.5) / this.dataset.width;
			const y = (Math.floor(pixel / this.dataset.width) + 0.5) / this.dataset.height;
			const timeCenter = frameTime(frame, this.dataset.frameCount);
			const velocity = estimateLocalMotionVelocity(this.dataset, frame, pixel, frameVelocities[frame] ?? [0, 0]);
			const vx = Math.min(0.14, Math.max(-0.14, velocity[0] * 0.55));
			const vy = Math.min(0.14, Math.max(-0.14, velocity[1] * 0.55));
			const tc = timeCenter * 2 - 1;
			const color = sampleDatasetColor(this.dataset, frame, x, y);
			const base = index * SPLAT_FLOATS;
			params[base] = Math.min(1.1, Math.max(-0.1, x - vx * tc));
			params[base + 1] = Math.min(1.1, Math.max(-0.1, y - vy * tc));
			params[base + 2] = timeCenter;
			params[base + 3] = 0.010;
			params[base + 4] = vx;
			params[base + 5] = vy;
			params[base + 6] = 0;
			params[base + 7] = 0;
			params[base + 8] = color[0];
			params[base + 9] = color[1];
			params[base + 10] = color[2];
			params[base + 11] = -1.80;
			const byteOffset = base * 4;
			const replacement = params.subarray(base, base + SPLAT_FLOATS);
			this.device.queue.writeBuffer(this.buffers.params[0], byteOffset, replacement);
			this.device.queue.writeBuffer(this.buffers.params[1], byteOffset, replacement);
			const zeroMoment = new Float32Array(SPLAT_FLOATS);
			this.device.queue.writeBuffer(this.buffers.firstMoment, byteOffset, zeroMoment);
			this.device.queue.writeBuffer(this.buffers.secondMoment, byteOffset, zeroMoment);
			this.device.queue.writeBuffer(this.buffers.splatStats, index * 16, new Float32Array(4));
		}
		this.lastRecycleCount = recycleCount;
		this.totalRecycled += recycleCount;
		return recycleCount;
	}

	async readValidationLoss({ modelMode = 0, temporalSigma = 0.30, gridSize = 32 } = {}) {
		const metrics = await this.readValidationMetrics({ modelMode, temporalSigma, gridSize });
		return metrics.gridLoss;
	}

	async readValidationMetrics({ modelMode = 0, temporalSigma = 0.30, gridSize = 32 } = {}) {
		const params = await this.readParams();
		const metrics = computeGridValidationMetrics(this.dataset, params, this.splatCount, {
			modelMode,
			temporalSigma,
			gridSize,
		});
		let parameterDelta = 0;
		for (let i = 0; i < params.length; i += 1) {
			parameterDelta += Math.abs(params[i] - this.initialParams[i]);
		}
		metrics.parameterDelta = parameterDelta / Math.max(1, params.length);
		metrics.totalRecycled = this.totalRecycled;
		return metrics;
	}

	async readPreviewErrorImage({ time = 0.35, modelMode = 0, temporalSigma = 0.30 } = {}) {
		const params = await this.readParams();
		const frameIndex = Math.min(
			this.dataset.frameCount - 1,
			Math.max(0, Math.round(time * (this.dataset.frameCount - 1))),
		);
		const frame = {
			index: frameIndex,
			time: frameTime(frameIndex, this.dataset.frameCount),
			pixel: 0,
		};
		const pixels = this.dataset.width * this.dataset.height;
		const data = new Uint8ClampedArray(pixels * 4);
		let meanAbs = 0;
		let maxAbs = 0;
		for (let pixel = 0; pixel < pixels; pixel += 1) {
			const x = pixel % this.dataset.width;
			const y = Math.floor(pixel / this.dataset.width);
			const px = (x + 0.5) / this.dataset.width;
			const py = (y + 0.5) / this.dataset.height;
			frame.pixel = pixel;
			const [r, g, b] = evalModelCpu(
				this.dataset,
				params,
				this.splatCount,
				px,
				py,
				frame,
				modelMode,
				temporalSigma,
			);
			const targetBase = (frameIndex * pixels + pixel) * 4;
			const dr = Math.abs(r - this.dataset.frames[targetBase]);
			const dg = Math.abs(g - this.dataset.frames[targetBase + 1]);
			const db = Math.abs(b - this.dataset.frames[targetBase + 2]);
			const error = Math.sqrt((dr * dr + dg * dg + db * db) / 3);
			meanAbs += (dr + dg + db) / 3;
			maxAbs = Math.max(maxAbs, error);
			const heat = Math.min(1, error * 8);
			const idx = pixel * 4;
			data[idx] = Math.round(255 * Math.min(1, heat * 1.45));
			data[idx + 1] = Math.round(255 * Math.max(0, Math.min(1, heat * 1.9 - 0.25)));
			data[idx + 2] = Math.round(255 * Math.max(0.04, 0.45 - heat * 0.35));
			data[idx + 3] = 255;
		}
		return {
			frame: frameIndex,
			width: this.dataset.width,
			height: this.dataset.height,
			data,
			meanAbs: meanAbs / Math.max(1, pixels),
			maxAbs,
		};
	}

	resizeCanvas() {
		const dpr = window.devicePixelRatio || 1;
		const rect = this.canvas.getBoundingClientRect();
		const width = Math.max(1, Math.floor(rect.width * dpr));
		const height = Math.max(1, Math.floor(rect.height * dpr));
		if (this.canvas.width !== width || this.canvas.height !== height) {
			this.canvas.width = width;
			this.canvas.height = height;
		}
	}

		render(time = 0.35, modelMode = 0, temporalSigma = 0.30, renderMode = 0) {
		this.resizeCanvas();
		writeRenderConfig(this.renderConfigBytes, {
			width: this.canvas.width,
			height: this.canvas.height,
			time,
			splatCount: this.splatCount,
			pointScale: 1,
			modelMode,
			targetAspect: this.dataset.width / Math.max(1, this.dataset.height),
			temporalSigma,
				targetWidth: this.dataset.width,
				targetHeight: this.dataset.height,
				renderMode,
			});
		this.device.queue.writeBuffer(this.buffers.renderConfig, 0, this.renderConfigBytes);

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginRenderPass({
			colorAttachments: [
				{
					view: this.context.getCurrentTexture().createView(),
					clearValue: { r: 0.014, g: 0.017, b: 0.022, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
		});
		pass.setPipeline(this.pipelines.background);
		pass.setBindGroup(0, this.bindGroups.background);
		pass.draw(4, 1, 0, 0);
		pass.setPipeline(this.pipelines.render);
		pass.setBindGroup(0, this.bindGroups.render[this.currentIndex]);
		pass.setVertexBuffer(0, this.buffers.quad);
		pass.draw(4, this.splatCount, 0, 0);
		pass.end();
		this.device.queue.submit([encoder.finish()]);
	}
}
