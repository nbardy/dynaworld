import {
	assertStorageBufferFits,
	DynamicSplatWebGpu3dTrainer,
	FILTER_SIGMA_PIXELS,
	MAX_SPLAT_COLOR,
	SPLAT_FLOATS,
	makeInitialSplats,
	rgbaFloatFrameBytes,
} from "./trainerWebGpu3d.js";
import {
	BROWSER_ADAM_BETA1,
	BROWSER_ADAM_BETA2,
	BROWSER_ADAM_EPSILON,
	DENSITY_STAT_DECAY,
	browserLearningRates,
} from "./trainingSchedule.js";

const SPLAT_BYTES = SPLAT_FLOATS * 4;
const TILE_SIZE = 16;
const MIN_CHECKPOINT_STRIDE = 16;
const PROJECTION_BYTES = 12 * 16;
const SSIM_STATS_BYTES = 5 * 16;
const DENSITY_START_STEP = 600;
const DENSITY_INTERVAL = 100;
const DENSITY_DISPATCHES = 4;
const TILED_CONFIG_BYTES = 160;
export const DEFAULT_MAX_TILE_CAPACITY = 4096;
export const DEFAULT_CHECKPOINT_PRECISION = "packed-f16";
export const DEFAULT_STATIC_WARMUP_STEPS = 2048;
export const MAX_WORKGROUPS_PER_DIMENSION = 65535;
export const SCALE_LR_FROM_COLOR = 0.30;
export const ROTATION_LR_FROM_MOTION = 1.25;
// This is an optimizer/performance trust region, not a roundness prior. A 6:1
// standard-deviation ratio still allows 36:1 covariance conditioning; larger
// needles increase tile pairs and were worse on heldout in the matched 12:1 run.
export const MAX_SCALE_ASPECT_RATIO = 6;

function hashU32(value) {
	let result = value >>> 0;
	result = Math.imul(result ^ (result >>> 16), 0x7feb352d);
	result = Math.imul(result ^ (result >>> 15), 0x846ca68b);
	return (result ^ (result >>> 16)) >>> 0;
}

export function packedTrainingBackgroundForStep(step, enabled = true) {
	if (!Number.isSafeInteger(step) || step < 0) {
		throw new RangeError("step must be a non-negative safe integer.");
	}
	if (!enabled) return 0;
	const seed = step >>> 0;
	const channel = (salt) => hashU32(seed ^ salt) & 0x3ff;
	return (0x80000000
		| channel(0x68bc21eb)
		| (channel(0x02e5be93) << 10)
		| (channel(0x967a889b) << 20)) >>> 0;
}

export function trainingBackgroundForStep(step) {
	const packed = packedTrainingBackgroundForStep(step);
	return [
		(packed & 0x3ff) / 1023,
		((packed >>> 10) & 0x3ff) / 1023,
		((packed >>> 20) & 0x3ff) / 1023,
	];
}

export function resolveSsimRadius(value = 5) {
	if (!Number.isSafeInteger(value) || value < 0 || value > 15) {
		throw new RangeError("ssimRadius must be an integer from 0 through 15.");
	}
	return value;
}

function ceilDiv(value, divisor) {
	return Math.floor((value + divisor - 1) / divisor);
}

function nextPowerOfTwo(value) {
	return 2 ** Math.ceil(Math.log2(Math.max(1, value)));
}

export function resolveTiledCapacity(initialSplats, requestedCapacity = null) {
	if (!Number.isSafeInteger(initialSplats) || initialSplats < 8) {
		throw new RangeError("initialSplats must be an integer of at least 8.");
	}
	const capacity = requestedCapacity == null ? initialSplats * 3 : Number(requestedCapacity);
	return Math.min(4096, Math.max(initialSplats, Math.floor(capacity)));
}

export function resolveTileCapacity(splatCount, requestedCapacity = null) {
	if (!Number.isSafeInteger(splatCount) || splatCount < 8 || splatCount > 4096) {
		throw new RangeError("splatCount must be an integer from 8 through 4096.");
	}
	const required = nextPowerOfTwo(splatCount);
	const requested = requestedCapacity == null ? required : Math.floor(Number(requestedCapacity));
	if (!Number.isSafeInteger(requested) || requested < splatCount
		|| requested > DEFAULT_MAX_TILE_CAPACITY) {
		throw new RangeError(`tileCapacity must cover all ${splatCount} splats and be at most `
			+ `${DEFAULT_MAX_TILE_CAPACITY}.`);
	}
	return nextPowerOfTwo(requested);
}

export function resolveCheckpointPrecision(value = DEFAULT_CHECKPOINT_PRECISION) {
	if (value !== "f32" && value !== "packed-f16") {
		throw new RangeError('checkpointPrecision must be "f32" or "packed-f16".');
	}
	return value;
}

export function resolveStaticWarmupSteps(value = 0) {
	const steps = Number(value);
	if (!Number.isSafeInteger(steps) || steps < 0 || steps > 1_000_000) {
		throw new RangeError("staticWarmupSteps must be an integer from 0 through 1000000.");
	}
	return steps;
}

export function resolvePairDispatch(pairCount) {
	if (!Number.isSafeInteger(pairCount) || pairCount < 0) {
		throw new RangeError("pairCount must be a non-negative safe integer.");
	}
	return {
		x: Math.min(pairCount, MAX_WORKGROUPS_PER_DIMENSION),
		y: Math.max(1, ceilDiv(pairCount, MAX_WORKGROUPS_PER_DIMENSION)),
		z: 1,
	};
}

export function resolveCheckpointLayout(pixelCount, tileCapacity, storageLimit, bytesPerCheckpoint = 16) {
	if (!Number.isSafeInteger(pixelCount) || pixelCount < 1
		|| !Number.isSafeInteger(tileCapacity) || tileCapacity < MIN_CHECKPOINT_STRIDE
		|| !Number.isSafeInteger(storageLimit) || storageLimit < 16
		|| (bytesPerCheckpoint !== 8 && bytesPerCheckpoint !== 16)) {
		throw new RangeError("Checkpoint layout inputs must be positive safe integers.");
	}
	for (let stride = MIN_CHECKPOINT_STRIDE; stride <= tileCapacity; stride *= 2) {
		const blocksPerTile = ceilDiv(tileCapacity, stride);
		const byteLength = pixelCount * blocksPerTile * bytesPerCheckpoint;
		if (Number.isSafeInteger(byteLength) && byteLength <= storageLimit) {
			return { stride, blocksPerTile, byteLength };
		}
	}
	throw new RangeError(`Even one checkpoint record per pixel exceeds the ${storageLimit}-byte storage limit.`);
}

export function densityDispatchesForStep(initialSplats, capacity, step) {
	const hiddenSlots = Math.max(0, Math.floor(capacity) - Math.floor(initialSplats));
	const fillEvents = Math.ceil(hiddenSlots / (DENSITY_DISPATCHES * 4));
	const fillEvent = Math.floor((step - DENSITY_START_STEP) / DENSITY_INTERVAL);
	if (step >= DENSITY_START_STEP
		&& (step - DENSITY_START_STEP) % DENSITY_INTERVAL === 0
		&& fillEvent >= 0 && fillEvent < fillEvents) {
		const remaining = hiddenSlots - fillEvent * DENSITY_DISPATCHES * 4;
		return Math.min(DENSITY_DISPATCHES, Math.ceil(remaining / 4));
	}
	// Once reserved capacity is full, keep the SfM scaffold stable. The former
	// perpetual recycling repeatedly erased useful seeds without a residual-
	// guided replacement test.
	return 0;
}

export function fullFramePairForStep(trainViewIndices, frameCount, step) {
	if (!Array.isArray(trainViewIndices) || trainViewIndices.length < 1) {
		throw new Error("At least one train view is required.");
	}
	const safeStep = Math.max(0, Math.floor(step));
	const pairsPerCycle = trainViewIndices.length * Math.max(1, frameCount);
	let stride = Math.max(1, Math.floor(pairsPerCycle * 0.618)) | 1;
	const gcd = (left, right) => {
		let a = left; let b = right;
		while (b !== 0) [a, b] = [b, a % b];
		return a;
	};
	while (gcd(stride, pairsPerCycle) !== 1) stride += 2;
	const pairIndex = (safeStep * stride) % pairsPerCycle;
	const viewSlot = pairIndex % trainViewIndices.length;
	const frameIndex = Math.floor(pairIndex / trainViewIndices.length);
	return { viewSlot, viewIndex: trainViewIndices[viewSlot], frameIndex };
}

export function trainingPairForStep(trainViewIndices, frameCount, step, staticWarmupSteps = 0) {
	const safeStep = Math.max(0, Math.floor(step));
	const warmupSteps = resolveStaticWarmupSteps(staticWarmupSteps);
	if (safeStep < warmupSteps) {
		const selected = fullFramePairForStep(trainViewIndices, 1, safeStep);
		return {
			...selected,
			frameIndex: Math.floor((Math.max(1, frameCount) - 1) / 2),
			staticWarmup: true,
		};
	}
	return {
		...fullFramePairForStep(trainViewIndices, frameCount, safeStep - warmupSteps),
		staticWarmup: false,
	};
}

export function opacityAwarePixelBounds(projection, peakAlpha, width, height, alphaThreshold = 1 / 255) {
	if (!projection?.valid || !(peakAlpha > alphaThreshold)) return null;
	const qLimit = Math.min(9, 2 * Math.log(peakAlpha / alphaThreshold));
	if (!(qLimit > 0)) return null;
	const centerX = projection.center[0] * height;
	const centerY = projection.center[1] * height;
	const radiusX = Math.sqrt(Math.max(0, qLimit * projection.covariance[0])) * height;
	const radiusY = Math.sqrt(Math.max(0, qLimit * projection.covariance[2])) * height;
	const minX = Math.max(0, Math.floor(centerX - radiusX));
	const maxX = Math.min(width - 1, Math.ceil(centerX + radiusX));
	const minY = Math.max(0, Math.floor(centerY - radiusY));
	const maxY = Math.min(height - 1, Math.ceil(centerY + radiusY));
	return minX <= maxX && minY <= maxY ? { minX, maxX, minY, maxY, qLimit } : null;
}

export function ellipseIntersectsRect(center, conic, qLimit, rectangle) {
	const [mx, my] = center; const [a, b, c] = conic;
	const { minX, minY, maxX, maxY } = rectangle;
	const dx0 = minX - mx; const dx1 = maxX - mx;
	const dy0 = minY - my; const dy1 = maxY - my;
	if (mx >= minX && mx <= maxX && my >= minY && my <= maxY) return true;
	const quadratic = (dx, dy) => a * dx * dx + 2 * b * dx * dy + c * dy * dy;
	let minimum = Math.min(
		quadratic(dx0, dy0), quadratic(dx0, dy1),
		quadratic(dx1, dy0), quadratic(dx1, dy1),
	);
	if (c > 1e-8) {
		minimum = Math.min(minimum,
			quadratic(dx0, Math.min(dy1, Math.max(dy0, -(b / c) * dx0))),
			quadratic(dx1, Math.min(dy1, Math.max(dy0, -(b / c) * dx1))));
	}
	if (a > 1e-8) {
		minimum = Math.min(minimum,
			quadratic(Math.min(dx1, Math.max(dx0, -(b / a) * dy0)), dy0),
			quadratic(Math.min(dx1, Math.max(dx0, -(b / a) * dy1)), dy1));
	}
	return minimum <= qLimit;
}

function reflectIndex(index, size) {
	if (size <= 1) return 0;
	let value = index;
	const maximum = size - 1;
	while (value < 0 || value > maximum) {
		if (value < 0) value = -value;
		if (value > maximum) value = 2 * maximum - value;
	}
	return value;
}

const SSIM_GAUSSIAN_11 = Object.freeze([
	0.0010283800844791101,
	0.0075987581352391850,
	0.036000772128430829,
	0.10936068950970002,
	0.21300553771125369,
	0.26601172486179436,
	0.21300553771125369,
	0.10936068950970002,
	0.036000772128430829,
	0.0075987581352391850,
	0.0010283800844791101,
]);

function ssimKernel1d(radius) {
	if (radius === 5) return SSIM_GAUSSIAN_11;
	const side = radius * 2 + 1;
	return Array.from({ length: side }, () => 1 / side);
}

function reflectedKernelWeight(center, pixel, size, radius, kernel) {
	const weightAt = (offset) => Math.abs(offset) <= radius ? kernel[offset + radius] : 0;
	let weight = weightAt(pixel - center);
	if (pixel > 0) weight += weightAt(-pixel - center);
	const maximum = size - 1;
	if (pixel < maximum) weight += weightAt(2 * maximum - pixel - center);
	return weight;
}

export function windowedL1DssimCpu(prediction, target, width, height, {
	l1Weight = 0.8,
	dssimWeight = 0.2,
	radius = 5,
	c1 = 0.0001,
	c2 = 0.0009,
	computeGradient = true,
	pixelWeights = null,
} = {}) {
	if (prediction.length !== target.length || prediction.length !== width * height * 3) {
		throw new RangeError("prediction and target must be packed RGB images.");
	}
	const pixels = width * height;
	if (pixelWeights && pixelWeights.length !== pixels) {
		throw new RangeError("pixelWeights must contain one value per image pixel.");
	}
	const kernel = ssimKernel1d(radius);
	const stats = Array.from({ length: pixels }, () => null);
	let l1 = 0;
	let ssimSum = 0;
	let weightSum = 0;
	for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
		const sums = Array.from({ length: 5 }, () => [0, 0, 0]);
		for (let oy = -radius; oy <= radius; oy += 1) {
			const sy = reflectIndex(y + oy, height);
			for (let ox = -radius; ox <= radius; ox += 1) {
				const sx = reflectIndex(x + ox, width);
				const base = (sy * width + sx) * 3;
				const weight = kernel[oy + radius] * kernel[ox + radius];
				for (let channel = 0; channel < 3; channel += 1) {
					const px = prediction[base + channel];
					const py = target[base + channel];
					sums[0][channel] += weight * px;
					sums[1][channel] += weight * py;
					sums[2][channel] += weight * px * px;
					sums[3][channel] += weight * py * py;
					sums[4][channel] += weight * px * py;
				}
			}
		}
		const muX = sums[0];
		const muY = sums[1];
		const varX = sums[2].map((value, channel) => value - muX[channel] ** 2);
		const varY = sums[3].map((value, channel) => value - muY[channel] ** 2);
		const cov = sums[4].map((value, channel) => value - muX[channel] * muY[channel]);
		stats[y * width + x] = { muX, muY, varX, varY, cov };
		const centerWeight = pixelWeights?.[y * width + x] ?? 1;
		weightSum += centerWeight;
		for (let channel = 0; channel < 3; channel += 1) {
			const numerator = (2 * muX[channel] * muY[channel] + c1) * (2 * cov[channel] + c2);
			const denominator = (muX[channel] ** 2 + muY[channel] ** 2 + c1)
				* (varX[channel] + varY[channel] + c2);
			ssimSum += centerWeight * numerator / Math.max(denominator, 1e-12);
			const base = (y * width + x) * 3 + channel;
			l1 += centerWeight * Math.abs(prediction[base] - target[base]);
		}
	}
	const objectiveDenominator = Math.max(1e-12, weightSum * 3);
	l1 /= objectiveDenominator;
	const dssim = 1 - ssimSum / objectiveDenominator;
	const loss = l1Weight * l1 + dssimWeight * dssim;
	if (!computeGradient) return { loss, l1, dssim, gradient: null };
	const gradient = new Float32Array(prediction.length);
	for (let py = 0; py < height; py += 1) for (let px = 0; px < width; px += 1) {
		const pixel = py * width + px;
		for (let channel = 0; channel < 3; channel += 1) {
			const packed = pixel * 3 + channel;
			const error = prediction[packed] - target[packed];
			let dssimGradient = 0;
			for (let cy = Math.max(0, py - radius); cy <= Math.min(height - 1, py + radius); cy += 1) {
				const yWeight = reflectedKernelWeight(cy, py, height, radius, kernel);
				for (let cx = Math.max(0, px - radius); cx <= Math.min(width - 1, px + radius); cx += 1) {
					const centerWeight = pixelWeights?.[cy * width + cx] ?? 1;
					const weight = centerWeight * yWeight
						* reflectedKernelWeight(cx, px, width, radius, kernel);
					if (weight === 0) continue;
					const center = stats[cy * width + cx];
					const mx = center.muX[channel]; const my = center.muY[channel];
					const vx = center.varX[channel]; const vy = center.varY[channel];
					const covariance = center.cov[channel];
					const a = 2 * mx * my + c1; const b = 2 * covariance + c2;
					const c = mx * mx + my * my + c1; const d = vx + vy + c2;
					const da = 2 * my * weight;
					const db = 2 * weight * (target[packed] - my);
					const dc = 2 * mx * weight;
					const dd = 2 * weight * (prediction[packed] - mx);
						const denominator = Math.max(c * d, 1e-12);
						dssimGradient -= (((da * b + a * db) * denominator)
							- (a * b) * (dc * d + c * dd)) / (denominator ** 2 * objectiveDenominator);
				}
			}
			const ownWeight = pixelWeights?.[pixel] ?? 1;
			gradient[packed] = l1Weight * ownWeight * Math.sign(error) / objectiveDenominator
				+ dssimWeight * dssimGradient;
		}
	}
	return { loss, l1, dssim, gradient };
}

function writeTiledConfig(buffer, values) {
	const view = new DataView(buffer);
	const u32 = (offset, value) => view.setUint32(offset, value, true);
	const f32 = (offset, value) => view.setFloat32(offset, value, true);
	u32(0, values.width); u32(4, values.height); u32(8, values.splatCount); u32(12, TILE_SIZE);
	u32(16, values.tilesX); u32(20, values.tilesY); u32(24, values.tileCapacity);
	u32(28, values.blocksPerTile);
	u32(32, values.viewIndex); u32(36, values.frameIndex); u32(40, values.step);
	u32(44, values.modelMode); u32(48, values.targetOffset); u32(52, values.pixelCount);
	u32(56, values.pairCapacity); u32(60, values.checkpointStride);
	f32(64, values.targetAspect); f32(68, values.temporalSigma); f32(72, values.alphaThreshold);
	f32(76, values.transmittanceThreshold); f32(80, values.lrPosition); f32(84, values.lrColor);
	f32(88, values.lrOpacity); f32(92, values.lrMotion); f32(96, values.geometryScale);
	f32(100, values.l1Weight); f32(104, values.dssimWeight); f32(108, values.statDecay);
	f32(112, BROWSER_ADAM_BETA1); f32(116, BROWSER_ADAM_BETA2);
	f32(120, BROWSER_ADAM_EPSILON);
	u32(124, packedTrainingBackgroundForStep(values.step, values.randomBackground));
	u32(128, values.ssimRadius); u32(132, values.frameCount);
	u32(136, values.staticWarmup ? 1 : 0); u32(140, values.motionWeighting ? 1 : 0);
	f32(144, 0.0001); f32(148, 0.0009);
	f32(152, 0.03 * values.geometryScale); f32(156, values.geometryScale);
}

const CONFIG_WGSL = `
	struct TiledConfig {
		width:u32, height:u32, splatCount:u32, tileSize:u32,
		tilesX:u32, tilesY:u32, tileCapacity:u32, blocksPerTile:u32,
		viewIndex:u32, frameIndex:u32, step:u32, modelMode:u32,
		targetOffset:u32, pixelCount:u32, pairCapacity:u32, checkpointStride:u32,
		targetAspect:f32, temporalSigma:f32, alphaThreshold:f32, transmittanceThreshold:f32,
		lrPosition:f32, lrColor:f32, lrOpacity:f32, lrMotion:f32,
		geometryScale:f32, l1Weight:f32, dssimWeight:f32, statDecay:f32,
		beta1:f32, beta2:f32, adamEpsilon:f32, trainingBackgroundPacked:u32,
		ssimRadius:u32, frameCount:u32, staticWarmup:u32, motionWeighting:u32,
		c1:f32, c2:f32, minScale:f32, maxScale:f32,
	};
	struct Splat {
		centerStatic:vec4<f32>, velocityTime:vec4<f32>, harmonicPad:vec4<f32>,
		logScalePad:vec4<f32>, rotation:vec4<f32>, colorOpacity:vec4<f32>,
	};
	struct Projection {
		screenConic0:vec4<f32>, conicDepthAlpha:vec4<f32>, cameraPointValid:vec4<f32>,
		jacobian0:vec4<f32>, jacobian1:vec4<f32>,
		basis0:vec4<f32>, basis1:vec4<f32>, basis2:vec4<f32>,
		camera0:vec4<f32>, camera1:vec4<f32>, camera2:vec4<f32>, variancesPad:vec4<f32>,
	};
	struct Camera {
		row0:vec4<f32>, row1:vec4<f32>, row2:vec4<f32>, row3:vec4<f32>, intrinsics:vec4<f32>,
	};
	fn sigmoid(x:f32)->f32 { return 1.0/(1.0+exp(-x)); }
	fn safe_quaternion(raw:vec4<f32>)->vec4<f32> {
		let n2=dot(raw,raw); let normalized=raw*inverseSqrt(max(n2,1e-16));
		return select(vec4<f32>(0.0,0.0,0.0,1.0),normalized,n2>1e-16);
	}
	fn quaternion_matrix(raw:vec4<f32>)->mat3x3<f32> {
		let q=safe_quaternion(raw); let x=q.x; let y=q.y; let z=q.z; let w=q.w;
		return mat3x3<f32>(
			vec3<f32>(1.0-2.0*(y*y+z*z),2.0*(x*y+z*w),2.0*(x*z-y*w)),
			vec3<f32>(2.0*(x*y-z*w),1.0-2.0*(x*x+z*z),2.0*(y*z+x*w)),
			vec3<f32>(2.0*(x*z+y*w),2.0*(y*z-x*w),1.0-2.0*(x*x+y*y)));
	}
	fn outer3(a:vec3<f32>,b:vec3<f32>)->mat3x3<f32> {
		return mat3x3<f32>(a*b.x,a*b.y,a*b.z);
	}
	fn world_center(p:Splat,t:f32,modelMode:u32)->vec3<f32> {
		let tc=t*2.0-1.0; var center=p.centerStatic.xyz+p.velocityTime.xyz*tc;
		if(modelMode==0u){center+=p.harmonicPad.xyz*sin(t*6.28318530718);}
		return center;
	}
	fn temporal_gate(p:Splat,t:f32,sigmaValue:f32)->f32 {
		let sigma=clamp(sigmaValue,0.12,0.36);
		let floorValue=clamp(sigma*0.30,0.035,0.12);
		let dt=t-clamp(p.velocityTime.w,0.0,1.0);
		let dynamicGate=floorValue+(1.0-floorValue)*exp(-0.5*dt*dt/(sigma*sigma));
		return mix(dynamicGate,1.0,clamp(p.centerStatic.w,0.0,1.0));
	}
	fn frame_time(cfg:TiledConfig)->f32 {
		if(cfg.staticWarmup!=0u){return 0.5;}
		return select(0.0,f32(cfg.frameIndex)/f32(max(1u,cfg.frameCount-1u)),cfg.frameCount>1u);
	}
	fn training_background(packed:u32)->vec3<f32> {
		let rgb=vec3<f32>(
			f32(packed&0x3ffu),
			f32((packed>>10u)&0x3ffu),
			f32((packed>>20u)&0x3ffu))*(1.0/1023.0);
		return select(vec3<f32>(0.0),rgb,(packed&0x80000000u)!=0u);
	}
`;

function projectWgsl() {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> cameras:array<Camera>;
	@group(0) @binding(3) var<storage,read_write> tileCounts:array<atomic<u32>>;
	@group(0) @binding(4) var<storage,read_write> pairData:array<u32>;
	@group(0) @binding(5) var<storage,read_write> projections:array<Projection>;
	@group(0) @binding(6) var<storage,read_write> counters:array<atomic<u32>>;
	fn quadratic(d:vec2<f32>,q:vec3<f32>)->f32 {
		return q.x*d.x*d.x+2.0*q.y*d.x*d.y+q.z*d.y*d.y;
	}
	fn ellipse_intersects_rect(m:vec2<f32>,conic:vec3<f32>,tau:f32,
		minimum:vec2<f32>,maximum:vec2<f32>)->bool {
		let d0=minimum-m;let d1=maximum-m;
		if(all(m>=minimum)&&all(m<=maximum)){return true;}
		var qmin=min(min(quadratic(vec2<f32>(d0.x,d0.y),conic),
			quadratic(vec2<f32>(d0.x,d1.y),conic)),
			min(quadratic(vec2<f32>(d1.x,d0.y),conic),quadratic(vec2<f32>(d1.x,d1.y),conic)));
		if(conic.z>1e-8){
			qmin=min(qmin,quadratic(vec2<f32>(d0.x,clamp(-(conic.y/conic.z)*d0.x,d0.y,d1.y)),conic));
			qmin=min(qmin,quadratic(vec2<f32>(d1.x,clamp(-(conic.y/conic.z)*d1.x,d0.y,d1.y)),conic));
		}
		if(conic.x>1e-8){
			qmin=min(qmin,quadratic(vec2<f32>(clamp(-(conic.y/conic.x)*d0.y,d0.x,d1.x),d0.y),conic));
			qmin=min(qmin,quadratic(vec2<f32>(clamp(-(conic.y/conic.x)*d1.y,d0.x,d1.x),d1.y),conic));
		}
		return qmin<=tau;
	}
	@compute @workgroup_size(64)
	fn project_and_bin(@builtin(global_invocation_id) gid:vec3<u32>){
		let i=gid.x; if(i>=cfg.splatCount){return;} let p=params[i]; let camera=cameras[cfg.viewIndex];
		let t=frame_time(cfg);
		let h=vec4<f32>(world_center(p,t,cfg.modelMode),1.0);
		let cp=vec3<f32>(dot(camera.row0,h),dot(camera.row1,h),dot(camera.row2,h));
		var out=Projection(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(cp,0.0),
			vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),
			vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));
		if(cp.z<=0.1){projections[i]=out;return;}
		let cameraRotation=mat3x3<f32>(
			vec3<f32>(camera.row0.x,camera.row1.x,camera.row2.x),
			vec3<f32>(camera.row0.y,camera.row1.y,camera.row2.y),
			vec3<f32>(camera.row0.z,camera.row1.z,camera.row2.z));
		let basis=cameraRotation*quaternion_matrix(p.rotation);
		let variances=exp(2.0*clamp(p.logScalePad.xyz,vec3<f32>(-16.0),vec3<f32>(4.0)));
		let sigmaCamera=variances.x*outer3(basis[0],basis[0])
			+variances.y*outer3(basis[1],basis[1])+variances.z*outer3(basis[2],basis[2]);
		let invZ=1.0/cp.z; let horizontalFocal=cfg.targetAspect*camera.intrinsics.x;
		let j0=vec3<f32>(horizontalFocal*invZ,0.0,-horizontalFocal*cp.x*invZ*invZ);
		let j1=vec3<f32>(0.0,camera.intrinsics.y*invZ,-camera.intrinsics.y*cp.y*invZ*invZ);
		// Conservative screen-space footprint floor. This is point-sampled
		// EWA-style filtering, not Mip-Splatting's compensated pixel filter.
		let filterVariance=pow(${FILTER_SIGMA_PIXELS}/max(1.0,f32(cfg.height)),2.0);
		let covariance=vec3<f32>(dot(j0,sigmaCamera*j0)+filterVariance,
			dot(j0,sigmaCamera*j1),dot(j1,sigmaCamera*j1)+filterVariance);
		let determinant=covariance.x*covariance.z-covariance.y*covariance.y;
		if(determinant<=1e-16){projections[i]=out;return;}
		let center=vec2<f32>(cfg.targetAspect*(camera.intrinsics.x*cp.x*invZ+camera.intrinsics.z),
			camera.intrinsics.y*cp.y*invZ+camera.intrinsics.w);
		let conic=vec3<f32>(covariance.z,-covariance.y,covariance.x)/determinant;
		let opacity=sigmoid(p.colorOpacity.w);
		let timeWeight=select(temporal_gate(p,t,cfg.temporalSigma),1.0,cfg.staticWarmup!=0u);
		out=Projection(vec4<f32>(center,conic.xy),vec4<f32>(conic.z,cp.z,opacity,timeWeight),
			vec4<f32>(cp,1.0),vec4<f32>(j0,0.0),vec4<f32>(j1,0.0),
			vec4<f32>(basis[0],0.0),vec4<f32>(basis[1],0.0),vec4<f32>(basis[2],0.0),
			vec4<f32>(cameraRotation[0],0.0),vec4<f32>(cameraRotation[1],0.0),
			vec4<f32>(cameraRotation[2],0.0),vec4<f32>(variances,0.0));
		projections[i]=out;
		let peak=opacity*timeWeight; if(peak<=cfg.alphaThreshold){return;}
		let qLimit=min(9.0,2.0*log(peak/cfg.alphaThreshold));
		let centerPx=vec2<f32>(center.x*f32(cfg.height),center.y*f32(cfg.height));
		let radiusPx=vec2<f32>(sqrt(max(0.0,qLimit*covariance.x)),
			sqrt(max(0.0,qLimit*covariance.z)))*f32(cfg.height);
		let minPixel=vec2<i32>(max(vec2<i32>(0),vec2<i32>(floor(centerPx-radiusPx-vec2<f32>(0.5)))));
		let maxPixel=min(vec2<i32>(i32(cfg.width)-1,i32(cfg.height)-1),
			vec2<i32>(ceil(centerPx+radiusPx-vec2<f32>(0.5))));
		if(any(minPixel>maxPixel)){return;}
		let minTile=vec2<u32>(minPixel)/cfg.tileSize; let maxTile=vec2<u32>(maxPixel)/cfg.tileSize;
		for(var ty=minTile.y;ty<=maxTile.y;ty++){
			for(var tx=minTile.x;tx<=maxTile.x;tx++){
				let pixelMin=vec2<f32>(f32(tx*cfg.tileSize)+0.5,f32(ty*cfg.tileSize)+0.5)/f32(cfg.height);
				let pixelMax=vec2<f32>(f32(min(cfg.width-1u,(tx+1u)*cfg.tileSize-1u))+0.5,
					f32(min(cfg.height-1u,(ty+1u)*cfg.tileSize-1u))+0.5)/f32(cfg.height);
				if(ellipse_intersects_rect(center,conic,qLimit,pixelMin,pixelMax)){
					let tile=ty*cfg.tilesX+tx; let slot=atomicAdd(&tileCounts[tile],1u);
					if(slot<cfg.tileCapacity){
						pairData[tile*cfg.tileCapacity+slot]=i;
					}else{atomicAdd(&counters[1],1u);}
				}
			}
		}
	}`;
}

function clearWgsl() {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read_write> tileCounts:array<atomic<u32>>;
	@group(0) @binding(2) var<storage,read_write> counters:array<atomic<u32>>;
	@group(0) @binding(3) var<storage,read_write> indirectArgs:array<u32>;
	@group(0) @binding(4) var<storage,read_write> metrics:array<vec4<f32>>;
	@group(0) @binding(5) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	@compute @workgroup_size(64)
	fn clear_step(@builtin(global_invocation_id) gid:vec3<u32>){
		let tileCount=cfg.tilesX*cfg.tilesY;
		if(gid.x<tileCount){atomicStore(&tileCounts[gid.x],0u);}
		if(gid.x<cfg.splatCount*24u){atomicStore(&gradientAtoms[gid.x],0u);}
		if(gid.x==0u){
			atomicStore(&counters[0],0u);atomicStore(&counters[1],0u);
			indirectArgs[0]=0u;indirectArgs[1]=1u;indirectArgs[2]=1u;metrics[0]=vec4<f32>(0.0);
		}
	}`;
}

function sortWgsl(tileCapacity) {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> projections:array<Projection>;
	@group(0) @binding(2) var<storage,read_write> tileCounts:array<atomic<u32>>;
	@group(0) @binding(3) var<storage,read_write> pairData:array<u32>;
	@group(0) @binding(4) var<storage,read_write> counters:array<atomic<u32>>;
	var<workgroup> depthKeys:array<u32,${tileCapacity}>;
	var<workgroup> tileSortCount:u32;
	var<workgroup> pairBase:u32;
	@compute @workgroup_size(256)
	fn sort_tiles(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
		let tile=wid.x;let tileCount=cfg.tilesX*cfg.tilesY;if(tile>=tileCount){return;}
		let count=min(atomicLoad(&tileCounts[tile]),cfg.tileCapacity);
		if(lid.x==0u){
			var span=1u;
			loop{
				if(span>=max(count,1u)){break;}
				span*=2u;
			}
			tileSortCount=span;
		}
		let sortCount=workgroupUniformLoad(&tileSortCount);
		for(var index=lid.x;index<sortCount;index+=256u){
			if(index<count){
				let id=pairData[tile*cfg.tileCapacity+index];
				let depthBits=bitcast<u32>(max(0.0,projections[id].conicDepthAlpha.y));
				depthKeys[index]=(depthBits&0xfffff000u)|(id&0x00000fffu);
			}else{depthKeys[index]=0xffffffffu;}
		}
		workgroupBarrier();
		for(var width=2u;width<=sortCount;width*=2u){
			var stride=width/2u;
			loop{
				for(var index=lid.x;index<sortCount;index+=256u){
					let partner=index^stride;
					if(partner>index){
						let ascending=(index&width)==0u;
						let swap=select(depthKeys[index]<depthKeys[partner],
							depthKeys[index]>depthKeys[partner],ascending);
						if(swap){
							let key=depthKeys[index];depthKeys[index]=depthKeys[partner];depthKeys[partner]=key;
						}
					}
				}
				workgroupBarrier();if(stride==1u){break;}stride/=2u;
			}
		}
		for(var index=lid.x;index<count;index+=256u){
			let slot=tile*cfg.tileCapacity+index;let id=depthKeys[index]&0x00000fffu;pairData[slot]=id;
		}
		if(lid.x==0u){pairBase=atomicAdd(&counters[0],count);}
		workgroupBarrier();
		for(var index=lid.x;index<count;index+=256u){
			pairData[cfg.pairCapacity+pairBase+index]=tile*cfg.tileCapacity+index;
		}
	}`;
}

function finalizeWgsl() {
	return `
	@group(0) @binding(0) var<storage,read_write> counters:array<atomic<u32>>;
	@group(0) @binding(1) var<storage,read_write> indirectArgs:array<u32>;
	@compute @workgroup_size(1) fn finalize_pairs(){
		let pairCount=atomicLoad(&counters[0]);
		indirectArgs[0]=min(pairCount,${MAX_WORKGROUPS_PER_DIMENSION}u);
		indirectArgs[1]=max(1u,(pairCount+${MAX_WORKGROUPS_PER_DIMENSION - 1}u)
			/${MAX_WORKGROUPS_PER_DIMENSION}u);
		indirectArgs[2]=1u;
	}
	`;
}

function checkpointForwardWgsl(precision) {
	return precision === "packed-f16" ? `
	@group(0) @binding(6) var<storage,read_write> checkpoints:array<vec2<u32>>;
	fn write_checkpoint(index:u32,state:vec4<f32>){
		checkpoints[index]=vec2<u32>(pack2x16float(state.xy),pack2x16float(state.zw));
	}` : `
	@group(0) @binding(6) var<storage,read_write> checkpoints:array<vec4<f32>>;
	fn write_checkpoint(index:u32,state:vec4<f32>){checkpoints[index]=state;}
	`;
}

function checkpointBackwardWgsl(precision) {
	return precision === "packed-f16" ? `
	@group(0) @binding(5) var<storage,read> checkpoints:array<vec2<u32>>;
	fn read_checkpoint(index:u32)->vec4<f32>{
		let packed=checkpoints[index];
		return vec4<f32>(unpack2x16float(packed.x),unpack2x16float(packed.y));
	}` : `
	@group(0) @binding(5) var<storage,read> checkpoints:array<vec4<f32>>;
	fn read_checkpoint(index:u32)->vec4<f32>{return checkpoints[index];}
	`;
}

function forwardWgsl(checkpointPrecision) {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> projections:array<Projection>;
	@group(0) @binding(3) var<storage,read_write> tileCounts:array<atomic<u32>>;
	@group(0) @binding(4) var<storage,read> pairData:array<u32>;
	@group(0) @binding(5) var<storage,read_write> rendered:array<vec4<f32>>;
	${checkpointForwardWgsl(checkpointPrecision)}
	@group(0) @binding(7) var<storage,read_write> stopRanks:array<u32>;
	fn alpha_at(proj:Projection,point:vec2<f32>)->f32{
		if(proj.cameraPointValid.w<0.5){return 0.0;}
		let d=point-proj.screenConic0.xy;
		let q=proj.screenConic0.z*d.x*d.x+2.0*proj.screenConic0.w*d.x*d.y
			+proj.conicDepthAlpha.x*d.y*d.y;
		if(q<0.0||q>9.0){return 0.0;}
		let raw=proj.conicDepthAlpha.z*proj.conicDepthAlpha.w*exp(-0.5*q);
		return select(0.0,min(0.99,raw),raw>=cfg.alphaThreshold);
	}
	@compute @workgroup_size(16,16)
	fn raster_forward(@builtin(global_invocation_id) gid:vec3<u32>){
		if(gid.x>=cfg.width||gid.y>=cfg.height){return;}
		let pixel=gid.y*cfg.width+gid.x;let tile=(gid.y/cfg.tileSize)*cfg.tilesX+(gid.x/cfg.tileSize);
		let count=min(atomicLoad(&tileCounts[tile]),cfg.tileCapacity);
		let point=vec2<f32>((f32(gid.x)+0.5)/f32(cfg.height),(f32(gid.y)+0.5)/f32(cfg.height));
		var color=vec3<f32>(0.0);var transmittance=1.0;var stop=count;
		// Depth-sorted source-over is the model's occlusion/transmittance law.
		// A softmax over contributors would normalize away this visibility state.
		for(var rank=0u;rank<count;rank++){
			if(rank%cfg.checkpointStride==0u){
				write_checkpoint(pixel*cfg.blocksPerTile+rank/cfg.checkpointStride,vec4<f32>(color,transmittance));
			}
			let id=pairData[tile*cfg.tileCapacity+rank];let alpha=alpha_at(projections[id],point);
			color+=transmittance*alpha*params[id].colorOpacity.xyz;transmittance*=1.0-alpha;
			if(transmittance<cfg.transmittanceThreshold){stop=rank+1u;break;}
		}
		// Randomizing only the train underlay breaks the color/opacity shortcut
		// without injecting a camera image. Alpha remains true splat coverage.
		let background=training_background(cfg.trainingBackgroundPacked);
		rendered[pixel]=vec4<f32>(color+transmittance*background,1.0-transmittance);
		stopRanks[pixel]=stop;
	}`;
}

function ssimStatsWgsl() {
	return `${CONFIG_WGSL}
	struct SsimStats { muX:vec4<f32>,muY:vec4<f32>,varX:vec4<f32>,varY:vec4<f32>,cov:vec4<f32> };
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> rendered:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read> targets:array<vec4<f32>>;
	@group(0) @binding(3) var<storage,read_write> stats:array<SsimStats>;
	@group(0) @binding(4) var<storage,read_write> pixelLoss:array<vec4<f32>>;
	fn loss_weight(targetPixel:vec4<f32>)->f32 {
		return select(1.0,targetPixel.w,cfg.motionWeighting!=0u);
	}
	fn reflect_index(index:i32,size:u32)->u32 {
		if(size<=1u){return 0u;}var resolved=index;let maximum=i32(size)-1;
		loop{
			if(resolved>=0&&resolved<=maximum){break;}
			if(resolved<0){resolved=-resolved;}
			if(resolved>maximum){resolved=2*maximum-resolved;}
		}
		return u32(resolved);
	}
	fn ssim_weight(offset:i32,radius:i32)->f32 {
		if(radius==0){return 1.0;}
		if(radius!=5){return 1.0/f32(radius*2+1);}
		switch abs(offset) {
			case 0: { return 0.2660117149; }
			case 1: { return 0.2130055428; }
			case 2: { return 0.1093606874; }
			case 3: { return 0.0360007733; }
			case 4: { return 0.0075987582; }
			case 5: { return 0.0010283801; }
			default: { return 0.0; }
		}
	}
	@compute @workgroup_size(64)
	fn ssim_stats(@builtin(global_invocation_id) gid:vec3<u32>){
		let pixel=gid.x;if(pixel>=cfg.pixelCount){return;}let x=i32(pixel%cfg.width);let y=i32(pixel/cfg.width);
		var mux=vec3<f32>(0.0);var muy=vec3<f32>(0.0);
		var ex2=vec3<f32>(0.0);var ey2=vec3<f32>(0.0);var exy=vec3<f32>(0.0);
		let radius=i32(cfg.ssimRadius);
		for(var oy=-radius;oy<=radius;oy++){
			let sy=reflect_index(y+oy,cfg.height);
			for(var ox=-radius;ox<=radius;ox++){
				let sx=reflect_index(x+ox,cfg.width);let sample=sy*cfg.width+sx;
				let px=rendered[sample].xyz;let py=targets[cfg.targetOffset+sample].xyz;
				let weight=ssim_weight(oy,radius)*ssim_weight(ox,radius);
				mux+=weight*px;muy+=weight*py;ex2+=weight*px*px;
				ey2+=weight*py*py;exy+=weight*px*py;
			}
		}
		let vx=ex2-mux*mux;
		let vy=ey2-muy*muy;let covariance=exy-mux*muy;
		stats[pixel]=SsimStats(vec4<f32>(mux,0.0),vec4<f32>(muy,0.0),vec4<f32>(vx,0.0),
			vec4<f32>(vy,0.0),vec4<f32>(covariance,0.0));
		let a=2.0*mux*muy+vec3<f32>(cfg.c1);let b=2.0*covariance+vec3<f32>(cfg.c2);
		let c=mux*mux+muy*muy+vec3<f32>(cfg.c1);let d=vx+vy+vec3<f32>(cfg.c2);
		let ssim=(a*b)/max(c*d,vec3<f32>(1e-12));
		let targetPixel=targets[cfg.targetOffset+pixel];
		let objectiveWeight=loss_weight(targetPixel);
		let err=rendered[pixel].xyz-targetPixel.xyz;
		pixelLoss[pixel]=vec4<f32>(objectiveWeight*(abs(err.x)+abs(err.y)+abs(err.z))/3.0,
			objectiveWeight*(1.0-(ssim.x+ssim.y+ssim.z)/3.0),rendered[pixel].w,0.0);
	}`;
}

function ssimGradientWgsl() {
	return `${CONFIG_WGSL}
	struct SsimStats { muX:vec4<f32>,muY:vec4<f32>,varX:vec4<f32>,varY:vec4<f32>,cov:vec4<f32> };
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> rendered:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read> targets:array<vec4<f32>>;
	@group(0) @binding(3) var<storage,read> stats:array<SsimStats>;
	@group(0) @binding(4) var<storage,read> stopRanks:array<u32>;
	@group(0) @binding(5) var<storage,read_write> pixelGrad:array<vec4<f32>>;
	fn loss_weight(targetPixel:vec4<f32>)->f32 {
		return select(1.0,targetPixel.w,cfg.motionWeighting!=0u);
	}
	fn ssim_weight(offset:i32,radius:i32)->f32 {
		if(radius==0){return 1.0;}
		if(radius!=5){return 1.0/f32(radius*2+1);}
		switch abs(offset) {
			case 0: { return 0.2660117149; }
			case 1: { return 0.2130055428; }
			case 2: { return 0.1093606874; }
			case 3: { return 0.0360007733; }
			case 4: { return 0.0075987582; }
			case 5: { return 0.0010283801; }
			default: { return 0.0; }
		}
	}
	fn reflected_weight(center:i32,pixel:i32,size:i32,radius:i32)->f32 {
		var weight=select(0.0,ssim_weight(pixel-center,radius),abs(center-pixel)<=radius);
		if(pixel>0&&abs(center+pixel)<=radius){weight+=ssim_weight(-pixel-center,radius);}
		let maximum=size-1;
		let rightOffset=2*maximum-pixel-center;
		if(pixel<maximum&&abs(rightOffset)<=radius){weight+=ssim_weight(rightOffset,radius);}
		return weight;
	}
	@compute @workgroup_size(64)
	fn ssim_gradient(@builtin(global_invocation_id) gid:vec3<u32>){
		let pixel=gid.x;if(pixel>=cfg.pixelCount){return;}
		let px=i32(pixel%cfg.width);let py=i32(pixel/cfg.width);
		let prediction=rendered[pixel].xyz;let targetColor=targets[cfg.targetOffset+pixel].xyz;
		var dssim=vec3<f32>(0.0);let radius=i32(cfg.ssimRadius);
		for(var cy=max(0,py-radius);cy<=min(i32(cfg.height)-1,py+radius);cy++){
			let yWeight=reflected_weight(cy,py,i32(cfg.height),radius);
			for(var cx=max(0,px-radius);cx<=min(i32(cfg.width)-1,px+radius);cx++){
				let center=u32(cy)*cfg.width+u32(cx);
				let weight=loss_weight(targets[cfg.targetOffset+center])
					*yWeight*reflected_weight(cx,px,i32(cfg.width),radius);
				if(weight==0.0){continue;}
				let s=stats[center];
				let mx=s.muX.xyz;let my=s.muY.xyz;let vx=s.varX.xyz;let vy=s.varY.xyz;let covariance=s.cov.xyz;
				let a=2.0*mx*my+vec3<f32>(cfg.c1);let b=2.0*covariance+vec3<f32>(cfg.c2);
				let c=mx*mx+my*my+vec3<f32>(cfg.c1);let d=vx+vy+vec3<f32>(cfg.c2);
				let da=2.0*my*weight;let db=2.0*weight*(targetColor-my);
				let dc=2.0*mx*weight;let dd=2.0*weight*(prediction-mx);
				let denominatorRaw=c*d;let denominator=max(denominatorRaw,vec3<f32>(1e-12));
				let dDenominator=select(dc*d+c*dd,vec3<f32>(0.0),denominatorRaw<vec3<f32>(1e-12));
				dssim-=(((da*b+a*db)*denominator)-(a*b)*dDenominator)
					/(denominator*denominator*f32(cfg.pixelCount)*3.0);
			}
		}
		let l1=loss_weight(targets[cfg.targetOffset+pixel])
			*sign(prediction-targetColor)/(f32(cfg.pixelCount)*3.0);
		pixelGrad[pixel]=vec4<f32>(cfg.l1Weight*l1+cfg.dssimWeight*dssim,
			f32(stopRanks[pixel]));
	}`;
}

function metricsWgsl() {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> pixelLoss:array<vec4<f32>>;
	@group(0) @binding(2) var<storage,read_write> counters:array<atomic<u32>>;
	@group(0) @binding(3) var<storage,read_write> metrics:array<vec4<f32>>;
	var<workgroup> scratch:array<vec4<f32>,256>;
	@compute @workgroup_size(256)
	fn reduce_metrics(@builtin(local_invocation_id) lid:vec3<u32>){
		var total=vec4<f32>(0.0);
		for(var pixel=lid.x;pixel<cfg.pixelCount;pixel+=256u){total+=pixelLoss[pixel];}
		scratch[lid.x]=total;workgroupBarrier();
		var stride=128u;loop{
			if(lid.x<stride){scratch[lid.x]+=scratch[lid.x+stride];}
			workgroupBarrier();if(stride==1u){break;}stride/=2u;
		}
		if(lid.x==0u){
			let mean=scratch[0]/f32(cfg.pixelCount);
			metrics[0]=vec4<f32>(cfg.l1Weight*mean.x+cfg.dssimWeight*mean.y,mean.x,mean.y,
				f32(atomicLoad(&counters[1])));
		}
	}`;
}

function backwardWgsl(checkpointPrecision) {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> params:array<Splat>;
	@group(0) @binding(2) var<storage,read> projections:array<Projection>;
	@group(0) @binding(3) var<storage,read> pairData:array<u32>;
	@group(0) @binding(4) var<storage,read> rendered:array<vec4<f32>>;
	${checkpointBackwardWgsl(checkpointPrecision)}
	@group(0) @binding(6) var<storage,read> pixelGrad:array<vec4<f32>>;
	@group(0) @binding(7) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	@group(0) @binding(8) var<storage,read_write> counters:array<atomic<u32>>;
	var<workgroup> gradientScratch:array<Splat,256>;
	fn zero_splat()->Splat{return Splat(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));}
	fn add_splat(a:Splat,b:Splat)->Splat{return Splat(a.centerStatic+b.centerStatic,a.velocityTime+b.velocityTime,
		a.harmonicPad+b.harmonicPad,a.logScalePad+b.logScalePad,a.rotation+b.rotation,a.colorOpacity+b.colorOpacity);}
	fn atomic_add_f32(index:u32,value:f32){
		if(value==0.0){return;}
		var oldBits=atomicLoad(&gradientAtoms[index]);
		loop{
			let newBits=bitcast<u32>(bitcast<f32>(oldBits)+value);
			let result=atomicCompareExchangeWeak(&gradientAtoms[index],oldBits,newBits);
			if(result.exchanged){break;}
			oldBits=result.old_value;
		}
	}
	fn accumulate_splat(id:u32,gradient:Splat){
		let base=id*24u;
		for(var component=0u;component<4u;component++){
			atomic_add_f32(base+component,gradient.centerStatic[component]);
			atomic_add_f32(base+4u+component,gradient.velocityTime[component]);
			atomic_add_f32(base+8u+component,gradient.harmonicPad[component]);
			atomic_add_f32(base+12u+component,gradient.logScalePad[component]);
			atomic_add_f32(base+16u+component,gradient.rotation[component]);
			atomic_add_f32(base+20u+component,gradient.colorOpacity[component]);
		}
	}
	fn alpha_at(proj:Projection,point:vec2<f32>)->f32{
		if(proj.cameraPointValid.w<0.5){return 0.0;}let d=point-proj.screenConic0.xy;
		let q=proj.screenConic0.z*d.x*d.x+2.0*proj.screenConic0.w*d.x*d.y+proj.conicDepthAlpha.x*d.y*d.y;
		if(q<0.0||q>9.0){return 0.0;}
		let raw=proj.conicDepthAlpha.z*proj.conicDepthAlpha.w*exp(-0.5*q);
		return select(0.0,min(0.99,raw),raw>=cfg.alphaThreshold);
	}
	@compute @workgroup_size(16,16)
	fn pair_backward(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
		let lane=lid.y*16u+lid.x;let pair=wid.y*${MAX_WORKGROUPS_PER_DIMENSION}u+wid.x;
		let pairValid=pair<atomicLoad(&counters[0]);var id=0u;var gradient=zero_splat();
		if(pairValid){
			let slot=pairData[cfg.pairCapacity+pair];let tile=slot/cfg.tileCapacity;
			let rank=slot%cfg.tileCapacity;id=pairData[slot];let tileX=tile%cfg.tilesX;let tileY=tile/cfg.tilesX;
			let x=tileX*cfg.tileSize+lid.x;let y=tileY*cfg.tileSize+lid.y;
			if(x<cfg.width&&y<cfg.height){
				let pixel=y*cfg.width+x;
				if(rank<u32(pixelGrad[pixel].w)){
					let point=vec2<f32>((f32(x)+0.5)/f32(cfg.height),(f32(y)+0.5)/f32(cfg.height));
					let block=rank/cfg.checkpointStride;
					let checkpoint=read_checkpoint(pixel*cfg.blocksPerTile+block);
					var before=checkpoint.xyz;var transmittance=checkpoint.w;
					for(var replay=block*cfg.checkpointStride;replay<rank;replay++){
						let prior=pairData[tile*cfg.tileCapacity+replay];let alpha=alpha_at(projections[prior],point);
						before+=transmittance*alpha*params[prior].colorOpacity.xyz;transmittance*=1.0-alpha;
					}
					let p=params[id];let proj=projections[id];let d=point-proj.screenConic0.xy;
					let qform=proj.screenConic0.z*d.x*d.x+2.0*proj.screenConic0.w*d.x*d.y+proj.conicDepthAlpha.x*d.y*d.y;
					if(qform>=0.0&&qform<=9.0&&transmittance>cfg.transmittanceThreshold){
						let gaussian=exp(-0.5*qform);
						let rawAlpha=proj.conicDepthAlpha.z*proj.conicDepthAlpha.w*gaussian;
						let alpha=select(0.0,min(0.99,rawAlpha),rawAlpha>=cfg.alphaThreshold);
						let denominator=transmittance*(1.0-alpha);
						// rendered.rgb already includes the train background, so
						// replay recovers deeper splats plus that same underlay.
						let behind=select(vec3<f32>(0.0),(rendered[pixel].xyz-before-transmittance*alpha*p.colorOpacity.xyz)
							/max(denominator,1e-8),denominator>1e-8);
						let imageGrad=pixelGrad[pixel].xyz;
						let alphaGrad=dot(imageGrad,transmittance*(p.colorOpacity.xyz-behind));
						let clampGate=select(0.0,1.0,rawAlpha<0.99&&rawAlpha>=cfg.alphaThreshold);
						let barQform=-0.5*alphaGrad*rawAlpha*clampGate;
						let conicDelta=vec2<f32>(proj.screenConic0.z*d.x+proj.screenConic0.w*d.y,
							proj.screenConic0.w*d.x+proj.conicDepthAlpha.x*d.y);
						let barMu=-2.0*barQform*conicDelta;
						let barC00=-barQform*conicDelta.x*conicDelta.x;
						let barC01=-barQform*conicDelta.x*conicDelta.y;
						let barC11=-barQform*conicDelta.y*conicDelta.y;
						let j0=proj.jacobian0.xyz;let j1=proj.jacobian1.xyz;
						let barSigma=barC00*outer3(j0,j0)+barC01*(outer3(j0,j1)+outer3(j1,j0))+barC11*outer3(j1,j1);
						let basis=mat3x3<f32>(proj.basis0.xyz,proj.basis1.xyz,proj.basis2.xyz);
						let variances=proj.variancesPad.xyz;
						let sigmaCamera=variances.x*outer3(basis[0],basis[0])+variances.y*outer3(basis[1],basis[1])
							+variances.z*outer3(basis[2],basis[2]);
						let sigmaJ0=sigmaCamera*j0;let sigmaJ1=sigmaCamera*j1;
						let barJ0=2.0*(barC00*sigmaJ0+barC01*sigmaJ1);
						let barJ1=2.0*(barC01*sigmaJ0+barC11*sigmaJ1);
						let cp=proj.cameraPointValid.xyz;let invZ=1.0/cp.z;
						let horizontalFocal=proj.jacobian0.x*cp.z;let verticalFocal=proj.jacobian1.y*cp.z;
						let cameraGrad=vec3<f32>(
							barMu.x*horizontalFocal*invZ-barJ0.z*horizontalFocal*invZ*invZ,
							barMu.y*verticalFocal*invZ-barJ1.z*verticalFocal*invZ*invZ,
							-barMu.x*horizontalFocal*cp.x*invZ*invZ-barMu.y*verticalFocal*cp.y*invZ*invZ
							-barJ0.x*horizontalFocal*invZ*invZ+barJ0.z*2.0*horizontalFocal*cp.x*invZ*invZ*invZ
							-barJ1.y*verticalFocal*invZ*invZ+barJ1.z*2.0*verticalFocal*cp.y*invZ*invZ*invZ);
						let cameraRotation=mat3x3<f32>(proj.camera0.xyz,proj.camera1.xyz,proj.camera2.xyz);
						let worldGrad=transpose(cameraRotation)*cameraGrad;
						var gradLogScale=vec3<f32>(0.0);
						for(var axis=0u;axis<3u;axis++){let column=basis[axis];
							gradLogScale[axis]=2.0*variances[axis]*dot(column,barSigma*column);}
						let barBasis=mat3x3<f32>(2.0*variances.x*(barSigma*basis[0]),
							2.0*variances.y*(barSigma*basis[1]),2.0*variances.z*(barSigma*basis[2]));
						let barRotation=transpose(cameraRotation)*barBasis;let q=safe_quaternion(p.rotation);
						let h00=barRotation[0].x;let h01=barRotation[1].x;let h02=barRotation[2].x;
						let h10=barRotation[0].y;let h11=barRotation[1].y;let h12=barRotation[2].y;
						let h20=barRotation[0].z;let h21=barRotation[1].z;let h22=barRotation[2].z;
						let normalizedQuatGrad=vec4<f32>(
							-4.0*q.x*(h11+h22)+2.0*q.y*(h01+h10)+2.0*q.z*(h02+h20)+2.0*q.w*(h21-h12),
							-4.0*q.y*(h00+h22)+2.0*q.x*(h01+h10)+2.0*q.z*(h12+h21)+2.0*q.w*(h02-h20),
							-4.0*q.z*(h00+h11)+2.0*q.x*(h02+h20)+2.0*q.y*(h12+h21)+2.0*q.w*(h10-h01),
							2.0*q.z*(h10-h01)+2.0*q.y*(h02-h20)+2.0*q.x*(h21-h12));
						let rawNorm2=dot(p.rotation,p.rotation);var gradRotation=vec4<f32>(0.0);
						if(rawNorm2>1e-16){gradRotation=(normalizedQuatGrad-q*dot(q,normalizedQuatGrad))*inverseSqrt(rawNorm2);}
						let staticWarmup=cfg.staticWarmup!=0u;let t=frame_time(cfg);
						let tc=select(t*2.0-1.0,0.0,staticWarmup);
						let wave=select(sin(t*6.28318530718),0.0,staticWarmup);
						let sigma=clamp(cfg.temporalSigma,0.12,0.36);let staticMix=clamp(p.centerStatic.w,0.0,1.0);
						let temporalFloor=clamp(sigma*0.30,0.035,0.12);let timeDelta=t-clamp(p.velocityTime.w,0.0,1.0);
						let dynamicGate=temporalFloor+(1.0-temporalFloor)*exp(-0.5*timeDelta*timeDelta/(sigma*sigma));
						let dynamicCore=max(0.0,(proj.conicDepthAlpha.w-staticMix-temporalFloor*(1.0-staticMix))
							/max(1e-6,1.0-staticMix));
						let gradTime=select(clampGate*alphaGrad*proj.conicDepthAlpha.z*gaussian
							*(1.0-staticMix)*dynamicCore*(t-p.velocityTime.w)/(sigma*sigma),0.0,staticWarmup);
						let gradStaticMix=select(
							clampGate*alphaGrad*proj.conicDepthAlpha.z*gaussian*(1.0-dynamicGate),
							0.0,staticWarmup);
						let gradOpacity=clampGate*alphaGrad*gaussian*proj.conicDepthAlpha.w
							*proj.conicDepthAlpha.z*(1.0-proj.conicDepthAlpha.z);
						let gradColor=imageGrad*transmittance*alpha;
						gradient=Splat(vec4<f32>(worldGrad,gradStaticMix),vec4<f32>(worldGrad*tc,gradTime),
							vec4<f32>(select(vec3<f32>(0.0),worldGrad*wave,cfg.modelMode==0u),
								alpha/f32(cfg.pixelCount)),vec4<f32>(gradLogScale,length(barMu)),gradRotation,
							vec4<f32>(gradColor,gradOpacity));
					}
				}
			}
		}
		gradientScratch[lane]=gradient;workgroupBarrier();
		var stride=128u;loop{
			if(lane<stride){gradientScratch[lane]=add_splat(gradientScratch[lane],gradientScratch[lane+stride]);}
			workgroupBarrier();if(stride==1u){break;}stride/=2u;
		}
		if(lane==0u&&pairValid){accumulate_splat(id,gradientScratch[0]);}
	}`;
}

function updateWgsl() {
	return `${CONFIG_WGSL}
	@group(0) @binding(0) var<uniform> cfg:TiledConfig;
	@group(0) @binding(1) var<storage,read> paramsIn:array<Splat>;
	@group(0) @binding(2) var<storage,read_write> paramsOut:array<Splat>;
	@group(0) @binding(3) var<storage,read_write> firstMoment:array<Splat>;
	@group(0) @binding(4) var<storage,read_write> secondMoment:array<Splat>;
	@group(0) @binding(5) var<storage,read_write> splatStats:array<vec4<f32>>;
	@group(0) @binding(6) var<storage,read_write> gradientAtoms:array<atomic<u32>>;
	fn load_gradient(id:u32)->Splat{
		let base=id*24u;
		return Splat(
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+1u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+2u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+3u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+4u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+5u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+6u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+7u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+8u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+9u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+10u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+11u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+12u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+13u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+14u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+15u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+16u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+17u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+18u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+19u]))),
			vec4<f32>(bitcast<f32>(atomicLoad(&gradientAtoms[base+20u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+21u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+22u])),
				bitcast<f32>(atomicLoad(&gradientAtoms[base+23u])))
		);
	}
	@compute @workgroup_size(64)
	fn reduce_update(@builtin(global_invocation_id) gid:vec3<u32>){
		let i=gid.x;if(i>=cfg.splatCount){return;}
		var gradient=load_gradient(i);
		let meanAlpha=gradient.harmonicPad.w;gradient.harmonicPad.w=0.0;
		let screenGradient=gradient.logScalePad.w;gradient.logScalePad.w=0.0;
		var p=paramsIn[i];var m=firstMoment[i];var v=secondMoment[i];
		m.centerStatic=cfg.beta1*m.centerStatic+(1.0-cfg.beta1)*gradient.centerStatic;
		m.velocityTime=cfg.beta1*m.velocityTime+(1.0-cfg.beta1)*gradient.velocityTime;
		m.harmonicPad=cfg.beta1*m.harmonicPad+(1.0-cfg.beta1)*gradient.harmonicPad;
		m.logScalePad=cfg.beta1*m.logScalePad+(1.0-cfg.beta1)*gradient.logScalePad;
		m.rotation=cfg.beta1*m.rotation+(1.0-cfg.beta1)*gradient.rotation;
		m.colorOpacity=cfg.beta1*m.colorOpacity+(1.0-cfg.beta1)*gradient.colorOpacity;
		v.centerStatic=cfg.beta2*v.centerStatic+(1.0-cfg.beta2)*gradient.centerStatic*gradient.centerStatic;
		v.velocityTime=cfg.beta2*v.velocityTime+(1.0-cfg.beta2)*gradient.velocityTime*gradient.velocityTime;
		v.harmonicPad=cfg.beta2*v.harmonicPad+(1.0-cfg.beta2)*gradient.harmonicPad*gradient.harmonicPad;
		v.logScalePad=cfg.beta2*v.logScalePad+(1.0-cfg.beta2)*gradient.logScalePad*gradient.logScalePad;
		v.rotation=cfg.beta2*v.rotation+(1.0-cfg.beta2)*gradient.rotation*gradient.rotation;
		v.colorOpacity=cfg.beta2*v.colorOpacity+(1.0-cfg.beta2)*gradient.colorOpacity*gradient.colorOpacity;
		firstMoment[i]=m;secondMoment[i]=v;let adamStep=f32(cfg.step+1u);
		let mc=max(1e-6,1.0-pow(cfg.beta1,adamStep));let vc=max(1e-6,1.0-pow(cfg.beta2,adamStep));
		let posUpdate=(m.centerStatic/mc)/(sqrt(v.centerStatic/vc)+vec4<f32>(cfg.adamEpsilon));
		let velocityUpdate=(m.velocityTime/mc)/(sqrt(v.velocityTime/vc)+vec4<f32>(cfg.adamEpsilon));
		let harmonicUpdate=(m.harmonicPad/mc)/(sqrt(v.harmonicPad/vc)+vec4<f32>(cfg.adamEpsilon));
		let scaleUpdate=(m.logScalePad/mc)/(sqrt(v.logScalePad/vc)+vec4<f32>(cfg.adamEpsilon));
		let rotationUpdate=(m.rotation/mc)/(sqrt(v.rotation/vc)+vec4<f32>(cfg.adamEpsilon));
		let colorUpdate=(m.colorOpacity/mc)/(sqrt(v.colorOpacity/vc)+vec4<f32>(cfg.adamEpsilon));
		p.centerStatic=vec4<f32>(p.centerStatic.xyz-cfg.lrPosition*posUpdate.xyz,
			clamp(p.centerStatic.w-cfg.lrMotion*posUpdate.w,0.0,1.0));
		p.velocityTime=vec4<f32>(clamp(p.velocityTime.xyz-cfg.lrMotion*velocityUpdate.xyz,
			vec3<f32>(-2.0*cfg.geometryScale),vec3<f32>(2.0*cfg.geometryScale)),
			clamp(p.velocityTime.w-cfg.lrMotion*velocityUpdate.w,0.0,1.0));
		p.harmonicPad=vec4<f32>(clamp(p.harmonicPad.xyz-cfg.lrMotion*harmonicUpdate.xyz,
			vec3<f32>(-1.5*cfg.geometryScale),vec3<f32>(1.5*cfg.geometryScale)),p.harmonicPad.w);
		var nextLogScale=clamp(p.logScalePad.xyz-${SCALE_LR_FROM_COLOR}*cfg.lrColor*scaleUpdate.xyz,
			vec3<f32>(log(cfg.minScale)),vec3<f32>(log(cfg.maxScale)));
		let meanLog=(nextLogScale.x+nextLogScale.y+nextLogScale.z)/3.0;
		// Center the trust region in log scale so the ratio bound is symmetric
		// across axes and does not select a preferred world direction.
		let halfLogAspect=0.5*log(${MAX_SCALE_ASPECT_RATIO}.0);
		nextLogScale=clamp(nextLogScale,vec3<f32>(meanLog-halfLogAspect),vec3<f32>(meanLog+halfLogAspect));
		p.logScalePad=vec4<f32>(nextLogScale,p.logScalePad.w);
		let rotationTrial=p.rotation-${ROTATION_LR_FROM_MOTION}*cfg.lrMotion*rotationUpdate;
		let rotationNorm2=dot(rotationTrial,rotationTrial);
		p.rotation=select(vec4<f32>(0.0,0.0,0.0,1.0),rotationTrial*inverseSqrt(max(rotationNorm2,1e-16)),rotationNorm2>1e-16);
		p.colorOpacity=vec4<f32>(clamp(p.colorOpacity.xyz-cfg.lrColor*colorUpdate.xyz,
			vec3<f32>(0.0),vec3<f32>(${MAX_SPLAT_COLOR}.0)),
			clamp(p.colorOpacity.w-cfg.lrOpacity*colorUpdate.w,-12.0,3.0));
		paramsOut[i]=p;let observed=vec4<f32>(screenGradient,meanAlpha,
			abs(gradient.colorOpacity.w),length(gradient.velocityTime.xyz));
		splatStats[i]=cfg.statDecay*splatStats[i]+(1.0-cfg.statDecay)*observed;
	}`;
}

async function checkedModule(device, name, code) {
	const module = device.createShaderModule({ label: name, code });
	const info = await module.getCompilationInfo();
	const errors = info.messages.filter((message) => message.type === "error")
		.map((message) => `${name}:${message.lineNum}:${message.linePos} ${message.message}`);
	if (errors.length) throw new Error(`WGSL compilation failed:\n${errors.join("\n")}`);
	return module;
}

export class DynamicSplatWebGpu3dTiledTrainer extends DynamicSplatWebGpu3dTrainer {
	constructor(canvas) {
		super(canvas);
		this.initialSplatCount = 1536;
		this.skipSampleGradientAllocation = true;
		this.tiledConfigBytes = new ArrayBuffer(TILED_CONFIG_BYTES);
	}

	targetBufferByteLength(dataset) {
		return rgbaFloatFrameBytes(dataset);
	}

	uploadTargetPage(target, viewIndex, frameIndex, { staticWarmup = false } = {}) {
		const pixelCount = this.dataset.width * this.dataset.height;
		const sourcePixelOffset = staticWarmup
			? viewIndex * pixelCount
			: (viewIndex * this.dataset.frameCount + frameIndex) * pixelCount;
		const pageKey = staticWarmup ? `background:${viewIndex}` : `frame:${viewIndex}:${frameIndex}`;
		if (this.targetPageKey !== pageKey) {
			const sourceElementOffset = sourcePixelOffset * 4;
			const source = staticWarmup ? this.dataset.backgrounds : this.dataset.frames;
			this.device.queue.writeBuffer(target, 0, source.subarray(
				sourceElementOffset,
				sourceElementOffset + pixelCount * 4,
			));
			this.targetPageKey = pageKey;
		}
		return sourcePixelOffset;
	}

	initializeTargetBuffer(target) {
		this.targetPageKey = null;
		this.uploadTargetPage(target, this.trainViewIndices[0], 0);
	}

	async init(dataset, {
		splatCount = 1536,
		growthCapacity = null,
		tileCapacity = null,
		checkpointPrecision = DEFAULT_CHECKPOINT_PRECISION,
		staticWarmupSteps = 0,
	} = {}) {
		this.initialSplatCount = splatCount;
		this.requestedTileCapacity = tileCapacity;
		this.checkpointPrecision = resolveCheckpointPrecision(checkpointPrecision);
		this.staticWarmupSteps = resolveStaticWarmupSteps(staticWarmupSteps);
		const capacity = resolveTiledCapacity(splatCount, growthCapacity);
		await super.init(dataset, { splatCount: capacity, requiredWorkgroupStorageSize: 24576 });
		this.adapterName = `${this.adapterName} · tiled full-frame`;
	}

	async createPipelines() {
		await super.createPipelines();
		this.tileCapacity = resolveTileCapacity(this.splatCount, this.requestedTileCapacity);
		const modules = await Promise.all([
			checkedModule(this.device, "tiled-clear", clearWgsl()),
			checkedModule(this.device, "tiled-project", projectWgsl()),
			checkedModule(this.device, "tiled-sort", sortWgsl(this.tileCapacity)),
			checkedModule(this.device, "tiled-finalize", finalizeWgsl()),
			checkedModule(this.device, "tiled-forward", forwardWgsl(this.checkpointPrecision)),
			checkedModule(this.device, "tiled-ssim-stats", ssimStatsWgsl()),
			checkedModule(this.device, "tiled-ssim-gradient", ssimGradientWgsl()),
			checkedModule(this.device, "tiled-metrics", metricsWgsl()),
			checkedModule(this.device, "tiled-backward", backwardWgsl(this.checkpointPrecision)),
			checkedModule(this.device, "tiled-update", updateWgsl()),
		]);
		const pipeline = (module, entryPoint) => this.device.createComputePipeline({
			label: `tiled-${entryPoint}`, layout: "auto", compute: { module, entryPoint },
		});
		this.device.pushErrorScope("validation");
		this.tiledPipelines = {
			clear: pipeline(modules[0], "clear_step"),
			project: pipeline(modules[1], "project_and_bin"),
			sort: pipeline(modules[2], "sort_tiles"),
			finalize: pipeline(modules[3], "finalize_pairs"),
			forward: pipeline(modules[4], "raster_forward"),
			ssimStats: pipeline(modules[5], "ssim_stats"),
			ssimGradient: pipeline(modules[6], "ssim_gradient"),
			metrics: pipeline(modules[7], "reduce_metrics"),
			backward: pipeline(modules[8], "pair_backward"),
			update: pipeline(modules[9], "reduce_update"),
		};
		const pipelineError = await this.device.popErrorScope();
		if (pipelineError) throw new Error(`Tiled WebGPU pipeline validation failed: ${pipelineError.message}`);
	}

	createBuffers() {
		super.createBuffers();
		const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
		const makeBuffer = (size, bufferUsage = usage) => this.device.createBuffer({
			size: Math.max(4, Math.ceil(size / 4) * 4), usage: bufferUsage,
		});
		const active = makeInitialSplats(this.dataset, this.initialSplatCount);
		const initial = new Float32Array(this.splatCount * SPLAT_FLOATS);
		initial.set(active);
		for (let i = this.initialSplatCount; i < this.splatCount; i += 1) {
			const source = (i % this.initialSplatCount) * SPLAT_FLOATS;
			const base = i * SPLAT_FLOATS;
			initial.set(active.subarray(source, source + SPLAT_FLOATS), base);
			initial[base + 23] = -12;
		}
		this.initialParams = initial.slice();
		for (const params of this.buffers.params) this.device.queue.writeBuffer(params, 0, initial);
		this.buffers.sampleGradients.destroy();
		this.buffers.sampleGradients = makeBuffer(this.splatCount * SPLAT_BYTES);
		this.tilesX = ceilDiv(this.dataset.width, TILE_SIZE);
		this.tilesY = ceilDiv(this.dataset.height, TILE_SIZE);
		this.tileCount = this.tilesX * this.tilesY;
		this.pixelCount = this.dataset.width * this.dataset.height;
		this.pairCapacity = this.tileCount * this.tileCapacity;
		const checkpointLayout = resolveCheckpointLayout(
			this.pixelCount,
			this.tileCapacity,
			this.storageBufferLimit,
			this.checkpointPrecision === "packed-f16" ? 8 : 16,
		);
		this.checkpointStride = checkpointLayout.stride;
		this.blocksPerTile = checkpointLayout.blocksPerTile;
		const tiledBufferBytes = {
			pairData: this.pairCapacity * 8,
			checkpoints: checkpointLayout.byteLength,
			gradientAccumulator: this.splatCount * SPLAT_BYTES,
		};
		for (const [label, byteLength] of Object.entries(tiledBufferBytes)) {
			assertStorageBufferFits(`The tiled ${label} buffer`, byteLength, this.storageBufferLimit);
		}
		this.memoryPlan = Object.freeze({
			targetPageBytes: this.targetBufferByteLength(this.dataset),
			tileCapacity: this.tileCapacity,
			pairCapacity: this.pairCapacity,
			checkpointStride: this.checkpointStride,
			checkpointPrecision: this.checkpointPrecision,
			checkpointBytes: tiledBufferBytes.checkpoints,
			pairDataBytes: tiledBufferBytes.pairData,
			gradientAccumulatorBytes: tiledBufferBytes.gradientAccumulator,
			pairGradientBytes: 0,
			storageBufferLimit: this.storageBufferLimit,
			nativeShaderF16: this.supportsShaderF16,
			staticWarmupSteps: this.staticWarmupSteps,
		});
		Object.assign(this.buffers, {
			tiledConfig: makeBuffer(TILED_CONFIG_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
			projections: makeBuffer(this.splatCount * PROJECTION_BYTES),
			tileCounts: makeBuffer(this.tileCount * 4),
			pairData: makeBuffer(tiledBufferBytes.pairData),
			counters: makeBuffer(16),
			indirectArgs: makeBuffer(12, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT
				| GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC),
			renderedTrain: makeBuffer(this.pixelCount * 16),
			checkpoints: makeBuffer(tiledBufferBytes.checkpoints),
			stopRanks: makeBuffer(this.pixelCount * 4),
			ssimStats: makeBuffer(this.pixelCount * SSIM_STATS_BYTES),
			pixelLoss: makeBuffer(this.pixelCount * 16),
			pixelGrad: makeBuffer(this.pixelCount * 16),
			gradientAtoms: makeBuffer(tiledBufferBytes.gradientAccumulator),
			tiledMetrics: makeBuffer(16),
		});
	}

	createBindGroups() {
		super.createBindGroups();
		const group = (pipeline, entries, index = 0) => this.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(index), entries,
		});
		const buffer = (binding, value) => ({ binding, resource: { buffer: value } });
		this.tiledBindGroups = {
			clear: group(this.tiledPipelines.clear, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.tileCounts),
				buffer(2, this.buffers.counters), buffer(3, this.buffers.indirectArgs),
				buffer(4, this.buffers.tiledMetrics), buffer(5, this.buffers.gradientAtoms),
			]),
			project: this.buffers.params.map((params) => group(this.tiledPipelines.project, [
				buffer(0, this.buffers.tiledConfig), buffer(1, params), buffer(2, this.buffers.cameras),
				buffer(3, this.buffers.tileCounts), buffer(4, this.buffers.pairData),
				buffer(5, this.buffers.projections), buffer(6, this.buffers.counters),
			])),
			sort: group(this.tiledPipelines.sort, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.projections),
				buffer(2, this.buffers.tileCounts), buffer(3, this.buffers.pairData),
				buffer(4, this.buffers.counters),
			]),
			finalize: group(this.tiledPipelines.finalize, [
				buffer(0, this.buffers.counters), buffer(1, this.buffers.indirectArgs),
			]),
			forward: this.buffers.params.map((params) => group(this.tiledPipelines.forward, [
				buffer(0, this.buffers.tiledConfig), buffer(1, params), buffer(2, this.buffers.projections),
				buffer(3, this.buffers.tileCounts), buffer(4, this.buffers.pairData),
				buffer(5, this.buffers.renderedTrain), buffer(6, this.buffers.checkpoints),
				buffer(7, this.buffers.stopRanks),
			])),
			ssimStats: group(this.tiledPipelines.ssimStats, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.renderedTrain),
				buffer(2, this.buffers.target), buffer(3, this.buffers.ssimStats),
				buffer(4, this.buffers.pixelLoss),
			]),
			ssimGradient: group(this.tiledPipelines.ssimGradient, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.renderedTrain),
				buffer(2, this.buffers.target), buffer(3, this.buffers.ssimStats),
				buffer(4, this.buffers.stopRanks), buffer(5, this.buffers.pixelGrad),
			]),
			metrics: group(this.tiledPipelines.metrics, [
				buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.pixelLoss),
				buffer(2, this.buffers.counters), buffer(3, this.buffers.tiledMetrics),
			]),
			backward: this.buffers.params.map((params) => group(this.tiledPipelines.backward, [
				buffer(0, this.buffers.tiledConfig), buffer(1, params), buffer(2, this.buffers.projections),
				buffer(3, this.buffers.pairData), buffer(4, this.buffers.renderedTrain),
				buffer(5, this.buffers.checkpoints), buffer(6, this.buffers.pixelGrad),
				buffer(7, this.buffers.gradientAtoms), buffer(8, this.buffers.counters),
			])),
			update: [
				group(this.tiledPipelines.update, [
					buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.params[0]),
					buffer(2, this.buffers.params[1]), buffer(3, this.buffers.firstMoment),
					buffer(4, this.buffers.secondMoment), buffer(5, this.buffers.stats),
					buffer(6, this.buffers.gradientAtoms),
				]),
				group(this.tiledPipelines.update, [
					buffer(0, this.buffers.tiledConfig), buffer(1, this.buffers.params[1]),
					buffer(2, this.buffers.params[0]), buffer(3, this.buffers.firstMoment),
					buffer(4, this.buffers.secondMoment), buffer(5, this.buffers.stats),
					buffer(6, this.buffers.gradientAtoms),
				]),
			],
		};
	}

	encodePass(encoder, pipeline, bindGroup, x, y = 1, z = 1) {
		const pass = encoder.beginComputePass();
		pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(x, y, z); pass.end();
	}

	trainStep({ learningRate = 1, learningRateDecay = false, modelMode = 0,
		temporalSigma = 0.30, ssimRadius = 5, motionWeighting = false,
		randomBackground = false } = {}) {
		const validateSubmission = this.stepCount === 0;
		const resolvedSsimRadius = resolveSsimRadius(ssimRadius);
		const rates = browserLearningRates(learningRate, this.stepCount, learningRateDecay);
		this.lastLearningRateMultipliers = {
			geometry: rates.geometry,
			appearance: rates.appearance,
			progress: rates.progress,
		};
		if (validateSubmission) this.device.pushErrorScope("validation");
		const selected = trainingPairForStep(
			this.trainViewIndices,
			this.dataset.frameCount,
			this.stepCount,
			this.staticWarmupSteps,
		);
		this.lastCameraBatch = [selected.viewIndex]; this.lastCameraBatchStart = selected.viewSlot;
		this.lastFrameIndex = selected.frameIndex;
		this.lastTrainingPhase = selected.staticWarmup ? "static_warmup" : "dynamic_fit";
		this.lastTargetSourceOffset = this.uploadTargetPage(
			this.buffers.target,
			selected.viewIndex,
			selected.frameIndex,
			{ staticWarmup: selected.staticWarmup },
		);
		const targetOffset = 0;
		this.lastTargetOffset = targetOffset;
		writeTiledConfig(this.tiledConfigBytes, {
			width: this.dataset.width, height: this.dataset.height, splatCount: this.splatCount,
			tilesX: this.tilesX, tilesY: this.tilesY, tileCapacity: this.tileCapacity,
			blocksPerTile: this.blocksPerTile, viewIndex: selected.viewIndex, frameIndex: selected.frameIndex,
			step: this.stepCount, modelMode, targetOffset, pixelCount: this.pixelCount,
			pairCapacity: this.pairCapacity, targetAspect: this.dataset.width / this.dataset.height,
			temporalSigma, alphaThreshold: 1 / 255, transmittanceThreshold: 1e-4,
			lrPosition: rates.position, lrColor: rates.color,
			lrOpacity: rates.opacity, lrMotion: rates.motion,
			geometryScale: this.dataset.geometryScale, l1Weight: 0.8, dssimWeight: 0.2,
			statDecay: DENSITY_STAT_DECAY, ssimRadius: resolvedSsimRadius,
			frameCount: this.dataset.frameCount,
			staticWarmup: selected.staticWarmup,
			motionWeighting,
			randomBackground,
			checkpointStride: this.checkpointStride,
		});
		this.device.queue.writeBuffer(this.buffers.tiledConfig, 0, this.tiledConfigBytes);
		const encoder = this.device.createCommandEncoder();
		this.encodePass(encoder, this.tiledPipelines.clear, this.tiledBindGroups.clear,
			ceilDiv(Math.max(this.tileCount, this.splatCount * SPLAT_FLOATS), 64));
		this.encodePass(encoder, this.tiledPipelines.project, this.tiledBindGroups.project[this.currentIndex],
			ceilDiv(this.splatCount, 64));
		this.encodePass(encoder, this.tiledPipelines.sort, this.tiledBindGroups.sort, this.tileCount);
		this.encodePass(encoder, this.tiledPipelines.finalize, this.tiledBindGroups.finalize, 1);
		this.encodePass(encoder, this.tiledPipelines.forward, this.tiledBindGroups.forward[this.currentIndex],
			this.tilesX, this.tilesY);
		this.encodePass(encoder, this.tiledPipelines.ssimStats, this.tiledBindGroups.ssimStats,
			ceilDiv(this.pixelCount, 64));
		this.encodePass(encoder, this.tiledPipelines.ssimGradient, this.tiledBindGroups.ssimGradient,
			ceilDiv(this.pixelCount, 64));
		this.encodePass(encoder, this.tiledPipelines.metrics, this.tiledBindGroups.metrics, 1);
		const backward = encoder.beginComputePass();
		backward.setPipeline(this.tiledPipelines.backward);
		backward.setBindGroup(0, this.tiledBindGroups.backward[this.currentIndex]);
		backward.dispatchWorkgroupsIndirect(this.buffers.indirectArgs, 0); backward.end();
		this.encodePass(encoder, this.tiledPipelines.update, this.tiledBindGroups.update[this.currentIndex],
			ceilDiv(this.splatCount, 64));
		const nextStep = this.stepCount + 1;
		const densityDispatches = densityDispatchesForStep(
			this.initialSplatCount, this.splatCount, nextStep);
		if (densityDispatches > 0) {
			const maintenance = encoder.beginComputePass();
			maintenance.setPipeline(this.pipelines.maintenance);
			maintenance.setBindGroup(0, this.bindGroups.maintenance[1 - this.currentIndex]);
			for (let pass = 0; pass < densityDispatches; pass += 1) maintenance.dispatchWorkgroups(1);
			maintenance.end(); this.totalRecycled += densityDispatches * 4;
		}
		this.device.queue.submit([encoder.finish()]);
		if (validateSubmission) {
			this.firstStepValidation = this.device.popErrorScope();
		}
		this.lastSampleCount = this.pixelCount;
		this.currentIndex = 1 - this.currentIndex; this.stepCount = nextStep;
	}

	async readLoss() {
		const submissionError = await this.firstStepValidation;
		this.firstStepValidation = null;
		if (submissionError) throw new Error(`Tiled full-frame submission failed: ${submissionError.message}`);
		this.device.pushErrorScope("validation");
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.buffers.tiledMetrics, 0, this.buffers.metricsReadback, 0, 16);
		this.device.queue.submit([encoder.finish()]);
		const readbackError = await this.device.popErrorScope();
		if (readbackError) throw new Error(`Tiled metric readback failed: ${readbackError.message}`);
		await this.buffers.metricsReadback.mapAsync(GPUMapMode.READ, 0, 16);
		const values = new Float32Array(this.buffers.metricsReadback.getMappedRange(0, 16).slice(0));
		this.buffers.metricsReadback.unmap();
		this.lastLossBreakdown = {
			loss: values[0], l1: values[1], dssim: values[2], tileOverflow: values[3],
		};
		return values[0];
	}

	async readTiledStepDebugStateUnlocked() {
		const renderedBytes = this.pixelCount * 16;
		const gradientBytes = this.splatCount * SPLAT_BYTES;
		const metricsOffset = renderedBytes + gradientBytes;
		const readback = this.device.createBuffer({
			label: "tiled-step-debug-readback",
			size: metricsOffset + 16,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});
		try {
			const encoder = this.device.createCommandEncoder();
			encoder.copyBufferToBuffer(this.buffers.renderedTrain, 0, readback, 0, renderedBytes);
			encoder.copyBufferToBuffer(this.buffers.gradientAtoms, 0, readback, renderedBytes, gradientBytes);
			encoder.copyBufferToBuffer(this.buffers.tiledMetrics, 0, readback, metricsOffset, 16);
			this.device.queue.submit([encoder.finish()]);
			await readback.mapAsync(GPUMapMode.READ);
			const bytes = readback.getMappedRange().slice(0);
			return {
				step: this.stepCount,
				viewIndex: this.lastCameraBatch?.[0] ?? null,
				frameIndex: this.lastFrameIndex ?? null,
				renderedRgba: new Float32Array(bytes, 0, this.pixelCount * 4).slice(),
				gradients: new Float32Array(bytes, renderedBytes, this.splatCount * SPLAT_FLOATS).slice(),
				metrics: new Float32Array(bytes, metricsOffset, 4).slice(),
			};
		} finally {
			if (readback.mapState === "mapped") readback.unmap();
			readback.destroy();
		}
	}

	readTiledStepDebugState() {
		const read = this.readbackChain.then(() => this.readTiledStepDebugStateUnlocked());
		this.readbackChain = read.then(() => undefined, () => undefined);
		return read;
	}
}
