function normalizeKey(path) {
	return path.replace(/^\.?\//, "").replace(/\\/g, "/");
}

function buildFileIndex(files) {
	const index = new Map();
	for (const file of files) {
		const relative = normalizeKey(file.webkitRelativePath || file.name);
		index.set(relative, file);
		const parts = relative.split("/");
		index.set(parts[parts.length - 1], file);
	}
	return index;
}

async function readJsonFile(file) {
	return JSON.parse(await file.text());
}

async function readJsonUrl(url) {
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
	}
	return response.json();
}

async function loadTensorFromFile(fileIndex, entry) {
	const file = fileIndex.get(normalizeKey(entry.path));
	if (!file) {
		throw new Error(`Bundle file missing: ${entry.path}`);
	}
	const buffer = await file.arrayBuffer();
	const data = new Float32Array(buffer);
	if (typeof entry.count === "number" && data.length !== entry.count) {
		throw new Error(
			`Tensor ${entry.path} expected ${entry.count} float32 values, found ${data.length}.`,
		);
	}
	return { data, shape: entry.shape };
}

async function loadTensorFromUrl(baseUrl, entry) {
	const response = await fetch(`${baseUrl}/${entry.path}`);
	if (!response.ok) {
		throw new Error(
			`Failed to fetch tensor ${entry.path} from ${baseUrl}: HTTP ${response.status}`,
		);
	}
	const buffer = await response.arrayBuffer();
	const data = new Float32Array(buffer);
	if (typeof entry.count === "number" && data.length !== entry.count) {
		throw new Error(
			`Tensor ${entry.path} expected ${entry.count} float32 values, found ${data.length}.`,
		);
	}
	return { data, shape: entry.shape };
}

async function loadBundleWithTensorLoader(manifest, loadTensor) {
	if (manifest.version !== "dynaworld_token_head_bundle/v2") {
		throw new Error(
			`Unsupported bundle version ${JSON.stringify(manifest.version)}. Expected dynaworld_token_head_bundle/v2.`,
		);
	}
	const tensors = {};
	for (const [key, entry] of Object.entries(manifest.tensors)) {
		tensors[key] = await loadTensor(entry);
	}
	return {
		manifest,
		tensors,
		totalCount: manifest.counts.total_gaussians,
		decoded: null,
	};
}

export async function loadBundleFromFiles(files) {
	const fileIndex = buildFileIndex(files);
	const manifestFile =
		fileIndex.get("manifest.json") ||
		Array.from(fileIndex.values()).find((file) => file.name === "manifest.json");
	if (!manifestFile) {
		throw new Error("Could not find manifest.json in the selected directory.");
	}
	const manifest = await readJsonFile(manifestFile);
	return loadBundleWithTensorLoader(manifest, (entry) =>
		loadTensorFromFile(fileIndex, entry),
	);
}

export async function loadBundleFromBaseUrl(baseUrl) {
	const root = baseUrl.replace(/\/$/, "");
	const manifest = await readJsonUrl(`${root}/manifest.json`);
	return loadBundleWithTensorLoader(manifest, (entry) =>
		loadTensorFromUrl(root, entry),
	);
}

function tensor(bundle, key) {
	const value = bundle.tensors[key];
	if (!value) {
		throw new Error(`Bundle tensor missing: ${key}`);
	}
	return value;
}

function sigmoid(value) {
	return 1 / (1 + Math.exp(-value));
}

function erf(value) {
	const sign = value < 0 ? -1 : 1;
	const x = Math.abs(value);
	const t = 1 / (1 + 0.3275911 * x);
	const y =
		1 -
		(((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
			0.284496736) *
			t +
			0.254829592) *
			t) *
			Math.exp(-x * x);
	return sign * y;
}

function gelu(value) {
	return 0.5 * value * (1 + erf(value / Math.SQRT2));
}

function sortedLinearLayerIndices(bundle, prefix) {
	return Object.keys(bundle.tensors)
		.map((key) => {
			const match = key.match(new RegExp(`^${prefix.replace(/\./g, "\\.")}\\.(\\d+)\\.weight$`));
			return match ? Number(match[1]) : null;
		})
		.filter((value) => value !== null)
		.sort((left, right) => left - right);
}

function linear(input, rows, inDim, weightTensor, biasTensor) {
	const [outDim, weightInDim] = weightTensor.shape;
	if (weightInDim !== inDim) {
		throw new Error(`Linear input dimension mismatch: expected ${weightInDim}, got ${inDim}.`);
	}
	const output = new Float32Array(rows * outDim);
	const weight = weightTensor.data;
	const bias = biasTensor.data;
	for (let row = 0; row < rows; row += 1) {
		const inputBase = row * inDim;
		const outputBase = row * outDim;
		for (let outIndex = 0; outIndex < outDim; outIndex += 1) {
			let sum = bias[outIndex];
			const weightBase = outIndex * inDim;
			for (let inIndex = 0; inIndex < inDim; inIndex += 1) {
				sum += input[inputBase + inIndex] * weight[weightBase + inIndex];
			}
			output[outputBase + outIndex] = sum;
		}
	}
	return { data: output, shape: [rows, outDim] };
}

function runMlp(bundle, prefix, inputTensor) {
	const layerIndices = sortedLinearLayerIndices(bundle, prefix);
	if (layerIndices.length === 0) {
		throw new Error(`No linear layers found for ${prefix}.`);
	}
	let current = inputTensor;
	for (let index = 0; index < layerIndices.length; index += 1) {
		const layer = layerIndices[index];
		current = linear(
			current.data,
			current.shape[0],
			current.shape[1],
			tensor(bundle, `${prefix}.${layer}.weight`),
			tensor(bundle, `${prefix}.${layer}.bias`),
		);
		if (index !== layerIndices.length - 1) {
			for (let valueIndex = 0; valueIndex < current.data.length; valueIndex += 1) {
				current.data[valueIndex] = gelu(current.data[valueIndex]);
			}
		}
	}
	return current;
}

function normalizeQuatInFrame(frame, base) {
	const norm = Math.hypot(
		frame[base + 8],
		frame[base + 9],
		frame[base + 10],
		frame[base + 11],
	);
	const invNorm = norm > 1e-8 ? 1 / norm : 1;
	frame[base + 8] *= invNorm;
	frame[base + 9] *= invNorm;
	frame[base + 10] *= invNorm;
	frame[base + 11] *= invNorm;
}

function decodeGaussianHeads(bundle, tokenKey, prefix, meta) {
	const tokens = tensor(bundle, tokenKey);
	const tokenCount = tokens.shape[0];
	const gaussiansPerToken = meta.gaussians_per_token;
	const gaussianCount = tokenCount * gaussiansPerToken;
	const xyzRaw = runMlp(bundle, `${prefix}.xyz_head`, tokens).data;
	const scaleRaw = runMlp(bundle, `${prefix}.scale_head`, tokens).data;
	const rotRaw = runMlp(bundle, `${prefix}.rot_head`, tokens).data;
	const opacityRaw = runMlp(bundle, `${prefix}.opacity_head`, tokens).data;
	const rgbRaw = runMlp(bundle, `${prefix}.rgb_head`, tokens).data;
	const frame = new Float32Array(gaussianCount * 16);
	const logScaleInit = Math.log(meta.scale_init);

	for (let index = 0; index < gaussianCount; index += 1) {
		const out = index * 16;
		const xyz = index * 3;
		const rot = index * 4;
		frame[out] = Math.tanh(xyzRaw[xyz]) * meta.xy_extent;
		frame[out + 1] = Math.tanh(xyzRaw[xyz + 1]) * meta.xy_extent;
		frame[out + 2] = sigmoid(xyzRaw[xyz + 2]) * meta.z_extent + meta.z_min;
		frame[out + 3] = opacityRaw[index];
		frame[out + 4] = scaleRaw[xyz] + logScaleInit;
		frame[out + 5] = scaleRaw[xyz + 1] + logScaleInit;
		frame[out + 6] = scaleRaw[xyz + 2] + logScaleInit;
		frame[out + 7] = 0;
		frame[out + 8] = rotRaw[rot];
		frame[out + 9] = rotRaw[rot + 1];
		frame[out + 10] = rotRaw[rot + 2];
		frame[out + 11] = rotRaw[rot + 3];
		normalizeQuatInFrame(frame, out);
		frame[out + 12] = sigmoid(rgbRaw[xyz]);
		frame[out + 13] = sigmoid(rgbRaw[xyz + 1]);
		frame[out + 14] = sigmoid(rgbRaw[xyz + 2]);
		frame[out + 15] = 0;
	}
	return frame;
}

function reshapeDynamicCoefficients(raw, gaussianCount, basisCount, channels, extent) {
	const values = new Float32Array(gaussianCount * basisCount * channels);
	for (let index = 0; index < values.length; index += 1) {
		values[index] = Math.tanh(raw[index]) * extent;
	}
	return values;
}

function decodeDynamicBank(bundle) {
	const meta = bundle.manifest.decoder.dynamic_gaussian_heads;
	const tokens = tensor(bundle, "dynamic_query_tokens");
	const tokenCount = tokens.shape[0];
	const gaussiansPerToken = meta.base_heads.gaussians_per_token;
	const gaussianCount = tokenCount * gaussiansPerToken;
	const basisCount = meta.time_basis_count;
	const baseInterleaved = decodeGaussianHeads(
		bundle,
		"dynamic_query_tokens",
		"dynamic_gaussian_heads.base_heads",
		meta.base_heads,
	);
	const motionRaw = runMlp(bundle, "dynamic_gaussian_heads.motion_head", tokens).data;
	const rotationRaw = runMlp(bundle, "dynamic_gaussian_heads.rotation_head", tokens).data;
	const alphaRaw = runMlp(bundle, "dynamic_gaussian_heads.alpha_head", tokens).data;
	return {
		baseInterleaved,
		motion: reshapeDynamicCoefficients(
			motionRaw,
			gaussianCount,
			basisCount,
			3,
			meta.motion_extent,
		),
		rotation: reshapeDynamicCoefficients(
			rotationRaw,
			gaussianCount,
			basisCount,
			3,
			meta.rotation_radians,
		),
		alpha: reshapeDynamicCoefficients(
			alphaRaw,
			gaussianCount,
			basisCount,
			1,
			meta.alpha_logit_extent,
		),
	};
}

function ensureDecoded(bundle) {
	if (bundle.decoded) {
		return bundle.decoded;
	}
	const decoded = {
		staticInterleaved: decodeGaussianHeads(
			bundle,
			"static_query_tokens",
			"static_gaussian_heads",
			bundle.manifest.decoder.static_gaussian_heads,
		),
		dynamic: decodeDynamicBank(bundle),
	};
	bundle.decoded = decoded;
	return decoded;
}

function temporalBasis(timeValue, basisCount, maxFrequency) {
	if (basisCount < 1) {
		throw new Error(`basisCount must be >= 1, got ${basisCount}.`);
	}
	const values = new Float32Array(basisCount);
	const pairCount = Math.floor(basisCount / 2);
	let cursor = 0;
	if (pairCount > 0) {
		const log2Max = Math.log2(maxFrequency);
		for (let index = 0; index < pairCount; index += 1) {
			const alpha = pairCount === 1 ? 0 : index / (pairCount - 1);
			const frequency = 2 ** (alpha * log2Max);
			values[cursor] = Math.sin(2 * Math.PI * timeValue * frequency);
			cursor += 1;
		}
		for (let index = 0; index < pairCount; index += 1) {
			const alpha = pairCount === 1 ? 0 : index / (pairCount - 1);
			const frequency = 2 ** (alpha * log2Max);
			values[cursor] = Math.cos(2 * Math.PI * timeValue * frequency);
			cursor += 1;
		}
	}
	if (basisCount % 2 === 1) {
		values[cursor] = timeValue * 2 - 1;
	}
	return values;
}

function axisAngleToQuat(x, y, z) {
	const angle = Math.hypot(x, y, z);
	if (angle < 1e-6) {
		return [1, 0.5 * x, 0.5 * y, 0.5 * z];
	}
	const halfAngle = 0.5 * angle;
	const sinHalfOverAngle = Math.sin(halfAngle) / Math.max(angle, 1e-12);
	return [
		Math.cos(halfAngle),
		x * sinHalfOverAngle,
		y * sinHalfOverAngle,
		z * sinHalfOverAngle,
	];
}

function quatMultiply(lhs, rhs) {
	const [lw, lx, ly, lz] = lhs;
	const [rw, rx, ry, rz] = rhs;
	return [
		lw * rw - lx * rx - ly * ry - lz * rz,
		lw * rx + lx * rw + ly * rz - lz * ry,
		lw * ry - lx * rz + ly * rw + lz * rx,
		lw * rz + lx * ry - ly * rx + lz * rw,
	];
}

function normalizeQuat(quat) {
	const norm = Math.hypot(quat[0], quat[1], quat[2], quat[3]);
	const invNorm = norm > 1e-8 ? 1 / norm : 1;
	return [
		quat[0] * invNorm,
		quat[1] * invNorm,
		quat[2] * invNorm,
		quat[3] * invNorm,
	];
}

function clampTime(bundle, timeValue) {
	const [start, end] = bundle.manifest.viewer_defaults.time_domain;
	return Math.min(end, Math.max(start, timeValue));
}

export function buildFrameInterleaved(bundle, timeValue) {
	const decoded = ensureDecoded(bundle);
	const clampedTime = clampTime(bundle, timeValue);
	const basis = temporalBasis(
		clampedTime,
		bundle.manifest.model.dynamic_time_basis_count,
		bundle.manifest.model.dynamic_time_max_frequency,
	);
	const frame = new Float32Array(bundle.totalCount * 16);
	frame.set(decoded.staticInterleaved, 0);

	const staticCount = bundle.manifest.counts.static_gaussians;
	const dynamicCount = bundle.manifest.counts.dynamic_gaussians;
	const dynamicOffset = staticCount * 16;
	const basisCount = bundle.manifest.model.dynamic_time_basis_count;
	const dynamic = decoded.dynamic;

	for (let gaussianIndex = 0; gaussianIndex < dynamicCount; gaussianIndex += 1) {
		const inBase = gaussianIndex * 16;
		const outBase = dynamicOffset + inBase;
		frame[outBase + 4] = dynamic.baseInterleaved[inBase + 4];
		frame[outBase + 5] = dynamic.baseInterleaved[inBase + 5];
		frame[outBase + 6] = dynamic.baseInterleaved[inBase + 6];
		frame[outBase + 7] = 0;
		frame[outBase + 12] = dynamic.baseInterleaved[inBase + 12];
		frame[outBase + 13] = dynamic.baseInterleaved[inBase + 13];
		frame[outBase + 14] = dynamic.baseInterleaved[inBase + 14];
		frame[outBase + 15] = 0;

		let deltaX = 0;
		let deltaY = 0;
		let deltaZ = 0;
		let rotX = 0;
		let rotY = 0;
		let rotZ = 0;
		let alphaDelta = 0;
		const basisBase = gaussianIndex * basisCount;
		for (let basisIndex = 0; basisIndex < basisCount; basisIndex += 1) {
			const weight = basis[basisIndex];
			const coeffBase = (basisBase + basisIndex) * 3;
			deltaX += weight * dynamic.motion[coeffBase];
			deltaY += weight * dynamic.motion[coeffBase + 1];
			deltaZ += weight * dynamic.motion[coeffBase + 2];
			rotX += weight * dynamic.rotation[coeffBase];
			rotY += weight * dynamic.rotation[coeffBase + 1];
			rotZ += weight * dynamic.rotation[coeffBase + 2];
			alphaDelta += weight * dynamic.alpha[basisBase + basisIndex];
		}

		frame[outBase] = dynamic.baseInterleaved[inBase] + deltaX;
		frame[outBase + 1] = dynamic.baseInterleaved[inBase + 1] + deltaY;
		frame[outBase + 2] = dynamic.baseInterleaved[inBase + 2] + deltaZ;
		frame[outBase + 3] = dynamic.baseInterleaved[inBase + 3] + alphaDelta;

		const baseQuat = [
			dynamic.baseInterleaved[inBase + 8],
			dynamic.baseInterleaved[inBase + 9],
			dynamic.baseInterleaved[inBase + 10],
			dynamic.baseInterleaved[inBase + 11],
		];
		const residualQuat = axisAngleToQuat(rotX, rotY, rotZ);
		const quat = normalizeQuat(quatMultiply(baseQuat, residualQuat));
		frame[outBase + 8] = quat[0];
		frame[outBase + 9] = quat[1];
		frame[outBase + 10] = quat[2];
		frame[outBase + 11] = quat[3];
	}

	return frame;
}
