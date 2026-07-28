const MAX_TUBES = 32;
const PARAM_FLOATS = 20;
const PARAM_BYTES = PARAM_FLOATS * 4;
const CONFIG_BYTES = 48;

export const STAR_AFFINE_OPTIMIZED_COMPONENTS = Object.freeze([
	0, 1, 2,
	4, 5, 6, 7, 8, 9,
	16, 17, 18, 19,
]);

function assert(condition, message) {
	if (!condition) throw new Error(message);
}

function qTimesDelta(p, dx, dy, dt) {
	return [
		p[4] * dx + p[5] * dy + p[6] * dt,
		p[5] * dx + p[7] * dy + p[8] * dt,
		p[6] * dx + p[8] * dy + p[9] * dt,
	];
}

function tubeAlphaDepth(p, sample, options) {
	const dx = sample.x - p[0];
	const dy = sample.y - p[1];
	const dt = sample.t - p[2];
	const qd = qTimesDelta(p, dx, dy, dt);
	const qv = dx * qd[0] + dy * qd[1] + dt * qd[2];
	const gaussian = Math.exp(-0.5 * qv);
	const rawAlpha = p[19] * gaussian;
	return {
		delta: [dx, dy, dt], qd, gaussian, rawAlpha,
		alpha: Math.min(options.maxAlpha, rawAlpha),
		depth: p[12] + p[13] * dx + p[14] * dy + p[15] * dt,
	};
}

function validatePackedState(state) {
	assert(state instanceof Float32Array, "STAR state must be a Float32Array");
	assert(state.length % PARAM_FLOATS === 0, `STAR state length must be divisible by ${PARAM_FLOATS}`);
	assert(state.length > 0 && state.length / PARAM_FLOATS <= MAX_TUBES, `STAR supports 1-${MAX_TUBES} tubes`);
}

export function createAffineQ({
	precisionU,
	precisionV,
	temporalPrecision,
	velocityU = 0,
	velocityV = 0,
}) {
	assert(precisionU > 0 && precisionV > 0 && temporalPrecision > 0, "STAR precisions must be positive");
	return [
		precisionU,
		0,
		-precisionU * velocityU,
		precisionV,
		-precisionV * velocityV,
		temporalPrecision + precisionU * velocityU ** 2 + precisionV * velocityV ** 2,
	];
}

export function compileCameraSpaceWorldTubes(camera, tubes) {
	const { fx, fy, cx, cy } = camera;
	assert([fx, fy, cx, cy].every(Number.isFinite) && fx > 0 && fy > 0, "camera needs finite positive fx/fy and finite cx/cy");
	assert(Array.isArray(tubes) && tubes.length > 0 && tubes.length <= MAX_TUBES, `expected 1-${MAX_TUBES} camera-space tubes`);
	const packed = new Float32Array(tubes.length * PARAM_FLOATS);
	for (let index = 0; index < tubes.length; index += 1) {
		const tube = tubes[index];
		const [x, y, z] = tube.position;
		const [vx, vy, vz] = tube.velocity ?? [0, 0, 0];
		assert(z > 0 && [x, y, z, vx, vy, vz].every(Number.isFinite), "camera-space tube must stay in front of the pinhole at its reference time");
		const u = fx * x / z + cx;
		const v = fy * y / z + cy;
		const velocityU = fx * (vx * z - x * vz) / (z * z);
		const velocityV = fy * (vy * z - y * vz) / (z * z);
		const q = createAffineQ({
			precisionU: 1 / tube.sigmaPixels[0] ** 2,
			precisionV: 1 / tube.sigmaPixels[1] ** 2,
			temporalPrecision: 1 / tube.sigmaTime ** 2,
			velocityU,
			velocityV,
		});
		const offset = index * PARAM_FLOATS;
		packed.set([u, v, tube.centerTime ?? 0, 0], offset);
		packed.set([q[0], q[1], q[2], q[3]], offset + 4);
		packed.set([q[4], q[5], 0, 0], offset + 8);
		packed.set([z, 0, 0, vz], offset + 12);
		packed.set([tube.color[0], tube.color[1], tube.color[2], tube.opacity], offset + 16);
	}
	return packed;
}

export function renderAffineStarSample(state, sample, options = {}) {
	validatePackedState(state);
	const config = {
		alphaThreshold: options.alphaThreshold ?? 1 / 255,
		transmittanceThreshold: options.transmittanceThreshold ?? 1e-4,
		maxAlpha: options.maxAlpha ?? 0.99,
		background: options.background ?? [0, 0, 0],
	};
	const candidates = [];
	for (let tube = 0; tube < state.length / PARAM_FLOATS; tube += 1) {
		const p = state.subarray(tube * PARAM_FLOATS, (tube + 1) * PARAM_FLOATS);
		const evaluation = tubeAlphaDepth(p, sample, config);
		if (evaluation.alpha >= config.alphaThreshold) candidates.push({ tube, ...evaluation });
	}
	candidates.sort((a, b) => a.depth - b.depth || a.tube - b.tube);
	const rgb = [0, 0, 0];
	let transmittance = 1;
	for (const candidate of candidates) {
		const offset = candidate.tube * PARAM_FLOATS;
		const weight = transmittance * candidate.alpha;
		for (let channel = 0; channel < 3; channel += 1) rgb[channel] += weight * state[offset + 16 + channel];
		transmittance *= 1 - candidate.alpha;
		if (transmittance <= config.transmittanceThreshold) break;
	}
	for (let channel = 0; channel < 3; channel += 1) rgb[channel] += transmittance * config.background[channel];
	return rgb;
}

export function affineStarLossAndGradients(state, samples, options = {}) {
	validatePackedState(state);
	assert(Array.isArray(samples) && samples.length > 0, "STAR loss needs at least one sample");
	const config = {
		alphaThreshold: options.alphaThreshold ?? 1 / 255,
		transmittanceThreshold: options.transmittanceThreshold ?? 1e-4,
		maxAlpha: options.maxAlpha ?? 0.99,
		background: options.background ?? [0, 0, 0],
	};
	const gradients = new Float32Array(state.length);
	let loss = 0;
	for (const sample of samples) {
		const candidates = [];
		for (let tube = 0; tube < state.length / PARAM_FLOATS; tube += 1) {
			const offset = tube * PARAM_FLOATS;
			const evaluation = tubeAlphaDepth(state.subarray(offset, offset + PARAM_FLOATS), sample, config);
			if (evaluation.alpha >= config.alphaThreshold) candidates.push({ tube, offset, ...evaluation });
		}
		candidates.sort((a, b) => a.depth - b.depth || a.tube - b.tube);
		const used = [];
		const prediction = [0, 0, 0];
		let transmittance = 1;
		for (const candidate of candidates) {
			candidate.transmittance = transmittance;
			used.push(candidate);
			const weight = transmittance * candidate.alpha;
			for (let channel = 0; channel < 3; channel += 1) prediction[channel] += weight * state[candidate.offset + 16 + channel];
			transmittance *= 1 - candidate.alpha;
			if (transmittance <= config.transmittanceThreshold) break;
		}
		for (let channel = 0; channel < 3; channel += 1) prediction[channel] += transmittance * config.background[channel];
		const imageGradient = [0, 0, 0];
		for (let channel = 0; channel < 3; channel += 1) {
			const residual = prediction[channel] - sample.target[channel];
			loss += residual * residual / (samples.length * 3);
			imageGradient[channel] = 2 * residual / (samples.length * 3);
		}
		const suffix = [...config.background];
		for (let order = used.length - 1; order >= 0; order -= 1) {
			const candidate = used[order];
			const color = [state[candidate.offset + 16], state[candidate.offset + 17], state[candidate.offset + 18]];
			let gradAlpha = 0;
			for (let channel = 0; channel < 3; channel += 1) {
				gradAlpha += imageGradient[channel] * candidate.transmittance * (color[channel] - suffix[channel]);
				gradients[candidate.offset + 16 + channel] += imageGradient[channel] * candidate.transmittance * candidate.alpha;
				suffix[channel] = candidate.alpha * color[channel] + (1 - candidate.alpha) * suffix[channel];
			}
			if (candidate.rawAlpha < config.maxAlpha) {
				const gradQv = -0.5 * candidate.alpha * gradAlpha;
				const [dx, dy, dt] = candidate.delta;
				gradients[candidate.offset] += -2 * candidate.qd[0] * gradQv;
				gradients[candidate.offset + 1] += -2 * candidate.qd[1] * gradQv;
				gradients[candidate.offset + 2] += -2 * candidate.qd[2] * gradQv;
				gradients[candidate.offset + 4] += dx * dx * gradQv;
				gradients[candidate.offset + 5] += 2 * dx * dy * gradQv;
				gradients[candidate.offset + 6] += 2 * dx * dt * gradQv;
				gradients[candidate.offset + 7] += dy * dy * gradQv;
				gradients[candidate.offset + 8] += 2 * dy * dt * gradQv;
				gradients[candidate.offset + 9] += dt * dt * gradQv;
				gradients[candidate.offset + 19] += candidate.gaussian * gradAlpha;
			}
		}
	}
	return { loss, gradients };
}

export function finiteDifferenceAffineStar(state, samples, options = {}) {
	const epsilon = options.epsilon ?? 1e-3;
	const numerical = new Float32Array(state.length);
	for (let tube = 0; tube < state.length / PARAM_FLOATS; tube += 1) {
		for (const component of STAR_AFFINE_OPTIMIZED_COMPONENTS) {
			const index = tube * PARAM_FLOATS + component;
			const plus = state.slice();
			const minus = state.slice();
			plus[index] += epsilon;
			minus[index] -= epsilon;
			numerical[index] = (
				affineStarLossAndGradients(plus, samples, options).loss
				- affineStarLossAndGradients(minus, samples, options).loss
			) / (2 * epsilon);
		}
	}
	return numerical;
}

function packSamples(samples) {
	const packed = new Float32Array(samples.length * 8);
	for (let index = 0; index < samples.length; index += 1) {
		const sample = samples[index];
		packed.set([sample.x, sample.y, sample.t, 0, ...sample.target, 0], index * 8);
	}
	return packed;
}

function shaderSource() {
	return /* wgsl */ `
const MAX_TUBES: u32 = ${MAX_TUBES}u;
struct Param { ma: vec4<f32>, q0: vec4<f32>, q1: vec4<f32>, depth: vec4<f32>, appearance: vec4<f32> }
struct Sample { a: vec4<f32>, expected: vec4<f32> }
struct Config {
  tubeCount: u32, sampleCount: u32, step: u32, _pad: u32,
  alphaThreshold: f32, transmittanceThreshold: f32, maxAlpha: f32, learningRate: f32,
  background: vec4<f32>,
}
@group(0) @binding(0) var<storage, read_write> params: array<Param>;
@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read_write> sampleGrads: array<Param>;
@group(0) @binding(3) var<storage, read_write> reducedGrads: array<Param>;
@group(0) @binding(4) var<uniform> cfg: Config;

fn zero_param() -> Param {
  return Param(vec4<f32>(0), vec4<f32>(0), vec4<f32>(0), vec4<f32>(0), vec4<f32>(0));
}
fn q_delta(p: Param, d: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    p.q0.x*d.x + p.q0.y*d.y + p.q0.z*d.z,
    p.q0.y*d.x + p.q0.w*d.y + p.q1.x*d.z,
    p.q0.z*d.x + p.q1.x*d.y + p.q1.y*d.z);
}
fn depth_at(p: Param, a: vec3<f32>) -> f32 { return p.depth.x + dot(p.depth.yzw, a - p.ma.xyz); }

@compute @workgroup_size(64)
fn shared_adjoint(@builtin(global_invocation_id) gid: vec3<u32>) {
  let sampleId = gid.x;
  if (sampleId >= cfg.sampleCount) { return; }
  let sample = samples[sampleId];
  let a = sample.a.xyz;
  var ids: array<u32, ${MAX_TUBES}>;
  var alphas: array<f32, ${MAX_TUBES}>;
  var gaussian: array<f32, ${MAX_TUBES}>;
  var rawAlpha: array<f32, ${MAX_TUBES}>;
  var transBefore: array<f32, ${MAX_TUBES}>;
  var count = 0u;
  for (var tube = 0u; tube < cfg.tubeCount; tube++) {
    sampleGrads[sampleId * cfg.tubeCount + tube] = zero_param();
    let p = params[tube];
    let delta = a - p.ma.xyz;
    let qd = q_delta(p, delta);
    let g = exp(-0.5 * dot(delta, qd));
    let raw = p.appearance.w * g;
    let alpha = min(cfg.maxAlpha, raw);
    gaussian[tube] = g; rawAlpha[tube] = raw; alphas[tube] = alpha;
    if (alpha >= cfg.alphaThreshold) {
      var at = count;
      let d = depth_at(p, a);
      loop {
        if (at == 0u) { break; }
        let previous = ids[at - 1u];
        let previousDepth = depth_at(params[previous], a);
        if (previousDepth < d || (previousDepth == d && previous < tube)) { break; }
        ids[at] = previous; at--;
      }
      ids[at] = tube; count++;
    }
  }
  var prediction = vec3<f32>(0);
  var transmittance = 1.0;
  var used = 0u;
  for (var order = 0u; order < count; order++) {
    let tube = ids[order];
    transBefore[tube] = transmittance;
    prediction += transmittance * alphas[tube] * params[tube].appearance.xyz;
    transmittance *= 1.0 - alphas[tube];
    used++;
    if (transmittance <= cfg.transmittanceThreshold) { break; }
  }
  prediction += transmittance * cfg.background.xyz;
  let imageGrad = 2.0 * (prediction - sample.expected.xyz) / (f32(cfg.sampleCount) * 3.0);
  var suffix = cfg.background.xyz;
  for (var reverse = used; reverse > 0u; reverse--) {
    let tube = ids[reverse - 1u];
    let p = params[tube];
    let alpha = alphas[tube];
    let gradAlpha = dot(imageGrad, transBefore[tube] * (p.appearance.xyz - suffix));
    var grad = zero_param();
    let gradColor = imageGrad * transBefore[tube] * alpha;
    var gradOpacity = 0.0;
    if (rawAlpha[tube] < cfg.maxAlpha) {
      let delta = a - p.ma.xyz;
      let qd = q_delta(p, delta);
      let gradQv = -0.5 * alpha * gradAlpha;
      grad.ma = vec4<f32>(-2.0 * qd * gradQv, 0.0);
      grad.q0 = vec4<f32>(delta.x*delta.x, 2.0*delta.x*delta.y, 2.0*delta.x*delta.z, delta.y*delta.y) * gradQv;
      grad.q1 = vec4<f32>(vec2<f32>(2.0*delta.y*delta.z, delta.z*delta.z) * gradQv, 0.0, 0.0);
      gradOpacity = gaussian[tube] * gradAlpha;
    }
    grad.appearance = vec4<f32>(gradColor, gradOpacity);
    sampleGrads[sampleId * cfg.tubeCount + tube] = grad;
    suffix = alpha * p.appearance.xyz + (1.0 - alpha) * suffix;
  }
}

@compute @workgroup_size(64)
fn reduce_gradients(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tube = gid.x;
  if (tube >= cfg.tubeCount) { return; }
  var total = zero_param();
  for (var sample = 0u; sample < cfg.sampleCount; sample++) {
    let g = sampleGrads[sample * cfg.tubeCount + tube];
    total.ma += g.ma; total.q0 += g.q0; total.q1 += g.q1; total.appearance += g.appearance;
  }
  reducedGrads[tube] = total;
}

@compute @workgroup_size(64)
fn sgd_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tube = gid.x;
  if (tube >= cfg.tubeCount) { return; }
  var p = params[tube];
  let g = reducedGrads[tube];
  p.ma = vec4<f32>(p.ma.xyz - cfg.learningRate * g.ma.xyz, p.ma.w);
  p.q0 -= cfg.learningRate * g.q0;
  p.q1 = vec4<f32>(p.q1.xy - cfg.learningRate * g.q1.xy, p.q1.zw);
  p.appearance -= cfg.learningRate * g.appearance;
  p.q0 = vec4<f32>(max(p.q0.x, 1e-4), p.q0.yz, max(p.q0.w, 1e-4));
  p.q1 = vec4<f32>(p.q1.x, max(p.q1.y, 1e-4), p.q1.zw);
  p.appearance = vec4<f32>(clamp(p.appearance.xyz, vec3<f32>(0), vec3<f32>(1)), clamp(p.appearance.w, 1e-4, cfg.maxAlpha));
  params[tube] = p;
}
`;
}

export class AffineStarWebGpuTrainer {
	constructor(device) {
		this.device = device;
		this.step = 0;
	}

	static async create(state, samples, options = {}) {
		assert(globalThis.navigator?.gpu, "WebGPU is unavailable");
		const adapter = await navigator.gpu.requestAdapter();
		assert(adapter, "No WebGPU adapter found");
		const device = await adapter.requestDevice();
		const trainer = new AffineStarWebGpuTrainer(device);
		await trainer.init(state, samples, options);
		trainer.adapterName = adapter.info?.description ?? adapter.info?.device ?? "WebGPU adapter";
		return trainer;
	}

	async init(state, samples, options = {}) {
		validatePackedState(state);
		assert(samples.length > 0, "STAR trainer needs samples");
		this.tubeCount = state.length / PARAM_FLOATS;
		this.sampleCount = samples.length;
		this.options = {
			alphaThreshold: options.alphaThreshold ?? 1 / 255,
			transmittanceThreshold: options.transmittanceThreshold ?? 1e-4,
			maxAlpha: options.maxAlpha ?? 0.99,
			background: options.background ?? [0, 0, 0],
		};
		const usage = GPUBufferUsage;
		this.buffers = {
			params: this.device.createBuffer({ size: state.byteLength, usage: usage.STORAGE | usage.COPY_DST | usage.COPY_SRC }),
			samples: this.device.createBuffer({ size: samples.length * 32, usage: usage.STORAGE | usage.COPY_DST }),
			sampleGrads: this.device.createBuffer({ size: state.byteLength * samples.length, usage: usage.STORAGE }),
			reducedGrads: this.device.createBuffer({ size: state.byteLength, usage: usage.STORAGE | usage.COPY_SRC }),
			config: this.device.createBuffer({ size: CONFIG_BYTES, usage: usage.UNIFORM | usage.COPY_DST }),
		};
		this.device.queue.writeBuffer(this.buffers.params, 0, state);
		this.device.queue.writeBuffer(this.buffers.samples, 0, packSamples(samples));
		const module = this.device.createShaderModule({ code: shaderSource() });
		const compilation = await module.getCompilationInfo();
		const shaderErrors = compilation.messages.filter((message) => message.type === "error");
		if (shaderErrors.length) {
			throw new Error(`Affine STAR WGSL compilation failed:\n${shaderErrors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n")}`);
		}
		const bindGroupLayout = this.device.createBindGroupLayout({ entries: [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
			{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
			{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
			{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
		] });
		const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
		this.pipelines = {
			adjoint: await this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "shared_adjoint" } }),
			reduce: await this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "reduce_gradients" } }),
			optimizer: await this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "sgd_step" } }),
		};
		const bindGroup = this.device.createBindGroup({ layout: bindGroupLayout, entries: [
			{ binding: 0, resource: { buffer: this.buffers.params } },
			{ binding: 1, resource: { buffer: this.buffers.samples } },
			{ binding: 2, resource: { buffer: this.buffers.sampleGrads } },
			{ binding: 3, resource: { buffer: this.buffers.reducedGrads } },
			{ binding: 4, resource: { buffer: this.buffers.config } },
		] });
		this.bindGroups = { adjoint: bindGroup, reduce: bindGroup, optimizer: bindGroup };
	}

	writeConfig(learningRate) {
		const bytes = new ArrayBuffer(CONFIG_BYTES);
		const u32 = new Uint32Array(bytes);
		const f32 = new Float32Array(bytes);
		u32.set([this.tubeCount, this.sampleCount, this.step, 0], 0);
		f32.set([this.options.alphaThreshold, this.options.transmittanceThreshold, this.options.maxAlpha, learningRate], 4);
		f32.set([...this.options.background, 0], 8);
		this.device.queue.writeBuffer(this.buffers.config, 0, bytes);
	}

	encodeGradients(encoder) {
		let pass = encoder.beginComputePass();
		pass.setPipeline(this.pipelines.adjoint);
		pass.setBindGroup(0, this.bindGroups.adjoint);
		pass.dispatchWorkgroups(Math.ceil(this.sampleCount / 64));
		pass.end();
		pass = encoder.beginComputePass();
		pass.setPipeline(this.pipelines.reduce);
		pass.setBindGroup(0, this.bindGroups.reduce);
		pass.dispatchWorkgroups(Math.ceil(this.tubeCount / 64));
		pass.end();
	}

	trainStep({ learningRate = 0.01 } = {}) {
		this.writeConfig(learningRate);
		const encoder = this.device.createCommandEncoder();
		this.encodeGradients(encoder);
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipelines.optimizer);
		pass.setBindGroup(0, this.bindGroups.optimizer);
		pass.dispatchWorkgroups(Math.ceil(this.tubeCount / 64));
		pass.end();
		this.device.queue.submit([encoder.finish()]);
		this.step += 1;
	}

	async readBuffer(source, size) {
		const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(source, 0, staging, 0, size);
		this.device.queue.submit([encoder.finish()]);
		await staging.mapAsync(GPUMapMode.READ);
		const result = new Float32Array(staging.getMappedRange().slice(0));
		staging.unmap();
		staging.destroy();
		return result;
	}

	async readParams() {
		return this.readBuffer(this.buffers.params, this.tubeCount * PARAM_BYTES);
	}

	async readGradients() {
		this.device.pushErrorScope("validation");
		this.writeConfig(0);
		const encoder = this.device.createCommandEncoder();
		this.encodeGradients(encoder);
		this.device.queue.submit([encoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();
		const validationError = await this.device.popErrorScope();
		if (validationError) throw new Error(`Affine STAR WebGPU validation: ${validationError.message}`);
		return this.readBuffer(this.buffers.reducedGrads, this.tubeCount * PARAM_BYTES);
	}

	async gradientCheck(samples, { epsilon = 1e-3, absoluteTolerance = 2e-4, relativeTolerance = 2e-2 } = {}) {
		assert(samples.length === this.sampleCount, "gradient-check samples must match initialized samples");
		const state = await this.readParams();
		const gpu = await this.readGradients();
		const numerical = finiteDifferenceAffineStar(state, samples, { ...this.options, epsilon });
		let maxAbsoluteError = 0;
		let maxRelativeError = 0;
		let worstAbsolute = null;
		let worstRelative = null;
		for (let tube = 0; tube < this.tubeCount; tube += 1) {
			for (const component of STAR_AFFINE_OPTIMIZED_COMPONENTS) {
				const index = tube * PARAM_FLOATS + component;
				const absolute = Math.abs(gpu[index] - numerical[index]);
				const relative = absolute / Math.max(1e-5, Math.abs(gpu[index]), Math.abs(numerical[index]));
				if (absolute > maxAbsoluteError) {
					maxAbsoluteError = absolute;
					worstAbsolute = { tube, component, gpu: gpu[index], numerical: numerical[index], error: absolute };
				}
				if (relative > maxRelativeError) {
					maxRelativeError = relative;
					worstRelative = { tube, component, gpu: gpu[index], numerical: numerical[index], error: relative };
				}
			}
		}
		return {
			passed: maxAbsoluteError <= absoluteTolerance || maxRelativeError <= relativeTolerance,
			maxAbsoluteError,
			maxRelativeError,
			worstAbsolute,
			worstRelative,
			absoluteTolerance,
			relativeTolerance,
		};
	}

	dispose() {
		for (const buffer of Object.values(this.buffers ?? {})) buffer.destroy();
		this.device.destroy();
	}
}

export function createTinyAffineStarFixture() {
	const camera = { fx: 24, fy: 24, cx: 8, cy: 8 };
	const trueState = compileCameraSpaceWorldTubes(camera, [
		{ position: [-0.18, -0.05, 1.8], velocity: [0.025, 0.012, 0], sigmaPixels: [2.3, 2.0], sigmaTime: 2.8, centerTime: 0, color: [0.9, 0.18, 0.08], opacity: 0.72 },
		{ position: [0.16, 0.08, 2.2], velocity: [-0.018, -0.01, 0], sigmaPixels: [2.0, 2.4], sigmaTime: 3.2, centerTime: 0, color: [0.08, 0.3, 0.92], opacity: 0.64 },
	]);
	const samples = [];
	for (let frame = 0; frame < 4; frame += 1) {
		const t = frame - 1.5;
		for (let y = 3; y < 14; y += 2) {
			for (let x = 3; x < 14; x += 2) samples.push({ x: x + 0.5, y: y + 0.5, t, target: renderAffineStarSample(trueState, { x: x + 0.5, y: y + 0.5, t }) });
		}
	}
	const initialState = trueState.slice();
	initialState[0] += 0.35; initialState[1] -= 0.2; initialState[16] -= 0.12; initialState[19] -= 0.08;
	initialState[PARAM_FLOATS] -= 0.25; initialState[PARAM_FLOATS + 17] -= 0.1;
	return { camera, trueState, initialState, samples, width: 16, height: 16, frames: 4 };
}

export const AFFINE_STAR_BROWSER_CONTRACT = Object.freeze({
	name: "affine STAR UVT / World Tubes browser subset",
	state: "time-shared ma, symmetric q_uvt, conditional depth, opacity, RGB",
	coordinates: "pixel-center u,v and centered frame time, matching star_uvt_v0",
	visibility: "alpha-threshold support, stable detached conditional-depth order, source-over alpha",
	backward: "shared sampled-ray reverse source-over replay reduced over time and pixels",
	omissions: [
		"UVT tile binning and interval atlases",
		"projective/rational moving-camera traces and camera-family gauges",
		"order-event certificates and fallback splitting",
		"finite exposure and rolling shutter",
		"gradients through discrete support, ordering, and depth parameters",
		"Metal fixed-point/atomic accumulation and production optimizer parity",
	],
	matchedCostKnobs: ["tubeCount", "sampleCount", "frameCount", "optimizerSteps", "alphaThreshold"],
});
