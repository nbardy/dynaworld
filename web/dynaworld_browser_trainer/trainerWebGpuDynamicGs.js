const MAX_SPLATS = 32;
const STATE_FLOATS = 16;
const STATE_BYTES = STATE_FLOATS * 4;
const WORKGROUP_SIZE = 64;

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const logit = (value) => Math.log(value / (1 - value));

function rotateByQuaternion([x, y, z], [qx, qy, qz, qw]) {
	const tx = 2 * (qy * z - qz * y);
	const ty = 2 * (qz * x - qx * z);
	const tz = 2 * (qx * y - qy * x);
	return [
		x + qw * tx + qy * tz - qz * ty,
		y + qw * ty + qz * tx - qx * tz,
		z + qw * tz + qx * ty - qy * tx,
	];
}

function cameraPoint(camera, mean) {
	const matrix = camera.worldToCamera;
	return [
		matrix[0] * mean[0] + matrix[1] * mean[1] + matrix[2] * mean[2] + matrix[3],
		matrix[4] * mean[0] + matrix[5] * mean[1] + matrix[6] * mean[2] + matrix[7],
		matrix[8] * mean[0] + matrix[9] * mean[1] + matrix[10] * mean[2] + matrix[11],
	];
}

function projectedGaussian(state, offset, camera, u, v) {
	const mean = [state[offset], state[offset + 1], state[offset + 2]];
	const point = cameraPoint(camera, mean);
	if (point[2] <= 1e-3 || state[offset + 7] <= 0) return null;
	const [fx, fy, cx, cy] = camera.intrinsics;
	const centerX = fx * point[0] / point[2] + cx;
	const centerY = fy * point[1] / point[2] + cy;
	const quaternion = [state[offset + 8], state[offset + 9], state[offset + 10], state[offset + 11]];
	const axes = [
		rotateByQuaternion([1, 0, 0], quaternion),
		rotateByQuaternion([0, 1, 0], quaternion),
		rotateByQuaternion([0, 0, 1], quaternion),
	];
	const matrix = camera.worldToCamera;
	const variances = [
		Math.exp(2 * state[offset + 4]),
		Math.exp(2 * state[offset + 5]),
		Math.exp(2 * state[offset + 6]),
	];
	let cxx = 0; let cxy = 0; let cxz = 0; let cyy = 0; let cyz = 0; let czz = 0;
	for (let axis = 0; axis < 3; axis += 1) {
		const vector = axes[axis];
		const rx = matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
		const ry = matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2];
		const rz = matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2];
		const variance = variances[axis];
		cxx += variance * rx * rx; cxy += variance * rx * ry; cxz += variance * rx * rz;
		cyy += variance * ry * ry; cyz += variance * ry * rz; czz += variance * rz * rz;
	}
	const invZ = 1 / point[2];
	const j00 = fx * invZ; const j02 = -fx * point[0] * invZ * invZ;
	const j11 = fy * invZ; const j12 = -fy * point[1] * invZ * invZ;
	const sxx = j00 * j00 * cxx + 2 * j00 * j02 * cxz + j02 * j02 * czz + 1e-6;
	const sxy = j00 * j11 * cxy + j00 * j12 * cxz + j02 * j11 * cyz + j02 * j12 * czz;
	const syy = j11 * j11 * cyy + 2 * j11 * j12 * cyz + j12 * j12 * czz + 1e-6;
	const determinant = sxx * syy - sxy * sxy;
	if (determinant <= 1e-12) return null;
	const dx = u - centerX; const dy = v - centerY;
	const q = (syy * dx * dx - 2 * sxy * dx * dy + sxx * dy * dy) / determinant;
	if (q > 18) return null;
	const opacity = sigmoid(state[offset + 3]);
	const rawAlpha = opacity * Math.exp(-0.5 * Math.max(0, q));
	return {
		depth: point[2],
		alpha: Math.min(0.99, rawAlpha),
		rawAlpha,
		opacity,
		color: [sigmoid(state[offset + 12]), sigmoid(state[offset + 13]), sigmoid(state[offset + 14])],
	};
}

export function evaluateDynamicGsSample({ state, splatCount, frame, camera, u, v, target = null }) {
	const projected = [];
	for (let splat = 0; splat < splatCount; splat += 1) {
		const offset = (frame * splatCount + splat) * STATE_FLOATS;
		const item = projectedGaussian(state, offset, camera, u, v);
		if (item) projected.push({ ...item, splat, offset });
	}
	projected.sort((a, b) => a.depth - b.depth || a.splat - b.splat);
	const transmittanceBefore = [];
	let transmittance = 1;
	const color = [0, 0, 0];
	for (const item of projected) {
		transmittanceBefore.push(transmittance);
		for (let channel = 0; channel < 3; channel += 1) color[channel] += transmittance * item.alpha * item.color[channel];
		transmittance *= 1 - item.alpha;
	}
	if (!target) return { color, transmittance, order: projected.map((item) => item.splat) };
	const dColor = color.map((value, channel) => 2 * (value - target[channel]) / 3);
	const gradients = new Float64Array(splatCount * 4);
	let behind = [0, 0, 0];
	for (let index = projected.length - 1; index >= 0; index -= 1) {
		const item = projected[index]; const before = transmittanceBefore[index];
		for (let channel = 0; channel < 3; channel += 1) {
			const value = item.color[channel];
			gradients[item.splat * 4 + channel] = dColor[channel] * before * item.alpha * value * (1 - value);
		}
		let alphaGradient = 0;
		for (let channel = 0; channel < 3; channel += 1) alphaGradient += dColor[channel] * before * (item.color[channel] - behind[channel]);
		if (item.rawAlpha < 0.99) gradients[item.splat * 4 + 3] = alphaGradient * (item.rawAlpha / item.opacity) * item.opacity * (1 - item.opacity);
		behind = item.color.map((value, channel) => item.alpha * value + (1 - item.alpha) * behind[channel]);
	}
	const loss = color.reduce((sum, value, channel) => sum + (value - target[channel]) ** 2, 0) / 3;
	return { color, transmittance, order: projected.map((item) => item.splat), loss, gradients };
}

export function makeDynamicGsState(dataset, { splatCount = 16, seed = 17 } = {}) {
	if (splatCount < 1 || splatCount > MAX_SPLATS) throw new RangeError(`splatCount must be in [1, ${MAX_SPLATS}]`);
	const state = new Float32Array(dataset.frameCount * splatCount * STATE_FLOATS);
	const seeds = dataset.seedPoints ?? dataset.seed_points_xyzrgb ?? [];
	const flatSeeds = ArrayBuffer.isView(seeds) && typeof seeds[0] === "number";
	const seedCount = flatSeeds ? Math.floor(seeds.length / 6) : seeds.length;
	const anchor = normalizeCamera(dataset.cameras[0]);
	const positiveDepths = [];
	for (let index = 0; index < seedCount; index += 1) {
		const source = flatSeeds ? Array.from(seeds.subarray(index * 6, index * 6 + 6)) : seeds[index];
		const xyz = source.xyz ?? source.slice(0, 3); const depth = cameraPoint(anchor, xyz)[2];
		if (Number.isFinite(depth) && depth > 0) positiveDepths.push(depth);
	}
	positiveDepths.sort((a, b) => a - b);
	const medianDepth = positiveDepths.length ? positiveDepths[Math.floor(positiveDepths.length / 2)] : 5;
	const baseScale = Math.max(1e-3, medianDepth * 0.018);
	let randomState = seed >>> 0;
	const random = () => { randomState ^= randomState << 13; randomState ^= randomState >>> 17; randomState ^= randomState << 5; return (randomState >>> 0) / 0xffffffff; };
	for (let frame = 0; frame < dataset.frameCount; frame += 1) {
		for (let splat = 0; splat < splatCount; splat += 1) {
			const offset = (frame * splatCount + splat) * STATE_FLOATS;
			const seedIndex = seedCount ? Math.min(seedCount - 1, Math.floor((splat + 0.5) * seedCount / splatCount)) : -1;
			const source = seedIndex < 0 ? null : (flatSeeds ? Array.from(seeds.subarray(seedIndex * 6, seedIndex * 6 + 6)) : seeds[seedIndex]);
			const xyz = source ? (source.xyz ?? source.slice(0, 3)) : [(random() - 0.5) * 2, (random() - 0.5) * 2, 5 + random() * 2];
			const rgb = source ? (source.rgb ?? source.slice(3, 6)) : [0.35 + random() * 0.3, 0.35 + random() * 0.3, 0.35 + random() * 0.3];
			state.set([xyz[0], xyz[1], xyz[2], logit(0.18), Math.log(baseScale), Math.log(baseScale * 0.6), Math.log(baseScale * 0.35), 1,
				0, 0, 0, 1, logit(clamp(rgb[0], 0.01, 0.99)), logit(clamp(rgb[1], 0.01, 0.99)),
				logit(clamp(rgb[2], 0.01, 0.99)), 0], offset);
		}
	}
	return state;
}

function normalizeCamera(camera) {
	const matrix = camera.worldToCamera ?? camera.world_to_camera;
	return { intrinsics: Array.from(camera.intrinsics), worldToCamera: Array.isArray(matrix[0]) ? matrix.flat() : Array.from(matrix) };
}

function packCameras(cameras) {
	const packed = new Float32Array(cameras.length * 16);
	for (let index = 0; index < cameras.length; index += 1) {
		const camera = normalizeCamera(cameras[index]);
		packed.set(camera.worldToCamera.slice(0, 4), index * 16);
		packed.set(camera.worldToCamera.slice(4, 8), index * 16 + 4);
		packed.set(camera.worldToCamera.slice(8, 12), index * 16 + 8);
		packed.set(camera.intrinsics, index * 16 + 12);
	}
	return packed;
}

const TRAIN_WGSL = /* wgsl */`
const MAX_SPLATS: u32 = 32u;
struct Config { width:u32, height:u32, frameCount:u32, splatCount:u32, sampleCount:u32, step:u32, cameraCount:u32, pad:u32, lr:f32, beta1:f32, beta2:f32, epsilon:f32 };
struct Splat { meanOpacity:vec4<f32>, logScaleActive:vec4<f32>, rotation:vec4<f32>, colorPad:vec4<f32> };
struct Camera { row0:vec4<f32>, row1:vec4<f32>, row2:vec4<f32>, intrinsics:vec4<f32> };
struct Sample { frame:u32, view:u32, pixel:u32, pad:u32 };
struct Grad { value:vec4<f32> };
struct Projection { depth:f32, alpha:f32, rawAlpha:f32, opacity:f32, color:vec3<f32>, valid:f32 };
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read> stateIn:array<Splat>;
@group(0) @binding(2) var<storage,read> cameras:array<Camera>;
@group(0) @binding(3) var<storage,read> targets:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> samples:array<Sample>;
@group(0) @binding(5) var<storage,read_write> gradients:array<Grad>;
@group(0) @binding(6) var<storage,read_write> losses:array<f32>;
fn sigmoid(x:f32)->f32{return 1.0/(1.0+exp(-x));}
fn qrotate(v:vec3<f32>,q:vec4<f32>)->vec3<f32>{let t=2.0*cross(q.xyz,v);return v+q.w*t+cross(q.xyz,t);}
fn project(p:Splat, camera:Camera, uv:vec2<f32>)->Projection {
	let cp=vec3<f32>(dot(camera.row0.xyz,p.meanOpacity.xyz)+camera.row0.w,dot(camera.row1.xyz,p.meanOpacity.xyz)+camera.row1.w,dot(camera.row2.xyz,p.meanOpacity.xyz)+camera.row2.w);
	if(cp.z<=0.001 || p.logScaleActive.w<=0.0){return Projection(0.0,0.0,0.0,0.0,vec3<f32>(0.0),0.0);}
	let center=vec2<f32>(camera.intrinsics.x*cp.x/cp.z+camera.intrinsics.z,camera.intrinsics.y*cp.y/cp.z+camera.intrinsics.w);
	let a0=qrotate(vec3<f32>(1,0,0),p.rotation);let a1=qrotate(vec3<f32>(0,1,0),p.rotation);let a2=qrotate(vec3<f32>(0,0,1),p.rotation);
	let b0=vec3<f32>(dot(camera.row0.xyz,a0),dot(camera.row1.xyz,a0),dot(camera.row2.xyz,a0));
	let b1=vec3<f32>(dot(camera.row0.xyz,a1),dot(camera.row1.xyz,a1),dot(camera.row2.xyz,a1));
	let b2=vec3<f32>(dot(camera.row0.xyz,a2),dot(camera.row1.xyz,a2),dot(camera.row2.xyz,a2));
	let variance=exp(2.0*p.logScaleActive.xyz);
	let cxx=variance.x*b0.x*b0.x+variance.y*b1.x*b1.x+variance.z*b2.x*b2.x;
	let cxy=variance.x*b0.x*b0.y+variance.y*b1.x*b1.y+variance.z*b2.x*b2.y;
	let cxz=variance.x*b0.x*b0.z+variance.y*b1.x*b1.z+variance.z*b2.x*b2.z;
	let cyy=variance.x*b0.y*b0.y+variance.y*b1.y*b1.y+variance.z*b2.y*b2.y;
	let cyz=variance.x*b0.y*b0.z+variance.y*b1.y*b1.z+variance.z*b2.y*b2.z;
	let czz=variance.x*b0.z*b0.z+variance.y*b1.z*b1.z+variance.z*b2.z*b2.z;
	let iz=1.0/cp.z;let j00=camera.intrinsics.x*iz;let j02=-camera.intrinsics.x*cp.x*iz*iz;let j11=camera.intrinsics.y*iz;let j12=-camera.intrinsics.y*cp.y*iz*iz;
	let sxx=j00*j00*cxx+2.0*j00*j02*cxz+j02*j02*czz+1e-6;
	let sxy=j00*j11*cxy+j00*j12*cxz+j02*j11*cyz+j02*j12*czz;
	let syy=j11*j11*cyy+2.0*j11*j12*cyz+j12*j12*czz+1e-6;let det=sxx*syy-sxy*sxy;
	if(det<=1e-12){return Projection(0.0,0.0,0.0,0.0,vec3<f32>(0.0),0.0);}let d=uv-center;let q=(syy*d.x*d.x-2.0*sxy*d.x*d.y+sxx*d.y*d.y)/det;
	if(q>18.0){return Projection(cp.z,0.0,0.0,0.0,vec3<f32>(0.0),0.0);}let opacity=sigmoid(p.meanOpacity.w);let raw=opacity*exp(-0.5*max(q,0.0));
	return Projection(cp.z,min(0.99,raw),raw,opacity,vec3<f32>(1.0)/(vec3<f32>(1.0)+exp(-p.colorPad.xyz)),1.0);
}
@compute @workgroup_size(64) fn sampleBackward(@builtin(global_invocation_id) gid:vec3<u32>){
	let sampleIndex=gid.x;if(sampleIndex>=cfg.sampleCount){return;}let sample=samples[sampleIndex];let x=sample.pixel%cfg.width;let y=sample.pixel/cfg.width;
	let uv=(vec2<f32>(f32(x),f32(y))+0.5)/vec2<f32>(f32(cfg.width),f32(cfg.height));let camera=cameras[sample.view];
	var ids:array<u32,32>;var depths:array<f32,32>;var projections:array<Projection,32>;var count=0u;
	for(var i=0u;i<MAX_SPLATS;i++){if(i>=cfg.splatCount){break;}let projection=project(stateIn[sample.frame*cfg.splatCount+i],camera,uv);if(projection.valid>0.5){var insert=count;loop{if(insert==0u||depths[insert-1u]<=projection.depth){break;}depths[insert]=depths[insert-1u];ids[insert]=ids[insert-1u];projections[insert]=projections[insert-1u];insert--;}depths[insert]=projection.depth;ids[insert]=i;projections[insert]=projection;count++;}}
	var before:array<f32,32>;var prediction=vec3<f32>(0.0);var transmittance=1.0;for(var i=0u;i<count;i++){before[i]=transmittance;prediction+=transmittance*projections[i].alpha*projections[i].color;transmittance*=1.0-projections[i].alpha;}
	let targetIndex=(sample.view*cfg.frameCount+sample.frame)*cfg.width*cfg.height+sample.pixel;let wanted=targets[targetIndex].xyz;let error=prediction-wanted;losses[sampleIndex]=dot(error,error)/3.0;let dColor=2.0*error/3.0;
	for(var i=0u;i<cfg.splatCount;i++){gradients[sampleIndex*cfg.splatCount+i].value=vec4<f32>(0.0);}var behind=vec3<f32>(0.0);var reverse=count;loop{if(reverse==0u){break;}reverse--;let p=projections[reverse];let id=ids[reverse];let colorGrad=dColor*before[reverse]*p.alpha*p.color*(1.0-p.color);let alphaGrad=dot(dColor,before[reverse]*(p.color-behind));var opacityGrad=0.0;if(p.rawAlpha<0.99){opacityGrad=alphaGrad*(p.rawAlpha/max(p.opacity,1e-8))*p.opacity*(1.0-p.opacity);}gradients[sampleIndex*cfg.splatCount+id].value=vec4<f32>(colorGrad,opacityGrad);behind=p.alpha*p.color+(1.0-p.alpha)*behind;}
}
`;

const UPDATE_WGSL = /* wgsl */`
struct Config { width:u32,height:u32,frameCount:u32,splatCount:u32,sampleCount:u32,step:u32,cameraCount:u32,pad:u32,lr:f32,beta1:f32,beta2:f32,epsilon:f32 };
struct Splat { meanOpacity:vec4<f32>,logScaleActive:vec4<f32>,rotation:vec4<f32>,colorPad:vec4<f32> };struct Sample { frame:u32,view:u32,pixel:u32,pad:u32 };struct Grad{value:vec4<f32>};
@group(0) @binding(0)var<uniform>cfg:Config;@group(0) @binding(1)var<storage,read>stateIn:array<Splat>;@group(0) @binding(2)var<storage,read_write>stateOut:array<Splat>;@group(0) @binding(3)var<storage,read_write>m1:array<vec4<f32>>;@group(0) @binding(4)var<storage,read_write>m2:array<vec4<f32>>;@group(0) @binding(5)var<storage,read>samples:array<Sample>;@group(0) @binding(6)var<storage,read>gradients:array<Grad>;
@compute @workgroup_size(64) fn update(@builtin(global_invocation_id)gid:vec3<u32>){let index=gid.x;let total=cfg.frameCount*cfg.splatCount;if(index>=total){return;}let frame=index/cfg.splatCount;var gradient=vec4<f32>(0.0);var matches=0.0;for(var sample=0u;sample<cfg.sampleCount;sample++){if(samples[sample].frame==frame){gradient+=gradients[sample*cfg.splatCount+(index%cfg.splatCount)].value;matches+=1.0;}}gradient/=max(matches,1.0);let first=cfg.beta1*m1[index]+(1.0-cfg.beta1)*gradient;let second=cfg.beta2*m2[index]+(1.0-cfg.beta2)*gradient*gradient;m1[index]=first;m2[index]=second;let t=f32(cfg.step+1u);let adjusted=cfg.lr*(first/(1.0-pow(cfg.beta1,t)))/(sqrt(second/(1.0-pow(cfg.beta2,t)))+cfg.epsilon);var output=stateIn[index];output.colorPad=vec4<f32>(clamp(output.colorPad.xyz-adjusted.xyz,vec3<f32>(-8.0),vec3<f32>(8.0)),output.colorPad.w);output.meanOpacity=vec4<f32>(output.meanOpacity.xyz,clamp(output.meanOpacity.w-adjusted.w,-8.0,8.0));stateOut[index]=output;}
`;

export class DynamicGsWebGpuTrainer {
	static async create(dataset, options = {}) {
		if (!navigator.gpu) throw new Error("WebGPU is unavailable.");
		const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
		if (!adapter) throw new Error("No WebGPU adapter found.");
		const device = await adapter.requestDevice(); device.pushErrorScope("validation");
		const trainer = new DynamicGsWebGpuTrainer(device, dataset, options); const error = await device.popErrorScope();
		if (error) { trainer.dispose(); throw error; }
		return trainer;
	}

	constructor(device, dataset, { splatCount = 16, state = null } = {}) {
		if (!dataset?.cameras?.length || !dataset?.frames) throw new TypeError("dataset must provide calibrated cameras and view-major RGBA frames");
		this.device = device; this.dataset = dataset; this.splatCount = splatCount; this.stepCount = 0; this.current = 0; this.lastSampleCount = 0;
		this.state = state ?? makeDynamicGsState(dataset, { splatCount }); this.configBytes = new ArrayBuffer(48);
		this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: TRAIN_WGSL }), entryPoint: "sampleBackward" } });
		this.updatePipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: UPDATE_WGSL }), entryPoint: "update" } });
		this.createBuffers(); this.createBindGroups();
	}

	createBuffers() {
		const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC; const make = (size, usage = storage) => this.device.createBuffer({ size, usage });
		const states = [make(this.state.byteLength), make(this.state.byteLength)]; states.forEach((buffer) => this.device.queue.writeBuffer(buffer, 0, this.state));
		const cameras = packCameras(this.dataset.cameras); const cameraBuffer = make(cameras.byteLength); this.device.queue.writeBuffer(cameraBuffer, 0, cameras);
		const target = make(this.dataset.frames.byteLength); this.device.queue.writeBuffer(target, 0, this.dataset.frames);
		const maxSamples = 256; const samples = make(maxSamples * 16); const gradients = make(maxSamples * this.splatCount * 16); const losses = make(maxSamples * 4);
		const moments = [make(this.dataset.frameCount * this.splatCount * 16), make(this.dataset.frameCount * this.splatCount * 16)];
		const readback = make(maxSamples * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
		this.buffers = { states, cameraBuffer, target, samples, gradients, losses, moments, readback, config: make(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST) }; this.maxSamples = maxSamples;
	}

	createBindGroups() {
		const trainEntries = (state) => [{ binding: 0, resource: { buffer: this.buffers.config } }, { binding: 1, resource: { buffer: state } }, { binding: 2, resource: { buffer: this.buffers.cameraBuffer } }, { binding: 3, resource: { buffer: this.buffers.target } }, { binding: 4, resource: { buffer: this.buffers.samples } }, { binding: 5, resource: { buffer: this.buffers.gradients } }, { binding: 6, resource: { buffer: this.buffers.losses } }];
		const updateEntries = (input, output) => [{ binding: 0, resource: { buffer: this.buffers.config } }, { binding: 1, resource: { buffer: input } }, { binding: 2, resource: { buffer: output } }, { binding: 3, resource: { buffer: this.buffers.moments[0] } }, { binding: 4, resource: { buffer: this.buffers.moments[1] } }, { binding: 5, resource: { buffer: this.buffers.samples } }, { binding: 6, resource: { buffer: this.buffers.gradients } }];
		this.trainGroups = this.buffers.states.map((state) => this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: trainEntries(state) }));
		this.updateGroups = [this.device.createBindGroup({ layout: this.updatePipeline.getBindGroupLayout(0), entries: updateEntries(this.buffers.states[0], this.buffers.states[1]) }), this.device.createBindGroup({ layout: this.updatePipeline.getBindGroupLayout(0), entries: updateEntries(this.buffers.states[1], this.buffers.states[0]) })];
	}

	trainStep({ samplesPerStep = 64, learningRate = 0.01 } = {}) {
		const count = clamp(Math.floor(samplesPerStep), 1, this.maxSamples); const samples = new Uint32Array(count * 4); const trainViews = this.dataset.cameras.map((camera, index) => camera.role === "heldout" ? -1 : index).filter((index) => index >= 0);
		const pixels = this.dataset.width * this.dataset.height; const motionSamples = this.dataset.motionSamples;
		for (let index = 0; index < count; index += 1) {
			const value = (this.stepCount * count + index) >>> 0;
			if (motionSamples?.length) {
				const packed = motionSamples[(value * 2654435761 >>> 0) % motionSamples.length];
				samples[index * 4] = Math.floor(packed / pixels) % this.dataset.frameCount;
				samples[index * 4 + 1] = Math.floor(packed / (pixels * this.dataset.frameCount));
				samples[index * 4 + 2] = packed % pixels;
			} else {
				samples[index * 4] = value % this.dataset.frameCount;
				samples[index * 4 + 1] = trainViews[Math.floor(value / this.dataset.frameCount) % trainViews.length];
				samples[index * 4 + 2] = (value * 2654435761 >>> 0) % pixels;
			}
		}
		this.device.queue.writeBuffer(this.buffers.samples, 0, samples); const u32 = new Uint32Array(this.configBytes); const f32 = new Float32Array(this.configBytes); u32.set([this.dataset.width, this.dataset.height, this.dataset.frameCount, this.splatCount, count, this.stepCount, this.dataset.cameras.length, 0]); f32.set([learningRate, 0.9, 0.999, 1e-8], 8); this.device.queue.writeBuffer(this.buffers.config, 0, this.configBytes);
		const encoder = this.device.createCommandEncoder(); let pass = encoder.beginComputePass(); pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.trainGroups[this.current]); pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP_SIZE)); pass.end(); pass = encoder.beginComputePass(); pass.setPipeline(this.updatePipeline); pass.setBindGroup(0, this.updateGroups[this.current]); pass.dispatchWorkgroups(Math.ceil(this.dataset.frameCount * this.splatCount / WORKGROUP_SIZE)); pass.end(); this.device.queue.submit([encoder.finish()]); this.current = 1 - this.current; this.stepCount += 1; this.lastSampleCount = count;
	}

	async readLoss() {
		const byteLength = this.lastSampleCount * 4; const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.buffers.losses, 0, this.buffers.readback, 0, byteLength); this.device.queue.submit([encoder.finish()]); await this.buffers.readback.mapAsync(GPUMapMode.READ, 0, byteLength); const values = new Float32Array(this.buffers.readback.getMappedRange(0, byteLength).slice(0)); this.buffers.readback.unmap(); return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
	}

	dispose() { for (const value of Object.values(this.buffers)) { if (Array.isArray(value)) value.forEach((buffer) => buffer.destroy()); else value.destroy(); } }
}

export const DYNAMIC_GS_LIMITS = Object.freeze({ maxSplats: MAX_SPLATS, stateFloats: STATE_FLOATS, optimized: ["per-frame RGB logits", "per-frame opacity logit"], fixed: ["means", "anisotropic scales", "rotations"], depthOrder: "camera-space ascending per sample", compositing: "front-to-back alpha over black" });
