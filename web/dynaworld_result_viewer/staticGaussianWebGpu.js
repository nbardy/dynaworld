function invert4x4(matrix, out = new Float32Array(16)) {
	const a = matrix;
	const b00 = a[0] * a[5] - a[1] * a[4];
	const b01 = a[0] * a[6] - a[2] * a[4];
	const b02 = a[0] * a[7] - a[3] * a[4];
	const b03 = a[1] * a[6] - a[2] * a[5];
	const b04 = a[1] * a[7] - a[3] * a[5];
	const b05 = a[2] * a[7] - a[3] * a[6];
	const b06 = a[8] * a[13] - a[9] * a[12];
	const b07 = a[8] * a[14] - a[10] * a[12];
	const b08 = a[8] * a[15] - a[11] * a[12];
	const b09 = a[9] * a[14] - a[10] * a[13];
	const b10 = a[9] * a[15] - a[11] * a[13];
	const b11 = a[10] * a[15] - a[11] * a[14];
	const det =
		b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
	if (!det) {
		out.set(matrix);
		return out;
	}
	const invDet = 1 / det;
	out[0] = (a[5] * b11 - a[6] * b10 + a[7] * b09) * invDet;
	out[1] = (-a[1] * b11 + a[2] * b10 - a[3] * b09) * invDet;
	out[2] = (a[13] * b05 - a[14] * b04 + a[15] * b03) * invDet;
	out[3] = (-a[9] * b05 + a[10] * b04 - a[11] * b03) * invDet;
	out[4] = (-a[4] * b11 + a[6] * b08 - a[7] * b07) * invDet;
	out[5] = (a[0] * b11 - a[2] * b08 + a[3] * b07) * invDet;
	out[6] = (-a[12] * b05 + a[14] * b02 - a[15] * b01) * invDet;
	out[7] = (a[8] * b05 - a[10] * b02 + a[11] * b01) * invDet;
	out[8] = (a[4] * b10 - a[5] * b08 + a[7] * b06) * invDet;
	out[9] = (-a[0] * b10 + a[1] * b08 - a[3] * b06) * invDet;
	out[10] = (a[12] * b04 - a[13] * b02 + a[15] * b00) * invDet;
	out[11] = (-a[8] * b04 + a[9] * b02 - a[11] * b00) * invDet;
	out[12] = (-a[4] * b09 + a[5] * b07 - a[6] * b06) * invDet;
	out[13] = (a[0] * b09 - a[1] * b07 + a[2] * b06) * invDet;
	out[14] = (-a[12] * b03 + a[13] * b01 - a[14] * b00) * invDet;
	out[15] = (a[8] * b03 - a[9] * b01 + a[10] * b00) * invDet;
	return out;
}

function toColumnMajor(matrix, rowMajor = false, out = new Float32Array(16)) {
	if (!rowMajor) {
		out.set(matrix);
		return out;
	}
	out[0] = matrix[0];
	out[1] = matrix[4];
	out[2] = matrix[8];
	out[3] = matrix[12];
	out[4] = matrix[1];
	out[5] = matrix[5];
	out[6] = matrix[9];
	out[7] = matrix[13];
	out[8] = matrix[2];
	out[9] = matrix[6];
	out[10] = matrix[10];
	out[11] = matrix[14];
	out[12] = matrix[3];
	out[13] = matrix[7];
	out[14] = matrix[11];
	out[15] = matrix[15];
	return out;
}

export async function assertWebGpuAvailable() {
	if (!navigator.gpu) {
		throw new Error("WebGPU unavailable in this browser.");
	}
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		throw new Error("WebGPU adapter unavailable.");
	}
}

export function createStaticGaussianWebGpuRenderer(
	renderCanvas,
	logMessage = console.log,
) {
	let state = null;
	const columnMajor = new Float32Array(16);
	const inverted = new Float32Array(16);
	const uniforms = new Float32Array(27);

	const commonWGSL = `
		struct Params {
			w2c: mat4x4<f32>,
			fx: f32, fy: f32, cx: f32, cy: f32,
			width: f32, height: f32, scaleMul: f32, colorGain: f32,
			near: f32, far: f32, pad0: f32
		};
		@group(0) @binding(0) var<uniform> u: Params;

		struct SourceSplat {
			posOpac: vec4<f32>,
			scale: vec4<f32>,
			rotation: vec4<f32>,
			color: vec4<f32>
		};

		struct PrecomputedSplat {
			posRadius: vec4<f32>,
			invCov: vec4<f32>,
			color: vec4<f32>
		};

		fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
			let len = length(q);
			let nq = q / max(len, 1e-8);
			let w = nq.x; let x = nq.y; let y = nq.z; let z = nq.w;
			return mat3x3<f32>(
				vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
				vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
				vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))
			);
		}
	`;

	const destroyFrameResources = () => {
		if (!state?.bufs.frame) {
			return;
		}
		for (const buffer of state.bufs.frame.all) {
			buffer.destroy();
		}
		state.bufs.frame = null;
		state.bgs = null;
		state.data = null;
	};

	const ensureState = async () => {
		if (state) {
			return state;
		}
		if (!navigator.gpu) {
			throw new Error("WebGPU unavailable in this browser.");
		}
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) {
			throw new Error("WebGPU adapter unavailable.");
		}
		const device = await adapter.requestDevice({
			requiredLimits: {
				maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
				maxBufferSize: adapter.limits.maxBufferSize,
			},
		});
		const context = renderCanvas.getContext("webgpu");
		if (!context) {
			throw new Error("WebGPU canvas context unavailable.");
		}
		const format = "rgba16float";
		context.configure({ device, format, alphaMode: "premultiplied" });

		const computeModule = device.createShaderModule({
			code:
				commonWGSL +
				`
				@group(0) @binding(1) var<storage, read> src: array<SourceSplat>;
				@group(0) @binding(2) var<storage, read_write> splats: array<PrecomputedSplat>;
				@group(0) @binding(3) var<storage, read_write> bucketCounts: array<atomic<u32>>;
				@group(0) @binding(4) var<storage, read_write> bucketOf: array<u32>;
				@group(0) @binding(5) var<storage, read_write> localIndex: array<u32>;
				@group(0) @binding(6) var<storage, read_write> culledIndices: array<u32>;
				@group(0) @binding(7) var<storage, read_write> culledCount: atomic<u32>;

				const NUM_BUCKETS: u32 = 65536u;

				@compute @workgroup_size(256)
				fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
					let i = gid.x + gid.y * 65535u * 256u;
					if (i >= arrayLength(&src)) { return; }

					let s_in = src[i];
					let viewPos = u.w2c * vec4<f32>(s_in.posOpac.xyz, 1.0);
					let cz = viewPos.z;
					if (cz <= u.near || cz > u.far) { return; }

					let W = mat3x3<f32>(u.w2c[0].xyz, u.w2c[1].xyz, u.w2c[2].xyz);
					let R = quatToMat3(s_in.rotation);
					let sx = exp(s_in.scale.x);
					let sy = exp(s_in.scale.y);
					let sz = exp(s_in.scale.z);
					let RS = R * mat3x3<f32>(
						vec3<f32>(sx, 0.0, 0.0),
						vec3<f32>(0.0, sy, 0.0),
						vec3<f32>(0.0, 0.0, sz)
					);

					let invZ = 1.0 / cz;
					let J = mat3x2<f32>(
						vec2<f32>(u.fx * invZ, 0.0),
						vec2<f32>(0.0, u.fy * invZ),
						vec2<f32>(-u.fx * viewPos.x * invZ * invZ, -u.fy * viewPos.y * invZ * invZ)
					);
					let WRS = W * RS;
					let T = J * WRS;
					let cov2d = T * transpose(T) + mat2x2<f32>(0.3, 0.0, 0.0, 0.3);
					let det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
					if (det <= 1e-9) { return; }
					let invDet = 1.0 / det;
					let trace = cov2d[0][0] + cov2d[1][1];
					let disc = max(0.0, trace * trace * 0.25 - det);
					let radius = 3.0 * sqrt(trace * 0.5 + sqrt(disc));

					let splatIdx = atomicAdd(&culledCount, 1u);
					culledIndices[splatIdx] = i;

					var s_out: PrecomputedSplat;
					let opacity = 1.0 / (1.0 + exp(-s_in.posOpac.w));
					s_out.posRadius = vec4<f32>(u.fx * viewPos.x * invZ + u.cx, u.fy * viewPos.y * invZ + u.cy, radius, opacity);
					s_out.invCov = vec4<f32>(cov2d[1][1] * invDet, -cov2d[0][1] * invDet, cov2d[0][0] * invDet, 0.0);
					s_out.color = vec4<f32>(s_in.color.xyz, 0.0);
					splats[i] = s_out;

					let depthNorm = clamp((cz - u.near) / (u.far - u.near), 0.0, 1.0);
					let bucket = min(u32((1.0 - depthNorm) * f32(NUM_BUCKETS - 1u)), NUM_BUCKETS - 1u);
					bucketOf[i] = bucket;
					localIndex[i] = atomicAdd(&bucketCounts[bucket], 1u);
				}
			`,
		});

		const clearModule = device.createShaderModule({
			code: `
				@group(0) @binding(0) var<storage, read_write> bucketCounts: array<atomic<u32>>;
				@group(0) @binding(1) var<storage, read_write> culledCount: atomic<u32>;
				@compute @workgroup_size(256)
				fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
					if (gid.x < 65536u) { atomicStore(&bucketCounts[gid.x], 0u); }
					if (gid.x == 0u) { atomicStore(&culledCount, 0u); }
				}
			`,
		});

		const prefixModule = device.createShaderModule({
			code: `
				@group(0) @binding(0) var<storage, read_write> bucketCounts: array<u32>;
				@group(0) @binding(1) var<storage, read_write> bucketOffsets: array<u32>;
				@compute @workgroup_size(1)
				fn main() {
					var sum: u32 = 0u;
					bucketOffsets[0] = 0u;
					for (var i: u32 = 0u; i < 65536u; i = i + 1u) {
						sum = sum + bucketCounts[i];
						bucketOffsets[i + 1u] = sum;
					}
				}
			`,
		});

		const scatterModule = device.createShaderModule({
			code: `
				@group(0) @binding(0) var<storage, read> bucketOf: array<u32>;
				@group(0) @binding(1) var<storage, read> localIndex: array<u32>;
				@group(0) @binding(2) var<storage, read> bucketOffsets: array<u32>;
				@group(0) @binding(3) var<storage, read_write> sortedIndices: array<u32>;
				@group(0) @binding(4) var<storage, read> culledIndices: array<u32>;
				@group(0) @binding(5) var<storage, read> culledCount: u32;

				@compute @workgroup_size(256)
				fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
					let globalIndex = gid.x + gid.y * 65535u * 256u;
					if (globalIndex >= culledCount) { return; }
					let sourceIndex = culledIndices[globalIndex];
					let bucket = bucketOf[sourceIndex];
					sortedIndices[bucketOffsets[bucket] + localIndex[sourceIndex]] = sourceIndex;
				}
			`,
		});

		const sortBucketsModule = device.createShaderModule({
			code: `
				@group(0) @binding(0) var<storage, read> bucketCounts: array<u32>;
				@group(0) @binding(1) var<storage, read> bucketOffsets: array<u32>;
				@group(0) @binding(2) var<storage, read_write> sortedIndices: array<u32>;

				@compute @workgroup_size(256)
				fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
					let bucket = gid.x + gid.y * 65535u * 256u;
					if (bucket >= 65536u) { return; }
					let count = bucketCounts[bucket];
					if (count <= 1u) { return; }
					let start = bucketOffsets[bucket];
					let maxSort = min(count, 2048u);
					for (var i = 1u; i < maxSort; i = i + 1u) {
						let current = sortedIndices[start + i];
						var j = i;
						while (j > 0u) {
							let previous = sortedIndices[start + j - 1u];
							if (current > previous) {
								sortedIndices[start + j] = previous;
								j = j - 1u;
							} else {
								break;
							}
						}
						sortedIndices[start + j] = current;
					}
				}
			`,
		});

		const renderModule = device.createShaderModule({
			code:
				commonWGSL +
				`
				@group(0) @binding(1) var<storage, read> splats: array<PrecomputedSplat>;
				@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;
				@group(0) @binding(3) var<storage, read> culledCount: u32;

				struct VSOut {
					@builtin(position) pos: vec4<f32>,
					@location(0) uv: vec2<f32>,
					@location(1) col: vec3<f32>,
					@location(2) opac: f32,
					@location(3) invCov: vec3<f32>
				};

				@vertex
				fn vs_main(@builtin(instance_index) iid: u32, @location(0) quad: vec2<f32>) -> VSOut {
					if (iid >= culledCount) {
						return VSOut(vec4<f32>(0.0), vec2<f32>(0.0), vec3<f32>(0.0), 0.0, vec3<f32>(0.0));
					}
					let index = sortedIndices[iid];
					let splat = splats[index];
					let radius = max(0.5, splat.posRadius.z * u.scaleMul);
					var out: VSOut;
					out.pos = vec4<f32>(
						(splat.posRadius.x / u.width) * 2.0 - 1.0 + quad.x * (radius / u.width) * 2.0,
						1.0 - (splat.posRadius.y / u.height) * 2.0 + quad.y * (radius / u.height) * 2.0,
						0.5,
						1.0
					);
					out.uv = vec2<f32>(quad.x * radius, -quad.y * radius);
					out.col = splat.color.xyz;
					out.opac = splat.posRadius.w;
					out.invCov = splat.invCov.xyz;
					return out;
				}

				@fragment
				fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
					let q =
						input.invCov.x * input.uv.x * input.uv.x +
						2.0 * input.invCov.y * input.uv.x * input.uv.y +
						input.invCov.z * input.uv.y * input.uv.y;
					let alpha = input.opac * exp(-0.5 * q);
					return vec4<f32>(input.col * alpha * u.colorGain, alpha);
				}
			`,
		});

		const createBuffer = (size, usage) => device.createBuffer({ size, usage });
		const uniformBuffer = createBuffer(
			112,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		);
		const quadBuffer = createBuffer(
			32,
			GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		);
		device.queue.writeBuffer(
			quadBuffer,
			0,
			new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
		);

		const computePipe = device.createComputePipeline({
			layout: "auto",
			compute: { module: computeModule, entryPoint: "main" },
		});
		const clearPipe = device.createComputePipeline({
			layout: "auto",
			compute: { module: clearModule, entryPoint: "main" },
		});
		const prefixPipe = device.createComputePipeline({
			layout: "auto",
			compute: { module: prefixModule, entryPoint: "main" },
		});
		const scatterPipe = device.createComputePipeline({
			layout: "auto",
			compute: { module: scatterModule, entryPoint: "main" },
		});
		const sortBucketsPipe = device.createComputePipeline({
			layout: "auto",
			compute: { module: sortBucketsModule, entryPoint: "main" },
		});
		const renderPipe = device.createRenderPipeline({
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
						format,
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
		});

		state = {
			device,
			context,
			format,
			data: null,
			pipes: {
				clearPipe,
				computePipe,
				prefixPipe,
				scatterPipe,
				sortBucketsPipe,
				renderPipe,
			},
			bgs: null,
			bufs: {
				uniformBuffer,
				quadBuffer,
				frame: null,
			},
		};
		logMessage("WebGPU renderer ready.");
		return state;
	};

	const createFrameResources = (rendererState, data) => {
		const { device, pipes } = rendererState;
		const createBuffer = (size, usage) => device.createBuffer({ size, usage });
		const sourceBuffer = createBuffer(
			data.interleaved.byteLength,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		);
		const splatBuffer = createBuffer(data.count * 48, GPUBufferUsage.STORAGE);
		const countBuffer = createBuffer(
			65536 * 4,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		);
		const bucketOfBuffer = createBuffer(data.count * 4, GPUBufferUsage.STORAGE);
		const localIndexBuffer = createBuffer(data.count * 4, GPUBufferUsage.STORAGE);
		const offsetBuffer = createBuffer(65537 * 4, GPUBufferUsage.STORAGE);
		const sortedBuffer = createBuffer(data.count * 4, GPUBufferUsage.STORAGE);
		const culledBuffer = createBuffer(data.count * 4, GPUBufferUsage.STORAGE);
		const culledCountBuffer = createBuffer(
			4,
			GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		);
		device.queue.writeBuffer(sourceBuffer, 0, data.interleaved);

		const computeBG = device.createBindGroup({
			layout: pipes.computePipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: rendererState.bufs.uniformBuffer } },
				{ binding: 1, resource: { buffer: sourceBuffer } },
				{ binding: 2, resource: { buffer: splatBuffer } },
				{ binding: 3, resource: { buffer: countBuffer } },
				{ binding: 4, resource: { buffer: bucketOfBuffer } },
				{ binding: 5, resource: { buffer: localIndexBuffer } },
				{ binding: 6, resource: { buffer: culledBuffer } },
				{ binding: 7, resource: { buffer: culledCountBuffer } },
			],
		});
		const clearBG = device.createBindGroup({
			layout: pipes.clearPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: countBuffer } },
				{ binding: 1, resource: { buffer: culledCountBuffer } },
			],
		});
		const prefixBG = device.createBindGroup({
			layout: pipes.prefixPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: countBuffer } },
				{ binding: 1, resource: { buffer: offsetBuffer } },
			],
		});
		const scatterBG = device.createBindGroup({
			layout: pipes.scatterPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: bucketOfBuffer } },
				{ binding: 1, resource: { buffer: localIndexBuffer } },
				{ binding: 2, resource: { buffer: offsetBuffer } },
				{ binding: 3, resource: { buffer: sortedBuffer } },
				{ binding: 4, resource: { buffer: culledBuffer } },
				{ binding: 5, resource: { buffer: culledCountBuffer } },
			],
		});
		const sortBucketsBG = device.createBindGroup({
			layout: pipes.sortBucketsPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: countBuffer } },
				{ binding: 1, resource: { buffer: offsetBuffer } },
				{ binding: 2, resource: { buffer: sortedBuffer } },
			],
		});
		const renderBG = device.createBindGroup({
			layout: pipes.renderPipe.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: rendererState.bufs.uniformBuffer } },
				{ binding: 1, resource: { buffer: splatBuffer } },
				{ binding: 2, resource: { buffer: sortedBuffer } },
				{ binding: 3, resource: { buffer: culledCountBuffer } },
			],
		});

		return {
			sourceBuffer,
			bgs: { clearBG, computeBG, prefixBG, scatterBG, sortBucketsBG, renderBG },
			bufs: {
				sourceBuffer,
				splatBuffer,
				countBuffer,
				bucketOfBuffer,
				localIndexBuffer,
				offsetBuffer,
				sortedBuffer,
				culledBuffer,
				culledCountBuffer,
				all: [
					sourceBuffer,
					splatBuffer,
					countBuffer,
					bucketOfBuffer,
					localIndexBuffer,
					offsetBuffer,
					sortedBuffer,
					culledBuffer,
					culledCountBuffer,
				],
			},
		};
	};

	const loadFrameData = async (data) => {
		const rendererState = await ensureState();
		if (
			rendererState.data &&
			rendererState.data.count === data.count &&
			rendererState.bufs.frame
		) {
			rendererState.device.queue.writeBuffer(
				rendererState.bufs.frame.sourceBuffer,
				0,
				data.interleaved,
			);
			rendererState.data = data;
			return data.count;
		}

		destroyFrameResources();
		const resources = createFrameResources(rendererState, data);
		rendererState.data = data;
		rendererState.bgs = resources.bgs;
		rendererState.bufs.frame = resources.bufs;
		return data.count;
	};

	const renderFrameFromCamera = async (camera) => {
		const rendererState = await ensureState();
		if (!rendererState.data || !rendererState.bgs || !rendererState.bufs.frame) {
			throw new Error("Renderer not initialized with frame data.");
		}

		const w2c = camera.w2c
			? toColumnMajor(camera.w2c, camera.rowMajor, columnMajor)
			: invert4x4(
					toColumnMajor(camera.c2w || [], camera.rowMajor, columnMajor),
					inverted,
				);

		renderCanvas.width = camera.width;
		renderCanvas.height = camera.height;

		uniforms.set(w2c, 0);
		uniforms[16] = camera.fx;
		uniforms[17] = camera.fy;
		uniforms[18] = camera.cx;
		uniforms[19] = camera.cy;
		uniforms[20] = camera.width;
		uniforms[21] = camera.height;
		uniforms[22] = 1;
		uniforms[23] = 1;
		uniforms[24] = camera.near ?? 0.01;
		uniforms[25] = camera.far ?? 1000;
		uniforms[26] = 0;
		rendererState.device.queue.writeBuffer(
			rendererState.bufs.uniformBuffer,
			0,
			uniforms,
		);

		const { device, context, pipes, bgs, bufs } = rendererState;
		const encoder = device.createCommandEncoder();
		const passClear = encoder.beginComputePass();
		passClear.setPipeline(pipes.clearPipe);
		passClear.setBindGroup(0, bgs.clearBG);
		passClear.dispatchWorkgroups(Math.ceil(65536 / 256));
		passClear.end();

		const count = rendererState.data.count;
		const numWorkgroups = Math.ceil(count / 256);
		const workgroupX = Math.min(numWorkgroups, 65535);
		const workgroupY = Math.ceil(numWorkgroups / 65535);

		const passCompute = encoder.beginComputePass();
		passCompute.setPipeline(pipes.computePipe);
		passCompute.setBindGroup(0, bgs.computeBG);
		passCompute.dispatchWorkgroups(workgroupX, workgroupY);
		passCompute.end();

		const passPrefix = encoder.beginComputePass();
		passPrefix.setPipeline(pipes.prefixPipe);
		passPrefix.setBindGroup(0, bgs.prefixBG);
		passPrefix.dispatchWorkgroups(1);
		passPrefix.end();

		const passScatter = encoder.beginComputePass();
		passScatter.setPipeline(pipes.scatterPipe);
		passScatter.setBindGroup(0, bgs.scatterBG);
		passScatter.dispatchWorkgroups(workgroupX, workgroupY);
		passScatter.end();

		const passSortBuckets = encoder.beginComputePass();
		passSortBuckets.setPipeline(pipes.sortBucketsPipe);
		passSortBuckets.setBindGroup(0, bgs.sortBucketsBG);
		passSortBuckets.dispatchWorkgroups(Math.ceil(65536 / 256));
		passSortBuckets.end();

		const renderPass = encoder.beginRenderPass({
			colorAttachments: [
				{
					view: context.getCurrentTexture().createView(),
					clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
		});
		renderPass.setPipeline(pipes.renderPipe);
		renderPass.setBindGroup(0, bgs.renderBG);
		renderPass.setVertexBuffer(0, bufs.quadBuffer);
		renderPass.draw(4, count, 0, 0);
		renderPass.end();
		device.queue.submit([encoder.finish()]);
		return rendererState.data.count;
	};

	return {
		loadFrameData,
		renderFrameFromCamera,
		dispose() {
			destroyFrameResources();
			if (state) {
				state.bufs.uniformBuffer.destroy();
				state.bufs.quadBuffer.destroy();
				state.context.unconfigure();
			}
			state = null;
		},
	};
}
