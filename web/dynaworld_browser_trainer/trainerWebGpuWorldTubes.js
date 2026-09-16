import {
	DynamicSplatWebGpu3dTrainer, makeInitialSplats, normalizeDatasetGeometry,
	packPreviewCamera, resolveCamerasPerStep,
	assertStorageBufferFits,
} from "./trainerWebGpu3d.js";
import { frameTime01, nearestFrameIndex, decodeFrameRgb, resolveFrameBank, FRAME_BANK_FORMAT_RGBA8 } from "./dataset.js";
import { computeSnapshotMetrics, renderSnapshotFrame } from "./snapshotMetrics.js";
import { browserLearningRates } from "./trainingSchedule.js";
import { WORLD_TUBES_MODEL_MODE, WORLD_TUBES_PARAMETER_SCHEMA } from "./worldTubesMath.js";
import { worldTubesTrainWgsl, worldTubesUpdateWgsl, worldTubesRenderWgsl,
	WORLD_TUBES_PRESENT_WGSL } from "./worldTubesWgsl.js";
export { WORLD_TUBES_MODEL_MODE, WORLD_TUBES_PARAMETER_SCHEMA };
const MAX_SAMPLES = 192, MAX_CAMERAS = 4, MAX_TUBES = 4096;

export function makeWorldTubeInitialParams(source, { temporalSigma = 0.30 } = {}) {
	if (!(source instanceof Float32Array) || source.length % 24) throw new TypeError("Expected complete 24-float world atoms.");
	if (!(temporalSigma > 0 && temporalSigma <= 1)) throw new RangeError("temporalSigma must be in (0, 1].");
	const params = source.slice();
	for (let b=0;b<params.length;b+=24) {
		params[b+3]=Math.log(Math.max(Math.exp(-4),temporalSigma));
		params[b+7]=Math.min(1,Math.max(0,params[b+7]));
		params.fill(0,b+8,b+12);
	}
	return params;
}
export function worldTubeTemporalDensity(params, base, time) {
	return Math.exp(-0.5*((time-params[base+7])/Math.exp(params[base+3]))**2);
}
function hash(value) {
	value=Math.imul(value^(value>>>16),0x7feb352d);value=Math.imul(value^(value>>>15),0x846ca68b);
	return (value^(value>>>16))>>>0;
}
function cameraIdentity(dataset) {
	return JSON.stringify(dataset.cameras.map(c=>[c.name,c.role,...c.worldToCamera,...c.intrinsics]));
}

// Inherit only device/readback/continuation plumbing. No 3DGS render or backward
// pipeline is created or dispatched by this backend.
export class WorldTubesWebGpuTrainer extends DynamicSplatWebGpu3dTrainer {
	constructor(canvas) {
		super(canvas);this.skipSampleGradientAllocation=true;
		this.sampleData=new Float32Array(MAX_SAMPLES*8);
	}
	targetBufferByteLength() { return 16; }
	async init(dataset, options={}) {
		const count=options.splatCount??768;
		if(!Number.isSafeInteger(count)||count<1||count>MAX_TUBES)throw new RangeError("World Tubes supports 1-4096 atoms.");
		this.initialTemporalSigma=options.temporalSigma??0.30;
		this.replayCompiler=Boolean(options.replayCompiler);
		await super.init(dataset,{splatCount:count});
		this.initialSplatCount=count;this.activeSplatCount=count;
		return this;
	}
	async createPipelines() {
		const sources={train:worldTubesTrainWgsl({replayCompiler:this.replayCompiler}),update:worldTubesUpdateWgsl(),render:worldTubesRenderWgsl(),present:WORLD_TUBES_PRESENT_WGSL};
		const modules={};
		for(const [name,code] of Object.entries(sources)) {
			const module=this.device.createShaderModule({label:"world-tubes:"+name,code});
			const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
			if(errors.length)throw new Error(name+" WGSL: "+errors.map(m=>m.lineNum+": "+m.message).join("\n"));
			modules[name]=module;
		}
		this.pipelines={};
		for(const [key,module,entryPoint] of [["compile","train","compile_traces"],["train","train","train_rays"],["update","update","update"],["render","render","render"]]) {
			this.pipelines[key]=await this.device.createComputePipelineAsync({label:"world-tubes:"+key,layout:"auto",compute:{module:modules[module],entryPoint}});
		}
		this.pipelines.present=await this.device.createRenderPipelineAsync({layout:"auto",
			vertex:{module:modules.present,entryPoint:"vs"},fragment:{module:modules.present,entryPoint:"fs",targets:[{format:this.format}]}});
	}
	createBuffers() {
		const make=(size,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC)=> {
			assertStorageBufferFits("World Tubes buffer",size,this.storageBufferLimit);
			return this.device.createBuffer({size,usage});
		};
		this.initialParams=makeWorldTubeInitialParams(makeInitialSplats(this.dataset,this.splatCount),{temporalSigma:this.initialTemporalSigma});
		this.buffers={params:[make(this.initialParams.byteLength),make(this.initialParams.byteLength)],
			firstMoment:make(this.initialParams.byteLength),secondMoment:make(this.initialParams.byteLength),
			stats:make(this.splatCount*16),worldGrads:make(this.initialParams.byteLength),
			paramsReadback:make(this.initialParams.byteLength,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),
			samples:make(MAX_SAMPLES*32),tape:make(MAX_SAMPLES*this.splatCount*16),pixels:make(MAX_SAMPLES*16)};
		for(const buffer of this.buffers.params)this.device.queue.writeBuffer(buffer,0,this.initialParams);
		for(const lane of ["train","preview"]) {
			const scale=lane==="train"?1:Math.min(1,144/this.dataset.height);
			const width=Math.max(1,Math.round(this.dataset.width*scale)),height=Math.max(1,Math.round(this.dataset.height*scale));
			const tiles=Math.ceil(width/16)*Math.ceil(height/16);
			this[lane+"Size"]={width,height};
			this.buffers[lane+"Config"]=make(64,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
			this.buffers[lane+"Cameras"]=make(MAX_CAMERAS*80);
			this.buffers[lane+"Traces"]=make(MAX_CAMERAS*this.splatCount*80);
			this.buffers[lane+"Bins"]=make(MAX_CAMERAS*tiles*(1+this.splatCount)*4);
		}
		this.buffers.traceGrads=make(MAX_CAMERAS*this.splatCount*80);
		this.previewTexture=this.device.createTexture({size:[this.previewSize.width*MAX_CAMERAS,this.previewSize.height],format:"rgba8unorm",
			usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
		const bufferBytes=Object.fromEntries(Object.entries(this.buffers).map(([k,v])=>[k,Array.isArray(v)?v.reduce((n,b)=>n+b.size,0):v.size]));
		this.memoryPlan=Object.freeze({parameterSchema:WORLD_TUBES_PARAMETER_SCHEMA,trainingUnit:"uniform pixel/time rays",
			checkpointPrecision:"f32",bufferBytes,allocatedBytes:Object.values(bufferBytes).reduce((a,b)=>a+b,0)+this.previewSize.width*MAX_CAMERAS*this.previewSize.height*4,
			traceBytes:this.buffers.trainTraces.size,rayTapeBytes:this.buffers.tape.size,staticWarmupSteps:0,
			compileUnit:"one atom/camera/optimizer step; shared across requested times",maxSamples:MAX_SAMPLES,maxCameras:MAX_CAMERAS});
	}
	group(pipeline,entries) {
		return this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:entries.map(([binding,value])=>({binding,resource:value.createView?value.createView():{buffer:value}}))});
	}
	createBindGroups() {
		const b=this.buffers,p=this.pipelines;
		this.bindGroups={};
		for(const lane of ["train","preview"]) {
			this.bindGroups[lane+"Compile"]=b.params.map(params=>this.group(p.compile,[[0,b[lane+"Config"]],[1,params],[2,b[lane+"Cameras"]],[3,b[lane+"Traces"]],[4,b[lane+"Bins"]]]));
		}
		this.bindGroups.train=b.params.map(params=>this.group(p.train,[[0,b.trainConfig],
			...(this.replayCompiler?[[1,params],[2,b.trainCameras]]:[[3,b.trainTraces]]),
			[4,b.trainBins],[5,b.traceGrads],[6,b.samples],[7,b.tape],[8,b.pixels]]));
		this.bindGroups.update=b.params.map((params,i)=>this.group(p.update,[[0,b.trainConfig],[1,params],[2,b.params[1-i]],
			[3,b.firstMoment],[4,b.secondMoment],[5,b.trainCameras],[6,b.traceGrads],[7,b.worldGrads]]));
		this.bindGroups.render=this.group(p.render,[[0,b.previewConfig],[3,b.previewTraces],[4,b.previewBins],[5,this.previewTexture]]);
		this.bindGroups.present=this.group(p.present,[[0,this.previewTexture],[1,b.previewConfig]]);
	}
	writeConfig(lane,cameraCount,sampleCount=0,options={}) {
		const {width,height}=this[lane+"Size"],bytes=new ArrayBuffer(64),u=new Uint32Array(bytes),f=new Float32Array(bytes);
		u.set([this.splatCount,cameraCount,width,height,sampleCount,this.stepCount,Math.ceil(width/16),Math.ceil(height/16)]);
		const lr=browserLearningRates(options.learningRate??1.25,this.stepCount,options.learningRateDecay??true);
		f.set([lr.position,lr.motion,lr.color,lr.opacity,width/height,(0.3/height)**2,options.time??0.35,options.renderMode??0],8);
		this.device.queue.writeBuffer(this.buffers[lane+"Config"],0,bytes);
	}
	writeCameras(lane,cameras,scale=1) {
		const data=new Float32Array(cameras.length*20);
		cameras.forEach((camera,i)=>data.set(packPreviewCamera(camera,scale),i*20));
		this.device.queue.writeBuffer(this.buffers[lane+"Cameras"],0,data);
	}
	pass(encoder,pipeline,group,x,y=1) {
		const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(x,y);pass.end();
	}
	compile(encoder,lane,cameraCount) {
		encoder.clearBuffer(this.buffers[lane+"Bins"]);
		this.pass(encoder,this.pipelines.compile,this.bindGroups[lane+"Compile"][this.currentIndex],Math.ceil(this.splatCount*cameraCount/64));
	}
	// Uniform sampling is explicit: no hidden support guard or static-target loss.
	// Stratify times within the camera batch so one compile serves multiple times.
	trainingSamples(options={}) {
		const cameraCount=Math.min(MAX_CAMERAS,resolveCamerasPerStep(this.trainViewIndices.length,options.camerasPerStep));
		const count=options.samplesPerStep??96;
		if(!Number.isSafeInteger(count)||count<1||count>MAX_SAMPLES)throw new RangeError("samplesPerStep must be 1-192.");
		const views=Array.from({length:Math.min(count,cameraCount)},(_,i)=>this.trainViewIndices[(this.stepCount*cameraCount+i)%this.trainViewIndices.length]);
		this.lastCameraBatch=views;this.lastCameraBatchStart=(this.stepCount*cameraCount)%this.trainViewIndices.length;
		const {width,height,frameCount}=this.dataset,bank=resolveFrameBank(this.dataset),scale=bank.format===FRAME_BANK_FORMAT_RGBA8?1/255:1;
		for(let i=0;i<count;i++) {
			const local=i%views.length,frame=(Math.floor(i/views.length)+this.stepCount)%frameCount;
			const pixel=hash(Math.imul(this.stepCount+1,MAX_SAMPLES)+i)%(width*height);
			const base=((views[local]*frameCount+frame)*width*height+pixel)*4;
			this.sampleData.set([(pixel%width+0.5)/height,(Math.floor(pixel/width)+0.5)/height,frameTime01(this.dataset,frame),local,
				bank.data[base]*scale,bank.data[base+1]*scale,bank.data[base+2]*scale,0],i*8);
		}
		return {data:this.sampleData.subarray(0,count*8),views,count};
	}
	trainStep(options={}) {
		const samples=this.trainingSamples(options);
		this.submitSamples(samples.data,samples.views.map(i=>this.dataset.cameras[i]),options);
	}
	submitSamples(data,cameras,options={}) {
		const count=data.length/8;
		if(count<1||count>MAX_SAMPLES||!Number.isInteger(count)||cameras.length<1||cameras.length>MAX_CAMERAS)throw new RangeError("Invalid world-tube ray batch.");
		this.writeConfig("train",cameras.length,count,options);this.writeCameras("train",cameras);
		this.device.queue.writeBuffer(this.buffers.samples,0,data);
		const encoder=this.device.createCommandEncoder();this.compile(encoder,"train",cameras.length);
		encoder.clearBuffer(this.buffers.traceGrads);
		this.pass(encoder,this.pipelines.train,this.bindGroups.train[this.currentIndex],Math.ceil(count/64));
		this.pass(encoder,this.pipelines.update,this.bindGroups.update[this.currentIndex],Math.ceil(this.splatCount/64));
		this.device.queue.submit([encoder.finish()]);this.currentIndex=1-this.currentIndex;this.stepCount++;this.lastSampleCount=count;
	}
	async readBuffer(buffer,size=buffer.size) {
		const staging=this.device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
		try {
			const encoder=this.device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,staging,0,size);this.device.queue.submit([encoder.finish()]);
			await staging.mapAsync(GPUMapMode.READ);const data=new Float32Array(staging.getMappedRange().slice(0));staging.unmap();return data;
		} finally {staging.destroy();}
	}
	async readLoss() {
		if(!this.lastSampleCount)return Number.NaN;
		const count=this.lastSampleCount,pixels=await this.readBuffer(this.buffers.pixels,count*16);
		let loss=0;for(let i=0;i<count;i++)loss+=pixels[i*4+3]/count;
		this.lastLossBreakdown={rgbMse:loss,sampleCount:count,compiledAtoms:this.splatCount*(this.lastCameraBatch?.length??1)};
		return loss;
	}
	async readValidationMetrics() {
		const params=await this.readParams();
		const train=computeSnapshotMetrics(this.dataset,params,{modelMode:2,views:this.trainViewIndices,frames:"all"});
		const views=this.dataset.cameras.flatMap((c,i)=>c.role==="heldout"?[i]:[]);
		const heldout=views.length?computeSnapshotMetrics(this.dataset,params,{modelMode:2,views,frames:"all"}):null;
		return {gridLoss:train.mse,gridMae:train.mae,gridPsnr:train.psnr,gridSsim:train.ssim,motionCoverage:train.coverage,
			heldoutLoss:heldout?.mse,heldoutPsnr:heldout?.psnr,heldoutSsim:heldout?.ssim};
	}
	async readPreviewErrorImage({time=0.35,viewIndex=this.trainViewIndices[0]}={}) {
		const frame=nearestFrameIndex(this.dataset,time),params=await this.readParams();
		const image=renderSnapshotFrame(this.dataset,params,{modelMode:2,viewIndex,frameIndex:frame});
		const target=decodeFrameRgb(this.dataset,viewIndex,frame),data=new Uint8ClampedArray(image.width*image.height*4);
		let meanAbs=0;
		for(let pixel=0;pixel<image.width*image.height;pixel++) {
			let squared=0;for(let c=0;c<3;c++)squared+=(image.rgb[pixel*3+c]-target[pixel*3+c])**2;
			const error=Math.sqrt(squared/3),heat=Math.min(1,error*3);meanAbs+=error;
			data.set([255*heat,255*Math.max(0,heat*1.6-0.3),255*Math.max(0.05,0.4-heat*0.3),255],pixel*4);
		}
		return {frame,viewIndex,width:image.width,height:image.height,data,meanAbs:meanAbs/(image.width*image.height),maxAbs:1};
	}
	maintainDensity() { return 0; }
	continuationContract() {return {...super.continuationContract(),parameterSchema:WORLD_TUBES_PARAMETER_SCHEMA,cameraIdentity:cameraIdentity(this.dataset)};}
	assertContinuationStateCompatible(state) {
		super.assertContinuationStateCompatible(state);
		if(state.contract.cameraIdentity!==cameraIdentity(this.dataset))throw new Error("Continuation calibrated cameras do not match.");
		return state;
	}
	replaceTemporalPage(dataset) {
		const next=normalizeDatasetGeometry(dataset);
		if(next.width!==this.dataset.width||next.height!==this.dataset.height||cameraIdentity(next)!==cameraIdentity(this.dataset)
			||Math.abs(next.geometryScale-this.dataset.geometryScale)>1e-8)throw new Error("Temporal page must preserve geometry, cameras, and raster dimensions.");
		for(let i=0;i<next.frameCount;i++)frameTime01(next,i);
		this.dataset=next;return next;
	}
	render(time=0.35,_mode=2,_sigma=0.30,renderMode=0,viewIndex=0,viewIndices=null,previewCameras=null) {
		if(!this.context)return;
		const views=viewIndices??[viewIndex];
		const cameras=previewCameras?.length?previewCameras.slice(0,3):views.slice(0,3).map(i=>this.dataset.cameras[i]);
		this.writeConfig("preview",cameras.length,0,{time,renderMode});
		this.writeCameras("preview",cameras,previewCameras?.length?this.dataset.geometryScale:1);
		const encoder=this.device.createCommandEncoder();this.compile(encoder,"preview",cameras.length);
		this.pass(encoder,this.pipelines.render,this.bindGroups.render,Math.ceil(this.previewSize.width*cameras.length/8),Math.ceil(this.previewSize.height/8));
		const pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});
		// The texture reserves four camera slots. Crop presentation to live panels.
		pass.setPipeline(this.pipelines.present);pass.setBindGroup(0,this.bindGroups.present);
		pass.setViewport(0,0,this.canvas.width,this.canvas.height,0,1);
		pass.setScissorRect(0,0,this.canvas.width,this.canvas.height);pass.draw(3);pass.end();this.device.queue.submit([encoder.finish()]);
	}
	dispose() {this.previewTexture?.destroy();super.dispose();this.device?.destroy();}
}

export const WORLD_TUBES_BROWSER_CONTRACT=Object.freeze({
	name:"World Tubes · affine STAR WebGPU",
	state:"shared SPD(4) world atoms; no per-camera trainable parameters",
	training:"uniform RGB MSE; UVT ray adjoint and world-compiler VJP plus Adam in WGSL",
	rendering:"compiled UVT marginal and per-ray conditional-depth source-over, peak-preserving opacity",
	omissions:Object.freeze(["projective interval-atlas and moving-camera charts","visibility-event derivatives and retained-fiber integration","adaptive density growth"]),
});
