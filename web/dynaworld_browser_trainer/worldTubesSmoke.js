import { WorldTubesWebGpuTrainer } from "./trainerWebGpuWorldTubes.js";
import { WORLD_COMPONENTS } from "./worldTubesMath.js";
import { renderSnapshotFrame } from "./snapshotMetrics.js";
import { createNonblockingTrainer } from "./nonblockingTrainerClient.js";
import { WorkerEvent } from "./workerProtocol.js";

function check(condition,message) { if(!condition)throw new Error(message); }
function maxError(actual,expected) {
	check(actual.length===expected.length,"comparison dimensions differ");
	return Math.max(...Array.from(actual,(v,i)=>Math.abs(v-expected[i])));
}
function compare(actual,expected,absolute,relative,label) {
	let error=0;
	check(actual.length===expected.length,label+" dimensions");
	for(let i=0;i<actual.length;i++){
		const difference=Math.abs(actual[i]-expected[i]);
		check(Number.isFinite(actual[i])&&difference<=absolute+relative*Math.abs(expected[i]),label+"["+i+"]: "+actual[i]+" != "+expected[i]);
		error=Math.max(error,difference);
	}
	return error;
}
function fixtureDataset(reference,{scale=1,times=[0,0.13,0.61,1],page=0}={}) {
	const width=reference.width*scale,height=reference.height*scale;
	const cameras=[...reference.cameras,{...reference.cameras[0],name:"heldout",role:"heldout"}];
	const frames=new Float32Array(width*height*times.length*cameras.length*4);
	for(let i=0;i<frames.length;i+=4)frames.set([0.35,0.25,0.45,1],i);
	return {name:"World Tubes independent parity",width,height,frameCount:times.length,viewCount:3,
		trainViewCount:2,trainViewIndices:[0,1],cameras,frames,backgrounds:new Float32Array(width*height*3*4),
		frameTimesNormalized:times,frameIndices:times.map(t=>Math.round(t*100)),durationSeconds:1,
		temporalPageIndex:page,seedPointCount:2,seedPoints:Float32Array.of(-0.07,0.02,1,0.7,0.2,0.4,0.06,-0.04,1,0.2,0.7,0.4),
		motionSamples:new Uint32Array(),staticSamples:new Uint32Array(),comparisonViewIndices:[0,1,2],
		heldoutViewIndex:2,sourceViewIndex:0,targetViewIndex:1};
}
async function readPreview(trainer) {
	const {width,height}=trainer.previewSize;
	const bytesPerRow=Math.ceil(width*4/256)*256;
	const buffer=trainer.device.createBuffer({size:bytesPerRow*height,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
	try {
		const encoder=trainer.device.createCommandEncoder();
		encoder.copyTextureToBuffer({texture:trainer.previewTexture},{buffer,bytesPerRow},[width,height]);
		trainer.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);
		const raw=new Uint8Array(buffer.getMappedRange()),rgb=new Float32Array(width*height*3);
		for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let c=0;c<3;c++)rgb[(y*width+x)*3+c]=raw[y*bytesPerRow+x*4+c]/255;
		buffer.unmap();return rgb;
	}finally{buffer.destroy();}
}
function event(client,type,action) {
	return new Promise((resolve,reject)=>{
		const timeout=setTimeout(()=>finish(new Error("Timed out waiting for "+type)),30000);
		const done=e=>finish(null,e.detail),fail=e=>finish(new Error(e.detail.message));
		function finish(error,value){clearTimeout(timeout);client.removeEventListener(type,done);client.removeEventListener(WorkerEvent.ERROR,fail);error?reject(error):resolve(value);}
		client.addEventListener(type,done);client.addEventListener(WorkerEvent.ERROR,fail);action();
	});
}
async function run() {
	const report={passed:false},reference=await fetch("./tests/fixtures/world_tubes_reference.json").then(r=>r.json());
	globalThis.__worldTubesReport=report;
	const dataset=fixtureDataset(reference),data=Float32Array.from(reference.samples.flat());
	const trainer=new WorldTubesWebGpuTrainer(document.querySelector("#smokeCanvas"));
	try {
		await trainer.init(dataset,{splatCount:2});trainer.device.pushErrorScope("validation");
		const params=Float32Array.from(reference.params);
		for(const buffer of trainer.buffers.params)trainer.device.queue.writeBuffer(buffer,0,params);
		trainer.submitSamples(data,reference.cameras,{learningRate:0});
		const traces=await trainer.readBuffer(trainer.buffers.trainTraces,reference.traces.length*4);
		report.traceMaxError=compare(traces,reference.traces,2e-4,3e-5,"UVT trace");
		const pixels=await trainer.readBuffer(trainer.buffers.pixels,data.length/8*16);
		const rgb=Float32Array.from(reference.rgb,(_,i)=>pixels[Math.floor(i/3)*4+i%3]);
		report.rgbMaxError=compare(rgb,reference.rgb,2e-6,2e-5,"render RGB");
		const gradient=await trainer.readBuffer(trainer.buffers.worldGrads);
		report.worldGradientMaxError=compare(gradient,reference.gradients,2e-6,3e-4,"world gradient");
		report.gradientFamilies=WORLD_COMPONENTS.length;
		report.lossError=Math.abs(await trainer.readLoss()-reference.loss);
		check(report.lossError<2e-6,"loss mismatch");
		trainer.render(0.61,2,0.30,0,0);
		const preview=await readPreview(trainer);
		const expected=renderSnapshotFrame(trainer.dataset,params,{modelMode:2,frameIndex:2,viewIndex:0});
		report.previewMaxError=compare(preview,expected.rgb,1/255+2e-5,0,"preview");
		check(preview.some(v=>v>0.1),"blank preview");
		const movedCamera={...reference.cameras[0],worldToCamera:reference.cameras[0].worldToCamera.slice()};
		movedCamera.worldToCamera[3]+=0.06;
		trainer.render(0.61,2,0.30,0,0,null,[movedCamera]);
		const moved=await readPreview(trainer);
		const expectedMoved=renderSnapshotFrame(trainer.dataset,params,{modelMode:2,frameIndex:2,camera:movedCamera});
		report.orbitMaxError=compare(moved,expectedMoved.rgb,1/255+2e-5,0,"orbit");
		check(maxError(moved,preview)>0.01,"orbit did not change pixels");
		report.initialLoss=await trainer.readLoss();
		for(let step=0;step<64;step++)trainer.submitSamples(data,reference.cameras,{learningRate:2,learningRateDecay:false});
		report.finalLoss=await trainer.readLoss();
		check(report.finalLoss<report.initialLoss*0.9,"optimizer did not improve fixture loss by 10%");
		const state=await trainer.exportContinuationState();
		trainer.submitSamples(data,reference.cameras,{learningRate:2});
		await trainer.restoreContinuationState(state);
		const restored=await trainer.exportContinuationState();
		for(const key of ["params","firstMoment","secondMoment"])check(maxError(restored[key],state[key])===0,"restore changed "+key);
		report.restoredStep=restored.stepCount;
		const gpuError=await trainer.device.popErrorScope();check(!gpuError,gpuError?.message);
		report.adapter=trainer.adapterName;
	} finally {trainer.dispose();}

	// Exercise the same worker, validation worker, OffscreenCanvas and transition
	// messages as the SPA, not a second bespoke trainer interface.
	const canvas=document.createElement("canvas");canvas.width=240;canvas.height=180;document.body.append(canvas);
	const client=createNonblockingTrainer();
	try {
		const ready=await client.init({dataset,canvas,trainerOptions:{backend:"world-tubes",splatCount:2},
			trainOptions:{samplesPerStep:32,camerasPerStep:2},schedule:{renderFps:0}});
		check(ready.capabilities.offscreenRender,"OffscreenCanvas not active");
		await event(client,WorkerEvent.METRICS,()=>client.step(4));
		const page=fixtureDataset(reference,{times:[0.31,0.87],page:1});
		await client.switchTemporalPage(page);
		await event(client,WorkerEvent.METRICS,()=>client.step(2));
		const higher=fixtureDataset(reference,{scale:4,times:[0.31,0.87],page:1});
		const stage=await client.switchDataset(higher);
		check(stage.step===6,"resolution continuation lost optimizer step");
		await event(client,WorkerEvent.METRICS,()=>client.step(2));
		const validation=await event(client,WorkerEvent.VALIDATION,()=>client.requestValidation());
		check(Number.isFinite(validation.metrics.gridLoss)&&Number.isFinite(validation.metrics.heldoutLoss),"worker validation invalid");
		report.worker={offscreen:true,paging:true,resolutionContinuation:true,step:validation.step,trainLoss:validation.metrics.gridLoss,heldoutLoss:validation.metrics.heldoutLoss};
	} finally {await event(client,WorkerEvent.DISPOSED,()=>client.dispose());}
	report.passed=true;
	globalThis.__worldTubesReport=report;
	document.querySelector("#result").textContent=JSON.stringify(report,null,2);
	document.documentElement.dataset.smoke="passed";
}
run().catch(error=>{
	document.querySelector("#result").textContent=error.stack??String(error);
	document.documentElement.dataset.smoke="failed";console.error(error);
});
