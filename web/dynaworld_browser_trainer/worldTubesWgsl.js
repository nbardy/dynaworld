import { worldTubeCompilerWgsl } from "./worldTubesMath.js";

const COMMON = /* wgsl */ `
struct Config { dims:vec4<u32>, args:vec4<u32>, rates:vec4<f32>, view:vec4<f32> }
struct Sample { a:vec4<f32>, expected:vec4<f32> }
@group(0) @binding(0) var<uniform> cfg:Config;
fn tile_count()->u32{return cfg.args.z*cfg.args.w;}
fn tile_at(a:vec2<f32>)->u32 {
  let xy=min(vec2<u32>(max(a,vec2<f32>(0))*f32(cfg.dims.w))/16u,vec2<u32>(cfg.args.z-1u,cfg.args.w-1u));
  return xy.y*cfg.args.z+xy.x;
}
fn alpha_depth(p:array<f32,20>,a:vec3<f32>)->vec2<f32>{
  let dt=a.z-p[2]; let det=p[4]*p[7]-p[5]*p[5];
  let velocity=vec2<f32>(-(p[7]*p[6]-p[5]*p[8]),-(-p[5]*p[6]+p[4]*p[8]))/det;
  let delta=a.xy-vec2<f32>(p[0],p[1])-velocity*dt;
  let spatialQ=p[4]*delta.x*delta.x+2.0*p[5]*delta.x*delta.y+p[7]*delta.y*delta.y;
  let temporalQ=max(0.0,p[9]+dot(vec2<f32>(p[6],p[8]),velocity))*dt*dt;
  let raw=p[19]*exp(-0.5*(spatialQ+temporalQ));
  let d=a-vec3<f32>(p[0],p[1],p[2]);
  let depth=p[12]+dot(vec3<f32>(p[13],p[14],p[15]),d);
  if(p[3]<0.5||spatialQ>9.0||raw<1.0/255.0||depth<=0.1){return vec2<f32>(0.0,depth);}
  return vec2<f32>(min(0.99,raw),depth);
}
`;

export function worldTubesTrainWgsl({ replayCompiler = false } = {}) {
	return COMMON + worldTubeCompilerWgsl() + /* wgsl */ `
@group(0) @binding(1) var<storage,read> params:array<array<f32,24>>;
@group(0) @binding(2) var<storage,read> cameras:array<array<f32,20>>;
@group(0) @binding(3) var<storage,read_write> traces:array<array<f32,20>>;
@group(0) @binding(4) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> grads:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read> samples:array<Sample>;
// Per-ray tape is bounded by actual atom count: never truncate visibility.
@group(0) @binding(7) var<storage,read_write> tape:array<vec4<f32>>;
@group(0) @binding(8) var<storage,read_write> pixels:array<vec4<f32>>;
fn load_trace(id:u32)->array<f32,20>{
  ${replayCompiler
		? "return compile_tube(params[id%cfg.dims.x],cameras[id/cfg.dims.x],cfg.view.x,cfg.view.y);"
		: "return traces[id];"}
}
fn bin_base(tile:u32)->u32{return cfg.dims.y*tile_count()+tile*cfg.dims.x;}
@compute @workgroup_size(64)
fn compile_traces(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=cfg.dims.x*cfg.dims.y){return;}
  let p=compile_tube(params[i%cfg.dims.x],cameras[i/cfg.dims.x],cfg.view.x,cfg.view.y);
  traces[i]=p;if(p[3]<0.5||p[19]<1.0/255.0){return;}
  let det=p[4]*p[7]-p[5]*p[5];
  let variance=vec2<f32>(p[7],p[4])/det;
  let velocity=vec2<f32>(-(p[7]*p[6]-p[5]*p[8]),-(-p[5]*p[6]+p[4]*p[8]))/det;
  let m0=vec2<f32>(p[0],p[1])-velocity*p[2];let m1=m0+velocity;
  let lower=(min(m0,m1)-3.0*sqrt(variance))*f32(cfg.dims.w);
  let upper=(max(m0,m1)+3.0*sqrt(variance))*f32(cfg.dims.w);
  if(any(upper<vec2<f32>(0))||any(lower>vec2<f32>(f32(cfg.dims.z),f32(cfg.dims.w)))){return;}
  let lo=vec2<u32>(max(lower,vec2<f32>(0)))/16u;
  let hi=min(vec2<u32>(max(upper,vec2<f32>(0)))/16u,vec2<u32>(cfg.args.z-1u,cfg.args.w-1u));
  for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
    let tile=(i/cfg.dims.x)*tile_count()+y*cfg.args.z+x;
    let slot=atomicAdd(&bins[tile],1u);atomicStore(&bins[bin_base(tile)+slot],i);
  }}
}
fn add_grad(index:u32,value:f32){
  if(value==0.0){return;}var old=atomicLoad(&grads[index]);
  loop{let exchanged=atomicCompareExchangeWeak(&grads[index],old,bitcast<u32>(bitcast<f32>(old)+value));
    if(exchanged.exchanged){break;}old=exchanged.old_value;}
}
@compute @workgroup_size(64)
fn train_rays(@builtin(global_invocation_id) gid:vec3<u32>){
  let ray=gid.x;if(ray>=cfg.args.x){return;}
  let sample=samples[ray];let a=sample.a.xyz;
  let tile=u32(sample.a.w)*tile_count()+tile_at(a.xy);
  let binCount=atomicLoad(&bins[tile]);let base=ray*cfg.dims.x;
  var count=0u;
  for(var k=0u;k<binCount;k++){
    let id=atomicLoad(&bins[bin_base(tile)+k]);let ad=alpha_depth(load_trace(id),a);
    if(ad.x==0.0){continue;}var at=count;
    loop{if(at==0u){break;}let prev=tape[base+at-1u];
      if(prev.x<ad.y||(prev.x==ad.y&&u32(prev.w)<id)){break;}
      tape[base+at]=prev;at--;}
    tape[base+at]=vec4<f32>(ad.y,ad.x,0.0,f32(id));count++;
  }
  var rgb=vec3<f32>(0);var trans=1.0;var used=0u;
  for(var k=0u;k<count;k++){
    var entry=tape[base+k];let p=load_trace(u32(entry.w));entry.z=trans;tape[base+k]=entry;
    rgb+=trans*entry.y*vec3<f32>(p[16],p[17],p[18]);trans*=1.0-entry.y;used++;
    if(trans<1e-4){break;}
  }
  let residual=rgb-sample.expected.xyz;pixels[ray]=vec4<f32>(rgb,dot(residual,residual)/3.0);
  let imageGrad=2.0*residual/(3.0*f32(cfg.args.x));var suffix=vec3<f32>(0);
  for(var reverse=used;reverse>0u;reverse--){
    let entry=tape[base+reverse-1u];let id=u32(entry.w);let p=load_trace(id);
    let color=vec3<f32>(p[16],p[17],p[18]);
    let ga=dot(imageGrad,entry.z*(color-suffix));let gc=imageGrad*entry.z*entry.y;
    add_grad(id*20u+16u,gc.x);add_grad(id*20u+17u,gc.y);add_grad(id*20u+18u,gc.z);
    if(entry.y<0.99){
      let d=a-vec3<f32>(p[0],p[1],p[2]);
      let qd=vec3<f32>(p[4]*d.x+p[5]*d.y+p[6]*d.z,p[5]*d.x+p[7]*d.y+p[8]*d.z,p[6]*d.x+p[8]*d.y+p[9]*d.z);
      let gq=-0.5*entry.y*ga;
      for(var axis=0u;axis<3u;axis++){add_grad(id*20u+axis,-2.0*qd[axis]*gq);}
      let dq=array<f32,6>(d.x*d.x,2.0*d.x*d.y,2.0*d.x*d.z,d.y*d.y,2.0*d.y*d.z,d.z*d.z);
      for(var j=0u;j<6u;j++){add_grad(id*20u+4u+j,dq[j]*gq);}
      add_grad(id*20u+19u,entry.y/p[19]*ga);
    }
    suffix=entry.y*color+(1.0-entry.y)*suffix;
  }
}
`;
}

export function worldTubesUpdateWgsl() {
	return COMMON + worldTubeCompilerWgsl() + /* wgsl */ `
@group(0) @binding(1) var<storage,read> params:array<array<f32,24>>;
@group(0) @binding(2) var<storage,read_write> next:array<array<f32,24>>;
@group(0) @binding(3) var<storage,read_write> first:array<array<f32,24>>;
@group(0) @binding(4) var<storage,read_write> second:array<array<f32,24>>;
@group(0) @binding(5) var<storage,read> cameras:array<array<f32,20>>;
@group(0) @binding(6) var<storage,read> traceGrads:array<array<f32,20>>;
@group(0) @binding(7) var<storage,read_write> worldGrads:array<array<f32,24>>;
@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid:vec3<u32>){
  let id=gid.x;if(id>=cfg.dims.x){return;}let p=params[id];var g:array<f32,24>;
  for(var camera=0u;camera<cfg.dims.y;camera++){
    let bar=compiler_vjp(p,cameras[camera],cfg.view.x,cfg.view.y,traceGrads[camera*cfg.dims.x+id]);
    for(var j=0u;j<24u;j++){g[j]+=bar[j];}
  }
  worldGrads[id]=g;var result=p;
  for(var j=0u;j<24u;j++){
    if((j>=8u&&j<=11u)||j==15u){continue;}
    let gradient=clamp(g[j],-100.0,100.0);
    let m=0.9*first[id][j]+0.1*gradient;let v=0.999*second[id][j]+0.001*gradient*gradient;
    first[id][j]=m;second[id][j]=v;
    let delta=(m/(1.0-pow(0.9,f32(cfg.args.y+1u))))/(sqrt(v/(1.0-pow(0.999,f32(cfg.args.y+1u))))+1e-8);
    var lr=cfg.rates.y;
    if(j<3u){lr=cfg.rates.x;}if(j>=12u&&j<=14u){lr=cfg.rates.x*1.285714;}
    if(j>=20u&&j<=22u){lr=cfg.rates.z;}if(j==23u){lr=cfg.rates.w;}
    result[j]-=lr*delta;
    if(j==3u){result[j]=clamp(result[j],-4.0,0.0);}
    if(j==7u){result[j]=clamp(result[j],0.0,1.0);}
    if(j>=12u&&j<=14u){result[j]=clamp(result[j],-10.0,0.0);}
    if(j>=20u&&j<=22u){result[j]=clamp(result[j],0.0,1.0);}
    if(j==23u){result[j]=clamp(result[j],-12.0,4.6);}
  }
  next[id]=result;
}`;
}

export function worldTubesRenderWgsl() {
	return COMMON + /* wgsl */ `
@group(0) @binding(3) var<storage,read> traces:array<array<f32,20>>;
@group(0) @binding(4) var<storage,read> bins:array<u32>;
@group(0) @binding(5) var output:texture_storage_2d<rgba8unorm,write>;
fn bin_base(tile:u32)->u32{return cfg.dims.y*tile_count()+tile*cfg.dims.x;}
@compute @workgroup_size(8,8)
fn render(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x>=cfg.dims.z*cfg.dims.y||gid.y>=cfg.dims.w){return;}
  let panel=gid.x/cfg.dims.z;let a=vec3<f32>((vec2<f32>(f32(gid.x%cfg.dims.z),f32(gid.y))+0.5)/f32(cfg.dims.w),cfg.view.z);
  let tile=panel*tile_count()+tile_at(a.xy);let binCount=bins[tile];
  var ids:array<u32,64>;var depths:array<f32,64>;var count=0u;var overflow=false;
  for(var k=0u;k<binCount;k++){
    let id=bins[bin_base(tile)+k];let ad=alpha_depth(traces[id],a);if(ad.x==0.0){continue;}
    if(count==64u){overflow=true;break;}var at=count;
    loop{if(at==0u){break;}let d=depths[at-1u];if(d<ad.y||(d==ad.y&&ids[at-1u]<id)){break;}
      depths[at]=d;ids[at]=ids[at-1u];at--;}
    ids[at]=id;depths[at]=ad.y;count++;
  }
  var rgb=vec3<f32>(0);var trans=1.0;var previousDepth=-1e30;var previousId=0u;
  // Dense rays use exact depth peeling, not a silently truncated candidate cap.
  let limit=select(count,binCount,overflow);
  for(var rank=0u;rank<limit;rank++){
    var id=0xffffffffu;
    if(!overflow){id=ids[rank];}else{
      var nearest=1e30;
      for(var k=0u;k<binCount;k++){
        let candidate=bins[bin_base(tile)+k];let ad=alpha_depth(traces[candidate],a);
        let after=ad.y>previousDepth||(ad.y==previousDepth&&candidate>previousId);
        if(ad.x>0.0&&after&&(ad.y<nearest||(ad.y==nearest&&candidate<id))){id=candidate;nearest=ad.y;}
      }
      previousDepth=nearest;previousId=id;
    }
    if(id==0xffffffffu){break;}
    let p=traces[id];let alpha=alpha_depth(p,a).x;rgb+=trans*alpha*vec3<f32>(p[16],p[17],p[18]);
    trans*=1.0-alpha;if(trans<1e-4){break;}
  }
  if(cfg.view.w==2.0){rgb=vec3<f32>(1.0-trans);}
  textureStore(output,vec2<i32>(gid.xy),vec4<f32>(rgb,1.0));
}`;
}

export const WORLD_TUBES_PRESENT_WGSL = /* wgsl */ `
@group(0) @binding(0) var tex:texture_2d<f32>;
@group(0) @binding(1) var<uniform> config:array<vec4<u32>,4>;
struct Vertex{@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>}
@vertex fn vs(@builtin(vertex_index) i:u32)->Vertex{
  let pos=array<vec2<f32>,3>(vec2<f32>(-1,-1),vec2<f32>(3,-1),vec2<f32>(-1,3))[i];
  return Vertex(vec4<f32>(pos,0,1),vec2<f32>(pos.x*0.5+0.5,0.5-pos.y*0.5));}
@fragment fn fs(v:Vertex)->@location(0) vec4<f32>{
  let size=textureDimensions(tex);let live=vec2<f32>(f32(config[0].y)/4.0,1.0);
  return textureLoad(tex,min(vec2<u32>(v.uv*live*vec2<f32>(size)),size-1u),0);}
`;
