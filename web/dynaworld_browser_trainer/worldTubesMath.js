// Lossless SPD(4) conditional block chart -> affine pinhole UVT marginal.
// The camera is linearized at each atom's mean time. This is not a projective atlas.
// World ABI: x(.5), log sigma_t, half-velocity, mean time, reserved4,
// log spatial scales + pad, quaternion xyzw, RGB + opacity logit (24 floats).
// Trace ABI is STAR's 20 floats: ma4, packed Q4+4, conditional depth4, RGBA4.
export const WORLD_TUBES_PARAMETER_SCHEMA = "world-tube-affine-star-24f/v2";
export const WORLD_TUBES_MODEL_MODE = 2;
export const WORLD_COMPONENTS = Object.freeze([0,1,2,3,4,5,6,7,12,13,14,16,17,18,19,20,21,22,23]);

// A small scalar expression tape emits an analytic reverse-mode WGSL compiler.
// It runs once per atom/camera, not once per pixel or optimizer parameter.
function compilerGraph() {
	const nodes = [];
	const input = (name) => { const id = nodes.length; nodes.push({ name }); return id; };
	const op = (kind, ...args) => { const id = nodes.length; nodes.push({ kind, args }); return id; };
	const p = Array.from({length:24}, (_, i) => input(`p[${i}]`));
	const c = Array.from({length:20}, (_, i) => input(`c[${i}]`));
	const aspect = input("aspect"), filter = input("filterVariance");
	const zero = input("0.0"), one = input("1.0"), two = input("2.0");
	const add = (a,b) => op("add",a,b), mul = (a,b) => op("mul",a,b);
	const neg = a => op("neg",a), sub = (a,b) => add(a,neg(b));
	const div = (a,b) => op("div",a,b), exp = a => op("exp",a);
	const sum = a => a.reduce(add,zero), dot = (a,b) => sum(a.map((v,i) => mul(v,b[i])));
	const mu = [0,1,2].map(i => add(p[i],mul(p[4+i],sub(mul(two,p[7]),one))));
	const cp = [0,1,2].map(i => add(dot(c.slice(i*4,i*4+3),mu),c[i*4+3]));
	const qnorm = op("sqrt",add(sum(p.slice(16,20).map(v=>mul(v,v))),input("1e-16")));
	const [x,y,z,w] = p.slice(16,20).map(v=>div(v,qnorm));
	const twice = a => mul(two,a);
	const rotation = [
		[sub(one,twice(add(mul(y,y),mul(z,z)))),twice(sub(mul(x,y),mul(z,w))),twice(add(mul(x,z),mul(y,w)))],
		[twice(add(mul(x,y),mul(z,w))),sub(one,twice(add(mul(x,x),mul(z,z)))),twice(sub(mul(y,z),mul(x,w)))],
		[twice(sub(mul(x,z),mul(y,w))),twice(add(mul(y,z),mul(x,w))),sub(one,twice(add(mul(x,x),mul(y,y))))],
	];
	const basis = [0,1,2].map(r => [0,1,2].map(k => dot(c.slice(r*4,r*4+3),rotation.map(row=>row[k]))));
	const variance = p.slice(12,15).map(v=>exp(twice(v)));
	const covariance = [0,1,2].map(r => [0,1,2].map(s => sum(variance.map((v,k)=>mul(mul(basis[r][k],v),basis[s][k])))));
	const fx = mul(aspect,c[16]), fy = c[17], iz = div(one,cp[2]);
	const j = [[mul(fx,iz),zero,neg(mul(mul(fx,cp[0]),mul(iz,iz)))],
		[zero,mul(fy,iz),neg(mul(mul(fy,cp[1]),mul(iz,iz)))]];
	const Cj = j.map(row=>covariance.map(r=>dot(r,row)));
	const s00=add(dot(j[0],Cj[0]),filter), s01=dot(j[0],Cj[1]), s11=add(dot(j[1],Cj[1]),filter);
	const det=sub(mul(s00,s11),mul(s01,s01));
	const a=div(s11,det), b=neg(div(s01,det)), d=div(s00,det);
	const velocity=[0,1,2].map(r=>twice(dot(c.slice(r*4,r*4+3),p.slice(4,7))));
	const uvVelocity=j.map(row=>dot(row,velocity));
	const qb=[add(mul(a,uvVelocity[0]),mul(b,uvVelocity[1])),add(mul(b,uvVelocity[0]),mul(d,uvVelocity[1]))];
	const invTimeVar=exp(neg(twice(p[3])));
	const beta=[add(mul(Cj[0][2],a),mul(Cj[1][2],b)),add(mul(Cj[0][2],b),mul(Cj[1][2],d))];
	const outputs=[add(mul(fx,mul(cp[0],iz)),mul(aspect,c[18])),add(mul(fy,mul(cp[1],iz)),c[19]),p[7],one,
		a,b,neg(qb[0]),d,neg(qb[1]),add(invTimeVar,dot(uvVelocity,qb)),
		sub(covariance[2][2],dot(beta,[Cj[0][2],Cj[1][2]])),zero,
		cp[2],...beta,sub(velocity[2],dot(beta,uvVelocity)),
		p[20],p[21],p[22],div(one,add(one,exp(neg(p[23]))))];
	return { nodes, outputs, depth:cp[2] };
}
const GRAPH = compilerGraph();

export function compileWorldTubeCpu(p, camera, aspect, filterVariance) {
	const c = [...camera.worldToCamera,...camera.intrinsics];
	const values = new Float64Array(GRAPH.nodes.length);
	for (let i=0;i<GRAPH.nodes.length;i++) {
		const n=GRAPH.nodes[i];
		if(n.name) {
			values[i]=i<24?p[i]:i<44?c[i-24]:n.name==="aspect"?aspect:n.name==="filterVariance"?filterVariance:Number(n.name);
		} else {
			const [a,b]=n.args.map(id=>values[id]);
			values[i]=n.kind==="add"?a+b:n.kind==="mul"?a*b:n.kind==="div"?a/b:n.kind==="neg"?-a:n.kind==="exp"?Math.exp(a):Math.sqrt(a);
		}
	}
	if(values[GRAPH.depth]<=0.1) return new Float64Array(20);
	return Float64Array.from(GRAPH.outputs,id=>values[id]);
}

export function worldTubeCompilerWgsl() {
	const ref=id=>`v${id}`;
	const forward=GRAPH.nodes.map((n,i)=> {
		const [a,b]=(n.args??[]).map(ref);
		const expr=n.name??(n.kind==="add"?`${a}+${b}`:n.kind==="mul"?`${a}*${b}`:n.kind==="div"?`${a}/${b}`:n.kind==="neg"?`-${a}`:`${n.kind}(${a})`);
		return `let v${i}:f32=${expr};`;
	}).join("\n");
	const live=new Set();
	function visit(i) { if(live.has(i))return; live.add(i); for(const a of GRAPH.nodes[i].args??[])visit(a); }
	// Conditional depth/order and support are detached, as in the STAR oracle.
	const components=[0,1,2,4,5,6,7,8,9,16,17,18,19];
	for(const k of components)visit(GRAPH.outputs[k]);
	const reverse=[];
	for(const i of live)reverse.push(`var b${i}:f32=0.0;`);
	for(const k of components)reverse.push(`b${GRAPH.outputs[k]}+=g[${k}];`);
	for(let i=GRAPH.nodes.length-1;i>=0;i--) {
		if(!live.has(i)||!GRAPH.nodes[i].args)continue;
		const {kind,args:[a,b]}=GRAPH.nodes[i];
		if(kind==="add")reverse.push(`b${a}+=b${i};b${b}+=b${i};`);
		if(kind==="mul")reverse.push(`b${a}+=b${i}*v${b};b${b}+=b${i}*v${a};`);
		if(kind==="div")reverse.push(`b${a}+=b${i}/v${b};b${b}-=b${i}*v${i}/v${b};`);
		if(kind==="neg")reverse.push(`b${a}-=b${i};`);
		if(kind==="exp")reverse.push(`b${a}+=b${i}*v${i};`);
		if(kind==="sqrt")reverse.push(`b${a}+=b${i}*0.5/v${i};`);
	}
	const args="p:array<f32,24>,c:array<f32,20>,aspect:f32,filterVariance:f32";
	return `fn compile_tube(${args})->array<f32,20>{${forward}
	if(v${GRAPH.depth}<=0.1){return array<f32,20>();}
	return array<f32,20>(${GRAPH.outputs.map(ref).join(",")});}
	fn compiler_vjp(${args},g:array<f32,20>)->array<f32,24>{${forward}
	if(v${GRAPH.depth}<=0.1){return array<f32,24>();}
	${reverse.join("\n")}
	return array<f32,24>(${Array.from({length:24},(_,i)=>live.has(i)?`b${i}`:"0.0").join(",")});}`;
}

export function sliceWorldTube(trace,time) {
	const [a,b,,d]=trace.slice(4,8), det=a*d-b*b;
	if(!trace[3]||!(det>0))return {valid:false};
	const covariance=[d/det,-b/det,a/det];
	const velocity=[-(covariance[0]*trace[6]+covariance[1]*trace[8]),-(covariance[1]*trace[6]+covariance[2]*trace[8])];
	const dt=time-trace[2];
	const qt=trace[9]+trace[6]*velocity[0]+trace[8]*velocity[1];
	return {valid:true,center:[trace[0]+velocity[0]*dt,trace[1]+velocity[1]*dt],
		covariance,conic:[a,b,d],peakAlpha:trace[19]*Math.exp(-0.5*qt*dt*dt),
		depthAt:(u,v)=>trace[12]+trace[13]*(u-trace[0])+trace[14]*(v-trace[1])+trace[15]*dt};
}
