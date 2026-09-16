#!/usr/bin/env python3
"""Reproducible CPU mathematical checks; not a GPU/dataset benchmark.
Requires Python 3.10+ and NumPy. Run: python checks.py --out results.json
"""
from __future__ import annotations
import argparse, json, platform, sys
from pathlib import Path
import numpy as np


def phi_values(s: float, length: float) -> tuple[float,float]:
    x=s*length
    if abs(x)<1e-4:
        phi=length*(1-x/2+x*x/6-x**3/24+x**4/120)
        ds=length**2*(-.5+x/3-x*x/8+x**3/30-x**4/144)
    else:
        phi=-np.expm1(-x)/s
        ds=((1+x)*np.exp(-x)-1)/(s*s)
    return phi,ds

# Each of two boxes has A0(9), A1(9), b0(3), v(3), log mass(1), RGB(3).
STRIDE=28

def make_parameters() -> np.ndarray:
    boxes=[]
    for i in range(2):
        A=np.array([[.65+.08*i,.04,.02],[.01,.55,.03],[.02,-.015,.7+.07*i]])
        Ad=np.array([[.03,-.01,.006],[.005,-.012,.008],[.0,.008,.015]])*(1 if i==0 else -.7)
        b=np.array([-.1+.23*i,.04-.07*i,.15+.6*i])
        v=np.array([.06-.09*i,.018,.025])
        rgb=np.array([.85,.2,.12]) if i==0 else np.array([.12,.25,.88])
        boxes.extend(np.r_[A.ravel(),Ad.ravel(),b,v,np.log(1.2+.2*i),rgb])
    return np.r_[boxes,[-.13,.075,-3.0],[.033,-.017,1.0]]


def box_render(theta: np.ndarray, t: float, upstream: np.ndarray | None=None):
    """Exact point-ray interval emission-absorption and manual reverse pass.
    Valid derivatives require simple slab endpoints, no ties and det(A)>0.
    Includes camera origin and normalized raw direction derivatives.
    """
    theta=np.asarray(theta,dtype=np.float64)
    grad=np.zeros_like(theta)
    o=theta[-6:-3]; raw=theta[-3:]; norm=np.linalg.norm(raw)
    if norm==0: raise ValueError('zero ray direction')
    d=raw/norm; bg=np.array([.10,.12,.20]); events=[]; boxes=[]
    for i in range(2):
        p=theta[i*STRIDE:(i+1)*STRIDE]
        A=p[:9].reshape(3,3)+t*p[9:18].reshape(3,3)
        b=p[18:21]+t*p[21:24]
        det=np.linalg.det(A)
        if det<=0: raise ValueError('nonpositive determinant')
        inv=np.linalg.inv(A); q=inv@(o-b); v=inv@d
        rho=np.exp(p[24])/(8*det); rgb=p[25:28]
        lows=[]; highs=[]
        for k in range(3):
            if abs(v[k])<1e-14:
                if abs(q[k])>1: lows=[(np.inf,k,0)]; highs=[(-np.inf,k,0)]; break
                lows.append((-np.inf,k,-1)); highs.append((np.inf,k,1))
            else:
                a=((-1-q[k])/v[k],k,-1); z=((1-q[k])/v[k],k,1)
                lows.append(min(a,z,key=lambda z:z[0])); highs.append(max(a,z,key=lambda z:z[0]))
        lo=max(lows,key=lambda z:z[0]); hi=min(highs,key=lambda z:z[0])
        boxes.append((A,inv,q,v,rho,rgb))
        if hi[0]>max(lo[0],0):
            if lo[0]<=0: raise ValueError('test oracle assumes cameras outside every box')
            events.extend([(lo[0],i,+1,lo[1],lo[2]),(hi[0],i,-1,hi[1],hi[2])])
    events.sort(key=lambda x:x[0]); K=len(events)
    if K==0: return bg,grad
    sigma=0.; em=np.zeros(3); intervals=[]; trans=1.
    for j in range(K-1):
        _,i,sign,_,_=events[j]; rho=boxes[i][4]; rgb=boxes[i][5]
        sigma+=sign*rho; em=em+sign*rho*rgb
        length=events[j+1][0]-events[j][0]
        a=np.exp(-sigma*length); phi,dphi=phi_values(sigma,length)
        intervals.append((sigma,em.copy(),length,a,phi,dphi,trans))
        trans*=a
    back=bg.copy(); behind=[None]*(K-1)
    for j in range(K-2,-1,-1):
        sigma,em,length,a,phi,_,_=intervals[j]
        behind[j]=back.copy(); back=em*phi+a*back
    color=back
    if upstream is None: return color,grad
    g=np.asarray(upstream)
    gs=np.zeros(K); gj=np.zeros((K,3)); ge=np.zeros(K)
    for j,(sigma,em,length,a,phi,dphi,pre) in enumerate(intervals):
        gs[j]=pre*np.dot(g,em*dphi-length*a*behind[j])
        gj[j]=pre*phi*g
        gl=pre*np.dot(g,a*(em-sigma*behind[j]))
        ge[j]-=gl; ge[j+1]+=gl
    ds=np.cumsum(gs[::-1])[::-1]; dj=np.cumsum(gj[::-1],axis=0)[::-1]
    grho=np.zeros(2); gc=np.zeros((2,3)); gq=np.zeros((2,3)); gv=np.zeros((2,3))
    for j,(s,i,sign,k,face) in enumerate(events):
        _,_,q,v,rho,rgb=boxes[i]
        grho[i]+=sign*(ds[j]+np.dot(dj[j],rgb)); gc[i]+=sign*rho*dj[j]
        gq[i,k]-=ge[j]/v[k]; gv[i,k]-=ge[j]*s/v[k]
    go=np.zeros(3); gd=np.zeros(3)
    for i,(A,inv,q,v,rho,rgb) in enumerate(boxes):
        oq=inv.T@gq[i]; dv=inv.T@gv[i]
        ga=-np.outer(oq,q)-np.outer(dv,v)-grho[i]*rho*inv.T
        sl=slice(i*STRIDE,(i+1)*STRIDE); pgrad=np.zeros(STRIDE)
        pgrad[:9]=ga.ravel(); pgrad[9:18]=(t*ga).ravel()
        pgrad[18:21]=-oq; pgrad[21:24]=-t*oq
        pgrad[24]=grho[i]*rho; pgrad[25:28]=gc[i]
        grad[sl]=pgrad; go+=oq; gd+=dv
    grad[-6:-3]=go; grad[-3:]=(gd-d*np.dot(d,gd))/norm
    return color,grad


def midpoint_volume(theta,t,n=131072):
    """Independent uniform midpoint integration; no endpoint sorting."""
    o=theta[-6:-3]; d=theta[-3:]/np.linalg.norm(theta[-3:]); ds=6./n
    s=(np.arange(n)+.5)*ds; pts=o+s[:,None]*d
    sigma=np.zeros(n); em=np.zeros((n,3))
    for i in range(2):
        p=theta[i*STRIDE:(i+1)*STRIDE]
        A=p[:9].reshape(3,3)+t*p[9:18].reshape(3,3); b=p[18:21]+t*p[21:24]
        q=(pts-b)@np.linalg.inv(A).T; mask=np.all(np.abs(q)<=1,axis=1)
        rho=np.exp(p[24])/(8*np.linalg.det(A))
        sigma[mask]+=rho; em[mask]+=rho*p[25:28]
    tau=sigma*ds; pre=np.exp(-np.r_[0,np.cumsum(tau[:-1])]); local=np.full(n,ds)
    nz=sigma>0; local[nz]=-np.expm1(-tau[nz])/sigma[nz]
    return (pre[:,None]*em*local[:,None]).sum(0)+np.exp(-tau.sum())*np.array([.1,.12,.2])


def smooth_sphere(x,t,theta):
    """Exact orthographic ray color + six analytic parameter derivatives.
    x,t broadcast. Domain stays strictly inside the sphere's silhouette.
    """
    x,t=np.broadcast_arrays(x,t)
    b=theta[0]+theta[1]*t; r0=np.exp(theta[2]); R=r0*(1+theta[3]*t)
    h=R*R-(x-b)**2-.1**2
    if np.any(h<=0): raise ValueError('smooth test crossed silhouette')
    sq=np.sqrt(h); length=2*sq; sigma=np.exp(theta[4])/(4*np.pi*R**3/3)
    tau=sigma*length; c=1/(1+np.exp(-theta[5])); tr=np.exp(-tau)
    tb=sigma*2*(x-b)/sq; tradius=sigma*(2*R/sq-3*length/R)
    dtau=np.stack([tb,t*tb,tradius*R,tradius*r0*t,tau,np.zeros_like(tau)],axis=-1)
    jac=(c*tr)[...,None]*dtau
    jac[...,5]=c*(1-c)*(1-tr)
    return c*(1-tr),jac


def bilinear(values,x,t,n):
    u=(x+.25)/.5*n; v=t*n
    i=np.minimum(np.floor(u).astype(int),n-1); j=np.minimum(np.floor(v).astype(int),n-1)
    a=u-i; b=v-j
    extra=(1,)*(values.ndim-2)
    a=a.reshape(a.shape+extra); b=b.reshape(b.shape+extra)
    return (1-a)*(1-b)*values[i,j]+a*(1-b)*values[i+1,j]+(1-a)*b*values[i,j+1]+a*b*values[i+1,j+1]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='results.json'); args=parser.parse_args()
    rng=np.random.default_rng(1729); theta=make_parameters(); times=np.array([.03,.37,.81])
    gs=rng.normal(size=(len(times),3)); ana=sum(box_render(theta,t,g)[1] for t,g in zip(times,gs))
    def loss(th): return sum(np.dot(g,box_render(th,t)[0]) for t,g in zip(times,gs))
    fd_results=[]
    for h in [1e-4,1e-5,1e-6]:
        fd=np.zeros_like(theta)
        for k in range(len(theta)):
            e=np.zeros_like(theta); e[k]=h
            fd[k]=(loss(theta+e)-loss(theta-e))/(2*h)
        fd_results.append({'step':h,'max_abs_error':float(np.max(np.abs(fd-ana))),
          'relative_l2_error':float(np.linalg.norm(fd-ana)/np.linalg.norm(fd))})
    rgb,_=box_render(theta,.37); qrgb=midpoint_volume(theta,.37)
    th=np.array([.03,.12,np.log(.90),.12,np.log(1.4),.7])
    P,T=64,256; xx=np.linspace(-.249,.249,P); tt=np.linspace(.001,.999,T)
    x,t=np.meshgrid(xx,tt,indexing='ij'); exact,J=smooth_sphere(x,t,th)
    G=rng.normal(size=(16,P*T)); G/=np.linalg.norm(G,axis=1)[:,None]
    Jm=J.reshape(-1,6); refG=G@Jm
    rows=[]
    for n in [8,16,32,64]:
        gx,gt=np.meshgrid(np.linspace(-.25,.25,n+1),np.linspace(0,1,n+1),indexing='ij')
        V,VJ=smooth_sphere(gx,gt,th); pred=bilinear(V,x,t,n); PJ=bilinear(VJ,x,t,n)
        DJ=PJ.reshape(-1,6)-Jm; appG=G@PJ.reshape(-1,6)
        rows.append({'cells_per_axis':n,'node_ray_evaluations':int((n+1)**2),
          'query_rays':P*T,'max_image_error':float(np.max(np.abs(pred-exact))),
          'max_row_jacobian_l2_error':float(np.max(np.linalg.norm(DJ,axis=1))),
          'jacobian_spectral_relative_error':float(np.linalg.norm(DJ,2)/np.linalg.norm(Jm,2)),
          'max_16_random_vjp_relative_error':float(np.max(np.linalg.norm(appG-refG,axis=1)/np.linalg.norm(refG,axis=1)))})
    # Bilinear pullback is an exact transpose for the interpolant, independent of its accuracy.
    n=16; val=rng.normal(size=(n+1,n+1)); g=rng.normal(size=x.shape); a=(x+.25)/.5*n; b=t*n
    i=np.minimum(np.floor(a).astype(int),n-1); j=np.minimum(np.floor(b).astype(int),n-1); a-=i;b-=j
    adj=np.zeros_like(val)
    for di,dj,w in [(0,0,(1-a)*(1-b)),(1,0,a*(1-b)),(0,1,(1-a)*b),(1,1,a*b)]:
        np.add.at(adj,(i,j) if (di,dj)==(0,0) else (i+di,j+dj),w*g)
    lhs=np.sum(g*bilinear(val,x,t,n)); rhs=np.sum(val*adj)
    # The tangent-blind trap: every node misses a real parameter derivative.
    nodes=np.linspace(0,1,17); query=(np.arange(4096)+.37)/4096
    truth=np.sin(16*np.pi*query); vals=np.sin(16*np.pi*nodes); approx=np.interp(query,nodes,vals)
    trap_error=np.linalg.norm(approx-truth)/np.linalg.norm(truth)
    overlap_tau=np.log(2.); a=np.exp(-overlap_tau); red=np.array([1.,0,0]); blue=np.array([0.,0,1.])
    forward=(1-a)*red+a*(1-a)*blue; reverse=(1-a)*blue+a*(1-a)*red
    results={'scope':'CPU mathematical diagnostics only: no training, no GPU benchmark, no external dataset',
      'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'platform':platform.platform()},
      'box_manual_reverse':{'parameters':len(theta),'rays':len(times),'finite_differences':fd_results,
        'interval_rgb':rgb.tolist(),'independent_midpoint_samples':131072,'midpoint_rgb':qrgb.tolist(),
        'max_midpoint_rgb_discrepancy':float(np.max(np.abs(rgb-qrgb)))},
      'ray_operator_smooth_2D_slice':rows,
      'interpolant_transpose_relative_error':float(abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)),
      'tangent_blind_aliasing':{'image_error_at_theta_zero':0.,'derivative_relative_l2_error':float(trap_error)},
      'order_loss_counterexample':{'red_front':forward.tolist(),'blue_front':reverse.tolist()},
      'scaling_counts_n16':[{'T':T,'P':64,'scene_node_evaluations':289,'ray_queries':64*T,'query_stencil_terms':4*64*T} for T in [16,64,256,1024]]}
    Path(args.out).write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2))
    assert fd_results[-1]['relative_l2_error']<1e-7
    assert results['box_manual_reverse']['max_midpoint_rgb_discrepancy'] < 1e-4
    assert results['interpolant_transpose_relative_error']<1e-11
    assert rows[-1]['max_image_error']<1e-4
    assert trap_error>.99

if __name__=='__main__': main()
