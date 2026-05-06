Yes. I’d split this into **three levels**:

1. **MVP dynamic foam**: spline the foam parameters over time.
2. **Better dynamic foam**: canonical foam + learned deformation field.
3. **Best / paper-ish version**: dynamic **gauge foam**, where each cell carries a local material frame and features live in that frame.

My strong take: **the simplest useful version is “Dynamic PowerFoam = PowerFoam + persistent site trajectories + temporal regularization.”** The better version is **SE(3)/Sim(3) gauge trajectories over the foam graph**, because then geometry, normals, texture, and features move as coherent local material patches instead of swimming around like cursed soup.

---

## 1. What does “dynamic foam” need to parameterize?

For static Radiant Foam, the core state is basically:

[
\theta_i = {p_i,\sigma_i,c_i}
]

where (p_i) is a Voronoi site, (\sigma_i) is density/opacity-ish state, and (c_i) is view-dependent color. The important rendering trick is that Voronoi adjacency gives fast ray traversal by stepping cell-to-cell through neighboring faces. Radiant Foam’s page describes using a Delaunay-derived adjacency structure and ray-cell intersections by iterating over neighbors. ([Radiant Foam][1])

Power Foam is the better base for dynamics, though, because it moves from unbounded Voronoi cells to **bounded power diagrams** with controllable extents, adds oriented surfaces, and decouples geometry from appearance via differentiable texture on surfaces. ([Power Foam][2]) It also uses a Čech-style overlap graph as a cheap superset of the α-complex, which is very relevant for dynamic scenes because moving cells need robust neighbor candidates. ([Power Foam][2])

So for dynamic foam, each primitive should become:

[
\theta_i(t)=
{p_i(t), r_i(t), R_i(t), \alpha_i(t), f_i(t), \text{texture}_i(t)}
]

where:

* (p_i(t)): time-varying site position.
* (r_i(t)): time-varying power radius / support / lifetime.
* (R_i(t)): orientation for an oriented surface, dipole, or local chart.
* (\alpha_i(t)): opacity/density.
* (f_i(t)): feature or appearance code.
* (\text{texture}_i(t)): optional local surface texture/displacement.

That alone gets you a **4D foam** in the practical sense: a 3D foam whose parameters are continuous functions of time.

---

## 2. The absolute simplest version

The simplest thing I’d actually implement is:

> **Persistent sites with B-spline trajectories.**

You initialize a static PowerFoam/RadiantFoam, then give every cell a small time trajectory.

For normalized time (t), use basis functions (B_m(t)), maybe cubic B-splines:

[
p_i(t)=p_i^0+\sum_m B_m(t)\Delta p_{i,m}
]

[
r_i(t)=\text{softplus}\left(r_i^0+\sum_m B_m(t)\Delta r_{i,m}\right)
]

[
\alpha_i(t)=\sigma\left(\alpha_i^0+\sum_m B_m(t)\Delta \alpha_{i,m}\right)
]

[
f_i(t)=f_i^0+\sum_m B_m(t)\Delta f_{i,m}
]

If you use PowerFoam-style oriented surface/dipoles, add rotations:

[
R_i(t)=\exp\left(\sum_m B_m(t)\omega_{i,m}\right)R_i^0
]

Then render frame (t) by evaluating all current parameters and ray tracing/rasterizing the foam at that time.

This is directly analogous to the “persistent primitive” idea from **Dynamic 3D Gaussians**, where Gaussians move and rotate over time while maintaining persistent color, opacity, and size, with local-rigidity regularization. ([arXiv][3]) Foam can copy that recipe almost embarrassingly directly.

### Training losses

The minimum loss is:

[
\mathcal{L}
===========

\mathcal{L}*{rgb}
+
\lambda*{temp}\mathcal{L}*{temp}
+
\lambda*{arap}\mathcal{L}*{arap}
+
\lambda*{sparse}\mathcal{L}_{sparse}
]

Where:

[
\mathcal{L}_{temp}
==================

\sum_i
\left|
p_i(t+\Delta t)-2p_i(t)+p_i(t-\Delta t)
\right|^2
]

for acceleration smoothness, and:

[
\mathcal{L}_{arap}
==================

\sum_{(i,j)\in E}
\left(
|p_i(t)-p_j(t)|-|p_i^0-p_j^0|
\right)^2
]

for local rigidity.

That is the “first build.” Not the sexiest, but it would answer the question: **can foam be fit over videos?**

---

## 3. The first non-toy version: canonical foam + deformation field

The next version should not store independent spline coefficients per primitive. Instead:

> Learn a canonical foam, then deform it to each time.

[
\theta_i^0 = {p_i^0,r_i^0,R_i^0,\alpha_i^0,f_i^0}
]

[
D_\psi(p_i^0,t,z_i)
\rightarrow
(\Delta p_i(t), \Delta r_i(t), \Delta R_i(t), \Delta f_i(t))
]

Then:

[
p_i(t)=p_i^0+\Delta p_i(t)
]

[
R_i(t)=\Delta R_i(t)R_i^0
]

[
r_i(t)=r_i^0+\Delta r_i(t)
]

[
f_i(t)=f_i^0+\Delta f_i(t)
]

This is basically **Deformable 3DGS, but foam-shaped**. Deformable 3D Gaussians learns Gaussians in canonical space and uses a deformation field to model monocular dynamic scenes. ([arXiv][4]) 4D Gaussian Splatting similarly uses 3D Gaussians plus 4D neural voxels / HexPlane-style encodings and a lightweight MLP to predict Gaussian deformations at novel timestamps. ([arXiv][5])

For foam, the deformation network could be:

[
D_\psi(x,t) = \text{MLP}(\gamma(x), \gamma(t))
]

or better:

[
D_\psi(x,t) = \text{MLP}(\text{HexPlane}(x,t))
]

where HexPlane / multi-plane features keep the model compact and fast.

This version is much better for interpolation, lower storage, and regular motion. The major implementation question becomes: **how do we handle the adjacency graph as cells move?**

---

## 4. The graph problem is the annoying bit

In static Radiant Foam, traversal depends on the Voronoi/Delaunay adjacency. In dynamic foam, as (p_i(t)) changes, the true adjacency can change.

You have three options.

### Option A: recompute the graph per frame

For each training/rendering timestamp, rebuild Delaunay / regular triangulation / Čech candidates.

This is clean and probably okay for an offline research prototype. But it is expensive and may be annoying if you want real-time interactive rendering.

### Option B: fixed conservative candidate graph

Build a large neighbor graph once:

[
E = \text{kNN}(p^0) \cup \text{Čech}(r_{\max})
]

Then during traversal, when in cell (i), test all candidate neighbors (j \in E_i), compute valid face intersections, and take the nearest valid crossing.

This is not as elegant, but it is the MVP path. PowerFoam already motivates using a Čech graph as a superset of the α-complex, which is exactly the kind of conservative neighbor graph you’d want for moving bounded cells. ([Power Foam][2])

### Option C: dynamic local graph via spatial hash

At each timestamp, build a local overlap graph using a GPU spatial hash over bounded supports. This is probably the real implementation path if you want speed.

For dynamic scenes, I would strongly favor **bounded PowerFoam** over unbounded Radiant Foam, because bounded supports make dynamic neighbor search way less gross.

---

## 5. Births, deaths, occlusions, and topology changes

Dynamic scenes need primitives to appear and disappear.

Simplest trick:

[
a_i(t)=\sigma(g_i(t))
]

where (a_i(t)) is an activity/lifetime gate. Then opacity is:

[
\tilde{\alpha}_i(t)=a_i(t)\alpha_i(t)
]

You can parameterize (a_i(t)) with a spline, an MLP, or a temporal Gaussian:

[
a_i(t)=\exp\left(-\frac{(t-\tau_i)^2}{2s_i^2}\right)
]

That gives each cell a **temporal support window**. Very useful for objects entering/exiting, disocclusions, and topology changes.

For a slightly more geometric version, use a **space-time power weight**:

[
w_i(t)=w_i^0-\lambda(t-\tau_i)^2
]

So a cell is naturally more active near its temporal center (\tau_i). This is a neat bridge toward “real” 4D foam without fully committing to 4D Delaunay hell.

---

## 6. A true 4D foam formulation

You *could* define sites in spacetime:

[
q_i=(p_i,\tau_i)\in\mathbb{R}^4
]

and a 4D power distance:

[
D_i(x,t)
========

|x-p_i|^2
+
\lambda(t-\tau_i)^2
-------------------

w_i
]

At a fixed timestamp (t), this becomes a 3D power diagram with an effective time-dependent weight:

[
D_i(x,t)
========

## |x-p_i|^2

\left[w_i-\lambda(t-\tau_i)^2\right]
]

So slicing a 4D power diagram gives you a time-varying 3D foam.

Cute. Maybe even paper-cute.

But by itself, static 4D point sites mostly give you changing cell weights and moving boundaries, not rich object motion. For actual video dynamics, I’d combine this with worldline sites:

[
p_i(t)=p_i^0 + v_i t
]

or spline/deformation-field trajectories:

[
p_i(t)=p_i^0 + D_\psi(p_i^0,t)
]

So the practical version is:

> **canonical/deformed 3D foam + temporal activity gates**, not pure 4D Voronoi.

Pure 4D foam is elegant, but it’s not the first build. It’s the “maybe SIGGRAPH diagram looks sick” build.

---

## 7. The better version using gauges

Yes — and I think gauges are more compelling for **dynamic** foam than for static foam.

The gauge version is:

> Each cell has a local material coordinate frame that evolves over time.

Let each foam cell carry a gauge:

[
G_i(t)\in SE(3)
]

or, better, (Sim(3)) if you want local scale:

[
G_i(t)\in Sim(3)
]

Then:

[
G_i(t)=\exp(\xi_i(t))G_i^0
]

where (\xi_i(t)) is a Lie algebra trajectory predicted by splines or a deformation network.

The site position comes from the gauge:

[
p_i(t)=G_i(t)p_i^{local}
]

The local chart coordinates are:

[
u_i(x,t)
========

G_i(t)^{-1}x
]

Then geometry, displacement, features, and texture are defined in **local material coordinates**:

[
F_i(u,v)
]

[
d_i(u,v)
]

[
C_i = \text{MLP}(F_i(u,v), n_i(t), \omega, t)
]

This matters because now texture/features do not swim in world space. They are attached to the moving local patch.

That is the real win.

---

## 8. Minimal gauge dynamic foam

The simplest gauge version is:

[
G_i(t)=\exp\left(\sum_m B_m(t)\xi_{i,m}\right)G_i^0
]

where (\xi_{i,m}\in \mathfrak{se}(3)) are learned twist coefficients.

Then:

[
p_i(t)=G_i(t)[0,0,0,1]^T
]

[
R_i(t)=\text{Rot}(G_i(t))
]

[
f_i(t)=f_i^0
]

or:

[
f_i(t)=f_i^0+\Delta f_i(t)
]

but I’d start with mostly persistent features.

Add local connection regularization over the foam graph. For neighboring cells (i,j), define:

[
H_{ij}(t)=G_i(t)^{-1}G_j(t)
]

and canonical relative transform:

[
H_{ij}^0=G_i(0)^{-1}G_j(0)
]

Then penalize:

[
\mathcal{L}_{conn}
==================

\sum_{(i,j)\in E}
\rho
\left(
\left|
\log\left((H_{ij}^0)^{-1}H_{ij}(t)\right)
\right|
\right)
]

Use a robust penalty (\rho), not pure L2, so articulated boundaries can move.

This gives you a foam version of local rigidity, but in **SE(3)** rather than just Euclidean distances. It preserves local orientation and relative frames, not only point distances. That’s much more geometrically meaningful.

---

## 9. Why gauges are probably better than plain deformations

A plain dynamic foam says:

[
p_i(t)=p_i^0+\Delta p_i(t)
]

That moves sites, but it does not tell you how the local surface patch rotated, how its texture advected, or how neighboring patches should agree.

A gauge dynamic foam says:

[
\text{cell } i \text{ has a moving local coordinate system}
]

so:

* texture lives in material coordinates;
* normals rotate consistently;
* features are stable over time;
* local shape/detail moves with the cell;
* neighboring cells can be regularized by relative transforms;
* temporal interpolation becomes less floaty.

This is especially important for dynamic NVS because the failure mode is not just bad photometric reconstruction. It’s **feature/texture swimming**, depth drift, and local geometry explaining away motion. Tiny ghost goblin artifacts everywhere.

---

## 10. Do not start with anisotropic “gauge distance”

There is one tempting but dangerous idea:

[
D_i(x,t)
========

(x-p_i(t))^\top G_i(t)(x-p_i(t)) - w_i(t)
]

where each cell has its own metric tensor (G_i(t)).

That sounds like a gauge-aware foam. But if every cell has a different metric, boundaries between cells are no longer planes; they become quadrics. Then you lose a big reason foam is fast and clean: simple planar face intersections.

So my recommendation is:

> **Do not put the gauge inside the cell-assignment metric first.**

Keep PowerFoam’s power diagram / bounded cells for traversal. Use gauges for:

* local surface charts,
* feature coordinates,
* motion parameterization,
* normal/frame evolution,
* neighbor consistency.

That gives you most of the benefit without nuking the traversal algorithm.

Later, maybe try anisotropic/Riemannian foam. But that is not the MVP.

---

## 11. The clean research proposal

I’d pitch the system as:

### **Dynamic Gauge Foam**

A dynamic scene representation using bounded power-diagram foam cells with time-varying local gauge frames.

Each primitive has:

[
\theta_i =
{
p_i^0,
r_i^0,
G_i(t),
a_i(t),
F_i(u,v),
d_i(u,v)
}
]

Rendering at time (t):

1. Evaluate each cell’s gauge (G_i(t)), position (p_i(t)), radius (r_i(t)), and activity (a_i(t)).
2. Build/use a conservative dynamic neighbor graph.
3. Ray trace or rasterize the bounded foam.
4. At intersections, map world point (x) into local gauge coordinates:

[
u_i = G_i(t)^{-1}x
]

5. Read charted feature/displacement.
6. Decode color with:

[
c = \text{MLP}(F_i(u_i), n_i(t), \omega, t)
]

Losses:

[
\mathcal{L}
===========

\mathcal{L}*{rgb}
+
\lambda*{conn}\mathcal{L}*{conn}
+
\lambda*{temp}\mathcal{L}*{temp}
+
\lambda*{flow}\mathcal{L}*{flow}
+
\lambda*{sparse}\mathcal{L}*{sparse}
+
\lambda*{depth}\mathcal{L}_{depth}
]

Optional but good:

[
\mathcal{L}_{chart}
===================

\sum_{(i,j)}
|
F_i(u_i)-F_j(u_j)
|
]

for overlapping/neighboring chart consistency.

---

## 12. What I would actually build first

The build order should be brutally pragmatic:

### Version 0: Per-frame foam

Train independent foams per keyframe. Terrible storage, but gives a quality oracle.

### Version 1: Spline Dynamic PowerFoam

Same cells, time-varying:

[
p_i(t), r_i(t), \alpha_i(t), f_i(t)
]

with temporal smoothness and local rigidity.

This is the minimum viable dynamic foam.

### Version 2: Canonical Deformation Foam

Replace per-site splines with:

[
D_\psi(p_i^0,t)
]

using MLP or HexPlane features.

This is the fair comparison to dynamic Gaussian methods.

### Version 3: SE(3) Gauge Foam

Replace additive offsets with local gauge trajectories:

[
G_i(t)\in SE(3)
]

Attach features/textures to local coordinates. Add connection regularization.

This is the version I’d be excited about.

### Version 4: Dynamic Gauge Atlas Foam

Learn chart transition maps between neighboring cells:

[
T_{ij}(t): u_i \rightarrow u_j
]

and enforce chart/feature/surface agreement. This becomes a real dynamic manifold atlas over a foam graph.

That is the paper version.

---

## Bottom line

The simplest dynamic foam is:

[
\boxed{
\text{PowerFoam} + \text{persistent sites} + \text{B-spline trajectories} + \text{temporal/ARAP losses}
}
]

The better version is:

[
\boxed{
\text{canonical PowerFoam} + D_\psi(x,t) \text{ deformation field}
}
]

The version I’d bet has the most interesting geometry is:

[
\boxed{
\text{Dynamic Gauge Foam: bounded foam traversal + local } SE(3)/Sim(3) \text{ material gauges}
}
]

The key design choice: **keep the foam/power diagram for traversal, but use gauges for motion, local charts, texture/features, and geometry consistency.** That preserves the speed story while attacking the actual dynamic-NVS failure mode: incoherent geometry and swimming appearance.

[1]: https://radfoam.github.io/ "Radiant Foam"
[2]: https://powerfoam.github.io/ "Power Foam"
[3]: https://arxiv.org/abs/2308.09713 "[2308.09713] Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis"
[4]: https://arxiv.org/abs/2309.13101 "[2309.13101] Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction"
[5]: https://arxiv.org/abs/2310.08528 "[2310.08528] 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"

