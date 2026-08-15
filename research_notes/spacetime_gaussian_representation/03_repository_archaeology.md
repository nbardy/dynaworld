# Repository Archaeology: What We Had, What Narrowed, What Survived

## Scope and method

At the start of this audit the repository contained 1,559 Markdown files. The
audit:

1. recursively searched the complete Markdown inventory for representation
   terms (`4D Gaussian`, `spacetime`, `world tube`, `UVT`, `q_uvt`, covariance,
   precision, quaternion, rotation, deformation, transported metric, and
   projective trace variants);
2. fully read the relevant architecture, handoff, experiment, loose-note, and
   renderer-contract hits;
3. traced the corresponding code fields and projection paths;
4. searched Git and available submodule history for removed or superseded
   formulations;
5. compared the recovered objects by mathematical semantics rather than name.

Unrelated dataset, UI, product, and infrastructure Markdown was indexed but not
linearly reread. No relevant deleted Markdown formulation was found in the
available Git history. Some third-party files entered the visible history in a
later submodule consolidation, so their exact original creation dates cannot
be established from current refs.

## Executive reconstruction

Two objects were conflated in recent discussion:

1. **Canonical world object:** a full Gaussian field on \((x,y,z,t)\), stored
   as one \(\mu_4\) and one SPD \(4\times4\) covariance or precision.
2. **Current paper scaffold:** `x0[3]`, `velocity[3]`, `t0`,
   `precision_xy[2]`, `lambda_t`, opacity, and RGB.

The second is not the final mathematical reduction of the first. It is a
fronto-parallel implementation slice chosen to reach the existing UVT renderer
contract. The repository says explicitly that it does not yet cover full
anisotropic 3D covariance integration.

The elegant formulation was therefore **not lost**. It survived in the
spacetime rasterizer handoff, the later STAR derivation, and the gauged
ray-fiber theory. What narrowed was the active world-to-UVT implementation.

## Chronology

### 1. April 18–20: time-conditioned ordinary 3DGS

The earliest “4D Gaussian” design was actually a family of ordinary 3D
Gaussians whose parameters varied with time. It proposed polynomial
coefficients for position, quaternion rotation, and opacity while keeping
simple/fixed appearance:

- [`KEY_ARCHITECTURE_DECISIONS.md`](../KEY_ARCHITECTURE_DECISIONS.md)
- [`chats/chatgpt_apr_20th.md`](../chats/chatgpt_apr_20th.md)
- [`SESSION_Q_AND_A_SYNTHESIS.md`](../SESSION_Q_AND_A_SYNTHESIS.md)

That line also preserved several alternatives:

- anchor splats plus low-rank temporal bases;
- grouped \(SE(3)\) transforms with residual deformation;
- a fixed recurrent pool with velocity/acceleration updates;
- a global parallel pool with temporal centers and widths.

The recorded OOM objection concerned retaining a global video-backbone graph,
not an impossibility of polynomial or rotating splats. Treating that experiment
as a rejection of time-varying rotation would be a historical error.

### 2. April 25–28: material coordinates and transported covariance

The next geometric family used persistent material coordinates:

\[
x_i(t)=\Phi_t(q_i),
\qquad
\Sigma_i(t)=D\Phi_t(q_i)G_iD\Phi_t(q_i)^\top+\varepsilon I.
\]

This couples center and covariance evolution through one transport map. It can
rotate and deform spatial support over time, unlike one joint Gaussian.

Sources:

- [`incidence_kernels_and_material_objects.md`](../incidence_kernels_and_material_objects.md)
- [`research_experiments/gauge_fields/README.md`](../../research_experiments/gauge_fields/README.md)
- [`2026-04-27_19-20-00_gauge_world_ball_same_count_matrix.md`](../../agent_notes/loose_notes/2026-04-27_19-20-00_gauge_world_ball_same_count_matrix.md)

Tested variants included screen disks, oriented slabs, transported isotropic
world balls, full-rank transported metrics, and exact ray–Gaussian line
integration. The pure-Torch variants were generally slower and did not beat
the per-frame upper baseline on heldout quality, but that does not invalidate
the object class.

### 3. May lineage: native full spacetime Gaussian

The strongest recovered “one position plus one covariance” formulation is
[`spacetime_v0/docs/handoff.md`](../../third_party/fast-mac-gsplat/variants/spacetime_v0/docs/handoff.md).
It defines

\[
G_i(z)\propto
\exp\!\left[-\frac12(z-\mu_i)^\top Q_i(z-\mu_i)\right],
\qquad z=(x,y,z,t),
\]

and stores:

```text
mu:      float4
Q:       float4x4 full precision
logNorm: amplitude / normalization
color:   float3
```

It then lifts camera rays into spacetime and integrates ray depth:

\[
\rho_i(u,v,\tau)=
\int \alpha_iG_i(b+A(u,v,\tau)+sd)\,ds.
\]

The projected sensor-time precision follows from the depth-orthogonalized
precision

\[
Q_\perp=Q-Qd(d^\top Qd)^{-1}d^\top Q,
\qquad
H=A^\top Q_\perp A.
\]

Its acceptance suite explicitly requires a constant-velocity 3D Gaussian to
equal one tilted 4D Gaussian. This is the direct ancestor of the clean object
recovered in the current audit.

### 4. May 10–12: STAR UVT contract, then restricted world scaffold

STAR's renderer first accepted an already-compiled screen-time Gaussian:

```text
ma          [N,3]    UVT mean
q_uvt       [N,6]    packed symmetric 3x3 UVT precision
depth0      [N]
depth_beta  [N,3]
opacity     [N]
color       [N,3]
```

See [`star_uvt_v0/README.md`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/README.md).

World-space Phase 2 then introduced

\[
x(t)=x_0+v(t-t_0),
\qquad
\alpha_t=e^{-\lambda_t(t-t_0)^2/2},
\]

with the exact precision blocks

\[
\Lambda_{xx}=A,
\qquad
\Lambda_{xt}=-Av,
\qquad
\Lambda_{tt}=v^\top Av+\lambda_t.
\]

This formula is general if \(A\in\operatorname{SPD}(3)\). The implementation,
however, stored only `precision_xy[2]`. The source document labels the
fronto-parallel projection an intentionally weaker scaffold and states that it
does not cover full anisotropic 3D covariance:

- [`phase_2_world_tube_projection.md`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/phases/phase_2_world_tube_projection.md)
- [`trainer_harness/world_tube.py`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py)

This is the exact narrowing point:

- six spatial covariance DOF became two;
- world \(z\)-extent disappeared;
- spatial orientation disappeared;
- depth became affine metadata rather than Gaussian support;
- conditional covariance was fixed/restricted;
- the world center remained a linear path.

### 5. Moving cameras: affine, segmented, and projective traces

The repository next explored moving-camera compilation:

- first-order affine camera-time UVT;
- piecewise camera-time segments;
- homogeneous/projective rational traces.

See:

- [`2026-05-12_variable_camera_attempts.md`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/attempts/2026-05-12_variable_camera_attempts.md)
- [`curve_tube_design.md`](../../third_party/fast-mac-gsplat/variants/star_prt_v0/docs/curve_tube_design.md)
- [`projective_rational.py`](../../third_party/fast-mac-gsplat/variants/star_uvt_prt_v0/research_project/trainer_harness/projective_rational.py)

PRT stores homogeneous image curves \(h(t)=(uw,vw,w)\) and recovers
\((u,v)=(h_x/h_z,h_y/h_z)\). This curvature is caused by camera/projective
geometry. It is **compiled screen-trace curvature**, not evidence that the
world primitive itself follows a curved trajectory.

### 6. May 23: the full object is restated explicitly

[`research_experiments/star_uvt_notes.md`](../../research_experiments/star_uvt_notes.md)
states that current world tubes are structured moving 4D primitives, not
generic 4DGS. It then reintroduces

\[
\rho(z)=\alpha e^{-\frac12(z-\mu)^\top Q(z-\mu)},
\qquad z\in\mathbb R^4,
\]

and later gives the gauge-charted primitive

\[
\mathcal G_i=(U_i,\chi_i,\eta_i,\Lambda_i,\alpha_i,c_i),
\quad
\eta_i\in\mathbb R^4,
\quad
\Lambda_i\in\mathbb S_{++}^4.
\]

The note explicitly treats constant-velocity tubes as a special chart. This is
the most important “lost-looking” formulation: it was deferred behind the
simpler PRT build, not rejected.

### 7. May 24 onward: the invariant becomes a ray-bundle compiler

The durable formulation became

\[
\bar\rho_i=\pi_*\Gamma^*\rho_i.
\]

The world primitive is pulled back to the camera-ray bundle and pushed forward
along ray depth into a reusable sensor-time trace. Relevant notes:

- [`00_bundle_foundations/README.md`](../gauged_uvt_trace_atlas/00_bundle_foundations/README.md)
- [`02_gaussian_fiber_pushforward/README.md`](../gauged_uvt_trace_atlas/02_gaussian_fiber_pushforward/README.md)
- [`DEPTH_FIBER_CROSS_TRACK_NOTE.md`](../gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md)
- [`paper/WORLD_TUBES_PAPER_DRAFT.md`](../gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md)

The local Gaussian bridge is the Schur complement

\[
S=H_{yy}-H_{yz}H_{zz}^{-1}H_{zy}.
\]

The notes map `ma` to the local UVT mean, `q_uvt` to the six coefficients of
\(S\), and `depth0/depth_beta` to a conditional-depth approximation. They also
name what the compact ABI still lacks: depth variance, gauge identity,
projective denominator coefficients, and fit/validity certificates.

This is more mature than “convert a 4D Gaussian to one 3D Gaussian per frame.”
It compiles a camera program once into local sensor-time records, then evaluates
the relevant temporal queries.

### 8. June–July: implementation boundary is frozen, not theory reversed

The completion handoff warns against reducing the theory to “a 4D Gaussian
becomes a 3D Gaussian in UVT” as the primary claim. Arbitrary world primitives
may induce Gaussian, polynomial, rational, or interval trace records with
support/order/fallback certificates.

The July renderer taxonomy separates:

- World Tubes paper method;
- STAR UVT implementation;
- PRT projective extension;
- Dynamic 3DGS baseline;
- WorldFoam's retained-depth/transmittance lane.

See [`renderer_lane_taxonomy.md`](../renderer_lane_taxonomy.md).

### 9. July 23: full 3D covariance returns in the browser prototype

The new browser path restores standard anisotropic 3D covariance:

\[
\Sigma_w=R(q)
\operatorname{diag}(e^{2\ell_x},e^{2\ell_y},e^{2\ell_z})
R(q)^\top.
\]

Sources:

- [`2026-07-23_00-24-30_browser_anisotropic_3d_gaussian_design.md`](../../agent_notes/loose_notes/2026-07-23_00-24-30_browser_anisotropic_3d_gaussian_design.md)
- [`2026-07-23_00-56-06_browser_anisotropic_capacity_live_ablation.md`](../../agent_notes/loose_notes/2026-07-23_00-56-06_browser_anisotropic_capacity_live_ablation.md)

Its center is linear plus optional harmonic motion, while quaternion and scales
remain time-shared. It is useful implementation evidence for the missing
spatial block, but it is a separate browser prototype and still has visibility
limitations.

## Current code inventory

| Path/model | Actual geometry | Important restriction |
|---|---|---|
| Ordinary 3DGS renderer | \(\mu_3+\operatorname{SPD}(3)\) | no intrinsic time |
| Paper free dynamic 3DGS | independent full 3DGS bank per frame | storage grows as \(TGN\) |
| `ScreenTimeTubeModel` | UVT mean + restricted UVT precision | no UV correlation |
| `FeatureScreenTimeTubeModel` | UVT mean + full SPD(3) UVT precision | camera/chart-specific; depth summarized |
| `WorldTubeBatch` | linear world center + XY widths + temporal gate | no \(z\)-width or full spatial rotation |
| Browser 3D dynamic path | moving center + fixed full SPD(3) | no time-varying covariance; fixed-order risks |
| Browser affine STAR | direct packed UVT precision | diagonal clamp does not guarantee SPD |

The feature STAR implementation is likely the remembered elegant “one UVT
center plus one UVT covariance” object. It is indeed a full Gaussian in
\((u,v,t)\), but it is camera-specific and has already reduced the world depth
fiber. It should be the compiler output, not the canonical multiview asset.

## What survived versus what narrowed

### Survived in theory

- full \(\mu_4\) and SPD(4) precision/covariance;
- all three space-time cross terms;
- gauge-charted or warped primitives;
- ray-depth pushforward before sensor-time slicing;
- low-rank, polynomial, recurrent, transported, and deformation alternatives;
- time-varying physical quaternion rotation as a richer candidate;
- projective camera traces and chart certificates.

### Narrowed in active World Tubes code

- full spatial SPD(3) to two XY precisions;
- world orientation to fronto-parallel axes;
- finite depth variance to affine depth metadata;
- arbitrary path to linear world translation;
- dynamic covariance to fixed/restricted support;
- dynamic appearance to fixed RGB;
- generic camera compiler to the currently exercised camera subsets.

The first four are geometry restrictions. Fixed RGB is an appearance choice,
not a consequence of 4D Gaussian geometry. Projective camera curvature is a
compiler feature, not a restoration of curved world motion.

## Historical conclusion

The clean anchor should be restored as:

\[
\boxed{
\text{world }(\mu_4,\Sigma_4,\alpha,\text{appearance})
\longrightarrow
\pi_*\Gamma^*
\longrightarrow
\text{gauged UVT + depth/order certificates}.
}
\]

The current `WorldTubeBatch` should be documented and benchmarked as a
restricted chart/scaffold of that object. The repository does not support the
stronger claim that its missing rotation, depth thickness, and full covariance
were found unnecessary.

