# Could a substantive camera-gauge formulation improve World Tubes?

User asks whether redesigning around gauges and gauge theory could improve the
method. This is a mathematical/design follow-up to the September 6 audit,
not authorization to replace the renderer or launch experiments. No production
code, manuscript, evidence ledger, or training state was changed. One independent
read-only mathematical review checked the moving-frame equations and caveats.

## Assessment

Yes, there are concrete possibilities: shared moving frames for compact motion
and compiler work, coordinate-consistent temporal derivatives, and equivariant
learned feature transport. None follows merely from naming coordinates gauges.
The best first experiment adds one shared rigid motion factor to the existing
compiler. A full gauge-equivariant learned world encoder is a separate, larger
research branch. Current rendering correctness defects still need their own
fixes and must not be obscured by this proposal.

The previous recommendation to remove gauge-theory framing applied to the
current demonstrated algorithm. It does not rule out an algorithm in which
gauge covariance is an actual implementation constraint.

## Exact moving-frame formulation

Use homogeneous points x(t) in a fixed world frame, world-to-camera extrinsic
C(t) in SE(3), and a differentiable coordinate change G(t) in SE(3). Intrinsics
K remain sensor properties. Define

```text
x' = G x
C' = C G^-1
C' x' = C x.
```

The physical scene/camera setup is unchanged; projected points agree exactly.
This does not identify physically different viewpoints. A world-to-camera
transform can itself be used as the coordinate frame, G=C, giving C'=I. Then
the scene coordinates inherit the camera motion. This cannot remove physical
screen motion or visibility events.

Let A = dot(G) G^-1. The derivative in the moving frame must be

```text
D_t x' = dot(x') - A x' = G dot(x).
```

Under another time-dependent frame change x''=H x', the connection transforms
as A''=dot(H) H^-1 + H A H^-1, yielding D_t x''=H D_t x'. This is a substantive
use of a connection and covariant derivative. A derived connection of this form
is pure gauge. On an ordinary time interval this does not introduce physical
curvature or a new dynamical field; adding a curvature penalty is not justified
by the camera trajectory alone.

For G=(R,p), ordinary velocities obey

```text
dot(x')_spatial = R dot(x)_spatial + dot(R) x_spatial + dot(p).
```

The covariant derivative subtracts the latter two coordinate-motion terms.
Motion regularization should preserve the original physical penalty, e.g.
`||spatial(D_t x')||^2`, rather than penalizing the raw moving-frame velocity.
Otherwise a learned moving frame could make motion appear free, or penalize
a static scene for apparent camera-induced motion. Acceleration uses D_t^2.
Positions need a transformed physical anchor when a position penalty is used.

SE(3) preserves spatial volume and ray lengths. Densities are pulled back as
scalar fields; view-dependent appearance must transform directional arguments.
This is simpler than arbitrary nonrigid coordinate maps, which also require
metric/measure changes and can destroy simple ray and splat geometry.

## Closure limitation: time-dependent gauges can make the model harder

At each fixed time, spatial Gaussian covariance transforms exactly as
Sigma'(t)=R(t) Sigma(t) R(t)^T. However, a varying rotation generally does not
preserve a joint spacetime Gaussian or a constant-velocity tube. Around a mean,
the spacetime Jacobian is

```text
J = [ R     dot(R) mean_x + dot(p) ]
    [ 0                 1         ]
Sigma'_xyzt approximately J Sigma_xyzt J^T.
```

This is a local approximation. Introducing gauges must not silently promote
it to global SPD4 closure. For example, rotation turns many affine world
trajectories into sinusoidal trajectories. Independently approximating Gx and
CG^-1 can destroy an exact cancellation between them. Compose shared transforms
before approximation where possible.

## Branch A: shared rigid motion plus residual tubes

Represent homogeneous primitive centers as

```text
x_i(t) = H(t) q_i(t)
C(t) x_i(t) = [C(t) H(t)] q_i(t).
```

H is a shared rigid motion stored as part of the world asset; q_i is a simple
residual trajectory. Compile C H once per shared group and reuse it. Spatial
covariances and directional appearance must use the same shared transform.

Hypothesis: coherent motion permits fewer residual coefficients, longer useful
approximation intervals, and less repeated lowering/VJP work at matched error.
For a rigid platform q_i can be constant. If the camera moves with that
platform, C=C0 H^-1, the product C H=C0 is constant: composing first exposes
an exact cancellation. The original physical image already has this property;
the factorization exposes it computationally.

There is a real gauge freedom inside the factorization: for arbitrary valid S,
q_i'=S q_i and H'=H S^-1 yield the same physical x_i. A compactness objective
can select a useful representative. Physical motion penalties remain on the
reconstructed x_i or their equivalent covariant form; residual simplicity is
a separate representation-cost choice.

Important alternative explanation: any improvement may come entirely from
ordinary shared rigid transforms/scene-graph instancing. That is a mandatory
comparison, not evidence for additional gauge machinery. Known H is an oracle
experiment; learning H, accounting for its parameters, and paying its optimization
cost are later tests. Start with one group, not hand-designed object segmentation.

The existing export contract is preserved when H and q_i are in W and depend
on time only. The query camera enters deterministic rendering through C H.
A camera-dependent learned scene generator would be a different contract.

Falsifiers: independent primitive motion eliminates compression; transform
overhead exceeds saved work; current compiler already captures the same
cancellation; or simple shared transforms match every purported extra benefit.

## Branch B: gauge-equivariant learned geometry and feature transport

This is the branch where local gauge structure can materially constrain a
learned operator. Let F_i be an orthonormal local frame and v_i a vector-valued
feature's coordinates. Under independently changed bases F_i'=F_i h_i,
v_i'=h_i^-1 v_i. Transport from frame j to i is

```text
U_ij = F_i^T F_j
U_ij' = h_i^-1 U_ij h_j
m_i = sum_j w_ij U_ij v_j.
```

For invariant scalar weights w_ij, m_i'=h_i^-1 m_i. Features are compared in a
consistent frame rather than concatenating arbitrary coordinate components.
More general feature representations need their corresponding group actions;
arbitrary MLPs/nonlinearities do not preserve this equation automatically.

Hypothesis: an equivariant encoder/update rule needs less data to handle
arbitrary world-frame orientation and behaves consistently across camera rigs.
That is a generalization/learning hypothesis, not a promised rasterizer speedup.
Coordinate equivariance does not reconstruct unseen surfaces, resolve missing
motion evidence, or guarantee novel-view quality.

Cheap tests: change only global and local coordinate bases for the same
physical observations; check equivariance of outputs and appropriately
transformed covectors. Then compare heldout reconstruction at matched capacity,
training budget, and wall time against ordinary frame normalization and rigid
augmentation. Stop if only synthetic coordinate tests improve while deployment
quality and cost do not.

Prior art establishes feasibility rather than novelty for this particular
proposal: [Cohen et al., Gauge Equivariant Convolutional Networks](https://proceedings.mlr.press/v97/cohen19d.html)
construct locally gauge-equivariant operators. [Xu et al., Equivariant Light
Field Convolution and Transformer](https://arxiv.org/abs/2212.14871) learn ray-space
operators for reconstruction/rendering that are equivariant to coordinate-frame
changes from relative camera poses. The older v1 title of the latter paper was
SE(3)-Equivariant Reconstruction from Light Field.

## Branch C: gauge fixing in joint camera/scene estimation

If both scene and cameras are optimized, redundant global pose (and scale for
an unanchored monocular setup) can create optimization null directions. Fixing
a reference frame/scale or optimizing the quotient can help conditioning.
This is established bundle-adjustment practice, not a new camera theory:
[Triggs et al., Bundle Adjustment—A Modern Synthesis](https://lear.inrialpes.fr/people/triggs/pubs/Triggs-va99.pdf).

Applicability is conditional. Fixed calibrated cameras can already anchor these
directions, and arbitrary per-time frame changes are not an extra physical
ambiguity when trajectories/dynamics are constrained. Test this branch only
when jointly estimated poses expose the relevant nullspace. Do not add camera
optimization merely to make gauge fixing useful.

## Decision experiment before any rebuild

Use identical synthetic physical scenes with exact known motion:

1. Coherently rotating/moving primitives, with static and co-moving cameras.
2. A static world with a moving camera.
3. Independently moving primitives with no useful common rigid motion.

Compare (a) current world-frame/Jacobian compiler, (b) ordinary shared H plus
residuals, and (c) that same factorization with the additional proposed gauge
selection/consistent derivatives. First establish exact forward equivalence,
then matched image and physical world-gradient error under approximation.
Raw coordinate gradients are covectors and must be transformed before comparison.

Measure coefficient/storage size including H, accepted interval count, support
and visibility events, compilation/VJP time, total forward/backward time, and
all transform/selection overhead. Do not claim eliminated physical visibility
events: different factorization can reduce conservative splitting only.

If (b) beats (a), keep the simple shared-motion representation. If (c) improves
conditioning, approximation efficiency, or learning beyond (b), retain the
additional machinery with that specific claim. If neither improves intended
workloads, keep the existing compiler. No new experiment was executed here.

Related learned-coordinate work also exists in [General Neural Gauge Fields](https://arxiv.org/abs/2305.03462).
Its broad learned-mapping terminology should not be confused with a proof that
every noninvertible neural coordinate map is an SE(3) gauge transformation.
This is a targeted literature check, not exhaustive novelty clearance.
