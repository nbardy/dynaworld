# WorldFoam Handoff

Date: 2026-07-05

Status: historical handoff. Its chronology is preserved, but its "current"
status and next-work queue are superseded by
`WORLD_FOAM_MEMORY_LIGHT_THEOREM_LEDGER_2026-08-03.md`,
`WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md`, and
`../../TODO/worldfoam_memory_light_native4d.md`. In particular, active kinetic
CPU compilation, multi-chart transfer, material-action certification, and a
frozen-program sparse geometry/material VJP have landed since this note.

Purpose: give a clean-thread agent enough context to continue the WorldFoam
paper/implementation lane without re-deriving the whole trail. This is about
what we tried, what the current solution is, which math is shared with World
Tubes, where WorldFoam differs, and what still blocks a strong paper claim.

## 0. One-Screen Summary

WorldFoam is the second-paper lane:

```text
WorldFoam in Gauged Camera Space:
Ray-Fiber Transmittance Fields for Dynamic Rendering
```

The idea is not just "PowerFoam on Metal" and not just "World Tubes but with
cells." It is:

```text
bounded world cells / foam matter
    -> pulled through a known camera-ray bundle
    -> lifted opacity field sigma(u,v,t,z)
    -> transmittance prefix along ray fibers
    -> frames/exposures as cheap evaluations or slices
```

Current honest state:

```text
Strong:
    - real implementation trail: PowerFoam/WorldFoam trainers, Metal variants,
      tests, configs, benchmark harnesses, and many saved result artifacts.
    - local speed evidence: clean Gate4/native-cutwalk microgates beat matched
      STAR UVT on total/backward timing at 2/4/8/16f.
    - trainability evidence: one-step real32 loader smoke has 32 loaded frames,
      no repeats, loss decrease, nonzero gradients, and parameter update.

Weak:
    - not yet RGB-quality competitive with STAR UVT or solid same-source
      baselines.
    - official CUDA/Warp parity is not complete.
    - current best speed gates are microgates, not a public benchmark paper.
    - topology/witness/holonomy math is promising but diagnostic, not proven.
```

Near-term paper shape:

```text
theory + prototype + speed/scale paper
```

Full rendering-paper shape only after quality/parity gates clear:

```text
synthetic exactness
same-representation per-frame replay baseline
gradient correctness
visibility-stress advantage
public quality floor
official parity or explicit scoped alternative
```

## 1. Where Things Live

Paper lane:

```text
research_notes/worldfoam_paper/README.md
research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md
research_notes/worldfoam_paper/WORLD_FOAM_EXPERIMENT_PLAN.md
research_notes/worldfoam_paper/WORLDFOAM_HANDOFF.md
```

Bridge to World Tubes / gauge bundle:

```text
research_notes/gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
research_notes/gauged_uvt_trace_atlas/08_worldfoam_bridge/README.md
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_EXPERIMENT_PLAN.md
```

WorldFoam proof scaffold:

```text
research_notes/worldfoam_paper/WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
research_notes/worldfoam_paper/scientist_notes/2026-07-05_depth_fiber_operator_ordering_intake.md
```

PowerFoam math and upstream reproduction notes:

```text
research_notes/foam_papers/powerfoam_mathematical_aspects_deep_dive.md
research_notes/foam_papers/powerfoam_rasterizer_notes.md
research_notes/foam_papers/powerfoam_reproduction_audit.md
research_notes/foam_papers/powerfoam_upstream_source.md
research_notes/gauge_powerfoam/2026-05-05_powerfoam_gauge_field_math_directions.md
research_notes/foam_papers/2026-05-05_powerfoam_cech_aabb_witnessed_geometry_math.md
```

Main local implementation surfaces:

```text
src/train/train_powerfoam_metal.py
src/train/powerfoam_metal_trainer.py
src/train/powerfoam_direct.py
src/train/powerfoam_adjacency.py
src/train/powerfoam_geometry.py
src/train/powerfoam_eval_render.py
src/train/objective/world_foam_frozen_rgb_mse.py
research_experiments/dynamic_foam/
research_experiments/world_foam_lane2/
third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_direct_v0/
```

Canonical status surfaces:

```text
PROJECT_INDEX.md
BASELINES.md
EXPERIMENTS.md
README.md
TODO/README.md
```

## 2. What We Tried

### 2.1 Upstream PowerFoam / Radiant-Foam-Derived Math

The inherited math is from the Radiant Foam -> PowerFoam family.

Radiant Foam starting point:

```text
ordinary Voronoi cells:
V_i = { x : ||x - p_i|| <= ||x - p_j|| for all j }
```

This is elegant for ray traversal because cells form a partition-like
adjacency structure, but ordinary Voronoi cells are unbounded and awkward for
tile rasterization/culling.

PowerFoam move:

```text
pow_i(x) = ||x - p_i||^2 - r_i^2

B_i = { x : ||x - p_i|| <= r_i
            and pow_i(x) <= pow_j(x) for neighbors j }
```

The same radius `r_i` is both:

```text
support radius
power weight that moves radical faces
```

This is useful because changing `r_i` gives gradients both for support extent
and for cell boundaries.

Radical face against neighbor `j`:

```text
n_ij = p_j - p_i
h_ij = 0.5 (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)

x dot n_ij <= h_ij
```

Ray crossing:

```text
x(s) = o + s d
s_face = (h_ij - o dot n_ij) / (d dot n_ij)
```

That part is the clean inherited geometry. It is the reason foam is attractive:
cell support, cell adjacency, and ray order are not independent hacks.

### 2.2 Local PowerFoam Metal Reproduction

We explored local PowerFoam/Metal paths with:

```text
bounded power cells
Cech/AABB adjacency
quaternion height+SV primitives
tiled and raytrace paths
replay backward
normal-distance gradients
posed-camera trainer plumbing
synthetic 4K verifier artifacts
DeepView / Neural3D train/heldout probes
CUDA smoke scaffolding
official fixture/parity blockers
```

The strong lesson:

```text
basic Metal trainability and raytrace backward are real
```

The weak lesson:

```text
clean heldout RGB quality is not solved by the primitive alone
```

DeepView-style rows often sit below paper acceptance:

```text
heldout PSNR around 12.5-12.7 in better current rows
SSIM around 0.10-0.13
rough floor target: PSNR >= 13 and SSIM >= 0.15
```

Some Neural3D/EX4DGS-init rows are more encouraging, but they depend on an
external pretrained/init artifact and should not be treated as a clean local
COLMAP benchmark.

### 2.3 WorldFoam Lane2 Shader Variants

This is the dense shader research lane under:

```text
research_experiments/world_foam_lane2/
third_party/fast-mac-gsplat/variants/world_foam_lane2_*
```

Representative variants/ideas explored:

```text
moving ray slab compiler
shared real-ray replay
CSR candidate storage
segment tapes
delta tapes
boundary delta tapes
record delta tapes
owner-run tapes
endpoint-run tapes
endpoint-record edit replay
block4 / blockcoeff / coeff16 variants
i16x3 / i16x4 / packed coefficient encodings
framegroup16 fused-MSE loss reduction
factorized coefficient recompute
framebitmask selectors
native owner-run cutwalk prep
matched STAR comparison wrapper and verifier
real32 loaded-frame contract
```

Important result pattern:

```text
More tape often helps compute but hurts memory.
More recompute helps memory but can hit traversal/ALU floors.
Per-channel or overly rich tapes are usually not viable.
Compact scalar/endpoint/owner-run records are closer to the right shape.
```

Selected/current compute keeper for speed evidence:

```text
native owner-run cutwalk / Gate4 fused-MSE microgate
```

Not because it is the final WorldFoam paper representation, but because it has
the cleanest local speed/scale evidence and guardrails.

### 2.4 Dynamic / Camera-Program Extension

We also explored the dynamic idea:

```text
world cells are stable objects
camera changes aggressively
compile cell-camera intersections over time
reuse the event structure across frames
```

For revolving cameras, this is the key distinction:

```text
screen footprints can whirl around and become ugly,
but world cells and cell-camera intersections may remain coherent.
```

The bridge note states the target clearly:

```text
cell-camera intersections
instead of relearning screen tubes from scratch
```

## 3. Current Math Model

### 3.1 Shared Bundle/Gauge Backbone

WorldFoam reuses the same camera-bundle math as World Tubes.

Sensor-time base:

```text
B = Omega x T
y = (u,v,t)
```

Camera ray bundle:

```text
pi: E_Gamma -> B
pi^{-1}(y) = F_y
```

Camera map to spacetime:

```text
Gamma: E_Gamma -> M
M = R^3 x R
```

Gauge/trivialization over a local domain:

```text
chi_l: E_Gamma | C_l -> C_l x Z_l
chi_l(e) = (y,z)
```

Gauge-coordinate camera map:

```text
Gamma_l(y,z) = Gamma(chi_l^{-1}(y,z))
```

Measure correction:

```text
dmu_y(e) = J_l(y,z) dz
```

The `J_l` Jacobian is important. It is the same lesson as World Tubes: ordinary
depth, log depth, inverse depth, and projective/orbit gauges are only physically
equivalent if the fiber measure is transformed correctly.

### 3.2 WorldFoam Pullback

Cells:

```text
W = { F_j, theta_j }_{j=1}^N
F_j subset M

sigma_j(x; theta_j) >= 0
c_j(x, omega; theta_j)
```

Pull each cell density into the camera gauge:

```text
rho_{j,l}(y,z)
  = 1_{Gamma_l(y,z) in F_j}
    sigma_j(Gamma_l(y,z))
    J_l(y,z)
```

Lifted opacity field:

```text
sigma_l(y,z) = sum_{j in A_l(y,z)} rho_{j,l}(y,z)
```

Lifted color numerator:

```text
q_l(y,z) = sum_j rho_{j,l}(y,z) c_j(Gamma_l(y,z))
```

Premultiplied/local color:

```text
c_l(y,z) = q_l(y,z) / sigma_l(y,z)
```

### 3.3 Transmittance Rendering

WorldFoam's visibility object is optical depth, not primitive sort.

```text
tau_l(y,z) = integral_{z_front}^{z} sigma_l(y,s) ds
T_l(y,z)   = exp(-tau_l(y,z))
I(y)       = integral T_l(y,z) sigma_l(y,z) c_l(y,z) dz
alpha(y)   = 1 - exp(- integral sigma_l(y,z) dz)
```

This is standard emission-absorption/Beer-Lambert rendering. The novelty is
not that equation. The novelty is the compact bounded-cell, camera-compiled
atlas and GPU strategy for reusing cell/ray event structure over sensor time.

### 3.4 Foam Atlas

Compiled object:

```text
K_Foam = { C_l, Z_l, A_l, H_l, S_l, P_l, E_l }_{l=1}^L

C_l     sensor-time gauge domain in (u,v,t)
Z_l     ray-fiber interval or depth-layer partition
A_l     active cells and cell-ray intersection records
H_l     local lifted opacity/radiance bases
S_l     support/intersection certificates
P_l     transmittance prefix summaries or depth-layer prefix scans
E_l     error/fallback metadata
```

Evaluation:

```text
locate C_l for y
load active cell/intersection records
scan depth/event layers front-to-back
update tau/T
accumulate I
fallback if the local basis/event certificate is invalid
```

Backward:

```text
front transmittance prefix
behind-radiance suffix
local basis derivatives
cell-parameter Jacobians
```

Continuous variation:

```text
delta I(y)
  = integral T sigma delta c dz
  + integral T (c(z) - I_behind(y,z)) delta sigma(z) dz
```

This is the continuous analog of front-to-back alpha compositing gradients:

```text
dI/d alpha_i = T_i (c_i - I_behind_i)
```

## 4. Is This As Elegant As World Tubes?

Short answer:

```text
World Tubes is more algebraically elegant.
WorldFoam is more physically/structurally elegant.
```

World Tubes has a crisp closed-form jewel:

```text
spacetime Gaussian
  + local affine camera gauge
  -> integrate ray depth
  -> Schur complement
  -> UVT Gaussian footprint + conditional depth
```

That works because Gaussians are closed under marginalization. The Schur
complement compresses the lifted `(u,v,t,z)` object into a small `(u,v,t)`
footprint plus conditional depth statistics.

WorldFoam does not generally have that free lunch:

```text
bounded power cell
  + hard radical faces
  + clipped ray intervals
  + topology changes
  -> event intervals and transmittance prefixes
```

There may be local analytic pieces, but not one universal Schur collapse.

So the elegance is different:

```text
World Tubes:
    marginalize analytic Gaussian atoms; keep GS compatibility.

WorldFoam:
    preserve lifted depth/fiber structure; make visibility physical.
```

WorldFoam's best math is:

```text
visibility = cumulative optical depth along a camera fiber
```

not:

```text
visibility = sorted list of projected primitives
```

The sharper operator-ordering version is:

```text
World Tubes:
    early pushforward pi_* Gamma^* primitive,
    then visibility/order certificates repair what depth marginalization loses.

WorldFoam:
    delayed pushforward R_z Gamma^* matter,
    where R_z is the transmittance-prefix renderer along the retained depth
    fiber.
```

This is the reason the depth fiber is useful in both tracks but not used the
same way.

The current paper-plan upgrade is to make WorldFoam an optical-transfer event
algebra, not just a sigma-prefix renderer:

```text
event element:        g = (beta, m)
composition:          g1 otimes g2 = (beta1 beta2, m1 + beta1 m2)
render:               G(y) = otimes_k g_k(y)
decode:               I(y) = m + beta I_bg(y)
visibility error:     commutator ~ opacity overlap * color contrast
cell-path renderer:   ray -> certified owner/event word -> monoid scan
World Tubes closure:  Schur complement
WorldFoam closure:    event intervals + optical-transfer monoid
```

Promote the monoid, transfer element, commutator theorem, event replay
equivalence, cell-path atlas definition, and monoid VJP. Keep Magnus
compression, Hessians, interface-flux boundary gradients, flux witness scores,
feature-gauge transfer, and universal ray-space transfer behind tests.

## 5. What Math Is Reused From World Tubes?

Reused:

```text
camera-ray bundle E_Gamma -> B
sensor-time base B = Omega x T
local gauges over camera programs
fiber coordinate z
measure Jacobian J_l(y,z)
known camera path compilation
event-certified domains
finite exposure / rolling shutter compatibility
compiled forward/backward reuse across time
fallback when local model fails
```

Changed:

```text
World Tubes pushes primitives down to UVT footprints:
    alpha_i(u,v,t), c_i(u,v,t), z_i(u,v,t)

WorldFoam keeps the lifted fiber:
    sigma(u,v,t,z), c(u,v,t,z), tau/T prefix
```

World Tubes needs a separate visibility gauge atlas because depth was partly
marginalized away:

```text
footprint atlas + lifted depth/order atlas
```

WorldFoam keeps depth/fiber visibility primary:

```text
lifted opacity atlas + transmittance prefix
```

## 6. What Breaks Or Needs To Be Different

### B1. No global Schur-complement compression

Bounded cells are not Gaussian tails. A cell has:

```text
spherical support boundary
radical face boundaries
entry/exit intervals
neighbor graph dependence
```

So the fast path is not covariance algebra. It is:

```text
ray/cell event detection
interval compression
depth/layer prefix scan
fallback for topology churn
```

### B2. Topology can change

Cell adjacency and ownership are not fixed under training:

```text
Cech/AABB edges appear/disappear
radical face winners change
endpoint winners change
support intervals enter/leave tiles
```

This makes the compiler less smooth than World Tubes. The right response is
not to pretend topology is differentiable everywhere. The right response is:

```text
compile hard fast structure
log topology churn
use witnessed/holonomy diagnostics
refresh/rebuild when needed
fallback or split chaotic domains
```

### B3. General backward needs behind-state information

For transmittance, density gradients depend on what is behind the point:

```text
dI/d sigma(z) = T(z) (c(z) - I_behind(z))
```

For general material laws, backward needs either:

```text
stored prefix/suffix state
compact scalar contribution/weight tape
endpoint-winner tape
or recomputation
```

The later finite-P0 implementation selected exact recomputation: one forward
gets the final affine transfer and a second front-to-back scan keeps only the
current prefix state. It needs no stored per-run suffix/reverse array. This is
why the earlier shader lane's tape variants are historical alternatives rather
than the current P0 memory contract.

### B4. Memory can become the whole problem

Naively caching `(u,v,t,z)` is too big. Naively storing per-channel tapes is
also too big. Viable memory shape is closer to:

```text
cell/intersection events
compressed endpoint records
depth-layer prefix summaries
scalar tapes, not feature-channel tapes
interval/framegroup compression
```

### B5. Geometry support is currently the quality bottleneck

The current speed gates prove the shader can be fast. They do not prove the
representation can recover good heldout geometry.

Observed blockers include:

```text
weak clean point-cloud support
mostly two-view tracks
source-view fit not predicting heldout quality
appearance absorbing geometry error
unwitnessed faces/cells
topology instability / false Cech edges
```

This is why the witnessed power complex and holonomy ideas matter.

### B6. Compatibility is different

World Tubes can be sold as:

```text
same dynamic Gaussian semantics, compiled through camera path
```

WorldFoam is not that. It changes semantics:

```text
from sorted splat alpha compositing
to lifted opacity/transmittance
```

That is cleaner physically, but harder to compare. Same-representation
WorldFoam replay is the main baseline; STAR/GS are contextual challengers.

## 7. Current Evidence

### 7.1 Clean Gate4 Native-Cutwalk Microgate

Artifact:

```text
research_experiments/world_foam_lane2/results/
2026-05-20_native_cutwalk_worldfoam_star_starretry.promotion_summary.json
```

WorldFoam 2/4/8/16f:

```text
mean total:
    3.008 / 3.014 / 3.323 / 4.095 ms

backward:
    2.739 / 2.517 / 2.561 / 3.796 ms

scale over 8x frame increase:
    1.361x total
    1.386x backward

train PSNR:
    11.770 / 11.783 / 12.150 / 12.248

heldout PSNR:
    12.352 / 12.406 / 12.589 / 12.857
```

Matched STAR:

```text
median total:
    5.003 / 5.943 / 8.092 / 9.794 ms

backward:
    2.629 / 3.411 / 5.083 / 6.768 ms
```

Interpretation:

```text
WorldFoam is faster in this microgate.
This is speed/scale evidence, not RGB system parity.
```

### 7.2 Repeated-Fixture 32f Smoke

WorldFoam 2/4/8/16/32f:

```text
total:
    2.829 / 3.248 / 4.414 / 4.643 / 6.371 ms

backward:
    2.557 / 2.965 / 4.054 / 4.254 / 6.001 ms

scale over requested 2 -> 32f:
    2.252x total
    2.347x backward
```

Critical caveat:

```text
32f repeats 16 loaded frames.
Use as speed-scaling smoke only.
```

### 7.3 Render96/Site48 Gate

WorldFoam 2/4/8f:

```text
total:
    3.760 / 4.125 / 4.619 ms

backward:
    3.480 / 3.847 / 4.331 ms
```

Matched STAR:

```text
total:
    5.773 / 7.583 / 9.692 ms

backward:
    3.614 / 5.161 / 6.719 ms
```

Interpretation:

```text
larger fused-MSE speed/scale gate
i32 framebitmask offsets fixed
not RGB system parity
```

### 7.4 Real32 Loader Smoke

Config:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Evidence:

```text
loaded_frame_count = 32
no repeat flags
loss decrease
nonzero gradient
parameter update
```

Caveat:

```text
benchmark environment was contended
correctness/data gate only
```

Warm real32 retries saw about:

```text
2.25-2.30 ms total
1.95-2.01 ms backward
```

but those were rejected by the wrapper because the environment became
contended before promotion, and no matched STAR comparison ran.

### 7.5 Quality Gap

Current honest stance:

```text
speed-competitive in clean microgates
not RGB-quality competitive with STAR UVT or solid same-source baselines
```

Recorded gap:

```text
best WorldFoam train PSNR:   about 12.248
best WorldFoam heldout PSNR: about 12.857
gap to solid same-source:    about 9.112 dB
gap to STAR UVT source:      about 17.575 dB
```

This is the main reason the WorldFoam paper should be separate from the World
Tubes paper.

## 8. Current Solution

There are two "current solutions," depending on what question is being asked.

### 8.1 Current engineering solution

For local speed evidence:

```text
native owner-run cutwalk / Gate4 fused-MSE microgate
strict wrapper + preflight + verifier
matched STAR comparison
real-frame contract when required
```

Use this to produce honest speed/scale artifacts. Do not use it to claim broad
quality.

### 8.2 Current paper/theory solution

For the second paper:

```text
WorldFoam atlas:
    bounded world cells
    camera-gauge pullback into sigma(u,v,t,z)
    interval/depth-layer event compression
    transmittance prefix rendering
    prefix/suffix adjoint reuse
    topology diagnostics for witness/holonomy
```

This is the model to describe and test. It should be compared first against
per-frame WorldFoam replay, not against arbitrary external methods.

## 9. Goals

### G0. Keep the claim honest

Do not claim:

```text
SOTA dynamic novel-view synthesis
replacement for STAR UVT / dynamic 3DGS
official PowerFoam parity
sublinear total work in output pixels
quality parity from speed microgates
```

Safe current claim:

```text
WorldFoam is a camera-gauged lifted opacity/transmittance formulation with a
Metal prototype. Focused local microgates show favorable speed/frame-count
scaling. Quality and official parity remain open.
```

### G1. Prove synthetic exactness

Need:

```text
constant-density sphere
crossing translucent slabs
crossing Gaussian density sheets
thin foreground occluder
dense semi-transparent cloud
moving cell complex
fast orbit camera
rolling shutter / finite exposure
```

Compare:

```text
dense raymarch / analytic reference
per-frame WorldFoam replay
compiled WorldFoam atlas
sorted splats / World Tubes compatibility mode where useful
```

### G2. Isolate compiler benefit

Required same-representation baselines:

```text
F0 dense/analytic reference
F1 per-frame WorldFoam replay
F2 replay with cached camera constants
F3 cached active set, live intersections
F4 compiled intersections, live transmittance
F5 compiled intersections + depth-layer prefix
F6 full compiled atlas + backward prefix/suffix
F7 fallback/reference cells
```

### G3. Get gradient correctness

Need tests for:

```text
prefix/suffix transmittance gradients
cell density gradients
color gradients
endpoint-winner / boundary gradients
topology refresh boundaries
finite exposure accumulation
```

### G4. Close or clearly scope quality

Minimum public-quality target before full paper:

```text
DeepView-like heldout PSNR >= 13
DeepView-like heldout SSIM >= 0.15
competitive with same-budget replay
not catastrophically behind STAR/dynamic-gsplat baselines
```

If this does not clear, publish only as:

```text
theory + prototype + speed/scale + synthetic visibility-stress paper
```

### G5. Decide official parity scope

Either:

```text
close CUDA/Warp official fixture parity
```

or explicitly scope the paper as:

```text
Metal prototype / independent WorldFoam implementation
```

Do not imply official PowerFoam reproduction if the official CUDA/Warp fixture
has not been run.

## 10. Next Work Queue

Immediate, highest value:

1. Build the cell-path optical-transfer fixture: constant-density owner-run
   word, monoid scan, same-representation replay, and exact VJP finite
   differences for `DeltaTau`, `sigma`, color, and run length. Use
   `experiment_designs/cell_path_optical_transfer_fixture.md` as the code-level
   file/function/test plan.
2. Build synthetic transmittance correctness suite.
3. Add crossing translucent slabs/Gaussian sheets as the visibility-stress
   benchmark.
4. Add per-frame WorldFoam replay baseline to isolate compiler benefit.
5. Add prefix/suffix gradient correctness test.
6. Run clean real loaded-frame 2/4/8/16/32f wrapper with
   `--require-real-loaded-frames` and matched STAR comparison when the MPS
   environment is quiet.

Then:

1. Log Cech/AABB/witness/holonomy diagnostics on trained foam checkpoints.
2. Test whether flux-witness/holonomy metrics predict heldout-free validation,
   source leave-one-camera-out error, topology churn, or traversal instability.
3. Run public DeepView/Neural3D/D-NeRF subset only after same-representation
   compiler gates are clean.
4. Convert paper draft into LaTeX only after deciding theory/prototype versus
   full rendering-paper scope.

Do not spend time on:

```text
long hyperparameter sweeps before synthetic exactness and replay baselines
another tape variant without a memory/gradient contract
claiming quality from source-view or fused-MSE microgates
topology regularizers before topology diagnostics correlate with failure
```

## 11. Falsification Tests

### F1. Compiler benefit is fake

Test:

```text
per-frame WorldFoam replay vs compiled atlas at equal quality
```

WorldFoam compiler weakens if:

```text
compiled path only beats because representation/loss differs
compile amortization point is too late
memory grows nearly per-frame
fallback fraction dominates
```

### F2. Transmittance semantics does not help visibility

Test:

```text
crossing translucent slabs/sheets
```

WorldFoam weakens if:

```text
sorted splats / World Tubes visibility gauge match or beat it
flicker/gradient stability do not improve
transmittance basis needs too many depth layers
```

### F3. Geometry support is the real blocker

Test:

```text
oracle/well-initialized cells vs clean train-only reconstruction
```

Supports current suspicion if:

```text
oracle/external-init improves substantially
clean local init stays poor
witness metrics correlate with heldout failure
```

### F4. Topology diagnostics are decorative

Test:

```text
witness score / holonomy / edge churn vs heldout-free validation
```

Cut the topology math from the paper if:

```text
metrics do not predict failure
regularizers improve metrics but hurt heldout
metrics mostly track alpha/coverage only
```

### F5. Real32 speed does not survive clean benchmarking

Test:

```text
strict wrapper, quiet MPS environment, real loaded 32f, matched STAR
```

Weakens speed claim if:

```text
clean real32 total/backward scales linearly
STAR catches up at realistic scene/cell counts
CPU prep dominates despite native cutwalk
```

## 12. Relation To World Tubes Paper

Keep these papers separate.

World Tubes:

```text
baseline-compatible dynamic Gaussian compiler
Schur-complement UVT footprints
visibility gauge atlas for order certification
compiled interval VJP
stronger near-term arXiv lane
```

WorldFoam:

```text
different rendering semantics
lifted opacity/transmittance
bounded cell complex
cell-ray event compiler
topology/witness diagnostics
speed/prototype evidence now, quality later
```

Shared math:

```text
camera bundle
gauges
fiber coordinate and Jacobian
known camera program compilation
event domains
backward reuse across time
```

Different math:

```text
World Tubes:
    integrate z away when possible, then certify order.

WorldFoam:
    keep z, because visibility is the prefix integral along z.
```

## 13. Best Next Goal Prompt

Use this to start a clean thread:

```text
Continue the WorldFoam second-paper lane in /Users/nicholasbardy/git/gsplats_browser/dynaworld.
Read AGENTS.md, PROJECT_INDEX.md, BASELINES.md WorldFoam rows, research_notes/worldfoam_paper/WORLDFOAM_HANDOFF.md, and research_notes/worldfoam_paper/WORLD_FOAM_EXPERIMENT_PLAN.md.

Goal: turn WorldFoam from a speed-prototype/math lane into a falsifiable paper lane.
Start with the synthetic exactness suite and same-representation per-frame replay baseline:
bounded cells pulled through a camera gauge into sigma(u,v,t,z), rendered by transmittance prefix, with gradient correctness against dense/analytic references.

Keep claims honest:
do not claim SOTA quality, official PowerFoam parity, or sublinear output-pixel work.
The safe claim is camera-gauged lifted opacity/transmittance with local Metal speed evidence.
The main blockers are quality, topology/witness support, memory/tape scaling, and official parity.
```
