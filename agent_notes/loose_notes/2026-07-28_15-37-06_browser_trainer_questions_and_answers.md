# July 28 Questions And Answers: Browser Dynamic Splat Trainer

Date: 2026-07-28

Status: evidence-backed project reflection and next-work framework

## Frozen State Ledger

This ledger prevents old sampled-ray observations from being mixed with the
current full-frame system.

| Item | Current state |
| --- | --- |
| Source state | `9cfe24d` (`Page and compact high-resolution browser training`) |
| Browser backend | `tiled3d`, full-frame 16x16 tiled raster and shared backward |
| Control backend | `sampled3d`, 96 sampled rays and RGB MSE/support guards |
| Dataset | Coffee Martini |
| Cameras | 17 train, `cam06` held out |
| Times | 16 synchronized frames |
| Training raster | 96x72 |
| Default splats/capacity | 4,096 / 4,096 |
| Initialization | 4,096 train-visible Ex4DGS SfM points in `cam04` coordinates |
| Objective | `0.8 L1 + 0.2 (1 - SSIM)` |
| SSIM implementation | 11x11 reflected uniform box window |
| Checkpoint precision | packed FP16 storage by default; training arithmetic remains FP32 |
| GPU target residency | one RGBA32F camera/time page |
| Host target residency | complete decoded Float32 tensor; not yet shared across workers |
| Motion representation | 3D covariance plus linear or linear+sinusoidal center motion and temporal gate |
| Topology | fixed allocation with GPU split/fill/recycle through step 60,000 |
| Validation | asynchronous 12x12 sparse grid and global-luma SSIM proxy |
| Live preview | two train cameras plus heldout, looping over time |
| Research status | browser systems prototype, not a validated paper baseline |

Evidence:

- `web/dynaworld_browser_trainer/README.md`
- `web/dynaworld_browser_trainer/trainerWebGpu3dTiled.js`
- `web/dynaworld_browser_trainer/coffee_martini_train17_holdout1.json`
- `web/dynaworld_browser_trainer/benchmark_results/2026-07-28_tiled_scaling_apple_m4.json`
- `web/dynaworld_browser_trainer/benchmark_results/2026-07-28_tiled_memory_precision_apple_m4.json`
- `agent_notes/loose_notes/2026-07-28_16-28-58_browser_target_paging_and_precision.md`
- `research_notes/browser_4dgs_baseline.md`

## 1. User Questions

The user questions across this browser-trainer work reduce to these themes.

1. Which dataset is actually loaded, and is it Coffee Martini?
2. How many cameras and times are used?
3. Should all cameras train, with one camera held out?
4. Should several cameras contribute to one optimizer step?
5. Do target and result panels use the same camera and aspect ratio?
6. Can the preview show several views and loop continuously over time?
7. Is initialization random, copied from target pixels, or based on COLMAP/SfM?
8. Why did an early version appear almost perfectly initialized?
9. Are SfM coordinates normalized to useful optimizer step sizes?
10. Is topology fixed, or do splats split, spawn, relocate, and prune?
11. Why did the result become sparse?
12. Why did Gaussians remain spherical, and are scales and rotations trainable?
13. Is the active model dynamic 3DGS, native 4DGS, World Tubes, or a mixture?
14. Which additional shaders exist but are not connected to the SPA?
15. Should every experimental shader be a selectable option?
16. Does training render a full image or only random rays?
17. Is full-raster training faster because forward and backward share splat work?
18. Does the WebGPU result update continuously while training?
19. Why did training run in bursts followed by freezes?
20. Can metrics, validation, preview, and UI work avoid blocking optimization?
21. Why did a worker version advance steps while reporting null loss and stale images?
22. Why did the old brute-force kernel reach only about 7.3 steps/s?
23. How does current WebGPU speed compare with recorded Metal numbers?
24. Should WebGPU be only 20-30% slower than Metal?
25. Is current steps/s much lower than earlier browser versions?
26. How does speed scale with raster resolution?
27. How does speed scale with splat count?
28. Is raster resolution cheap relative to splat count?
29. Where is time spent within binning, sorting, raster, loss, backward, and update?
30. Is the learning rate high enough, and should it be scheduled?
31. Why does loss descend and then plateau early?
32. Did speed work accidentally damage gradient correctness or convergence?
33. Is there an earlier shader or checkpoint that serves as a quality control?
34. Should SSIM/DSSIM be used for training rather than validation only?
35. Why use an 11x11 local SSIM window instead of whole-image statistics?
36. Is 11x11 standard, and is its gradient useful?
37. What speed and quality tradeoff comes from a smaller SSIM window?
38. Should the charts contain full history, log scaling, PSNR, and SSIM?
39. Are the displayed validation PSNR and SSIM real paper metrics or proxies?
40. Which standard regularizers should be included?
41. What are acceleration, velocity, opacity, scale, and temporal regularizers?
42. Which losses and regularizers already have paper ablations behind them?
43. Are there 3D-aware optimizers that converge faster than generic Adam?
44. How should learning rates differ by center, scale, rotation, opacity, color, and motion?
45. Which pruning, spawning, densification, and relocation schedule is appropriate?
46. What do Motion Mix, Static Mix, Support Guard, and Temporal Support mean?
47. Are those sampled-ray controls relevant to the full-frame backend?
48. Is Fudan native 4D Gaussian Splatting the right external baseline?
49. Is the current browser trainer already a solid baseline?
50. What work remains before the prototype can support trustworthy research claims?

## 2. Followup

### Is Current Speed Way Lower?

It depends on the comparison.

- The old 7.3 steps/s result was a failed occupancy design that serialized
  sampled rays behind one workgroup. It is not the current kernel.
- Historical 584-824, 793, and 878 steps/s observations used 768 sampled-ray
  splats, RGB MSE/support guards, fewer frames, or different preview schedules.
- The current live SPA at 4,096 splats has recently shown about 250-280
  completed steps/s.
- The isolated current full-frame kernel at 96x72 and 4,096 matched splats and
  capacity reaches 359 steps/s with FP32 checkpoints and 403 steps/s with
  packed-FP16 checkpoints.

The live value is therefore roughly 30-38% below the isolated packed-FP16
interval, while also running preview, status, metric, and validation services.
That gap is not yet phase-attributed, so it should not all be blamed on the UI.
It is not a return to the 7.3 steps/s structural failure.

### Resolution And Splat Scaling

Apple M4, 32 warmup plus 128 measured GPU-drained steps. The 384x288/4,096
FP32 endpoint is the median of three repeats; the other cells are single
intervals:

| Raster | 768 splats | 1,536 splats | 4,096 splats |
| --- | ---: | ---: | ---: |
| 96x72 | 1,233 steps/s | 863 steps/s | 359 steps/s |
| 192x144 | 675 steps/s | 462 steps/s | 240 steps/s |
| 384x288 | 266 steps/s | 182 steps/s | 118 steps/s |

At 4,096 splats, packed-FP16 checkpoints improve the three raster points to
403, 294, and 132 steps/s. Pixels are not free: 4x pixels from 96x72 to
192x144 retains 55%, 54%, and 67% of FP32 throughput, while another 4x pixels
retains 39%, 39%, and 49%. Splat work is important, but the current full-image
SSIM, checkpoint, and backward work make raster cost substantial.

The 384x288 benchmark is systems evidence only: it nearest-neighbor scales the
96x72 targets, so it adds compute but no image detail. True detail requires a
new bundle exported at the higher source resolution.

### Memory And Precision Followup

The former 384x288 blocker is fixed. The tiled backend pages one camera/time
RGBA32F target before each step and reuses that 1.69 MiB GPU buffer. It also
removed the 162 MiB pair-gradient slab, compacts pair IDs and references, and
uses one 0.375 MiB FP32 gradient record per splat.

At 384x288/4,096, the dominant GPU binding is now the forward checkpoint tape:

| Buffer | Current |
| --- | ---: |
| Target page | 1.69 MiB |
| Forward checkpoints | 108 MiB |
| Pair IDs and references | 13.5 MiB |
| FP32 gradient accumulator | 0.375 MiB |

Packed FP16 is used only for stored forward checkpoints. Projection,
covariance, compositing, SSIM, gradients, trainable parameters, and Adam
moments remain FP32. At 384x288 the planner spends FP16's saving on twice as
many checkpoints, using the same 108 MiB but reducing backward replay. The
matched median improves from 118 to 132 steps/s, and loss after 1,024
submissions differs from FP32 by only `1.13e-6`.

The unresolved memory problem is host-side. The all-camera/all-time 384x288
Float32 tensor is still about 486 MiB and may be cloned into the training and
validation workers. The next memory change should retain canonical RGBA8
atlas bytes, share them or decode inside the owning worker, and page a packed
target or texture. Requesting a larger GPU storage binding would preserve the
wrong ownership model.

### Code Quality Followup

The data and worker boundaries are mostly healthy: the browser export remains
a thin adapter over the canonical multicamera contract, the optimizer owns one
worker, validation owns another, and the backend registry names the two real
SPA choices.

The main code-health debt is inside the renderer:

- `trainerWebGpu3d.js` is 1,740 lines and `trainerWebGpu3dTiled.js` is 1,263;
- the tiled trainer subclasses the sampled trainer and calls
  `super.createPipelines()`, `super.createBuffers()`, and
  `super.createBindGroups()`;
- this compiles sampled-only pipelines, creates sampled-only buffers, then
  replaces some of them;
- the inherited initialization still requests nine storage buffers per shader
  stage, while the portable WebGPU guaranteed minimum is lower;
- the tiled sort/backward path requests 24 KiB of workgroup storage, also above
  the portable guaranteed minimum;
- many tests inspect generated source contracts, while complete WGSL
  value/gradient readback parity is still missing.

The right refactor is a small shared runtime for device, cameras,
initialization, parameters, render preview, and disposal, with sibling sampled
and tiled trainers owning only their pipelines and buffers. It should not
create a browser/Python trainer hierarchy or merge this demo into the paper
runner.

### Why 11x11 Rather Than Whole Image?

SSIM compares local luminance, contrast, and structure. A whole-image mean and
covariance can remain similar after edges or objects move to the wrong place.
Its gradient is global and weakly localized. It is differentiable, but it is
not the standard reconstruction signal used by Gaussian Splatting.

An 11x11 Gaussian with sigma 1.5 is canonical SSIM and is used by the official
3DGS implementation. The 3DGS training mixture is also commonly
`0.8 L1 + 0.2 (1 - SSIM)`.

The browser currently uses the standard window size and constants but a
uniform box weighting. Its analytic CPU gradient passes finite differences for
that exact box objective. It has not yet been checked against WGSL gradient
readback, and it is not canonical Gaussian SSIM.

A matched speed ablation at 96x72 and 1,536 splats measured:

| Window | Median steps/s |
| --- | ---: |
| 7x7 box | 1,174 |
| 11x11 box | 933 |

The 7x7 window is about 26% faster end to end. No matched quality run exists,
so it cannot be promoted. These absolute timings also predate the compact
gradient/checkpoint layout in `9cfe24d`; they are directional evidence about
window cost, not current throughput. The best standards-preserving path is a
separable 11-tap Gaussian forward and transpose backward, ideally with
target-only moments precomputed. That reduces each 2D convolution from 121
neighborhood taps to 22 while preserving the intended objective.

### Metal Comparison

There is still no matched Metal row for this complete browser step:

- 17 training cameras and 16 times;
- 96x72 full image;
- 4,096 splats;
- tiled projection, sorting, alpha compositing, local SSIM, raster backward,
  Adam, and topology bookkeeping.

The saved 7.18 ms Metal probe used 768 splats and omitted parts of the complete
optimizer contract. Other native rows use 8,192 or 65,536 splats, different
features, resolutions, or batch dimensions. Saying WebGPU is within 20-30% of
Metal would currently be invented. A matched phase-timed harness is required.

Faster-GS is useful design evidence, not a substitute for that harness. It
reports that splat-list processing is strongly memory-bound, emphasizes
opacity-aware ellipse bounds and front-to-back backward, and finds that
parameter update becomes important after raster optimizations. The browser now
has opacity-aware bounds and front-to-back replay, but it has not measured
whether SSIM, checkpoint replay, atomics, sorting, or Adam is the present
bottleneck. Port the paper's hypotheses as ablations, not its speedup as an
expectation.

Primary source:

- Faster-GS: <https://arxiv.org/abs/2602.09999>

## 3. How To Ask Our Own Good Questions

A useful project question should contain:

1. **Decision:** what choice will change if the answer is yes or no?
2. **Frozen contract:** exact build, backend, data, split, raster, splats,
   objective, initialization, and hardware.
3. **Hypothesis:** a causal claim, not merely a symptom.
4. **Intervention:** one controlled change or a small factorial screen.
5. **Observable:** a direct metric rather than an attractive proxy alone.
6. **Threshold:** a result that counts as promotion, rejection, or uncertainty.
7. **Budget:** steps, wall time, memory, seeds, and device scope.
8. **Artifact:** JSON, images, traces, tests, and command needed to reproduce it.
9. **Failure alternatives:** at least two other causes the experiment can
   distinguish.
10. **Lane boundary:** browser ergonomics, baseline validity, renderer research,
    or paper evidence.
11. **Evidence state:** observed, inferred, proposed, or externally reported.
12. **Information efficiency:** decisions unlocked per implementation hour and
    benchmark minute.

Bad questions usually omit the decision or comparison contract. “Can we make it
faster?” is open-ended. “At fixed 96x72, 4,096 splats, and identical objective,
does separable Gaussian SSIM reduce median phase time by at least 20% without
changing CPU/WGSL value or gradient beyond tolerance?” is actionable.

For this project, score a proposed question before executing it:

```text
priority =
    (correctness_risk * decisions_unlocked * transfer_value)
    / (implementation_hours + benchmark_hours + proxy_risk)
```

The score is ordinal rather than numerically scientific. Its purpose is to
force comparison. A cheap GPU gradient parity fixture ranks above a large
learning-rate sweep because it can invalidate every later convergence result.

## 4. Loop 1 Questions: Inventory (50)

### Contract

1. What exact commit and asset tag produced the visible SPA?
2. Which backend is selected by default and which backend is only a control?
3. What work is included in one reported optimizer step?
4. Which controls affect the selected backend and which are hidden as irrelevant?
5. Which claims in project docs describe superseded browser builds?

### Data

6. Which manifest row defines Coffee Martini for the browser?
7. Which 17 cameras train and why is `cam06` held out?
8. Which 16 source-frame indices are exported?
9. Are calibration and image conventions identical to the canonical loader?
10. Can heldout pixels influence initialization, tuning, or checkpoint choice?

### Initialization

11. What file supplies the 4,096 XYZRGB seeds?
12. In which coordinate frame are those seeds stored?
13. How are train-visible points selected?
14. How are initial scales, rotations, colors, and opacities assigned?
15. Which early visual result used target-pixel initialization instead?

### Representation

16. Which parameters are genuinely trainable in the active primitive?
17. Which parameters vary over time?
18. Is appearance view dependent?
19. Is covariance time dependent?
20. Which differences remain versus dynamic 3DGS, native 4DGS, and World Tubes?

### Raster And Backward

21. How are splats assigned to tiles?
22. How is depth order established per camera and time?
23. What state is checkpointed in the forward pass?
24. Which thread owns each backward contribution?
25. Which gradients have finite-difference or parity coverage?

### Objective

26. Is the optimizer loss L1, MSE, SSIM, or a mixture?
27. Is SSIM local or global?
28. Is its window Gaussian or uniform?
29. Does border handling match the intended baseline?
30. Does reported “DSSIM” mean `1-SSIM` or `(1-SSIM)/2`?

### Optimizer And Topology

31. Which Adam learning rate applies to each parameter family?
32. Which clamps or aspect guards alter updates?
33. When do hidden slots fill?
34. When and how are weak splats recycled?
35. Does the default 4,096-splat run ever increase its active count?

### Validation

36. Which cameras and times contribute to train validation?
37. Is heldout validation full image or sparse?
38. Is displayed SSIM the same quantity used by training?
39. How often are snapshots requested and can they block GPU submission?
40. Which metrics are sufficient for a baseline claim?

### Performance

41. How many steps/s does the isolated tiled loop achieve?
42. How much throughput is lost in the live SPA?
43. How does cost scale with pixels?
44. How does cost scale with splats?
45. Which buffer first prevents higher raster resolution?

### Evidence And Decisions

46. Which quality result is recorded in `BASELINES.md`?
47. Which historical speed observations are workload matched?
48. Which standalone shaders are complete enough to compare?
49. What evidence would make the browser trainer a solid baseline?
50. What single next artifact unlocks the most downstream decisions?

## 5. Why Loop 1 Questions Are Good Or Bad

### Good

- Coverage is broad: contract, math, systems, data, quality, and evidence.
- Questions expose category errors, especially sampled versus full-frame,
  proxy SSIM versus windowed SSIM, and harmonic motion versus native 4DGS.
- Several questions can be answered directly from code and artifacts.
- The inventory makes stale documentation visible before more experiments run.

### Bad

- Many questions are descriptive and do not change a decision.
- “How” questions can be answered by reading code without testing behavior.
- Fifty equal-looking questions hide priority and dependencies.
- Several questions overlap, especially topology, active count, and spawning.
- No threshold says when an answer is sufficient.
- No experiment budget prevents a question from expanding indefinitely.
- The list is vulnerable to question laundering: producing lots of questions
  can feel like progress without creating new evidence.

## 6. How To Adjust The Framework After Loop 1

For the next loop, rewrite each inventory item as a falsifiable decision:

```text
Decision -> hypothesis -> matched intervention -> metric -> threshold
-> budget -> artifact -> alternate explanation.
```

Additional rules:

- Every speed question must include a workload tuple.
- Every quality question must include train and heldout metrics.
- Every representation question must distinguish implemented, wired, tested,
  and benchmarked.
- “Not measured” is a valid answer.
- Questions with the same required experiment should be merged.
- No new renderer becomes a SPA option before it satisfies the shared data,
  worker, validation, and backward contracts.

## 7. Five Repetitions Of Questions, Critique, And Adjustment

### Loop 2: Falsifiable Questions (50)

#### Contract

1. Can a generated state ledger reproduce every visible default from one build hash?
2. Does one reported tiled step execute exactly one complete camera/time image?
3. Does changing a sampled-ray-only control leave tiled WGSL uniforms unchanged?
4. Does every selector label match the implemented parameterization?
5. Can stale browser claims be detected automatically against the active registry?

#### Data

6. Does browser export round-trip all camera matrices within `1e-6`?
7. Does the exported heldout index remain absent from every training schedule cycle?
8. Does one schedule cycle visit all `17 x 16 = 272` train pairs exactly once?
9. Does atlas decode reproduce canonical loader pixels within 8-bit quantization?
10. Can any target or seed path read heldout RGB during initialization?

#### Initialization

11. At step zero, what fraction of seeds project into each training camera?
12. Does train-visible SfM init beat deterministic random init at matched wall time?
13. Does farthest-point selection beat a random point-cloud subset across three seeds?
14. Does local-PCA anisotropy improve first-1,000-step heldout PSNR?
15. Does perturbing seed depth expose excessive dependence on external geometry?

#### Representation

16. Does harmonic motion lower heldout temporal error versus linear motion?
17. Does constant RGB cap heldout quality relative to a small view-dependent basis?
18. Does fixed covariance over time explain moving-object blur?
19. Does temporal gating reduce ghosting without starving gradients?
20. Can a calibrated dynamic-3DGS backend beat the current trajectory model at equal parameters?

#### Raster And Backward

21. Does tiled forward match a CPU compositor within a declared image tolerance?
22. Does every WGSL parameter gradient match finite differences on a tiny fixture?
23. Is tile overflow zero across the complete training schedule?
24. Does dynamic depth sorting match CPU order for every camera/time fixture?
25. Does pair-owned backward equal a slower pixel-owned reference?

#### Objective

26. Does browser box SSIM match its CPU analogue in value and RGB gradient?
27. Does canonical Gaussian SSIM improve heldout quality at matched wall time?
28. Does `0.2 * (1-SSIM)` outperform L1-only after equal seconds, not equal steps?
29. Does 7x7 lose measurable heldout quality versus 11x11?
30. Does precomputing target moments preserve objective values exactly?

#### Optimizer And Topology

31. Does parameter-family Adam beat one shared learning rate?
32. Which learning-rate multiplier maximizes heldout PSNR by a fixed wall time?
33. Does split/recycle beat fixed topology at equal capacity and time?
34. Does spatial separation outperform temporal separation for recycled children?
35. Does a prune threshold improve speed without reducing heldout quality?

#### Validation

36. How far can sparse-grid PSNR deviate from full-image PSNR?
37. How far can global-luma SSIM deviate from windowed RGB SSIM?
38. Does asynchronous snapshot validation measurably reduce live throughput?
39. Are validation results invariant to preview state?
40. Does `cam06` ranking predict performance on other heldout cameras?

#### Performance

41. Does 192x144 retain at least half of 96x72 step rate at each splat count?
42. Does steps/s decline monotonically over 768, 1,536, and 4,096 splats?
43. Which phase consumes more than 25% of a 4,096-splat step?
44. Does separable Gaussian SSIM reduce SSIM phase time by at least 3x?
45. Does paging targets make 384x288 pass without changing training math?

#### Evidence And Decisions

46. Can one command reproduce the scaling JSON within 15%?
47. Does any current result satisfy a predeclared baseline graduation gate?
48. Does native 4DGS solve a measured current failure that dynamic 3DGS cannot?
49. Which proposed experiment has the greatest expected information per GPU minute?
50. Which result would justify stopping browser research and returning to the canonical lane?

### Loop 2 Critical Review

Good:

- Most questions now produce a pass, fail, or bounded uncertainty.
- Matched wall time appears alongside matched steps where optimization speed
  matters.
- Correctness and quality are separated from throughput.
- The questions expose whether an external representation is actually needed.

Bad:

- A one-variable test can miss interactions such as resolution by splat count
  or SSIM by topology.
- Thresholds like 15% and 25% are provisional rather than derived from use.
- Fifty separate experiments would waste setup time and inflate false positives.
- Some “beat” questions still lack a minimum practically important quality gain.
- Coffee Martini and one M4 remain overrepresented.

### Loop 2 Framework Adjustment

The next loop should group symptoms into a causal graph:

```text
data -> initialization -> visibility/support -> image objective
-> gradient conditioning -> optimizer/topology -> quality

resource layout -> GPU occupancy/traffic -> throughput -> wall-time quality
```

Use small factorial screens when factors plausibly interact. Require each
question to identify which causal edge it tests and what competing cause it
rules out.

### Loop 3: Causal Bottleneck Questions (50)

#### Data And Calibration Causes

1. Is plateaued heldout quality caused by calibration error or model capacity?
2. Do reprojection residuals correlate with cameras that validate poorly?
3. Does atlas quantization measurably change gradients at 96x72?
4. Are synchronized frame indices truly synchronized across all 18 cameras?
5. Does the anchor-frame transform introduce scale or handedness error?

#### Initialization Causes

6. Is early blur caused by seed positions, seed scales, or initial opacity?
7. Do poorly supported heldout regions lack training-view seed coverage?
8. Does the optimizer repair 5% depth perturbations but fail at 20%?
9. Are local-PCA scale axes aligned with actual surfaces or sampling artifacts?
10. Is the early pretty 2D baseline explained entirely by target-pixel leakage?

#### Visibility And Raster Causes

11. Is streaking caused by wrong depth order or oversized covariance?
12. Is sparsity caused by opacity collapse or points leaving all cameras?
13. Are tile overflows silently dropping high-support contributors?
14. Does transmittance early termination remove useful low-alpha gradients?
15. Do 3-sigma support cuts omit gradients needed to grow coverage?

#### Objective Causes

16. Does L1 favor median-color blur under the current representation?
17. Does box SSIM over-smooth because 11 pixels span 15% of image height?
18. Does local SSIM conflict with sparse topology relocation?
19. Is heldout plateau hidden by the sparse global SSIM proxy?
20. Does black-background composition create an avoidable objective mismatch?

#### Gradient Causes

21. Which parameter family receives vanishing gradients after the early descent?
22. Do quaternion gradients disappear when scales remain near isotropic?
23. Are center gradients correctly scaled after geometry normalization?
24. Do temporal-center and velocity gradients disagree in sign on moving regions?
25. Does the shared backward omit gradients behind early alpha saturation?

#### Optimizer Causes

26. Is the plateau caused by learning rates that are too low or too high?
27. Are Adam second moments freezing parameters after large initial gradients?
28. Would per-family warmup improve early structure without later instability?
29. Does opacity saturation prevent geometry from continuing to move?
30. Does gradient clipping hide a scale or coordinate bug?

#### Topology Causes

31. Are recycled parents selected by useful screen-space error?
32. Are weak slots actually weak across cameras and times, or only in one sample?
33. Does recycling destroy useful rare-time support?
34. Does fixed capacity prevent growth in uncovered image regions?
35. Are child offsets large enough to escape their parent support?

#### Representation Causes

36. Does one covariance across time force motion blur?
37. Does one RGB across views force color averaging?
38. Does one sinusoid fail nonperiodic or articulated motion?
39. Does a scalar temporal gate conflate visibility and deformation?
40. Would independent per-frame Gaussians establish a higher-capacity oracle?

#### Systems Causes

41. Is live throughput loss caused by preview, metrics, validation, or queue probes?
42. Does a 32-step queue hide control updates or inflate apparent burstiness?
43. Is SSIM compute bound or storage-bandwidth bound?
44. Which allocation grows quadratically in raster or tile capacity?
45. Does worker message transfer copy any large tensor unexpectedly?

#### Research-Lane Causes

46. Is the browser blocked by WebGPU limitations or by the current algorithm?
47. Would the same model plateau under the native Metal renderer?
48. Is a browser-specific optimization useful to the paper lane?
49. Does implementing native 4DGS answer a measured causal uncertainty?
50. Which cause can be falsified most cheaply without adding a new backend?

### Loop 3 Critical Review

Good:

- Questions distinguish observed symptoms from plausible mechanisms.
- Competing explanations are explicit.
- Several cheap diagnostics can prevent expensive representation work.
- The model and systems causal chains are treated separately.

Bad:

- Correlation questions can be mistaken for causal proof.
- The list still assumes all factors are independently observable.
- Some diagnostics require instrumentation not yet present.
- A causal graph can become an excuse for endless profiling.
- Questions do not yet rank by downstream decision value.

### Loop 3 Framework Adjustment

Assign each candidate:

- expected information value;
- implementation hours;
- benchmark minutes;
- risk of misleading proxy results;
- number of later decisions unlocked;
- whether the result transfers to the canonical trainer.

Only retain questions on the Pareto frontier of information and cost.

### Loop 4: Decision-Portfolio Questions (50)

#### Tier 0: Freeze And Correctness

1. Can the current state ledger be generated and checked in one command?
2. Can a tiny CPU/WGSL forward fixture pass before any quality run?
3. Can all trainable WGSL gradients pass a readback parity gate?
4. Can tile overflow and order errors fail loudly rather than degrade silently?
5. Can validation prove heldout exclusion mechanically?

#### Tier 1: Timing

6. Can timestamp queries separate ten GPU phases with under 3% overhead?
7. Can one scaling harness sweep raster, splats, and SSIM window?
8. Can target-buffer memory be estimated before allocation?
9. Can live versus isolated overhead be decomposed by toggling one service?
10. Can a matched Metal harness execute the same mathematical step?

#### Tier 1: Quality Diagnostics

11. Can full-image heldout metrics run without pausing training?
12. Can moving-region and static-region errors be reported separately?
13. Can gradient norms and update norms be logged per parameter family?
14. Can support, opacity, scale, and aspect histograms reveal collapse?
15. Can temporal error be plotted for all 16 frames?

#### Tier 2: Initialization

16. Does SfM versus random init change 60-second heldout PSNR by at least 2 dB?
17. Does local-PCA versus isotropic init change early convergence?
18. Does opacity 0.05, 0.1, or 0.2 interact with topology?
19. Does seed count or final capacity explain more variance?
20. Which init ablation can reuse one compiled pipeline?

#### Tier 2: Objective

21. Does canonical Gaussian SSIM improve heldout PSNR or SSIM at equal time?
22. Does L1-only reach better PSNR because SSIM cost reduces step count?
23. Does 7x7 versus 11x11 change moving-edge quality?
24. Does precomputed target-moment SSIM preserve exact gradients?
25. Is LPIPS valuable for evaluation before it is considered for training?

#### Tier 2: Optimizer

26. Which three learning-rate multipliers cover the plausible stable range?
27. Does cosine decay beat a fixed rate by 60 seconds?
28. Does opacity reset unblock geometry after a plateau?
29. Does per-family Adam epsilon matter at browser precision?
30. Can update-to-parameter ratios define an automatic LR sanity gate?

#### Tier 2: Topology

31. Does fixed topology versus recycle produce a visible difference by 30 seconds?
32. Does selecting parents by screen gradient beat contribution alone?
33. Does selecting weak slots across a schedule beat one-frame statistics?
34. Does spatial versus temporal child separation address different residuals?
35. Does topology improve heldout quality per millisecond of added work?

#### Tier 3: Representation

36. What quality ceiling does independent per-frame dynamic GS establish?
37. Does calibrated dynamic 3DGS close the current plateau?
38. Does native 4D covariance add value after dynamic 3DGS is tuned?
39. Does World Tubes improve shared temporal efficiency at equal quality?
40. Which backend can reuse the current tiled raster and worker contracts?

#### Robustness

41. Does the best setting survive three random seeds?
42. Does it survive a different heldout camera?
43. Does it survive a second dynamic scene?
44. Does throughput ranking survive a second WebGPU device class?
45. Does quality ranking survive 96x72 and 192x144?

#### Stop And Promotion

46. What failure stops objective work and redirects to gradient correctness?
47. What failure stops LR tuning and redirects to representation capacity?
48. What result rejects topology complexity?
49. What evidence graduates the browser to baseline status?
50. What result archives the browser as a demo and ends further research investment?

### Loop 4 Critical Review

Good:

- Dependencies and tiers reduce wasted work.
- Questions now unlock later decisions.
- Stop conditions appear beside promotion conditions.
- Robustness enters before a baseline claim.

Bad:

- Tier labels may encode current bias rather than measured value.
- Full factorial robustness can exceed the browser demo's intended scope.
- A matched Metal harness is valuable but not necessarily cheap.
- “At least 2 dB” may be too coarse for short runs.
- The portfolio still lacks adversarial checks for leakage and metric gaming.

### Loop 4 Framework Adjustment

Add an adversarial reviewer to every proposed claim. The reviewer must ask:

- Can initialization explain the gain?
- Can heldout tuning explain the gain?
- Can a proxy metric hide a worse full image?
- Can hidden synchronization explain the speed?
- Are workloads and objectives matched?
- Did a failed allocation silently reduce work?
- Did documentation merge results from different generations?

### Loop 5: Adversarial Questions (50)

#### Leakage

1. Could heldout pixels enter the SfM point cloud used for initialization?
2. Was `cam06` used when choosing hyperparameters repeatedly?
3. Does checkpoint selection inspect heldout quality?
4. Did an early target-pixel initializer masquerade as 3D convergence?
5. Are train and heldout camera names ever reordered during export?

#### Workload Matching

6. Does a steps/s comparison include the same splats, pixels, and objective?
7. Does “full frame” include all loss and update phases?
8. Does one backend skip topology or metrics inside the timed interval?
9. Is capacity different from active splat count?
10. Are warmup, repeats, queue drain, and device contention recorded?

#### Correctness Attacks

11. Can a zero-pair indirect dispatch make a fast broken step look valid?
12. Can an invalid bind group leave stale output that appears successful?
13. Can tile overflow counters remain unread while splats are dropped?
14. Can front-to-back and back-to-front conventions cancel in one view only?
15. Can CPU finite differences pass while WGSL indexing is wrong?

#### Metric Gaming

16. Can sparse-grid PSNR miss thin moving structures?
17. Can global-luma SSIM reward spatially misplaced content?
18. Can black background dominate loss and hide dynamic failure?
19. Can a support guard inflate coverage while worsening RGB?
20. Can a denser blurry field improve SSIM but reduce usable geometry?

#### Initialization Advantage

21. Does external SfM encode information unavailable to an in-browser baseline?
22. Is train-only visibility filtering actually train-only?
23. Does farthest-point selection favor the heldout view accidentally?
24. Are colors copied from a point cloud reconstructed with all cameras?
25. Would a fair random-init method require more steps than the demo budget?

#### Systems Attacks

26. Does queue depth make submitted steps look completed?
27. Can UI polling trigger GPU readback stalls?
28. Can multiple open trainer tabs contaminate timing?
29. Does device recreation warm shader caches across repeats?
30. Does thermal or power state create the observed repeat trend?

#### Objective Attacks

31. Is the browser's “SSIM” actually canonical Gaussian SSIM?
32. Is `DSSIM` normalized consistently across browser and Python code?
33. Does reflected padding inflate border agreement?
34. Does 11x11 box averaging over-smooth a 72-pixel-high image?
35. Can a smaller window look faster only because it weakens the gradient?

#### Topology Attacks

36. Does recycle count activity without improving primitive placement?
37. Are parent statistics biased toward the latest camera/time?
38. Can recycling delete rare heldout-relevant geometry?
39. Does fixed allocation hide the absence of real pruning?
40. Can inactive low-opacity splats still consume most compute?

#### Representation Attacks

41. Is “World Tubes” only a label over sinusoidal motion?
42. Is the DynamicGs probe too partial to be a baseline?
43. Would native 4DGS add parameters rather than solve the observed cause?
44. Does constant color make geometry look worse than it is?
45. Does fixed covariance make motion look worse than it is?

#### Claim Attacks

46. Is any current quality number committed as a baseline row?
47. Can another person reproduce the visible run from a command?
48. Are source, artifact, and conclusion linked in the same document?
49. Is “solid baseline” defined before or after seeing results?
50. What evidence would force us to retract the strongest current claim?

### Loop 5 Critical Review

Good:

- It catches the largest risks in the current history: leakage-like init,
  unmatched speed, stale generations, proxy metrics, and incomplete backends.
- It treats surprisingly fast results as a correctness question.
- It makes “not yet a baseline” an evidence statement rather than pessimism.

Bad:

- Adversarial questioning can block every claim indefinitely.
- Several attacks are possible in principle but unlikely in this exact code.
- The list can reward defensive tests over useful model progress.
- No maximum evidence burden is set for a browser prototype.
- A skeptical list still does not schedule the work.

### Loop 5 Framework Adjustment

Use bounded skepticism:

- one cheap falsification test for each high-risk claim;
- one stronger test only if the cheap test fails or remains ambiguous;
- explicit acceptable residual risk;
- no paper-level burden for UI-only behavior;
- paper-level burden for quality, generalization, or representation claims.

Then convert surviving questions into dependency-ordered work packages.

### Loop 6: Execution-Ready Questions (50)

#### Work Package A: State And Reproduction

1. What script emits the frozen state ledger?
2. What command starts the isolated server?
3. What URL runs the synchronized benchmark?
4. What tests guard the export and browser contracts?
5. What exact artifact records the July 28 result?

#### Work Package B: GPU Correctness

6. What tiny fixture exercises overlapping depth-sorted splats?
7. Which rendered channels should be read back for parity?
8. Which parameter subset gives complete gradient-family coverage?
9. What absolute and relative tolerances fit f32 WGSL?
10. What failure message identifies tile, camera, time, and parameter?

#### Work Package C: Phase Timing

11. Which timestamp-query feature is available on target browsers?
12. Where should timestamps bracket each compute pass?
13. How many warmups and repeats stabilize the M4 result?
14. How should live-service overhead be measured independently?
15. What JSON schema stores phase and memory results?

#### Work Package D: Target Paging

16. Should the GPU hold one camera/time image, one camera, or a small ring?
17. How is the next target uploaded without blocking queued training?
18. Can target-only SSIM moments share the same paging unit?
19. What maximum raster passes the revised binding contract?
20. Does paging preserve exact target offsets and schedule order?

#### Work Package E: Canonical SSIM

21. What normalized 11-tap Gaussian coefficients are used?
22. Which five moment images need separable filtering?
23. Which intermediate derivatives are required for transpose backward?
24. Can target-only moments be precomputed per page?
25. What CPU/WGSL value-and-gradient fixture closes parity?

#### Work Package F: Quality Ablations

26. What fixed wall-clock budget is long enough to distinguish settings?
27. Which metrics are computed on full heldout images?
28. Which three seeds are deterministic and recorded?
29. Which ablations fit a small factorial design?
30. What endpoint and curve artifacts are saved?

#### Work Package G: Plateau Diagnosis

31. Which per-family gradient and update quantiles are logged?
32. Which moving/static region errors are plotted?
33. Which intervention distinguishes capacity from bad placement?
34. Which intervention distinguishes loss conflict from representation bias?
35. What threshold declares the plateau cause unresolved?

#### Work Package H: Topology

36. What fixed-topology flag disables all maintenance?
37. What statistics select parents and weak slots?
38. How are statistics aggregated across camera/time pairs?
39. What matched run measures quality per topology millisecond?
40. What result rejects further topology complexity?

#### Work Package I: Representation

41. What is the minimal complete calibrated dynamic-3DGS backend?
42. Which contracts can it reuse from `tiled3d`?
43. What oracle shows the current trajectory model lacks capacity?
44. What result would justify native 4D covariance?
45. What result would justify a World Tubes shared-temporal backend?

#### Work Package J: Graduation

46. What correctness gates must all be green?
47. What minimum heldout metrics and robustness are required?
48. What speed and memory envelope is acceptable?
49. What documentation and reproduction command are mandatory?
50. Who or what verifier declares the baseline gate passed?

### Loop 6 Critical Review

Good:

- Every question maps to a file, command, test, or artifact.
- Dependency order is visible.
- The list distinguishes prototype completion from research graduation.
- It gives future agents concrete entry points.

Bad:

- Fifty work-package questions are still too many for immediate execution.
- Several packages depend on the same foundational parity and timing tools.
- The list does not choose between quality work and new representation work.
- Graduation thresholds remain undefined until a reference baseline is run.
- Without a final reduction, this becomes a backlog rather than a strategy.

### Loop 6 Final Framework Adjustment

Reduce to ten questions. Each final question must:

- unlock at least two downstream decisions;
- have a named artifact;
- have a stop condition;
- state the current provisional answer;
- avoid creating another renderer lane unless earlier evidence demands it.

### Loop Rerun Outcome At `9cfe24d`

The five refinement passes were rerun against the post-paging source state.
This matters because several Loop 6 questions have changed status:

| Gate | State | Consequence |
| --- | --- | --- |
| Canonical data/split adapter | green | reuse it; do not create browser split semantics |
| One-frame GPU target paging | green for `tiled3d` | 384x288 binding blocker is closed |
| Packed-FP16 checkpoint storage | green | keep as default; retain FP32 math |
| 4,096 tile capacity and 2D indirect dispatch | green | former silent truncation/dispatch ceilings are closed |
| Complete CPU/WGSL image parity | red | blocks strong correctness claims |
| Complete CPU/WGSL gradient/update parity | red | blocks LR and convergence conclusions |
| Phase timing | red | blocks the next kernel choice and Metal comparison |
| Full-image heldout metrics | red | blocks quality promotion |
| Host RGBA8/shared target ownership | red | blocks normal high-resolution SPA use |
| Canonical separable Gaussian SSIM | red | blocks objective parity with 3DGS |
| Fixed versus recycle topology ablation | red | topology benefit is unknown |
| Complete dynamic-3DGS/native-4DGS browser baseline | red | blocks representation ranking |
| Tiled/sampled runtime separation | amber | code and portability debt, not a quality blocker |

This rerun changes the immediate sequence:

```text
GPU value/gradient parity
-> full-image metrics plus per-family update diagnostics
-> phase timing and host target ownership
-> small convergence screen
-> canonical SSIM/topology ablations
-> representation decision
```

The process rejected three tempting shortcuts:

1. Do not broaden FP16 into optimizer or covariance math before a measured
   memory/throughput need and a convergence parity run.
2. Do not tune learning rates from sparse proxy curves before WGSL update
   parity and dimensionless update ratios are visible.
3. Do not wire every shader into the SPA before one complete dynamic baseline
   establishes the residual capacity gap.

## 8. Final Set Of Ten Powerful Questions

### 1. What Exactly Is The Current Browser Training Contract?

Why it matters:

Every later speed, quality, and representation claim depends on identifying the
same system. This question prevents historical metric mixing.

Possible subquestions:

- What commit, asset tag, backend, and parameter schema are active?
- What cameras, frames, raster, splats, and capacity are used?
- What exactly happens in one optimizer step?
- Which objective, initialization, and topology schedule are active?
- Which controls affect this backend?

Artifact:

- generated state ledger plus resolved bundle and backend metadata.

Stop condition:

- one command emits the ledger and a test checks it against the UI defaults.

### 2. Are Tiled Forward, Visibility, And Analytic Backward Correct?

Why it matters:

Fast convergence work is meaningless if a projection, compositing, index, or
gradient error is present. Surprisingly high throughput should increase, not
decrease, the demand for this gate.

Possible subquestions:

- Does rendered RGB/alpha match a slow CPU reference?
- Does tile order match camera depth with stable tie breaking?
- Is every trainable parameter gradient within f32 tolerance?
- Are reflected borders and transmittance checkpoints correct?
- Is overflow always zero, and does nonzero overflow fail loudly?

Artifact:

- tiny fixture JSON, CPU outputs, WGSL readbacks, and a parity test report.

Stop condition:

- all value and gradient families pass declared tolerances on at least two
  overlapping-splat fixtures.

### 3. Where Do End-To-End Time And Memory Go, And How Do They Scale?

Why it matters:

The July 28 matrix shows useful aggregate scaling, and paging removes the old
GPU target-buffer wall. It still cannot identify the next kernel optimization,
the best checkpoint-memory policy, or a matched Metal claim.

Possible subquestions:

- What are bin, sort, raster, SSIM, backward, update, and topology times?
- What do preview, metrics, validation, and completion probes cost live?
- How do phases scale over raster by splat count?
- What is actual peak allocated, bound, and host-resident memory?
- Should packed-FP16 savings buy denser checkpoints or halve checkpoint memory?
- How many target copies exist across the main, train-worker, and validation
  paths, and can canonical RGBA8 or worker-owned decode remove them?
- What sampled-only pipelines and buffers does the tiled subclass still create?
- Can an identical mathematical harness run through Metal?

Artifact:

- GPU timestamp, device-memory, host-memory, and checkpoint-policy JSON for
  WebGPU, plus a separately labeled matched Metal result if implemented.

Stop condition:

- at least 90% of step time and 95% of resident bytes are assigned to measured
  owners, and repeat variance is below a declared bound.

### 4. Is Initialization The Dominant Quality Advantage Or Blocker?

Why it matters:

The current run starts from strong external SfM geometry. That is useful for a
demo, but it can hide model weakness and complicate baseline fairness.

Possible subquestions:

- How do SfM, random, perturbed-SfM, and train-only alternatives compare?
- Does local-PCA anisotropy help?
- Was the external source PLY reconstructed with the camera later called
  heldout, even though export-time visibility filtering uses train cameras?
- Can a train-camera-only seed cloud be generated under the canonical data
  contract?
- Does seed count or placement matter more than final capacity?
- How much of the early visual quality exists at step zero?

Artifact:

- matched initialization sweep with step-zero renders, equal-time curves, and
  explicit provenance.

Stop condition:

- confidence intervals identify whether initialization changes the preferred
  model or only convergence time.

### 5. What Causes The Observed Convergence Plateau?

Why it matters:

One million steps without clear structure means step count is not the scarce
resource. More tuning should target a diagnosed mechanism.

Possible subquestions:

- Are gradient or update norms vanishing by parameter family?
- Are dimensionless update-to-parameter ratios sensible for position, scale,
  rotation, color, opacity, and motion?
- Is active support collapsing or saturating?
- Does more capacity or raster detail help when placement is fixed?
- Does fixed topology versus recycle change the plateau?
- Do independent per-frame Gaussians establish a much higher oracle?
- Does the plateau persist in full-image heldout metrics, or only in the sparse
  UI proxy?

Artifact:

- plateau trace with per-family gradients/updates, support histograms, region
  errors, and a small set of discriminating interventions.

Stop condition:

- one cause or bounded combination explains the plateau and predicts an
  intervention that improves heldout quality in a repeat.

### 6. Does Windowed SSIM Improve Quality Enough To Justify Its Cost?

Why it matters:

SSIM is standard in 3DGS, but the current browser uses a noncanonical box
window. At 96x72, an 11x11 window also spans a large fraction of the image.
Window size, weighting, implementation, genuine detail, and wall-time cost are
separate questions.

Possible subquestions:

- How do L1-only, 7x7 box, 11x11 box, and 11x11 Gaussian compare?
- Are comparisons matched by steps and wall time?
- Does Gaussian weighting improve moving edges rather than blur?
- Can separable filtering and target moment caching recover the cost?
- Does CPU/WGSL value and gradient parity hold?

Artifact:

- objective parity fixture, phase timing, and matched quality curves.

Stop condition:

- promote the fastest objective on the heldout quality/time Pareto frontier;
  otherwise retain canonical Gaussian SSIM for comparability.

### 7. Does Split/Recycle Topology Improve The Quality-Speed Frontier?

Why it matters:

Dynamic maintenance now exists, but implementation activity is not evidence of
benefit. At default capacity it recycles rather than increases primitive count.

Possible subquestions:

- Does maintenance beat fixed topology?
- Are parent and weak-slot statistics representative across cameras and times?
- Is spatial or temporal separation useful?
- Are rare supports destroyed?
- What is quality gain per added millisecond?

Artifact:

- fixed/recycle/spatial/temporal matched ablation with utilization and support
  traces.

Stop condition:

- retain only a topology mode that improves heldout quality at acceptable cost
  across repeats.

### 8. Which Representation Deserves A Complete Browser Backend Next?

Why it matters:

The repository contains useful probes, but turning every shader into a selector
would create misleading choices and duplicate contracts.

Possible subquestions:

- What ceiling does the current trajectory model reach?
- What ceiling does complete calibrated dynamic 3DGS reach?
- Does Fudan native 4DGS, with a full 4D covariance and conditional 3D render,
  solve a failure that dynamic 3DGS leaves?
- Is the Wu et al. HexPlane/deformation method being mislabeled as the same
  "4D-GS" baseline, despite being a different representation?
- Does World Tubes provide measurable temporal sharing at equal quality?
- Which implementation can reuse the current tiled raster, worker, and data path?

Artifact:

- parameter- and protocol-matched oracle table before any new selector.

Stop condition:

- complete dynamic 3DGS first; promote native 4DGS or World Tubes only after a
  measured residual capability or efficiency gap remains.

### 9. Do Improvements Survive Novel Views, Time, Seeds, Scenes, And Devices?

Why it matters:

Coffee Martini with one heldout camera on one M4 can support a demo, not a
general claim.

Possible subquestions:

- Do results hold for three seeds?
- Do they hold with another heldout camera?
- Do they hold on a second dynamic scene?
- Is error concentrated at particular times?
- Does throughput ranking hold on another WebGPU class?

Artifact:

- compact robustness matrix with full-image PSNR, Gaussian-window SSIM, LPIPS,
  L1, wall time, memory, and rendered FPS.

Stop condition:

- effects keep direction and practical magnitude across the declared scope, or
  claims are narrowed explicitly.

### 10. What Evidence Graduates The Demo Into A Solid Baseline?

Why it matters:

“Solid baseline” must be a predeclared gate, not a feeling after a pleasing
render.

Possible subquestions:

- Which correctness gates are mandatory?
- Which external or internal baseline must be matched?
- What quality, speed, memory, and robustness thresholds apply?
- What commands and artifacts reproduce the result?
- Is the tiled runtime separated cleanly from sampled-ray pipelines and buffers?
- Do requested WebGPU limits respect portable minima or explicitly report the
  supported-device envelope?
- Are `PROJECT_INDEX.md`, `README.md`, `TODO/README.md`, `EXPERIMENTS.md`, and
  `BASELINES.md` synchronized with the implementation?
- Which failures keep the system in prototype status?

Artifact:

- one completion verifier tied to checked-in result JSON and an appended
  `BASELINES.md` row.

Stop condition:

- the verifier passes without waivers on a clean checkout.

## 9. Answers To The Final Questions

### Answer 1: Current Contract

The active source state is `9cfe24d`. Its resolved contract is:

- Coffee Martini has 18 selected cameras: 17 train cameras and `cam06` held
  out.
- Sixteen synchronized times are selected, giving 272 train camera/time pairs.
- Training uses a 96x72 raster and 4,096 requested splats at 4,096 capacity.
- Initialization takes 4,096 XYZRGB points from the external Ex4DGS
  `input.ply`, filters for train-camera visibility, and transforms them into
  `cam04` OpenCV coordinates.
- `tiled3d` is the default full-frame backend. One complete camera/time image
  is optimized per Adam step, while the scheduler covers all 272 pairs.
- Each primitive has anisotropic 3D scale and rotation, RGB, opacity, temporal
  support, and linear or linear-plus-sinusoidal center motion.
- The objective is `0.8 L1 + 0.2 (1 - 11x11 box SSIM)`.
- The effective default learning rates are `4.375e-4` center, `1.875e-3`
  color, `1.0e-3` opacity, `2.5e-4` motion, `5.625e-4` scale, and
  `3.125e-4` rotation.
- Allocation is fixed at 4,096. GPU maintenance can fill unused slots and
  recycle weak slots through step 60,000; at 4,096/4,096 it can only recycle.
- Stored forward checkpoints use packed FP16 by default. Parameters,
  projection, compositing, loss arithmetic, gradients, and Adam state remain
  FP32.
- Optimization and validation run in separate workers. The UI streams a
  looping three-camera preview without intentionally synchronizing every step.

The selected model is therefore a compact dynamic 3D Gaussian trajectory
model, not Fudan native 4DGS and not World Tubes. The README and result
artifacts describe the contract, but one generated, machine-checked state
ledger tied to UI defaults is still missing.

### Answer 2: Correctness

Correctness is partially established, not closed.

Established:

- CPU anisotropic projection and VJP tests cover center, scale, and quaternion.
- CPU windowed L1/SSIM analytic gradient passes finite differences.
- source and runtime tests protect full-frame scheduling, target-page ordering,
  tile capacity, two-dimensional indirect dispatch, exact stop-rank
  conversion, sorting structure, and shared backward.
- the post-paging browser gate currently passes 53 focused tests.
- live runs show finite loss and parameter motion with zero observed overflow.

Missing:

- CPU versus WGSL full rendered RGB/alpha parity;
- CPU versus WGSL per-parameter gradient readback parity;
- readback parity for one complete Adam update, not only CPU helpers;
- adversarial fixtures with overlapping, nearly opaque, clipped, and
  equal-depth splats;
- explicit failure on all resource and overflow conditions.

The recent capacity, dispatch, paging, and index fixes remove known structural
errors. They do not prove the WGSL calculus. No strong convergence comparison
should bypass the GPU value, gradient, and update parity gate.

### Answer 3: Timing, Scaling, Memory, And Precision

The synchronized Apple M4 FP32-checkpoint matrix is now:

| Raster | 768 | 1,536 | 4,096 |
| --- | ---: | ---: | ---: |
| 96x72 | 1,233 | 863 | 359 steps/s |
| 192x144 | 675 | 462 | 240 steps/s |
| 384x288 | 266 | 182 | 118 steps/s |

At 4,096 splats, packed-FP16 checkpoints reach 403, 294, and 132 steps/s
at the same three rasters. At 384x288, loss after 1,024 submissions differs
from the FP32-checkpoint run by only `1.13e-6`. This is encouraging storage
precision evidence, not a complete long-run quality equivalence test.

Two earlier intuitions need correction:

1. Resolution is not almost free. At 4,096 splats, each 4x increase in pixels
   retains about 67% and then 49% of FP32 throughput.
2. The 384x288 GPU target limit is fixed. One 1.69 MiB RGBA32F target page
   replaces the former 486 MiB all-target GPU binding.

The main 384x288/4,096 GPU allocations are approximately 108 MiB of forward
checkpoints, 13.5 MiB of pair IDs/references, 1.69 MiB of target page, and
0.375 MiB of FP32 gradient accumulation. The packed mode currently spends its
storage saving on checkpoints twice as frequently, reducing replay while
keeping the checkpoint allocation near 108 MiB. A memory-first policy that
keeps the old checkpoint spacing should reduce that allocation toward 54 MiB;
that is an inference and needs a measured mode.

The remaining large target-memory issue is on the host. A decoded 384x288
Float32 all-target tensor is about 486 MiB and may be duplicated across main,
training-worker, and validation-worker ownership. Canonical RGBA8 storage,
shared transfer where available, or worker-owned decode is the next memory
change. Raising the GPU storage-buffer limit would restore the wrong design.

Live 96x72/4,096 observations around 250-280 steps/s remain 30-38% below the
isolated packed result. We do not yet know how much belongs to raster,
checkpoint replay, SSIM, Adam, target transfer, preview, metrics, or
validation. GPU timestamps, host-memory accounting, and removal of inherited
sampled-only resources are needed before another optimization.

There is still no mathematically matched full-step Metal result. Existing
Metal probes cannot justify the claim that WebGPU is only 20-30% slower.

### Answer 4: Initialization

The active run is not random and is not initialized perfectly on target pixels.
It uses external SfM points and point colors, applies export-time
training-camera visibility filtering, and transforms them into the canonical
anchor frame. Initial anisotropy comes from local neighborhoods; opacity
begins at 0.1.

There is an important fairness caveat: the source PLY is an external Ex4DGS
artifact. The bundle records train-camera filtering, but its provenance does
not prove that `cam06` was excluded when the source reconstruction itself was
created. The image split is honest; the initialization may still contain
privileged heldout-camera geometry. A train-camera-only reconstruction is
required for a strict novel-view baseline.

The earliest visually flattering browser lane used image-space target-grid or
target-pixel color initialization and a temporal-mean background. That was much
closer to a source-view memorization prior and is not an honest 3D baseline.

The correct ablation is step-zero plus equal-wall-time comparison of external
SfM, train-only SfM, perturbed SfM, and random initialization, with identical
capacity and seeds. Until that exists, we do not know how much early quality
comes from the representation versus the seed cloud.

### Answer 5: Plateau

The plateau and rough loss curves are observed but not diagnosed. A curve that
mixes 272 heterogeneous camera/time tasks will naturally be noisier than a
single-view overfit, so sample loss must be separated from a fixed full-image
evaluation set. Plausible causes include:

- unverified WGSL gradients or poorly scaled updates in one parameter family;
- only 4,096 primitives for a multicamera dynamic scene;
- a 96x72 training signal with no true high-resolution detail;
- constant RGB across views;
- covariance fixed over time;
- a limited linear/sinusoidal trajectory basis;
- fixed-capacity topology driven by short-horizon statistics;
- a broad uniform 11x11 box objective at low raster resolution;
- privileged but imperfect geometry initialization.

The shortest causal sequence is:

1. pass tiny WGSL value/gradient/update parity;
2. log fixed-pair full-image loss, PSNR, canonical SSIM, support, and
   dimensionless update-to-parameter ratios by family;
3. compare fixed topology with recycle under the same schedule;
4. run small matched interventions for capacity, learning-rate scale, objective,
   and trajectory basis;
5. train independent per-frame Gaussian oracles to measure the cost of temporal
   sharing separately from raster or optimizer failure.

More raw steps should not be purchased until one of those tests predicts that
steps are the limiting resource. A successful diagnosis must improve a
predeclared heldout metric in a repeat, not merely lower sampled training loss.

### Answer 6: SSIM

An 11x11 local Gaussian window with sigma 1.5 is the conventional implementation
used by official 3DGS; whole-image moments are not a suitable replacement. The
local gradient identifies where luminance, contrast, and structure disagree,
whereas a global statistic can remain similar after an object moves.

The browser currently uses an 11x11 box window. Its gradient is mathematically
correct for the CPU analogue, but WGSL parity is unverified. At 96x72, eleven
rows are 15% of the image height, and a uniform box gives distant taps the same
weight as central taps. That is much broader than an 11x11 Gaussian's effective
support and may suppress detail; this is a hypothesis, not yet an ablation
result.

An older 96x72/1,536 ablation measured 1,174 steps/s for 7x7 box versus 933 for
11x11 box, about 26% faster. Those absolute rates predate `9cfe24d`, and no
quality result exists, so they are only directional evidence that window work
matters.

The recommended implementation is separable 11x11 Gaussian SSIM with
target-only moment caching and a transpose-filter backward. That preserves the
baseline objective while reducing each 2D filter from 121 to 22 taps. Compare
L1-only, current box, and canonical Gaussian objectives at both equal steps and
equal wall time. Use a genuinely higher-resolution bundle before claiming that
an objective recovered image detail.

Primary references:

- Wang et al., SSIM: <https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf>
- Official 3DGS loss implementation:
  <https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py>
- Official 3DGS training mixture:
  <https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py>

### Answer 7: Topology

Dynamic split/recycle is implemented. Earlier statements that topology was
entirely fixed are superseded.

The allocation remains fixed. At requested counts below capacity, hidden slots
can be filled. At the default 4,096/4,096 configuration, maintenance replaces
weak slots with children of selected parents; it cannot grow beyond 4,096.
There is no canonical variable-count prune/spawn buffer and no matched evidence
that recycle improves heldout quality.

So the truthful status is: dynamic placement within fixed capacity, implemented
but not yet validated as beneficial. The first ablation should expose fixed
topology as a control, accumulate parent/weak-slot evidence across the full
camera/time schedule, and report maintenance cost, utilization, rare-time
support, and heldout quality.

Original 3DGS gradient-driven clone/split/prune is the canonical comparison.
3DGS-MCMC offers a fixed-count relocation interpretation that is especially
relevant to a browser memory budget. Neither should be copied before current
recycle earns or loses its place under a matched test.

Primary references:

- Original 3DGS: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- 3DGS-MCMC: <https://arxiv.org/abs/2404.09591>

### Answer 8: Representation

The active backend is neither native 4DGS nor World Tubes. It is a compact
dynamic 3D Gaussian approximation with optional harmonic trajectory.

Three commonly conflated baselines must remain distinct:

1. a calibrated dynamic 3DGS baseline directly optimizing time-dependent 3D
   Gaussian attributes;
2. Wu et al. 4D-GS, which combines 3D Gaussians with a HexPlane-inspired 4D
   neural voxel encoding and deformation MLP;
3. Fudan native 4DGS, which optimizes native 4D Gaussian primitives and renders
   a time-conditioned 3D Gaussian plus temporal marginal.

World Tubes is a fourth research lane with a different temporal-sharing
contract. The standalone STAR, DynamicGs, and tube shaders are useful probes,
not complete SPA options. Exposing them now would imply data, objective,
optimizer, topology, validation, and backward parity they do not have.

The next complete baseline should be calibrated dynamic 3DGS because it is the
closest controlled extension and can reuse the tiled raster, data, worker,
preview, and validation contracts. Fudan native 4DGS is a relevant external
baseline, but should become a browser option only after its 4D covariance,
conditional 3D Gaussian, temporal marginal, appearance, analytic backward, and
topology are complete. The same gate applies to World Tubes.

Primary 4DGS sources:

- Wu et al. deformation 4D-GS: <https://arxiv.org/abs/2310.08528>
- Fudan native 4DGS: <https://fudan-zvg.github.io/4d-gaussian-splatting/>
- Fudan code: <https://github.com/fudan-zvg/4d-gaussian-splatting>

### Answer 9: Robustness

Robustness is not established. Current observations come from:

- one scene;
- one designated heldout camera;
- effectively one initialization;
- one browser/GPU class;
- sparse proxy validation rather than full paper metrics.

No result should be phrased as a general dynamic-view synthesis improvement.
The minimum useful extension is three seeds, another heldout camera, one more
scene, full-image metrics, and a second WebGPU device for performance claims.
Initialization provenance and true source resolution must be held explicit in
every row.

### Answer 10: Baseline Status

The browser trainer is a substantially better prototype than it was:

- calibrated multicamera data;
- a nominal train/heldout image split with an initialization-provenance caveat;
- strong nonrandom 3D initialization;
- full-frame tiled raster;
- shared analytic backward;
- L1 plus local SSIM training;
- trainable anisotropy and motion;
- fixed-capacity fill/recycle topology;
- paged GPU targets and packed-FP16 checkpoint storage;
- nonblocking workers, live multi-view time preview, and useful charts;
- a 3x3 raster/splat scaling benchmark and 53 focused browser tests.

It is not yet a solid research baseline. The missing graduation gates are:

1. complete CPU/WGSL value and gradient parity;
2. a train-camera-only initialization provenance path;
3. canonical Gaussian SSIM and full-image paper metrics;
4. phase timing and complete host/device memory accounting;
5. a diagnosed convergence plateau;
6. matched initialization, learning-rate, capacity, objective, and topology
   ablations;
7. robustness beyond one scene, heldout camera, seed, and device;
8. a reproducible accepted row in `BASELINES.md`.

The execution portfolio is:

| Priority | Work | Decision unlocked |
| --- | --- | --- |
| P0 | Tiny CPU/WGSL RGB, alpha, gradient, and Adam-update parity | Whether convergence tuning is meaningful |
| P0 | Fixed full-image validation set with PSNR, Gaussian SSIM, and L1 | Whether visual quality actually improves |
| P1 | GPU phase timestamps plus host/device resident-byte accounting | Which speed or memory optimization to implement |
| P1 | Per-family gradient and dimensionless update telemetry | Whether LR or gradient flow causes the plateau |
| P1 | Canonical separable Gaussian SSIM with target caching | A comparable and potentially faster objective |
| P1 | Canonical RGBA8/shared or worker-owned target decode | High-resolution operation without host multiplication |
| P2 | Matched screen of initialization, LR scale, capacity, topology, and trajectory | The first causal quality intervention |
| P2 | Fixed-count recycle versus canonical density-control comparison | Whether dynamic topology earns its complexity |
| P2 | Split sampled and tiled runtimes into sibling implementations | Remove dead resources and reduce correctness risk |
| P2 | Portable-limit mode or explicit supported-device report | Honest WebGPU portability |
| P3 | Complete calibrated dynamic 3DGS baseline | A controlled representation ceiling |
| P3 | Native 4DGS and World Tubes only after oracle evidence | Avoid decorative backend selectors |
| P3 | Multi-seed, multi-heldout, multi-scene, multi-device matrix | Scope the claim |
| P4 | Completion verifier, `BASELINES.md` row, and synchronized project docs | Graduate from demo to baseline |

This order improves detail, convergence, speed, memory, and code quality without
confounding them. New representation work should begin only after parity,
metrics, and diagnostics identify a residual capability gap rather than an
implementation defect.
