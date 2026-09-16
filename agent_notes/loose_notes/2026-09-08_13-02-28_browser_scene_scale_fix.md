# Browser geometric unit correction

Continues the September 6 source-unit counterexample. Changed the actual
initializer and sampled/tiled Gaussian training configuration together.

The dataset retains `geometryScale` for external pose/point conversion. Its new
`trainingSceneScale` uses the existing cameraRigRadius function on normalized
training cameras only: 1.1 times maximum center distance from their mean.
Coincident cameras use the existing fallback of 1. This heuristic is explicit;
it is not claimed optimal across rig designs.

Local PCA minimum/maximum sizes use 0.03/0.60 times trainingSceneScale. Shader
uniforms retain their ABI name but receive this scene length for size limits,
velocity/harmonic bounds, and tiled geometric consistency/variance scaling.
Preview poses and continuation's coordinate-conversion identity retain the
original conversion. Removed clipping of the inverse median depth: source
meters/millimeters must normalize identically as well.

Regression test exercises identical initialization at source multipliers
0.001, 0.1, 1, 10, 1000 and verifies heldout camera placement cannot alter the
training scene length. All 206 browser node tests pass. The audit now includes
camera roles explicitly and reports the actual new bounds. Existing September
6 JSON is retained unchanged as historical measurement.

Six real-bundle CPU cases are retained in the adjacent JSON. Deep3D's three unit
variants agree: 84/4096 fully capped splats, median aspect 2.84948, projected
sigma 0.749274 px, mean alpha 0.190191. Baseline was 3541 capped, aspect 1,
sigma 0.344917 px, alpha 0.049240. Coffee also becomes invariant, but its larger
rig raises the initial minimum size and changes its aspect distribution. That
is a reason to evaluate quality, not claim a blanket convergence improvement.
Alpha excludes temporal gating and Mip compensation, as recorded in JSON.

GPU preflight command failed before launch because Bun could not resolve
`puppeteer` from run_headless_kernel_benchmark.js. No GPU timing, optimizer-step
invariance, image-quality comparison, UI verification, or deployment is claimed.
The next runtime check is a fresh Deep3D/Coffee fit with drained step timing,
tile overflow, and fixed train/heldout RGB metrics. Increasing footprints can
increase raster work. Existing uncommitted World Tubes/paper changes were kept.
