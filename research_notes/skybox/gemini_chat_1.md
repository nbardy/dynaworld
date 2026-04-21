User:

What is the typical way to handle horizons or skybox /clouds at far distance with gaussians splats?

Is there any specific research around this

Handling horizons, skies, and distant backgrounds is a notorious challenge in Gaussian Splatting. Because standard 3DGS relies on Structure-from-Motion (SfM) point clouds for initialization, it struggles with the sky, which lacks distinct trackable features. Forcing standard anisotropic Gaussians to model these unbounded areas typically results in "floaters," splotchy artifacts, and massive memory bloat as the optimizer tries to stretch primitives into infinity.
Over the past year, the research community has settled on four primary architectural paradigms to solve this.
1. Hybrid Representations (Explicit Splats + Implicit Neural Fields)
Borrowed directly from the NeRF++ playbook, methods like Unbounded-GS and HyRF decompose the scene into a near-field and a far-field.
The Mechanism: The foreground is handled using standard, explicitly optimized 3D Gaussians for high-frequency detail. For the background and sky, the pipeline hands off to an implicit multi-layer perceptron (MLP). The MLP predicts far-away background colors directly from the positional encodings of the ray direction and view position.
Rendering: During the forward pass, the explicit splats are rasterized, and the remaining ray transmittance is filled by querying the MLP, blending the two seamlessly. This drastically reduces the parameter count and prevents Gaussians from bleeding into the sky.
2. Homogeneous Coordinates (HoGS)
Instead of partitioning the scene, Homogeneous Gaussian Splatting (HoGS) alters the underlying coordinate geometry of the rasterizer.
The Mechanism: It integrates homogeneous coordinates (projective geometry) directly into the 3DGS framework. By representing positions and scales in homogeneous space, an object at infinite distance (like the horizon) has a mathematically finite, optimizable coordinate representation.
Trade-offs: This is highly elegant because it avoids arbitrary bounding boxes or two-stage rendering pipelines. It unifies near and far object reconstruction seamlessly within a single forward pass, which is ideal if you are writing custom WebGPU/WGSL compute kernels and want to avoid branching logic.
3. Geodesic Shells (Two-Stage Gaussian Splatting)
If you want to keep the scene entirely explicit (no MLPs), recent frameworks physically constrain distant Gaussians to a bounding sphere.
The Mechanism: The architecture defines a geodesic spherical shell between an inner radius $R_i$ and an outer radius $R_o$. Distant scenery is initialized uniformly across this shell rather than relying on SfM depth.
The Math: To keep the optimizer from pulling these background splats into the foreground, a specific shell loss is applied to the point centers $p_i$ relative to the scene origin $O$:
$$\mathcal{L}_{\text{shell}} = \frac{1}{N} \sum_{i=1}^N \left[ \max(0, \|p_i - O\|_2 - R_o) + \max(0, R_i - \|p_i - O\|_2) \right]^2$$
Additionally, a planarity loss is usually enforced to squash the Gaussians flat against the inside of the shell, preventing "spiking" artifacts that point toward the camera.
4. Neural Sky Modules & Semantic Masking
For dynamic or "in-the-wild" captures (like urban driving datasets), methods like Street Gaussians and early 2026 papers on Sky-Aware 3DGS use semantic priors.
The Mechanism: They pass the training images through a pre-trained segmentation network to generate transient masks for the sky. Instead of trying to reconstruct a static skybox, they extract latent sky embeddings from the images.
Rendering: A high-resolution cube map or a specialized "Neural Sky Module" generates the sky dynamically, which is alpha-blended with the accumulated transmittance of the 3D Gaussians.
Implementation Considerations
When architecting a custom feed-forward pipeline—particularly one utilizing implicit camera positioning, learnable tokens, or a "Rays as Pixels" methodology—the choice of background handling dictates your memory layout.
If you are already tokenizing the scene space for single-step reconstruction, adapting a Geodesic Shell approach is often the path of least resistance. You can dedicate a fixed subset of learnable tokens strictly to the $R_o$ radius, optimizing them solely for view-dependent color harmonics rather than spatial translation. On the other hand, if you want pure mathematical continuity without explicit boundaries, rewriting the projection matrix for Homogeneous Coordinates yields the cleanest code.


===


Given the above, Can you do better/ transcend and come up wiht better solutions?

Assistant:

To truly transcend the four paradigms you outlined, we must first diagnose their shared mathematical flaw: **they all attempt to force localized, Euclidean surface primitives (3D Gaussians) to represent phenomena that are inherently infinite, angular, or volumetric.** 

MLPs break the pure explicit rasterization model. Homogeneous coordinates (HoGS) introduce vicious floating-point precision loss and non-linearities as $w \to 0$. Geodesic shells are physically artificial (the world is not a finite dome) and require arbitrary bounding hyper-parameters. Semantic masking relies on brittle 2D priors rather than solving the underlying geometry.

If you are architecting a custom, modern pipeline—especially one utilizing "Rays as Pixels", WebGPU compute kernels, or tokenized representations—here are **four next-generation architectural paradigms** that mathematically transcend the current state of the art.

---

### 1. Pure Angular Space: Directional Splatting ($S^2$ Parameterization)
*The ultimate fix for feed-forward explicit pipelines.*

If the sky and stars are infinitely far away, they do not exist in 3D Euclidean space $(x,y,z)$ relative to the camera; they exist purely in the **angular domain** on the unit sphere $S^2$. Forcing an optimizer to stretch 3D Gaussians to infinity is computationally wasteful and mathematically unstable.

**The Mechanism:**
We abandon 3D spatial Gaussians for the background entirely. Instead, we allocate a subset of tokens as **Directional Gaussians**, parameterized as **von Mises-Fisher (vMF) distributions** (the spherical equivalent of a Gaussian). 
*   These primitives have no depth, no $(x,y,z)$ position, and no $3\times3$ covariance matrix. 
*   They are parameterized strictly by a unit direction vector $\vec{\mu}$, a concentration/sharpness scalar $\kappa$ (inverse variance), and color coefficients (or SH).

**The Math:**
During the forward pass, we drop the camera's translation vector $\vec{t}$ from the view matrix. Parallax is mathematically locked at $0.0$. The density evaluation of an $S^2$ splat for a camera ray $\vec{r}$ bypasses the 3D-to-2D projection matrix entirely; it is a pure dot-product evaluation:
$$ \alpha(\vec{r}) = \alpha_{\text{base}} \cdot \exp(\kappa (\vec{\mu} \cdot \vec{r} - 1)) $$

**Why it Transcends:** Zero "floaters" can exist because these splats physically have no 3D bounds to bleed from. It avoids the heavy projection matrix math of HoGS. In a "Rays as Pixels" architecture, after you rasterize your standard 3D foreground, you evaluate the $S^2$ mixture on the remaining ray transmittance $T_{\text{final}}$ in an $O(1)$ pass.

### 2. Topological Transcendence: Conformal Space Contraction
*The mathematical fix for continuous depth and horizon blending.*

If you want to maintain a pure, unified 3D coordinate system (to seamlessly blend distant mountains into the sky) but without the severe precision loss of projective coordinates at infinity, you can borrow the topological trick from *Mip-NeRF 360* and adapt it for explicit splats.

**The Mechanism:**
The scene is partitioned into a finite primal ball (radius $R$) and an inverted dual space. Everything beyond $R$ is warped using a conformal contraction mapping, such as $f(\mathbf{x}) = (2 - \frac{R}{\|\mathbf{x}\|}) \frac{\mathbf{x}}{\|\mathbf{x}\|}$. In this space, absolute infinity mathematically folds perfectly onto the boundary of a sphere of radius 2.

**The Math:**
Warping a 3D Gaussian non-linearly usually destroys its closed-form shape. The transcendence here is to apply the **Jacobian $J$** of the contraction function to analytically warp the covariance matrices of the background splats:
$$ \Sigma_{\text{contracted}} = J \Sigma_{\text{world}} J^T $$

**Why it Transcends:** Unlike a Geodesic Shell (which acts as a hard backstop), Conformal Contraction maintains true, infinite depth ordering. A cloud at 10km and a mountain at 50km remain distinct and correctly sorted. As splats approach infinity, the Jacobian naturally "pancakes" their covariance matrices into flat sheets orthogonal to the camera rays, natively preventing spiking artifacts while keeping the optimizer bounded in pristine 32-bit float precision $[-2, 2]$.

### 3. Radiometric Transcendence: Analytic Atmospheric Volumetrics
*The physics-based fix for "splotchy" skies and horizon haze.*

Current 3DGS models skies as opaque blobs painted on a distant wall. In reality, the sky is not a surface; it is a participating medium governed by Rayleigh and Mie scattering. The optimizer creates artifacts because it is forced to use discrete geometry and Spherical Harmonics to fake volumetric aerial perspective.

**The Mechanism:**
We introduce **Atmospheric Gaussians**. Instead of carrying an RGB color and a discrete opacity $\alpha$, these background Gaussians carry physical parameters: scattering coefficients ($\beta_R, \beta_M$) and an asymmetry parameter $g$ for the phase function.

**The Math:**
Because the density field of a Gaussian is analytic, the integral of its density $\rho$ along any camera ray $\vec{r}(t) = \vec{o} + t\vec{d}$ to infinity has an exact, closed-form solution utilizing the Error Function ($\text{erf}$). Therefore, the total optical depth $\tau$ and in-scattering from a global Sun vector $\vec{l}$ can be evaluated continuously without ray-marching:
$$ \tau(\vec{r}) = \int_{0}^{\infty} \rho(t) dt \propto \text{erf}\left( \frac{\mu - \text{proj}}{\sigma} \right) $$

**Why it Transcends:** The color of the sky and horizon haze emerges organically from the physics of light scattering through this optimized medium. This natively solves the issue of distant mountains fading into the sky color. Furthermore, you can change the sun direction at runtime, and the horizon will physically transition to realistic sunset colors—something entirely impossible with baked MLPs or SH.

### 4. Asymptotic Ground Planes & Cylindrical Horizons
*The geometric fix for the "floating world" problem.*

Even with the best skybox, 3DGS struggles where the sky meets the earth. Because SfM point clouds rarely capture featureless ground planes stretching to infinity, optimizers leave a "gap" between the foreground floor and the background sky. 

**The Mechanism:**
We factorize the infinite boundary into two mathematically distinct infinite primitives:
1. **The Infinite Ground Splat:** A 2D Gaussian mathematically locked to the $XY$ plane ($Z=0$), but given an infinite variance (scale) in the $X$ and $Y$ axes. 
2. **The Cylindrical Horizon Splat:** A Gaussian projected onto an infinitely wide cylinder surrounding the scene.

**Why it Transcends:** By explicitly defining the mathematical asymptotes of the scene, you prevent the optimizer from trying to use hundreds of standard 3D Gaussians to build a "floor." The infinite ground splat guarantees a seamless horizon line where $Z \to 0$ meets $X,Y \to \infty$, grounding the geometry perfectly.

---

### Synthesizing the "Transcendent" Architecture

If you are building a unified, tokenized "Rays as Pixels" architecture and want the cleanest, most performant code without branching logic or neural network dependencies, the ultimate path is a fusion of **Paradigm 1** and **Paradigm 3**:

1. Dedicate $95\%$ of your learnable tokens to standard explicit 3D Gaussians (the near-field).
2. Dedicate $5\%$ to **Directional Atmospheric Splats** ($S^2$ parameters + scattering coefficients).
3. Run your standard foreground rasterization/accumulation.
4. In your final resolve kernel, take the accumulated transmittance $T$ (the "empty" space left in the ray), drop the camera translation vector, and do an $O(1)$ analytic evaluation of your Directional Splats based purely on the viewing angle and the sun vector.

This yields mathematically perfect continuity, zero "floaters", zero bounding-box logic, vastly reduced VRAM usage, and physically accurate horizons, all while remaining 100% feed-forward and explicitly optimizable.
