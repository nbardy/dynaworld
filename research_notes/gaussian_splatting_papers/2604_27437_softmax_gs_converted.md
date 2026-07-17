# Softmax-GS: Generalized Gaussians Learning When to Blend or Bound

Source:
    https://arxiv.org/abs/2604.27437

Downloaded:
    2026-05-24

Local artifacts:
    PDF: `pdfs/2604_27437_softmax_gs.pdf`
    arXiv source: `sources/2604_27437_softmax_gs/`
    Extracted text: `text/2604_27437_softmax_gs.txt`
    DynaWorld integration note: `2604_27437_softmax_gs_dynaworld_integration.md`

Authors:
    Chen Ziwen, Peng Wang, Hao Tan, Zexiang Xu, Li Fuxin

Note:
    This Markdown was converted from the arXiv TeX source with `latexpand` and
    `pandoc`. Some complex LaTeX tables/algorithms may be partially flattened;
    use the PDF/source for final numeric transcription.

# Abstract

3D Gaussian Splatting is efficient partly because the renderer assumes compact
Gaussian supports do not overlap in depth. Softmax-GS relaxes that assumption by
adding a softmax competition between overlapping Gaussians, learnable
competition strength, and generalized exponential footprints for individual
boundary sharpness. The method tries to keep order invariance for two-way
overlaps while preserving final transmittance, and reports better quality or
parameter efficiency on standard static 3DGS benchmarks.

# Introduction

<figure id="fig:smgs_rot" data-latex-placement="h">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/smgs_fig1.pdf" style="width:80.0%" />
<figcaption>Comparison between different versions of 3D GS and Softmax-GS under slight left and right rotation. Vanilla 3D GS suffers from diffuse boundary and view inconsistency (“popping effect") due to the no-overlap assumption. Only resorting<span class="citation" data-cites="stopthepop"></span> cannot fix view inconsistency for Gaussians on the same surface. Softmax-GS provides both boundary sharpness control and viewpoint consistency between overlapping Gaussians. 360° rotation videos are available on the project page.</figcaption>
</figure>

<figure id="fig:smgs_comp_main" data-latex-placement="h">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/smgs_fig2.pdf" style="width:95.0%" />
<figcaption>Softmax-GS provides flexible boundary sharpness control and introduces a viewpoint-consistent, softmax-based color-merging mechanism for overlapping Gaussians, enabling both smooth color blending and a winner-take-all behavior.</figcaption>
</figure>

3D Gaussian Splatting (3D GS)  has rapidly gained popularity over Neural Radiance Fields (NeRF)  due to its substantially higher efficiency in both training and rendering. This breakthrough has sparked many follow-up work aimed at further improving its capabilities, including advancements in the optimization pipeline , densification strategies , pruning techniques , incorporation of geometric priors and neural networks , compression for on-device deployment , and extensions to large-scale city-level reconstruction , among others.

In this paper, we focus on two primary limitations of 3D GS: blurry boundaries and view inconsistency. Due to the long tails of the Gaussian function, 3D Gaussians inherently exhibit blurry boundaries that cannot be directly controlled. As a result, a large number of Gaussians are required to capture sharp color transitions in the input views. Several works  have addressed this issue by making boundary sharpness adjustable, either through modifications to the Gaussian function or by replacing Gaussians with alternative geometric primitives.

However, simply adjusting the boundary sharpness of individual Gaussians cannot create sharp edges “between" two overlapping Gaussians in 3D space. In fact, the standard 3D GS algorithm assumes that the Gaussians **never** overlap with each other. This assumption is crucial to its rendering efficiency but also leads to view inconsistency—commonly known as the “popping effect”—when the camera undergoes even slight rotations. To mitigate this, several works  propose to re-sort the Gaussians based on pixel depth after the standard global sorting step. However, because these methods do not directly challenge the no-overlap assumption—the root cause of the inconsistency—they fail in cases where two flat Gaussians lie on the same surface, i.e., when their pixel-wise depth values coincide (see Fig. <a href="#fig:smgs_rot" data-reference-type="ref" data-reference="fig:smgs_rot">1</a>). explicitly recognize this limitation and propose replacing Gaussians with constant-density ellipsoids, deriving the exact density integration along camera rays. While effective, this approach reduces rendering speed by more than an order of magnitude and does not provide controllable boundary sharpness.

We propose Softmax-GS, an algorithm that enforces softmax competition among overlapping Gaussians with a tunable sharpness parameter. With some mathematical derivations, we are able to approximate the rendering equation with overlapping Gaussians using the original sorting and splatting approach. To enable individual-Gaussian edge sharpness control, we additionally replace the standard Gaussian kernel with the Generalized Exponential Function (GEF) introduced by GES . By directly relaxing the no-overlap assumption, Softmax-GS addresses both the view-inconsistency and blurry-boundary issues in a unified manner, supporting a continuous range of visual effects, from smooth color blending to crisp, well-defined boundaries.

Through optimization experiments on simple geometric color patterns, we demonstrate that neither individual boundary sharpness control nor softmax competition alone is sufficient for optimal performance; both components are essential for fully flexible optimization. We further evaluate Softmax-GS on established real-world benchmarks , where it can either **halve the number** of Gaussians without degrading rendering quality, or alternatively improve reconstruction fidelity while retaining a rendering speed close to that of the original 3D GS. By jointly addressing both the blurry-boundary and the view-inconsistency problem—two fundamental limitations of Gaussian Splatting—Softmax-GS provides a principled framework that advances the visual quality of 3D scene reconstruction.

# Related Work

In this section, we summarize prior approaches that address the two major challenges in 3D Gaussian Splatting: the blurry boundary issue and the view-inconsistency issue.

**Controllable boundary sharpness.** To address the challenge of controllable boundary sharpness, several approaches have been proposed. GES  replaces the Gaussian function with the Generalized Exponential Function (GEF), in which the squared distance term $`\|\cdot\|^2`$ is generalized to $`\|\cdot\|^{2\alpha}`$, where $`\alpha`$ is a learnable parameter. However, instead of directly implementing GEF, GES employs an approximate rasterization. In contrast, our method incorporates the exact GEF within our CUDA kernel. SSS  adopts the Student’s t-distribution as an alternative to the Gaussian. DisC-GS  models sharpness by optimizing Bézier curve control points, while 3DCS  replaces Gaussians with convex polygons, incorporating learnable corner smoothness and edge blurriness. DRK  constructs geometry primitives by connecting offset dots from the centers of the primitives. Finally, 3D-HGS  enforces sharpness by cutting a Gaussian in half, yielding a sharper representation at the cross-section. We choose to adopt GEF in this work but most of the above methods can be similarly integrated with the proposed Softmax-GS to control the sharpness of Gaussians.

**View inconsistency.** StopThePop  highlights the intrinsic “popping” effect in 3D Gaussian rendering, which arises because Gaussians are globally sorted by their center depths. When two Gaussians intersect, even a slight camera movement can abruptly change their relative order, causing sudden changes in the rendered images. StopThePop addresses this issue by recomputing per-pixel depths for the Gaussian splats and re-sorting them for each pixel according to these depths. However, since it retains the non-overlapping assumption of the vanilla 3D GS, it cannot resolve the artifacts when two flat Gaussians lie on the same surface, where their per-pixel depths coincide (see Fig. <a href="#fig:smgs_rot" data-reference-type="ref" data-reference="fig:smgs_rot">1</a>). LC-WSR  attempts to eliminate the computationally expensive re-sorting step of StopThePop by learning to predict mixing weights for each Gaussian from the depth values, but the learned weights do not necessarily respect the rendering equation outside the training views. Similarly, StochasticSplats  uses Monte Carlo sampling to mix pixel colors according to the transparency of overlapping Gaussians, demonstrating the ability to blend the colors of intersecting Gaussians. In comparison, our method provides additional control by enabling a smooth transition from normal color blending to sharp boundaries between overlapping Gaussians. Finally, EVER  explicitly addresses the non-overlapping assumption by replacing 3D Gaussians with constant-density ellipsoids and deriving the exact volumetric rendering equation along each camera ray. While effective, this approach reduces rendering speed by more than an order of magnitude compared to standard 3D GS and does not provide controllable boundary sharpness, which is suboptimal when the input view is a soft blending of colors.

# Method

In this section, we first revisit the non-overlapping assumption of 3D Gaussian Splatting, discussing why it is a critical prerequisite for the efficiency of the standard 3D GS algorithm. Building on this analysis, we introduce our modifications to the 3D GS framework to approximate the effects of applying softmax competition to overlapping 3D Gaussians.

## The non-overlapping assumption

Following , we start with the original volume rendering equation along a camera ray. Let $`\mathbf{x}=(x,y)`$ denote a pixel position on the image plane, which also specifies the ray passing through that pixel, and let $`I(\mathbf{x})`$ represent the light intensity reaching the pixel. Let $`o(\mathbf{x},l)`$ denote the extinction function, defining the rate of light occlusion at distance $`l`$ from the camera, and let $`c(\mathbf{x},l)`$ be the emission coefficient at distance $`l`$. Then the original volume rendering equation is
``` math
\begin{equation}
    I(\mathbf{x}) = \int_0^L c(\mathbf{x},l)o(\mathbf{x},l)T(o,\mathbf{x},l)dl
    \label{eq:vol_rend}
\end{equation}
```
where $`T(o,\mathbf{x},l)=e^{-\int_0^l o(\mathbf{x},\mu)d\mu}`$ denotes the transmittance until distance $`l`$. Now, suppose $`o`$ is the summation of $`K`$ kernel functions
``` math
\begin{equation}
    o(\mathbf{x},l) = \sum_{k=1}^K o_k(\mathbf{x},l).\label{eq:decom}
\end{equation}
```
Within the context of 3D GS, each $`o_k`$ is represented as a 1D-projection of a 3D Gaussian to the camera ray, associated with a constant color $`c_k`$, where each Gaussian is limited to a compact support (e.g., $`3\sigma`$). Now, 3D GS further makes the key assumption that *the support of all the trimmed Gaussian kernels do not overlap*, so that they can be sorted from nearest to farthest, indexed from $`1`$ to $`K`$. These assumptions allowed rewriting $`T(o, \mathbf{x}, l)`$ as the contribution of the first $`k_l-1`$ Gaussians that are closer to the camera than the distance $`l`$, and the contributions of the Gaussians can be viewed as independent to $`l`$ due to their non-overlapping support. In this manner, we can extract $`T(o, \mathbf{x}, l)`$ outside of the integral in Eq. (<a href="#eq:vol_rend" data-reference-type="ref" data-reference="eq:vol_rend">[eq:vol_rend]</a>), leading to a greatly simplified formulation commonly utilized in the literature.

We believe it is important to reproduce the original derivation of 3D GS (from the 1990s)  to highlight the assumptions that were used. Note we also ignores self-occlusion (of the $`k_l`$-th Gaussian which overlaps with $`l`$), and utilizes a Taylor expansion $`e^x \approx 1 + x`$ below:
``` math
\begin{align}
    T(o,\mathbf{x},l)&=e^{-\int_0^l o(\mathbf{x},\mu)d\mu}
    \approx e^{-\int_0^l \left(\sum_{j=1}^{k_l-1} o_j(\mathbf{x},\mu)\right)d\mu} \nonumber\\
    \tag*{\parbox[t]{.9\linewidth}{\raggedleft non-overlapping assumption, ignore self-occlusion}}\\
    &\approx e^{-\int_0^L \left(\sum_{j=1}^{k_l-1} o_j(\mathbf{x},\mu)\right)d\mu} \nonumber \\
    \tag*{\parbox[t]{.7\linewidth}{\raggedleft non-overlapping assumption}} \nonumber \\ &= e^{\sum_{j=1}^{k_l-1}\left(-\int_0^L  o_j(\mathbf{x},\mu)d\mu\right)}
    %
    = \prod_{j=1}^{k_l-1} e^{-\int_0^L  o_j(\mathbf{x},\mu)d\mu} \nonumber\\
    &\approx \prod_{j=1}^{k_l-1} \left(1-\int_0^L  o_j(\mathbf{x},\mu)d\mu\right). \label{eq:tapprox}\\
    \tag*{\parbox[t]{.7\linewidth}{\raggedleft Taylor expansion}}
\end{align}
```
where $`l`$ is changed to $`L`$ because the non-overlapping assumption dictates that the support of each $`o_j(\mathbf{x},\mu)`$ ends before the distance $`l`$ (hence $`o_j(\mathbf{x}, \mu) = 0`$ for $`\mu  > l`$). Importantly, $`T(\cdot)`$ is now only a piecewise-constant function of $`l`$, and its discontinuities coincide with the support of each $`o_j`$, hence the volume rendering equation becomes:
``` math
\begin{align}
    I(\mathbf{x}) &= \int_0^L \left(\sum_{k=1}^K c_k o_k(\mathbf{x},l)\right)T(o,\mathbf{x},l)dl\label{eq:orig} \\
 %
    & \approx \sum_{k=1}^K c_k \left(\int_0^L o_k(\mathbf{x},l)dl\right) \prod_{j=1}^{k-1} \left(1-\int_0^L o_j(\mathbf{x},\mu)d\mu\right)\nonumber
    %
\end{align}
```
where we have successfully switched the order of integration and summation utilizing the piecewise-constant $`T(\cdot)`$ approximation in Eq. (<a href="#eq:tapprox" data-reference-type="ref" data-reference="eq:tapprox">[eq:tapprox]</a>). Integrating each 1D Gaussian along the ray (denoted by $`a_k = \int_0^L o_k(\mathbf{x},l) dl`$), we arrive at the commonly used Gaussian splatting formulation
``` math
\begin{equation}
    I(\mathbf{x}) = \sum_{k=1}^K c_k a_k \prod_{j=1}^{k-1} \left(1-a_j\right).
    \label{eq:gs}
\end{equation}
```
From this derivation, it is clear that the non-overlapping assumption is key for the efficiency of 3D GS: it allows the 3D Gaussians to be integrated along the ray first, reducing the primitives to 2D, after which they can be composed efficiently using the standard splatting equation—i.e., “splat first, compose second."

Although the non-overlapping assumption simplifies the integration process, it is inherently unrealistic. In real-world scenes represented with 3D Gaussians, different colors are often adjacent, and Gaussians frequently lie in close proximity at color boundaries. Under the vanilla 3D GS framework, overlapping Gaussians are assigned an arbitrary fixed order and rendered according to this order-dependent equation. This leads to view inconsistency, since the ordering changes when the viewpoint is rotated, leading to sudden, significant difference in the rendering. To address these challenges, we propose Softmax-GS, a unified approach that relaxes the non-overlapping assumption and enforces softmax-based competition among overlapping Gaussians.

## Softmax competition and approximation

Due to the inherently diffuse boundaries of Gaussians, representing the sharp color transitions common in real-world scenes requires stacking many tiny Gaussians along each edge. Simply relaxing the non-overlap assumption and allowing equal color blending between Gaussians does not resolve this blurriness issue (top-left of Fig. <a href="#fig:smgs_param" data-reference-type="ref" data-reference="fig:smgs_param">4</a>). Methods like GEF can sharpen a single Gaussian’s boundary but cannot control how two overlapping Gaussians should behave—whether they should blend smoothly or produce a crisp separation.

Ideally, two overlapping Gaussians should be able to adaptively transition between a winner-take-all behavior—assigning each pixel to a single dominant Gaussian for sharp boundaries—and a more balanced blending when appropriate. To achieve this, we introduce a softmax-based competition between overlapping Gaussians, controlled by a tunable parameter $`\beta`$ that modulates the strength of competition. This formulation enables a continuous spectrum of behaviors, ranging from smooth color averaging to well-defined boundaries. Specifically, we re-formulate Eq. (<a href="#eq:decom" data-reference-type="ref" data-reference="eq:decom">[eq:decom]</a>) as:
``` math
\begin{equation}
    o(\mathbf{x},l) = \sum_{k=1}^K w_k (\mathbf{x}, l,p) o_k(\mathbf{x},l)\label{eq:decom2}
\end{equation}
```
where $`w_k`$ denotes the softmax weight resulting from the competition among the Gaussian exponents $`p_k`$:
``` math
\begin{equation}
    w_k(\mathbf{x},l,p) = \dfrac{\exp(\beta\cdot p_k(\mathbf{x},l))}{\sum_{j=1}^{K}\exp(\beta\cdot  p_j(\mathbf{x},l))}.
    \label{eq:softmax_weight}
\end{equation}
```
Here, $`o_k \sim \exp(p_k)`$, and $`\beta`$ controls the strength of the softmax competition. When $`\beta = 0`$, Gaussian colors are blended equally, whereas a large $`\beta`$ produces sharp boundaries between overlapping Gaussians (see Fig. <a href="#fig:smgs_param" data-reference-type="ref" data-reference="fig:smgs_param">4</a>). We operate on $`p_k`$ instead of $`o_k`$ because $`o_k`$ ranges only from 0 to $`\infty`$; using $`\exp(o_k)`$ would yield values no less than 1, making it difficult for the softmax weights to diminish and producing undesired visual effects. We purposefully keep separate notations for $`p_k`$ and $`o_k`$ because as we later add individual Gaussian edge sharpness control in Sec. <a href="#sec:smgs_alg" data-reference-type="ref" data-reference="sec:smgs_alg">3.3</a>, $`o_k \sim \exp(p_k)`$ will no longer hold. Substituting Eq. (<a href="#eq:decom2" data-reference-type="ref" data-reference="eq:decom2">[eq:decom2]</a>) into Eq. (<a href="#eq:orig" data-reference-type="ref" data-reference="eq:orig">[eq:orig]</a>) gives
``` math
\begin{equation}
    I(\mathbf{x}) = \int_0^L \left(\sum_{k=1}^K c_k w_k(\mathbf{x},l,p) o_k(\mathbf{x},l)\right)T(o,\mathbf{x},l)dl\label{eq:vr_sm}
\end{equation}
```
. However, directly evaluating Eq. (<a href="#eq:vr_sm" data-reference-type="ref" data-reference="eq:vr_sm">[eq:vr_sm]</a>) requires a full numerical integration, which is impractical for real-time rendering. Therefore, here we impose a mild assumption that the Gaussians can still be *approximately* sorted along the ray—such that once a Gaussian loses dominance in the softmax competition, it does not regain it in latter parts of the ray. Through this more realistic assumption, we can derive an efficient approximation. Specifically, we first compute the per-Gaussian softmax-weighted integral $`\mathring{a}_k = \int w_k(\mathbf{x},l,p) o_k(\mathbf{x},l)dl`$ and then apply the standard splatting formulation
``` math
\begin{equation}
I(\mathbf{x}) = \sum_{k=1}^K c_k \mathring{a}_k \prod_{j=1}^{k-1} (1 - \mathring{a}_j),
\end{equation}
```
retaining the strategy of splatting first and compositing afterwards. This approximation is most accurate when $`\beta`$ is large (the winner-take-all regime), where Gaussians effectively behave as non-overlapping after softmax suppression.

However, the weight $`w_k`$ still depends on all other Gaussians through the softmax, making the exact computation of $`\mathring{a}_k`$ difficult, even though the non-softmax-ed integrals $`a_k = \int_0^L o_k(\mathbf{x},l) dl`$ are readily available. We therefore propose to apply the softmax competition directly on the set of $`a_k`$’s to obtain $`\hat{a}_k`$. Consider the simple case where two Gaussians share the same shape and completely overlap, then $`\mathring{a}_k=\hat{a}_k`$; if they are fully separated, then $`\mathring{a}_k=a_k`$. Motivated by these two extremes, we approximate the full $`\mathring{a}_k`$ by interpolating between $`a_k`$ and $`\hat{a}_k`$. Fig. <a href="#fig:smgs_int" data-reference-type="ref" data-reference="fig:smgs_int">3</a> plots $`\mathring{a}_k`$ of a Gaussian $`o_k`$ under the influence of another identical Gaussian $`o_j`$ as a function of the distance $`|\mu_k - \mu_j|`$ between them. Note as they separate, $`\mathring{a}_k`$ decays back to $`a_k`$ approximately exponentially. Therefore, we propose to use an exponentially decaying factor $`s`$ to interpolate between $`a_k`$ and $`\hat{a}_k`$. Specifically, consider two 3D Gaussians that project to 2D splats with absorbance values $`a_j, a_k`$ and Gaussian exponents $`p_j, p_k`$ with $`a_{\{j,k\}}\sim \exp(p_{\{j,k\}})`$ at pixel $`\mathbf{x}`$. We compute
``` math
\begin{align}
    &\hat{a}_k = w_k a_k \ \ \ \  w_k=\tfrac{\exp(\beta p_k)}{\exp(\beta p_j)+\exp(\beta p_k)}\label{eq:sm}\\
    &s = \exp(-\gamma|d_k-d_j|)\label{eq:decay}\\
    &\bar{a}_k = s \hat{a}_k + (1-s)a_k\ \ \ \
    \bar{a}_j = s \hat{a}_j + (1-s)a_j\label{eq:interp}
\end{align}
```
where $`d_{\{j,k\}}`$ denote the depths of the splats from the camera, and $`s`$ is a $`\gamma`$-controlled decay factor that decreases with their depth difference. This formulation empirically captures how $`\mathring{a}_k`$ varies with the separation between two splats. Now we can present the central idea of Softmax-GS: *when two splats are extremely close, they are treated as belonging to the same surface and their colors are blended through softmax competition; as they separate, they are rapidly interpreted as distinct surfaces again*.

While the set of approximation equations above describe the two-Gaussian case, in practice a camera ray may intersect many Gaussians. This leads to our full Softmax-GS algorithm, which generalizes the above approximation to an arbitrary number of overlapping splats.

<figure id="fig:smgs_int" data-latex-placement="h">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/fig2.pdf" />
<figcaption> Visualization of softmax competition between two identical Gaussians (<span class="math inline"><em>o</em><sub><em>j</em></sub></span>, <span class="math inline"><em>o</em><sub><em>k</em></sub></span>). We plot their softmax-ed values (<span class="math inline"><em>ô</em><sub><em>j</em></sub>, <em>ô</em><sub><em>k</em></sub></span>) and the integral <span class="math inline"><em>å</em><sub><em>k</em></sub></span> of <span class="math inline"><em>ô</em><sub><em>k</em></sub></span> (blue dotted line). The influence of <span class="math inline"><em>o</em><sub><em>j</em></sub></span> on <span class="math inline"><em>å</em><sub><em>k</em></sub></span> decays nearly exponentially with distance, allowing <span class="math inline"><em>å</em><sub><em>k</em></sub></span> to be approximated via an exponentially weighted linear interpolation between the <span class="math inline"><em>å</em><sub><em>k</em></sub></span> values in the non-overlapping and fully overlapping cases (green dotted line). </figcaption>
</figure>

<figure id="fig:smgs_param" data-latex-placement="h">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/smgs_fig4_single_col.pdf" style="width:90.0%" />
<figcaption>We demonstrate the effect of parameter <span class="math inline"><em>α</em></span> and <span class="math inline"><em>β</em></span> with five overlapping Gaussians. Parameter <span class="math inline"><em>α</em></span> controls the sharpness of individual Gaussian boundary and <span class="math inline"><em>β</em></span> controls the harshness of the softmax competition between Gaussians, realizing smooth transition from color blending to clear boundary.</figcaption>
</figure>

## Softmax-GS algorithm

In this section, we present the full Softmax-GS algorithm. The method preserves the linear complexity of the rendering equation by processing Gaussians along each camera ray sequentially, from nearest to farthest. At the $`k`$-th Gaussian, we treat all previously processed Gaussians as a single entity and compute the softmax competition between this combined set and the current Gaussian. Specifically, let $`T_{k-1} = \prod_{j=1}^{k-1} (1 - a_j)`$ denote the current transmittance, and define $`a_{\text{past}} = 1 - T_{k-1}`$ as the accumulated absorbance. Let $`d_{\text{past}}`$ and $`p_{\text{past}}`$ be the moving average depth and Gaussian exponent of the past Gaussians, respectively, and let $`p_{k}, a_{k}, d_{k}`$ represent the exponent, absorbance, and depth of the current Gaussian. Applying the same set of equations (Eq. (<a href="#eq:sm" data-reference-type="ref" data-reference="eq:sm">[eq:sm]</a>)–(<a href="#eq:interp" data-reference-type="ref" data-reference="eq:interp">[eq:interp]</a>)) then yields the updated values $`\bar{a}_{\text{past}}`$ and $`\bar{a}_{k}`$.

**Order invariance and transmittance maintenance.** However, two challenges remain. First, although $`a_{\text{past}}`$ and $`a_{\text{k}}`$ appear symmetric in the computations above, in the 3D GS algorithm the current Gaussian color is multiplied by $`(1-\bar{a}_{\text{past}})\bar{a}_{k}`$ rather than just by $`a_{k}`$, which reduces the contribution of the current Gaussian. Consider two Gaussian splats lying at the same depth. The one processed second according to the sorting order will have its color discounted by a factor of $`1-\hat{a}_{\text{past}}`$ relative to the first. The second issue is that both $`\hat{a}_{\text{past}}<a_{\text{past}}`$ and $`\hat{a}_{k}<a_{k}`$, since the softmax weights are less than one. This causes the output transmittance after the two Gaussians to be higher than in the scenario without softmax competition, causing transmittance inconsistency between image regions with and without overlap. To address these issues, we modify $`\hat{a}_{\text{past}}`$ and $`\hat{a}_{k}`$ to ensure order invariance and maintain the original output transmittance. Formally, setting $`T_{k}=\prod_{j=1}^k (1-a_j) = (1-a_{\text{past}})(1-{a}_{k})`$, we aim to find $`\tilde{a}_{\text{past}}`$ and $`\tilde{a}_{k}`$ satisfying
``` math
\begin{align}
    \tfrac{(1-\tilde{a}_{\text{past}})\tilde{a}_{k}}{\tilde{a}_{\text{past}}}&=\tfrac{\hat{a}_{k}}{\hat{a}_{\text{past}}}\\
\tag*{\parbox[t]{.9\linewidth}{\raggedleft order invariance}}\\
    (1-\tilde{a}_{\text{past}})(1-\tilde{a}_{k}) &= T_{k}.\\
    \tag*{\parbox[t]{.9\linewidth}{\raggedleft maintain original transmittance}}
\end{align}
```
Solving this system of equations yields
``` math
\begin{align}
    \tilde{a}_{\text{past}}=\tfrac{\hat{a}_{\text{past}}(1-T_{k})}{\hat{a}_{\text{past}}+\hat{a}_{k}},
    \tilde{a}_{k}=\tfrac{\hat{a}_{k}(1-T_{k})}{\hat{a}_{k}+ \tilde{a}_{\text{past}}T_{k}}
\end{align}
```
We then apply Eq. (<a href="#eq:interp" data-reference-type="ref" data-reference="eq:interp">[eq:interp]</a>) to $`\tilde{a}_{\text{past}}`$ and $`\tilde{a}_{k}`$ to obtain $`\bar{a}_{\text{past}}`$ and $`\bar{a}_{k}`$. After interpolation, however, the output transmittance may again deviate from $`T_{k}`$. We therefore further correct the absorbance values by computing a scaling factor $`m`$ such that
``` math
\begin{equation}
    (1-m \bar{a}_{\text{past}})(1-m \bar{a}_{k})=T_{k}.
    \label{eq:m_solution}
\end{equation}
```
which leads to the final approximation $`\mathring{a}_{\text{past}} = m \bar{a}_{\text{past}}`$ and $`\mathring{a}_{k} = m \bar{a}_{k}`$. Note that $`m`$ can be solved exactly from eq. (<a href="#eq:m_solution" data-reference-type="ref" data-reference="eq:m_solution">[eq:m_solution]</a>), see the supplementary material.

**Single Gaussian boundary sharpness.** While softmax competition can enforce sharp boundaries between two or more Gaussians, it cannot sharpen an individual Gaussian by itself. To address this, we incorporate the Generalized Exponential Function (GEF) first adopted by GES , where the squared distance $`\|\cdot\|^2`$ in the Gaussian function is replaced by $`\|\cdot\|^{2\alpha}`$, allowing flexible control over the shape of Gaussian boundaries. Unlike GES, we implement the exact GEF directly in our CUDA kernel rather than using an approximate rasterization. Note that this modification affects only the absorbance value $`a`$, not the Gaussian exponent $`p`$, which participates in the softmax competition. Modifying $`p`$ directly would cause the softmax competition less effective at distinguishing between Gaussians as the boundary sharpness increases.

Collectively, we introduce three additional parameters for each Gaussian: $`\alpha`$, which controls the boundary sharpness of an individual Gaussian; $`\beta`$, which governs the strength of softmax competition between overlapping Gaussians; and $`\gamma`$, which determines the decay rate of the softmax competition with increasing distance between Gaussians. The complete forward pass of Softmax-GS rendering algorithm is provided in supplementary material.

**Efficient backward pass.** The backward pass of the original 3D GS algorithm also enjoys linear complexity, proceeding from the farthest Gaussian to the nearest along each ray and computing gradients in reverse. This is achieved by inferring the input transmittance $`T_{\text{in}}`$ from the output transmittance $`T_{\text{out}}`$ and the current absorbance $`a_{\text{cur}}`$:
``` math
\begin{equation}
    T_{\text{in}}=\frac{T_{\text{out}}}{1-a_{\text{cur}}}.
\end{equation}
```
However, in Softmax-GS this inference is no longer valid, and many other intermediate values necessary for gradient computation ( e.g. $`a_{\text{past}}`$ and $`d_{\text{past}}`$) cannot be directly obtained from the output values at each Gaussian iteration. To preserve the linear complexity of the backward pass, we apply Softmax-GS only to the closest $`K`$ Gaussians for each image patch. This allows us to create CUDA arrays of length $`K`$ and cache all intermediate values required for gradient computation during a single forward pass. During the backward pass, the $`k`$-th entry of these arrays can be directly accessed, eliminating the need to recompute the forward pass up to the $`k`$-th Gaussian. The hyperparameter $`K`$ can be empirically chosen based on the approximate total number of Gaussians in the scene. We use $`K=128`$ for all the real-world scene experiments, which covered $`70\%`$ - $`93\%`$ of the Gaussians.

# Experiments

<figure id="fig:smgs_simple" data-latex-placement="h">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/fig4.pdf" />
<figcaption>Simple geometry fitting with 4 Gaussians using 3D GS, Softmax-GS, and two variants ablating softmax competition or boundary sharpness. Only one component yields suboptimal results, while Softmax-GS achieves the best of both. Optimization process videos are available on the project page.</figcaption>
</figure>

<figure id="fig:smgs_simple2" data-latex-placement="h">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/fig5.pdf" />
<figcaption>Fitting experiment with 3D GS and Softmax-GS on <code>circle4</code> with different Gaussian number constraint. As the number of Gaussians increases, 3D GS achieves similar result as Softmax-GS, but still struggles with sharp corners.</figcaption>
</figure>

<div class="table*">

<div class="minipage">

<div class="center">

</div>

</div>

</div>

## Simple geometry patterns

To clearly demonstrate the functionality of Softmax-GS, we created several simple color patterns as shown in Fig. <a href="#fig:smgs_simple" data-reference-type="ref" data-reference="fig:smgs_simple">5</a>. We initialize 4 black Gaussians and run optimization for 10K steps without opacity reset using default rendering losses.

In Fig. <a href="#fig:smgs_simple" data-reference-type="ref" data-reference="fig:smgs_simple">5</a>, we show optimization results with only 4 Gaussians. Softmax-GS achieves results closest to the target, highlighting the complementary roles of edge sharpness control and softmax competition. Using edge sharpness alone improves PSNR significantly, but the Gaussians on the same surface are still enforced with an order, making color blending or correct boundary between Gaussians difficult to obtain. Conversely, using only softmax competition struggles to control the edge sharpness of the outside boundaries of the individual Gaussians.

<div class="table*">

<div class="minipage">

<div class="center">

</div>

</div>

</div>

<div class="minipage">

<div class="center">

</div>

</div>

We also conduct experiments varying the maximum number of Gaussians allowed during the densification process. Quantitative results in terms of PSNR are provided in Table <a href="#tb:smgs_simple" data-reference-type="ref" data-reference="tb:smgs_simple">[tb:smgs_simple]</a>. As expected, the gap between vanilla 3D GS and Softmax-GS decreases when more Gaussians are used, since 3D GS can approximate sharp boundaries by stacking many small Gaussians. Nevertheless, Softmax-GS maintains an advantage w.r.t. 3D GS and *sharp edge only* even with 1024 Gaussians. Qualitative comparisons are shown in Fig. <a href="#fig:smgs_simple2" data-reference-type="ref" data-reference="fig:smgs_simple2">6</a>.

## Real-world data

We evaluate Softmax-GS on a standard real-world benchmark suite, including seven scenes from Mip-NeRF360 , two scenes from Tanks&Temples  (`train` and `truck`), and two scenes from DeepBlending  (`drjohnson` and `playroom`). For DeepBlending, we use the original image resolution. For Mip-NeRF360 and Tanks&Temples, we preserve the aspect ratio while resizing the width to 1600.

For real-world scenes, we adopt the same densification and opacity resetting strategy as in the original 3D GS, while raising the Gaussian size pruning threshold to 0.4, as the GEF-based edge sharpening might shrink the effective size of the Gaussian splats, and our strategy allows the usage of larger and **fewer** Gaussians to represent color patterns. We also present the Softmax-GS$`_{5\%}`$, Softmax-GS$`_\text{mini}`$ and Softmax-GS$`_\text{light}`$ versions with significantly fewer Gaussians by tuning the densification parameter to 4<span class="smallcaps">e</span>-3, 5<span class="smallcaps">e</span>-4 and 3<span class="smallcaps">e</span>-4, respectively. In addition to running Softmax-GS with the standard 3D GS optimization, we also evaluate its performance combined with GS-MCMC’s improved optimization strategy. Notably, our transmittance-maintaining rendering algorithm integrates seamlessly with GS-MCMC’s cloning approach, as it ensures that overlapping “new” Gaussians conform to the overall transmittance assigned to them.

<figure id="fig:smgs_rw" data-latex-placement="htb">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/fig6.pdf" style="width:81.0%" />
<figcaption>Qualitative comparison with 3D GS, GES and StopThePop on real world datasets.</figcaption>
</figure>

Quantitative comparisons are shown in Table <a href="#tb:smgs_real" data-reference-type="ref" data-reference="tb:smgs_real">[tb:smgs_real]</a> in three categories: (1) methods evaluated under extreme sparsity constraints, (2) methods designed to produce sparse scene representations , and (3) methods using a number of Gaussians comparable to standard 3D GS. In the first group, under a stringent budget of only 5% of the original Gaussians, Softmax-GS consistently outperforms 3D GS by $`>`$<!-- -->1 dB PSNR across all datasets, demonstrating stronger robustness to extreme sparsity. In the second group, Softmax-GS$`_{\text{light}}`$ substantially outperforms all competing sparse approaches—particularly in PSNR—demonstrating its ability to achieve higher fidelity with fewer Gaussians. Remarkably, Softmax-GS$`_{\text{light}}`$ attains performance on par with the original 3D GS while using only about **half** as many Gaussians. In the third group, both Softmax-GS and Softmax-GS-MCMC consistently outperform 3D GS and GS-MCMC, respectively. In particular, Softmax-GS improves PSNR by approximately +0.3 dB across all datasets compared to 3D GS.

The qualitative comparison between Softmax-GS and the baselines is provided in Fig. <a href="#fig:smgs_rw" data-reference-type="ref" data-reference="fig:smgs_rw">7</a>. Softmax-GS achieves significantly better fidelity on thin structures such as (from left to right) the radiator, the antenna, the text and the details of the building. Such details are important for faithful reconstruction of the scene.

## Training and rendering speed

Table <a href="#tb:smgs_profile" data-reference-type="ref" data-reference="tb:smgs_profile">[tb:smgs_profile]</a> presents the training and rendering speed in two representative scenes, `bicycle` and `train`. Our method preserves linear time complexity with respect to the number of Gaussians per ray for both forward and backward passes. The additional softmax-competition and transmittance-maintenance steps add only about 1.2× increase in training time, while rendering runs at roughly 80% of the original 3D GS speed. Furthermore, Softmax-GS$`_\text{light}`$ matches the training and rendering speeds of standard 3D GS due to the reduced Gaussian count.

# Conclusion

We propose Softmax-GS, a unified solution to the view-inconsistency and diffuse-boundary issues in the original 3D GS algorithm. Our method introduces a softmax-based color-merging mechanism for overlapping Gaussians with controllable competition strength, enabling a continuum of visual behaviors ranging from smooth color blending to sharp, winner-take-all boundaries. Softmax-GS is derived by carefully modifying the assumption from the original volume rendering equation and designed to ensure order invariance for overlapping Gaussians, while maintaining consistent transmittance across the overlapping and non-overlapping regions, thereby preventing discontinuity artifacts. Softmax-GS achieves state-of-the-art performance on real-world benchmarks for both sparse and dense Gaussian reconstructions. Providing both boundary sharpness control and view consistency, Softmax-GS offers a flexible and efficient framework for real-world scene reconstruction.

# Acknowledgements

Chen Ziwen and Li Fuxin are partially supported by Oregon State University College of Agricultural Sciences Seed grant AGD010-AS06, NSF grants 1751402 and 2321851, Oregon Department of Agriculture 2023 Specialty Crop Block Grant, DARPA TIAMAT grant HR0011-24-9-0423 and an Adobe Fellowship.

# More implementation details

For simple-geometry fitting experiments, we place a camera at the origin facing the +z direction, and initialize four black Gaussians with identical shapes slightly apart at a depth of 1 unit in front of the camera. Optimization is run for 10K steps without opacity reset against the target image using default rendering losses. We set the learning rates for $`\alpha`$, $`\beta`$, and $`\gamma`$ to $`0.0003`$, $`0.0003`$, and $`0.0002`$, respectively. For real-world benchmarks, we set the learning rates for $`\alpha`$, $`\beta`$, and $`\gamma`$ to $`0.0008`$, $`0.008`$, and $`0.0004`$, respectively. To accelerate training, we adopt the tile culling strategies proposed in .

To improve stability, we further introduce a variance regularization on $`\beta`$ and $`\gamma`$ along each camera ray. This term penalizes abrupt changes in these parameters when the depth ordering of Gaussians changes. Specifically, the variance is computed over Gaussians intersecting a ray and weighted by the inverse distance to the nearest (front-most) Gaussian, so that the regularization primarily affects the first visible cluster. The weights for both variance terms are set to $`\lambda = 0.01`$.

For our quantitative comparison, since previous works may have used different image resolutions for their benchmarks, we download the official code of the baselines and re-run all algorithms at a consistent resolution. We keep all methods’ default hyperparameters unchanged, and all scenes are optimized for 30K steps. For GS-MCMC , SSS , and Softmax-GS-MCMC, all of which employ Monte Carlo-based optimization, we initialize from the SfM point cloud and cap the maximum number of Gaussians to match that of the original 3D GS.

<div class="table*">

<div class="minipage">

<div class="center">

</div>

</div>

</div>

# Measurement of view consistency

We evaluate the view consistency of Softmax-GS following the protocol from StopThePop . We first render a video from the reconstructed Gaussians and, for each frame $`\mathbf{I}_i`$, pair it with the frame seven steps ahead, $`\mathbf{I}_{i+7}`$. We then apply the RAFT optical-flow method  to warp $`\mathbf{I}_i`$ to $`\mathbf{I}_{i+7}`$, producing $`\hat{\mathbf{I}}_i`$, and compute MSE and LIP$`_7`$ similarity metrics between $`\hat{\mathbf{I}}_i`$ and $`\mathbf{I}_{i+7}`$. Because does not disclose the camera-trajectory generation process, we simply insert 128 interpolated frames between each pair of target frames. Following , we run RAFT using the model pre-trained on SINTEL  and resize the input frames to SINTEL’s resolution of 1024×436.

We present detailed evaluation results on five representative scenes in Table <a href="#tb:view_consist" data-reference-type="ref" data-reference="tb:view_consist">[tb:view_consist]</a>, reporting both rendering quality (PSNR) and view-consistency metrics (MSE and LIP). All methods are evaluated under the same protocol. We observe that Softmax-GS, by addressing overlapping Gaussians, and StopThePop, by introducing a re-sorting mechanism, each improves view consistency individually. However, as shown in the rendered videos provided the supplementary materials, Softmax-GS exhibits significantly fewer “floater" artifacts compared to StopThePop (especially in the `kitchen` scene), indicating that our approach leads to more stable optimization behavior.

# Depth rendering of Softmax-GS

We visualize the depth rendering of Softmax-GS in Fig. <a href="#fig:dep_synth" data-reference-type="ref" data-reference="fig:dep_synth">8</a> for a synthetic pattern and in Fig. <a href="#fig:dep_bicycle" data-reference-type="ref" data-reference="fig:dep_bicycle">9</a> for a real-world scene. Note the smooth depth transitions at Gaussian boundaries.

# Non-coplanar intersection

We visualize two Gaussians crossing at 30$`^\circ`$ in Fig. <a href="#fig:cross" data-reference-type="ref" data-reference="fig:cross">10</a>. In contrast to the popping artifacts of 3D GS and the abrupt color changes of STP, Softmax-GS produces smooth, flicker-free transitions at the crossing. Note Softmax-GS effectively merges intersecting Gaussians into a single surface, consistent with the physical assumption that real-world object surfaces do not cross.

<figure id="fig:dep_synth">
<div class="minipage">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/rebuttal/4.pdf" style="width:99.0%" />
</div>
<figcaption>Pixel-wise depth rendering of synthetic pattern.</figcaption>
</figure>

<figure id="fig:dep_bicycle">
<div class="minipage">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/rebuttal/5.pdf" style="width:99.0%" />
</div>
<figcaption>Depth rendering comparison with 3D GS.</figcaption>
</figure>

<figure id="fig:cross">
<embed src="/Users/nicholasbardy/git/gsplats_browser/dynaworld/research_notes/gaussian_splatting_papers/media/2604_27437_softmax_gs/figs/rebuttal/3.pdf" style="width:99.0%" />
<figcaption>Two Gaussians crossing at 30<span class="math inline"><sup>∘</sup></span> angle.</figcaption>
</figure>

<div class="table*">

<div class="minipage">

<div class="center">

</div>

</div>

</div>

# Per-scene results

Per-scene comparisons of PSNR and Gaussian counts are presented in Table <a href="#tb:smgs_real_indiv" data-reference-type="ref" data-reference="tb:smgs_real_indiv">[tb:smgs_real_indiv]</a>, showing that Softmax-GS achieves higher rendering quality with a similar number of Gaussians across all scenes.

# Full Algorithm

We provide the complete forward-pass of the Softmax-GS algorithm in Algorithm <a href="#alg:smgs" data-reference-type="ref" data-reference="alg:smgs">[alg:smgs]</a>.

<div class="algorithm*">

<div class="algorithmic">

$`T_{\text{past}}=1, c_{\text{past}}=0`$ $`d_{\text{past}}=0, p_{\text{past}}=0`$ $`K, \textbf{x}_{\text{pixel}}`$ $`\textbf{x}[K], \mathbf{\sigma}[K],o[K],c[K],d[K]`$ $`\alpha[K],\beta[K], \gamma[K]`$ $`k\gets 1`$ $`\textbf{x}'=\textbf{x}[k]-\textbf{x}_{\text{pixel}}`$ $`p_{\text{cur}}\gets -0.5\cdot \text{Mahalanobis\_distance}(\textbf{x}',\sigma[k])`$ $`a_{\text{cur}} \gets o[k]\cdot \exp(-(-p_{\text{cur}})^{\alpha[k]})`$

$`T_{\text{orig}}\gets T_{\text{past}}\cdot (1-{a}_{\text{cur}})`$ $`a_{\text{past}}\gets 1-T_{\text{past}}`$ $`w_{\text{cur}}\gets 1 / (1+\exp(\beta[k] \cdot (p_{\text{past}}-p_{\text{cur}}))`$ $`\hat{a}_{\text{cur}}\gets w_{\text{cur}}\cdot a_{\text{cur}}`$ $`w_{\text{past}}\gets 1 - w_{\text{cur}}`$ $`\hat{a}_{\text{past}}\gets w_{\text{past}}\cdot a_{\text{past}}`$ $`\hat{a}_{\text{past}}\gets \frac{\hat{a}_{\text{past}}(1-T_{\text{orig}})}{\hat{a}_{\text{past}}+\hat{a}_{\text{cur}}}`$ $`\hat{a}_{\text{cur}}=\frac{\hat{a}_{\text{cur}}(1-T_{\text{orig}})}{\hat{a}_{\text{cur}}+ \hat{a}_{\text{past}}T_{\text{orig}}}`$ $`s\gets \exp(-\gamma[k]|d[k]-d_\text{past}|)`$ $`\bar{a}_{\text{past}} \gets s\cdot \hat{a}_{\text{past}}+(1-s)\cdot a_{\text{past}}`$ $`a_{\text{cur}} \gets s\cdot \hat{a}_{\text{cur}}+(1-s)\cdot a_{\text{cur}}`$ $`m\gets \frac{a_{\text{cur}}+\bar{a}_{\text{past}}-\sqrt{(a_{\text{cur}}+\bar{a}_{\text{past}})^2-4(1-T_{\text{orig}})a_{\text{cur}}\bar{a}_{\text{past}}}}{2a_{\text{cur}}\bar{a}_{\text{past}}}`$ $`\bar{a}_{\text{past}}\gets m\cdot \bar{a}_{\text{past}}`$ $`a_{\text{cur}}\gets m\cdot a_{\text{cur}}`$ $`T_{\text{past}}\gets 1 - \bar{a}_{\text{past}}`$ $`c_{\text{past}}\gets c_{\text{past}} \cdot \bar{a}_{\text{past}} / a_{\text{past}}`$ $`c_{\text{past}}\gets c_{\text{past}}+c[k]\cdot a_{\text{cur}}\cdot T_{\text{past}}`$ $`d_{\text{past}} \gets \frac{d_{\text{past}}\cdot (1-T_{\text{past}})+d[k]\cdot a_{\text{cur}}\cdot T_{\text{past}}}{1-T_{\text{past}}+a_{\text{cur}}\cdot T_{\text{past}}}`$ $`p_{\text{past}} \gets \frac{p_{\text{past}}\cdot (1-T_{\text{past}})+p_{\text{cur}}\cdot a_{\text{cur}}\cdot T_{\text{past}}}{1-T_{\text{past}}+a_{\text{cur}}\cdot T_{\text{past}}}`$ $`T_{\text{past}}\gets T_{\text{past}}\cdot (1-a_{\text{cur}})`$ $`k\gets k+1`$

</div>

</div>

# Limitations

Softmax-GS has three main limitations. First, the proposed splatting algorithm is applied only to the first 128 Gaussians along each ray in order to maintain linear complexity in the backward pass. As a result, coverage is incomplete: on Mip-NeRF360 indoor scenes, Softmax-GS accounts for approximately 85% of pixels across test images, while for outdoor scenes the coverage drops to around 70%. Second, the order-invariance mechanism in Softmax-GS does not extend to cases with three or more overlapping Gaussians with distinct colors. This limitation arises from the bookkeeping complexity required to preserve permutation invariance under a strict linear-time constraint. Third, the current formulation struggles with semi-transparent Gaussians. In particular, distant semi-transparent Gaussians can bias the running estimates of accumulated depth and opacity toward intermediate values, which in turn affects the softmax-based competition among subsequent Gaussians along the ray. Addressing these limitations is an important direction for future work.
