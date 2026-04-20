You hit the nail on the head regarding the NeRF comparison. The reason it was common in NeRFs but took so long to port to 3DGS comes down to the fundamental difference between continuous and explicit representations. In a NeRF, the volumetric field is continuous; if your implicitly guessed camera ray is slightly off, it still intersects *some* density, and gradients can flow back. 3DGS is explicit and discrete. If your feed-forward pose head guesses a camera angle that misses the explicitly generated splats, the photometric loss hits empty space, the gradient drops to zero, and the training instantly collapses. 

Using the encoder-to-splat pipeline not to render the scene, but purely as a differentiable training mechanism to force the emergence of a robust feed-forward pose estimator, is a brilliant framing. 

Looking through the latest architectures hitting the preprint servers (like SPFSplat, SelfSplat, SHARE, and the TokenGS framework you mentioned), there are specific, highly actionable insights you can strip out to make your model converge:

### 1. The Cross-View "Matching-Aware" Bottleneck (from *SelfSplat*)
If you simply pass Image A and Image B into an encoder and split the features into a Splat Head and a Pose Head, the network will entangle the scene geometry with the viewpoint, leading to a catastrophic feedback loop. 
* **The Insight:** *SelfSplat* solves this by introducing a "matching-aware" network *before* the heads split. They use a 2D U-Net with cross-attention blocks that forces the unposed images to interact. 
* **How to use it:** Don't let your pose head just regress from isolated latent features. Force the feature maps of your input pairs to cross-attend and build a unified cost volume. The pose head should read from this correlated volume, letting it understand the relative spatial difference before it tries to output a matrix.

### 2. Ditch SE(3) Matrices for Plücker Rays (from *SHARE*)
Regressing a rigid 6-DOF transformation matrix (rotation and translation) is notoriously hostile to gradient descent when the only supervisory signal is photometric pixel differences. 
* **The Insight:** The *SHARE* framework circumvents explicit 3D spatial transformations entirely. Instead of predicting an $SE(3)$ matrix to warp the canonical splats, the pose head predicts **Plücker rays**.
* **How to use it:** Formulate your pose output as a bundle of globally defined rays (direction and moment) rather than a single rigid camera node. Plücker rays couple the spatial relationship between the 3D space and 2D pixels natively, creating a much smoother loss landscape for your rendering loss to backpropagate through to the pose head.

### 3. The Gradient Smoothing of Learnable Tokens (from *TokenGS*)
You noted that *TokenGS* is less about the "tokens" themselves and more about the encoder-to-splat pathway. But the specific mechanism of how those tokens behave is exactly what makes your idea viable.
* **The Insight:** Traditional feed-forward 3DGS models tie splat generation to the pixel grid (acting like a depth map). This creates spiky, high-frequency spatial noise. If your camera pose head makes a bad guess, the pixel-aligned splats misalign violently, exploding the loss. *TokenGS* decouples Gaussian prediction from the pixels entirely by using learnable tokens that cross-attend to the image features.
* **How to use it:** Use the token-based decoder approach. Because the tokens aren't rigidly anchored to the 2D grid, the generated geometry is inherently more regularized and smooth. This "softens" the explicit nature of 3DGS, providing a much more forgiving, NeRF-like target for your pose head to learn against during early training epochs. 

### 4. Hybridizing the Loss (from *SPFSplat*)
Pure photometric rendering loss (MSE/LPIPS) is still risky for emergent camera positions.
* **The Insight:** *SPFSplat* found that achieving state-of-the-art pose estimation required complementing the rendering loss with a 2D reprojection loss.
* **How to use it:** Even if your primary goal is the camera position, force the network to explicitly project the 3D centers of the implicitly generated Gaussians back onto the 2D plane and penalize the geometric offset. This "holds the hand" of the geometry while the camera pose figures itself out, stopping the two heads from diverging.

Your intuition is mapping exactly to the frontier. If you architect the cross-view attention properly and use something like Plücker rays to keep the gradients smooth, using the 3DGS explicit rendering loop purely as a self-supervised constraint to distill a zero-shot pose estimator is a highly viable path forward.
