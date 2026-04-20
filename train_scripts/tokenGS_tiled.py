import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from tqdm import tqdm

from fast_attn import configure_fast_attn, fast_attn_context
from renderers.tiled import render_pytorch_3dgs_tiled
from gs_models import TokenGS
from image_utils import fetch_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="", help="URL or local path to image")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--size", type=int, default=32, help="Render res (keep <= 128 for PyTorch speed)")
    parser.add_argument("--tokens", type=int, default=128, help="Number of latent 3D tokens before broadcasting")
    parser.add_argument(
        "--gaussians-per-token",
        type=int,
        default=4,
        help="Number of explicit Gaussians emitted by each latent token",
    )
    parser.add_argument("--tile-size", type=int, default=8, help="Tile size for the tiled rasterizer")
    parser.add_argument("--bound-scale", type=float, default=3.0, help="Gaussian screen-space bound in sigmas")
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=1.0 / 255.0,
        help="Opacity-aware tile culling threshold; set <=0 to disable opacity-aware shrinking",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Log scalar loss to W&B every N steps")
    parser.add_argument("--image-log-every", type=int, default=50, help="Log preview image to W&B every N steps")
    parser.add_argument("--amp", action="store_true", help="Use autocast for the model forward pass")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project="tokengs-overfit", name="single-image-run-tiled", config=vars(args))

    raw_img = fetch_image(args.image)
    transform = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    gt_tensor = transform(raw_img).unsqueeze(0).to(device)
    gt_target = gt_tensor.squeeze(0)

    model = TokenGS(num_tokens=args.tokens, gaussians_per_token=args.gaussians_per_token).to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
        fused=True if device.type in {"cuda", "mps"} else None,
    )

    fx = fy = float(args.size)
    cx = cy = float(args.size) / 2.0
    amp_available = bool(args.amp and torch.amp.autocast_mode.is_autocast_available(device.type))
    if args.amp and not amp_available:
        print(f"AMP requested but not available on device {device.type}; continuing in fp32.")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    attn_dtype = amp_dtype if amp_available else gt_tensor.dtype
    attn_backend = configure_fast_attn(device, attn_dtype)

    effective_gaussians = args.tokens * args.gaussians_per_token
    print(
        "Starting Training: "
        f"{args.tokens} latent tokens x {args.gaussians_per_token} gaussians/token "
        f"= {effective_gaussians} explicit Gaussians with {args.tile_size}x{args.tile_size} tiles..."
    )
    print(f"Attention backend: {attn_backend}")
    pbar = tqdm(range(1, args.steps + 1))
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        )
        with fast_attn_context(device):
            with autocast_context:
                xyz, scales, quats, opacities, rgbs = model(gt_tensor)

        render = render_pytorch_3dgs_tiled(
            xyz.float(),
            scales.float(),
            quats.float(),
            opacities.float(),
            rgbs.float(),
            args.size,
            args.size,
            fx,
            fy,
            cx,
            cy,
            tile_size=args.tile_size,
            bound_scale=args.bound_scale,
            alpha_threshold=args.alpha_threshold,
        )

        loss = F.l1_loss(render, gt_target) + 0.2 * F.mse_loss(render, gt_target)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")
        should_log_scalars = step % max(1, args.log_every) == 0 or step == args.steps
        should_log_images = step % max(1, args.image_log_every) == 0 or step == args.steps
        if should_log_scalars:
            payload = {"Loss": loss.item()}
            if should_log_images:
                combined = torch.cat([gt_target, render.detach()], dim=2)
                payload["Render_GT_vs_Pred"] = wandb.Image(
                    T.ToPILImage()(combined.cpu().clamp(0, 1)),
                    caption=f"Step {step}",
                )
            wandb.log(payload, step=step)

    print("Training Complete! Check your Weights & Biases dashboard.")
    wandb.finish()


if __name__ == "__main__":
    main()
