import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from camera import make_default_camera
from config_utils import load_config_file, resolved_config, serialize_config_value
from fast_attn import configure_fast_attn, fast_attn_context
from gs_models import TokenGS
from image_utils import fetch_image
from renderers.common import build_pixel_grid
from rendering import pick_renderer_mode as resolve_renderer_mode
from rendering import render_gaussian_frame
from tqdm import tqdm
from train_logging import make_preview_image


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    return resolved_config(config, ("data", "model", "render", "train", "logging"))


def pick_renderer_mode(config: dict[str, Any]) -> tuple[str, int]:
    model_cfg = config["model"]
    render_cfg = config["render"]
    effective_gaussians = model_cfg["tokens"] * model_cfg["gaussians_per_token"]
    renderer_mode = resolve_renderer_mode(
        renderer=render_cfg["renderer"],
        gaussian_count=effective_gaussians,
        height=model_cfg["size"],
        width=model_cfg["size"],
        auto_dense_limit=render_cfg["auto_dense_limit"],
    )
    return renderer_mode, effective_gaussians


def render_single_frame(renderer_mode, config, dense_grid, camera, decoded):
    model_cfg = config["model"]
    render_cfg = config["render"]
    return render_gaussian_frame(
        decoded.frame(0),
        camera=camera,
        height=model_cfg["size"],
        width=model_cfg["size"],
        mode=renderer_mode,
        dense_grid=dense_grid,
        tile_size=render_cfg["tile_size"],
        bound_scale=render_cfg["bound_scale"],
        alpha_threshold=render_cfg["alpha_threshold"],
    )


def run_training(config: dict[str, Any]):
    cfg = resolve_config(config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    logging_cfg = cfg["logging"]

    device = pick_device()
    print(f"Using device: {device}")

    raw_img = fetch_image(data_cfg["image"])
    transform = T.Compose([T.Resize((model_cfg["size"], model_cfg["size"])), T.ToTensor()])
    gt_tensor = transform(raw_img).unsqueeze(0).to(device)
    gt_target = gt_tensor.squeeze(0)
    camera = make_default_camera(image_size=model_cfg["size"], device=device)

    wandb.init(
        project=logging_cfg["wandb_project"],
        name=logging_cfg["wandb_run_name"],
        tags=logging_cfg.get("wandb_tags"),
        config=serialize_config_value(cfg),
    )

    model = TokenGS(num_tokens=model_cfg["tokens"], gaussians_per_token=model_cfg["gaussians_per_token"]).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], fused=device.type in {"cuda", "mps"})

    dense_grid = build_pixel_grid(model_cfg["size"], model_cfg["size"], device)
    amp_available = bool(train_cfg["amp"] and torch.amp.autocast_mode.is_autocast_available(device.type))
    if train_cfg["amp"] and not amp_available:
        print(f"AMP requested but not available on device {device.type}; continuing in fp32.")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    attn_dtype = amp_dtype if amp_available else gt_tensor.dtype
    attn_backend = configure_fast_attn(device, attn_dtype)
    renderer_mode, effective_gaussians = pick_renderer_mode(cfg)

    print(
        "Starting TokenGS single-image training: "
        f"{model_cfg['tokens']} latent tokens x {model_cfg['gaussians_per_token']} gaussians/token "
        f"= {effective_gaussians} explicit Gaussians with {renderer_mode} renderer..."
    )
    print(f"Attention backend: {attn_backend}")

    pbar = tqdm(range(1, train_cfg["steps"] + 1))
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            decoded = model(gt_tensor, camera=camera)

        render = render_single_frame(renderer_mode, cfg, dense_grid, camera, decoded)
        loss = F.l1_loss(render, gt_target) + 0.2 * F.mse_loss(render, gt_target)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")
        should_log_scalars = step % max(1, logging_cfg["log_every"]) == 0 or (
            logging_cfg["always_log_last_step"] and step == train_cfg["steps"]
        )
        should_log_images = step % max(1, logging_cfg["image_log_every"]) == 0 or (
            logging_cfg["always_log_last_step"] and step == train_cfg["steps"]
        )
        if should_log_scalars:
            payload = {"Loss": loss.item()}
            if should_log_images:
                payload["Render_GT_vs_Pred"] = make_preview_image(gt_target, render.detach(), caption=f"Step {step}")
            wandb.log(payload, step=step)

    print("TokenGS single-image training complete. Check your Weights & Biases dashboard.")
    wandb.finish()


def main(config: dict[str, Any] | str | Path) -> None:
    if isinstance(config, (str, Path)):
        run_training(load_config_file(config))
    else:
        run_training(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python src/train/tokenGS.py src/train_configs/local_mac_overfit_single_image.jsonc"
        )
    main(sys.argv[1])
