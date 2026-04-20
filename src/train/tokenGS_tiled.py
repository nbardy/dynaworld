import sys

from tokenGS import main


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit(
            "Usage: uv run python src/train/tokenGS_tiled.py "
            "[src/train_configs/local_mac_overfit_single_image_tiled.jsonc]"
        )
    main(sys.argv[1] if len(sys.argv) == 2 else "src/train_configs/local_mac_overfit_single_image_tiled.jsonc")
