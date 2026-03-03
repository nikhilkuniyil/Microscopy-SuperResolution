from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from microscopy_sr.data import MicroscopySuperResolutionDataset
from microscopy_sr.diffusion import SRDiffusion
from microscopy_sr.eval import denorm, save_image
from microscopy_sr.models import ConditionalUNet
from microscopy_sr.utils.config import load_yaml


def save_uncertainty_map(std_map: torch.Tensor, out_path: str) -> None:
    arr = std_map.squeeze().cpu().numpy()
    arr = arr / (arr.max() + 1e-8)
    arr = (arr * 255.0).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample_uncertainty.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MicroscopySuperResolutionDataset(
        root_dir=cfg["data"]["root_dir"],
        split="val",
        train_ratio=cfg["data"].get("train_ratio", 0.9),
        seed=cfg.get("seed", 42),
        patch_size=cfg["data"].get("patch_size", 128),
        upscale_factor=cfg["data"].get("upscale_factor", 4),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ConditionalUNet(
        in_channels=2,
        out_channels=1,
        base_channels=cfg["model"].get("base_channels", 64),
        t_dim=cfg["model"].get("t_dim", 256),
    ).to(device)

    state = torch.load(cfg["sample"]["checkpoint"], map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    diffusion = SRDiffusion(
        model,
        timesteps=cfg["diffusion"].get("timesteps", 1000),
        beta_start=cfg["diffusion"].get("beta_start", 1e-4),
        beta_end=cfg["diffusion"].get("beta_end", 2e-2),
    ).to(device)

    n_examples = cfg["sample"].get("n_examples", 8)
    n_samples = cfg["sample"].get("n_samples", 8)
    out_dir = Path(cfg["sample"].get("output_dir", "outputs/uncertainty"))

    for i, batch in enumerate(loader):
        if i >= n_examples:
            break
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        draws = []
        for j in range(n_samples):
            pred = diffusion.sample(lr, shape=hr.shape)
            draws.append(pred)
            save_image(pred[0], str(out_dir / f"sample_{i:03d}" / f"draw_{j:02d}.png"))

        stack = torch.stack(draws, dim=0)
        mean_pred = stack.mean(0)
        std_map = stack.std(0)

        save_image(hr[0], str(out_dir / f"sample_{i:03d}" / "target_hr.png"))
        save_image(mean_pred[0], str(out_dir / f"sample_{i:03d}" / "mean_pred.png"))
        save_uncertainty_map(std_map[0], str(out_dir / f"sample_{i:03d}" / "uncertainty_std.png"))

        print(f"saved sample {i}")


if __name__ == "__main__":
    main()
