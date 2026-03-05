from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from microscopy_sr.data import MicroscopySuperResolutionDataset
from microscopy_sr.diffusion import SRDiffusion
from microscopy_sr.eval import (
    calibration_error,
    denorm,
    lpips_metric,
    mean_uncertainty,
    psnr,
    ssim,
)
from microscopy_sr.models import ConditionalUNet, apply_lora_to_model
from microscopy_sr.utils.config import load_yaml
from microscopy_sr.utils.seed import set_seed


def load_model(
    cfg: dict,
    ckpt_path: str,
    device: torch.device,
    lora: bool = False,
) -> ConditionalUNet:
    model = ConditionalUNet(
        in_channels=2,
        out_channels=1,
        base_channels=cfg["model"].get("base_channels", 64),
        t_dim=cfg["model"].get("t_dim", 256),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if lora:
        lora_cfg = state.get("lora", {})
        apply_lora_to_model(
            model,
            rank=lora_cfg.get("rank", 4),
            alpha=lora_cfg.get("alpha", 1.0),
        )
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def mc_sample(
    diffusion: SRDiffusion,
    lr: torch.Tensor,
    hr_shape: tuple,
    n_samples: int,
    ddim_steps: int = 50,
    eta: float = 1.0,
) -> torch.Tensor:
    """Run n_samples independent reverse chains. Returns (M, B, C, H, W)."""
    draws = [diffusion.sample(lr, shape=hr_shape, ddim_steps=ddim_steps, eta=eta) for _ in range(n_samples)]
    return torch.stack(draws, dim=0)


def evaluate_model(
    model: ConditionalUNet,
    diffusion: SRDiffusion,
    loader: DataLoader,
    n_samples: int,
    device: torch.device,
    label: str,
) -> dict:
    psnr_vals, ssim_vals, lpips_vals, cal_vals, mu_vals = [], [], [], [], []

    for batch in loader:
        lr = batch["lr"].to(device)   # (1, 1, lr_H, lr_W)
        hr = batch["hr"].to(device)   # (1, 1, H, W)

        # MC sampling: (M, 1, 1, H, W)
        samples_5d = mc_sample(diffusion, lr, hr.shape, n_samples,
                               ddim_steps=cfg.get("sample", {}).get("ddim_steps", 50),
                               eta=cfg.get("sample", {}).get("eta", 1.0))
        # Remove batch dim: (M, 1, H, W)
        samples = samples_5d.squeeze(1)

        mean_pred = samples.mean(dim=0)    # (1, H, W)
        std_map = samples.std(dim=0)       # (1, H, W)
        hr_single = hr.squeeze(0)          # (1, H, W)

        psnr_vals.append(psnr(denorm(mean_pred), denorm(hr_single)))
        ssim_vals.append(ssim(mean_pred.unsqueeze(0), hr_single.unsqueeze(0)))
        lpips_vals.append(lpips_metric(mean_pred.unsqueeze(0), hr_single.unsqueeze(0)))
        cal_vals.append(calibration_error(samples, hr_single))
        mu_vals.append(mean_uncertainty(std_map))

    def avg(lst: list) -> float:
        return sum(lst) / len(lst)

    return {
        "label": label,
        "psnr": avg(psnr_vals),
        "ssim": avg(ssim_vals),
        "lpips": avg(lpips_vals),
        "cal_err": avg(cal_vals),
        "mean_unc": avg(mu_vals),
    }


def evaluate_bicubic(loader: DataLoader, device: torch.device) -> dict:
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        hr_h, hr_w = hr.shape[-2:]
        bicubic = F.interpolate(lr, size=(hr_h, hr_w), mode="bicubic", align_corners=False).clamp(-1.0, 1.0)
        psnr_vals.append(psnr(denorm(bicubic.squeeze(0)), denorm(hr.squeeze(0))))
        ssim_vals.append(ssim(bicubic, hr))
        lpips_vals.append(lpips_metric(bicubic, hr))

    def avg(lst: list) -> float:
        return sum(lst) / len(lst)

    return {
        "label": "bicubic",
        "psnr": avg(psnr_vals),
        "ssim": avg(ssim_vals),
        "lpips": avg(lpips_vals),
        "cal_err": float("nan"),
        "mean_unc": float("nan"),
    }


def print_table(results: list[dict]) -> None:
    header = f"{'Label':<22} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'CalErr':>8} {'MeanUnc':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<22} {r['psnr']:>8.3f} {r['ssim']:>8.4f} {r['lpips']:>8.4f} "
            f"{r['cal_err']:>8.4f} {r['mean_unc']:>10.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with all metrics")
    parser.add_argument("--config", default="configs/sample_uncertainty.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--lora", action="store_true",
                        help="Set if checkpoint was saved with LoRA weights")
    parser.add_argument("--label", default="model", help="Label for results table row")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of MC posterior samples (M in the proposal)")
    parser.add_argument("--bicubic_baseline", action="store_true",
                        help="Also compute and print bicubic baseline metrics")
    parser.add_argument("--out_json", default=None,
                        help="Save results dict to this JSON file")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
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
    print(f"Evaluating on {len(dataset)} validation samples")

    model = load_model(cfg, args.checkpoint, device, lora=args.lora)
    diffusion = SRDiffusion(
        model,
        timesteps=cfg["diffusion"].get("timesteps", 1000),
        beta_start=cfg["diffusion"].get("beta_start", 1e-4),
        beta_end=cfg["diffusion"].get("beta_end", 2e-2),
    ).to(device)

    results = []
    if args.bicubic_baseline:
        results.append(evaluate_bicubic(loader, device))
    results.append(evaluate_model(model, diffusion, loader, args.n_samples, device, label=args.label))
    print_table(results)

    if args.out_json:
        import json
        from pathlib import Path
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
