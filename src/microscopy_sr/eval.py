from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    Structural similarity index (pure PyTorch, no extra dependencies).

    pred and target should be in [-1, 1]. Returns a scalar in roughly [0, 1].
    Accepts shapes (B, C, H, W) or (C, H, W).
    """
    p = denorm(pred)
    t = denorm(target)
    if p.dim() == 3:
        p = p.unsqueeze(0)
        t = t.unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    sigma = 1.5

    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)

    channels = p.shape[1]
    kernel = kernel_2d.expand(channels, 1, window_size, window_size)
    pad = window_size // 2

    mu_p = F.conv2d(p, kernel, padding=pad, groups=channels)
    mu_t = F.conv2d(t, kernel, padding=pad, groups=channels)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sig_p_sq = F.conv2d(p * p, kernel, padding=pad, groups=channels) - mu_p_sq
    sig_t_sq = F.conv2d(t * t, kernel, padding=pad, groups=channels) - mu_t_sq
    sig_pt = F.conv2d(p * t, kernel, padding=pad, groups=channels) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / (
        (mu_p_sq + mu_t_sq + C1) * (sig_p_sq + sig_t_sq + C2)
    )
    return ssim_map.mean().item()


_lpips_cache: dict = {}


def lpips_metric(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> float:
    """
    Learned Perceptual Image Patch Similarity using the lpips package.

    pred and target should be in [-1, 1], shape (1, C, H, W) or (C, H, W).
    Grayscale images are replicated to 3 channels automatically.
    """
    try:
        import lpips as _lpips_lib
    except ImportError:
        raise ImportError("lpips package not found. Install with: pip install lpips")

    device = pred.device
    key = (net, str(device))
    if key not in _lpips_cache:
        _lpips_cache[key] = _lpips_lib.LPIPS(net=net, verbose=False).to(device)
    fn = _lpips_cache[key]

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    # LPIPS expects RGB; replicate grayscale channel to 3
    if pred.shape[1] == 1:
        pred = pred.expand(-1, 3, -1, -1)
        target = target.expand(-1, 3, -1, -1)

    with torch.no_grad():
        return fn(pred.float(), target.float()).mean().item()


def calibration_error(
    samples: torch.Tensor,
    target: torch.Tensor,
    alpha_levels: Sequence[float] | None = None,
) -> float:
    """
    Empirical calibration error: mean |Coverage(α) - α| over confidence levels α.

    Args:
        samples: (M, C, H, W) — M independent posterior draws for a single image
        target:  (C, H, W) or (1, C, H, W) — ground-truth HR image
        alpha_levels: confidence levels to evaluate (default: proposal spec)

    Returns:
        Scalar calibration error (lower is better; 0 means perfectly calibrated).
    """
    if alpha_levels is None:
        alpha_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    if target.dim() == 3:
        target = target.unsqueeze(0)  # (1, C, H, W)

    # samples: (M, C, H, W)
    mean_pred = samples.mean(dim=0, keepdim=True)   # (1, C, H, W)
    std_pred = samples.std(dim=0, keepdim=True).clamp(min=1e-8)  # (1, C, H, W)
    abs_err = (target - mean_pred).abs()            # (1, C, H, W)

    normal = torch.distributions.Normal(0.0, 1.0)
    errors = []
    for alpha in alpha_levels:
        z_alpha = normal.icdf(torch.tensor((1.0 + alpha) / 2.0))
        covered = (abs_err <= z_alpha * std_pred).float().mean().item()
        errors.append(abs(covered - alpha))

    return float(sum(errors) / len(errors))


def mean_uncertainty(std_map: torch.Tensor) -> float:
    """Average pixel-wise standard deviation across the uncertainty map."""
    return std_map.mean().item()


def save_image(x: torch.Tensor, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    arr = (denorm(x).squeeze().cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(out_path)
