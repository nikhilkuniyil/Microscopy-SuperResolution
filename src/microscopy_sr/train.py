from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from microscopy_sr.data import MicroscopySuperResolutionDataset
from microscopy_sr.diffusion import SRDiffusion
from microscopy_sr.models import ConditionalUNet


def build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_set = MicroscopySuperResolutionDataset(
        root_dir=data_cfg["root_dir"],
        split="train",
        train_ratio=data_cfg.get("train_ratio", 0.9),
        seed=cfg.get("seed", 42),
        patch_size=data_cfg.get("patch_size", 128),
        upscale_factor=data_cfg.get("upscale_factor", 4),
        few_shot_k=data_cfg.get("few_shot_k"),
    )
    val_set = MicroscopySuperResolutionDataset(
        root_dir=data_cfg["root_dir"],
        split="val",
        train_ratio=data_cfg.get("train_ratio", 0.9),
        seed=cfg.get("seed", 42),
        patch_size=data_cfg.get("patch_size", 128),
        upscale_factor=data_cfg.get("upscale_factor", 4),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"].get("batch_size", 8),
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"].get("batch_size", 8),
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )
    return train_loader, val_loader


def build_system(cfg: dict[str, Any], device: torch.device) -> tuple[ConditionalUNet, SRDiffusion, torch.optim.Optimizer]:
    model_cfg = cfg["model"]
    model = ConditionalUNet(
        in_channels=2,
        out_channels=1,
        base_channels=model_cfg.get("base_channels", 64),
        t_dim=model_cfg.get("t_dim", 256),
    ).to(device)

    diff_cfg = cfg["diffusion"]
    diffusion = SRDiffusion(
        model,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 2e-2),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    return model, diffusion, optimizer


def maybe_load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, ckpt_path: str | None, device: torch.device) -> int:
    if not ckpt_path:
        return 0
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)) + 1


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, out_dir: str) -> str:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    ckpt_path = path / f"epoch_{epoch:03d}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
    return str(ckpt_path)


def set_finetune_mode(model: torch.nn.Module, trainable_prefixes: list[str]) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if any(n.startswith(prefix) for prefix in trainable_prefixes):
            p.requires_grad = True


def run_epoch(model: torch.nn.Module, diffusion: SRDiffusion, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, train: bool = True) -> float:
    model.train(train)
    total_loss = 0.0

    for batch in tqdm(loader, leave=False):
        hr = batch["hr"].to(device)
        lr = batch["lr"].to(device)

        t = torch.randint(0, diffusion.timesteps, (hr.size(0),), device=device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            loss = diffusion.p_losses(hr, lr, t)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * hr.size(0)

    return total_loss / len(loader.dataset)
