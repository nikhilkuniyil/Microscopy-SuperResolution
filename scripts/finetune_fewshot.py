from __future__ import annotations

import argparse
import itertools
import os

import torch
from torch.utils.data import DataLoader

from microscopy_sr.data import MicroscopySuperResolutionDataset
from microscopy_sr.models import ConditionalUNet, apply_lora_to_model, freeze_non_lora
from microscopy_sr.train import build_system
from microscopy_sr.utils.config import load_yaml
from microscopy_sr.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot LoRA or full fine-tune")
    parser.add_argument("--config", default="configs/fewshot_finetune.yaml")
    parser.add_argument("--k", type=int, default=None,
                        help="Override few_shot_k from config (number of target examples)")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Checkpoint filename stem (e.g. lora_k20). Auto-generated if omitted.")
    parser.add_argument("--full_ft", action="store_true",
                        help="Full fine-tune (all params); default is LoRA adaptation")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override checkpoint output directory from config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.k is not None:
        cfg["data"]["few_shot_k"] = args.k

    k_val = cfg["data"].get("few_shot_k", "all")

    # Build model and diffusion
    model, diffusion, _ = build_system(cfg, device)

    # Load pretrained base checkpoint (BEFORE applying LoRA so key names match)
    base_ckpt = cfg["train"]["base_checkpoint"]
    state = torch.load(base_ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    print(f"Loaded base checkpoint: {base_ckpt}")

    lora_cfg = cfg.get("lora", {"rank": 4, "alpha": 1.0})

    if args.full_ft:
        for p in model.parameters():
            p.requires_grad = True
        stem = args.out_name or f"fullft_k{k_val}"
        mode_label = "Full fine-tune"
    else:
        apply_lora_to_model(model, rank=lora_cfg.get("rank", 4), alpha=lora_cfg.get("alpha", 1.0))
        freeze_non_lora(model)
        stem = args.out_name or f"lora_k{k_val}"
        mode_label = "LoRA adaptation"

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{mode_label} | Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"].get("lr", 5e-5),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

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
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"].get("batch_size", 4),
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        drop_last=len(train_set) >= cfg["train"].get("batch_size", 4),
    )
    print(f"K={k_val} | {len(train_set)} training samples")

    total_iters = cfg["train"].get("iterations", 500)
    model.train()
    loader_iter = itertools.cycle(train_loader)

    for step in range(total_iters):
        batch = next(loader_iter)
        hr = batch["hr"].to(device)
        lr = batch["lr"].to(device)
        t = torch.randint(0, diffusion.timesteps, (hr.size(0),), device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.p_losses(hr, lr, t)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == total_iters - 1:
            print(f"  step={step}/{total_iters}  loss={loss.item():.6f}")

    out_dir = args.out_dir or cfg["train"].get("checkpoint_dir", "checkpoints/fewshot")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = f"{out_dir}/{stem}.pt"

    payload: dict = {"model": model.state_dict()}
    if not args.full_ft:
        payload["lora"] = lora_cfg
    torch.save(payload, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
