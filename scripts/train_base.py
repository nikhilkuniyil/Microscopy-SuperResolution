from __future__ import annotations

import argparse
from pathlib import Path

import torch

from microscopy_sr.train import build_loaders, build_system, run_epoch
from microscopy_sr.utils.config import load_yaml
from microscopy_sr.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_train.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(cfg)
    model, diffusion, optimizer = build_system(cfg, device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_val = float("inf")
    epochs = cfg["train"].get("epochs", 10)
    out_dir = Path(cfg["train"].get("checkpoint_dir", "checkpoints/base"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = run_epoch(model, diffusion, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, diffusion, val_loader, optimizer, device, train=False)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # Always overwrite latest.pt so training can be resumed
        latest = out_dir / "latest.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, latest)

        # Only save a new best.pt when val loss improves
        if val_loss < best_val:
            best_val = val_loss
            best = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, best)
            print(f"new_best: {best} (val_loss={val_loss:.6f})")


if __name__ == "__main__":
    main()
