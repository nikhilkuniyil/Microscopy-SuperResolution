from __future__ import annotations

"""
Ablation sweep: runs all baselines and K-sweeps described in the proposal.

Usage:
    python scripts/run_ablations.py \
        --base_ckpt checkpoints/base/best.pt \
        --config_ft configs/fewshot_finetune.yaml \
        --config_eval configs/sample_uncertainty.yaml

This script orchestrates the following comparisons (Table 1 in the paper):
  - Bicubic baseline
  - Base model (no adaptation)
  - LoRA-K  for K in {5, 10, 20, 50}
  - FullFT-K for K in {5, 10, 20, 50}
"""

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    """Run command, streaming output, raise on failure."""
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full ablation sweep over K and adaptation method")
    parser.add_argument("--base_ckpt", required=True,
                        help="Path to the pretrained base checkpoint")
    parser.add_argument("--config_ft", default="configs/fewshot_finetune.yaml",
                        help="Config for finetune_fewshot.py")
    parser.add_argument("--config_eval", default="configs/sample_uncertainty.yaml",
                        help="Config for evaluate.py")
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 20, 50],
                        help="Few-shot K values to sweep")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of MC posterior samples for evaluation (M)")
    parser.add_argument("--ft_dir", default="checkpoints/ablations",
                        help="Directory for fine-tuned checkpoints")
    args = parser.parse_args()

    py = sys.executable

    # ------------------------------------------------------------------ #
    # 1. Bicubic baseline (no model inference needed — just upsample LR)
    # ------------------------------------------------------------------ #
    run([
        py, "scripts/evaluate.py",
        "--config", args.config_eval,
        "--checkpoint", args.base_ckpt,
        "--label", "bicubic",
        "--n_samples", "1",
        "--bicubic_baseline",
    ])

    # ------------------------------------------------------------------ #
    # 2. Base model — no adaptation
    # ------------------------------------------------------------------ #
    run([
        py, "scripts/evaluate.py",
        "--config", args.config_eval,
        "--checkpoint", args.base_ckpt,
        "--label", "base_no_adapt",
        "--n_samples", str(args.n_samples),
    ])

    # ------------------------------------------------------------------ #
    # 3. LoRA-K and FullFT-K sweeps
    # ------------------------------------------------------------------ #
    for k in args.k_values:

        # --- LoRA adaptation ---
        lora_stem = f"lora_k{k}"
        lora_ckpt = f"{args.ft_dir}/{lora_stem}.pt"
        run([
            py, "scripts/finetune_fewshot.py",
            "--config", args.config_ft,
            "--k", str(k),
            "--out_name", lora_stem,
            "--out_dir", args.ft_dir,
        ])
        run([
            py, "scripts/evaluate.py",
            "--config", args.config_eval,
            "--checkpoint", lora_ckpt,
            "--lora",
            "--label", f"LoRA-K{k}",
            "--n_samples", str(args.n_samples),
        ])

        # --- Full fine-tune ---
        fullft_stem = f"fullft_k{k}"
        fullft_ckpt = f"{args.ft_dir}/{fullft_stem}.pt"
        run([
            py, "scripts/finetune_fewshot.py",
            "--config", args.config_ft,
            "--k", str(k),
            "--out_name", fullft_stem,
            "--out_dir", args.ft_dir,
            "--full_ft",
        ])
        run([
            py, "scripts/evaluate.py",
            "--config", args.config_eval,
            "--checkpoint", fullft_ckpt,
            "--label", f"FullFT-K{k}",
            "--n_samples", str(args.n_samples),
        ])

    print("\nAblation sweep complete.")


if __name__ == "__main__":
    main()
