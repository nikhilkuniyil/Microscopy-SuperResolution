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

Resumable: completed steps are skipped automatically on restart.
Results are saved as JSON in --results_dir and aggregated into a final table.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run command, streaming output, raise on failure."""
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def run_eval(py: str, config_eval: str, checkpoint: str, label: str,
             n_samples: int, results_dir: str,
             lora: bool = False, bicubic_baseline: bool = False) -> bool:
    """Run evaluate.py for one configuration. Returns True if skipped."""
    json_path = f"{results_dir}/{label}.json"
    if Path(json_path).exists():
        print(f"[skip] {label} — results already exist at {json_path}")
        return True

    cmd = [
        py, "scripts/evaluate.py",
        "--config", config_eval,
        "--checkpoint", checkpoint,
        "--label", label,
        "--n_samples", str(n_samples),
        "--out_json", json_path,
    ]
    if lora:
        cmd.append("--lora")
    if bicubic_baseline:
        cmd.append("--bicubic_baseline")
    run(cmd)
    return False


def run_finetune(py: str, config_ft: str, k: int, stem: str,
                 ft_dir: str, full_ft: bool = False) -> bool:
    """Run finetune_fewshot.py. Returns True if skipped."""
    ckpt = f"{ft_dir}/{stem}.pt"
    if Path(ckpt).exists():
        print(f"[skip] {stem} — checkpoint already exists at {ckpt}")
        return True

    cmd = [
        py, "scripts/finetune_fewshot.py",
        "--config", config_ft,
        "--k", str(k),
        "--out_name", stem,
        "--out_dir", ft_dir,
    ]
    if full_ft:
        cmd.append("--full_ft")
    run(cmd)
    return False


def print_final_table(results_dir: str) -> None:
    results = []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            results.extend(json.load(f))

    if not results:
        print("No results found.")
        return

    header = f"{'Label':<22} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'CalErr':>8} {'MeanUnc':>10}"
    print("\n" + "=" * len(header))
    print("FINAL RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        psnr = r.get("psnr", float("nan"))
        ssim = r.get("ssim", float("nan"))
        lpips = r.get("lpips", float("nan"))
        cal = r.get("cal_err", float("nan"))
        unc = r.get("mean_unc", float("nan"))
        print(f"{r['label']:<22} {psnr:>8.3f} {ssim:>8.4f} {lpips:>8.4f} {cal:>8.4f} {unc:>10.5f}")


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
    parser.add_argument("--results_dir", default="results/ablations",
                        help="Directory to save per-step JSON results")
    args = parser.parse_args()

    Path(args.ft_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # ------------------------------------------------------------------ #
    # 1. Bicubic baseline (saved separately; no model inference needed)
    # ------------------------------------------------------------------ #
    json_path = f"{args.results_dir}/bicubic.json"
    if Path(json_path).exists():
        print(f"[skip] bicubic — results already exist at {json_path}")
    else:
        run([
            py, "scripts/evaluate.py",
            "--config", args.config_eval,
            "--checkpoint", args.base_ckpt,
            "--label", "bicubic",
            "--n_samples", "1",
            "--bicubic_baseline",
            "--bicubic_only",
            "--out_json", json_path,
        ])

    # ------------------------------------------------------------------ #
    # 2. Base model — no adaptation
    # ------------------------------------------------------------------ #
    run_eval(py, args.config_eval, args.base_ckpt, "base_no_adapt",
             n_samples=args.n_samples, results_dir=args.results_dir)

    # ------------------------------------------------------------------ #
    # 3. LoRA-K and FullFT-K sweeps
    # ------------------------------------------------------------------ #
    for k in args.k_values:

        lora_stem = f"lora_k{k}"
        run_finetune(py, args.config_ft, k, lora_stem, args.ft_dir, full_ft=False)
        run_eval(py, args.config_eval, f"{args.ft_dir}/{lora_stem}.pt",
                 f"LoRA-K{k}", n_samples=args.n_samples,
                 results_dir=args.results_dir, lora=True)

        fullft_stem = f"fullft_k{k}"
        run_finetune(py, args.config_ft, k, fullft_stem, args.ft_dir, full_ft=True)
        run_eval(py, args.config_eval, f"{args.ft_dir}/{fullft_stem}.pt",
                 f"FullFT-K{k}", n_samples=args.n_samples,
                 results_dir=args.results_dir, lora=False)

    # ------------------------------------------------------------------ #
    # 4. Print aggregated results table
    # ------------------------------------------------------------------ #
    print_final_table(args.results_dir)
    print("\nAblation sweep complete.")


if __name__ == "__main__":
    main()
