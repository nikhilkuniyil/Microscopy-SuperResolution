# Uncertainty-Aware Few-Shot Super-Resolution via Conditional Diffusion Models with Low-Rank Adaptation

Base scaffold for the proposal: **uncertainty-aware few-shot diffusion super-resolution for microscopy images**.

## What is implemented
- Conditional DDPM baseline for super-resolution (`lr -> hr`).
- Base training entrypoint.
- Few-shot fine-tuning entrypoint (parameter-efficient by prefix freezing).
- Uncertainty sampling script (multiple draws + per-pixel std map).
- Microscopy dataset loader that creates LR/HR pairs from local HR images by bicubic downsampling.

## Project structure
- `src/microscopy_sr/data/dataset.py`: microscopy dataset and LR/HR pair generation.
- `src/microscopy_sr/models/unet.py`: conditional U-Net noise predictor.
- `src/microscopy_sr/diffusion/ddpm.py`: DDPM training loss and sampling loop.
- `src/microscopy_sr/train.py`: training utilities and checkpointing.
- `scripts/train_base.py`: base pretraining script.
- `scripts/finetune_fewshot.py`: few-shot adaptation script.
- `scripts/sample_uncertainty.py`: multi-sample reconstruction + uncertainty maps.
- `configs/*.yaml`: default configs.

## Setup
```bash
python3 -m pip install -e .
```

## Run
```bash
python3 scripts/train_base.py --config configs/base_train.yaml
python3 scripts/finetune_fewshot.py --config configs/fewshot_finetune.yaml
python3 scripts/sample_uncertainty.py --config configs/sample_uncertainty.yaml
```

## Notes
- Current config uses `images/` as the dataset root and assumes raw microscopy images are available there.
- Dataset split is random but reproducible via `seed`.
- Images are converted to grayscale and normalized to `[-1, 1]`.
- This is the base code scaffold; evaluation extensions (SSIM, calibration metrics, per-cell-type splits) can be added next.
