from __future__ import annotations

"""
Generate side-by-side comparison figures from uncertainty sampling outputs.

Each figure shows a 4-panel row per sample:
  LR input (bicubic upsampled) | Mean prediction | Uncertainty (std) | HR ground truth

Usage:
    python scripts/visualize_uncertainty.py \
        --samples_dir outputs/uncertainty \
        --out_dir outputs/figures
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_LABELS = ["LR Input (bicubic)", "Mean Prediction", "Uncertainty (std)", "HR Ground Truth"]
COLORMAP_STEPS = 256


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def apply_heatmap(arr: np.ndarray) -> np.ndarray:
    """Map a [0,1] grayscale array to a blue-yellow heatmap (uint8 RGB)."""
    arr = arr.clip(0, 1)
    r = to_uint8(arr)
    g = to_uint8(arr * 0.8)
    b = to_uint8(1.0 - arr)
    return np.stack([r, g, b], axis=-1)


def add_label(img: Image.Image, text: str, font_size: int = 14) -> Image.Image:
    w, h = img.size
    label_h = font_size + 8
    canvas = Image.new("RGB", (w, h + label_h), color=(30, 30, 30))
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 4), text, fill=(220, 220, 220), font=font)
    return canvas


def make_figure(sample_dir: Path, bicubic_up: np.ndarray | None = None) -> Image.Image | None:
    mean_path = sample_dir / "mean_pred.png"
    std_path = sample_dir / "uncertainty_std.png"
    hr_path = sample_dir / "target_hr.png"

    if not (mean_path.exists() and std_path.exists() and hr_path.exists()):
        return None

    mean_arr = load_gray(mean_path)
    std_arr = load_gray(std_path)
    hr_arr = load_gray(hr_path)
    H, W = hr_arr.shape

    # Build LR panel: average all draw files, then bicubic upsample
    draws = sorted(sample_dir.glob("draw_*.png"))
    if bicubic_up is not None:
        lr_panel = bicubic_up
    elif draws:
        # Use first draw resized to show LR quality
        d = load_gray(draws[0])
        lr_small = Image.fromarray(to_uint8(d)).resize((W // 4, H // 4), Image.BICUBIC)
        lr_panel = np.array(lr_small.resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0
    else:
        lr_panel = np.zeros((H, W), dtype=np.float32)

    panels = [
        Image.fromarray(to_uint8(lr_panel)).convert("RGB"),
        Image.fromarray(to_uint8(mean_arr)).convert("RGB"),
        Image.fromarray(apply_heatmap(std_arr)),
        Image.fromarray(to_uint8(hr_arr)).convert("RGB"),
    ]

    labeled = [add_label(p, lbl) for p, lbl in zip(panels, PANEL_LABELS)]
    pw, ph = labeled[0].size
    gap = 4
    total_w = pw * 4 + gap * 3
    figure = Image.new("RGB", (total_w, ph), color=(15, 15, 15))
    for i, panel in enumerate(labeled):
        figure.paste(panel, (i * (pw + gap), 0))

    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", default="outputs/uncertainty",
                        help="Directory containing sample_NNN subdirectories")
    parser.add_argument("--out_dir", default="outputs/figures",
                        help="Where to save the output figures")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(samples_dir.glob("sample_*"))
    if not sample_dirs:
        raise SystemExit(f"No sample directories found in {samples_dir}")

    saved = 0
    for sd in sample_dirs:
        fig = make_figure(sd)
        if fig is None:
            print(f"[skip] {sd.name} — missing required files")
            continue
        out_path = out_dir / f"{sd.name}.png"
        fig.save(out_path)
        print(f"Saved {out_path}")
        saved += 1

    print(f"\nDone. {saved} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
