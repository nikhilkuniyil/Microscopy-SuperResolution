from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = "#E8E0D7"
NAVY = "#0A3554"
TEXT = "#353F5A"
PANEL = "#F7F5F0"
BORDER = "#C9C5E8"
GRID = "#DDD8EE"
PURPLE = "#5B50D6"
GOLD = "#C7942B"
GREEN = "#8DAE95"

WIDTH = 1200
HEIGHT = 620
MARGIN = 28


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
                  font: ImageFont.ImageFont, fill: str) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    x = left + (right - left - (bbox[2] - bbox[0])) / 2
    y = top + (bottom - top - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, color: str,
               title_font: ImageFont.ImageFont, value_font: ImageFont.ImageFont) -> None:
    w, h = 220, 84
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=PANEL, outline=color, width=3)
    draw.text((x + 18, y + 16), label, font=title_font, fill=TEXT)
    draw.text((x + 18, y + 42), value, font=value_font, fill=color)


def main() -> None:
    out_dir = Path("report/figures/slides/slide_conclusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    title_font = load_font(28, bold=True)
    axis_font = load_font(22, bold=True)
    label_font = load_font(20, bold=True)
    body_font = load_font(18)
    badge_title_font = load_font(16, bold=True)
    badge_value_font = load_font(24, bold=True)

    draw.rounded_rectangle(
        (MARGIN, MARGIN, WIDTH - MARGIN, HEIGHT - MARGIN),
        radius=26,
        fill=PANEL,
        outline=BORDER,
        width=3,
    )
    draw.text((60, 48), "Fidelity-Calibration Tradeoff", font=title_font, fill=NAVY)
    draw.text(
        (60, 92),
        "Conceptual summary of the result: the best reconstruction and best calibration occur at different shot counts.",
        font=body_font,
        fill=TEXT,
    )

    plot_left, plot_top = 120, 160
    plot_right, plot_bottom = 820, 470

    for t in range(1, 5):
        x = plot_left + t * (plot_right - plot_left) / 5
        y = plot_top + t * (plot_bottom - plot_top) / 5
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=2)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=TEXT, width=4)
    draw.line((plot_left, plot_bottom, plot_left, plot_top), fill=TEXT, width=4)

    draw.polygon(
        [(plot_right, plot_bottom), (plot_right - 14, plot_bottom - 7), (plot_right - 14, plot_bottom + 7)],
        fill=TEXT,
    )
    draw.polygon(
        [(plot_left, plot_top), (plot_left - 7, plot_top + 14), (plot_left + 7, plot_top + 14)],
        fill=TEXT,
    )

    draw.text((plot_right - 180, plot_bottom + 24), "Reconstruction fidelity", font=axis_font, fill=TEXT)
    draw.text((40, plot_top - 10), "Uncertainty", font=axis_font, fill=TEXT)
    draw.text((52, plot_top + 20), "calibration", font=axis_font, fill=TEXT)

    points = {
        "Bicubic": ((250, 415), "#9DA6BA"),
        "Base": ((420, 335), GREEN),
        "LoRA": ((520, 305), GOLD),
        "FullFT-K20": ((610, 225), PURPLE),
        "FullFT-K50": ((735, 285), NAVY),
    }

    line_points = [points["Bicubic"][0], points["Base"][0], points["LoRA"][0], points["FullFT-K20"][0], points["FullFT-K50"][0]]
    draw.line(line_points, fill="#B8B2D9", width=4)

    for name, (xy, color) in points.items():
        x, y = xy
        r = 11 if "FullFT" in name else 9
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=PANEL, width=3)
        if name == "FullFT-K20":
            draw.text((x - 10, y - 42), name, font=label_font, fill=color)
        elif name == "FullFT-K50":
            draw.text((x - 5, y + 18), name, font=label_font, fill=color)
        else:
            draw.text((x + 14, y - 12), name, font=body_font, fill=TEXT)

    draw.text((plot_left + 20, plot_bottom + 64), "lower", font=body_font, fill=TEXT)
    draw.text((plot_right - 56, plot_bottom + 64), "higher", font=body_font, fill=TEXT)
    draw.text((34, plot_bottom - 6), "lower", font=body_font, fill=TEXT)
    draw.text((34, plot_top - 20), "higher", font=body_font, fill=TEXT)

    note_left = 880
    draw.rounded_rectangle((note_left, 170, 1125, 365), radius=20, fill=BG, outline=BORDER, width=2)
    draw.text((note_left + 24, 198), "Interpretation", font=axis_font, fill=NAVY)
    draw.text((note_left + 24, 248), "K50 maximizes fidelity", font=body_font, fill=TEXT)
    draw.text((note_left + 24, 282), "K20 gives better", font=body_font, fill=TEXT)
    draw.text((note_left + 24, 312), "uncertainty calibration", font=body_font, fill=TEXT)

    draw_badge(draw, 860, 420, "Best PSNR / SSIM", "FullFT-K50", PURPLE, badge_title_font, badge_value_font)
    draw_badge(draw, 860, 516, "Best Calibration", "FullFT-K20", GOLD, badge_title_font, badge_value_font)

    image.save(out_dir / "tradeoff_concept.png")

    notes = out_dir / "NOTES.txt"
    notes.write_text(
        "Conclusion slide visuals\n\n"
        "Primary figure:\n"
        "- tradeoff_concept.png: conceptual tradeoff between reconstruction fidelity and uncertainty calibration.\n\n"
        "Suggested use:\n"
        "- Place on the right side of the conclusion slide.\n"
        "- Pair with 3 short bullets on the left.\n"
    )

    print(out_dir / "tradeoff_concept.png")


if __name__ == "__main__":
    main()
