#!/usr/bin/env python3
"""
Stage 1 follow-up: synthesize variants of the cleanest reference per brand.

Picks the best existing ref per brand (sharpness * short_side), then generates
6 variants that perturb the embedding without changing logo identity:

  scale_0.5  - downscale to 0.5x then back up    (simulates small crops)
  scale_0.7  - downscale to 0.7x then back up    (simulates medium crops)
  bright_up  - brightness +25%
  bright_dn  - brightness -25%
  rot_+5     - rotate +5 degrees, white fill
  blur_s1.5  - Gaussian blur sigma=1.5            (simulates soft capture)

NOT included on purpose: large rotations, mirroring, hue shifts, inversion -
those change logo identity, not just embedding location.

Variants are saved into the same brand folder with prefix `_var_`.
"""

import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


CLEAN_DIR = Path("clean_test")
REF_DIRS = ["america_first", "mountain_america", "utah_credit_union", "utah_jazz"]


def laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    )
    return float(out[1:-1, 1:-1].var())


def quality_score(path: Path) -> float:
    """Higher is better. Sharpness * contrast wins, with a hard floor on short side
    (anything <200px is unusable for 224-input ViTs no matter how sharp)."""
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        return -1.0
    rgb = img.convert("RGB")
    arr = np.asarray(rgb)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    sharp = laplacian_variance(gray)
    contrast = float(gray.std())
    short = min(rgb.size)
    if short < 200:
        return -1.0
    return sharp * contrast


def list_originals(d: Path):
    """Real refs only - skip prior variants (`_var_` prefix)."""
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
        and not p.name.startswith("_var_")
    )


def to_rgb_white_bg(img: Image.Image) -> Image.Image:
    """Composite onto white if there's transparency, then return RGB."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, (0, 0), img)
        return bg.convert("RGB")
    return img.convert("RGB")


def variant_scale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)


def variant_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def variant_rotate(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(255, 255, 255), expand=False)


def variant_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


VARIANTS = [
    ("scale_0.5", lambda im: variant_scale(im, 0.5)),
    ("scale_0.7", lambda im: variant_scale(im, 0.7)),
    ("bright_up", lambda im: variant_brightness(im, 1.25)),
    ("bright_dn", lambda im: variant_brightness(im, 0.75)),
    ("rot_+5",    lambda im: variant_rotate(im, 5)),
    ("blur_s1.5", lambda im: variant_blur(im, 1.5)),
]


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    for brand in REF_DIRS:
        d = CLEAN_DIR / brand
        originals = list_originals(d)
        if not originals:
            print(f"[{brand}] no originals — skip")
            continue

        scored = sorted(((quality_score(p), p) for p in originals), reverse=True)
        best_score, best = scored[0]
        print(f"\n[{brand}] picked best ref: {best.name}  (score={best_score:.1f})")
        print(f"  candidates: {[(p.name, round(s, 1)) for s, p in scored]}")

        base = to_rgb_white_bg(Image.open(best))

        for tag, fn in VARIANTS:
            try:
                v = fn(base)
            except Exception as e:
                print(f"  ! {tag} failed: {e}")
                continue
            out = d / f"_var_{best.stem}_{tag}.png"
            v.save(out, format="PNG")
            print(f"  + {out.name}")


if __name__ == "__main__":
    main()
