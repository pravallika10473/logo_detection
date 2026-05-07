#!/usr/bin/env python3
"""
Stage 1: Image quality audit.

For every test image and reference image:
  - resolution, file size, format, mode
  - sharpness (variance of Laplacian — low = blurry)
  - mean brightness, contrast (std of luminance)
  - flag: tiny (<200px shorter side), low-sharpness, low-contrast, alpha/palette quirks
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np


CLEAN_DIR = Path("clean_test")
TEST_DIR = CLEAN_DIR / "test_images"
REF_DIRS = ["america_first", "mountain_america", "utah_credit_union", "utah_jazz"]

TINY_THRESH = 200          # shorter side, pixels
SHARPNESS_THRESH = 80.0    # variance-of-Laplacian; <80 typically blurry for natural photos
CONTRAST_THRESH = 25.0     # luminance std; <25 means flat


def laplacian_variance(gray: np.ndarray) -> float:
    """3x3 Laplacian, no scipy needed."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray.astype(np.float32)
    h, w = g.shape
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    )
    return float(out[1:-1, 1:-1].var())


def audit(path: Path) -> dict:
    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        return {"path": str(path), "error": str(e)}

    fmt = img.format
    mode = img.mode
    w, h = img.size
    size_kb = path.stat().st_size / 1024

    rgb = img.convert("RGB")
    arr = np.asarray(rgb)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])

    sharpness = laplacian_variance(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    flags = []
    if min(w, h) < TINY_THRESH:
        flags.append(f"TINY({min(w, h)}px)")
    if sharpness < SHARPNESS_THRESH:
        flags.append(f"BLUR({sharpness:.0f})")
    if contrast < CONTRAST_THRESH:
        flags.append(f"FLAT({contrast:.0f})")
    if mode == "P":
        flags.append("PALETTE")
    if mode == "RGBA":
        # alpha channel — fine, but worth knowing
        flags.append("ALPHA")

    return {
        "path": str(path),
        "name": path.name,
        "format": fmt,
        "mode": mode,
        "w": w,
        "h": h,
        "size_kb": round(size_kb, 1),
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "flags": flags,
    }


def list_images(d: Path):
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
    )


def report(title: str, rows: list):
    print(f"\n{'='*78}\n{title}  ({len(rows)} images)\n{'='*78}")
    print(f"{'name':<46} {'fmt':<5} {'mode':<5} {'WxH':<11} {'kb':>6} "
          f"{'sharp':>6} {'bri':>5} {'con':>5}  flags")
    print("-" * 110)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<46}  ERROR: {r['error']}")
            continue
        wxh = f"{r['w']}x{r['h']}"
        flags = " ".join(r["flags"]) if r["flags"] else ""
        print(f"{r['name'][:46]:<46} {str(r['format'] or '?'):<5} {r['mode']:<5} "
              f"{wxh:<11} {r['size_kb']:>6} "
              f"{r['sharpness']:>6} {r['brightness']:>5} {r['contrast']:>5}  {flags}")


def summarize(label: str, rows: list):
    bad = [r for r in rows if r.get("flags")]
    errors = [r for r in rows if "error" in r]
    print(f"\n[{label}] {len(rows)} images  |  flagged: {len(bad)}  |  errors: {len(errors)}")
    if not rows or errors:
        return
    sharps = [r["sharpness"] for r in rows if "sharpness" in r]
    cons = [r["contrast"] for r in rows if "contrast" in r]
    sides = [min(r["w"], r["h"]) for r in rows if "w" in r]
    print(f"   sharpness: min={min(sharps):.0f}  median={sorted(sharps)[len(sharps)//2]:.0f}  max={max(sharps):.0f}")
    print(f"   contrast : min={min(cons):.0f}  median={sorted(cons)[len(cons)//2]:.0f}  max={max(cons):.0f}")
    print(f"   short side: min={min(sides)}  median={sorted(sides)[len(sides)//2]}  max={max(sides)}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    test_rows = [audit(p) for p in list_images(TEST_DIR)]
    report("TEST IMAGES (clean_test/test_images)", test_rows)

    ref_results = {}
    for name in REF_DIRS:
        rows = [audit(p) for p in list_images(CLEAN_DIR / name)]
        ref_results[name] = rows
        report(f"REFERENCE: {name}", rows)

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    summarize("test", test_rows)
    for name, rows in ref_results.items():
        summarize(f"ref:{name}", rows)


if __name__ == "__main__":
    main()
