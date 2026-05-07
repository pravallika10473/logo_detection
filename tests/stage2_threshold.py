#!/usr/bin/env python3
"""
Stage 2 follow-up: apply thresholding + NMS to DETR detections.

Re-runs DETR (cheap, 18 images) with a low score floor, then filters by:
  score >= SCORE_MIN
  min(w, h) >= MIN_SIDE (pixels)
  aspect ratio in [MIN_AR, MAX_AR]
  area <= MAX_AREA_PCT% of image
  greedy NMS at IoU >= NMS_IOU

Output:
  stage2_filtered/
    <stem>/
      annotated_kept.png      (only kept boxes, green)
      annotated_dropped.png   (only dropped boxes, red, with reason)
      crop_<i>_<reason>.png   (kept crops only)
    summary.txt
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import pipeline


TEST_DIR = Path("clean_test/test_images")
OUT_DIR = Path("tests/results/stage2_filtered")
MODEL_ID = "Pravallika6/detr-finetuned-logo-detection_v2"

# Initial detection threshold (we filter further below)
DETECT_THRESH = 0.20

# Filter rules
SCORE_MIN = 0.50
# Confidence-tiered min side: high-conf detections get more leeway because
# DETR is more sure they're real, even if the crop is small.
HIGHCONF_SCORE = 0.85
HIGHCONF_MIN_SIDE = 20
MIN_SIDE = 28
MIN_AR = 0.2
MAX_AR = 5.0
MAX_AREA_PCT = 80.0
NMS_IOU = 0.5


def get_font(size=14):
    for c in ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def filter_detections(dets, W, H):
    """Return (kept, dropped) where each item is dict with reason annotated."""
    annotated = []
    for d in dets:
        b = d["box"]
        x0, y0, x1, y1 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
        w, h = x1 - x0, y1 - y0
        ar = w / h if h else 0
        area_pct = 100 * (w * h) / (W * H) if W * H > 0 else 0

        reasons = []
        if d["score"] < SCORE_MIN:
            reasons.append(f"score<{SCORE_MIN}")
        # Confidence-tiered min size: high-conf detections only need HIGHCONF_MIN_SIDE
        side_floor = HIGHCONF_MIN_SIDE if d["score"] >= HIGHCONF_SCORE else MIN_SIDE
        if min(w, h) < side_floor:
            reasons.append(f"side<{side_floor}")
        if not (MIN_AR <= ar <= MAX_AR):
            reasons.append(f"ar={ar:.2f}")
        if area_pct > MAX_AREA_PCT:
            reasons.append(f"area={area_pct:.0f}%")

        annotated.append({
            **d,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "w": w, "h": h, "ar": ar, "area_pct": area_pct,
            "drop_reasons": reasons,
        })

    # Apply rule-based drops
    survivors = [d for d in annotated if not d["drop_reasons"]]
    rule_dropped = [d for d in annotated if d["drop_reasons"]]

    # NMS on survivors
    survivors.sort(key=lambda d: -d["score"])
    kept = []
    nms_dropped = []
    for d in survivors:
        bbox = (d["x0"], d["y0"], d["x1"], d["y1"])
        suppressed = False
        for k in kept:
            kbox = (k["x0"], k["y0"], k["x1"], k["y1"])
            if iou(bbox, kbox) >= NMS_IOU:
                suppressed = True
                d["drop_reasons"] = [f"nms(by#{k['idx']})"]
                nms_dropped.append(d)
                break
        if not suppressed:
            d["idx"] = len(kept) + 1
            kept.append(d)

    return kept, rule_dropped + nms_dropped


def draw_boxes(img, dets, font, color_fn, label_fn):
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for d in dets:
        color = color_fn(d)
        draw.rectangle([d["x0"], d["y0"], d["x1"], d["y1"]], outline=color, width=3)
        label = label_fn(d)
        tw = draw.textlength(label, font=font)
        th = 16
        ty = max(0, d["y0"] - th)
        draw.rectangle([d["x0"], ty, d["x0"] + tw + 6, ty + th], fill=color)
        draw.text((d["x0"] + 3, ty + 1), label, fill="black", font=font)
    return out


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    OUT_DIR.mkdir(exist_ok=True)

    device_index = 0 if torch.cuda.is_available() else -1
    print(f"Loading DETR ({MODEL_ID})...")
    detr = pipeline(task="object-detection", model=MODEL_ID,
                    device=device_index, use_fast=True)
    font = get_font()

    test_imgs = sorted(
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
    )
    print(f"Filtering with: score>={SCORE_MIN}  "
          f"min_side>={MIN_SIDE} (>={HIGHCONF_MIN_SIDE} if score>={HIGHCONF_SCORE})  "
          f"ar in [{MIN_AR},{MAX_AR}]  area<={MAX_AREA_PCT}%  NMS_iou={NMS_IOU}\n")

    summary = []
    summary.append(f"{'image':<48} {'WxH':<11} {'raw':>3} {'kept':>4} {'drop':>4}")
    summary.append("-" * 76)

    total_raw, total_kept = 0, 0
    no_kept = []

    for path in test_imgs:
        img = Image.open(path).convert("RGB")
        W, H = img.size
        raw = detr(img, threshold=DETECT_THRESH)
        kept, dropped = filter_detections(raw, W, H)

        out_dir = OUT_DIR / path.stem
        out_dir.mkdir(exist_ok=True)

        if kept:
            draw_boxes(
                img, kept, font,
                color_fn=lambda d: "lime",
                label_fn=lambda d: f"#{d['idx']} {d['score']:.2f}",
            ).save(out_dir / "annotated_kept.png")
        else:
            img.save(out_dir / "annotated_kept.png")

        if dropped:
            draw_boxes(
                img, dropped, font,
                color_fn=lambda d: "red",
                label_fn=lambda d: f"{d['score']:.2f} ({','.join(d['drop_reasons'])[:30]})",
            ).save(out_dir / "annotated_dropped.png")

        # save kept crops only + bbox metadata for downstream stages
        kept_meta = []
        for d in kept:
            crop = img.crop((d["x0"], d["y0"], d["x1"], d["y1"]))
            crop_name = f"crop_{d['idx']:02d}_s{d['score']:.3f}_{d['w']}x{d['h']}.png"
            crop.save(out_dir / crop_name)
            kept_meta.append({
                "crop": crop_name,
                "idx": d["idx"],
                "score": float(d["score"]),
                "x0": int(d["x0"]), "y0": int(d["y0"]),
                "x1": int(d["x1"]), "y1": int(d["y1"]),
                "w": int(d["w"]), "h": int(d["h"]),
            })
        (out_dir / "kept_boxes.json").write_text(json.dumps({
            "image": path.name,
            "W": W, "H": H,
            "boxes": kept_meta,
        }, indent=2))

        line = (f"{path.name:<48} {W}x{H:<6}  {len(raw):>3} {len(kept):>4} {len(dropped):>4}")
        summary.append(line)
        print(line)
        total_raw += len(raw)
        total_kept += len(kept)
        if not kept:
            no_kept.append(path.name)

    summary.append("")
    summary.append(f"TOTAL raw detections: {total_raw}")
    summary.append(f"TOTAL kept after filter: {total_kept}  ({100*total_kept/total_raw:.0f}%)")
    summary.append(f"Images with NO kept detections: {len(no_kept)}")
    for n in no_kept:
        summary.append(f"  - {n}")

    (OUT_DIR / "summary.txt").write_text("\n".join(summary))

    print(f"\n{'='*60}")
    print(f"Raw:  {total_raw}   Kept: {total_kept} ({100*total_kept/total_raw:.0f}%)")
    print(f"Images with no kept boxes: {len(no_kept)}")
    if no_kept:
        for n in no_kept:
            print(f"  - {n}")
    print(f"\nSee {OUT_DIR}/<stem>/annotated_kept.png and annotated_dropped.png")


if __name__ == "__main__":
    main()
