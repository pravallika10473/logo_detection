#!/usr/bin/env python3
"""
Stage 2 ensemble: union DETR crops + GDINO crops.

Reads kept_boxes.json from both stage2_filtered/ and stage2_filtered_gdino/,
unions the boxes per image, runs cross-detector NMS at IoU >= 0.7 (high
threshold so we keep slightly-different views from each detector), re-indexes,
and writes a combined output to stage2_filtered_ensemble/.

The DETR and GDINO score scales differ (DETR 0.5-0.99, GDINO 0.3-0.7), so
within-detector comparisons drove the per-detector NMS in their stage 2 runs.
For cross-detector NMS we just take the higher score on near-duplicates;
accepting some bias toward DETR is fine because near-duplicates only matter
when both already agree on the same region.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


TEST_DIR = Path("clean_test/test_images")
DETR_DIR = Path("tests/results/stage2_filtered")
GDINO_DIR = Path("tests/results/stage2_filtered_gdino")
OUT_DIR = Path("tests/results/stage2_filtered_ensemble")

CROSS_NMS_IOU = 0.7   # high so we preserve slightly-different views


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


def load_meta(d: Path, src_tag: str):
    """Returns list of {x0,y0,x1,y1,w,h,score,src,orig_idx}."""
    p = d / "kept_boxes.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    out = []
    for b in data["boxes"]:
        out.append({
            "x0": b["x0"], "y0": b["y0"], "x1": b["x1"], "y1": b["y1"],
            "w": b["w"], "h": b["h"], "score": b["score"],
            "src": src_tag, "orig_idx": b["idx"],
        })
    return out, data.get("image"), data.get("W"), data.get("H")


def cross_nms(boxes):
    """Within the union, suppress near-duplicates (IoU >= CROSS_NMS_IOU).
    Keep higher-score one. Stable: sort desc by score."""
    sorted_boxes = sorted(boxes, key=lambda b: -b["score"])
    kept = []
    for b in sorted_boxes:
        bbox = (b["x0"], b["y0"], b["x1"], b["y1"])
        suppressed = False
        for k in kept:
            kbox = (k["x0"], k["y0"], k["x1"], k["y1"])
            if iou(bbox, kbox) >= CROSS_NMS_IOU:
                suppressed = True
                break
        if not suppressed:
            kept.append(b)
    return kept


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    OUT_DIR.mkdir(exist_ok=True)
    font = get_font()

    # discover all image stems from both detector outputs
    stems = set()
    for d in (DETR_DIR, GDINO_DIR):
        if d.exists():
            stems.update(p.name for p in d.iterdir() if p.is_dir())
    stems = sorted(stems)

    summary = [f"{'image':<48} {'DETR':>5} {'GDINO':>6} {'union':>6} {'kept':>5}"]
    summary.append("-" * 76)

    total_detr, total_gdino, total_kept = 0, 0, 0
    no_kept = []

    for stem in stems:
        # gather meta from each detector
        detr_data = load_meta(DETR_DIR / stem, "detr") if (DETR_DIR / stem).exists() else ([], None, None, None)
        gdino_data = load_meta(GDINO_DIR / stem, "gdino") if (GDINO_DIR / stem).exists() else ([], None, None, None)
        detr_boxes = detr_data[0] if detr_data else []
        gdino_boxes = gdino_data[0] if gdino_data else []

        image_name = (detr_data[1] if detr_data and detr_data[1]
                      else (gdino_data[1] if gdino_data and gdino_data[1] else None))
        W = detr_data[2] if detr_data and detr_data[2] else (gdino_data[2] if gdino_data else 0)
        H = detr_data[3] if detr_data and detr_data[3] else (gdino_data[3] if gdino_data else 0)
        if image_name is None:
            continue

        union = detr_boxes + gdino_boxes
        kept = cross_nms(union)

        out_dir = OUT_DIR / stem
        out_dir.mkdir(exist_ok=True)

        # locate the original image
        orig_path = TEST_DIR / image_name
        if not orig_path.exists():
            continue
        img = Image.open(orig_path).convert("RGB")

        # save crops + meta + annotation
        kept_meta = []
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        for i, b in enumerate(kept, start=1):
            crop = img.crop((b["x0"], b["y0"], b["x1"], b["y1"]))
            crop_name = f"crop_{i:02d}_s{b['score']:.3f}_{int(b['w'])}x{int(b['h'])}.png"
            crop.save(out_dir / crop_name)
            kept_meta.append({
                "crop": crop_name,
                "idx": i,
                "score": float(b["score"]),
                "src": b["src"],
                "x0": int(b["x0"]), "y0": int(b["y0"]),
                "x1": int(b["x1"]), "y1": int(b["y1"]),
                "w": int(b["w"]), "h": int(b["h"]),
            })
            color = "lime" if b["src"] == "detr" else "cyan"
            draw.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=color, width=3)
            label = f"#{i} [{b['src']}] {b['score']:.2f}"
            tw = draw.textlength(label, font=font)
            ty = max(0, b["y0"] - 16)
            draw.rectangle([b["x0"], ty, b["x0"] + tw + 6, ty + 16], fill=color)
            draw.text((b["x0"] + 3, ty + 1), label, fill="black", font=font)

        annotated.save(out_dir / "annotated_kept.png")
        (out_dir / "kept_boxes.json").write_text(json.dumps({
            "image": image_name,
            "W": W, "H": H,
            "boxes": kept_meta,
        }, indent=2))

        line = (f"{image_name:<48} {len(detr_boxes):>5} {len(gdino_boxes):>6} "
                f"{len(union):>6} {len(kept):>5}")
        summary.append(line)
        print(line)
        total_detr += len(detr_boxes)
        total_gdino += len(gdino_boxes)
        total_kept += len(kept)
        if not kept:
            no_kept.append(stem)

    summary.append("")
    summary.append(f"TOTAL  DETR: {total_detr}  GDINO: {total_gdino}  -> ensemble kept: {total_kept}")
    summary.append(f"Images with no kept boxes: {len(no_kept)}")
    for n in no_kept:
        summary.append(f"  - {n}")
    (OUT_DIR / "summary.txt").write_text("\n".join(summary))

    print(f"\n{'='*60}")
    print(f"DETR crops: {total_detr}   GDINO crops: {total_gdino}")
    print(f"Ensemble kept (after cross-NMS): {total_kept}")
    print(f"Images with no kept boxes: {len(no_kept)}")
    for n in no_kept:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
