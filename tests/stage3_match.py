#!/usr/bin/env python3
"""
Stage 3: similarity matching on Stage 2 crops.

For each crop in stage2_filtered/<stem>/crop_*.png:
  1. Embed with DINOv2 (CLS, patch-mean, 0.3*CLS + 0.7*patch combo)
  2. Compare to each brand's reference centroid
  3. Predict best brand per crop
  4. Image-level vote: similarity-weighted, ignore crops below SIM_FLOOR,
     UNCERTAIN if no crop passes the floor.

Outputs:
  stage3_match_results.txt    - text report
  stage3_annotated/<stem>.png - original image with each kept bbox color-coded
                                 by predicted brand and labeled with similarity
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


CLEAN_DIR = Path("clean_test")
TEST_DIR = Path("clean_test/test_images")
# CROPS_DIR / annotated_dir are set in main() from CLI args (defaults below)
DEFAULT_CROPS_DIR = Path("tests/results/stage2_filtered")
DEFAULT_ANNOTATED_DIR = Path("tests/results/stage3_annotated")
RESULTS_DIR = Path("tests/results")
BRANDS = ["america_first", "mountain_america", "utah_credit_union", "utah_jazz"]

BRAND_COLORS = {
    "america_first":     "red",
    "mountain_america":  "blue",
    "utah_credit_union": "lime",
    "utah_jazz":         "magenta",
    "UNCERTAIN":         "gray",
}

# Image-level voting: drop crops where best-brand sim is below this floor
# (intra-ref mean is ~0.78 patch; below 0.50 the embedding has no useful signal).
SIM_FLOOR = 0.50

# Embedder used for the visualizations + image-level prediction
PRIMARY_KEY = "patch"

GROUND_TRUTH = {
    "258s":                                  "mountain_america",
    "326d6595f2b58c917c57b15d3e7ceb33":      None,
    "348s":                                  None,
    "3UC3DJJBW5GCFKBJYTOFBK62RA.jpg":        None,
    "Nevada-Reno-123":                       "mountain_america",
    "St.-George-IMG_7866-scaled":            None,
    "U5A4700.jpg":                           None,
    "Untitled-1_02.jpg":                     None,
    "af":                                    "america_first",
    "americafirstcujpg*1200xx4032-2265-0-380": "america_first",
    "b8kbravu0sg4dssqa5ri":                  None,
    "bps1":                                  "america_first",
    "first":                                 "america_first",
    "images":                                "america_first",
    "images_jazz":                           "utah_jazz",
    "imagesl":                               "utah_jazz",
    "maxresdefault":                         "mountain_america",
    "spanish-fork-branch":                   "america_first",
}


def load_dinov2(device):
    from transformers import AutoImageProcessor, Dinov2Model
    print("Loading DINOv2-base...")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    return model, proc


def embed_dinov2(img, model, proc, device):
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        h = model(**inputs).last_hidden_state
    cls = F.normalize(h[:, 0, :], dim=-1).cpu()
    patch = F.normalize(h[:, 1:, :].mean(dim=1), dim=-1).cpu()
    combo = F.normalize(0.3 * cls + 0.7 * patch, dim=-1)
    return {"cls": cls, "patch": patch, "combo": combo}


def list_imgs(d: Path):
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )


def centroid(vecs):
    return F.normalize(torch.cat(vecs).mean(0, keepdim=True), dim=-1)


def compute_brand_centroids(model, proc, device):
    print("\nEmbedding references...")
    centroids = {}
    for brand in BRANDS:
        d = CLEAN_DIR / brand
        embs = []
        for p in list_imgs(d):
            img = Image.open(p).convert("RGB")
            embs.append(embed_dinov2(img, model, proc, device))
        centroids[brand] = {
            key: centroid([e[key] for e in embs])
            for key in ("cls", "patch", "combo")
        }
        print(f"  {brand}: {len(embs)} refs")
    return centroids


def match(crop_emb, centroids, key):
    """Return list of (brand, sim) sorted by sim desc."""
    sims = [(brand, F.cosine_similarity(crop_emb[key], c[key]).item())
            for brand, c in centroids.items()]
    sims.sort(key=lambda x: -x[1])
    return sims


def vote(per_crop, key):
    """Weighted vote across crops:
      - drop crops whose top-brand sim < SIM_FLOOR
      - each surviving crop contributes its top sim to that brand's tally
      - return (brand or 'UNCERTAIN', total_score, n_voters)
    """
    tallies = defaultdict(float)
    voters = 0
    for crop in per_crop:
        ranked = crop[key]["ranked"]
        top_brand, top_sim = ranked[0]
        if top_sim < SIM_FLOOR:
            continue
        tallies[top_brand] += top_sim
        voters += 1
    if not tallies:
        return "UNCERTAIN", 0.0, 0
    best_brand = max(tallies, key=tallies.get)
    return best_brand, tallies[best_brand], voters


def get_font(size=18):
    for c in ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def annotate(orig_path, boxes_meta, per_crop, image_pred, gt, key, font_box, font_title):
    """Draw each kept bbox in brand color, label with predicted brand + sim."""
    img = Image.open(orig_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # title bar at top
    title = f"PRED: {image_pred}"
    if gt:
        title += f"   GT: {gt}   {'OK' if image_pred == gt else 'MISS'}"
    else:
        title += "   GT: ?"
    tw = draw.textlength(title, font=font_title)
    th = 28
    draw.rectangle([0, 0, tw + 10, th], fill="black")
    draw.text((5, 4), title, fill="white", font=font_title)

    # per-bbox annotations
    for box, crop_pred in zip(boxes_meta, per_crop):
        ranked = crop_pred[key]["ranked"]
        top_brand, top_sim = ranked[0]
        margin = top_sim - ranked[1][1]
        below_floor = top_sim < SIM_FLOOR

        color = BRAND_COLORS.get(top_brand, "white")
        # dim if below voting floor
        if below_floor:
            label_prefix = "(weak) "
        else:
            label_prefix = ""

        x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
        width = 4 if not below_floor else 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

        label = f"{label_prefix}{top_brand} {top_sim:.2f} (Δ{margin:+.2f})"
        tw = draw.textlength(label, font=font_box)
        lh = 20
        ly = max(th + 2, y0 - lh)
        draw.rectangle([x0, ly, x0 + tw + 6, ly + lh], fill=color)
        text_color = "black" if color in ("lime", "white") else "white"
        draw.text((x0 + 3, ly + 1), label, fill=text_color, font=font_box)

    return img


def crop_meta(name):
    m = re.match(r"crop_(\d+)_s([\d.]+)_(\d+)x(\d+)\.png", name)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", default=str(DEFAULT_CROPS_DIR),
                        help="Directory of stage 2 output (one subdir per test image)")
    parser.add_argument("--annotated-dir", default=None,
                        help="Where to write annotated visualizations "
                             "(default: stage3_annotated_<crops-dir-suffix>)")
    parser.add_argument("--results-file", default=None,
                        help="Where to write text results (default derived from --crops-dir)")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    suffix = crops_dir.name.replace("stage2_filtered", "").lstrip("_")
    annotated_dir = Path(args.annotated_dir) if args.annotated_dir else (
        RESULTS_DIR / f"stage3_annotated_{suffix}" if suffix else DEFAULT_ANNOTATED_DIR)
    results_file = Path(args.results_file) if args.results_file else (
        RESULTS_DIR / f"stage3_match_results_{suffix}.txt" if suffix
        else RESULTS_DIR / "stage3_match_results.txt")

    annotated_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, proc = load_dinov2(device)
    centroids = compute_brand_centroids(model, proc, device)

    if not crops_dir.exists():
        print(f"\nERROR: {crops_dir}/ not found. Run stage2 first.")
        return

    image_dirs = sorted(d for d in crops_dir.iterdir() if d.is_dir())
    print(f"\nMatching crops from {crops_dir}/ ({len(image_dirs)} images), "
          f"floor={SIM_FLOOR}, primary={PRIMARY_KEY}\n")
    print(f"Annotated -> {annotated_dir}/, results -> {results_file}\n")

    font_box = get_font(14)
    font_title = get_font(18)

    lines = []
    tally = {k: {"correct": 0, "wrong": 0, "uncertain": 0, "no_gt": 0, "no_crops": 0}
             for k in ("cls", "patch", "combo")}
    confusion = defaultdict(int)

    for d in image_dirs:
        meta_path = d / "kept_boxes.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())

        gt = GROUND_TRUTH.get(d.name, None)
        gt_str = gt if gt else "?"
        crops = sorted([p for p in list_imgs(d) if p.name.startswith("crop_")])

        header = f"\n{'='*78}\n{d.name}    [GT: {gt_str}]    crops: {len(crops)}\n{'='*78}"
        lines.append(header)
        print(header)

        if not crops:
            for k in tally:
                tally[k]["no_crops"] += 1
            # still emit an annotated frame (just the title) for completeness
            orig = TEST_DIR / meta["image"]
            if orig.exists():
                img = Image.open(orig).convert("RGB")
                draw = ImageDraw.Draw(img)
                title = f"PRED: NO_CROPS   GT: {gt_str}"
                draw.rectangle([0, 0, draw.textlength(title, font=font_title) + 10, 28], fill="black")
                draw.text((5, 4), title, fill="white", font=font_title)
                img.save(annotated_dir / f"{d.name}.png")
            continue

        # embed each crop, compute per-key ranked matches
        per_crop = []
        for cp in crops:
            img = Image.open(cp).convert("RGB")
            emb = embed_dinov2(img, model, proc, device)
            per_crop_entry = {}
            for k in ("cls", "patch", "combo"):
                per_crop_entry[k] = {"ranked": match(emb, centroids, k)}
            per_crop.append(per_crop_entry)

            # log per-crop for primary key
            ranked = per_crop_entry[PRIMARY_KEY]["ranked"]
            top, top_s = ranked[0]
            margin = top_s - ranked[1][1]
            cm = crop_meta(cp.name)
            tag = f"crop#{cm[0]:>2} score={cm[1]:.2f} {cm[2]}x{cm[3]}" if cm else cp.name
            score_str = "  ".join(f"{b[:14]:>14}={s:.3f}" for b, s in ranked)
            ok = (gt and top == gt)
            mark = "OK" if ok else (f"MISS (gt={gt})" if gt else "")
            below = "  (BELOW FLOOR)" if top_s < SIM_FLOOR else ""
            lines.append(f"  [{PRIMARY_KEY}] {tag} -> {top} (m{margin:+.2f}) {mark}{below}")
            lines.append(f"        {score_str}")

        # image-level vote per key
        for k in ("cls", "patch", "combo"):
            pred, score, n_voters = vote(per_crop, k)
            if k == PRIMARY_KEY:
                lines.append(f"  IMAGE [{k}] -> {pred}  (vote_score={score:.2f}, voters={n_voters})")

            if pred == "UNCERTAIN":
                tally[k]["uncertain"] += 1
            elif gt is None:
                tally[k]["no_gt"] += 1
            elif pred == gt:
                tally[k]["correct"] += 1
            else:
                tally[k]["wrong"] += 1

            if k == PRIMARY_KEY and gt:
                confusion[(gt, pred)] += 1

        # annotated visualization (uses primary key)
        orig = TEST_DIR / meta["image"]
        if orig.exists():
            image_pred, _, _ = vote(per_crop, PRIMARY_KEY)
            ann = annotate(orig, meta["boxes"], per_crop, image_pred, gt,
                           PRIMARY_KEY, font_box, font_title)
            ann.save(annotated_dir / f"{d.name}.png")

    # summary
    summary = ["", "=" * 78, "STAGE 3 SUMMARY  "
               f"(floor={SIM_FLOOR}, primary={PRIMARY_KEY})", "=" * 78]
    for k in ("cls", "patch", "combo"):
        t = tally[k]
        n_with_gt_decisive = t["correct"] + t["wrong"]
        acc = 100 * t["correct"] / n_with_gt_decisive if n_with_gt_decisive else 0
        summary.append(
            f"  {k:>5}: correct={t['correct']}  wrong={t['wrong']}  "
            f"uncertain={t['uncertain']}  no_gt={t['no_gt']}  no_crops={t['no_crops']}  "
            f"-> {acc:.0f}% on {n_with_gt_decisive} decisive predictions with GT"
        )

    summary.append("")
    summary.append(f"Confusion matrix ({PRIMARY_KEY}, GT-known images only):")
    for (gt, pred), n in sorted(confusion.items()):
        summary.append(f"  {gt:<22} -> {pred:<22}  (x{n})")

    summary.append("")
    summary.append(f"Annotated images: {annotated_dir}/")
    out = "\n".join(lines + summary)
    print("\n".join(summary))
    results_file.write_text(out)


if __name__ == "__main__":
    main()
