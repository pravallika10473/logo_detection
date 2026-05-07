#!/usr/bin/env python3
"""
Stage 3 (CLIP variant): same pipeline as stage3_match.py but with CLIP-Large
as the embedder.

Image-level vote: similarity-weighted across crops above SIM_FLOOR. Outputs
annotated images at stage3_annotated_clip/.
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
DEFAULT_CROPS_DIR = Path("tests/results/stage2_filtered")
DEFAULT_ANNOTATED_DIR = Path("tests/results/stage3_annotated_clip")
RESULTS_DIR = Path("tests/results")
BRANDS = ["america_first", "mountain_america", "utah_credit_union", "utah_jazz"]

BRAND_COLORS = {
    "america_first":     "red",
    "mountain_america":  "blue",
    "utah_credit_union": "lime",
    "utah_jazz":         "magenta",
    "UNCERTAIN":         "gray",
}

# CLIP ref calibration (measured): intra mean 0.844, inter mean 0.535.
# But noisy DETR crops match much weaker than clean ref-vs-ref pairs (0.55-0.65
# range observed). Floor must sit in that working range, not the ref baseline.
SIM_FLOOR = 0.60
CLIP_MODEL = "openai/clip-vit-large-patch14"

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


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    print(f"Loading {CLIP_MODEL}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, proc


def embed_clip(img, model, proc, device):
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1).cpu()


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
    intra_pairs = []  # for floor calibration / sanity
    for brand in BRANDS:
        d = CLEAN_DIR / brand
        embs = []
        for p in list_imgs(d):
            img = Image.open(p).convert("RGB")
            embs.append(embed_clip(img, model, proc, device))
        centroids[brand] = centroid(embs)
        # log intra-class similarity
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                intra_pairs.append(F.cosine_similarity(embs[i], embs[j]).item())
        print(f"  {brand}: {len(embs)} refs")
    if intra_pairs:
        print(f"\n  intra-class sim (CLIP): "
              f"mean={sum(intra_pairs)/len(intra_pairs):.3f}  "
              f"min={min(intra_pairs):.3f}  max={max(intra_pairs):.3f}")
    return centroids


def match(crop_emb, centroids):
    sims = [(brand, F.cosine_similarity(crop_emb, c).item())
            for brand, c in centroids.items()]
    sims.sort(key=lambda x: -x[1])
    return sims


def vote(per_crop_ranked):
    """Weighted vote across crops: drop those below SIM_FLOOR, tally by top sim."""
    tallies = defaultdict(float)
    voters = 0
    for ranked in per_crop_ranked:
        top_brand, top_sim = ranked[0]
        if top_sim < SIM_FLOOR:
            continue
        tallies[top_brand] += top_sim
        voters += 1
    if not tallies:
        return "UNCERTAIN", 0.0, 0
    best = max(tallies, key=tallies.get)
    return best, tallies[best], voters


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


def annotate(orig_path, boxes_meta, per_crop_ranked, image_pred, gt,
             font_box, font_title):
    img = Image.open(orig_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    title = f"PRED: {image_pred}"
    if gt:
        title += f"   GT: {gt}   {'OK' if image_pred == gt else 'MISS'}"
    else:
        title += "   GT: ?"
    tw = draw.textlength(title, font=font_title)
    draw.rectangle([0, 0, tw + 10, 28], fill="black")
    draw.text((5, 4), title, fill="white", font=font_title)

    for box, ranked in zip(boxes_meta, per_crop_ranked):
        top_brand, top_sim = ranked[0]
        margin = top_sim - ranked[1][1]
        below_floor = top_sim < SIM_FLOOR
        color = BRAND_COLORS.get(top_brand, "white")
        x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
        draw.rectangle([x0, y0, x1, y1], outline=color,
                       width=4 if not below_floor else 2)
        prefix = "(weak) " if below_floor else ""
        label = f"{prefix}{top_brand} {top_sim:.2f} (Δ{margin:+.2f})"
        tw = draw.textlength(label, font=font_box)
        ly = max(30, y0 - 20)
        draw.rectangle([x0, ly, x0 + tw + 6, ly + 20], fill=color)
        text_color = "black" if color in ("lime", "white") else "white"
        draw.text((x0 + 3, ly + 1), label, fill=text_color, font=font_box)
    return img


def crop_meta(name):
    m = re.match(r"crop_(\d+)_s([\d.]+)_(\d+)x(\d+)\.png", name)
    return (int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))) if m else None


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", default=str(DEFAULT_CROPS_DIR))
    parser.add_argument("--annotated-dir", default=None)
    parser.add_argument("--results-file", default=None)
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    suffix = crops_dir.name.replace("stage2_filtered", "").lstrip("_")
    annotated_dir = Path(args.annotated_dir) if args.annotated_dir else (
        RESULTS_DIR / f"stage3_annotated_clip_{suffix}" if suffix else DEFAULT_ANNOTATED_DIR)
    results_file = Path(args.results_file) if args.results_file else (
        RESULTS_DIR / f"stage3_match_clip_results_{suffix}.txt" if suffix
        else RESULTS_DIR / "stage3_match_clip_results.txt")

    annotated_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, proc = load_clip(device)
    centroids = compute_brand_centroids(model, proc, device)

    image_dirs = sorted(d for d in crops_dir.iterdir() if d.is_dir())
    print(f"\nMatching crops from {crops_dir}/ ({len(image_dirs)} images), "
          f"floor={SIM_FLOOR}\n")
    print(f"Annotated -> {annotated_dir}/, results -> {results_file}\n")

    font_box = get_font(14)
    font_title = get_font(18)

    lines = []
    tally = {"correct": 0, "wrong": 0, "uncertain": 0, "no_gt": 0, "no_crops": 0}
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
            tally["no_crops"] += 1
            orig = TEST_DIR / meta["image"]
            if orig.exists():
                img = Image.open(orig).convert("RGB")
                draw = ImageDraw.Draw(img)
                title = f"PRED: NO_CROPS   GT: {gt_str}"
                draw.rectangle([0, 0, draw.textlength(title, font=font_title) + 10, 28], fill="black")
                draw.text((5, 4), title, fill="white", font=font_title)
                img.save(annotated_dir / f"{d.name}.png")
            continue

        per_crop_ranked = []
        for cp in crops:
            img = Image.open(cp).convert("RGB")
            emb = embed_clip(img, model, proc, device)
            ranked = match(emb, centroids)
            per_crop_ranked.append(ranked)

            top, top_s = ranked[0]
            margin = top_s - ranked[1][1]
            cm = crop_meta(cp.name)
            tag = f"crop#{cm[0]:>2} score={cm[1]:.2f} {cm[2]}x{cm[3]}" if cm else cp.name
            score_str = "  ".join(f"{b[:14]:>14}={s:.3f}" for b, s in ranked)
            ok = (gt and top == gt)
            mark = "OK" if ok else (f"MISS (gt={gt})" if gt else "")
            below = "  (BELOW FLOOR)" if top_s < SIM_FLOOR else ""
            lines.append(f"  {tag} -> {top} (m{margin:+.2f}) {mark}{below}")
            lines.append(f"        {score_str}")

        pred, score, n = vote(per_crop_ranked)
        lines.append(f"  IMAGE -> {pred}  (vote_score={score:.2f}, voters={n})")

        if pred == "UNCERTAIN":
            tally["uncertain"] += 1
        elif gt is None:
            tally["no_gt"] += 1
        elif pred == gt:
            tally["correct"] += 1
        else:
            tally["wrong"] += 1

        if gt:
            confusion[(gt, pred)] += 1

        orig = TEST_DIR / meta["image"]
        if orig.exists():
            ann = annotate(orig, meta["boxes"], per_crop_ranked, pred, gt,
                           font_box, font_title)
            ann.save(annotated_dir / f"{d.name}.png")

    n_decisive = tally["correct"] + tally["wrong"]
    acc = 100 * tally["correct"] / n_decisive if n_decisive else 0
    summary = ["", "=" * 78, f"STAGE 3 (CLIP) SUMMARY  floor={SIM_FLOOR}", "=" * 78]
    summary.append(
        f"  correct={tally['correct']}  wrong={tally['wrong']}  "
        f"uncertain={tally['uncertain']}  no_gt={tally['no_gt']}  "
        f"no_crops={tally['no_crops']}  -> {acc:.0f}% on {n_decisive} decisive w/ GT"
    )
    summary.append("")
    summary.append("Confusion (CLIP, GT-known images only):")
    for (gt, pred), n in sorted(confusion.items()):
        summary.append(f"  {gt:<22} -> {pred:<22}  (x{n})")

    summary.append("")
    summary.append(f"Annotated images: {annotated_dir}/")
    out = "\n".join(lines + summary)
    print("\n".join(summary))
    results_file.write_text(out)


if __name__ == "__main__":
    main()
