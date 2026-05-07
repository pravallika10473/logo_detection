#!/usr/bin/env python3
"""
Stage 3 (dual-embedder ensemble): DINOv2 patch + CLIP-Large.

For each crop:
  - Embed with both
  - Get per-brand cosine sim from each
  - Combine: average per-brand sim across embedders
  - Top brand = argmax(combined_sim)

For each image:
  - Each crop votes for its top combined brand with weight = combined sim
  - Drop crops below SIM_FLOOR (calibrated to combined-sim scale)
  - Image-level prediction = brand with max tally; UNCERTAIN if no voters

Also reports per-crop agreement: when both embedders pick the same top brand
(stronger signal) vs. disagreement (weaker — one of them is wrong).

Outputs:
  stage3_match_dual_results_<suffix>.txt
  stage3_annotated_dual_<suffix>/<stem>.png
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
DEFAULT_CROPS_DIR = Path("tests/results/stage2_filtered_ensemble")
RESULTS_DIR = Path("tests/results")
BRANDS = ["america_first", "mountain_america", "utah_credit_union", "utah_jazz"]

BRAND_COLORS = {
    "america_first":     "red",
    "mountain_america":  "blue",
    "utah_credit_union": "lime",
    "utah_jazz":         "magenta",
    "UNCERTAIN":         "gray",
}

# Per-embedder floors. Each crop emits up to 2 independent votes (one per
# embedder) using its own calibrated floor. A uniform floor punishes whichever
# embedder runs lower; per-embedder floors let each contribute when confident.
DINO_FLOOR = 0.50
CLIP_FLOOR = 0.60
# Kept for the per-crop "combined" max ranking shown in annotations
SIM_FLOOR = DINO_FLOOR
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


def load_models(device):
    from transformers import AutoImageProcessor, Dinov2Model, CLIPModel, CLIPProcessor
    print("Loading DINOv2-base...")
    dino = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    print(f"Loading {CLIP_MODEL}...")
    clip = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return dino, dino_proc, clip, clip_proc


def embed_dino_patch(img, model, proc, device):
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        h = model(**inputs).last_hidden_state
    return F.normalize(h[:, 1:, :].mean(dim=1), dim=-1).cpu()


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


def compute_centroids(dino, dino_proc, clip, clip_proc, device):
    print("\nEmbedding references with both models...")
    dino_c, clip_c = {}, {}
    for brand in BRANDS:
        d = CLEAN_DIR / brand
        dino_embs, clip_embs = [], []
        for p in list_imgs(d):
            img = Image.open(p).convert("RGB")
            dino_embs.append(embed_dino_patch(img, dino, dino_proc, device))
            clip_embs.append(embed_clip(img, clip, clip_proc, device))
        dino_c[brand] = centroid(dino_embs)
        clip_c[brand] = centroid(clip_embs)
        print(f"  {brand}: {len(dino_embs)} refs")
    return dino_c, clip_c


def per_brand_sims(emb, centroids):
    return {brand: F.cosine_similarity(emb, c).item()
            for brand, c in centroids.items()}


def combine(dino_sims, clip_sims):
    """Per-brand combined = max(dino_sim, clip_sim).
    Either embedder can carry a strong signal; we don't penalize the weaker one."""
    combined = {b: max(dino_sims[b], clip_sims[b]) for b in BRANDS}
    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return combined, ranked


def vote(per_crop):
    """Each crop emits up to 2 independent votes (DINOv2 + CLIP), each with
    its own per-embedder floor. Brand tally = sum of all contributing sims.
    Returns (brand, tally, n_votes_total)."""
    tallies = defaultdict(float)
    voters = 0
    for crop in per_crop:
        d_top = crop["dino_top"]
        d_sim = crop["dino_sims"][d_top]
        if d_sim >= DINO_FLOOR:
            tallies[d_top] += d_sim
            voters += 1
        c_top = crop["clip_top"]
        c_sim = crop["clip_sims"][c_top]
        if c_sim >= CLIP_FLOOR:
            tallies[c_top] += c_sim
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


def annotate(orig_path, boxes_meta, per_crop, image_pred, gt, font_box, font_title):
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

    for box, crop in zip(boxes_meta, per_crop):
        ranked = crop["combined_ranked"]
        top, top_sim = ranked[0]
        margin = top_sim - ranked[1][1]
        below = top_sim < SIM_FLOOR
        agree = crop["agree"]

        color = BRAND_COLORS.get(top, "white")
        x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
        draw.rectangle([x0, y0, x1, y1], outline=color,
                       width=4 if not below else 2)
        prefix = "(weak) " if below else ""
        agree_tag = "" if agree else " [disagree]"
        label = f"{prefix}{top} {top_sim:.2f} (Δ{margin:+.2f}){agree_tag}"
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
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    suffix = crops_dir.name.replace("stage2_filtered", "").lstrip("_") or "default"
    annotated_dir = RESULTS_DIR / f"stage3_annotated_dual_{suffix}"
    results_file = RESULTS_DIR / f"stage3_match_dual_results_{suffix}.txt"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino, dino_proc, clip, clip_proc = load_models(device)
    dino_c, clip_c = compute_centroids(dino, dino_proc, clip, clip_proc, device)

    image_dirs = sorted(d for d in crops_dir.iterdir() if d.is_dir())
    print(f"\nMatching crops from {crops_dir}/ ({len(image_dirs)} images), "
          f"floor={SIM_FLOOR}\n")
    print(f"Annotated -> {annotated_dir}/, results -> {results_file}\n")

    font_box = get_font(14)
    font_title = get_font(18)

    lines = []
    tally = {"correct": 0, "wrong": 0, "uncertain": 0, "no_gt": 0, "no_crops": 0}
    confusion = defaultdict(int)
    n_agree, n_disagree = 0, 0

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

        per_crop = []
        for cp in crops:
            img = Image.open(cp).convert("RGB")
            d_emb = embed_dino_patch(img, dino, dino_proc, device)
            c_emb = embed_clip(img, clip, clip_proc, device)
            d_sims = per_brand_sims(d_emb, dino_c)
            c_sims = per_brand_sims(c_emb, clip_c)
            d_top = max(d_sims, key=d_sims.get)
            c_top = max(c_sims, key=c_sims.get)
            agree = d_top == c_top
            if agree:
                n_agree += 1
            else:
                n_disagree += 1
            combined, ranked = combine(d_sims, c_sims)
            per_crop.append({
                "dino_sims": d_sims, "clip_sims": c_sims,
                "combined": combined, "combined_ranked": ranked,
                "dino_top": d_top, "clip_top": c_top, "agree": agree,
            })

            cm = crop_meta(cp.name)
            tag = f"crop#{cm[0]:>2} score={cm[1]:.2f} {cm[2]}x{cm[3]}" if cm else cp.name
            top, top_s = ranked[0]
            margin = top_s - ranked[1][1]
            ok = (gt and top == gt)
            mark = "OK" if ok else (f"MISS (gt={gt})" if gt else "")
            below = "  (BELOW FLOOR)" if top_s < SIM_FLOOR else ""
            agree_tag = "" if agree else f"  [DINO->{d_top}  CLIP->{c_top}]"
            lines.append(f"  {tag} -> {top} (m{margin:+.2f}) {mark}{below}{agree_tag}")
            score_str = "  ".join(f"{b[:14]:>14}={s:.3f}" for b, s in ranked)
            lines.append(f"        combined: {score_str}")
            lines.append(f"        dino:     " +
                         "  ".join(f"{b[:14]:>14}={d_sims[b]:.3f}" for b in BRANDS))
            lines.append(f"        clip:     " +
                         "  ".join(f"{b[:14]:>14}={c_sims[b]:.3f}" for b in BRANDS))

        pred, score, n_voters = vote(per_crop)
        lines.append(f"  IMAGE -> {pred}  (vote_score={score:.2f}, voters={n_voters})")

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
            ann = annotate(orig, meta["boxes"], per_crop, pred, gt,
                           font_box, font_title)
            ann.save(annotated_dir / f"{d.name}.png")

    n_decisive = tally["correct"] + tally["wrong"]
    acc = 100 * tally["correct"] / n_decisive if n_decisive else 0
    summary = ["", "=" * 78,
               f"STAGE 3 (dual-embedder ensemble)  floor={SIM_FLOOR}",
               "=" * 78]
    summary.append(
        f"  correct={tally['correct']}  wrong={tally['wrong']}  "
        f"uncertain={tally['uncertain']}  no_gt={tally['no_gt']}  "
        f"no_crops={tally['no_crops']}  -> {acc:.0f}% on {n_decisive} decisive w/ GT"
    )
    summary.append(f"  per-crop agreement: {n_agree} agree, {n_disagree} disagree "
                   f"({100*n_agree/(n_agree+n_disagree):.0f}% agree)")
    summary.append("")
    summary.append("Confusion (dual, GT-known images only):")
    for (gt, pred), n in sorted(confusion.items()):
        summary.append(f"  {gt:<22} -> {pred:<22}  (x{n})")
    summary.append("")
    summary.append(f"Annotated images: {annotated_dir}/")
    out = "\n".join(lines + summary)
    print("\n".join(summary))
    results_file.write_text(out)


if __name__ == "__main__":
    main()
