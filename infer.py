#!/usr/bin/env python3
"""
Logo detection + brand classification: end-to-end inference.

Loads the winning pipeline (DETR + GroundingDINO ensemble crops -> DINOv2 +
CLIP dual embedder voting), runs it on a single image or a directory, and
writes annotated PNGs (and optional per-image JSON).

Brands are discovered automatically: every subdirectory of --refs-dir whose
name does not start with "_" is treated as a brand, and all images inside
are used as references for that brand.

Usage:
    python infer.py path/to/image.jpg
    python infer.py path/to/image.jpg --refs-dir clean_test --output-dir out/
    python infer.py path/to/image_dir/ --refs-dir my_refs/ --json

Programmatic:
    from infer import LogoClassifier
    clf = LogoClassifier(refs_dir="clean_test")
    result = clf.predict("path/to/image.jpg")
    # -> {"prediction": "...", "score": ..., "voters": ..., "crops": [...]}
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


# ============================================================================
# Configuration (defaults match the winning Utah pipeline -> 88% accuracy)
# ============================================================================
#
# TUNING GUIDE
# ------------
# All knobs below are tunable. The defaults are calibrated to the Utah
# credit-union testbed (4 brands, ~250-2500px test images, clean PNG refs).
# For other domains, expect to adjust at least the *_FLOOR values after
# calibrating on a small labeled sample.
#
# Models -- swap to change the underlying backbones:
#   DETR_MODEL    : fine-tuned detector. Wrong domain -> retrain or set to
#                   a different fine-tune. Currently trained on credit-union
#                   storefronts.
#   GDINO_MODEL   : zero-shot detector. -base is fine for CPU; -large needs
#                   GPU but better recall.
#   GDINO_PROMPT  : the most important knob for GDINO. Domain-specific.
#                   Examples:
#                     credit-union storefront: "logo . sign . brand . emblem"
#                     stadium sponsors:        "logo . sign . brand . sponsor
#                                               . hoarding . advertisement"
#                     product packaging:       "logo . brand . label . trademark"
#   DINO_MODEL    : try -large for better fine-grained discrimination.
#   CLIP_MODEL    : try google/siglip-large-patch16-256 for better retrieval.
#
# Detection thresholds:
#   DETECT_THRESH : raw detector floor before filter. Keep low (0.15-0.25);
#                   the filter does the real culling.
#
# Per-detector filter (calibrated separately because score scales differ):
#   SCORE_MIN_DETR  (default 0.50): DETR scores skew 0.5-0.99.
#   SCORE_MIN_GDINO (default 0.35): GDINO scores skew 0.3-0.7. Always keep
#                   this LOWER than SCORE_MIN_DETR.
#   HIGHCONF_SCORE      (0.85): above this, detections get small-size leeway.
#   HIGHCONF_MIN_SIDE   (20  ): min crop side for high-conf detections (px).
#   MIN_SIDE            (28  ): min crop side for normal detections.
#                                Drop to 20 for tiny images, raise to 40+
#                                for high-res sets where logos are large.
#   MIN_AR / MAX_AR     (0.2 / 5.0): aspect ratio bounds.
#                                Raise MAX_AR to 8+ for horizontal banner
#                                logos (e.g. stadium hoardings).
#   MAX_AREA_PCT        (80  ): drops "whole-image" detections. Raise if
#                                your logos legitimately fill most of frame.
#   NMS_IOU             (0.5 ): within-detector NMS. Lower = more dedup.
#   CROSS_NMS_IOU       (0.7 ): across-detector NMS. HIGH on purpose --
#                                preserves slightly-different views from
#                                DETR vs GDINO so both can vote independently.
#
# Voting floors (biggest knob for precision vs UNCERTAIN rate):
#   DINO_FLOOR  (0.50): below this, DINOv2 abstains.
#   CLIP_FLOOR  (0.60): below this, CLIP abstains.
#                       Calibrate per-dataset. Reference intra-class similarity
#                       runs high (~0.78 DINOv2, ~0.84 CLIP); test-crop sims
#                       run lower (~0.50-0.65 DINOv2, ~0.55-0.70 CLIP) because
#                       crops are noisier than clean refs. Floor must sit in
#                       the working range, not the ref baseline.
#                       CLIP usually 0.10 higher than DINOv2 on the same data.
#
# Tuning recipes
# --------------
# Logos are tiny in test images (stadium hoardings, distant signs):
#   MIN_SIDE = 20, HIGHCONF_MIN_SIDE = 12, MAX_AR = 8
#   DINO_FLOOR = 0.45, CLIP_FLOOR = 0.55
#
# Too many false positives in output:
#   SCORE_MIN_DETR = 0.65, SCORE_MIN_GDINO = 0.45
#   raise both *_FLOOR by 0.05
#
# Too many UNCERTAIN, you want a guess:
#   DINO_FLOOR = 0.40, CLIP_FLOOR = 0.50
#   (expect more wrong predictions)
#
# Different domain entirely:
#   Update GDINO_PROMPT first (highest leverage)
#   Run on a labeled sample, look at the working range of similarities,
#   set floors at ~10-15% below that range's mean.
# ============================================================================

DETR_MODEL = "Pravallika6/detr-finetuned-logo-detection_v2"
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
GDINO_PROMPT = "logo . sign . brand . emblem"
DINO_MODEL = "facebook/dinov2-base"
CLIP_MODEL = "openai/clip-vit-large-patch14"

# Detection - low threshold, filter applies after
DETECT_THRESH = 0.20

# Filter rules
SCORE_MIN_DETR = 0.50
SCORE_MIN_GDINO = 0.35
HIGHCONF_SCORE = 0.85       # high-conf detections get smaller min-side allowed
HIGHCONF_MIN_SIDE = 20
MIN_SIDE = 28
MIN_AR = 0.2
MAX_AR = 5.0
MAX_AREA_PCT = 80.0
NMS_IOU = 0.5               # within-detector
CROSS_NMS_IOU = 0.7         # across detectors (high - preserves different views)

# Voting (per-embedder floors, sum-per-brand)
DINO_FLOOR = 0.50
CLIP_FLOOR = 0.60

# Visualization
BRAND_COLOR_PALETTE = [
    "red", "blue", "lime", "magenta", "orange", "cyan",
    "yellow", "purple", "pink", "turquoise"
]


# ============================================================================
# Helpers
# ============================================================================

def list_images(d: Path):
    return sorted(p for p in d.iterdir()
                  if p.is_file()
                  and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def iou(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0


def get_font(size=14):
    for c in ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def assign_brand_colors(brands):
    return {b: BRAND_COLOR_PALETTE[i % len(BRAND_COLOR_PALETTE)]
            for i, b in enumerate(brands)} | {"UNCERTAIN": "gray"}


# ============================================================================
# Core classifier
# ============================================================================

class LogoClassifier:
    """Loads all four models once, computes brand centroids once, then
    predicts on as many images as you call .predict() with."""

    def __init__(self,
                 refs_dir: Union[str, Path],
                 brands: Optional[List[str]] = None,
                 device: Optional[str] = None,
                 verbose: bool = True):
        self.refs_dir = Path(refs_dir)
        self.verbose = verbose
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Discover brands. Skip subdirs starting with "_" (convention for
        # quarantined / scratch dirs) and common known-non-brand names so
        # users with a refs dir that also contains test images don't have to
        # specify --brands explicitly.
        skip_names = {"test_images", "tests", "ground_truth", "gt", "outputs"}
        if brands is None:
            brands = sorted(d.name for d in self.refs_dir.iterdir()
                            if d.is_dir()
                            and not d.name.startswith("_")
                            and d.name.lower() not in skip_names)
        if not brands:
            raise ValueError(f"No brand subdirectories found in {self.refs_dir}")
        self.brands = brands
        self.colors = assign_brand_colors(brands)

        self._log(f"Brands: {brands}")
        self._log(f"Device: {self.device}")

        self._load_detectors()
        self._load_embedders()
        self._compute_centroids()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ---- model loading ----------------------------------------------------

    def _load_detectors(self):
        from transformers import (
            pipeline, AutoModelForZeroShotObjectDetection, AutoProcessor)
        self._log(f"Loading DETR ({DETR_MODEL})...")
        self.detr = pipeline(
            task="object-detection", model=DETR_MODEL,
            device=0 if self.device.type == "cuda" else -1, use_fast=True)
        self._log(f"Loading GroundingDINO ({GDINO_MODEL})...")
        self.gdino_proc = AutoProcessor.from_pretrained(GDINO_MODEL)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_MODEL).to(self.device).eval()

    def _load_embedders(self):
        from transformers import (
            AutoImageProcessor, Dinov2Model, CLIPModel, CLIPProcessor)
        self._log(f"Loading DINOv2 ({DINO_MODEL})...")
        self.dino = Dinov2Model.from_pretrained(DINO_MODEL).to(self.device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
        self._log(f"Loading CLIP ({CLIP_MODEL})...")
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    def _compute_centroids(self):
        self._log("Computing brand centroids from references...")
        self.dino_centroids, self.clip_centroids = {}, {}
        for brand in self.brands:
            d = self.refs_dir / brand
            d_embs, c_embs = [], []
            for p in list_images(d):
                img = Image.open(p).convert("RGB")
                d_embs.append(self._embed_dino(img))
                c_embs.append(self._embed_clip(img))
            if not d_embs:
                raise ValueError(
                    f"No reference images found in {d} for brand '{brand}'")
            self.dino_centroids[brand] = self._centroid(d_embs)
            self.clip_centroids[brand] = self._centroid(c_embs)
            self._log(f"  {brand}: {len(d_embs)} refs")

    # ---- low-level ops ----------------------------------------------------

    @staticmethod
    def _centroid(vecs):
        return F.normalize(torch.cat(vecs).mean(0, keepdim=True), dim=-1)

    def _embed_dino(self, img):
        inp = self.dino_proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.dino(**inp).last_hidden_state
        return F.normalize(h[:, 1:, :].mean(dim=1), dim=-1).cpu()

    def _embed_clip(self, img):
        inp = self.clip_proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            f = self.clip.get_image_features(**inp)
        return F.normalize(f, dim=-1).cpu()

    def _detect_detr(self, img):
        raw = self.detr(img, threshold=DETECT_THRESH)
        return [{"src": "detr", "score": float(d["score"]),
                 "x0": d["box"]["xmin"], "y0": d["box"]["ymin"],
                 "x1": d["box"]["xmax"], "y1": d["box"]["ymax"]}
                for d in raw]

    def _detect_gdino(self, img):
        inputs = self.gdino_proc(images=img, text=GDINO_PROMPT,
                                 return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gdino_model(**inputs)
        try:
            r = self.gdino_proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], box_threshold=DETECT_THRESH,
                target_sizes=[img.size[::-1]])[0]
        except TypeError:
            r = self.gdino_proc.post_process_grounded_object_detection(
                out, threshold=DETECT_THRESH, input_ids=inputs["input_ids"],
                target_sizes=[img.size[::-1]])[0]
        boxes = r["boxes"].cpu().tolist()
        scores = r["scores"].cpu().tolist()
        return [{"src": "gdino", "score": float(s),
                 "x0": float(b[0]), "y0": float(b[1]),
                 "x1": float(b[2]), "y1": float(b[3])}
                for b, s in zip(boxes, scores)]

    @staticmethod
    def _filter_one_detector(dets, W, H, score_min):
        kept = []
        for d in dets:
            w, h = d["x1"] - d["x0"], d["y1"] - d["y0"]
            ar = w / h if h else 0
            area_pct = 100 * (w * h) / (W * H) if W * H else 0
            if d["score"] < score_min:
                continue
            side_floor = (HIGHCONF_MIN_SIDE if d["score"] >= HIGHCONF_SCORE
                          else MIN_SIDE)
            if min(w, h) < side_floor or not (MIN_AR <= ar <= MAX_AR):
                continue
            if area_pct > MAX_AREA_PCT:
                continue
            kept.append({**d, "w": w, "h": h})
        kept.sort(key=lambda d: -d["score"])
        out = []
        for d in kept:
            bbox = (d["x0"], d["y0"], d["x1"], d["y1"])
            if any(iou(bbox, (k["x0"], k["y0"], k["x1"], k["y1"])) >= NMS_IOU
                   for k in out):
                continue
            out.append(d)
        return out

    @staticmethod
    def _cross_nms(boxes):
        boxes = sorted(boxes, key=lambda b: -b["score"])
        out = []
        for d in boxes:
            bbox = (d["x0"], d["y0"], d["x1"], d["y1"])
            if any(iou(bbox, (k["x0"], k["y0"], k["x1"], k["y1"])) >= CROSS_NMS_IOU
                   for k in out):
                continue
            out.append(d)
        return out

    def _per_brand_sims(self, emb, centroids):
        return {b: F.cosine_similarity(emb, c).item()
                for b, c in centroids.items()}

    def _vote(self, per_crop):
        tallies = defaultdict(float)
        voters = 0
        for c in per_crop:
            d_top = max(c["dino_sims"], key=c["dino_sims"].get)
            if c["dino_sims"][d_top] >= DINO_FLOOR:
                tallies[d_top] += c["dino_sims"][d_top]
                voters += 1
            c_top = max(c["clip_sims"], key=c["clip_sims"].get)
            if c["clip_sims"][c_top] >= CLIP_FLOOR:
                tallies[c_top] += c["clip_sims"][c_top]
                voters += 1
        if not tallies:
            return "UNCERTAIN", 0.0, 0
        best = max(tallies, key=tallies.get)
        return best, tallies[best], voters

    # ---- public API -------------------------------------------------------

    def predict(self, image: Union[str, Path, Image.Image]) -> Dict:
        """Run the full pipeline on one image. Returns a dict with the
        prediction + per-crop details."""
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img = Image.open(image).convert("RGB")
        else:
            image_path = None
            img = image.convert("RGB")
        W, H = img.size

        detr_kept = self._filter_one_detector(
            self._detect_detr(img), W, H, SCORE_MIN_DETR)
        gdino_kept = self._filter_one_detector(
            self._detect_gdino(img), W, H, SCORE_MIN_GDINO)
        ensemble = self._cross_nms(detr_kept + gdino_kept)

        per_crop = []
        for d in ensemble:
            crop = img.crop((d["x0"], d["y0"], d["x1"], d["y1"]))
            d_emb = self._embed_dino(crop)
            c_emb = self._embed_clip(crop)
            per_crop.append({
                "bbox": [int(d["x0"]), int(d["y0"]), int(d["x1"]), int(d["y1"])],
                "src": d["src"],
                "detector_score": float(d["score"]),
                "dino_sims": self._per_brand_sims(d_emb, self.dino_centroids),
                "clip_sims": self._per_brand_sims(c_emb, self.clip_centroids),
            })

        prediction, vote_score, voters = self._vote(per_crop)

        return {
            "image": image_path,
            "size": [W, H],
            "prediction": prediction,
            "vote_score": vote_score,
            "voters": voters,
            "n_raw_detections": len(detr_kept) + len(gdino_kept),
            "n_kept_after_cross_nms": len(ensemble),
            "crops": per_crop,
        }

    def annotate(self, image: Union[str, Path, Image.Image],
                 result: Dict) -> Image.Image:
        """Render winning bboxes on the original image. Pass the output of
        predict() as `result`."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        draw = ImageDraw.Draw(img)
        font_box = get_font(14)
        font_title = get_font(18)

        prediction = result["prediction"]
        title = f"PRED: {prediction}"
        if result.get("voters"):
            title += f"  (vote_score={result['vote_score']:.2f}, voters={result['voters']})"
        tw = draw.textlength(title, font=font_title)
        draw.rectangle([0, 0, tw + 10, 28], fill="black")
        draw.text((5, 4), title, fill="white", font=font_title)

        if prediction == "UNCERTAIN":
            return img

        for crop in result["crops"]:
            d_top = max(crop["dino_sims"], key=crop["dino_sims"].get)
            c_top = max(crop["clip_sims"], key=crop["clip_sims"].get)
            d_voted = (d_top == prediction
                       and crop["dino_sims"][d_top] >= DINO_FLOOR)
            c_voted = (c_top == prediction
                       and crop["clip_sims"][c_top] >= CLIP_FLOOR)
            if not (d_voted or c_voted):
                continue
            srcs, sims = [], []
            if d_voted: srcs.append("DINO"); sims.append(crop["dino_sims"][d_top])
            if c_voted: srcs.append("CLIP"); sims.append(crop["clip_sims"][c_top])
            color = self.colors.get(prediction, "white")
            x0, y0, x1, y1 = crop["bbox"]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
            label = f"{prediction} {max(sims):.2f} [{'+'.join(srcs)}]"
            tw = draw.textlength(label, font=font_box)
            ly = max(30, y0 - 20)
            draw.rectangle([x0, ly, x0 + tw + 6, ly + 20], fill=color)
            text_color = ("black" if color in ("orange", "yellow", "lime",
                                                "cyan", "white", "pink", "turquoise")
                          else "white")
            draw.text((x0 + 3, ly + 1), label, fill=text_color, font=font_box)
        return img


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Logo detection + brand classification (DETR + GDINO "
                    "ensemble crops -> DINOv2 + CLIP dual-embedder voting)")
    parser.add_argument("input",
                        help="Image file or directory of images")
    parser.add_argument("--refs-dir", default="clean_test",
                        help="Directory containing one subdir per brand "
                             "(default: clean_test)")
    parser.add_argument("--brands", nargs="+", default=None,
                        help="Restrict to specific brand subdirs (default: all)")
    parser.add_argument("--output-dir", default="inference_output",
                        help="Where to save annotated images "
                             "(default: inference_output/)")
    parser.add_argument("--json", action="store_true",
                        help="Also write per-image JSON with full crop details")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (default: auto-detect)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect input images
    if in_path.is_dir():
        images = list_images(in_path)
        if not images:
            print(f"No images found in {in_path}")
            return
    elif in_path.is_file():
        images = [in_path]
    else:
        print(f"Input not found: {in_path}")
        return

    clf = LogoClassifier(refs_dir=args.refs_dir, brands=args.brands,
                         device=args.device)

    print(f"\nProcessing {len(images)} image(s) -> {out_dir}/\n")
    counts = defaultdict(int)
    for i, p in enumerate(images, start=1):
        result = clf.predict(p)
        annotated = clf.annotate(p, result)
        out_png = out_dir / f"{p.stem}_pred.png"
        annotated.save(out_png)
        if args.json:
            (out_dir / f"{p.stem}.json").write_text(json.dumps(result, indent=2))
        counts[result["prediction"]] += 1
        print(f"[{i:>3}/{len(images)}] {p.name:<48}  -> {result['prediction']:<20} "
              f"(voters={result['voters']}, "
              f"vote_score={result['vote_score']:.2f}, "
              f"crops={result['n_kept_after_cross_nms']})  saved {out_png.name}")

    print("\n" + "=" * 60)
    print("Summary:")
    for pred in sorted(counts, key=lambda x: -counts[x]):
        print(f"  {pred:<20}  {counts[pred]}")
    print(f"\nAnnotated images: {out_dir}/")


if __name__ == "__main__":
    main()
