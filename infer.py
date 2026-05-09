#!/usr/bin/env python3
"""
Logo detection + brand classification: end-to-end inference.

Loads the winning pipeline (DETR + GroundingDINO ensemble crops -> DINOv2 +
CLIP dual embedder voting), runs it on a single image or a directory, and
writes annotated PNGs (and optional per-image JSON).

Brands are discovered automatically: every subdirectory of --refs-dir whose
name does not start with "_" is treated as a brand, and all images inside
are used as references for that brand.

Features:
  - Reference embeddings cached on disk by file hash + model ID, so re-runs
    skip the ~30s ref-embedding step when refs haven't changed.
  - Crop-level abstain: a crop only votes if its top-brand similarity is both
    above SIM_FLOOR AND has a margin of MARGIN_FLOOR over the second brand.
    This drops false-positive non-logo crops where all brands score similarly.
  - Multi-brand prediction: returns every brand whose tally is at least
    MULTI_BRAND_RATIO * top_brand_tally. Annotated images color each bbox by
    its voting brand. Set --multi-brand-ratio to 1.0 for old single-brand mode.
  - Every tunable constant is exposed as a CLI flag (see --help).

Usage:
    python infer.py path/to/image.jpg
    python infer.py path/to/image.jpg --refs-dir clean_test --output-dir out/
    python infer.py path/to/image_dir/ --refs-dir my_refs/ --json
    python infer.py img.jpg --dino-floor 0.45 --clip-floor 0.55  # tune from CLI

Programmatic:
    from infer import LogoClassifier
    clf = LogoClassifier(refs_dir="clean_test")
    result = clf.predict("path/to/image.jpg")
    # -> {"prediction": "...", "all_brands": [...], "voters": ..., "crops": [...]}
"""

import argparse
import hashlib
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
# All knobs below are tunable - via CLI (--help shows every flag) or by
# constructing LogoClassifier with kwargs. The defaults are calibrated to
# the Utah credit-union testbed (4 brands, ~250-2500px test images, clean
# PNG refs). For other domains, expect to adjust at least the *_FLOOR
# values after calibrating on a small labeled sample.
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
#   SCORE_MIN_DETR  (0.50): DETR scores skew 0.5-0.99.
#   SCORE_MIN_GDINO (0.35): GDINO scores skew 0.3-0.7. Keep this LOWER
#                           than SCORE_MIN_DETR.
#   HIGHCONF_SCORE      (0.85): above this, detections get small-size leeway.
#   HIGHCONF_MIN_SIDE   (20  ): min crop side for high-conf detections (px).
#   MIN_SIDE            (28  ): min crop side for normal detections.
#   MIN_AR / MAX_AR     (0.2 / 5.0): aspect ratio bounds. Raise MAX_AR to 8+
#                                     for horizontal banner logos.
#   MAX_AREA_PCT        (80  ): drops "whole-image" detections.
#   NMS_IOU             (0.5 ): within-detector NMS.
#   CROSS_NMS_IOU       (0.7 ): across-detector NMS. HIGH on purpose --
#                                preserves slightly-different views from
#                                DETR vs GDINO so both can vote independently.
#
# Voting (per-embedder floors + margin gates + multi-brand reporting):
#   DINO_FLOOR        (0.50): below this, DINOv2 doesn't vote.
#   CLIP_FLOOR        (0.60): below this, CLIP doesn't vote.
#                             Calibrate per-dataset. Reference intra-class
#                             similarity runs ~0.78 (DINO) / ~0.84 (CLIP);
#                             test-crop sims run ~0.50-0.65 / ~0.55-0.70.
#                             CLIP usually 0.10 higher than DINOv2.
#   DINO_MARGIN_FLOOR (0.03): a crop also needs (top_sim - 2nd_sim) >= this
#                             for the embedder to vote. Drops non-logo crops
#                             where all brands score similarly.
#   CLIP_MARGIN_FLOOR (0.03): same idea for CLIP.
#   MULTI_BRAND_RATIO (0.50): a brand is reported as a co-prediction if its
#                             tally >= ratio * top_brand_tally. 1.0 = old
#                             single-brand behavior; 0.0 = report any brand
#                             with at least one vote.
#
# Cache:
#   EMBEDDING_CACHE_DIR (.embedding_cache): per-image embeddings stored on
#                       disk, keyed by file content hash + model id. Skipped
#                       transparently when refs change (different hash ->
#                       different cache entry).
#
# Tuning recipes
# --------------
# Logos are tiny in test images:
#   --min-side 20 --highconf-min-side 12 --max-ar 8
#   --dino-floor 0.45 --clip-floor 0.55
#
# Too many false positives in output:
#   --score-min-detr 0.65 --score-min-gdino 0.45
#   --dino-margin-floor 0.05 --clip-margin-floor 0.05
#
# Too many UNCERTAIN, you want a guess:
#   --dino-floor 0.40 --clip-floor 0.50
#
# Multi-brand images (each photo can have several sponsors):
#   --multi-brand-ratio 0.5     (default)
#   or --multi-brand-ratio 0.7  (only co-predict close-second brands)
#
# Different domain entirely:
#   --gdino-prompt "logo . sign . brand . sponsor . hoarding . advertisement"
#   then re-calibrate floors against a labeled sample.
# ============================================================================

DETR_MODEL = "Pravallika6/detr-finetuned-logo-detection_v2"
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
GDINO_PROMPT = "logo . sign . brand . emblem"
DINO_MODEL = "facebook/dinov2-base"
CLIP_MODEL = "openai/clip-vit-large-patch14"

DETECT_THRESH = 0.20

SCORE_MIN_DETR = 0.50
SCORE_MIN_GDINO = 0.35
HIGHCONF_SCORE = 0.85
HIGHCONF_MIN_SIDE = 20
MIN_SIDE = 28
MIN_AR = 0.2
MAX_AR = 5.0
MAX_AREA_PCT = 80.0
NMS_IOU = 0.5
CROSS_NMS_IOU = 0.7

DINO_FLOOR = 0.50
CLIP_FLOOR = 0.60
DINO_MARGIN_FLOOR = 0.03
CLIP_MARGIN_FLOOR = 0.03
MULTI_BRAND_RATIO = 0.50

EMBEDDING_CACHE_DIR = ".embedding_cache"

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


def _model_slug(model_id: str) -> str:
    """Filesystem-safe short name for a model ID."""
    return model_id.replace("/", "__").replace("-", "_")


# ============================================================================
# Embedding cache
# ============================================================================

class EmbeddingCache:
    """Per-image embedding cache keyed by SHA1(file bytes) + model id.
    Cache hits skip the model forward pass; cache misses compute and save."""

    def __init__(self, cache_dir: Union[str, Path], enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, image_path: Path, model_id: str) -> Path:
        with open(image_path, "rb") as f:
            digest = hashlib.sha1(f.read()).hexdigest()[:16]
        return self.cache_dir / _model_slug(model_id) / f"{digest}.pt"

    def get(self, image_path: Path, model_id: str):
        if not self.enabled:
            return None
        p = self._key(image_path, model_id)
        if p.exists():
            try:
                tensor = torch.load(p, map_location="cpu", weights_only=True)
                self.hits += 1
                return tensor
            except Exception:
                return None
        return None

    def put(self, image_path: Path, model_id: str, tensor: torch.Tensor):
        if not self.enabled:
            return
        p = self._key(image_path, model_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, p)
        self.misses += 1


# ============================================================================
# Core classifier
# ============================================================================

class LogoClassifier:
    """Loads all four models once, computes brand centroids once (with cache),
    then predicts on as many images as you call .predict() with.

    All tunable constants are accepted as kwargs (defaults are the module-level
    constants). Useful for programmatic per-call overrides without globals."""

    def __init__(self,
                 refs_dir: Union[str, Path],
                 brands: Optional[List[str]] = None,
                 device: Optional[str] = None,
                 verbose: bool = True,
                 *,
                 # models
                 detr_model: str = DETR_MODEL,
                 gdino_model: str = GDINO_MODEL,
                 gdino_prompt: str = GDINO_PROMPT,
                 dino_model: str = DINO_MODEL,
                 clip_model: str = CLIP_MODEL,
                 # detection
                 detect_thresh: float = DETECT_THRESH,
                 # filter
                 score_min_detr: float = SCORE_MIN_DETR,
                 score_min_gdino: float = SCORE_MIN_GDINO,
                 highconf_score: float = HIGHCONF_SCORE,
                 highconf_min_side: int = HIGHCONF_MIN_SIDE,
                 min_side: int = MIN_SIDE,
                 min_ar: float = MIN_AR,
                 max_ar: float = MAX_AR,
                 max_area_pct: float = MAX_AREA_PCT,
                 nms_iou: float = NMS_IOU,
                 cross_nms_iou: float = CROSS_NMS_IOU,
                 # voting
                 dino_floor: float = DINO_FLOOR,
                 clip_floor: float = CLIP_FLOOR,
                 dino_margin_floor: float = DINO_MARGIN_FLOOR,
                 clip_margin_floor: float = CLIP_MARGIN_FLOOR,
                 multi_brand_ratio: float = MULTI_BRAND_RATIO,
                 # cache
                 cache_dir: Union[str, Path] = EMBEDDING_CACHE_DIR,
                 use_cache: bool = True):
        self.refs_dir = Path(refs_dir)
        self.verbose = verbose
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Store every tunable on self
        self.detr_model = detr_model
        self.gdino_model = gdino_model
        self.gdino_prompt = gdino_prompt
        self.dino_model = dino_model
        self.clip_model = clip_model
        self.detect_thresh = detect_thresh
        self.score_min_detr = score_min_detr
        self.score_min_gdino = score_min_gdino
        self.highconf_score = highconf_score
        self.highconf_min_side = highconf_min_side
        self.min_side = min_side
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.max_area_pct = max_area_pct
        self.nms_iou = nms_iou
        self.cross_nms_iou = cross_nms_iou
        self.dino_floor = dino_floor
        self.clip_floor = clip_floor
        self.dino_margin_floor = dino_margin_floor
        self.clip_margin_floor = clip_margin_floor
        self.multi_brand_ratio = multi_brand_ratio

        # Discover brands. Skip "_*" subdirs (convention) and known-non-brand
        # names so users with refs dirs that also contain test images don't
        # have to specify --brands explicitly.
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

        # Cache
        self.cache = EmbeddingCache(cache_dir, enabled=use_cache)
        if use_cache:
            self._log(f"Embedding cache: {self.cache.cache_dir}/")

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
        self._log(f"Loading DETR ({self.detr_model})...")
        self.detr = pipeline(
            task="object-detection", model=self.detr_model,
            device=0 if self.device.type == "cuda" else -1, use_fast=True)
        self._log(f"Loading GroundingDINO ({self.gdino_model})...")
        self.gdino_proc = AutoProcessor.from_pretrained(self.gdino_model)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.gdino_model).to(self.device).eval()

    def _load_embedders(self):
        from transformers import (
            AutoImageProcessor, Dinov2Model, CLIPModel, CLIPProcessor)
        self._log(f"Loading DINOv2 ({self.dino_model})...")
        self.dino = Dinov2Model.from_pretrained(self.dino_model).to(self.device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(self.dino_model)
        self._log(f"Loading CLIP ({self.clip_model})...")
        self.clip = CLIPModel.from_pretrained(self.clip_model).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(self.clip_model)

    def _compute_centroids(self):
        self._log("Computing brand centroids from references...")
        self.dino_centroids, self.clip_centroids = {}, {}
        for brand in self.brands:
            d = self.refs_dir / brand
            d_embs, c_embs = [], []
            for p in list_images(d):
                d_emb = self.cache.get(p, self.dino_model)
                if d_emb is None:
                    img = Image.open(p).convert("RGB")
                    d_emb = self._embed_dino(img)
                    self.cache.put(p, self.dino_model, d_emb)
                c_emb = self.cache.get(p, self.clip_model)
                if c_emb is None:
                    img = Image.open(p).convert("RGB")
                    c_emb = self._embed_clip(img)
                    self.cache.put(p, self.clip_model, c_emb)
                d_embs.append(d_emb)
                c_embs.append(c_emb)
            if not d_embs:
                raise ValueError(
                    f"No reference images found in {d} for brand '{brand}'")
            self.dino_centroids[brand] = self._centroid(d_embs)
            self.clip_centroids[brand] = self._centroid(c_embs)
            self._log(f"  {brand}: {len(d_embs)} refs")
        if self.cache.enabled:
            self._log(f"  cache: {self.cache.hits} hits, {self.cache.misses} misses")

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
        raw = self.detr(img, threshold=self.detect_thresh)
        return [{"src": "detr", "score": float(d["score"]),
                 "x0": d["box"]["xmin"], "y0": d["box"]["ymin"],
                 "x1": d["box"]["xmax"], "y1": d["box"]["ymax"]}
                for d in raw]

    def _detect_gdino(self, img):
        inputs = self.gdino_proc(images=img, text=self.gdino_prompt,
                                 return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gdino(**inputs)
        try:
            r = self.gdino_proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], box_threshold=self.detect_thresh,
                target_sizes=[img.size[::-1]])[0]
        except TypeError:
            r = self.gdino_proc.post_process_grounded_object_detection(
                out, threshold=self.detect_thresh, input_ids=inputs["input_ids"],
                target_sizes=[img.size[::-1]])[0]
        boxes = r["boxes"].cpu().tolist()
        scores = r["scores"].cpu().tolist()
        return [{"src": "gdino", "score": float(s),
                 "x0": float(b[0]), "y0": float(b[1]),
                 "x1": float(b[2]), "y1": float(b[3])}
                for b, s in zip(boxes, scores)]

    def _filter_one_detector(self, dets, W, H, score_min):
        kept = []
        for d in dets:
            w, h = d["x1"] - d["x0"], d["y1"] - d["y0"]
            ar = w / h if h else 0
            area_pct = 100 * (w * h) / (W * H) if W * H else 0
            if d["score"] < score_min:
                continue
            side_floor = (self.highconf_min_side if d["score"] >= self.highconf_score
                          else self.min_side)
            if min(w, h) < side_floor or not (self.min_ar <= ar <= self.max_ar):
                continue
            if area_pct > self.max_area_pct:
                continue
            kept.append({**d, "w": w, "h": h})
        kept.sort(key=lambda d: -d["score"])
        out = []
        for d in kept:
            bbox = (d["x0"], d["y0"], d["x1"], d["y1"])
            if any(iou(bbox, (k["x0"], k["y0"], k["x1"], k["y1"])) >= self.nms_iou
                   for k in out):
                continue
            out.append(d)
        return out

    def _cross_nms(self, boxes):
        boxes = sorted(boxes, key=lambda b: -b["score"])
        out = []
        for d in boxes:
            bbox = (d["x0"], d["y0"], d["x1"], d["y1"])
            if any(iou(bbox, (k["x0"], k["y0"], k["x1"], k["y1"])) >= self.cross_nms_iou
                   for k in out):
                continue
            out.append(d)
        return out

    def _per_brand_sims(self, emb, centroids):
        return {b: F.cosine_similarity(emb, c).item()
                for b, c in centroids.items()}

    @staticmethod
    def _embedder_vote(sims: Dict[str, float], floor: float, margin_floor: float):
        """Returns (top_brand, top_sim) if this embedder votes, else None."""
        if not sims:
            return None
        ranked = sorted(sims.items(), key=lambda x: -x[1])
        top_brand, top_sim = ranked[0]
        if top_sim < floor:
            return None
        margin = top_sim - (ranked[1][1] if len(ranked) > 1 else 0)
        if margin < margin_floor:
            return None
        return top_brand, top_sim

    def _vote(self, per_crop):
        """Each crop emits up to 2 independent votes (DINOv2 + CLIP), each
        with its own floor + margin gate. Brand tally = sum of contributing
        sims. Returns sorted list of {brand, score, voters}."""
        tallies = defaultdict(lambda: {"score": 0.0, "voters": 0})
        for c in per_crop:
            d_vote = self._embedder_vote(
                c["dino_sims"], self.dino_floor, self.dino_margin_floor)
            if d_vote:
                brand, sim = d_vote
                tallies[brand]["score"] += sim
                tallies[brand]["voters"] += 1
            c_vote = self._embedder_vote(
                c["clip_sims"], self.clip_floor, self.clip_margin_floor)
            if c_vote:
                brand, sim = c_vote
                tallies[brand]["score"] += sim
                tallies[brand]["voters"] += 1
        all_brands = sorted(
            [{"brand": b, "score": v["score"], "voters": v["voters"]}
             for b, v in tallies.items()],
            key=lambda x: -x["score"])
        return all_brands

    def _select_multi_brands(self, all_brands):
        """Filter all_brands to those within multi_brand_ratio of the top."""
        if not all_brands:
            return []
        top = all_brands[0]["score"]
        return [b for b in all_brands
                if b["score"] >= self.multi_brand_ratio * top]

    # ---- public API -------------------------------------------------------

    def predict(self, image: Union[str, Path, Image.Image]) -> Dict:
        """Run the full pipeline on one image. Returns:
            {
                "image":              str or None,
                "size":               [W, H],
                "prediction":         top brand or "UNCERTAIN",
                "all_brands":         [{brand, score, voters}, ...]  (all that voted),
                "multi_brands":       [{brand, score, voters}, ...]  (within multi_brand_ratio of top),
                "vote_score":         float (top brand's score),
                "voters":             int (total embedder votes cast),
                "n_raw_detections":   int,
                "n_kept_after_cross_nms": int,
                "crops":              [{bbox, src, detector_score, dino_sims, clip_sims}, ...]
            }
        """
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img = Image.open(image).convert("RGB")
        else:
            image_path = None
            img = image.convert("RGB")
        W, H = img.size

        detr_kept = self._filter_one_detector(
            self._detect_detr(img), W, H, self.score_min_detr)
        gdino_kept = self._filter_one_detector(
            self._detect_gdino(img), W, H, self.score_min_gdino)
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

        all_brands = self._vote(per_crop)
        multi_brands = self._select_multi_brands(all_brands)

        if all_brands:
            prediction = all_brands[0]["brand"]
            vote_score = all_brands[0]["score"]
        else:
            prediction = "UNCERTAIN"
            vote_score = 0.0
        total_voters = sum(b["voters"] for b in all_brands)

        return {
            "image": image_path,
            "size": [W, H],
            "prediction": prediction,
            "all_brands": all_brands,
            "multi_brands": multi_brands,
            "vote_score": vote_score,
            "voters": total_voters,
            "n_raw_detections": len(detr_kept) + len(gdino_kept),
            "n_kept_after_cross_nms": len(ensemble),
            "crops": per_crop,
        }

    def annotate(self, image: Union[str, Path, Image.Image],
                 result: Dict) -> Image.Image:
        """Render winning bboxes on the original image. Each kept bbox is
        colored by the brand it voted for (in multi-brand mode this can
        differ per box). Pass the output of predict() as `result`."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        draw = ImageDraw.Draw(img)
        font_box = get_font(14)
        font_title = get_font(18)

        multi_brands = result.get("multi_brands", [])
        multi_set = {b["brand"] for b in multi_brands}

        if not multi_brands:
            title = "PRED: UNCERTAIN"
        elif len(multi_brands) == 1:
            title = f"PRED: {multi_brands[0]['brand']}"
        else:
            title = "PRED: " + " + ".join(b["brand"] for b in multi_brands)
        if result.get("voters"):
            title += f"  ({result['voters']} voters)"

        tw = draw.textlength(title, font=font_title)
        draw.rectangle([0, 0, tw + 10, 28], fill="black")
        draw.text((5, 4), title, fill="white", font=font_title)

        if not multi_brands:
            return img

        for crop in result["crops"]:
            d_vote = self._embedder_vote(
                crop["dino_sims"], self.dino_floor, self.dino_margin_floor)
            c_vote = self._embedder_vote(
                crop["clip_sims"], self.clip_floor, self.clip_margin_floor)

            # Decide which brand to draw this crop as. Prefer agreement; if
            # they disagree, prefer whichever is in multi_set with the higher
            # margin. If neither voted brand is in multi_set, skip.
            crop_brand, crop_sim, srcs = None, 0.0, []
            d_in = d_vote and d_vote[0] in multi_set
            c_in = c_vote and c_vote[0] in multi_set
            if d_vote and c_vote and d_vote[0] == c_vote[0] and d_in:
                crop_brand = d_vote[0]
                crop_sim = max(d_vote[1], c_vote[1])
                srcs = ["DINO", "CLIP"]
            elif d_in and c_in and d_vote[0] != c_vote[0]:
                # disagreement, both in multi_set - pick higher sim
                if d_vote[1] >= c_vote[1]:
                    crop_brand, crop_sim, srcs = d_vote[0], d_vote[1], ["DINO"]
                else:
                    crop_brand, crop_sim, srcs = c_vote[0], c_vote[1], ["CLIP"]
            elif d_in:
                crop_brand, crop_sim, srcs = d_vote[0], d_vote[1], ["DINO"]
            elif c_in:
                crop_brand, crop_sim, srcs = c_vote[0], c_vote[1], ["CLIP"]
            else:
                continue

            color = self.colors.get(crop_brand, "white")
            x0, y0, x1, y1 = crop["bbox"]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
            label = f"{crop_brand} {crop_sim:.2f} [{'+'.join(srcs)}]"
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

def _build_parser():
    p = argparse.ArgumentParser(
        description="Logo detection + brand classification (DETR + GDINO "
                    "ensemble crops -> DINOv2 + CLIP dual-embedder voting). "
                    "All tunable constants are exposed as flags below; defaults "
                    "match the winning Utah pipeline. See module docstring for "
                    "tuning recipes.")
    p.add_argument("input", help="Image file or directory of images")

    g = p.add_argument_group("inputs/outputs")
    g.add_argument("--refs-dir", default="clean_test",
                   help="Directory containing one subdir per brand "
                        "(default: clean_test)")
    g.add_argument("--brands", nargs="+", default=None,
                   help="Restrict to specific brand subdirs (default: auto-discover)")
    g.add_argument("--output-dir", default="inference_output",
                   help="Where to save annotated images (default: inference_output/)")
    g.add_argument("--json", action="store_true",
                   help="Also write per-image JSON with full crop details")
    g.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    g.add_argument("--no-cache", action="store_true",
                   help="Disable embedding cache (recompute every run)")
    g.add_argument("--cache-dir", default=EMBEDDING_CACHE_DIR,
                   help=f"Embedding cache directory (default: {EMBEDDING_CACHE_DIR})")

    g = p.add_argument_group("models")
    g.add_argument("--detr-model", default=DETR_MODEL)
    g.add_argument("--gdino-model", default=GDINO_MODEL)
    g.add_argument("--gdino-prompt", default=GDINO_PROMPT,
                   help="Text prompt for GroundingDINO (highest-leverage knob "
                        "for new domains)")
    g.add_argument("--dino-model", default=DINO_MODEL)
    g.add_argument("--clip-model", default=CLIP_MODEL)

    g = p.add_argument_group("detection")
    g.add_argument("--detect-thresh", type=float, default=DETECT_THRESH,
                   help=f"Raw detector score floor before filter (default: {DETECT_THRESH})")

    g = p.add_argument_group("filter")
    g.add_argument("--score-min-detr", type=float, default=SCORE_MIN_DETR,
                   help=f"DETR score floor (default: {SCORE_MIN_DETR})")
    g.add_argument("--score-min-gdino", type=float, default=SCORE_MIN_GDINO,
                   help=f"GDINO score floor (default: {SCORE_MIN_GDINO})")
    g.add_argument("--highconf-score", type=float, default=HIGHCONF_SCORE,
                   help=f"Above this, detections get smaller-side leeway "
                        f"(default: {HIGHCONF_SCORE})")
    g.add_argument("--highconf-min-side", type=int, default=HIGHCONF_MIN_SIDE,
                   help=f"Min crop side in px for high-conf detections "
                        f"(default: {HIGHCONF_MIN_SIDE})")
    g.add_argument("--min-side", type=int, default=MIN_SIDE,
                   help=f"Min crop side in px for normal detections "
                        f"(default: {MIN_SIDE})")
    g.add_argument("--min-ar", type=float, default=MIN_AR,
                   help=f"Min aspect ratio (default: {MIN_AR})")
    g.add_argument("--max-ar", type=float, default=MAX_AR,
                   help=f"Max aspect ratio. Raise to 8+ for banner logos "
                        f"(default: {MAX_AR})")
    g.add_argument("--max-area-pct", type=float, default=MAX_AREA_PCT,
                   help=f"Drops crops covering more than this %% of image "
                        f"(default: {MAX_AREA_PCT})")
    g.add_argument("--nms-iou", type=float, default=NMS_IOU,
                   help=f"Within-detector NMS IoU (default: {NMS_IOU})")
    g.add_argument("--cross-nms-iou", type=float, default=CROSS_NMS_IOU,
                   help=f"Across-detector NMS IoU (default: {CROSS_NMS_IOU})")

    g = p.add_argument_group("voting")
    g.add_argument("--dino-floor", type=float, default=DINO_FLOOR,
                   help=f"DINOv2 sim floor below which it abstains "
                        f"(default: {DINO_FLOOR})")
    g.add_argument("--clip-floor", type=float, default=CLIP_FLOOR,
                   help=f"CLIP sim floor below which it abstains "
                        f"(default: {CLIP_FLOOR})")
    g.add_argument("--dino-margin-floor", type=float, default=DINO_MARGIN_FLOOR,
                   help=f"Min (top_sim - 2nd_sim) for DINOv2 to vote "
                        f"(default: {DINO_MARGIN_FLOOR})")
    g.add_argument("--clip-margin-floor", type=float, default=CLIP_MARGIN_FLOOR,
                   help=f"Min (top_sim - 2nd_sim) for CLIP to vote "
                        f"(default: {CLIP_MARGIN_FLOOR})")
    g.add_argument("--multi-brand-ratio", type=float, default=MULTI_BRAND_RATIO,
                   help=f"Report a brand as co-prediction if its tally >= "
                        f"ratio * top_brand_tally. Set to 1.0 for old "
                        f"single-brand-only behavior (default: {MULTI_BRAND_RATIO})")
    return p


def main():
    args = _build_parser().parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    clf = LogoClassifier(
        refs_dir=args.refs_dir, brands=args.brands, device=args.device,
        detr_model=args.detr_model, gdino_model=args.gdino_model,
        gdino_prompt=args.gdino_prompt, dino_model=args.dino_model,
        clip_model=args.clip_model, detect_thresh=args.detect_thresh,
        score_min_detr=args.score_min_detr, score_min_gdino=args.score_min_gdino,
        highconf_score=args.highconf_score, highconf_min_side=args.highconf_min_side,
        min_side=args.min_side, min_ar=args.min_ar, max_ar=args.max_ar,
        max_area_pct=args.max_area_pct, nms_iou=args.nms_iou,
        cross_nms_iou=args.cross_nms_iou, dino_floor=args.dino_floor,
        clip_floor=args.clip_floor, dino_margin_floor=args.dino_margin_floor,
        clip_margin_floor=args.clip_margin_floor,
        multi_brand_ratio=args.multi_brand_ratio,
        cache_dir=args.cache_dir, use_cache=not args.no_cache)

    print(f"\nProcessing {len(images)} image(s) -> {out_dir}/\n")
    pred_counts = defaultdict(int)
    for i, p in enumerate(images, start=1):
        result = clf.predict(p)
        annotated = clf.annotate(p, result)
        out_png = out_dir / f"{p.stem}_pred.png"
        annotated.save(out_png)
        if args.json:
            (out_dir / f"{p.stem}.json").write_text(json.dumps(result, indent=2))
        # display string: multi-brand uses "+", single uses brand alone
        mb = result.get("multi_brands", [])
        if not mb:
            display = "UNCERTAIN"
        else:
            display = "+".join(b["brand"] for b in mb)
        pred_counts[display] += 1
        print(f"[{i:>3}/{len(images)}] {p.name:<48}  -> {display:<30} "
              f"(voters={result['voters']}, "
              f"vote_score={result['vote_score']:.2f}, "
              f"crops={result['n_kept_after_cross_nms']})  saved {out_png.name}")

    print("\n" + "=" * 60)
    print("Summary:")
    for pred in sorted(pred_counts, key=lambda x: -pred_counts[x]):
        print(f"  {pred:<30}  {pred_counts[pred]}")
    print(f"\nAnnotated images: {out_dir}/")


if __name__ == "__main__":
    main()
