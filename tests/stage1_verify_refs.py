#!/usr/bin/env python3
"""
Stage 1 verification: are the new references separable?

Loads every ref (originals + variants), embeds with DINOv2 (CLS, patch-mean,
and a 0.3*CLS + 0.7*patch combo), then for each ref asks:
  Does it match its own brand's centroid better than any other brand's?

Reports per-brand leave-one-out accuracy and an intra/inter similarity gap.
If gap is large and accuracy is high -> refs are good, move to Stage 2.
If accuracy is low -> the refs themselves can't separate the brands; no
amount of better DETR cropping will fix matching.
"""

import os
import statistics
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F


CLEAN_DIR = Path("clean_test")
BRANDS = ["america_first", "mountain_america", "utah_credit_union", "utah_jazz"]


def load_refs():
    refs = {}
    for brand in BRANDS:
        d = CLEAN_DIR / brand
        files = sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        refs[brand] = [(p.name, Image.open(p).convert("RGB")) for p in files]
        print(f"  {brand}: {len(refs[brand])} refs")
    return refs


def embed_all(refs, model, proc, device):
    out = {}
    for brand, items in refs.items():
        out[brand] = []
        for name, img in items:
            inputs = proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                h = model(**inputs).last_hidden_state
            cls = F.normalize(h[:, 0, :], dim=-1)
            patch = F.normalize(h[:, 1:, :].mean(dim=1), dim=-1)
            out[brand].append({"name": name, "cls": cls.cpu(), "patch": patch.cpu()})
    return out


def centroid(vecs):
    return F.normalize(torch.cat(vecs).mean(0, keepdim=True), dim=-1)


def leave_one_out(emb, key):
    """For each ref, build centroids excluding it (own class) or including all (other classes)
    and check if its own centroid wins.
    Returns: per-brand correct/total + list of misses with margins."""
    results = {b: {"correct": 0, "total": 0, "misses": []} for b in emb}

    for brand, items in emb.items():
        for i, q in enumerate(items):
            same_vecs = [r[key] for j, r in enumerate(items) if j != i]
            if not same_vecs:
                continue
            same_c = centroid(same_vecs)
            same_sim = F.cosine_similarity(q[key], same_c).item()

            best_other_brand, best_other_sim = None, -1.0
            for other in emb:
                if other == brand:
                    continue
                other_c = centroid([r[key] for r in emb[other]])
                s = F.cosine_similarity(q[key], other_c).item()
                if s > best_other_sim:
                    best_other_sim, best_other_brand = s, other

            ok = same_sim > best_other_sim
            results[brand]["total"] += 1
            if ok:
                results[brand]["correct"] += 1
            else:
                results[brand]["misses"].append(
                    (q["name"], same_sim, best_other_brand, best_other_sim)
                )
    return results


def separation(emb, key):
    intra, inter = [], []
    brands = list(emb.keys())
    for b1 in brands:
        for b2 in brands:
            for r1 in emb[b1]:
                for r2 in emb[b2]:
                    if r1 is r2:
                        continue
                    s = F.cosine_similarity(r1[key], r2[key]).item()
                    (intra if b1 == b2 else inter).append(s)
    return intra, inter


def report(label, results, intra, inter):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    total_c, total_n = 0, 0
    for brand, r in results.items():
        pct = 100 * r["correct"] / r["total"] if r["total"] else 0
        print(f"  {brand:<22} {r['correct']:>2}/{r['total']:<2}  ({pct:>5.1f}%)")
        for name, ss, ob, os_ in r["misses"]:
            print(f"      MISS {name:<55}  same={ss:.3f}  other({ob})={os_:.3f}  margin={os_-ss:+.3f}")
        total_c += r["correct"]
        total_n += r["total"]
    overall = 100 * total_c / total_n if total_n else 0
    print(f"  {'OVERALL':<22} {total_c:>2}/{total_n:<2}  ({overall:>5.1f}%)")

    intra_mean = statistics.mean(intra)
    inter_mean = statistics.mean(inter)
    gap = intra_mean - inter_mean
    print(f"\n  intra-class similarity: mean={intra_mean:.3f}  min={min(intra):.3f}  max={max(intra):.3f}")
    print(f"  inter-class similarity: mean={inter_mean:.3f}  min={min(inter):.3f}  max={max(inter):.3f}")
    print(f"  gap (intra-inter):      {gap:+.3f}")
    if min(intra) > max(inter):
        print(f"  PERFECTLY SEPARABLE  (threshold {(min(intra)+max(inter))/2:.3f})")
    else:
        overlap = sum(1 for s in intra if s <= max(inter))
        print(f"  OVERLAP: {overlap}/{len(intra)} intra-class pairs fall in inter-class range")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("Loading DINOv2-base...")
    from transformers import AutoImageProcessor, Dinov2Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    print("\nLoading references...")
    refs = load_refs()
    print("\nEmbedding...")
    emb = embed_all(refs, model, proc, device)

    # Add a combined embedding: 0.3*CLS + 0.7*patch (matches improved_pipeline.py)
    for items in emb.values():
        for r in items:
            combo = F.normalize(0.3 * r["cls"] + 0.7 * r["patch"], dim=-1)
            r["combo"] = combo

    for key, label in [
        ("cls",   "DINOv2 CLS only"),
        ("patch", "DINOv2 patch-mean only"),
        ("combo", "DINOv2 0.3*CLS + 0.7*patch"),
    ]:
        results = leave_one_out(emb, key)
        intra, inter = separation(emb, key)
        report(label, results, intra, inter)


if __name__ == "__main__":
    main()
