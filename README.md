# Logo Detection

Detect and classify brand logos in images using an ensemble of detectors
(fine-tuned DETR + GroundingDINO) and a dual embedder (DINOv2 + CLIP) for
similarity matching against a reference set.

Reaches **88% accuracy** on the Utah credit-union testbed (4 brands, 18 test
images). See [`findings.md`](findings.md) for the full debugging writeup,
per-stage improvement guide, and limitations.

---

## Quick start

```bash
# Single image
python infer.py path/to/image.jpg --refs-dir clean_test

# Batch
python infer.py path/to/image_dir/ --refs-dir clean_test --output-dir out/ --json
```

References live in `<refs-dir>/<brand_name>/*.jpg`. Brand names are
auto-discovered from subdirectory names (skips `_*`, `test_images`, `tests`,
`ground_truth`, `gt`, `outputs`).

Programmatic use:

```python
from infer import LogoClassifier

clf = LogoClassifier(refs_dir="clean_test")
result = clf.predict("path/to/image.jpg")
# result = {"prediction": "...", "vote_score": ..., "voters": ..., "crops": [...]}

annotated = clf.annotate("path/to/image.jpg", result)
annotated.save("out.png")
```

---

## How it works

```
test image
    │
    ├──► DETR (Pravallika6/detr-finetuned-logo-detection_v2)
    ├──► GroundingDINO (IDEA-Research/grounding-dino-base, prompt
    │    "logo . sign . brand . emblem")
    │
    │    Both run independently; each filtered with its own score floor,
    │    confidence-tiered min-size, aspect ratio bounds, and intra-detector
    │    NMS at IoU 0.5.
    │
    ├──► Cross-detector NMS at IoU 0.7 (high - preserves slightly-different
    │    views from each detector instead of collapsing them)
    │
    └──► For each surviving crop:
           - DINOv2 patch-mean embedding -> cosine sim vs each brand centroid
           - CLIP-Large image embedding   -> cosine sim vs each brand centroid

         Vote: each crop emits up to 2 independent votes. DINOv2 votes for
         its top brand if sim >= 0.50; CLIP votes for its top brand if sim
         >= 0.60. Brand tally = sum of contributing similarities. Image
         prediction = argmax. UNCERTAIN if no crop passes any floor.
```

Brand centroids are precomputed from references on classifier init.
Subsequent inference is detector + embedder forward passes only.

---

## File structure

```
.
├── README.md                       # this file
├── findings.md                     # full pipeline writeup + improvement guide
├── infer.py                        # standalone end-to-end inference (production)
│
├── clean_test/                     # Utah credit-union testbed (gitignored)
│   ├── america_first/              #   reference images per brand
│   ├── mountain_america/
│   ├── utah_credit_union/
│   ├── utah_jazz/
│   ├── _quarantined/               #   refs dropped by stage1 audit (recoverable)
│   └── test_images/                #   18 test images
│
├── photonode/                      # Burnley FC sponsor data (Barnfield/Vertu, gitignored)
│   ├── barnfield_reference_images/
│   ├── vertu_reference_images/
│   └── burnley_test_images/        #   516 match photos
│
└── tests/                          # staged debugging pipeline
    ├── stage1_image_quality.py     #   audit refs + tests (resolution, sharpness, contrast)
    ├── stage1_synthesize_variants.py  # generate 6 variants per brand from cleanest base
    ├── stage1_verify_refs.py       #   leave-one-out separability check (DINOv2)
    ├── stage2_threshold.py         #   DETR + filter
    ├── stage2_threshold_gdino.py   #   GroundingDINO + filter
    ├── stage2_ensemble.py          #   cross-detector NMS union
    ├── stage3_match.py             #   DINOv2-only matching
    ├── stage3_match_clip.py        #   CLIP-only matching
    ├── stage3_match_dual.py        #   WINNING: dual embedder + per-embedder voting
    ├── photonode_dual.py           #   end-to-end runner for the photonode dataset
    └── results/                    #   text logs (committed) + annotated PNGs (gitignored)
        ├── stage*_results*.txt
        ├── stage2_filtered*/       #   crops + metadata per detector
        └── stage3_annotated*/      #   per-image predictions visualized
```

The staged debugging scripts (`tests/stage*.py`) all `os.chdir` to the project
root, so they can be run from anywhere:

```bash
python tests/stage1_image_quality.py
python tests/stage2_threshold.py
python tests/stage2_threshold_gdino.py
python tests/stage2_ensemble.py
python tests/stage3_match_dual.py --crops-dir tests/results/stage2_filtered_ensemble
```

---

## Tunable parameters

All 16 tunable constants are documented at the top of [`infer.py`](infer.py)
with what each does, default values, and recipes for common scenarios:

- **Models**: detector + embedder model IDs, GDINO prompt
- **Detection thresholds**: initial score floor, per-detector score floors
- **Filter rules**: min crop side (with confidence tiers), aspect ratio bounds,
  area cap, NMS IoU thresholds (within and across detectors)
- **Voting floors**: DINOv2 floor, CLIP floor

The most important knobs by impact:

| Knob | Effect |
|---|---|
| `GDINO_PROMPT` | Domain-specific. For sports sponsors, use `"logo . sign . brand . sponsor . hoarding . advertisement"` |
| `DINO_FLOOR` / `CLIP_FLOOR` | Higher = more UNCERTAIN, fewer wrong. Lower = more guesses. Calibrate per dataset. |
| `MIN_SIDE` | Drop to 20 for tiny-logo data; raise to 40+ for high-res |
| `MAX_AR` | Raise to 8+ for horizontal banner logos |

---

## Reference set requirements

For reliable matching:

- **3-5 reference images per brand** minimum (originals)
- **500px+ short side** (DINOv2 / CLIP resize to 224x224 internally;
  smaller refs become mostly interpolated pixels)
- **Sharp** (Laplacian variance >= 1500 for at least one ref per brand)
- **Decent contrast** (luminance std >= 30; lower causes centroid bias)
- **Aspect ratio within 1.5:1** of square (sliver crops get distorted on resize)
- **Background diversity** if test images come from varied contexts
  (transparent + a couple of solid-bg variants is usually enough)

If reference centroids can't separate themselves in leave-one-out
verification (`tests/stage1_verify_refs.py`), no Stage 2/3 trick will save
the test-set accuracy. Fix references first.

---

## Pipeline limitations

- **One brand per image.** The current vote returns a single image-level
  prediction. Multi-logo images get the strongest match, not all brands.
- **Non-logo crops still get assigned a brand.** A false-positive box on
  background may have its top similarity above the floor and contribute
  to the vote.
- **Sub-30px logos** are inherently lossy after the 224x224 embedder resize.
  The pipeline accepts them through the confidence-tier filter, but
  Stage 3 often returns UNCERTAIN.
- **Domain shift between refs and test crops** (e.g., refs are clean PNGs
  but test crops are tilted phone photos) hurts disproportionately.
  Use refs from the same visual domain as expected test crops.

See [`findings.md`](findings.md) for what each stage's limitations are
specifically and what would move the needle further.

---

## Related scripts (legacy / experimental)

These predate the staged debugging pipeline and are kept for reference:

- `clip.py`, `clip_logo_inference.py` — early CLIP-only experiments
- `db_embeddings.py` — embedding database experiments
- `inference.py`, `single_inference.py`, `logo_inference.py` — older inference
  scripts
- `iou.py`, `object_detection.py` — early detection experiments
- `slurms/` — SLURM scripts for CHPC cluster runs
