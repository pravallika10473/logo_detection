# Logo Detection Pipeline: Staged Debugging Findings

Test set: 18 images in `clean_test/test_images/` against 4 brands
(`america_first`, `mountain_america`, `utah_credit_union`, `utah_jazz`).
Original symptom: "everything matches mountain_america with ~0.05 margin"
across all detections.

We isolated the failure by stage:
1. Image quality (refs + tests)
2. Detector output (DETR, then GroundingDINO, then ensemble)
3. Similarity matching (DINOv2, CLIP, dual ensemble)

Final state: **88% accuracy** on 8 decisive predictions (7/8) using DETR+GDINO
ensemble crops + dual-embedder voting with per-embedder floors. Started at
75% with the symptom above.

---

## Stage 1 — Image quality and reference hygiene

### What we found
- The **original symptom (mountain_america dominance) was a contrast bias**, not
  a model issue. mountain_america refs had median contrast 80; utah_credit_union
  had 30. High-contrast refs produce sharper, more dominant centroids.
- `utah_credit_union` had only 4 refs, one outright blurry
  (`UFC_Logo_Horizontal-Boxed_Orange_RGB.jpg`, sharpness 66) and one near-flat
  (`unnamed.png`, contrast 22).
- `utah_jazz` was visually incoherent: 4 different color schemes (yellow/black,
  navy/orange, purple, green/distressed), 2 logo families (J-note alone vs
  "UTAH JAZZ" wordmark), inconsistent backgrounds.
- 5 of 18 test images are <300px short side. Logos in them are <30px after
  cropping — too small to embed meaningfully even before stage 3 runs.

### What hurt
- **Variant base picker first preferred image size over sharpness.**
  For `america_first` it picked `image.webp` (1024px, sharp 200) over
  `America_First_Credit_Union_Logo.jpg` (300px, sharp 1567). Variants
  inherit the base's softness.
- **Variants of a non-canonical base** (e.g., `images.png` for utah_jazz —
  yellow J-note on black) pulled the centroid into a region that didn't match
  what test images actually look like (purple jerseys, contextual photos).

### What we changed
- Quarantined the worst refs to `clean_test/_quarantined/`:
  utah_credit_union: `UFC_Logo_...`, `unnamed.png`, `images.png`;
  utah_jazz: `images.png` (yellow on black), `images5.jpeg` (green distressed),
  `utah-jazz-jersey-2023-...thumb.png` (B&W, 200x133).
- Synthesized 6 variants per brand from the cleanest base:
  `scale_0.5`, `scale_0.7`, `bright_up`, `bright_dn`, `rot_+5`, `blur_s1.5`.
  Deliberately excluded heavy rotations, mirroring, and hue shifts (those
  change identity, not just embedding location).
- Refined the picker to score by `sharpness * contrast` with a 200px short-side
  floor (size alone was misleading).

### Result (DINOv2 leave-one-out on refs)
- Before fixes: 90.7% (utah_jazz brand alone: 76.9%)
- After fixes: **95% combo / 92.5% patch** (utah_jazz alone: **100%**)
- Contrast disparity collapsed: all four brands now have median contrast 72-82
  (was 30-83).

---

## Stage 2 — Detector output

### Two detectors, complementary failure modes

**Fine-tuned DETR (`Pravallika6/detr-finetuned-logo-detection_v2`)**
- Over-fires on text and high-contrast rectangular regions: `bps1.png` got
  12 raw boxes, `spanish-fork-branch.png` got 24.
- Strong on small clean logos (`348s.jpg` 0.95 on a 51x32 logo).
- Misses on small contextual images: `first.jpeg`, `images.jpeg`,
  `images_jazz.jpeg` got nothing usable.

**Zero-shot GroundingDINO (`IDEA-Research/grounding-dino-base`,
prompt: `logo . sign . brand . emblem`)**
- Recovered `first.jpeg` (real America First sign found) and `images.jpeg`
  (jersey sponsor area) — DETR failures.
- Lost cases DETR handled cleanly: `258s.jpg`, `326d6595...`, `U5A4700`,
  `af.jpeg`, `imagesl.jpeg` (scores below floor).
- Boxed wrong objects on a couple of cases (`348s.jpg` boxed a car, not the
  logo).

### What hurt — and how we fixed it
- **Initial `MIN_SIDE = 40px` killed every clean small detection** including
  `348s.jpg` 0.95 / 51x32 and `Nevada-Reno-123.jpg` 0.99. The test images are
  small; logos in them are naturally small in absolute pixels.
- **Confidence-tiered min size**: high-confidence detections (score >= 0.85)
  only need 20px on a side; lower-confidence detections still need 28px.
  Recovered the small clean logos without re-admitting the 8x27 noise.
- DETR scores skew 0.5-0.99; GDINO scores skew 0.3-0.7. We used different
  `SCORE_MIN` per detector (DETR 0.50, GDINO 0.35) — uniform thresholds would
  drop one detector's good output entirely.
- **Cross-detector NMS at IoU 0.7** for the ensemble (high threshold so we
  preserve slightly-different views from each detector instead of suppressing
  them).

### Coverage after each variant
| Pipeline | Total kept crops | Images with no crops |
|---|---:|---:|
| DETR alone | 39 | 3 (`first`, `images`, `images_jazz`) |
| GDINO alone | 18 | 6 (lost some DETR-clean cases) |
| **Ensemble** | **55** | **1** (`images_jazz`) |

---

## Stage 3 — Similarity matching

### What we tried
- **DINOv2** (`facebook/dinov2-base`): CLS, patch-mean, and 0.3*CLS+0.7*patch
  combo. Patch-mean was best on this data.
- **CLIP-Large** (`openai/clip-vit-large-patch14`): Image-feature cosine sim.
- **Dual ensemble**: DINOv2 patch + CLIP-Large together.

### Findings
- **DINOv2 patch and CLIP have complementary failure modes.** DINOv2 is strong
  on small clean crops (258s small MA logo); CLIP is strong on text-heavy
  contextual crops (`first.jpeg`'s tight America First sign).
- CLIP cosine sims for **noisy DETR crops run much lower** than ref-vs-ref pairs
  (working range 0.55-0.65 vs ref intra mean 0.84). Tuning the floor to ref
  intra-class baseline (0.75) was wildly too aggressive — only 1 prediction
  out of 18.
- The `utah_jazz` ref centroid was an attractor in CLIP space too (every small
  noisy crop predicted utah_jazz) until we re-picked the base in Stage 1.
  Same root pattern as the original mountain_america bias in DINOv2.

### What hurt — and how we fixed it
- **"Highest single-crop similarity wins" is brittle.** False-positive crops
  can score higher than real-logo crops by chance. On `bps1.png`, crop #4
  (real check region) correctly matched america_first at 0.44, but crop #1
  (false positive on background) matched mountain_america at 0.54 and won.
- **Solution**: switched to **similarity-weighted voting** — each crop above
  a `SIM_FLOOR` contributes its top sim to that brand's tally; image-level
  prediction = max tally; UNCERTAIN if no crop passes the floor.
- **Naive averaging across embedders dilutes strong signals.** For
  `first.jpeg` the GDINO crop scored CLIP=0.696 (well above CLIP floor 0.60)
  but DINOv2=0.400 (below DINOv2 floor 0.50). Average = 0.548, just below the
  combined floor → UNCERTAIN. The weakest embedder dragged down the strongest.
- **Solution**: **per-embedder floors**. Each crop emits up to 2 independent
  votes, each with its own calibrated floor (DINOv2: 0.50, CLIP: 0.60). Brand
  tally sums all contributions. Either embedder can carry the day when it's
  confident; the other isn't required to agree.

### Accuracy progression
| Pipeline | Decisive | Correct | Wrong | Accuracy |
|---|---:|---:|---:|---:|
| Original (DETR + DINOv2 max-sim) | 8 | 6 | 2 | 75% |
| DETR + DINOv2 patch (weighted vote) | 6 | 5 | 1 | 83% |
| DETR + CLIP | 6 | 5 | 1 | 83% |
| GDINO + CLIP | 6 | 5 | 1 | 83% |
| **Ensemble + CLIP** | 7 | 6 | 1 | **86%** |
| **Ensemble + dual (per-embedder floors)** | **8** | **7** | **1** | **88%** |

---

## Final state

**Winning configuration**:
- Refs: 11 america_first / 12 mountain_america / 7 utah_credit_union / 10 utah_jazz
  (originals + 6 variants per brand from the sharpest base).
- Detector: DETR + GroundingDINO union, cross-NMS at IoU 0.7.
- Filter: score floors per detector, confidence-tiered min-side
  (>=20px if score>=0.85, else >=28px), aspect ratio in [0.2, 5.0],
  area <=80%, intra-detector NMS at IoU 0.5.
- Embedder: DINOv2-base patch-mean + CLIP-Large image features.
- Voting: per-embedder floors (DINOv2 0.50, CLIP 0.60); each crop emits
  up to 2 independent votes; brand tally = sum of contributions; UNCERTAIN
  when no crop passes any floor.

**Numbers**:
- 88% accuracy on 8 decisive predictions (7/8 correct).
- Coverage: 17/18 test images have at least one crop.
- 1 wrong prediction (`bps1.png`); 0 cases where the system predicts
  confidently and is wrong on a clean image.

**Output paths**:
- Best visualizations: `stage3_annotated_dual_ensemble/<stem>.png`
- Best per-crop log: `stage3_match_dual_results_ensemble.txt`
- Quarantined refs (recoverable): `clean_test/_quarantined/`

---

## Limitations

- **`bps1.png` (giveaway-check photo)**: Both DINOv2 and CLIP read the photo
  as mountain_america across most crops (CLIP: MA 0.62 vs AF 0.55). The big
  full-check crop correctly identifies AF at 0.36, but well below floor.
  The image is fundamentally a contextual photo, not a logo image.
- **`images_jazz.jpg` (300x168, no crops)**: Both detectors fail to confidently
  find the small jersey logo. Inherent limit of 30-300px logo regions in
  300px-wide images.
- **`utah_jazz` ↔ `america_first` confusability**: Both DINOv2 and CLIP show
  some confusion between these brands (predicted in Stage 1 verification).
  Both have navy on light backgrounds with text-heavy layouts. Embeddings
  collapse them in some regions of feature space.
- **Sub-30px logos** are inherently lossy when upscaled to 224 input. The
  pipeline accepts them through the confidence-tiered filter, but Stage 3
  often returns UNCERTAIN.
- **Per-image vote logic assumes one dominant logo per image.** Images with
  multiple legitimate brands (e.g., a sponsorship board) are not handled
  cleanly — current logic returns one image-level prediction.
- **Variant generation can't add new visual perspectives**, only perturbations
  of an existing base. Brands need at least one canonical, sharp, high-resolution
  reference for variants to be effective. utah_credit_union currently has
  only one usable original after quarantine.
- **Ground truth labels were assigned by eye** for 11 of 18 test images;
  7 are marked unknown in `GROUND_TRUTH` so accuracy is computed only over
  the labeled subset.

---

## Pipeline scripts (in execution order)

All scripts live under `tests/`. Outputs are written to `tests/results/`.
Each script does `os.chdir(Path(__file__).resolve().parent.parent)` so paths
to `clean_test/` and `tests/results/` resolve from the project root regardless
of how the script is invoked.

1. `tests/stage1_image_quality.py` — audit refs and tests; flag tiny/blurry/flat
2. `tests/stage1_synthesize_variants.py` — pick best base per brand, generate 6 variants
3. `tests/stage1_verify_refs.py` — leave-one-out check that refs are separable
4. `tests/stage2_threshold.py` — DETR + filter, writes `tests/results/stage2_filtered/`
5. `tests/stage2_threshold_gdino.py` — GroundingDINO + filter, writes `tests/results/stage2_filtered_gdino/`
6. `tests/stage2_ensemble.py` — union DETR + GDINO crops with cross-NMS,
   writes `tests/results/stage2_filtered_ensemble/`
7. `tests/stage3_match.py` — DINOv2 matching with weighted voting
   (`--crops-dir <dir>` selects which detector output)
8. `tests/stage3_match_clip.py` — CLIP matching with weighted voting
9. `tests/stage3_match_dual.py` — dual embedder ensemble with per-embedder floors
   (the winning configuration)

---

## How to improve each stage further

Generalized levers — independent of the specific failure cases on this dataset.

### Stage 1 (references)

What would move the needle:

- **Add canonical references per brand from official sources** (logo PNGs from
  brand sites, transparent backgrounds, vector-rendered to large sizes). One
  high-quality canonical ref typically helps more than many noisy refs because
  the centroid is dominated by what's most representative. Aim for >=3 sharp
  refs per brand before relying on variants.
- **Split brands with multiple visually distinct logo families.** When a brand
  has both a wordmark and an icon (e.g., utah_jazz: J-note vs "UTAH JAZZ"),
  the centroid averages incompatible visual identities. Treat them as two
  sub-brands and merge the prediction at output time. Same applies to
  primary/secondary color schemes.
- **Audit ref distribution against expected test conditions.** If test images
  are mostly contextual photos (jerseys, signs in scenes), references should
  include at least one in-context example, not only standalone-on-white logos.
  The geometry of "logo on jersey under stadium light" differs from "logo as
  PNG sticker."
- **Generate variants only for the perturbations you expect at inference**:
  scale, brightness, mild blur. Do not rotate beyond ~5 degrees, do not
  mirror, do not hue-shift. These change identity, not just embedding location.
  Cap variant count: more than ~6 variants from one base biases the centroid
  toward whatever quirks that base has.
- **Compute and log centroid quality** before downstream stages: leave-one-out
  accuracy + intra/inter similarity gap. If a brand's leave-one-out is below
  ~85%, no Stage 3 trick will rescue it on test data.

### Stage 2 (detection + filtering)

What would move the needle:

- **Calibrate score floors per detector empirically.** Detectors return
  different score distributions (DETR: 0.5-0.99; GDINO: 0.3-0.7). A uniform
  threshold either drops one detector's good output or admits the other's
  noise. Look at the score histogram on a held-out set and pick the elbow.
- **Use confidence-tiered min-size filters.** High-confidence detections
  deserve smaller-size leeway because the detector is more sure. A flat
  min-size threshold either kills small clean logos (high recall loss) or
  admits tiny noise (precision loss).
- **Ensemble detectors when they have complementary failure modes** (zero-shot
  vs fine-tuned, or different architectures). Cross-detector NMS at high IoU
  (~0.7) preserves slightly-different views from each detector instead of
  collapsing them. Different detectors framing the same logo differently can
  give independent voting signals downstream.
- **Track precision and recall separately** when tuning. For each filter
  knob, look at how it shifts both. The right setting depends on whether the
  downstream embedder is more hurt by noise (drop more) or by missing logos
  (drop less).
- **Consider an image-level pre-filter for tiny images.** Test images with
  short side <300px have logos that are sub-30px after cropping; downstream
  embedders can't usefully distinguish them. Either skip these, upscale them
  before detection (AI super-resolution), or accept they're out of scope.
- **Validate by hand on a sample.** Annotated overlays per image are cheap
  and tell you instantly whether the detector is finding logos vs framing
  background. Score thresholds can lie; overlays don't.

### Stage 3 (similarity matching)

What would move the needle:

- **Calibrate the similarity floor to the working range, not the ref baseline.**
  Ref-vs-ref intra-class similarities run high (often 0.80+); test crop sims
  run much lower (often 0.55-0.70) because crops are noisier than clean refs.
  A floor calibrated to ref-baseline will reject most real predictions.
- **Avoid "max single-crop similarity wins" voting.** False-positive crops
  can score higher than real-logo crops by chance alignment. Weighted voting
  across all valid crops (with a similarity floor) is more robust.
- **Per-embedder floors when ensembling embedders.** Different embedders have
  different working ranges (DINOv2 patch ~0.50, CLIP-L ~0.60). A uniform
  combined floor punishes whichever embedder runs lower. Per-embedder floors
  let each contribute when confident, no agreement required.
- **Use max-per-brand (not average) when combining per-brand similarities
  across embedders.** Averaging dilutes a strong signal from one embedder
  when the other is weak. Max preserves the best signal each embedder has
  to offer for each brand.
- **Add an UNCERTAIN class.** Better to abstain than to guess when no crop
  passes any embedder's floor. UNCERTAIN raises precision without changing
  the count of correct predictions.
- **Embedder ensembles beat single-embedder upgrades on small test sets.**
  CLIP and DINOv2 fail on different cases; combining them was a bigger win
  than swapping in a stronger single embedder. Try SigLIP next only if the
  ensemble plateaus.
- **For brands that confuse easily** (e.g., similar color and layout), add
  references that emphasize the discriminating features, or post-process
  with a brand-specific tiebreaker (color histogram, OCR on the crop).
  Embeddings alone can collapse visually similar brands.

### Cross-cutting

- **Always isolate failure by stage.** When the end-to-end output is wrong,
  the bug could be in refs, crops, or matching. Tools that only show the
  final answer hide the layer that's actually broken. The staged logs and
  per-stage visualizations in this pipeline (`tests/results/stage*/`) make
  it possible to walk a single image through every step.
- **Add ground truth lazily but completely for the cases you debug.** A
  partial GT (~60% labeled here) means accuracy numbers carry an error bar.
  When investigating a specific failure, label that image first, then
  evaluate.
- **Cache embeddings.** Re-embedding refs every script run (currently ~5
  seconds DINOv2 + ~30 seconds CLIP-L) wastes time when iterating on filter
  knobs. A small cache keyed by ref file hash would speed iteration.
