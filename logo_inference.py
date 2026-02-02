"""
Logo Detection and Recognition Pipeline

1. DETR detects logo bounding boxes
2. Crops are compared against pre-built embedding database
3. Best matching brand labels are assigned
4. Results are visualized with labeled bounding boxes
"""

from transformers import pipeline
from PIL import Image
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from db_embeddings import load_embedding_model, load_embeddings


# ============================================================
# Visualization
# ============================================================
def visualize_boxes(image, predictions, output_path):
    """Draw labeled bounding boxes on the image."""
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()

    for pred in predictions:
        box = pred.get("box", {})
        label = pred.get("label", "unknown")
        detr_score = pred.get("score", 0.0)
        similarity = pred.get("similarity", 0.0)

        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)

        # Draw bounding box
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color="green",
                linewidth=3,
            )
        )

        # Draw label
        text = f"{label} ({similarity:.2f})"
        ax.text(
            xmin,
            ymin - 5,
            text,
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="green", linewidth=2),
            color="green",
        )

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()
    print(f"Saved visualization to: {output_path}")


# ============================================================
# Main Inference Pipeline
# ============================================================
def run_inference(
    image_path,
    output_path,
    embeddings_path,
    model_type="dinov2",
    similarity_threshold=0.5,
    save_crops=False,
):
    """
    Run logo detection and recognition.
    
    Args:
        image_path: path to input image
        output_path: path to save output visualization
        embeddings_path: path to pre-built embeddings database (.pkl)
        model_type: embedding model used ("clip", "dinov2", or "siglip")
        similarity_threshold: minimum similarity to accept a match
        save_crops: if True, save cropped logo regions
    """
    print("=" * 60)
    print("Logo Detection and Recognition Pipeline")
    print("=" * 60)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_index = 0 if torch.cuda.is_available() else -1
    device = torch.device(device_str)
    print(f"Device: {device_str}")

    # --------------------------------------------------------
    # Step 1: Load DETR Logo Detector
    # --------------------------------------------------------
    print("\n[1/4] Loading DETR logo detector...")
    start_load = time.time()
    detector = pipeline(
        task="object-detection",
        model="Pravallika6/detr-finetuned-logo-detection_v2",
        device=device_index,
        use_fast=True,
    )
    print(f"  DETR loaded in {time.time() - start_load:.2f}s")

    # --------------------------------------------------------
    # Step 2: Load Embedding Model and Database
    # --------------------------------------------------------
    print(f"\n[2/4] Loading {model_type} embedding model...")
    start_load = time.time()
    model, processor, embed_fn = load_embedding_model(model_type, device_str)
    print(f"  Model loaded in {time.time() - start_load:.2f}s")

    print(f"\n[3/4] Loading embedding database from {embeddings_path}...")
    embeddings, metadata = load_embeddings(embeddings_path, device)
    
    brand_labels = list(embeddings.keys())
    print(f"  Database contains {len(brand_labels)} brands: {brand_labels}")

    # --------------------------------------------------------
    # Step 3: Detect Logos with DETR
    # --------------------------------------------------------
    print(f"\n[4/4] Running detection on {image_path}...")
    image = Image.open(image_path).convert("RGB")

    start_inference = time.time()
    detections = detector(image)
    print(f"  DETR found {len(detections)} logo regions in {time.time() - start_inference:.2f}s")

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_name = os.path.splitext(output_path)[0]

    # --------------------------------------------------------
    # Step 4: Match Each Detection to Database
    # --------------------------------------------------------
    print("\nMatching detections to database...")
    results = []

    for i, det in enumerate(detections):
        box = det.get("box", {})
        xmin, ymin = box.get("xmin", 0), box.get("ymin", 0)
        xmax, ymax = box.get("xmax", 0), box.get("ymax", 0)

        # Crop the detected region
        crop = image.crop((xmin, ymin, xmax, ymax))

        # Save crop if requested
        if save_crops:
            crop_path = f"{base_name}_crop_{i}.png"
            crop.save(crop_path)
            print(f"  Saved crop: {crop_path}")

        # Compute embedding for the crop
        crop_embedding = embed_fn(crop)

        # Compare with all brands in database
        similarities = {}
        for brand, brand_emb in embeddings.items():
            sim = F.cosine_similarity(crop_embedding, brand_emb).item()
            similarities[brand] = sim

        # Find best match
        best_brand = max(similarities, key=similarities.get)
        best_score = similarities[best_brand]

        # Print results for this detection
        print(f"\n  Detection {i}: DETR score={det.get('score', 0):.3f}")
        for brand, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            marker = " <-- BEST" if brand == best_brand else ""
            print(f"    {brand}: {sim:.4f}{marker}")

        # Accept or reject based on threshold
        if best_score >= similarity_threshold:
            result = {
                "box": box,
                "label": best_brand,
                "score": det.get("score", 0),
                "similarity": best_score,
                "all_similarities": similarities,
            }
            results.append(result)
            print(f"    -> Matched: {best_brand}")
        else:
            print(f"    -> Rejected (similarity {best_score:.3f} < threshold {similarity_threshold})")

    # --------------------------------------------------------
    # Step 5: Visualize Results
    # --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Input: {image_path}")
    print(f"DETR detections: {len(detections)}")
    print(f"Matched logos: {len(results)}")

    for i, res in enumerate(results):
        print(f"  {i+1}. {res['label']}: similarity={res['similarity']:.3f}, DETR={res['score']:.3f}")

    # Save visualization
    visualize_boxes(image, results, output_path)
    print(f"\nOutput saved to: {output_path}")

    return results


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logo detection and recognition")
    parser.add_argument(
        "--image",
        type=str,
        default="/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/test_images/picture_line.png",
        help="Input image path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/output/result.png",
        help="Output visualization path",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="/scratch/general/vast/u1475870/logo_detection/logo_embeddings.pkl",
        help="Path to embedding database (.pkl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2",
        choices=["clip", "dinov2", "siglip"],
        help="Embedding model (must match database)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Similarity threshold for matching",
    )
    parser.add_argument(
        "--save_crops",
        action="store_true",
        help="Save cropped logo regions",
    )

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        output_path=args.output,
        embeddings_path=args.embeddings,
        model_type=args.model,
        similarity_threshold=args.threshold,
        save_crops=args.save_crops,
    )
