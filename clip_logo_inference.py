from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


# ============================================================
# Visualization
# ============================================================
def visualize_boxes(image, predictions, output_path, title_suffix=""):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()

    for pred in predictions:
        box = pred.get("box", {})
        label = pred.get("label", "")
        score = pred.get("score", 0.0)

        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)

        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color="red",
                linewidth=2,
            )
        )

        sim = pred.get("clip_similarity", None)
        if sim is not None:
            text = f"{label}: DETR={score:.2f}, CLIP={sim:.2f}"
        else:
            text = f"{label}: {score:.2f}"

        ax.text(
            xmin,
            ymin - 5,
            text,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"),
            color="red",
        )

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()
    print(f"Saved visualization to: {output_path}")


# ============================================================
# Load Logo Embeddings (supports subfolders for multiple refs)
# ============================================================
def load_logo_embeddings(logo_dir, processor, model, device):
    """
    Supports two structures:
    1. Flat: logos/brand1.png, logos/brand2.png
    2. Subfolders: logos/brand1/img1.png, logos/brand1/img2.png
    
    For subfolders, embeddings are averaged per class.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    
    logo_embeddings = {}  # label -> list of embeddings
    
    for item in os.listdir(logo_dir):
        item_path = os.path.join(logo_dir, item)
        
        if os.path.isdir(item_path):
            # Subfolder structure: folder name is the label
            label = item
            logo_embeddings[label] = []
            
            for fname in os.listdir(item_path):
                if not any(fname.lower().endswith(ext) for ext in image_extensions):
                    continue
                    
                file_path = os.path.join(item_path, fname)
                try:
                    logo_img = Image.open(file_path).convert("RGB")
                    inputs = processor(images=logo_img, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        features = model.get_image_features(**inputs)
                        features = F.normalize(features, dim=-1)
                    
                    logo_embeddings[label].append(features)
                    print(f"  Loaded: {label}/{fname}")
                except Exception as e:
                    print(f"  Warning: Could not load {file_path}: {e}")
                    
        elif os.path.isfile(item_path):
            # Flat structure: filename (without ext) is the label
            if not any(item.lower().endswith(ext) for ext in image_extensions):
                continue
                
            label = os.path.splitext(item)[0]
            
            try:
                logo_img = Image.open(item_path).convert("RGB")
                inputs = processor(images=logo_img, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                    features = F.normalize(features, dim=-1)
                
                if label not in logo_embeddings:
                    logo_embeddings[label] = []
                logo_embeddings[label].append(features)
                print(f"  Loaded: {item}")
            except Exception as e:
                print(f"  Warning: Could not load {item}: {e}")
    
    # Average embeddings per class
    averaged_embeddings = {}
    for label, emb_list in logo_embeddings.items():
        if len(emb_list) > 0:
            stacked = torch.cat(emb_list, dim=0)  # (N, D)
            avg_emb = stacked.mean(dim=0, keepdim=True)  # (1, D)
            avg_emb = F.normalize(avg_emb, dim=-1)
            averaged_embeddings[label] = avg_emb
            print(f"  {label}: {len(emb_list)} reference(s), averaged")
    
    return averaged_embeddings


# ============================================================
# MAIN
# ============================================================
def main(image_path, output_path, logo_dir, similarity_threshold=0.75):
    print("Starting inference...")

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_index = 0 if torch.cuda.is_available() else -1
    device = torch.device(device_str)
    print(f"Device set to use {device_str}")

    # --------------------------------------------------------
    # DETR LOGO DETECTOR
    # --------------------------------------------------------
    start_load = time.time()
    pipe = pipeline(
        task="object-detection",
        model="Pravallika6/detr-finetuned-logo-detection_v2",
        device=device_index,
        use_fast=True,
    )
    load_time = time.time() - start_load
    print(f"DETR model loading time: {load_time:.2f} seconds")

    # Load image and run DETR inference
    image = Image.open(image_path).convert("RGB")

    start_inference = time.time()
    predictions = pipe(image)
    inference_time = time.time() - start_inference
    print(f"DETR inference time: {inference_time:.2f} seconds")
    print(f"DETR detected {len(predictions)} boxes")

    # --------------------------------------------------------
    # CREATE DETR VISUALIZATION
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_name = os.path.splitext(output_path)[0]

    detr_output = f"{base_name}_detr.png"
    visualize_boxes(image, predictions, detr_output, "DETR")

    # --------------------------------------------------------
    # CROP BOUNDING BOXES AND SAVE
    # --------------------------------------------------------
    bbox_images = []
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    for i, pred in enumerate(predictions):
        box = pred.get("box", {})
        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)
        crop = image.crop((xmin, ymin, xmax, ymax))
        bbox_images.append((crop, pred))
        
        # Save cropped bbox
        bbox_filename = f"{base_name}_detected_bbox_{i}.png"
        crop.save(bbox_filename)
        print(f"Saved cropped bbox to: {bbox_filename}")

    # --------------------------------------------------------
    # LOAD CLIP
    # --------------------------------------------------------
    print("Loading CLIP model...")
    start_clip_load = time.time()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    clip_load_time = time.time() - start_clip_load
    print(f"CLIP model loading time: {clip_load_time:.2f} seconds")

    # --------------------------------------------------------
    # LOAD LOGO EMBEDDINGS
    # --------------------------------------------------------
    print(f"Loading logos from {logo_dir}...")
    logo_embeddings = load_logo_embeddings(logo_dir, clip_processor, clip_model, device)
    
    if len(logo_embeddings) == 0:
        raise ValueError(f"No valid logo images found in {logo_dir}")
    
    print(f"Loaded {len(logo_embeddings)} logo classes")
    
    logo_labels = list(logo_embeddings.keys())
    logo_features_list = [logo_embeddings[label] for label in logo_labels]

    # --------------------------------------------------------
    # CLIP SIMILARITY MATCHING
    # --------------------------------------------------------
    print("Computing CLIP similarities...")
    filtered_predictions = []

    for i, (bbox_img, pred) in enumerate(bbox_images):
        inputs = clip_processor(images=bbox_img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            bbox_features = clip_model.get_image_features(**inputs)
            bbox_features = F.normalize(bbox_features, dim=-1)

        # Compute similarity with all logos
        similarities = []
        similarity_details = []
        for j, logo_features in enumerate(logo_features_list):
            sim = F.cosine_similarity(logo_features, bbox_features).item()
            similarities.append(sim)
            similarity_details.append((logo_labels[j], sim))

        # Find best match
        best_idx = int(torch.tensor(similarities).argmax())
        best_score = similarities[best_idx]
        best_label = logo_labels[best_idx]

        pred["clip_similarity"] = best_score
        pred["best_logo"] = best_label
        pred["all_similarities"] = dict(similarity_details)

        # Print detailed info
        print(f"  Bbox {i}: DETR score={pred.get('score', 0.0):.3f}")
        for logo_label, sim in sorted(similarity_details, key=lambda x: x[1], reverse=True):
            marker = " <-- BEST" if logo_label == best_label else ""
            print(f"    {logo_label}: {sim:.4f}{marker}")

        if best_score >= similarity_threshold:
            pred["label"] = best_label
            filtered_predictions.append(pred)
        else:
            print(f"    -> Filtered out (below threshold {similarity_threshold})")

    # --------------------------------------------------------
    # CREATE CLIP VISUALIZATION
    # --------------------------------------------------------
    clip_output = f"{base_name}_clip.png"
    visualize_boxes(image, filtered_predictions, clip_output, "CLIP")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print(f"\nProcessed {image_path}:")
    print(f"DETR detected: {len(predictions)} logos")
    print(f"CLIP filtered: {len(filtered_predictions)} logos (threshold: {similarity_threshold})")
    for i, pred in enumerate(filtered_predictions):
        label = pred.get("label", "unknown")
        detr_score = pred.get("score", 0.0)
        clip_score = pred.get("clip_similarity", 0.0)
        print(f"  {i+1}. {label}: DETR={detr_score:.3f}, CLIP={clip_score:.3f}")


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    image_path = "/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/test_images/utah_first.png"
    output_path = "/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/output_clip/utah_first.png"
    logo_dir = "/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/logos"

    main(
        image_path=image_path,
        output_path=output_path,
        logo_dir=logo_dir,
        similarity_threshold=0.75,
    )
