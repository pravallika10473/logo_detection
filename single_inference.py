from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


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
        
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                    fill=False, color='red', linewidth=2))
        
        text = f'{label}: {score:.2f}'
        ax.text(xmin, ymin - 5, text, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'),
                color='red')
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Saved visualization to: {output_path}")

def main(image_path, output_path, logo_dir, similarity_threshold=0.7):
    print("Starting inference...")
    
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_index = 0 if torch.cuda.is_available() else -1
    device = torch.device(device_str)
    print(f"Device set to use {device_str}")
    
    # Initialize DETR pipeline
    start_load = time.time()
    pipe = pipeline(
        task="object-detection",
        model="Pravallika6/detr-finetuned-logo-detection_v2",
        device=device_index,
        use_fast=True,
    )
    load_time = time.time() - start_load
    print(f"Model loading time: {load_time:.2f} seconds")
    
    # Load image and run DETR inference
    image = Image.open(image_path).convert("RGB")
    
    start_inference = time.time()
    predictions = pipe(image)
    inference_time = time.time() - start_inference
    print(f"Inference time: {inference_time:.2f} seconds")
    
    # Create DETR visualization (all predictions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_name = os.path.splitext(output_path)[0]
    detr_output = f"{base_name}_detr.png"
    visualize_boxes(image, predictions, detr_output, "DETR")
    
    # Extract bounding box images
    temp_dir = "temp_bboxes"
    os.makedirs(temp_dir, exist_ok=True)
    bbox_images = []
    for i, pred in enumerate(predictions):
        box = pred.get("box", {})
        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)
        bbox_image = image.crop((xmin, ymin, xmax, ymax))
        temp_path = os.path.join(temp_dir, f"bbox_{i}.png")
        bbox_image.save(temp_path)
        bbox_images.append((bbox_image, pred))
    
    # Load CLIP model and all logos from directory
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load all logos from directory
    print(f"Loading logos from {logo_dir}...")
    logo_files = []
    logo_features_list = []
    logo_labels = []
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    
    for filename in os.listdir(logo_dir):
        file_path = os.path.join(logo_dir, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                logo_image = Image.open(file_path).convert("RGB")
                logo_inputs = clip_processor(images=logo_image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    logo_features = clip_model.get_image_features(**logo_inputs)
                    logo_features = F.normalize(logo_features, dim=-1)
                
                logo_files.append(file_path)
                logo_features_list.append(logo_features)
                # Use filename without extension as label
                logo_label = os.path.splitext(filename)[0]
                logo_labels.append(logo_label)
                print(f"  Loaded logo: {filename} -> label: {logo_label}")
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")
    
    if len(logo_files) == 0:
        raise ValueError(f"No valid logo images found in {logo_dir}")
    
    print(f"Loaded {len(logo_files)} logos")
    
    # Compute cosine similarity for each bbox with all logos
    print("Computing CLIP similarities...")
    filtered_predictions = []
    for i, (bbox_image, pred) in enumerate(bbox_images):
        bbox_inputs = clip_processor(images=bbox_image, return_tensors="pt").to(device)
        with torch.no_grad():
            bbox_features = clip_model.get_image_features(**bbox_inputs)
            bbox_features = F.normalize(bbox_features, dim=-1)
            
            # Compute similarity with all logos
            similarities = []
            similarity_details = []
            for j, logo_features in enumerate(logo_features_list):
                # Ensure both tensors have the same shape for cosine similarity
                # Both should be [1, feature_dim] after normalization
                similarity = F.cosine_similarity(logo_features, bbox_features, dim=-1).item()
                similarities.append(similarity)
                similarity_details.append((logo_labels[j], similarity))
            
            # Find the logo with highest similarity
            max_similarity = max(similarities)
            best_logo_idx = similarities.index(max_similarity)
            best_logo_label = logo_labels[best_logo_idx]
        
        pred['clip_similarity'] = max_similarity
        pred['best_logo'] = best_logo_label
        pred['all_similarities'] = dict(similarity_details)
        
        # Print detailed similarity information
        print(f"  Bbox {i}: DETR score={pred.get('score', 0.0):.3f}")
        for logo_label, sim in sorted(similarity_details, key=lambda x: x[1], reverse=True):
            marker = " <-- BEST" if logo_label == best_logo_label else ""
            print(f"    {logo_label}: {sim:.4f}{marker}")
        print(f"    Best match: {best_logo_label} (similarity: {max_similarity:.4f})")
        
        if max_similarity > similarity_threshold:
            pred['label'] = best_logo_label
            filtered_predictions.append(pred)
        else:
            print(f"    -> Filtered out (below threshold {similarity_threshold})")
    
    # Create CLIP-filtered visualization
    clip_output = f"{base_name}_clip.png"
    visualize_boxes(image, filtered_predictions, clip_output, "CLIP")
    
    # Print summary
    print(f"\nProcessed {image_path}:")
    print(f"DETR detected: {len(predictions)} logos")
    print(f"CLIP filtered: {len(filtered_predictions)} logos (threshold: {similarity_threshold})")
    for i, pred in enumerate(filtered_predictions):
        label = pred.get('label', 'unknown')
        print(f"  {i+1}. {label}: DETR={pred.get('score', 0.0):.3f}, CLIP={pred.get('clip_similarity', 0.0):.3f}")
    

if __name__ == "__main__":
    image_path = "/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/test_images/utah_first.png"
    output_path = "/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/output_final/utah_first.png"
    logo_dir = "/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/logos"
    similarity_threshold = 0.7
    
    main(image_path, output_path, logo_dir, similarity_threshold)