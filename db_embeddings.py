"""
Logo Embeddings Database Builder

This module creates and manages a database of logo embeddings.
Supports multiple embedding models:
- CLIP (openai/clip-vit-large-patch14)
- DINOv2 (facebook/dinov2-base) - Best for visual similarity
- SigLIP (google/siglip-base-patch16-224)

Directory structures supported:
1. Flat: logos/brand1.png, logos/brand2.png
2. Subfolder: logos/brand1/img1.png, logos/brand1/img2.png
"""

from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModel,
    Dinov2Model, AutoImageProcessor,
)
from PIL import Image
import torch
import torch.nn.functional as F
import os
import pickle


# ============================================================
# Model Loading
# ============================================================
def load_embedding_model(model_type="dinov2", device="cuda"):
    """
    Load an embedding model.
    
    Args:
        model_type: "clip", "dinov2", or "siglip"
        device: torch device
        
    Returns:
        tuple: (model, processor, embed_fn)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    if model_type == "clip":
        model_name = "openai/clip-vit-large-patch14"
        print(f"Loading CLIP: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        def embed_fn(image):
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            return F.normalize(features, dim=-1)
        
    elif model_type == "dinov2":
        model_name = "facebook/dinov2-base"  # 768D embeddings
        print(f"Loading DINOv2: {model_name}")
        model = Dinov2Model.from_pretrained(model_name).to(device)
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        def embed_fn(image):
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token embedding
                features = outputs.last_hidden_state[:, 0, :]
            return F.normalize(features, dim=-1)
        
    elif model_type == "siglip":
        model_name = "google/siglip-base-patch16-224"
        print(f"Loading SigLIP: {model_name}")
        model = AutoModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        
        def embed_fn(image):
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            return F.normalize(features, dim=-1)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'clip', 'dinov2', or 'siglip'")
    
    model.eval()
    print(f"Model loaded on {device}")
    
    return model, processor, embed_fn


# ============================================================
# Load Logo Embeddings from Directory
# ============================================================
def load_logo_embeddings(logo_dir, embed_fn):
    """
    Load logo images and compute their embeddings.
    
    Supports two structures:
    1. Flat: logos/brand1.png, logos/brand2.png
    2. Subfolders: logos/brand1/img1.png, logos/brand1/img2.png
    
    For subfolders, embeddings are averaged per class.
    
    Args:
        logo_dir: directory containing logo images
        embed_fn: function that takes PIL Image and returns embedding tensor
        
    Returns:
        dict: brand_name -> normalized embedding tensor (1, D)
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
                    features = embed_fn(logo_img)
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
                features = embed_fn(logo_img)
                
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
# Save/Load Embeddings
# ============================================================
def save_embeddings(embeddings, output_path, metadata=None):
    """
    Save embeddings dictionary to a pickle file.
    
    Args:
        embeddings: dict of brand_name -> tensor
        output_path: path to save the pickle file
        metadata: optional dict with model info, etc.
    """
    data = {
        "embeddings": {k: v.cpu() for k, v in embeddings.items()},
        "metadata": metadata or {}
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved {len(embeddings)} brand embeddings to: {output_path}")


def load_embeddings(input_path, device=None):
    """
    Load embeddings dictionary from a pickle file.
    
    Args:
        input_path: path to the pickle file
        device: optional device to move tensors to
        
    Returns:
        tuple: (embeddings dict, metadata dict)
    """
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    
    # Handle both old format (just dict) and new format (dict with metadata)
    if isinstance(data, dict) and "embeddings" in data:
        embeddings = data["embeddings"]
        metadata = data.get("metadata", {})
    else:
        embeddings = data
        metadata = {}
    
    if device is not None:
        embeddings = {k: v.to(device) for k, v in embeddings.items()}
    
    print(f"Loaded {len(embeddings)} brand embeddings from: {input_path}")
    if metadata:
        print(f"  Model: {metadata.get('model_type', 'unknown')}")
    
    return embeddings, metadata


# ============================================================
# Build Embedding Database
# ============================================================
def build_embedding_database(logo_dir, output_path=None, model_type="dinov2"):
    """
    Build a logo embedding database from a directory of logo images.
    
    Args:
        logo_dir: directory containing logo images (flat or subfolder structure)
        output_path: optional path to save the embeddings pickle file
        model_type: "clip", "dinov2", or "siglip"
        
    Returns:
        dict: brand_name -> normalized embedding tensor
    """
    print(f"Building embedding database from: {logo_dir}")
    print(f"Using model: {model_type}")
    
    # Load model
    model, processor, embed_fn = load_embedding_model(model_type)
    
    # Load logo embeddings
    print(f"Processing logos from {logo_dir}...")
    embeddings = load_logo_embeddings(logo_dir, embed_fn)
    
    if len(embeddings) == 0:
        raise ValueError(f"No valid logo images found in {logo_dir}")
    
    print(f"\nBuilt database with {len(embeddings)} brands:")
    for brand in sorted(embeddings.keys()):
        emb_dim = embeddings[brand].shape[-1]
        print(f"  - {brand} ({emb_dim}D)")
    
    # Save if output path provided
    if output_path is not None:
        metadata = {
            "model_type": model_type,
            "embedding_dim": emb_dim,
            "num_brands": len(embeddings),
        }
        save_embeddings(embeddings, output_path, metadata)
    
    return embeddings


# ============================================================
# Similarity Search
# ============================================================
def find_similar_logos(query_image, embeddings, embed_fn, top_k=5):
    """
    Find the most similar logos to a query image.
    
    Args:
        query_image: PIL Image or path to image
        embeddings: dict of brand_name -> embedding tensor
        embed_fn: embedding function
        top_k: number of top matches to return
        
    Returns:
        list of (brand_name, similarity_score) tuples
    """
    if isinstance(query_image, str):
        query_image = Image.open(query_image).convert("RGB")
    
    query_emb = embed_fn(query_image)
    
    similarities = []
    for brand, brand_emb in embeddings.items():
        sim = F.cosine_similarity(query_emb, brand_emb).item()
        similarities.append((brand, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build logo embedding database")
    parser.add_argument("--logo_dir", type=str, 
                        default="/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/logos",
                        help="Directory containing logo images")
    parser.add_argument("--output", type=str,
                        default="/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/logo_embeddings.pkl",
                        help="Output pickle file path")
    parser.add_argument("--model", type=str, default="dinov2",
                        choices=["clip", "dinov2", "siglip"],
                        help="Embedding model to use (default: dinov2)")
    
    args = parser.parse_args()
    
    # Build and save the embedding database
    brand_db = build_embedding_database(
        logo_dir=args.logo_dir,
        output_path=args.output,
        model_type=args.model,
    )
