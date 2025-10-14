from transformers import pipeline
from PIL import Image
import time
import torch
import matplotlib.pyplot as plt
import os


def main(image_path, output_path):
    print("Starting inference...")
    
    # Print CUDA availability
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_index = 0 if torch.cuda.is_available() else -1
    print(f"Device set to use {device_str}")
    
    # Initialize the pipeline
    start_load = time.time()
    pipe = pipeline(
        task="object-detection",
        model="Pravallika6/detr-finetuned-logo-detection_v2",
        device=device_index,
        use_fast=True,
    )
    load_time = time.time() - start_load
    print(f"Model loading time: {load_time:.2f} seconds")
    
    # Load image and run inference
    image = Image.open(image_path).convert("RGB")
    
    start_inference = time.time()
    predictions = pipe(image)
    inference_time = time.time() - start_inference
    print(f"Inference time: {inference_time:.2f} seconds")
    
    # Create visualization with predicted boxes
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    # Plot predicted boxes with confidence scores
    for pred in predictions:
        box = pred.get("box", {})
        label = pred.get("label", "")
        score = pred.get("score", 0.0)
        
        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)
        
        # Draw bounding box
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                    fill=False, color='red', linewidth=2))
        
        # Add confidence score label
        text = f'{label}: {score:.2f}'
        ax.text(xmin, ymin - 5, text, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'),
                color='red')
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    print(f"Saved visualization to: {output_path}")
    
    # Print summary
    print(f"\nProcessed {image_path}:")
    print(f"Number of logos detected: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"  {i+1}. {pred.get('label', 'logo')}: {pred.get('score', 0.0):.3f}")

if __name__ == "__main__":
    image_path = "/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/test_images/utah_football.png"  # Replace with your image path
    output_path = "/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/outputs/utah_football.png"  # Replace with desired output path

    
    main(image_path, output_path)