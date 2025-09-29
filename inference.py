import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from torchvision.ops import box_convert

# Set up model and processor with use_fast=True
processor = AutoImageProcessor.from_pretrained("Pravallika6/detr-finetuned-logo-detection", use_fast=True)
model = AutoModelForObjectDetection.from_pretrained("Pravallika6/detr-finetuned-logo-detection")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set model to evaluation mode
model.eval()

# Colors for visualization
PRED_COLOR = 'red'  # Color for predicted boxes
GT_COLOR = 'lime'   # Color for ground truth boxes

def plot_results(pil_img, pred_scores, pred_boxes, gt_boxes, output_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    # Plot ground truth boxes in green
    if gt_boxes is not None:
        for box in gt_boxes:
            # Ground truth box
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        fill=False, color=GT_COLOR, linewidth=2))
            # Add "GT" label
            ax.text(box[0], box[1] - 10, 'GT', fontsize=12,
                   bbox=dict(facecolor=GT_COLOR, alpha=0.5), color='black')
    
    # Plot predicted boxes in red with confidence scores
    for score, box in zip(pred_scores, pred_boxes):
        box = box.tolist()
        # Predicted box
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    fill=False, color=PRED_COLOR, linewidth=2))
        # Add confidence score with "Pred" label
        text = f'Pred:{score:0.2f}'
        ax.text(box[0], box[1] - 5, text, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=PRED_COLOR),
                color='red')
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def process_image(image_path):
    # Load and process image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Convert outputs to numpy
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8  # Confidence threshold
    
    # Convert boxes to image coordinates
    boxes = outputs.pred_boxes[0, keep]
    probas = probas[keep]
    
    # Convert boxes to [x0, y0, x1, y1] format
    boxes = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
    
    # Scale boxes to image size
    w, h = image.size
    boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32).to(device)
    
    return {
        'image': image,
        'scores': probas.max(-1).values,
        'boxes': boxes
    }

def get_ground_truth_boxes(annotations, img_file):
    """
    Extract ground truth boxes for a specific image from COCO-style annotations
    """
    try:
        # First find the image_id for the given filename
        image_id = None
        for img in annotations['images']:
            if os.path.basename(img['file_name']) == img_file:
                image_id = img['id']
                break
        
        if image_id is None:
            return None
            
        # Get all annotations for this image_id
        boxes = []
        for ann in annotations['annotations']:
            if ann['image_id'] == image_id:
                # bbox format is [x, y, width, height], convert to [x1, y1, x2, y2]
                bbox = ann['bbox']
                boxes.append([
                    bbox[0],                    # x1
                    bbox[1],                    # y1
                    bbox[0] + bbox[2],          # x2 = x1 + width
                    bbox[1] + bbox[3]           # y2 = y1 + height
                ])
        
        return boxes if boxes else None
    except Exception as e:
        print(f"Error extracting ground truth boxes: {str(e)}")
        return None

def main():
    base_dir = '/scratch/general/vast/u1475870/logo_detection/data/test'
    images_dir = os.path.join(base_dir, 'images')
    json_path = os.path.join(base_dir, 'results.json')
    output_dir = '/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/predictions'
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory '{images_dir}' does not exist!")
        return
        
    if not os.path.exists(json_path):
        print(f"Error: JSON file '{json_path}' does not exist!")
        return

    # Load JSON file with ground truth annotations
    try:
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        print(f"Successfully loaded annotations file with {len(annotations['images'])} images and {len(annotations['annotations'])} annotations")
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return

    # Get list of images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in '{images_dir}'")
        print("Supported formats: .jpg, .jpeg, .png")
        return
        
    print(f"Found {len(image_files)} images to process")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(images_dir, img_file)
        try:
            # Get ground truth boxes
            gt_boxes = get_ground_truth_boxes(annotations, img_file)
            
            # Process image
            result = process_image(image_path)
            
            # Save visualization with both predicted and ground truth boxes
            output_path = os.path.join(output_dir, f'pred_{os.path.splitext(img_file)[0]}.png')
            plot_results(
                result['image'],
                result['scores'],
                result['boxes'],
                gt_boxes,
                output_path
            )
            
            print(f"\nProcessed {img_file}:")
            print(f"Number of predictions: {len(result['scores'])}")
            print(f"Confidence scores: {[f'{s:.2f}' for s in result['scores']]}")
            if gt_boxes:
                print(f"Number of ground truth boxes: {len(gt_boxes)}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

if __name__ == "__main__":
    main()