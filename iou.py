import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import os
from torchvision.ops import box_convert
import json
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_detr(model, processor, val_dir):
    """
    Evaluate DETR model on COCO format dataset
    """
    annotation_file = os.path.join(val_dir, 'results.json')
    img_folder = os.path.join(val_dir, 'images')
    
    if not os.path.exists(img_folder):
        raise ValueError(f"Images folder not found at {img_folder}")
    
    # Load ground truth
    coco_gt = COCO(annotation_file)
    
    # Initialize results list for predictions
    results = []
    model.eval()
    
    # Get list of image ids
    img_ids = coco_gt.getImgIds()
    
    print(f"\nProcessing {len(img_ids)} images...")
    processed_count = 0
    
    with torch.no_grad():
        for img_id in tqdm(img_ids):
            # Load image info and image
            img_info = coco_gt.loadImgs(img_id)[0]
            file_name = img_info['file_name'].replace('images/', '')
            image_path = os.path.join(img_folder, file_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found. Skipping...")
                continue
                
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Get original image dimensions from COCO annotation
                orig_width = img_info['width']
                orig_height = img_info['height']
                
                # Process image
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get predictions
                outputs = model(**inputs)
                
                # Process predictions
                probas = outputs.logits.softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.5
                
                if not keep.any():
                    continue
                
                # Convert predicted boxes from CxCyWH to XYXY format
                pred_boxes = outputs.pred_boxes[0, keep].cpu()
                pred_boxes = box_convert(pred_boxes, 'cxcywh', 'xyxy')
                
                # Scale boxes to original image dimensions from COCO annotations
                scale_tensor = torch.tensor([orig_width, orig_height, 
                                          orig_width, orig_height])
                pred_boxes = pred_boxes * scale_tensor
                
                # Get scores and labels
                scores = probas[keep].max(-1).values.cpu()
                labels = probas[keep].argmax(-1).cpu()
                
                # Convert predictions to COCO format
                for box, score, label in zip(pred_boxes, scores, labels):
                    bbox = box.tolist()
                    # Convert to COCO box format [x,y,width,height]
                    bbox = [
                        bbox[0],  # x
                        bbox[1],  # y
                        bbox[2] - bbox[0],  # width
                        bbox[3] - bbox[1]   # height
                    ]
                    
                    result = {
                        'image_id': img_id,
                        'category_id': 0,  # Match ground truth category ID
                        'bbox': bbox,
                        'score': score.item()
                    }
                    results.append(result)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue
    
    print(f"\nSuccessfully processed {processed_count} out of {len(img_ids)} images")
    
    if not results:
        print("No predictions were generated!")
        return None
    
    # Debug: Print sample boxes for comparison
    print("\nSample Prediction vs Ground Truth:")
    sample_img_id = results[0]['image_id']
    sample_gt = next(ann for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=sample_img_id)))
    print(f"Ground Truth bbox: {sample_gt['bbox']}")
    print(f"Prediction bbox: {results[0]['bbox']}")
    
    # Save predictions
    pred_file = os.path.join(val_dir, 'detr_predictions.json')
    with open(pred_file, 'w') as f:
        json.dump(results, f)
    
    try:
        coco_dt = coco_gt.loadRes(pred_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        return None

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model and processor
processor = AutoImageProcessor.from_pretrained("Pravallika6/detr-finetuned-logo-detection", use_fast=True)
model = AutoModelForObjectDetection.from_pretrained("Pravallika6/detr-finetuned-logo-detection")
model = model.to(device)

# Set validation directory path
val_dir = '/scratch/general/vast/u1475870/logo_detection/data/test'

# Run evaluation
print("\nStarting evaluation...")
metrics = evaluate_detr(model, processor, val_dir)

if metrics is not None:
    metric_names = ['AP@0.5:0.95', 'AP@0.5', 'AP@0.75', 
                    'AP@small', 'AP@medium', 'AP@large',
                    'AR@1', 'AR@10', 'AR@100', 
                    'AR@small', 'AR@medium', 'AR@large']

    print("\nDetailed Metrics:")
    for name, value in zip(metric_names, metrics):
        print(f"{name}: {value:.3f}")
else:
    print("\nEvaluation failed to produce metrics. Please check the errors above.")