from transformers import pipeline
from PIL import Image
import time
import torch
from PIL import ImageFilter
import numpy as np
import easyocr

def extract_text_from_region(image, box, reader):
    """Extract text from a specific region using EasyOCR"""
    # Convert PIL image to numpy array if it isn't already
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    # Extract region coordinates
    x1, y1 = max(0, int(box['xmin'])), max(0, int(box['ymin']))
    x2, y2 = min(img_array.shape[1], int(box['xmax'])), min(img_array.shape[0], int(box['ymax']))
    
    # Crop region
    region = img_array[y1:y2, x1:x2]
    
    # Perform OCR on the region
    results = reader.readtext(region)
    
    # Combine all detected text
    texts = [text for _, text, conf in results]
    return " ".join(texts)

def blur_credentials_fast(image, predictions, blur_radius=10):
    """Faster blur implementation using numpy array operations"""
    img_array = np.array(image)
    
    for pred in predictions:
        box = pred['box']
        x1, y1 = max(0, int(box['xmin'])), max(0, int(box['ymin']))
        x2, y2 = min(img_array.shape[1], int(box['xmax'])), min(img_array.shape[0], int(box['ymax']))
        region = Image.fromarray(img_array[y1:y2, x1:x2])
        blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img_array[y1:y2, x1:x2] = np.array(blurred_region)
    
    return Image.fromarray(img_array)

def main(image_path, output_path):
    # Initialize EasyOCR
    print("Initializing EasyOCR...")
    start_ocr_init = time.time()
    reader = easyocr.Reader(['en'], gpu=True)
    print(f"EasyOCR initialization time: {time.time() - start_ocr_init:.2f} seconds")
    
    # Print CUDA availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")
    
    # Initialize the pipeline
    start_load = time.time()
    pipe = pipeline("object-detection", 
                   model="Pravallika6/detr-finetuned-credentials",
                   device=device,
                   use_fast=True)
    load_time = time.time() - start_load
    print(f"Model loading time: {load_time:.2f} seconds")
    
    # Load image and run inference
    image = Image.open(image_path)
    
    start_inference = time.time()
    predictions = pipe(image)
    inference_time = time.time() - start_inference
    print(f"Inference time: {inference_time:.2f} seconds")
    
    # Extract text from each credential region
    start_ocr = time.time()
    for idx, pred in enumerate(predictions):
        text = extract_text_from_region(image, pred['box'], reader)
        print(f"\nCredential {idx + 1}:")
        print(f"Confidence: {pred['score']:.2f}")
        print(f"Location: {pred['box']}")
        print(f"Extracted Text: {text}")
    ocr_time = time.time() - start_ocr
    print(f"\nOCR processing time: {ocr_time:.2f} seconds")
    
    # Apply blur and save
    start_blur = time.time()
    blurred_image = blur_credentials_fast(image, predictions)
    blurred_image.save(output_path, 'JPEG', quality=90, optimize=True)
    blur_time = time.time() - start_blur
    print(f"Blur processing time: {blur_time:.2f} seconds")
    
    # Print total time
    total_time = load_time + inference_time + ocr_time + blur_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    # Print summary
    print(f"\nProcessed {image_path}:")
    print(f"Number of credentials detected and blurred: {len(predictions)}")

if __name__ == "__main__":
    image_path = "/uufs/chpc.utah.edu/common/home/u1475870/photonode/combined_dataset/val/images/78f2fa45-40N_1424.jpg"  # Replace with your image path
    output_path = "/uufs/chpc.utah.edu/common/home/u1475870/photonode/output.png"  # Replace with desired output path

    
    main(image_path, output_path)