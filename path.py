import json
import os

# Path to the JSON file
json_path = "/uufs/chpc.utah.edu/common/home/u1475870/photonode/val-names-coco/result.json"

# Base directory for images
image_base_dir = "/uufs/chpc.utah.edu/common/home/u1475870/photonode/val-names-coco/images"

# Load the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Update the file paths
for image in data['images']:
    # Get the current filename (without path)
    filename = image['file_name']
    
    # Create the full absolute path
    full_path = os.path.join(image_base_dir, filename)
    
    # Update the file_name field
    image['file_name'] = full_path

# Save the updated JSON
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Updated {len(data['images'])} image paths in {json_path}")
