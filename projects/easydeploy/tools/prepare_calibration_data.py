import json
import os
import random
import shutil
import csv
from PIL import Image # Need PIL to get image dimensions for normalization
import zipfile # Added for zipping

annotations_path = "data/coco/annotations/instances_val2017.json"
images_path = "data/coco/val2017"
output_path = "data/coco_calibration_data"
num_images = 15 # Updated to 20 as requested
random_seed = 42 # Fixed seed for reproducibility

if os.path.exists(output_path):
    print(f"Output directory {output_path} already exists. Please remove it before running the script.")
    print("rm -rf " + output_path)
    exit(1)

# Set random seed
random.seed(random_seed)
print(f"Set random seed to {random_seed}")

# Create output directories if they don't exist
output_images_dir = os.path.join(output_path, "images")
os.makedirs(output_images_dir, exist_ok=True)
print(f"Ensured output directory exists: {output_images_dir}")

# Load COCO annotations
print(f"Loading annotations from {annotations_path}...")
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)
print("Annotations loaded.")

images_info = {img['id']: img for img in coco_data['images']}
annotations_info = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_info:
        annotations_info[img_id] = []
    annotations_info[img_id].append(ann)

# Get all image IDs
all_image_ids = list(images_info.keys())
print(f"Found {len(all_image_ids)} images in the dataset.")

# Randomly sample image IDs
if len(all_image_ids) < num_images:
    print(f"Warning: Requested {num_images} images, but only {len(all_image_ids)} available. Sampling all available images.")
    sampled_image_ids = all_image_ids
else:
    sampled_image_ids = random.sample(all_image_ids, num_images)
print(f"Randomly sampled {len(sampled_image_ids)} image IDs.")

csv_data = []
csv_header = ['image_path', 'boxes', 'class_ids']

# Process sampled images
for i, image_id in enumerate(sampled_image_ids):
    img_info = images_info[image_id]
    file_name = img_info['file_name']
    img_width = img_info['width']
    img_height = img_info['height']

    source_image_path = os.path.join(images_path, file_name)
    relative_dest_image_path = os.path.join("images", file_name)
    dest_image_path = os.path.join(output_path, relative_dest_image_path)

    # Copy image
    print(f"[{i+1}/{len(sampled_image_ids)}] Processing image ID {image_id}: {file_name}")
    if os.path.exists(source_image_path):
        shutil.copy(source_image_path, dest_image_path)
        # print(f"  Copied {source_image_path} to {dest_image_path}")
    else:
        print(f"  Warning: Source image not found at {source_image_path}. Skipping copy.")
        continue # Skip if source image doesn't exist

    # Get annotations for this image
    img_annotations = annotations_info.get(image_id, [])
    boxes_normalized = []
    class_ids = []

    if not img_annotations:
         print(f"  Warning: No annotations found for image ID {image_id}.")

    for ann in img_annotations:
        # COCO format: [x_min, y_min, width, height]
        bbox = ann['bbox']
        x_min, y_min, w, h = bbox

        # Normalize coordinates
        norm_x = x_min / img_width
        norm_y = y_min / img_height
        norm_w = w / img_width
        norm_h = h / img_height

        boxes_normalized.append([norm_x, norm_y, norm_w, norm_h])
        class_ids.append(ann['category_id']) # Using category_id directly

    csv_data.append({
        'image_path': relative_dest_image_path,
        'boxes': json.dumps(boxes_normalized), # Store as JSON string in CSV
        'class_ids': json.dumps(class_ids)     # Store as JSON string in CSV
    })
    # print(f"  Found {len(boxes_normalized)} annotations.")

# Write data to CSV
csv_file_path = os.path.join(output_path, "data.csv")
print(f"Writing data to CSV file: {csv_file_path}")
try:
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_data)
    print("CSV file written successfully.")
except IOError:
    print(f"Error writing CSV file at {csv_file_path}")

print("Script finished.")

# Create zip archive of images and csv
zip_file_path = os.path.join(output_path, "coco_calibration_data.zip")
print(f"\nCreating zip archive: {zip_file_path}")
try:
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # Add images folder recursively
        for root, dirs, files in os.walk(output_images_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Arcname is the path inside the zip file
                arcname = os.path.relpath(file_path, output_path)
                zipf.write(file_path, arcname=arcname)
                # print(f"  Adding {arcname} to zip.")

        # Add csv file
        csv_arcname = os.path.relpath(csv_file_path, output_path)
        zipf.write(csv_file_path, arcname=csv_arcname)
        # print(f"  Adding {csv_arcname} to zip.")

    print("Zip archive created successfully.")
except Exception as e:
    print(f"Error creating zip archive: {e}")

