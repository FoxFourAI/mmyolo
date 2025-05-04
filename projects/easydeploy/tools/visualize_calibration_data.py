import csv
import json
import os
import cv2
import zipfile
import numpy as np # For color generation

# --- Configuration --- (Match this with prepare_calibration_data.py)
output_path = "data/coco_calibration_data"
csv_file_name = "data.csv"
visualized_images_dir_name = "visualized_images"
zip_file_name = "visuals.zip"
# ---------------------

csv_file_path = os.path.join(output_path, csv_file_name)
visualized_images_path = os.path.join(output_path, visualized_images_dir_name)
zip_file_path = os.path.join(output_path, zip_file_name)

# Create output directory for visualized images
os.makedirs(visualized_images_path, exist_ok=True)
print(f"Ensured output directory exists: {visualized_images_path}")

# Function to generate distinct colors
def get_colors(num_colors):
    colors = []
    np.random.seed(0) # Consistent colors
    for i in range(num_colors):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        colors.append(color)
    return colors

# Load category information if available (optional, for labels)
# Assuming coco_data might be accessible or we just use IDs
# You might need to load annotations again if you want category names
coco_categories = {} # Placeholder - populate if needed for names
max_class_id = 0

print(f"Reading data from CSV: {csv_file_path}")
visualized_files_count = 0

if not os.path.exists(csv_file_path):
    print(f"Error: CSV file not found at {csv_file_path}")
    exit()

try:
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader) # Read all rows to find max class ID first

        # Find max class ID to generate enough colors
        for row in rows:
            try:
                class_ids = json.loads(row['class_ids'])
                if class_ids:
                    max_class_id = max(max_class_id, max(class_ids))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse class_ids for {row['image_path']}. Skipping row for max_id check.")
            except ValueError:
                 print(f"Warning: Empty class_ids list for {row['image_path']}. Skipping row for max_id check.")

        colors = get_colors(max_class_id + 1)
        print(f"Generated {len(colors)} distinct colors for visualization.")

        # Process each image
        for i, row in enumerate(rows):
            relative_image_path = row['image_path']
            image_full_path = os.path.join(output_path, relative_image_path)

            print(f"[{i+1}/{len(rows)}] Processing {relative_image_path}...")

            if not os.path.exists(image_full_path):
                print(f"  Warning: Image file not found at {image_full_path}. Skipping.")
                continue

            # Read image
            img = cv2.imread(image_full_path)
            if img is None:
                print(f"  Warning: Could not read image {image_full_path}. Skipping.")
                continue

            img_height, img_width, _ = img.shape

            try:
                # Load boxes and class IDs
                boxes_normalized = json.loads(row['boxes'])
                class_ids = json.loads(row['class_ids'])

                if len(boxes_normalized) != len(class_ids):
                    print(f"  Warning: Mismatch between number of boxes ({len(boxes_normalized)}) and class IDs ({len(class_ids)}). Skipping drawing for this image.")
                    continue

                # Draw boxes
                for box_norm, class_id in zip(boxes_normalized, class_ids):
                    # Denormalize: [x_norm, y_norm, w_norm, h_norm] -> [x_min, y_min, x_max, y_max]
                    x_n, y_n, w_n, h_n = box_norm
                    x_min = int(x_n * img_width)
                    y_min = int(y_n * img_height)
                    w = int(w_n * img_width)
                    h = int(h_n * img_height)
                    x_max = x_min + w
                    y_max = y_min + h

                    color = colors[class_id] if class_id < len(colors) else (255, 255, 255) # Default white
                    label = str(class_id) # Use class ID as label
                    # category_name = coco_categories.get(class_id, {}).get('name', str(class_id))

                    # Draw rectangle
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

                    # Put label
                    label_pos = (x_min, y_min - 10 if y_min > 20 else y_min + 15)
                    cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Save visualized image
                base_filename = os.path.basename(relative_image_path)
                output_image_path = os.path.join(visualized_images_path, base_filename)
                cv2.imwrite(output_image_path, img)
                visualized_files_count += 1
                # print(f"  Saved visualized image to {output_image_path}")

            except json.JSONDecodeError:
                print(f"  Warning: Could not parse boxes/class_ids JSON for {relative_image_path}. Skipping drawing.")
            except Exception as e:
                print(f"  Error processing image {relative_image_path}: {e}")

except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV: {e}")
    exit()

print(f"\nFinished processing images. Visualized {visualized_files_count} images.")

# Create zip archive of visualized images
if visualized_files_count > 0:
    print(f"\nCreating zip archive: {zip_file_path}")
    try:
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for root, dirs, files in os.walk(visualized_images_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Arcname is the path inside the zip file relative to visualized_images_path
                    arcname = os.path.relpath(file_path, visualized_images_path)
                    zipf.write(file_path, arcname=os.path.join(visualized_images_dir_name, arcname))
                    # print(f"  Adding {arcname} to zip.")
        print("Zip archive created successfully.")
    except Exception as e:
        print(f"Error creating zip archive: {e}")
else:
    print("\nNo images were visualized, skipping zip archive creation.")

print("\nVisualization script finished.")
