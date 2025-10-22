#@title COCO to YAML Conversion

import os
import glob
import sys
import json
import numpy as np
from pathlib import Path
import shutil

# Remove all cache files
cache_files = glob.glob("/content/YoloV12-Object-Detection-Project/datasets/koala/*.cache")
for cache_file in cache_files:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed: {cache_file}")

REPO_ROOT = "/content/YoloV12-Object-Detection-Project"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Function to convert COCO JSON to YOLO labels
def convert_coco_to_yolo(coco_json_path, output_dir):
    """Converts COCO format annotations to YOLO format"""
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Extract skeleton information
    skeleton = None
    for cat in data.get('categories', []):
        if 'skeleton' in cat and cat.get('skeleton'):
            skeleton = cat['skeleton']
            print(f"Found skeleton with {len(skeleton)} connections")
            break

    # Create output directory for labels
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # Clean any existing label files to avoid duplicates
    for old_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        os.remove(old_file)

    # Create mappings for conversion
    image_map = {img['id']: img for img in data['images']}

    # Process annotations
    processed_images = set()
    annotation_count = 0

    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in image_map:
            continue

        img_info = image_map[img_id]
        img_w, img_h = img_info['width'], img_info['height']
        file_name = img_info['file_name']
        base_name = os.path.splitext(os.path.basename(file_name))[0]

        # Get category index (use 0 for all)
        cat_idx = 0

        # Get bounding box in YOLO format
        x, y, w, h = ann['bbox']
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        width = w / img_w
        height = h / img_h

        # Create label file path
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        # Start with class and bbox
        line = f"{cat_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        # Add keypoints if available
        if 'keypoints' in ann:
            keypoints = ann['keypoints']
            for i in range(0, len(keypoints), 3):
                kpt_x = keypoints[i] / img_w
                kpt_y = keypoints[i+1] / img_h
                kpt_v = keypoints[i+2]
                line += f" {kpt_x:.6f} {kpt_y:.6f} {kpt_v}"

        # Write to file (append if file exists)
        with open(label_path, 'a') as f:
            f.write(line + '\n')

        processed_images.add(img_id)
        annotation_count += 1

    print(f"Processed {annotation_count} annotations across {len(processed_images)} images")
    return skeleton

# Function to prepare data.yaml
def create_data_yaml(yaml_path, flip_idx=None, skeleton=None):
    """Create a data.yaml file with the right configuration"""
    content = """
# Dataset paths
train: /content/YoloV12-Object-Detection-Project/datasets/koala/train
val: /content/YoloV12-Object-Detection-Project/datasets/koala/valid
test: /content/YoloV12-Object-Detection-Project/datasets/koala/test

# Dataset root directory
path: /content/YoloV12-Object-Detection-Project/datasets/koala

# Keypoint configuration
kpt_shape: [17, 3]
"""

    if flip_idx:
        content += f"flip_idx: {flip_idx}\n"
    else:
        content += "flip_idx: [1, 0, 2, 4, 3, 6, 5, 8, 7, 9, 10, 11, 12, 14, 13, 16, 15]\n"

    # Save skeleton to a file if provided
    if skeleton:
        skeleton_path = os.path.join(os.path.dirname(yaml_path), "skeleton.json")
        with open(skeleton_path, 'w') as f:
            json.dump(skeleton, f)
        print(f"Saved skeleton to {skeleton_path}")

    content += """
# Classes
nc: 1
names: ['Koala']
"""

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"Created {yaml_path}")

# Process train and validation sets
print("Converting train set")
train_skeleton = convert_coco_to_yolo(
    "/content/YoloV12-Object-Detection-Project/datasets/koala/train/_annotations.coco.json",
    "/content/YoloV12-Object-Detection-Project/datasets/koala/train"
)

print("\nConverting validation set")
val_skeleton = convert_coco_to_yolo(
    "/content/YoloV12-Object-Detection-Project/datasets/koala/valid/_annotations.coco.json",
    "/content/YoloV12-Object-Detection-Project/datasets/koala/valid"
)

# Create data.yaml
yaml_dir = "/content/YoloV12-Object-Detection-Project/yaml"
os.makedirs(yaml_dir, exist_ok=True)
create_data_yaml(
    os.path.join(yaml_dir, "data.yaml"),
    skeleton=train_skeleton
)

print("\n✅ Conversion complete! Ready to train with standard YOLOv12.")