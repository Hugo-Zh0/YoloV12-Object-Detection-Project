#@title Check Directory Structure and Fix Labels

import os
import glob
import sys

REPO_ROOT = "/content/YoloV12-Object-Detection-Project"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Check directory structure
print("=== Directory Structure Analysis ===")

# Check train directory
train_dir = "/content/YoloV12-Object-Detection-Project/datasets/koala/train"
train_labels_dir = os.path.join(train_dir, "labels")
train_images = glob.glob(os.path.join(train_dir, "*.jpg")) + glob.glob(os.path.join(train_dir, "*.png"))
train_labels = glob.glob(os.path.join(train_labels_dir, "*.txt"))

print(f"Train images: {len(train_images)} (in {train_dir})")
print(f"Train labels: {len(train_labels)} (in {train_labels_dir})")

# Check validation directory
val_dir = "/content/YoloV12-Object-Detection-Project/datasets/koala/valid"
val_labels_dir = os.path.join(val_dir, "labels")
val_images = glob.glob(os.path.join(val_dir, "*.jpg")) + glob.glob(os.path.join(val_dir, "*.png"))
val_labels = glob.glob(os.path.join(val_labels_dir, "*.txt"))

print(f"Val images: {len(val_images)} (in {val_dir})")
print(f"Val labels: {len(val_labels)} (in {val_labels_dir})")

# Check example label files
print("\n=== Sample Label Contents ===")
if train_labels:
    with open(train_labels[0], 'r') as f:
        print(f"Sample train label ({os.path.basename(train_labels[0])}):")
        print(f.read()[:200] + "..." if len(f.read()) > 200 else f.read())
else:
    print("No train label files found")

# Fix 1: Make sure labels are in proper YOLO format
print("\n=== Fixing Directory Structure ===")

# YOLO expects images and labels to be in specific directories
train_images_dir = os.path.join(train_dir, "images")
val_images_dir = os.path.join(val_dir, "images")

# Create image directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)

# Move images to proper directories if needed
if train_images and not glob.glob(os.path.join(train_images_dir, "*.jpg")):
    print("Moving train images to proper directory...")
    for img in train_images:
        if not os.path.exists(os.path.join(train_images_dir, os.path.basename(img))):
            os.rename(img, os.path.join(train_images_dir, os.path.basename(img)))
    print(f"Moved {len(train_images)} train images")

if val_images and not glob.glob(os.path.join(val_images_dir, "*.jpg")):
    print("Moving validation images to proper directory...")
    for img in val_images:
        if not os.path.exists(os.path.join(val_images_dir, os.path.basename(img))):
            os.rename(img, os.path.join(val_images_dir, os.path.basename(img)))
    print(f"Moved {len(val_images)} validation images")

# Fix 2: Update data.yaml to use the proper directories
print("\n=== Updating data.yaml ===")

yaml_content = f"""
# Dataset paths with proper YOLO structure
train: {train_dir}
val: {val_dir}
test: {val_dir}  # Using val as test too

# Dataset root directory
path: /content/YoloV12-Object-Detection-Project/datasets/koala

# Keypoint configuration
kpt_shape: [17, 3]
flip_idx: [1, 0, 2, 4, 3, 6, 5, 8, 7, 9, 10, 11, 12, 14, 13, 16, 15]

# Classes
nc: 1
names: ['Koala']
"""

yaml_path = "/content/YoloV12-Object-Detection-Project/yaml/data.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
print(f"Updated {yaml_path}")

# Fix 3: Remove cache files again
cache_files = glob.glob("/content/YoloV12-Object-Detection-Project/datasets/koala/*.cache")
for cache_file in cache_files:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed cache file: {cache_file}")

print("\n=== Directory Structure Fixed ===")
print("YOLO training should now be able to find the label files.")
print("Run the training code again.")