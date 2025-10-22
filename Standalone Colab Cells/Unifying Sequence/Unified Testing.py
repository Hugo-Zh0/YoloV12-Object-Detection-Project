#@title Unified Sequential Testing (Object Detection + Pose)

import glob
import os
import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# Add the repository root to the Python path
REPO_ROOT = "/content/YoloV12-Object-Detection-Project"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.unified_pose_model import UnifiedYOLO

# ============================
# CONFIGURATION SETTINGS
# ============================
# Choose which models to test
RUN_OBJECT_DETECTION = True
RUN_POSE_ESTIMATION = True
RUN_SEQUENTIAL = True  # Run object detection followed by pose estimation

# Path settings
OBJECT_DETECTION_MODEL = "/content/YoloV12-Object-Detection-Project/runs/completed-training/koala-anatomy-features/train1/weights/best.pt"  # Change to best object detection model
POSE_ESTIMATION_MODEL = "/content/YoloV12-Object-Detection-Project/runs/completed-training/train10/weights/best.pt"  # Change to best pose model

# Data configuration
POSE_DATA_YAML = "/content/YoloV12-Object-Detection-Project/yaml/data.yaml"
OBJECT_DATA_YAML = "/content/YoloV12-Object-Detection-Project/yaml/koala-anatomy-features.yaml"

# Output paths
RESULTS_DIR = "/content/YoloV12-Object-Detection-Project/runs/test-results"
MODEL_TEST_DIR = "/content/YoloV12-Object-Detection-Project/runs/test-models/validation"
SEQUENTIAL_TEST_DIR = "/content/YoloV12-Object-Detection-Project/runs/test-models/sequential"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_TEST_DIR, exist_ok=True)
os.makedirs(SEQUENTIAL_TEST_DIR, exist_ok=True)

# Choose which test set to use for sequential evaluation
TEST_SET_SOURCE = "pose"  # Options: "pose", "object", or "both"(sequential)

# ============================
# HELPER FUNCTIONS
# ============================

def load_keypoint_names():
    """Load keypoint names from COCO JSON file"""
    keypoint_names = ["kpt"+str(i) for i in range(17)]  # Default names
    try:
        coco_path = "/content/YoloV12-Object-Detection-Project/datasets/koala/train/_annotations.coco.json"
        if os.path.exists(coco_path):
            with open(coco_path, 'r') as cf:
                coco_data = json.load(cf)
            for cat in coco_data.get('categories', []):
                if 'keypoints' in cat:
                    keypoint_names = cat['keypoints']
                    print(f"Found {len(keypoint_names)} keypoint names")
                    break
    except Exception as e:
        print(f"Could not load keypoint names: {e}")
    return keypoint_names

def sequential_prediction(object_model, pose_model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Run object detection first, then pass detections to pose estimation"""
    # Run object detection
    obj_results = object_model.predict(
        source=image_path, 
        conf=conf_threshold, 
        iou=iou_threshold,
        augment=False,
        verbose=False
    )[0]
    
    # Get the original image
    img = cv2.imread(image_path)
    orig_img = img.copy()
    h, w = img.shape[:2]
    
    # Process each detection
    detections = []
    for i, box in enumerate(obj_results.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        conf = float(obj_results.boxes.conf[i])
        cls = int(obj_results.boxes.cls[i])
        
        # Crop the detection with some margin
        margin = 20
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)
        crop_x2 = min(w, x2 + margin)
        crop_y2 = min(h, y2 + margin)
        
        # Crop the image
        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Skip if crop is empty
        if cropped_img.size == 0:
            continue
            
        # Save cropped image temporarily
        crop_path = f"/tmp/crop_{i}.jpg"
        cv2.imwrite(crop_path, cropped_img)
        
        # Run pose estimation on the crop
        pose_results = pose_model.predict(
            source=crop_path,
            task="pose",
            conf=conf_threshold,
            augment=False,
            verbose=False
        )[0]
        
        # Process keypoints if any
        if hasattr(pose_results, 'keypoints') and pose_results.keypoints is not None:
            keypoints = pose_results.keypoints.data[0].cpu().numpy()
            
            # Adjust keypoint coordinates to original image space
            if len(keypoints) > 0:
                # Only adjust x,y coordinates (not confidence)
                keypoints[:, 0] = keypoints[:, 0] + crop_x1
                keypoints[:, 1] = keypoints[:, 1] + crop_y1
                
                # Store the detection with keypoints
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'cls': cls,
                    'keypoints': keypoints
                })
        
        # Clean up
        if os.path.exists(crop_path):
            os.remove(crop_path)
    
    return detections, orig_img

def visualize_sequential_results(img, detections, keypoint_names, save_path=None):
    """Visualize results from sequential detection"""
    # Colors for visualization
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    # Draw each detection
    for i, det in enumerate(detections):
        # Draw bounding box
        x1, y1, x2, y2 = det['bbox']
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add class label and confidence
        label = f"Koala {det['conf']:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw keypoints
        if 'keypoints' in det:
            for j, kpt in enumerate(det['keypoints']):
                x, y, v = kpt
                if v > 0.5:  # Only draw visible keypoints
                    # Draw point
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                    
                    # Add keypoint name
                    if j < len(keypoint_names):
                        name = keypoint_names[j]
                        cv2.putText(img, name, (int(x), int(y)-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save the result
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
    
    return img

# ============================
# MAIN TESTING CODE
# ============================

# Load keypoint names
keypoint_names = load_keypoint_names()

# Run tests on individual models first
if RUN_OBJECT_DETECTION:
    print("\n=== Testing Object Detection Model ===")
    try:
        obj_model = YOLO(OBJECT_DETECTION_MODEL)
        obj_metrics = obj_model.val(
            data=OBJECT_DATA_YAML,
            split="test",
            imgsz=960,
            batch=8,
            augment=False,
            project=MODEL_TEST_DIR,
            name="object_detection",
            exist_ok=True
        )
        
        # Save metrics
        save_path = os.path.join(RESULTS_DIR, "object_detection_metrics.txt")
        with open(save_path, "w") as f:
            f.write(f"Object Detection Model: {OBJECT_DETECTION_MODEL}\n\n")
            f.write("=== Overall Metrics ===\n")
            for k, v in obj_metrics.results_dict.items():
                f.write(f"{k}: {v}\n")
        
        print(f"Object detection metrics saved to {save_path}")
    except Exception as e:
        print(f"Error testing object detection model: {e}")

if RUN_POSE_ESTIMATION:
    print("\n=== Testing Pose Estimation Model ===")
    try:
        pose_model = UnifiedYOLO(model=POSE_ESTIMATION_MODEL, task="pose")
        pose_metrics = pose_model.val(
            data=POSE_DATA_YAML,
            split="test",
            imgsz=960,
            batch=8,
            augment=False,
            project=MODEL_TEST_DIR,
            name="pose_estimation",
            exist_ok=True
        )
        
        # Save metrics
        save_path = os.path.join(RESULTS_DIR, "pose_estimation_metrics.txt")
        with open(save_path, "w") as f:
            f.write(f"Pose Estimation Model: {POSE_ESTIMATION_MODEL}\n\n")
            f.write("=== Overall Metrics ===\n")
            for k, v in pose_metrics.results_dict.items():
                f.write(f"{k}: {v}\n")
            
            # Keypoint-specific metrics
            if hasattr(pose_metrics, 'kpt_ap') and pose_metrics.kpt_ap is not None:
                f.write("\n=== Keypoint-specific AP ===\n")
                for i, ap in enumerate(pose_metrics.kpt_ap):
                    kpt_name = keypoint_names[i] if i < len(keypoint_names) else f"kpt{i}"
                    f.write(f"{kpt_name}: {ap:.4f}\n")
        
        print(f"Pose estimation metrics saved to {save_path}")
    except Exception as e:
        print(f"Error testing pose estimation model: {e}")

# Run sequential test
if RUN_SEQUENTIAL and RUN_OBJECT_DETECTION and RUN_POSE_ESTIMATION:
    print("\n=== Testing Sequential Pipeline ===")
    
    # Load both models
    obj_model = YOLO(OBJECT_DETECTION_MODEL)
    pose_model = UnifiedYOLO(model=POSE_ESTIMATION_MODEL, task="pose")
    
    # Get test images based on selected source
    if TEST_SET_SOURCE == "pose":
        test_dir = "/content/YoloV12-Object-Detection-Project/datasets/koala/test"
        test_images = glob.glob(os.path.join(test_dir, "images/*.jpg"))
        if not test_images:  # Try without "images/" subfolder as fallback
            test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
        print(f"Using pose estimation test set: {len(test_images)} images")
    
        if test_images:
            # Process a sample of test images
            sample_images = test_images[:5]  # Process first 5 images
            results = []
        
            for i, img_path in enumerate(sample_images):
                print(f"Processing test image {i+1}/{len(sample_images)}: {os.path.basename(img_path)}")
            
                # Run sequential prediction
                detections, orig_img = sequential_prediction(obj_model, pose_model, img_path)
            
                # Create visualization
                if detections:
                    viz_path = os.path.join(SEQUENTIAL_TEST_DIR, f"sequential_pose_{i}.jpg")
                    vis_img = visualize_sequential_results(orig_img.copy(), detections, keypoint_names, viz_path)
                
                    # Store results
                    results.append({
                        'image': os.path.basename(img_path),
                        'detections': len(detections),
                        'keypoints_found': sum(1 for d in detections if 'keypoints' in d),
                        'output': viz_path
                    })
            
            # Add summary code for pose case
            summary_path = os.path.join(RESULTS_DIR, "sequential_pipeline_pose_results.txt")
            with open(summary_path, "w") as f:
                f.write("Sequential Pipeline Test Results (Pose Test Set)\n")
                f.write(f"Object Detection Model: {OBJECT_DETECTION_MODEL}\n")
                f.write(f"Pose Estimation Model: {POSE_ESTIMATION_MODEL}\n\n")
                
                f.write("=== Results Summary ===\n")
                total_detections = sum(r['detections'] for r in results)
                total_with_keypoints = sum(r['keypoints_found'] for r in results)
                f.write(f"Images processed: {len(results)}\n")
                f.write(f"Total detections: {total_detections}\n")
                f.write(f"Detections with keypoints: {total_with_keypoints}\n")
                f.write(f"Success rate: {total_with_keypoints/max(1, total_detections):.2%}\n\n")
                
                f.write("=== Individual Results ===\n")
                for r in results:
                    f.write(f"Image: {r['image']}\n")
                    f.write(f"  Detections: {r['detections']}\n")
                    f.write(f"  With keypoints: {r['keypoints_found']}\n")
                    f.write(f"  Output: {r['output']}\n\n")
            
            print(f"Sequential pipeline results saved to {summary_path}")

    elif TEST_SET_SOURCE == "object":
        test_dir = "/content/YoloV12-Object-Detection-Project/datasets/koala-features/test"
        test_images = glob.glob(os.path.join(test_dir, "images/*.jpg"))
        if not test_images:  # Try without "images/" subfolder as fallback
            test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
        print(f"Using object detection test set: {len(test_images)} images")
    
        if test_images:
            # Process a sample of test images
            sample_images = test_images[:5]  # Process first 5 images
            results = []
            
            # Fixed indentation for the for loop
            for i, img_path in enumerate(sample_images):
                print(f"Processing test image {i+1}/{len(sample_images)}: {os.path.basename(img_path)}")
                
                # Run sequential prediction
                detections, orig_img = sequential_prediction(obj_model, pose_model, img_path)
                
                # Create visualization
                if detections:
                    viz_path = os.path.join(SEQUENTIAL_TEST_DIR, f"sequential_object_{i}.jpg")
                    vis_img = visualize_sequential_results(orig_img.copy(), detections, keypoint_names, viz_path)
                    
                    # Store results
                    results.append({
                        'image': os.path.basename(img_path),
                        'detections': len(detections),
                        'keypoints_found': sum(1 for d in detections if 'keypoints' in d),
                        'output': viz_path
                    })
            
            # Add summary code for object case
            summary_path = os.path.join(RESULTS_DIR, "sequential_pipeline_object_results.txt")
            with open(summary_path, "w") as f:
                f.write("Sequential Pipeline Test Results (Object Test Set)\n")
                f.write(f"Object Detection Model: {OBJECT_DETECTION_MODEL}\n")
                f.write(f"Pose Estimation Model: {POSE_ESTIMATION_MODEL}\n\n")
                
                f.write("=== Results Summary ===\n")
                total_detections = sum(r['detections'] for r in results)
                total_with_keypoints = sum(r['keypoints_found'] for r in results)
                f.write(f"Images processed: {len(results)}\n")
                f.write(f"Total detections: {total_detections}\n")
                f.write(f"Detections with keypoints: {total_with_keypoints}\n")
                f.write(f"Success rate: {total_with_keypoints/max(1, total_detections):.2%}\n\n")
                
                f.write("=== Individual Results ===\n")
                for r in results:
                    f.write(f"Image: {r['image']}\n")
                    f.write(f"  Detections: {r['detections']}\n")
                    f.write(f"  With keypoints: {r['keypoints_found']}\n")
                    f.write(f"  Output: {r['output']}\n\n")
            
            print(f"Sequential pipeline results saved to {summary_path}")

    else:  # "both"
        # Process both test sets separately
        pose_test_dir = "/content/YoloV12-Object-Detection-Project/datasets/koala/test"
        obj_test_dir = "/content/YoloV12-Object-Detection-Project/datasets/koala-features/test"
    
        pose_test_images = glob.glob(os.path.join(pose_test_dir, "images/*.jpg"))
        if not pose_test_images:  # Fallback
            pose_test_images = glob.glob(os.path.join(pose_test_dir, "*.jpg"))
        
        obj_test_images = glob.glob(os.path.join(obj_test_dir, "images/*.jpg"))
        if not obj_test_images:  # Fallback
            obj_test_images = glob.glob(os.path.join(obj_test_dir, "*.jpg"))
    
        # Create results container for both sets
        all_results = {}
    
        # Process each set separately
        for set_name, image_set in [("pose", pose_test_images[:5]), ("object", obj_test_images[:5])]:
            print(f"\n=== Processing {set_name} test set ({len(image_set)} images) ===")
            results = []
        
            for i, img_path in enumerate(image_set):
                print(f"Processing {set_name} image {i+1}/{len(image_set)}: {os.path.basename(img_path)}")
            
                # Run sequential prediction
                detections, orig_img = sequential_prediction(obj_model, pose_model, img_path)
            
                # Create visualization
                if detections:
                    viz_path = os.path.join(SEQUENTIAL_TEST_DIR, f"sequential_{set_name}_{i}.jpg")
                    vis_img = visualize_sequential_results(orig_img.copy(), detections, keypoint_names, viz_path)
                
                    # Store results
                    results.append({
                        'image': os.path.basename(img_path),
                        'detections': len(detections),
                        'keypoints_found': sum(1 for d in detections if 'keypoints' in d),
                        'output': viz_path
                    })
        
            # Store results for this set
            all_results[set_name] = results
    
        # Combine results for reporting
        combined_results = []
        for set_name, set_results in all_results.items():
            combined_results.extend(set_results)
    
        # Use combined results for the rest of the testing
        results = combined_results
        
        # Save summary
        summary_path = os.path.join(RESULTS_DIR, "sequential_pipeline_results.txt")
        with open(summary_path, "w") as f:
            f.write("Sequential Pipeline Test Results\n")
            f.write(f"Object Detection Model: {OBJECT_DETECTION_MODEL}\n")
            f.write(f"Pose Estimation Model: {POSE_ESTIMATION_MODEL}\n\n")
            
            f.write("=== Results Summary ===\n")
            total_detections = sum(r['detections'] for r in results)
            total_with_keypoints = sum(r['keypoints_found'] for r in results)
            f.write(f"Images processed: {len(results)}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Detections with keypoints: {total_with_keypoints}\n")
            f.write(f"Success rate: {total_with_keypoints/max(1, total_detections):.2%}\n\n")
            
            f.write("=== Individual Results ===\n")
            for r in results:
                f.write(f"Image: {r['image']}\n")
                f.write(f"  Detections: {r['detections']}\n")
                f.write(f"  With keypoints: {r['keypoints_found']}\n")
                f.write(f"  Output: {r['output']}\n\n")
        
        print(f"Sequential pipeline results saved to {summary_path}")
        
        # Display some results
        print("\n=== Sample Visualizations ===")
        fig, axes = plt.subplots(min(3, len(results)), 1, figsize=(10, 10))
        if len(results) == 1:
            axes = [axes]  # Make it iterable
            
        for i, r in enumerate(results[:3]):  # Show up to 3 results
            if os.path.exists(r['output']):
                img = cv2.imread(r['output'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].set_title(f"Image: {r['image']} - {r['keypoints_found']}/{r['detections']} keypoints")
                axes[i].axis('off')
        
                plt.tight_layout()
                plt.show()
            else:
                print("No test images found")

print("\n=== Testing Complete ===")