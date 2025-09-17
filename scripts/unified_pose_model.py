from ultralytics import YOLO
from pathlib import Path

class UnifiedYOLO(YOLO):
    # A YOLO subclass that can train on a unified dataset:
    # Standard object detection (Bounding Boxes)
    # Pose estimation (keypoints)


    def __init__(self, model="yolo12n.pt", task="detect", **kwargs):
        # Call the original constructor
        super().__init__(model, task=task, **kwargs)
        # Store a custom flag for unified training
        self.unified = True

    def train_unified(self, data, **kwargs):
    
        # Can train on a dataset that contains both boxes and keypoints.
        #`data` should point to a data.yaml that includes kpt_shape.
     
        # Force YOLO to run in 'pose' mode if keypoints present but keep detection outputs too
        kwargs.setdefault("task", "pose")
        return super().train(data=data, **kwargs)

    def predict_unified(self, source, **kwargs):
        #Run inference that returns boxes and keypoints.
        kwargs.setdefault("task", "pose")
        return super().predict(source=source, **kwargs)
