from pathlib import Path
from ultralytics import YOLO # type: ignore
import torch # type: ignore

# scripts/train.py -> go up to repo root
ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "models" / "yolov12_kangaroo_trained_1.pt"
data_yaml  = ROOT / "data" / "data.yaml"
prediction_source = ROOT / "scripts" / "predictions" / "kangaroo.jpg"

model = YOLO(str(model_path))

model.predict(source=str(prediction_source), show=True, save=True),
              