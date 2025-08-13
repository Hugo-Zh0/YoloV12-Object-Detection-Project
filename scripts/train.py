from pathlib import Path
from ultralytics import YOLO # type: ignore
import torch # type: ignore

# scripts/train.py -> go up to repo root
ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "models" / "yolov12s.pt"
data_yaml  = ROOT / "data" / "data.yaml"

# Optional: nudge CPU vs CUDA automatically
device = 0 if torch.cuda.is_available() else "cpu"

print(f"Using model: {model_path}")
print(f"Using data : {data_yaml}")
print(f"Device     : {device}")

model = YOLO(str(model_path))
model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=640,
    batch=12,         # consider lowering if CPU only
    workers=1,        # Windows-friendly
    device=device
)
