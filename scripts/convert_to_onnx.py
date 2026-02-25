from ultralytics import YOLO
from pathlib import Path
import shutil

# Racine du projet: .../Human_detection_rock5b
ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(ROOT / "runs/detect/models/mon_YOLOv82/weights/best.pt"))

# Export ONNX (opset 12 recommandé pour RKNN)
exported_path = model.export(format="onnx", imgsz=320, opset=12)

# Déplacer le fichier exporté vers /onnx à la racine du projet
exported_path = Path(exported_path)

target_path = ONNX_DIR / exported_path.name
shutil.move(str(exported_path), str(target_path))

print(f"ONNX sauvegardé ici: {target_path}")