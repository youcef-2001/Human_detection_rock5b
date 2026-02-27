from ultralytics import YOLO
from pathlib import Path
import shutil


def convert_to_onnx(   name='mon_YOLOv86',):
    # Racine du projet: .../Human_detection_rock5b
    ROOT = Path(__file__).resolve().parents[1]
    ONNX_DIR = ROOT / "onnx"
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ROOT / f"runs/detect/models/{name}/weights/best.pt"))

    # Export ONNX (opset 12 recommandé pour RKNN)
    exported_path = model.export(format="onnx", imgsz=320, opset=12)

    # Déplacer le fichier exporté vers /onnx à la racine du projet
    exported_path = Path(exported_path)

    target_path = ONNX_DIR / f"{name}.onnx"
    shutil.move(str(exported_path), str(target_path))

    print(f"ONNX sauvegardé ici: {target_path}")

if __name__ == "__main__":
    convert_to_onnx()