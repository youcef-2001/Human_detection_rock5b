from ultralytics import YOLO

model = YOLO('./../models/v1_320px/weights/best.pt')

# Exporter le modèle au format ONNX avec la bonne taille d'image
# opset=12 est généralement recommandé pour rknn-toolkit
success = model.export(format='onnx', imgsz=320, opset=12)