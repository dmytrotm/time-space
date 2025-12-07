# 1. Експорт PyTorch моделі в ONNX
from ultralytics import YOLO

model = YOLO('models/tape_detector.pt')
model.export(format='onnx', simplify=True)