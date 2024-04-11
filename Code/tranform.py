from ultralytics import YOLO
model = YOLO("../models/best_10_04_2023.pt")
success = model.export(format="onnx")
