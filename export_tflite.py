from ultralytics import YOLO

model_path = "models/fruit.pt"
model = YOLO(model_path)
model.export(format="tflite", imgsz=320, nms=True)