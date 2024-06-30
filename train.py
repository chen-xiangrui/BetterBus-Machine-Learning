from ultralytics import YOLO

# Load model
model = YOLO('betterbus-ml/yolov8s.pt')

# Train the model
model.train(data='betterbus-ml/data.yaml', epochs=30, imgsz=640, batch=32, workers=4)