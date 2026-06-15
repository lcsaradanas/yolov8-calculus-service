from ultralytics import YOLO

model = YOLO("mbest.pt")
model.export(format="onnx")