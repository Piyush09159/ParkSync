from ultralytics import YOLO

print("Downloading and exporting YOLOv8 to ONNX...")
# This automatically grabs the nano model and converts it
model = YOLO('yolov8n.pt')
model.export(format='onnx', simplify=True)
print("Export complete! You should now see 'yolov8n.onnx' in your folder.")