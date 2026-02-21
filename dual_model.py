# =========================================
# IMPORT PADDLE FIRST TO PREVENT DLL CRASHES
# =========================================
from paddleocr import PaddleOCR
import onnxruntime as ort

import os
import cv2
import numpy as np
import re

# =========================================
# INITIALIZE AI MODELS
# =========================================
print("Loading YOLOv8 (ONNX GPU Engine)...")
# Tell ONNX to specifically use your RTX 3050 (CUDA)
yolo_session = ort.InferenceSession("yolov8n.onnx", providers=['CUDAExecutionProvider'])
yolo_input_name = yolo_session.get_inputs()[0].name

print("Loading PaddleOCR (GPU ENABLED!)...")
ocr = PaddleOCR(lang='en', use_gpu=True, show_log=False) 
print("AI Models Ready!")

CONFIDENCE_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 3, 5, 7] # 2: car, 3: motorcycle, 5: bus, 7: truck
INDIAN_PLATE_REGEX = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'

def clean_and_validate(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("O", "0").replace("I", "1")
    if re.match(INDIAN_PLATE_REGEX, text):
        return text
    return None

def detect_red_zones(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 100, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([160, 80, 50]), np.array([179, 255, 255])
    
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    
    for i, c in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[0])):
        if cv2.contourArea(c) < 1500: continue
        poly = cv2.approxPolyDP(c, 5, True)
        M = cv2.moments(c)
        if M["m00"] != 0:
            zones.append({
                "zone_id": f"Z{i+1}",
                "poly": poly.reshape(-1,2),
                "centroid": (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            })
    return zones

def map_to_zone(point, zones):
    for z in zones:
        if cv2.pointPolygonTest(z["poly"], point, False) >= 0:
            return z["zone_id"]
    return None

# =========================================
# START WEBCAM
# =========================================
cap = cv2.VideoCapture(0)
print("Starting ParkSync Camera... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret: break

    zones = detect_red_zones(frame)

    for z in zones:
        cv2.polylines(frame, [z["poly"]], True, (0,255,0), 2)
        cv2.putText(frame, z["zone_id"], z["centroid"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # -----------------------------------------
    # STEP 1: RAW ONNX YOLOv8 INFERENCE
    # -----------------------------------------
    h_orig, w_orig = frame.shape[:2]
    
    # Format image for ONNX Engine (640x640, normalized, NCHW format)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # Fire the GPU inference
    outputs = yolo_session.run(None, {yolo_input_name: img})
    
    # -----------------------------------------
    # STEP 2: DECODE BOUNDING BOXES
    # -----------------------------------------
    predictions = np.squeeze(outputs[0]).T
    x_factor = w_orig / 640.0
    y_factor = h_orig / 640.0
    
    boxes, confidences, class_ids = [], [], []
    
    for row in predictions:
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence > CONFIDENCE_THRESHOLD and class_id in VEHICLE_CLASSES:
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            
            # Map back to standard webcam dimensions
            left = int((xc - w / 2) * x_factor)
            top = int((yc - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))

    # Apply Non-Maximum Suppression (Filter overlapping boxes)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.45)
    
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x1, y1, w, h = box[0], box[1], box[2], box[3]
            x2, y2 = x1 + w, y1 + h
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # -----------------------------------------
            # STEP 3: CROP & PADDLEOCR
            # -----------------------------------------
            y1_c, y2_c = max(0, y1), min(h_orig, y2)
            x1_c, x2_c = max(0, x1), min(w_orig, x2)
            vehicle_crop = frame[y1_c:y2_c, x1_c:x2_c]
            
            if vehicle_crop.shape[0] < 20 or vehicle_crop.shape[1] < 20: continue
                
            ocr_result = ocr.ocr(vehicle_crop)
            
            if ocr_result and ocr_result[0] is not None:
                for line in ocr_result[0]:
                    plate_box, (text, prob) = line

                    if prob < CONFIDENCE_THRESHOLD: continue

                    text = ''.join(ch for ch in text if ch.isalnum())
                    valid_plate = clean_and_validate(text)

                    if valid_plate:
                        plate_x_center = x1_c + int(np.mean(np.array(plate_box)[:, 0]))
                        plate_y_center = y1_c + int(np.mean(np.array(plate_box)[:, 1]))
                        zone = map_to_zone((plate_x_center, plate_y_center), zones)

                        cv2.putText(frame, f"{valid_plate} ({zone})", 
                                    (x1_c, y2_c + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        print(f"Detected: {valid_plate} in {zone}")

    cv2.imshow("ParkSync Dual-GPU Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()