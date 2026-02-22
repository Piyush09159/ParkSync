# =========================================
# ParkSync Live: Edge-Driven ALPR System
# =========================================
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
import onnxruntime as ort

import cv2
import numpy as np
import re

# =========================================
# INITIALIZE AI MODELS
# =========================================
print("Loading YOLOv8 (ONNX GPU Engine)...")
yolo_session = ort.InferenceSession("yolov8n.onnx", providers=['CUDAExecutionProvider'])
yolo_input_name = yolo_session.get_inputs()[0].name

print("Loading PaddleOCR (GPU ENABLED!)...")
ocr = PaddleOCR(lang='en', use_gpu=True, show_log=False) 
print("AI Models Ready!")

CONFIDENCE_THRESHOLD = 0.25
VEHICLE_CLASSES = [2, 3, 5, 7] # COCO IDs for vehicles
INDIAN_PLATE_REGEX = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'

def clean_and_validate(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("O", "0").replace("I", "1")
    if re.match(INDIAN_PLATE_REGEX, text):
        return text
    return None

# =========================================
# START WEBCAM
# =========================================
# Note: You used 1 in your script. Change to 0 if your main webcam doesn't turn on!
cap = cv2.VideoCapture(1) 
print("Starting ParkSync Live... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # =========================================================
    # THE MIRROR FIX: Flip the frame so text isn't backwards!
    # =========================================================
    frame = cv2.flip(frame, 1)
    
    h_orig, w_orig = frame.shape[:2]
    clean_frame = frame.copy() # Pristine copy for OCR

    # -----------------------------------------
    # STEP 1: RAW ONNX YOLOv8 INFERENCE
    # -----------------------------------------
    img = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = yolo_session.run(None, {yolo_input_name: img})

    # -----------------------------------------
    # STEP 2: DECODE & SORT BOUNDING BOXES
    # -----------------------------------------
    predictions = np.squeeze(outputs[0]).T
    x_factor = w_orig / 640.0
    y_factor = h_orig / 640.0

    boxes, confidences = [], []

    for row in predictions:
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence > CONFIDENCE_THRESHOLD and class_id in VEHICLE_CLASSES:
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            left = int((xc - w / 2) * x_factor)
            top = int((yc - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.45)

    final_cars = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_cars.append(boxes[i])
            
    # Sort cars from Left to Right geometrically
    final_cars.sort(key=lambda b: b[0]) 

    # -----------------------------------------
    # STEP 3: DYNAMIC SCENE CALIBRATION
    # -----------------------------------------
    virtual_zones = []
    zone_status = {}
    
    if len(final_cars) > 0:
        zone_boundaries = [0]
        for i in range(len(final_cars) - 1):
            x1_curr, y1_curr, w_curr, h_curr = final_cars[i]
            x1_next = final_cars[i+1][0]
            x2_curr = x1_curr + w_curr
            midpoint = (x2_curr + x1_next) // 2
            zone_boundaries.append(midpoint)
        zone_boundaries.append(w_orig)

        for i in range(len(final_cars)):
            x_left = zone_boundaries[i]
            x_right = zone_boundaries[i+1]
            zone_id = f"Slot-A{i+1}"
            
            y_offset = 30 if i % 2 == 0 else 100 
            
            virtual_zones.append({
                "zone_id": zone_id,
                "x_left": x_left,
                "x_right": x_right,
                "anchor_x": max(10, x_left + 10),  
                "anchor_y": y_offset
            })
            zone_status[zone_id] = "EMPTY"

    # -----------------------------------------
    # STEP 4: CROP, MAGNIFY & PADDLEOCR
    # -----------------------------------------
    for box in final_cars:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        
        # Draw quiet bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        vehicle_x_center = x1 + (w // 2)
        assigned_zone = "Unknown"
        for z in virtual_zones:
            if z["x_left"] <= vehicle_x_center <= z["x_right"]:
                assigned_zone = z["zone_id"]
                break
        
        y1_c, y2_c = max(0, y1), min(h_orig, y2)
        x1_c, x2_c = max(0, x1), min(w_orig, x2)
        
        vehicle_crop = clean_frame[y1_c:y2_c, x1_c:x2_c]
        if vehicle_crop.shape[0] < 20 or vehicle_crop.shape[1] < 20: continue
            
        # 2x Magnification for high-accuracy live reading
        zoomed_crop = cv2.resize(vehicle_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        ocr_result = ocr.ocr(zoomed_crop)
        
        if ocr_result and ocr_result[0] is not None:
            for line in ocr_result[0]:
                plate_box, (text, prob) = line

                if prob < 0.2: continue

                text = ''.join(ch for ch in text if ch.isalnum())
                valid_plate = clean_and_validate(text)

                if valid_plate and assigned_zone != "Unknown":
                    zone_status[assigned_zone] = valid_plate
                    print(f"LIVE DETECT: {valid_plate} -> {assigned_zone}")

    # =========================================
    # DRAW LIVE UI DASHBOARD
    # =========================================
    for z in virtual_zones:
        zone_id = z["zone_id"]
        plate = zone_status[zone_id]
        
        if z["x_left"] > 0: 
            cv2.line(frame, (z["x_left"], 0), (z["x_left"], h_orig // 2), (255, 0, 255), 2)
        
        if plate == "EMPTY":
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255) 
        else:
            bg_color = (0, 50, 0)
            text_color = (0, 255, 0)     
            
        (tw1, th1), _ = cv2.getTextSize(zone_id, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        (tw2, th2), _ = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        max_w = max(tw1, tw2) 
        ax, ay = z["anchor_x"], z["anchor_y"]
        
        cv2.rectangle(frame, (ax - 10, ay - th1 - 10), (ax + max_w + 10, ay + th1 + th2 + 15), bg_color, -1)
        cv2.putText(frame, zone_id, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(frame, plate, (ax, ay + th1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # Show the live feed!
    cv2.imshow("ParkSync Live Edge Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
