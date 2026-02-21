# =========================================
# ParkSync: Static Image Benchmarking Tool
# =========================================
from paddleocr import PaddleOCR
import onnxruntime as ort

import cv2
import numpy as np
import re
import os

# =========================================
# INITIALIZE AI MODELS
# =========================================
print("Loading YOLOv8 (ONNX GPU Engine)...")
yolo_session = ort.InferenceSession("yolov8n.onnx", providers=['CUDAExecutionProvider'])
yolo_input_name = yolo_session.get_inputs()[0].name

print("Loading PaddleOCR (GPU ENABLED!)...")
ocr = PaddleOCR(lang='en', use_gpu=True, show_log=False) 
print("AI Models Ready!")

CONFIDENCE_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 3, 5, 7] 
INDIAN_PLATE_REGEX = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'

def clean_and_validate(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("O", "0").replace("I", "1")
    if re.match(INDIAN_PLATE_REGEX, text):
        return text
    return None

def define_virtual_zones(width, height):
    zones = []
    slot_width = width // 4
    for i in range(4):
        x1 = i * slot_width
        y1 = height // 2  
        x2 = (i + 1) * slot_width
        y2 = height     
        
        # ----------------------------------------------------
        # THE FIX: Stagger the Y-anchors (Up, Down, Up, Down)
        # ----------------------------------------------------
        y_offset = 20 if i % 2 == 0 else 90 
        
        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        zones.append({
            "zone_id": f"Slot-A{i+1}",
            "poly": poly,
            "anchor_x": x1 + 15,  
            "anchor_y": y1 + y_offset   
        })
    return zones

def map_to_zone(point, zones):
    for z in zones:
        if cv2.pointPolygonTest(z["poly"], point, False) >= 0:
            return z["zone_id"]
    return "Unknown Area"

# =========================================
# LOAD AND PROCESS STATIC IMAGE
# =========================================
image_path = "car1.jpg" 

if not os.path.exists(image_path):
    print(f"❌ Error: Could not find '{image_path}'.")
    exit()

print(f"Processing '{image_path}'...")
frame = cv2.imread(image_path)
h_orig, w_orig = frame.shape[:2]

clean_frame = frame.copy() 
virtual_zones = define_virtual_zones(w_orig, h_orig)
zone_status = {z["zone_id"]: "EMPTY" for z in virtual_zones}

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
# STEP 2: DECODE BOUNDING BOXES
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

# -----------------------------------------
# STEP 3: CROP, MAGNIFY (2x) & PADDLEOCR
# -----------------------------------------
if len(indices) > 0:
    for i in indices.flatten():
        box = boxes[i]
        x1, y1, w, h = box[0], box[1], box[2], box[3]
        x2, y2 = x1 + w, y1 + h
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        # STANDARD CROP (Removed the 10px padding that broke the 4th car)
        y1_c, y2_c = max(0, y1), min(h_orig, y2)
        x1_c, x2_c = max(0, x1), min(w_orig, x2)
        
        vehicle_crop = clean_frame[y1_c:y2_c, x1_c:x2_c]
        if vehicle_crop.shape[0] < 20 or vehicle_crop.shape[1] < 20: continue
            
        # PURE 2x MAGNIFICATION
        zoomed_crop = cv2.resize(vehicle_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        ocr_result = ocr.ocr(zoomed_crop)
        
        if ocr_result and ocr_result[0] is not None:
            for line in ocr_result[0]:
                plate_box, (text, prob) = line

                if prob < 0.2: continue

                text = ''.join(ch for ch in text if ch.isalnum())
                valid_plate = clean_and_validate(text)

                if valid_plate:
                    plate_x_center = x1_c + int(np.mean(np.array(plate_box)[:, 0]) / 2.0) 
                    plate_y_center = y1_c + int(np.mean(np.array(plate_box)[:, 1]) / 2.0)
                    assigned_zone = map_to_zone((plate_x_center, plate_y_center), virtual_zones)
                    
                    if assigned_zone != "Unknown Area":
                        zone_status[assigned_zone] = valid_plate

# =========================================
# THE STAGGERED & STACKED UI DASHBOARD
# =========================================
for z in virtual_zones:
    zone_id = z["zone_id"]
    plate = zone_status[zone_id]
    
    cv2.polylines(frame, [z["poly"]], True, (255, 0, 255), 2)
    
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
    
    # Line 1: Zone ID
    cv2.putText(frame, zone_id, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    # Line 2: License Plate
    cv2.putText(frame, plate, (ax, ay + th1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

# =========================================
# DISPLAY RESULT
# =========================================
display_height = 800
scale = display_height / h_orig
display_width = int(w_orig * scale)
display_frame = cv2.resize(frame, (display_width, display_height))

cv2.imshow("ParkSync - Perfected UI Test", display_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()