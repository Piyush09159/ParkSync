# =========================================
# ParkSync Live: Cloud-Connected Edge Demo
# =========================================
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import re
import requests
import threading
from datetime import datetime, timezone
from paddleocr import PaddleOCR

# =========================================
# API CONFIGURATION & REAL DATABASE MAP
# =========================================
API_ENDPOINT = "http://parksync.onrender.com/api/v1/vision/event" 

# THE FIX: Mapped exactly to your active database bookings!
SLOT_UID_MAP = {
    # Booked slot for MH46AK7394
    "A2": ["38830b0e-1e2c-4cd8-9305-24bced882afc"], 
    
    # Booked slot for MH14AK4736
    "A4": ["a2fae571-43f6-4940-b8a6-b3393e6ce0e2"], 
    
    # A spare slot (reusing an old ID just so it doesn't crash if you test here)
    "A6": ["f48c1726-cde3-4fdc-93e3-e62e912d8f8d"]  
}

def send_to_api(vehicle_no, slot_id):
    """Sends JSON payload and parses the backend's verified/corrected response."""
    
    # =========================================================
    # THE FIX: Generate an offset-naive timestamp (No UTC, no 'Z')
    # Example output: "2026-02-22T08:15:11.824"
    # =========================================================
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    
    uid_list = SLOT_UID_MAP.get(slot_id, [slot_id])
    
    for uid in uid_list:
        payload = {
            "vehicle_no": vehicle_no,
            "slot_id": uid,
            "timestamp": timestamp
        }
        
        try:
            response = requests.post(API_ENDPOINT, json=payload, timeout=15.0)
            
            if response.status_code in [200, 201]:
                try:
                    server_msg = response.json().get("message", "Success")
                except:
                    server_msg = "Success (No JSON message)"
                    
                print(f"✅ API SUCCESS: {vehicle_no} in {slot_id} -> Server says: '{server_msg}'")
                return 
            else:
                print(f"⚠️ API REJECTED ({response.status_code}) for UID {uid}: {response.text}")
                if len(uid_list) > 1 and uid != uid_list[-1]:
                    print("🔄 Attempting fallback UID...")
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ API CONNECTION FAILED: {e}")
            return 
            
    if len(uid_list) > 1:
        print(f"❌ Exhausted all fallback options for {slot_id}.")
# =========================================
# INITIALIZE PADDLEOCR
# =========================================
# ... (Keep the rest of your script exactly as it is!)
# =========================================
# INITIALIZE PADDLEOCR
# =========================================
print("Loading PaddleOCR (GPU ENABLED!)...")
ocr = PaddleOCR(lang='en', use_gpu=True, show_log=False) 
print("PaddleOCR Ready!")

CONFIDENCE_THRESHOLD = 0.5
INDIAN_PLATE_REGEX = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'

def clean_and_validate(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("O", "0").replace("I", "1")
    if re.match(INDIAN_PLATE_REGEX, text):
        return text
    return None

# =========================================
# MATHEMATICAL GRID SLICER
# =========================================
def detect_red_zones(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 100, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([160, 80, 50]), np.array([179, 255, 255])
    
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return []

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 5000: return []

    x, y, w, h = cv2.boundingRect(largest_contour)
    slots_count = 3
    slot_width = w // slots_count
    zones = []

    slot_names = ["A2", "A4", "A6"]

    for i in range(slots_count):
        sx, sy, sw, sh = x + (i * slot_width), y, slot_width, h
        poly = np.array([[sx, sy], [sx+sw, sy], [sx+sw, sy+sh], [sx, sy+sh]])
        zones.append({
            "zone_id": slot_names[i], 
            "poly": poly,
            "draw_box": poly,
            "anchor_x": sx + 10,
            "anchor_y": max(30, sy - 10)
        })
    return zones

def map_to_zone(point, zones):
    for z in zones:
        if cv2.pointPolygonTest(z["poly"], point, False) >= 0:
            return z["zone_id"]
    return "Unknown Area"

# =========================================
# START WEBCAM & MEMORY
# =========================================
cap = cv2.VideoCapture(1) 
print("Starting Hardware Demo Webcam... Press Q to exit")

zone_memory = {}

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    zones = detect_red_zones(clean_frame)
    
    # Initialize Memory with API Anti-Spam state
    for z in zones:
        if z["zone_id"] not in zone_memory:
            zone_memory[z["zone_id"]] = {"text": "EMPTY", "life": 0, "last_api_plate": None}

    # Decay the memory 
    for z_id in zone_memory:
        if zone_memory[z_id]["life"] > 0:
            zone_memory[z_id]["life"] -= 1
        if zone_memory[z_id]["life"] == 0:
            zone_memory[z_id]["text"] = "EMPTY"
            zone_memory[z_id]["last_api_plate"] = None 

    big_frame = cv2.resize(clean_frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    result = ocr.ocr(big_frame)

    if result and result[0] is not None:
        for line in result[0]:
            box, (text, prob) = line

            if prob < CONFIDENCE_THRESHOLD: continue

            text = ''.join(ch for ch in text if ch.isalnum())
            valid_plate = clean_and_validate(text)

            if not valid_plate: continue

            box = (np.array(box).astype(float) / 2.0).astype(int)
            x_center = int(np.mean(box[:, 0]))
            y_center = int(np.mean(box[:, 1]))

            zone = map_to_zone((x_center, y_center), zones)
            
            if zone != "Unknown Area":
                zone_memory[zone]["text"] = valid_plate
                zone_memory[zone]["life"] = 15 # Keeps UI stable
                
                # API TRIGGER LOGIC
                if zone_memory[zone]["last_api_plate"] != valid_plate:
                    zone_memory[zone]["last_api_plate"] = valid_plate
                    threading.Thread(target=send_to_api, args=(valid_plate, zone)).start()

            cv2.polylines(frame, [box], True, (0,165,255), 3)

    # =========================================
    # DRAW CLEAN DASHBOARD UI
    # =========================================
    for z in zones:
        zone_id = z["zone_id"]
        plate = zone_memory[zone_id]["text"]
        
        cv2.polylines(frame, [z["draw_box"]], True, (255, 0, 255), 3)
        bg_color, text_color = ((0, 0, 0), (255, 255, 255)) if plate == "EMPTY" else ((0, 50, 0), (0, 255, 0))
            
        (tw1, th1), _ = cv2.getTextSize(zone_id, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        (tw2, th2), _ = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        max_w = max(tw1, tw2) 
        ax, ay = z["anchor_x"], z["anchor_y"]
        
        cv2.rectangle(frame, (ax - 10, ay - th1 - 10), (ax + max_w + 10, ay + th1 + th2 + 15), bg_color, -1)
        cv2.putText(frame, zone_id, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(frame, plate, (ax, ay + th1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    cv2.imshow("ParkSync - Live Hardware Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
