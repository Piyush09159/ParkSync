import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import re
from paddleocr import PaddleOCR

# =========================================
# INITIALIZE PADDLEOCR (GPU & v4 Enabled)
# =========================================
print("Loading PaddleOCR...")
# use_gpu=True explicitly targets your graphics card
# ocr_version='PP-OCRv4' bypasses the v5 PaddleX/OneDNN bug
# show_log=False stops PaddleOCR from spamming your console every frame
ocr = PaddleOCR(lang='en', ocr_version='PP-OCRv4') 
print("PaddleOCR Ready!")

CONFIDENCE_THRESHOLD = 0.5

# =========================================
# INDIAN NUMBER PLATE REGEX
# =========================================
INDIAN_PLATE_REGEX = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'

def clean_and_validate(text):
    text = text.upper()
    text = text.replace(" ", "")
    
    # Common OCR corrections
    text = text.replace("O", "0")
    text = text.replace("I", "1")

    if re.match(INDIAN_PLATE_REGEX, text):
        return text
    return None

# =========================================
# RED ZONE DETECTION
# =========================================
def detect_red_zones(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 100, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 80, 50])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    zones = []
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 1500:
            continue

        poly = cv2.approxPolyDP(c, 5, True)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])

        zones.append({
            "zone_id": f"Z{i+1}",
            "poly": poly.reshape(-1,2),
            "centroid": (cx, cy)
        })

    return zones

# =========================================
# MAP TEXT CENTER TO ZONE
# =========================================
def map_to_zone(point, zones):
    for z in zones:
        inside = cv2.pointPolygonTest(z["poly"], point, False)
        if inside >= 0:
            return z["zone_id"]
    return None

# =========================================
# MULTI-FRAME STABILITY
# =========================================
last_detected_plate = None
stable_counter = 0
STABLE_THRESHOLD = 3
                               
# =========================================
# START WEBCAM
# =========================================
cap = cv2.VideoCapture(0)

print("Starting Webcam... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    zones = detect_red_zones(frame)

    # Draw zones
    for z in zones:
        cv2.polylines(frame, [z["poly"]], True, (0,255,0), 2)
        cv2.putText(frame, z["zone_id"], z["centroid"],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # PaddleOCR inference
    result = ocr.ocr(frame)

    if result and result[0] is not None:
        for line in result[0]:
            box, (text, prob) = line

            if prob < CONFIDENCE_THRESHOLD:
                continue

            text = ''.join(ch for ch in text if ch.isalnum())
            valid_plate = clean_and_validate(text)

            if not valid_plate:
                continue

            box = np.array(box).astype(int)
            x_center = int(np.mean(box[:, 0]))
            y_center = int(np.mean(box[:, 1]))

            zone = map_to_zone((x_center, y_center), zones)

            # Stability logic
            if valid_plate == last_detected_plate:
                stable_counter += 1
            else:
                stable_counter = 0
                last_detected_plate = valid_plate

            if stable_counter >= STABLE_THRESHOLD:
                print("FINAL DETECTED:", valid_plate, "Zone:", zone)

            # Draw bounding box
            cv2.polylines(frame, [box], True, (255,0,0), 2)
            cv2.putText(frame, f"{valid_plate} ({zone})",
                        (box[0][0], box[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,0,0), 2)

    cv2.imshow("ParkSync Live - PaddleOCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()