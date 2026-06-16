import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

click_threshold = 30
last_click_time = 0
click_anim_time = 0
CLICK_ANIM_DURATION = 0.3

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

def draw_hand(frame, landmarks, h, w):
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

frame_timestamp_ms = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp_ms += 33
    result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    current_time = time.time()

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:

            draw_hand(frame, landmarks, h, w)

            # Index finger tip
            index_tip = landmarks[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y)

            # Thumb tip
            thumb_tip = landmarks[4]
            tx = int(thumb_tip.x * w)
            ty = int(thumb_tip.y * h)

            distance = np.hypot(x - tx, y - ty)

            # Midpoint between index and thumb
            mid_x = (x + tx) // 2
            mid_y = (y + ty) // 2

            pinching = distance < click_threshold

            if pinching:
                # Draw pinch indicator (red filled circle at midpoint)
                cv2.circle(frame, (mid_x, mid_y), 15, (0, 0, 255), -1)
                cv2.circle(frame, (mid_x, mid_y), 15, (255, 255, 255), 2)

                if current_time - last_click_time > 0.5:
                    pyautogui.click()
                    last_click_time = current_time
                    click_anim_time = current_time
            else:
                # Draw open pinch ring (green circle at midpoint)
                cv2.circle(frame, (mid_x, mid_y), 15, (0, 255, 0), 2)

            # Index tip dot
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            # Thumb tip dot
            cv2.circle(frame, (tx, ty), 10, (255, 0, 0), -1)

            # Distance bar at bottom left
            bar_max = 100
            bar_len = int(np.interp(distance, [0, bar_max], [0, 200]))
            bar_color = (0, 0, 255) if pinching else (0, 255, 0)
            cv2.rectangle(frame, (20, h - 40), (220, h - 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, h - 40), (20 + bar_len, h - 20), bar_color, -1)
            cv2.putText(frame, f"Pinch: {int(distance)}px", (20, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Click animation burst
    if current_time - click_anim_time < CLICK_ANIM_DURATION:
        elapsed = current_time - click_anim_time
        radius = int(np.interp(elapsed, [0, CLICK_ANIM_DURATION], [15, 60]))
        alpha = int(np.interp(elapsed, [0, CLICK_ANIM_DURATION], [255, 0]))
        overlay = frame.copy()
        cv2.circle(overlay, (w // 2, 60), radius, (0, 0, 255), 3)
        cv2.addWeighted(overlay, alpha / 255, frame, 1 - alpha / 255, 0, frame)
        cv2.putText(frame, "CLICK!", (w // 2 - 40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Air Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
