import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time

# ── pyautogui safety ──────────────────────────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

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

# ── thresholds ────────────────────────────────────────────────────────────────
click_threshold   = 30       # pixels – pinch distance to trigger click
CLICK_COOLDOWN    = 0.5      # seconds between clicks
CLICK_ANIM_DUR    = 0.3

# Head-scroll settings
SCROLL_DEAD_ZONE  = 0.015    # normalised Y offset before scroll starts (±)
SCROLL_SCALE      = 8        # multiplier: larger → faster scroll
SCROLL_INTERVAL   = 0.05     # seconds between scroll ticks (lower = smoother)
CALIBRATION_FRAMES = 30      # frames to average for neutral nose position

# ── state ─────────────────────────────────────────────────────────────────────
last_click_time  = 0
click_anim_time  = 0
last_scroll_time = 0

# Neutral nose-Y calibration
nose_samples     = []
nose_neutral_y   = None      # set after calibration

# ── build hand landmarker ─────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# ── build face landmarker ─────────────────────────────────────────────────────
# Requires face_landmarker.task  – download from:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
face_base = python.BaseOptions(model_asset_path='face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# ── helpers ───────────────────────────────────────────────────────────────────
def draw_hand(frame, landmarks, h, w):
    for s, e in HAND_CONNECTIONS:
        x1 = int(landmarks[s].x * w); y1 = int(landmarks[s].y * h)
        x2 = int(landmarks[e].x * w); y2 = int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)


def draw_scroll_indicator(frame, h, w, tilt, scrolling):
    """Draw a vertical bar on the left showing head tilt."""
    bar_x, bar_top, bar_bot = 15, 80, h - 80
    bar_h = bar_bot - bar_top
    mid   = bar_top + bar_h // 2

    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 14, bar_bot), (50, 50, 50), -1)

    # Clip fill between dead-zone and edge
    fill_len = int(abs(tilt) * bar_h * 3)
    fill_len = min(fill_len, bar_h // 2 - 2)
    color    = (0, 255, 255) if scrolling else (80, 80, 80)

    if tilt < -SCROLL_DEAD_ZONE:                          # tilt up → scroll up
        cv2.rectangle(frame, (bar_x, mid - fill_len), (bar_x + 14, mid), color, -1)
    elif tilt > SCROLL_DEAD_ZONE:                         # tilt down → scroll down
        cv2.rectangle(frame, (bar_x, mid), (bar_x + 14, mid + fill_len), color, -1)

    # Centre tick
    cv2.line(frame, (bar_x, mid), (bar_x + 14, mid), (180, 180, 180), 1)
    cv2.putText(frame, "SCROLL", (4, bar_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

# ── main loop ─────────────────────────────────────────────────────────────────
frame_ts = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_ts += 33
    hand_result = hand_detector.detect_for_video(mp_image, frame_ts)
    face_result = face_detector.detect_for_video(mp_image, frame_ts)

    current_time = time.time()

    # ── FACE / HEAD SCROLL ────────────────────────────────────────────────────
    tilt     = 0.0
    scrolling = False

    if face_result.face_landmarks:
        nose_tip = face_result.face_landmarks[0][1]   # landmark 1 = nose tip
        nose_y   = nose_tip.y                          # 0‥1 (normalised)

        # Calibrate neutral position from first N frames
        if nose_neutral_y is None:
            nose_samples.append(nose_y)
            if len(nose_samples) >= CALIBRATION_FRAMES:
                nose_neutral_y = float(np.mean(nose_samples))
                print(f"[HEAD SCROLL] Neutral nose Y calibrated: {nose_neutral_y:.4f}")

            # Show calibration progress
            pct = int(len(nose_samples) / CALIBRATION_FRAMES * 100)
            cv2.putText(frame, f"Calibrating head neutral... {pct}%",
                        (w // 2 - 160, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            tilt = nose_y - nose_neutral_y             # positive = head tilted down

            if abs(tilt) > SCROLL_DEAD_ZONE:
                scrolling = True
                if current_time - last_scroll_time > SCROLL_INTERVAL:
                    scroll_amount = int(tilt * SCROLL_SCALE * -10)  # negative = scroll down
                    # clamp to avoid huge jumps
                    scroll_amount = max(-5, min(5, scroll_amount))
                    pyautogui.scroll(scroll_amount)
                    last_scroll_time = current_time

        # Draw small nose dot
        nx = int(nose_tip.x * w)
        ny = int(nose_tip.y * h)
        cv2.circle(frame, (nx, ny), 5, (0, 255, 255), -1)

        draw_scroll_indicator(frame, h, w, tilt, scrolling)

        # Label
        direction = ""
        if tilt < -SCROLL_DEAD_ZONE:
            direction = "▲ UP"
        elif tilt > SCROLL_DEAD_ZONE:
            direction = "▼ DOWN"
        cv2.putText(frame, direction, (35, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255) if scrolling else (100, 100, 100), 2)

    # ── HAND / MOUSE + CLICK ──────────────────────────────────────────────────
    if hand_result.hand_landmarks:
        for landmarks in hand_result.hand_landmarks:
            draw_hand(frame, landmarks, h, w)

            # Index finger tip → cursor
            index_tip = landmarks[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y)

            # Thumb tip → pinch detection
            thumb_tip = landmarks[4]
            tx = int(thumb_tip.x * w)
            ty = int(thumb_tip.y * h)
            distance = np.hypot(x - tx, y - ty)
            mid_x = (x + tx) // 2
            mid_y = (y + ty) // 2
            pinching = distance < click_threshold

            if pinching:
                cv2.circle(frame, (mid_x, mid_y), 15, (0, 0, 255), -1)
                cv2.circle(frame, (mid_x, mid_y), 15, (255, 255, 255), 2)
                if current_time - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time  = current_time
                    click_anim_time  = current_time
            else:
                cv2.circle(frame, (mid_x, mid_y), 15, (0, 255, 0), 2)

            cv2.circle(frame, (x, y),   10, (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), 10, (255, 0, 0), -1)

            # Distance bar (bottom left)
            bar_len   = int(np.interp(distance, [0, 100], [0, 200]))
            bar_color = (0, 0, 255) if pinching else (0, 255, 0)
            cv2.rectangle(frame, (20, h - 40), (220, h - 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, h - 40), (20 + bar_len, h - 20), bar_color, -1)
            cv2.putText(frame, f"Pinch: {int(distance)}px", (20, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── CLICK ANIMATION ───────────────────────────────────────────────────────
    if current_time - click_anim_time < CLICK_ANIM_DUR:
        elapsed = current_time - click_anim_time
        radius  = int(np.interp(elapsed, [0, CLICK_ANIM_DUR], [15, 60]))
        alpha   = int(np.interp(elapsed, [0, CLICK_ANIM_DUR], [255, 0]))
        overlay = frame.copy()
        cv2.circle(overlay, (w // 2, 60), radius, (0, 0, 255), 3)
        cv2.addWeighted(overlay, alpha / 255, frame, 1 - alpha / 255, 0, frame)
        cv2.putText(frame, "CLICK!", (w // 2 - 40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ── HUD legend ────────────────────────────────────────────────────────────
    cv2.putText(frame, "Tilt head to scroll | Pinch to click | ESC to quit",
                (w // 2 - 230, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.imshow("Air Mouse + Head Scroll", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hand_detector.close()
face_detector.close()

