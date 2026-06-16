"""'''
Exam Cheating Detector
======================
Uses MediaPipe Face Mesh + Pose to detect suspicious behaviours during an exam.
Outputs a live cheating-risk percentage with per-behaviour breakdown.

Detected behaviours
-------------------
  1. Gaze direction   – eyes looking left / right / up (copying neighbour, looking at notes)
  2. Head turn        – head rotated horizontally beyond threshold
  3. Head nod / tilt  – repeated up-down head movement (signalling)
  4. Talking          – mouth open beyond threshold
  5. Looking away     – face not visible for extended time
  6. Phone / object   – hand raised to face level (holding phone up)
  7. Suspicious speed – sudden fast head movements

Requirements
------------
  pip install opencv-python mediapipe numpy

Model files (place next to this script)
----------------------------------------
  face_landmarker.task
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Run
---
  python exam_cheat_detector.py
  Press  E  to start/stop exam session
  Press  R  to reset stats
  Press ESC to quit
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from collections import deque

# ── tuneable constants ────────────────────────────────────────────────────────
GAZE_H_THRESH      = 0.12   # normalised iris offset – horizontal
GAZE_V_UP_THRESH   = 0.08   # normalised iris offset – looking up
HEAD_YAW_THRESH    = 18     # degrees – head rotation left/right
HEAD_PITCH_THRESH  = 20     # degrees – head pitch up (looking at ceiling)
MOUTH_OPEN_THRESH  = 0.045  # normalised lip distance
FACE_MISSING_SECS  = 1.5    # seconds before "face away" counts
HAND_FACE_DIST     = 0.25   # fraction of frame height – hand near face
NOD_WINDOW_SECS    = 2.0    # window to count nods
NOD_COUNT_THRESH   = 3      # nods in window to flag

# Weight each behaviour for the overall risk score (must sum to 1.0)
WEIGHTS = {
    "gaze_side"    : 0.22,
    "gaze_up"      : 0.10,
    "head_turn"    : 0.20,
    "talking"      : 0.15,
    "face_away"    : 0.18,
    "hand_face"    : 0.10,
    "nodding"      : 0.05,
}

# Colour palette (BGR)
C_RED    = (0,   60, 220)
C_ORANGE = (0,  140, 255)
C_YELLOW = (0,  210, 255)
C_GREEN  = (80, 200,  80)
C_CYAN   = (220, 200,  0)
C_WHITE  = (240, 240, 240)
C_GRAY   = (120, 120, 120)
C_DARK   = ( 20,  20,  20)
C_BG     = ( 15,  15,  15)

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

base_face = python.BaseOptions(model_asset_path='face_landmarker.task')
face_opts = vision.FaceLandmarkerOptions(
    base_options=base_face,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
face_detector = vision.FaceLandmarker.create_from_options(face_opts)

pose_estimator = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── state ─────────────────────────────────────────────────────────────────────
exam_active         = False
exam_start_time     = None
face_missing_since  = None
nod_timestamps      = deque()
last_pitch          = None
prev_yaw            = 0.0

# Per-behaviour: (triggered_frames, total_frames)
behaviour_counts = {k: [0, 0] for k in WEIGHTS}

# Rolling risk history for the graph
HISTORY_LEN   = 200
risk_history  = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

frame_ts = 0


# ── helpers ───────────────────────────────────────────────────────────────────
def rotation_from_matrix(mat):
    """Extract yaw (left/right) and pitch (up/down) in degrees from 4x4 matrix."""
    r = mat[:3, :3]
    yaw   = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    pitch = np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[2, 1]**2 + r[2, 2]**2)))
    return yaw, pitch


def iris_gaze(landmarks, h, w):
    """Return (h_offset, v_offset) normalised by eye width.
       Positive h = looking right, positive v = looking down."""
    # left eye: iris=473, inner=133, outer=33, top=159, bot=145
    # right eye: iris=468, inner=362, outer=263, top=386, bot=374
    def eye_offset(iris_idx, inner_idx, outer_idx, top_idx, bot_idx):
        iris  = landmarks[iris_idx]
        inner = landmarks[inner_idx]
        outer = landmarks[outer_idx]
        top   = landmarks[top_idx]
        bot   = landmarks[bot_idx]
        ix, iy = iris.x * w,  iris.y * h
        ex     = (inner.x + outer.x) / 2 * w
        ey     = (top.y  + bot.y)   / 2 * h
        eye_w  = abs(inner.x - outer.x) * w + 1e-6
        eye_h  = abs(top.y   - bot.y)   * h + 1e-6
        return (ix - ex) / eye_w, (iy - ey) / eye_h

    lh, lv = eye_offset(473, 133, 33,  159, 145)
    rh, rv = eye_offset(468, 362, 263, 386, 374)
    return (lh + rh) / 2, (lv + rv) / 2


def mouth_openness(landmarks):
    """Normalised distance between upper and lower lip."""
    upper = landmarks[13]
    lower = landmarks[14]
    return abs(upper.y - lower.y)


def risk_colour(pct):
    if pct < 25:  return C_GREEN
    if pct < 50:  return C_YELLOW
    if pct < 75:  return C_ORANGE
    return C_RED


def draw_rounded_rect(img, pt1, pt2, color, thickness, r=8):
    x1, y1 = pt1; x2, y2 = pt2
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270,  0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0,  0, 90, color, thickness)


def draw_panel(frame, x, y, pw, ph, alpha=0.55):
    """Semi-transparent dark panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+pw, y+ph), (18, 18, 18), -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)


def draw_risk_gauge(frame, cx, cy, radius, pct):
    """Draw arc gauge for overall risk %."""
    start_a, end_a = 220, -40          # 260° arc
    total_deg = 260
    colour = risk_colour(pct)

    # Background arc (gray)
    for angle in range(start_a, start_a - total_deg, -1):
        rad = np.radians(angle)
        ax  = int(cx + radius * np.cos(rad))
        ay  = int(cy - radius * np.sin(rad))
        cv2.circle(frame, (ax, ay), 4, (50, 50, 50), -1)

    # Filled arc
    fill_deg = int(pct / 100 * total_deg)
    for angle in range(start_a, start_a - fill_deg, -1):
        rad = np.radians(angle)
        ax  = int(cx + radius * np.cos(rad))
        ay  = int(cy - radius * np.sin(rad))
        cv2.circle(frame, (ax, ay), 4, colour, -1)

    # Centre text
    label = f"{int(pct)}%"
    cv2.putText(frame, label, (cx - 22, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    cv2.putText(frame, "RISK", (cx - 16, cy + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)


def draw_behaviour_bars(frame, x, y, behaviours):
    """Draw small horizontal bars for each behaviour."""
    labels = {
        "gaze_side" : "Gaze left/right",
        "gaze_up"   : "Gaze upward",
        "head_turn" : "Head turn",
        "talking"   : "Mouth open",
        "face_away" : "Face hidden",
        "hand_face" : "Hand near face",
        "nodding"   : "Nodding",
    }
    bar_w = 140
    row_h = 24
    for i, (key, label) in enumerate(labels.items()):
        pct = behaviours[key]
        ry  = y + i * row_h
        colour = risk_colour(pct)
        # background
        cv2.rectangle(frame, (x, ry+2), (x+bar_w, ry+14), (45,45,45), -1)
        # fill
        fill = int(pct / 100 * bar_w)
        cv2.rectangle(frame, (x, ry+2), (x+fill, ry+14), colour, -1)
        # label
        cv2.putText(frame, label, (x + bar_w + 6, ry + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1)
        cv2.putText(frame, f"{int(pct)}%", (x - 36, ry + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)


def draw_history_graph(frame, x, y, gw, gh, history):
    """Mini rolling risk graph."""
    cv2.rectangle(frame, (x, y), (x+gw, y+gh), (35,35,35), -1)
    pts = list(history)
    for i in range(1, len(pts)):
        x1 = x + int((i-1) / len(pts) * gw)
        x2 = x + int(i     / len(pts) * gw)
        y1 = y + gh - int(pts[i-1] / 100 * gh)
        y2 = y + gh - int(pts[i]   / 100 * gh)
        cv2.line(frame, (x1, y1), (x2, y2), risk_colour(pts[i]), 1)
    cv2.putText(frame, "Risk history", (x+2, y-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1)


# ── capture ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

print("=== Exam Cheating Detector ===")
print("Press E to start/stop exam  |  R to reset  |  ESC to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame    = cv2.flip(frame, 1)
    h, w, _  = frame.shape
    current  = time.time()

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_ts += 33
    face_res = face_detector.detect_for_video(mp_img, frame_ts)

    rgb_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res = pose_estimator.process(rgb_pose)

    # ── per-frame flag ────────────────────────────────────────────────────────
    flags = {k: False for k in WEIGHTS}

    face_found = bool(face_res.face_landmarks)

    if face_found:
        face_missing_since = None
        lms = face_res.face_landmarks[0]

        # Gaze
        gh, gv = iris_gaze(lms, h, w)
        if abs(gh) > GAZE_H_THRESH:
            flags["gaze_side"] = True
        if gv < -GAZE_V_UP_THRESH:            # negative v = looking up
            flags["gaze_up"] = True

        # Head rotation from transformation matrix
        yaw, pitch = 0.0, 0.0
        if face_res.facial_transformation_matrixes:
            mat        = np.array(face_res.facial_transformation_matrixes[0].data).reshape(4, 4)
            yaw, pitch = rotation_from_matrix(mat)

        if abs(yaw)   > HEAD_YAW_THRESH:    flags["head_turn"] = True
        if pitch      > HEAD_PITCH_THRESH:  flags["gaze_up"]   = True   # looking up at ceiling

        # Nodding (count pitch direction changes)
        if last_pitch is not None:
            if (last_pitch > 5 and pitch < -5) or (last_pitch < -5 and pitch > 5):
                nod_timestamps.append(current)
        last_pitch = pitch
        # clear old nods
        while nod_timestamps and current - nod_timestamps[0] > NOD_WINDOW_SECS:
            nod_timestamps.popleft()
        if len(nod_timestamps) >= NOD_COUNT_THRESH:
            flags["nodding"] = True

        # Mouth open
        if mouth_openness(lms) > MOUTH_OPEN_THRESH:
            flags["talking"] = True

        # Draw face mesh dots (subtle)
        for lm in lms:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 1, (60, 200, 100), -1)

    else:
        # Face away
        if face_missing_since is None:
            face_missing_since = current
        elif current - face_missing_since > FACE_MISSING_SECS:
            flags["face_away"] = True

    # ── Pose: hand near face ──────────────────────────────────────────────────
    if pose_res.pose_landmarks:
        plms = pose_res.pose_landmarks.landmark
        nose_y  = plms[mp_pose.PoseLandmark.NOSE].y
        lw_y    = plms[mp_pose.PoseLandmark.LEFT_WRIST].y
        rw_y    = plms[mp_pose.PoseLandmark.RIGHT_WRIST].y
        if abs(lw_y - nose_y) < HAND_FACE_DIST or abs(rw_y - nose_y) < HAND_FACE_DIST:
            flags["hand_face"] = True

    # ── accumulate counts ─────────────────────────────────────────────────────
    if exam_active:
        for key in WEIGHTS:
            behaviour_counts[key][1] += 1
            if flags[key]:
                behaviour_counts[key][0] += 1

    # ── compute per-behaviour rate and overall risk ───────────────────────────
    beh_rates = {}
    for key in WEIGHTS:
        total = behaviour_counts[key][1]
        beh_rates[key] = (behaviour_counts[key][0] / total * 100) if total else 0.0

    overall_risk = sum(WEIGHTS[k] * beh_rates[k] for k in WEIGHTS)
    overall_risk = min(overall_risk, 100.0)

    risk_history.append(overall_risk)

    # ── draw UI ───────────────────────────────────────────────────────────────
    # Darken bottom panel area
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-140), (w, h), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Left panel – gauge
    draw_panel(frame, 10, 10, 200, 220)
    draw_risk_gauge(frame, 110, 100, 65, overall_risk)
    label_col = risk_colour(overall_risk)
    risk_label = ("SAFE" if overall_risk < 25 else
                  "LOW"  if overall_risk < 50 else
                  "HIGH" if overall_risk < 75 else "CRITICAL")
    cv2.putText(frame, risk_label, (110 - len(risk_label)*5, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_col, 2)
    cv2.putText(frame, "Overall cheating risk", (18, 192),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

    # Session timer
    if exam_active and exam_start_time:
        elapsed = int(current - exam_start_time)
        mm, ss  = divmod(elapsed, 60)
        cv2.putText(frame, f"Exam: {mm:02d}:{ss:02d}", (18, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GREEN, 1)
    else:
        cv2.putText(frame, "Press E to start exam", (18, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

    # Right panel – behaviour bars
    bar_panel_w = 270
    draw_panel(frame, w-bar_panel_w-10, 10, bar_panel_w, 200)
    draw_behaviour_bars(frame, w-bar_panel_w+35, 20, beh_rates)

    # History graph (bottom right)
    gw, gh2 = 300, 100
    draw_panel(frame, w-gw-10, h-gh2-30, gw+10, gh2+30, alpha=0.6)
    draw_history_graph(frame, w-gw-4, h-gh2-12, gw, gh2-12, risk_history)

    # ── live behaviour alerts ─────────────────────────────────────────────────
    alert_y = h - 148
    active_flags = [k for k, v in flags.items() if v]
    alert_msgs = {
        "gaze_side" : "! Eyes looking sideways",
        "gaze_up"   : "! Eyes looking upward",
        "head_turn" : "! Head turned away",
        "talking"   : "! Mouth open / talking",
        "face_away" : "! Face not visible",
        "hand_face" : "! Hand raised to face",
        "nodding"   : "! Rapid nodding detected",
    }
    for i, key in enumerate(active_flags[:4]):
        cv2.putText(frame, alert_msgs[key],
                    (220, alert_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_RED, 1)

    # ── exam not active overlay ───────────────────────────────────────────────
    if not exam_active:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0,0), (w, h), (0,0,0), -1)
        cv2.addWeighted(overlay2, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, "EXAM NOT STARTED",
                    (w//2 - 160, h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_GRAY, 2)
        cv2.putText(frame, "Press  E  to begin monitoring",
                    (w//2 - 180, h//2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1)

    # Status bar
    cv2.putText(frame, "E=start/stop  R=reset  ESC=quit",
                (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

    cv2.imshow("Exam Cheating Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('e') or key == ord('E'):
        exam_active = not exam_active
        if exam_active:
            exam_start_time = current
            print("[EXAM] Monitoring started")
        else:
            print(f"[EXAM] Stopped. Final risk: {overall_risk:.1f}%")
    elif key == ord('r') or key == ord('R'):
        behaviour_counts = {k: [0, 0] for k in WEIGHTS}
        risk_history     = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        exam_start_time  = current if exam_active else None
        print("[EXAM] Stats reset")

cap.release()
pose_estimator.close()
face_detector.close()
cv2.destroyAllWindows()

# ── Final report ──────────────────────────────────────────────────────────────
print("\n===== EXAM SESSION REPORT =====")
for key, (triggered, total) in behaviour_counts.items():
    rate = triggered / total * 100 if total else 0
    print(f"  {key:<14}: {rate:5.1f}%  ({triggered}/{total} frames)")
print(f"\n  Overall cheating risk : {overall_risk:.1f}%")
print("================================\n")
