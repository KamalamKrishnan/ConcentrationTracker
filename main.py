import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Video Capture
cap = cv2.VideoCapture(0)

# For smoothing the gaze ratio
ratios = deque(maxlen=10)

# Distraction logic settings
distraction_frames = 0
FOCUS_FRAMES_REQUIRED = 5
DISTRACTED_FRAMES_THRESHOLD = 20  # ~1 sec if 20 fps
status = "FOCUSED"

# Eye ratio calculator


def calc_eye_ratio(landmarks, outer_idx, inner_idx, iris_idx):
    outer = landmarks[outer_idx]
    inner = landmarks[inner_idx]
    iris = landmarks[iris_idx]
    eye_width = inner.x - outer.x
    iris_offset = iris.x - outer.x
    return iris_offset / (eye_width + 1e-6)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    gaze_direction = "Undetected"

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]
        lm = face.landmark

        left_ratio = calc_eye_ratio(lm, 33, 133, 468)
        right_ratio = calc_eye_ratio(lm, 362, 263, 473)
        avg_ratio = (left_ratio + right_ratio) / 2
        ratios.append(avg_ratio)
        smooth_ratio = np.mean(ratios)

        # Determine gaze direction
        if smooth_ratio < 0.35:
            gaze_direction = "RIGHT"
        elif smooth_ratio > 0.65:
            gaze_direction = "LEFT"
        else:
            gaze_direction = "CENTER"

        # Distraction detection (Option B logic)
        if gaze_direction != "CENTER":
            distraction_frames += 1
            if distraction_frames >= DISTRACTED_FRAMES_THRESHOLD:
                status = "DISTRACTED"
        else:
            if distraction_frames > 0:
                distraction_frames -= FOCUS_FRAMES_REQUIRED
            if distraction_frames <= 0:
                distraction_frames = 0
                status = "FOCUSED"

        # Display debug info
        cv2.putText(frame, f"Gaze: {gaze_direction}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Ratio: {smooth_ratio:.2f}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"Status: {status}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if status == "DISTRACTED" else (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Face Not Detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Concentration Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
