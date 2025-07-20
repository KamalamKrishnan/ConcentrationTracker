import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)
ratios = deque(maxlen=10)

# Eye landmarks (MediaPipe indices)
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_IRIS_CENTER = 468

RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
RIGHT_IRIS_CENTER = 473


def calc_eye_ratio(landmarks, outer_idx, inner_idx, iris_idx):
    outer = landmarks[outer_idx]
    inner = landmarks[inner_idx]
    iris = landmarks[iris_idx]

    eye_width = inner.x - outer.x
    iris_offset = iris.x - outer.x
    ratio = iris_offset / (eye_width + 1e-6)
    return ratio


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

        # Draw iris and eye points
        def draw_point(index, color):
            x, y = int(lm[index].x * w), int(lm[index].y * h)
            cv2.circle(frame, (x, y), 2, color, -1)

        draw_point(LEFT_EYE_OUTER, (255, 0, 0))
        draw_point(LEFT_EYE_INNER, (255, 0, 0))
        draw_point(LEFT_IRIS_CENTER, (0, 255, 255))

        draw_point(RIGHT_EYE_OUTER, (0, 0, 255))
        draw_point(RIGHT_EYE_INNER, (0, 0, 255))
        draw_point(RIGHT_IRIS_CENTER, (0, 255, 255))

        # Get ratios
        left_ratio = calc_eye_ratio(
            lm, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_IRIS_CENTER)
        right_ratio = calc_eye_ratio(
            lm, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_IRIS_CENTER)
        avg_ratio = (left_ratio + right_ratio) / 2
        ratios.append(avg_ratio)
        smooth_ratio = np.mean(ratios)

        # Classify gaze direction
        if smooth_ratio < 0.40:
            gaze_direction = "RIGHT"
        elif smooth_ratio > 0.60:
            gaze_direction = "LEFT"
        else:
            gaze_direction = "CENTER"

        # Display text
        cv2.putText(frame, f"Gaze: {gaze_direction}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Smooth Ratio: {smooth_ratio:.2f}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Accurate Gaze Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
