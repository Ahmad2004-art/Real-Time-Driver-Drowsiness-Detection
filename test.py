import time
import os
import cv2
import mediapipe as mp
import pygame
import math
import numpy as np

ALARM_SOUND = "PROGRESSIVE_BLEEP_xvo.wav"
CLOSE_SECONDS = 5.0
EAR_THRESHOLD = 0.20

LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EAR = [362, 385, 387, 263, 373, 380]

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_ear(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)

def init_alarm():
    if not os.path.isfile(ALARM_SOUND):
        raise FileNotFoundError(f"{ALARM_SOUND}")
    pygame.mixer.init()
    pygame.mixer.music.load(ALARM_SOUND)
    pygame.mixer.music.set_volume(1.0)

def alarm_on():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(loops=-1)

def alarm_off():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def landmarks_to_points(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

def draw_eye_shape(frame, points, color):
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    if len(pts) >= 5:
        ellipse = cv2.fitEllipse(pts)
        cv2.ellipse(frame, ellipse, color, 2)

def main():
    init_alarm()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera error")

    closed_start = None
    alarm_playing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        now = time.time()
        status = "NO FACE"
        ear_val = None
        dur = 0.0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            left_ear_pts = landmarks_to_points(lm, LEFT_EAR, w, h)
            right_ear_pts = landmarks_to_points(lm, RIGHT_EAR, w, h)

            ear_left = compute_ear(left_ear_pts)
            ear_right = compute_ear(right_ear_pts)
            ear_val = (ear_left + ear_right) / 2.0

            is_closed = ear_val < EAR_THRESHOLD
            status = "CLOSED" if is_closed else "OPEN"

            left_contour = landmarks_to_points(lm, LEFT_EYE_CONTOUR, w, h)
            right_contour = landmarks_to_points(lm, RIGHT_EYE_CONTOUR, w, h)

            color = (0, 0, 255) if is_closed else (0, 255, 0)

            draw_eye_shape(frame, left_contour, color)
            draw_eye_shape(frame, right_contour, color)

            if is_closed:
                if closed_start is None:
                    closed_start = now
                dur = now - closed_start

                if dur >= CLOSE_SECONDS and not alarm_playing:
                    alarm_on()
                    alarm_playing = True
            else:
                closed_start = None
                dur = 0.0
                if alarm_playing:
                    alarm_off()
                    alarm_playing = False

        cv2.putText(frame, f"State: {status}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if ear_val is not None:
            cv2.putText(frame, f"EAR: {ear_val:.3f}  thr={EAR_THRESHOLD}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Closed time: {dur:.1f}s / {CLOSE_SECONDS}s", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Driver Eye Detection (EAR)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q')]:
            break
        if key in [ord('s'), ord('S')]:
            alarm_off()
            alarm_playing = False
            closed_start = None

    cap.release()
    alarm_off()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()