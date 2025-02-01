import cv2
import numpy as np
import mediapipe as mp
import dlib
import pygame
import time

# Initialize Mediapipe for face landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# OpenCV pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# dlib face detector (backup)
detector = dlib.get_frontal_face_detector()

# Initialize pygame for buzzer sound
pygame.mixer.init()
pygame.mixer.music.load("buzzer.wav")  # Ensure you have a "buzzer.wav" file in the same directory

# Open Camera
cap = cv2.VideoCapture(0)

# Timers & Counters
closed_eye_frame_count = 0
head_down_count = 0
head_tilt_count = 0  # Track head tilting (left/right)
yawning_timestamps = []  # Store yawning times
fps = 30  # Adjust based on actual FPS

# Frames required for 4 seconds of eye closure
EYE_CLOSED_THRESHOLD_FRAMES = fps * 4  

# Function to trigger the buzzer
def buzzer_alert():
    pygame.mixer.music.play()
    print("🚨 [BUZZER] ALERT: Drowsiness Detected! 🚨")

# Function to display alert on screen
def display_alert(frame, text):
    cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

# Start video loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(frame_rgb)

    # Face detection (backup using dlib if needed)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        dlib_faces = detector(gray)
        if len(dlib_faces) == 0:
            print("⚠️ No face detected!")
            closed_eye_frame_count = 0
            head_down_count = 0
            head_tilt_count = 0
            continue
    
    # If Mediapipe detects face landmarks, proceed
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract key points for eyes, mouth, and nose
            left_eye_pts = [face_landmarks.landmark[i] for i in [33, 133, 160, 158]]
            right_eye_pts = [face_landmarks.landmark[i] for i in [362, 263, 386, 374]]
            nose_tip = face_landmarks.landmark[1]  # Nose tip for head position detection

            # Convert to pixel coordinates
            def landmark_to_pixel(landmarks, frame_shape):
                return [(int(p.x * frame_shape[1]), int(p.y * frame_shape[0])) for p in landmarks]

            left_eye_bbox = landmark_to_pixel(left_eye_pts, frame.shape)
            right_eye_bbox = landmark_to_pixel(right_eye_pts, frame.shape)
            nose_tip_pixel = (int(nose_tip.x * frame.shape[1]), int(nose_tip.y * frame.shape[0]))

            # Eye closure detection
            eyes_closed = False
            detected_eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
            if len(detected_eyes) == 0:
                eyes_closed = True
                closed_eye_frame_count += 1
            else:
                closed_eye_frame_count = 0

            # **NEW FEATURE: Trigger buzzer immediately if eyes closed for 4 seconds**
            if closed_eye_frame_count >= EYE_CLOSED_THRESHOLD_FRAMES:
                display_alert(frame, "🚨 EYES CLOSED FOR 4 SECONDS! 🚨")
                buzzer_alert()
                closed_eye_frame_count = 0  # Reset counter

            # Yawning detection
            mouth_detected = mouth_cascade.detectMultiScale(gray, 1.5, 15)
            if len(mouth_detected) > 0:
                yawning_timestamps.append(time.time())

                # Remove yawns older than 60 seconds
                yawning_timestamps = [t for t in yawning_timestamps if time.time() - t < 60]

            # If 3 yawns happen within 60 seconds AND eyes close -> Trigger buzzer
            if len(yawning_timestamps) >= 4 and eyes_closed:
                display_alert(frame, "🚨 3 YAWNS + EYES CLOSED! 🚨")
                buzzer_alert()
                yawning_timestamps = []  # Reset after alert
                closed_eye_frame_count = 0

            # Head-down detection
            if nose_tip_pixel[1] > frame.shape[0] * 0.7:  # If nose tip moves too low
                head_down_count += 1
            else:
                head_down_count = 0

            # Head tilt detection (left or right)
            left_eye_x = sum([pt[0] for pt in left_eye_bbox]) / len(left_eye_bbox)
            right_eye_x = sum([pt[0] for pt in right_eye_bbox]) / len(right_eye_bbox)
            nose_x = nose_tip_pixel[0]

            # Calculate head tilt: If nose is significantly shifted left or right from eye center
            eye_center_x = (left_eye_x + right_eye_x) / 2
            if abs(nose_x - eye_center_x) > frame.shape[1] * 0.1:  # 10% of frame width
                head_tilt_count += 1
            else:
                head_tilt_count = 0

            # Trigger alert if head is down or tilted left/right
            if head_down_count >= 5 or head_tilt_count >= 5:  # 5 consecutive frames (approx. 0.3 sec)
                display_alert(frame, "🚨 HEAD DOWN OR TILTED! 🚨")
                buzzer_alert()
                head_down_count = 0
                head_tilt_count = 0  # Reset after alert

    # Display Frame
    cv2.imshow("Driver Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
