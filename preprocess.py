import cv2
import numpy as np
import mediapipe as mp
import dlib
import pygame
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Function to fix unsupported 'groups' parameter in DepthwiseConv2D
def custom_depthwise_conv2d(**kwargs):
    kwargs.pop("groups", None)  # Remove 'groups' parameter if it exists
    return DepthwiseConv2D(**kwargs)

# Load the trained Teachable Machine model with custom object scope
with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': custom_depthwise_conv2d}):
    model = load_model("keras_model.h5")

# Initialize Mediapipe for face landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# OpenCV pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

# Function to preprocess the image for model prediction
def preprocess_image(frame, target_size=(224, 224)):
    img = cv2.resize(frame, target_size)  # Resize to match model input size
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input
    return img

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

    # If at least one face is detected, proceed
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Predict using the Teachable Machine model
        processed_img = preprocess_image(face_img)
        predictions = model.predict(processed_img)[0]
        predicted_class = np.argmax(predictions)  # Get the class with the highest probability

        # Map predictions to labels
        labels = ["eye-closed", "mouth-closed", "eye-opened", "mouth-opened"]
        detected_label = labels[predicted_class]

        # Eye closure detection using model
        if detected_label == "eye-closed":
            closed_eye_frame_count += 1
        else:
            closed_eye_frame_count = 0

        # **IMPROVED: Trigger buzzer immediately if eyes closed for 4 seconds**
        if closed_eye_frame_count >= EYE_CLOSED_THRESHOLD_FRAMES:
            display_alert(frame, "🚨 EYES CLOSED FOR 4 SECONDS! 🚨")
            buzzer_alert()
            closed_eye_frame_count = 0  # Reset counter

        # Yawning detection using model
        if detected_label == "mouth-opened":  # If model detects yawning
            yawning_timestamps.append(time.time())

            # Remove yawns older than 60 seconds
            yawning_timestamps = [t for t in yawning_timestamps if time.time() - t < 60]

        # If 3 yawns happen within 60 seconds AND eyes close -> Trigger buzzer
        if len(yawning_timestamps) >= 3 and detected_label == "eye-closed":
            display_alert(frame, "🚨 3 YAWNS + EYES CLOSED! 🚨")
            buzzer_alert()
            yawning_timestamps = []  # Reset after alert
            closed_eye_frame_count = 0

        # Head-down detection
        nose_tip_pixel = (int(results.multi_face_landmarks[0].landmark[1].x * frame.shape[1]),
                          int(results.multi_face_landmarks[0].landmark[1].y * frame.shape[0]))

        if nose_tip_pixel[1] > frame.shape[0] * 0.7:  # If nose tip moves too low
            head_down_count += 1
        else:
            head_down_count = 0

        # Head tilt detection (left or right)
        eye_center_x = x + w // 2
        nose_x = nose_tip_pixel[0]

        if abs(nose_x - eye_center_x) > frame.shape[1] * 0.1:  # 10% of frame width
            head_tilt_count += 1
        else:
            head_tilt_count = 0

        # Trigger alert if head is down or tilted left/right
        if head_down_count >= 5 or head_tilt_count >= 5:  # 5 consecutive frames (~0.3 sec)
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
