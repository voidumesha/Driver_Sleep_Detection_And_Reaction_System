import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import RPi.GPIO as GPIO

# Load Pre-Trained Model
model = tf.keras.models.load_model("driver_drowsiness_model.h5")
labels = ['Closed_Eyes', 'Open_Eyes', 'Yawning', 'No_Yawning']

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Setup GPIO for Buzzer
BUZZER_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Open Pi Camera (For USB camera, use `cv2.VideoCapture(0)`)
cap = cv2.VideoCapture(0)

closed_eye_start_time = None  # Track eye closure duration
yawn_start_time = None  # Track yawning duration

def buzzer_alert():
    """ Function to turn on the buzzer for 3 seconds """
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(3)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye & mouth landmark coordinates
            left_eye = [face_landmarks.landmark[i] for i in [33, 133, 160, 158]]  # Left eye key points
            right_eye = [face_landmarks.landmark[i] for i in [362, 263, 386, 374]]  # Right eye key points
            mouth = [face_landmarks.landmark[i] for i in [13, 14, 78, 308]]  # Mouth key points

            # Convert to pixel coordinates
            def landmark_to_pixel(landmarks, frame_shape):
                return [(int(p.x * frame_shape[1]), int(p.y * frame_shape[0])) for p in landmarks]

            left_eye_pts = landmark_to_pixel(left_eye, frame.shape)
            right_eye_pts = landmark_to_pixel(right_eye, frame.shape)
            mouth_pts = landmark_to_pixel(mouth, frame.shape)

            # Define bounding boxes
            def get_bounding_box(pts, margin=10):
                x_min = min(p[0] for p in pts) - margin
                y_min = min(p[1] for p in pts) - margin
                x_max = max(p[0] for p in pts) + margin
                y_max = max(p[1] for p in pts) + margin
                return x_min, y_min, x_max, y_max

            left_eye_bbox = get_bounding_box(left_eye_pts)
            right_eye_bbox = get_bounding_box(right_eye_pts)
            mouth_bbox = get_bounding_box(mouth_pts, margin=20)

            # Crop the regions from frame
            def crop_region(frame, bbox):
                x_min, y_min, x_max, y_max = bbox
                return frame[max(0, y_min):min(frame.shape[0], y_max), max(0, x_min):min(frame.shape[1], x_max)]

            left_eye_region = crop_region(frame, left_eye_bbox)
            right_eye_region = crop_region(frame, right_eye_bbox)
            mouth_region = crop_region(frame, mouth_bbox)

            # Preprocess Image for CNN Model
            def preprocess(image):
                if image.shape[0] == 0 or image.shape[1] == 0:  # Avoid empty images
                    return None
                image = cv2.resize(image, (64, 64))
                image = image / 255.0  # Normalize
                image = np.expand_dims(image, axis=0)
                return image

            # Predict Eye State
            left_eye_input = preprocess(left_eye_region)
            right_eye_input = preprocess(right_eye_region)
            mouth_input = preprocess(mouth_region)

            if left_eye_input is not None and right_eye_input is not None:
                left_eye_prediction = model.predict(left_eye_input)
                right_eye_prediction = model.predict(right_eye_input)

                left_eye_label = labels[np.argmax(left_eye_prediction)]
                right_eye_label = labels[np.argmax(right_eye_prediction)]

                # Check if both eyes are closed
                if left_eye_label == "Closed_Eyes" and right_eye_label == "Closed_Eyes":
                    if closed_eye_start_time is None:
                        closed_eye_start_time = time.time()
                    elif time.time() - closed_eye_start_time >= 4:  # 4 seconds of closed eyes
                        print("🚨 Drowsiness Detected! Triggering Buzzer & LCD... 🚨")
                        buzzer_alert()
                else:
                    closed_eye_start_time = None  # Reset if eyes are open

            # Predict Mouth State
            if mouth_input is not None:
                mouth_prediction = model.predict(mouth_input)
                mouth_label = labels[np.argmax(mouth_prediction)]

                if mouth_label == "Yawning":
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    elif time.time() - yawn_start_time >= 3:  # 3 seconds of yawning
                        print("⚠️ Driver is Yawning Continuously! ⚠️")
                else:
                    yawn_start_time = None  # Reset if not yawning

            # Draw bounding boxes (Optional for debugging)
            cv2.rectangle(frame, (left_eye_bbox[0], left_eye_bbox[1]), (left_eye_bbox[2], left_eye_bbox[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (right_eye_bbox[0], right_eye_bbox[1]), (right_eye_bbox[2], right_eye_bbox[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (mouth_bbox[0], mouth_bbox[1]), (mouth_bbox[2], mouth_bbox[3]), (0, 255, 0), 2)

    cv2.imshow("Driver Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
