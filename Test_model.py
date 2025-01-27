import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('improved_driver_drowsiness_model.h5')

# Preprocess a single image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.6  # Lower confidence threshold for robust classification
labels = ['Closed_Eyes', 'Open_Eyes', 'Yawning', 'No_Yawning', 'Unknown']

# Test images
image_paths = ['car.jpg', 'coco.jpg', 'person_yawning.jpg']

for path in image_paths:
    image = preprocess_image(path)
    predictions = model.predict(image)
    confidence = np.max(predictions)
    predicted_label = labels[np.argmax(predictions)] if confidence >= CONFIDENCE_THRESHOLD else 'Unknown'
    
    print(f"Image: {path}, Predicted: {predicted_label} (Confidence: {confidence:.2f})")
