import tensorflow as tf
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.layers import DepthwiseConv2D
# DepthwiseConv2D

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Load model
try:
    model = tf.keras.models.load_model(
        'Violence_Detection/modelnew.h5',
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
    )
    print("Model loaded successfully.")
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Parameters
FRAME_SIZE = (128, 128)
WINDOW_SIZE = 10  # Number of frames to average predictions
PRED_QUEUE = deque(maxlen=WINDOW_SIZE)  # Store predictions

# Preprocessing function
def preprocess_frame(frame):
    frame = cv2.resize(frame, FRAME_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return frame

# Open webcam
video = 'D:\Project\Violence_Detection\Violence-Alert-System-main\Violence_Detection\\Testing videos\\nonv.mp4'
# video = 0
cap = cv2.VideoCapture(video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess and predict
    processed_frame = preprocess_frame(frame)
    input_frame = np.expand_dims(processed_frame, axis=0)
    predictions = model.predict(input_frame, verbose=0)
    PRED_QUEUE.append(predictions[0])  # Store prediction probabilities

    # Average predictions
    if len(PRED_QUEUE) == WINDOW_SIZE:
        avg_predictions = np.mean(PRED_QUEUE, axis=0)
        pred_class = np.argmax(avg_predictions)
        pred_prob = avg_predictions[pred_class]
        label = "Non-Violence" if pred_class == 1 else "Violence"
        confidence = pred_prob * 100
    else:
        label = "Initializing..."
        confidence = 0

    # Display result
    text = f"{label}: {confidence:.2f}%"
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Violence Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()