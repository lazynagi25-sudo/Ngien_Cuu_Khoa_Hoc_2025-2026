import streamlit as st
import pandas
import tensorflow as tf
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D class
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Load model
try:
    model = tf.keras.models.load_model(
        'modelnew.h5',
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
    )
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
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

# Function to get predictions from video
def get_predictions(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
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
            if pred_prob>0.7:
                label = "Non-Violence" if pred_class == 1 else "Violence"
                confidence = pred_prob * 100
            else:
                label = None
                confidence = pred_prob*0
        else:
            label = "Initializing..."
            confidence = 0

        # Display result
        if label:
            text = f"{label}: {confidence:.2f}%"
            color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if label=="Violence":
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Prediction Frame', use_column_width=True)

    cap.release()

# Streamlit UI
logo_path = "tay_thanh_logo.jpg"
st.image(logo_path, caption="Tay_Thanhlogo",width=200)
st.title("Violence Detection with Deep Learning")
st.write("Upload a video to detect violence scenes.")

# File uploader for video
video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Save the uploaded video file to a temporary location
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    st.video(video_file)

    st.write("Processing video...")

    # Call the function to process the video and get predictions
    get_predictions("uploaded_video.mp4")
