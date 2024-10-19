import streamlit as st
import cv2
import numpy as np

# Create a Streamlit app title
st.title("Live Video Stream in Black and White")

# Start capturing video from webcam
video_capture = cv2.VideoCapture(0)

# Streamlit component to show the video
if video_capture.isOpened():
    while True:
        ret, frame = video_capture.read()  # Read a frame from the webcam
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert the frame to black and white
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the frame to bytes for Streamlit
        _, buffer = cv2.imencode('.jpg', gray_frame)
        frame_bytes = buffer.tobytes()

        # Display the black and white video frame in Streamlit
        st.image(frame_bytes, channels="GRAY")

        # Add a stop button to end the video capture
        if st.button("Stop"):
            break

# Release the video capture when done
video_capture.release()
