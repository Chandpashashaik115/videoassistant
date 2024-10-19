import streamlit as st
import cv2
import numpy as np

# Initialize the camera state
if 'camera_enabled' not in st.session_state:
    st.session_state.camera_enabled = False

# Checkbox to enable camera
enable_camera = st.checkbox("Enable camera")

# Start and Stop buttons
if st.button("Start"):
    st.session_state.camera_enabled = True

if st.button("Stop"):
    st.session_state.camera_enabled = False

# Streaming live black-and-white feed using OpenCV
if enable_camera and st.session_state.camera_enabled:
    st.info("Streaming live black-and-white camera feed...")

    # Access the webcam (0 is typically the default webcam)
    cap = cv2.VideoCapture(0)

    # Create a placeholder in the Streamlit app for the video frames
    frame_placeholder = st.empty()

    # Loop to continuously capture frames and display them
    while st.session_state.camera_enabled and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture image.")
            break

        # Convert the frame to grayscale (black and white)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale frame back to RGB format for displaying in Streamlit
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Display the frame in the placeholder
        frame_placeholder.image(rgb_frame, channels='RGB')

    # Release the video capture object
    cap.release()

# Display camera status
if st.session_state.camera_enabled:
    st.success("Camera is active and streaming.")
else:
    st.warning("Camera is not active. Click 'Start' to enable.")
