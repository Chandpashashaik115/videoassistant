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
    if enable_camera:
        st.session_state.camera_enabled = True
    else:
        st.warning("Please enable the camera to start streaming.")

if st.button("Stop"):
    st.session_state.camera_enabled = False

# Streaming live black-and-white feed using OpenCV
if enable_camera and st.session_state.camera_enabled:
    st.info("Streaming live black-and-white camera feed...")

    # Create a placeholder in the Streamlit app for the video frames
    frame_placeholder = st.empty()

    # Loop to continuously capture frames and display them
    while st.session_state.camera_enabled:
        # Use the camera input from the user
        frame = st.camera_input("Take a picture", key="camera")

        if frame is not None:
            # Read the image as an array
            image = np.array(frame)

            # Convert the frame to grayscale (black and white)
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale frame back to RGB format for displaying in Streamlit
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Display the frame in the placeholder
            frame_placeholder.image(rgb_frame, channels='RGB')
        else:
            st.warning("Waiting for the camera input...")

# Display camera status
if st.session_state.camera_enabled:
    st.success("Camera is active and streaming.")
else:
    st.warning("Camera is not active. Click 'Start' to enable.")
