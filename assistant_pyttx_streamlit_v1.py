import cv2
import streamlit as st
import numpy as np

def main():
    st.title("Webcam Stream in Black and White using OpenCV and Streamlit")

    # Initialize session state for start/stop control
    if 'run' not in st.session_state:
        st.session_state['run'] = False

    # "Start" button to toggle webcam streaming
    if st.button("Start", key="start_button"):
        st.session_state['run'] = True

    # "Stop" button to toggle off webcam streaming
    if st.button("Stop", key="stop_button"):
        st.session_state['run'] = False

    # Placeholder for video stream
    frame_placeholder = st.empty()

    # If "Start" has been pressed, capture video from webcam
    if st.session_state['run']:
        cap = cv2.VideoCapture(0)  # Change 0 to 1 or another index if necessary

        # Check if webcam is opened correctly
        if not cap.isOpened():
            st.error("Cannot access the webcam! Please check permissions or webcam connection.")
            return

        # Loop to read and display frames in the Streamlit app
        while st.session_state['run']:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            # Convert the color from BGR (OpenCV format) to Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Update the image in the Streamlit app
            frame_placeholder.image(frame, channels="GRAY")

            # Break the loop if the user clicks the "Stop" button
            if not st.session_state['run']:
                break

        # Release the webcam and close the stream
        cap.release()

    # Give user a way to provide camera access if denied
    if st.button("Request Camera Access", key="request_camera_access"):
        st.warning("Please refresh the page and allow camera access when prompted.")

if __name__ == "__main__":
    main()
