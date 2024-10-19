import streamlit as st
import cv2
import numpy as np

# Set the title of the app
st.title("Live Video Stream in Black and White")

# Create a placeholder for the video stream
video_placeholder = st.empty()

# Create a camera input for taking a picture
img_file_buffer = st.camera_input("Take a picture")

# If a picture is taken, display it and process for black and white
if img_file_buffer is not None:
    # Read image file buffer with OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to black and white
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    # Display the black and white image
    st.image(gray_img, channels="GRAY", caption="Captured Image in Black and White")
    
# For continuous streaming, simulate a video feed
# Using OpenCV to create a fake video stream from webcam
if st.button("Start Video Stream"):
    cap = cv2.VideoCapture(0)  # Start video capture from webcam

    # Continuously capture video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        # Convert the frame to black and white
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert the frame to bytes for Streamlit
        _, buffer = cv2.imencode('.jpg', gray_frame)
        frame_bytes = buffer.tobytes()

        # Update the placeholder with the new frame
        video_placeholder.image(frame_bytes, channels="GRAY")

        # Stop streaming when a button is pressed
        if st.button("Stop Video Stream"):
            break

    # Release the video capture object
    cap.release()
