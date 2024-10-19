import base64
import os
import time
import wave
import numpy as np
import threading  # For running TTS asynchronously
import streamlit as st
import cv2
from cv2 import imencode
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from speech_recognition import Microphone, Recognizer, UnknownValueError
import pyttsx3

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.last_prompt = None
        self.last_response = None
        self.tts_lock = threading.Lock()

    def answer(self, prompt, image):
        if not prompt:
            return

        self.last_prompt = prompt
        print("Prompt:", prompt)

        # Encode the image as a JPEG
        _, buffer = cv2.imencode(".jpeg", image)
        
        # Convert the buffer to a base64 string
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image_base64},
            config={"configurable": {"session_id": "unused"}} 
        ).strip()

        self.last_response = response
        print("Response:", response)
        
        if response:
            self._tts(response)

    def _tts(self, response):
        """Convert the response text to speech using pyttsx3 in a separate thread."""
        def speak():
            with self.tts_lock:
                # Initialize pyttsx3 engine every time
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Set TTS speaking rate
                engine.setProperty('volume', 1.0)  # Set TTS volume
                
                engine.say(response)
                engine.runAndWait()  # Wait for TTS to finish speaking
                
                # Stop and reset engine after finishing
                engine.stop()

        # Run TTS in a separate thread to avoid blocking the main thread
        threading.Thread(target=speak).start()

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are being used to power a video assistant and you have knowledge on celebrities that will use the chat history and the image 
        provided by the user to answer its questions. Wait for the user prompt
        and greet them for the first time.

        recognize the actors in the image and answer the questions based on the actors in the image.

        recognize the image who is speaking with you and remember his name and whenever he asks respond accordingly.

        Do not use any emoticons or emojis. Do not use any special characters. Answer straight to the point. Don't tell
        the user about what you are learning.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """        

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

def audio_callback(recognizer, audio):
    """Process audio input and send it to the assistant."""
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, current_image)  # Process the audio input
    except UnknownValueError:
        print("There was an error processing the audio.")

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
assistant = Assistant(model)

# Initialize recognizer and microphone
recognizer = Recognizer()
microphone = Microphone()

# Start listening to the microphone in a separate thread
def start_listening():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        stop_listening_callback = recognizer.listen_in_background(microphone, audio_callback)

# Start the audio listening thread
listening_thread = threading.Thread(target=start_listening)
listening_thread.start()

# Streamlit UI
st.title("SAGE: A Video Assistant")
st.write("This application recognizes celebrities from your webcam feed and responds to your questions.")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Webcam Feed")
    current_image = st.camera_input("Take a picture")

with col2:
    st.write("### Chat")
    prompt_placeholder = st.empty()
    response_placeholder = st.empty()

# Loop to display chat messages
while True:
    if current_image is not None:
        # Process the image if available
        assistant.answer(assistant.last_prompt, current_image.read())
        
        # Display the chat history
        if assistant.last_prompt and assistant.last_response:
            prompt_placeholder.markdown(f"*Prompt:* {assistant.last_prompt}")
            response_placeholder.markdown(f"*Response:* {assistant.last_response}")
        elif assistant.last_response:
            response_placeholder.markdown(f"*Response:* {assistant.last_response}")
        else:
            prompt_placeholder.markdown("*Prompt:* Waiting for input...")
            response_placeholder.markdown("*Response:* Waiting for response...")

    else:
        # If no image is captured
        prompt_placeholder.markdown("*Prompt:* Waiting for input...")
        response_placeholder.markdown("*Response:* Waiting for response...")

    time.sleep(0.1)  # Adjust the sleep time as necessary

# Note: The thread will keep running and listening to audio input
