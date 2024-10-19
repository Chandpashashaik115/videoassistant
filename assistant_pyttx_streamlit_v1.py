import base64
import time
import cv2
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.last_prompt = None
        self.last_response = None

    def answer(self, prompt, image):
        if not prompt:
            return

        self.last_prompt = prompt
        print("Prompt:", prompt)

        # Encode the image as a JPEG
        _, buffer = cv2.imencode(".jpeg", image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image_base64}
        ).strip()

        self.last_response = response
        print("Response:", response)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are being used to power a video assistant and you have knowledge on celebrities...
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", [{"type": "text", "text": "{prompt}"}, {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_base64}"}]),
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

# Streamlit UI
st.title("SAGE: A Video Assistant")
st.write("This application recognizes celebrities from your webcam feed and responds to your questions.")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Webcam Feed")
    picture = st.camera_input("Take a picture")  # Using Streamlit's camera input
    if picture:
        image = cv2.imdecode(np.frombuffer(picture.getbuffer(), np.uint8), cv2.IMREAD_COLOR)

with col2:
    st.write("### Chat")
    prompt = st.text_input("Ask a question:")
    
    if st.button("Submit") and picture is not None:
        assistant.answer(prompt, image)

# Display chat history
if assistant.last_prompt and assistant.last_response:
    st.markdown(f"*Prompt:* {assistant.last_prompt}")
    st.markdown(f"*Response:* {assistant.last_response}")
else:
    st.markdown("*Prompt:* Waiting for input...")
    st.markdown("*Response:* Waiting for response...")
