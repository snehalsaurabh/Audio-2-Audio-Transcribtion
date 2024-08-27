import streamlit as st
import dotenv
import random
import os
from PIL import Image
import google.generativeai as genai
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

dotenv.load_dotenv()

google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# Step 1: Extract the text from the PDF
pdf_file_path = "uploads/text.pdf"
extracted_text = extract_text_from_pdf(pdf_file_path)

# Step 2: Convert text into embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = extracted_text.split('\n')  # Split text into sentences/paragraphs
embeddings = model.encode(sentences)

# Step 3: Create a FAISS index and add embeddings
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(embeddings))

# Function to query the vector database
def query_vector_db(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [sentences[idx] for idx in indices[0]]
    return results

def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages

# Function to query and stream the response from the LLM
def stream_llm_response(model_params, model_type="google", api_key=None):
    response_message = ""

    if model_type == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name = model_params["model"],
            generation_config={
                "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
            }
        )

        # Saving message history
        gemini_messages = messages_to_gemini(st.session_state.messages)

        for chunk in model.generate_content(
            contents=gemini_messages,
            stream=True,
        ):
            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="Mozarella",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>Mozarella</i> 💬</h1>""")

    # --- Main Content ---
    google_api_key = os.getenv("GOOGLE_API_KEY") if os.getenv("GOOGLE_API_KEY") is not None else ""
    if google_api_key == "" or google_api_key is None:
        st.write("#")
        st.warning("⬅️ Please introduce your Google API Key to continue...")

        with st.sidebar:
            st.write("#")
            st.write("#")
            st.video("https://www.youtube.com/watch?v=7i9j8M_zidA")
            st.write("📋[Medium Blog: Google Gemini](https://medium.com/@enricdomingo/how-i-add-gemini-1-5-pro-api-to-my-app-chat-with-videos-images-and-audios-f42171606143)")

    else:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        # Sidebar Model Options and Inputs
        with st.sidebar:
            st.divider()
            
            available_models = google_models if google_api_key else []
            model = st.selectbox("Select a model:", available_models, index=0)
            model_type = "google"
            
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            audio_response = st.toggle("Audio response", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()

            # Image Upload
            if model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
                    
                st.write("### **🖼️ Add an image or a video file:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        if img_type == "video/mp4":
                            # save the video file
                            video_id = random.randint(100000, 999999)
                            with open(f"video_{video_id}.mp4", "wb") as f:
                                f.write(st.session_state.uploaded_img.read())
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "video_file",
                                        "video_file": f"video_{video_id}.mp4",
                                    }]
                                }
                            )
                        else:
                            raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                            img = get_image_base64(raw_img)
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{img_type};base64,{img}"}
                                    }]
                                }
                            )

                cols_img = st.columns(2)

                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            "Upload an image or a video:", 
                            type=["png", "jpg", "jpeg", "mp4"], 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:                    
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )

            # Audio Upload
            st.write("#")
            st.write("### **🎤 Add an audio:**")

            audio_prompt = None
            audio_file_added = False
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None

            speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
            if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                st.session_state.prev_speech_hash = hash(speech_input)
                # save the audio file
                audio_id = random.randint(100000, 999999)
                with open(f"audio_{audio_id}.wav", "wb") as f:
                    f.write(speech_input)

                st.session_state.messages.append(
                    {
                        "role": "user", 
                        "content": [{
                            "type": "audio_file",
                            "audio_file": f"audio_{audio_id}.wav",
                        }]
                    }
                )

                audio_file_added = True

        # Chat input
        if user_query := st.chat_input("Hi! Ask me anything...") or audio_prompt or audio_file_added:
            if not audio_file_added:
                st.session_state.messages.append(
                    {
                        "role": "user", 
                        "content": [{
                            "type": "text",
                            "text": user_query or audio_prompt,
                        }]
                    }
                )
                
                # Display the new messages
                with st.chat_message("user"):
                    st.markdown(user_query)

            else:
                # Display the audio file
                with st.chat_message("user"):
                    st.audio(f"audio_{audio_id}.wav")

            # Step 1: Query the vector database for relevant text from the PDF
            relevant_texts = query_vector_db(user_query)
            
            # Step 2: Combine the relevant text with a response template
            response = f"Relevant information from the document:\n\n" + "\n".join(relevant_texts)
            
            # Add this combined response to the messages
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}]
                }
            )

            # Display the assistant's response
            with st.chat_message("assistant"):
                st.write(response)

            # Step 3: Generate an additional response from the LLM, if necessary
            with st.chat_message("assistant"):
                model2key = {
                    "google": google_api_key,
                }
                st.write_stream(
                    stream_llm_response(
                        model_params=model_params, 
                        model_type=model_type, 
                        api_key=model2key[model_type]
                    )
                )

if __name__ == "__main__":
    main()