import streamlit as st
import dotenv
import random
import os
from PIL import Image
import google.generativeai as genai
from audio_recorder_streamlit import st_audio_recorder
import base64
from io import BytesIO

dotenv.load_dotenv()

google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
