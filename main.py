from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to convert image to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

# Endpoint to upload image and get response from Google Gemini
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded image to the uploads folder
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Open the image and convert it to base64
    image = Image.open(file_path)
    image_base64 = get_image_base64(image)

    # Predefined question
    question = "What is happening in this image?"

    # Create the message for Google Gemini
    gemini_message = {
        "role": "user",
        "parts": [
            {
                "mime_type": "image/jpeg",  # Assuming the image is in JPEG format
                "data": image_base64
            },
            {
                "text": question
            }
        ]
    }

    # Send the message to Google Gemini and get the response
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config={
            "temperature": 0.3,
        }
    )

    response_message = ""
    for chunk in model.generate_content(
        contents=[gemini_message],
        stream=True,
    ):
        chunk_text = chunk.text or ""
        response_message += chunk_text

    # Return the response as JSON
    return JSONResponse(content={"response": response_message})

# Serve the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)