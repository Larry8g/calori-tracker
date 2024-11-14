import streamlit as st
from google.cloud import vision
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Ensure the environment variable for Google Vision API is set
vision_key_path = os.getenv("GOOGLE_VISION_API_KEY_PATH")
if not vision_key_path:
    raise ValueError("Google Vision API Key Path is not set in the .env file.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = vision_key_path

# Check if the Google Generative AI API key is available
genai_api_key = os.getenv("GOOGLE_API_KEY")
if not genai_api_key:
    raise ValueError("Google Generative AI API Key is not set in the .env file.")
genai.configure(api_key=genai_api_key)

def detect_food_items(image_path):
    """Uses Google Vision API to detect food items in an image."""
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Using label detection to identify food-related items in the image
    response = client.label_detection(image=image)
    
    food_items = []
    for label in response.label_annotations:
        if "food" in label.description.lower() or "fruit" in label.description.lower() or "vegetable" in label.description.lower():
            food_items.append(label.description)
    
    return food_items if food_items else ["Unable to detect specific food items."]

def get_gemini_response(input_prompt):
    """Generates nutritional analysis using Gemini API."""
    try:
        # Try using the 'generate_text' method instead of 'generate' (based on your error)
        response = genai.generate_text(
            model="gemini-1.5-flash",  # Verify this model name with the Gemini API docs
            prompt=input_prompt
        )
        
        # Assuming the response structure is like this
        return response['choices'][0]['text']
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit App Setup
st.set_page_config(page_title="Calori Advisor App")
st.header("Calori Advisor App")

uploaded_file = st.file_uploader("Choose an image of food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily for processing
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_path = "temp_image.jpg"
    image.save(image_path)  # Save temporarily to use with Vision API

    # Detect food items in the image
    food_items = detect_food_items(image_path)
    st.write("Detected food items:", ", ".join(food_items))

    # If food items are detected, proceed with nutritional analysis
    if food_items:
        input_prompt = f"""
        You are an expert nutritionist. Analyze the following food items and provide detailed nutritional information for each, including calories:
        {', '.join(food_items)}.

        For each item, provide:
        1. The number of calories
        2. The macronutrient breakdown (carbs, protein, fats)
        3. Any vitamins or minerals present

        Format the response like this:
        1. Item 1 - calories, carbs, protein, fat, vitamins, minerals
        2. Item 2 - calories, carbs, protein, fat, vitamins, minerals
        """
        
        # Get the Gemini API response
        response = get_gemini_response(input_prompt)
        
        # Display the nutritional analysis
        st.header("Nutritional Analysis")
        st.write(response)
else:
    st.write("Please upload an image.")
