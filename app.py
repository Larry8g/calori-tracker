import streamlit as st
from google.cloud import vision
import os
from dotenv import load_dotenv
import openai
from PIL import Image
import tempfile

# Load environment variables from .env file
load_dotenv()

# Set up Google Vision API and OpenAI API keys
vision_key_path = os.getenv("GOOGLE_VISION_API_KEY_PATH")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = vision_key_path

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Function to detect food items using Google Vision API
def detect_food_items(image_path):
    """Uses Google Vision API to detect food items in an image."""
    client = vision.ImageAnnotatorClient()
    try:
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.label_detection(image=image)

        # Extract food-related items
        food_items = [
            label.description for label in response.label_annotations
            if "food" in label.description.lower() or 
               "fruit" in label.description.lower() or 
               "vegetable" in label.description.lower()
        ]
        return food_items if food_items else ["Unable to detect specific food items."]
    except Exception as e:
        return [f"Error detecting food items: {str(e)}"]

# Function to get response from OpenAI API for nutritional analysis
def get_openai_response(input_prompt):
    """Generates nutritional analysis using OpenAI API."""
    try:
        # Use ChatCompletion API to generate the response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Specify the model (e.g., gpt-3.5-turbo or gpt-4)
            messages=[
                {"role": "system", "content": "You are a helpful nutritionist."},
                {"role": "user", "content": input_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        # Extract and return the response text
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit App Setup
st.set_page_config(page_title="Calori Advisor App")
st.header("Calori Advisor App")

# File uploader for food image
uploaded_file = st.file_uploader("Choose an image of food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    # Save the uploaded image temporarily for processing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        image.save(temp_image.name)
        food_items = detect_food_items(temp_image.name)
    
    # Display detected food items or errors
    if food_items == ["Unable to detect specific food items."]:
        st.warning("No specific food items detected. Try uploading a clearer image.")
    elif "Error detecting food items" in food_items[0]:
        st.error(food_items[0])  # Display the error message
    else:
        st.write("Detected food items:", ", ".join(food_items))
        
        # Create a prompt for nutritional analysis
        input_prompt = f"""
        Please analyze the following food items and provide detailed nutritional information for each, including calories:
        {', '.join(food_items)}.

        For each item, provide:
        1. The number of calories
        2. The macronutrient breakdown (carbs, protein, fats)
        3. Any vitamins or minerals present

        Format the response like this:
        1. Item 1 - calories, carbs, protein, fat, vitamins, minerals
        2. Item 2 - calories, carbs, protein, fat, vitamins, minerals
        """
        
        # Get response from OpenAI API
        response = get_openai_response(input_prompt)
        
        # Display nutritional analysis or error message
        st.header("Nutritional Analysis")
        if "Error generating response" in response:
            st.error(response)
        else:
            st.write(response)
else:
    st.write("Please upload an image.")
