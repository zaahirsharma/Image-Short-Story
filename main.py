# Use a pipeline as a high-level helper
# Can download hugging face model into local machine
from transformers import pipeline

from dotenv import load_dotenv, find_dotenv
import tkinter as tk
from tkinter import filedialog

from langchain import PromptTemplate, LLMChain, OpenAI

# Load in environment variables from .env file
load_dotenv(find_dotenv())


# Using huggingface transformers library to create a pipeline for image to text conversion
def img2Text(url):
    # Set up pipline for huggingface transformers with image to text model

    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Process the image URL, access the first element of the output and the value of 'generated_text' key
    text = pipe(url)[0]['generated_text']
    
    return text

# Allows user to select an image file using a from computer files and returns filepath
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        initialdir="/", # Starting directory for the dialog
        title="Select a Photo File",
        filetypes=(
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("PDF files", "*.pdf")
        )
    )
    
    # Check if a file was selected
    if file_path:
        print(f"Sucessfully loaded image from filepath: {file_path}")
    else:
        print("No file selected.")
        return None
    
    return file_path

# Generate a story by using a language model based on the scenario generated from image to text model
def generate_story(scenario):
    # Creating a prompt for the LLM to generate a story based on the scenario
    pre_prompt = '''
        You are a storyteller. 
        You will be given a scenario and you will generate a story based on that scenario.
        Your story should be engaging, creative, and well-structured, but must be limited to no more than 20 words:
        
        CONTEXT: {scenario}
        STORY: 
    '''
    # Create a PromptTemplate object with the pre_prompt and input variable
    prompt = PromptTemplate(input_variables=["scenario"], template=pre_prompt)
    
    # Choose the OpenAI model to use for story generation
    model = OpenAI(model="gpt-3.5-turbo", temperature=1, prmopt=prompt, verbose=True)
    
    # Generate the story using the model and the provided scenario
    final_story = model.predict(scenario=scenario)
    
    return final_story
    
    



if __name__ == "__main__":
    # Select an image file
    print("Please select an image file to process:\n") 
    image_url = select_image()
    img2Text(image_url)