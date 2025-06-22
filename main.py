# Use a pipeline as a high-level helper
# Can download hugging face model into local machine
from transformers import pipeline

from dotenv import load_dotenv, find_dotenv
import tkinter as tk
from tkinter import filedialog

# Python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import os
# For numerical operations
import numpy as np
# For saving the audio file
import soundfile as sf

# For loading speaker embeddings required for the TTS model
from datasets import load_dataset

# Load in environment variables from .env file
load_dotenv(find_dotenv())
# Loading in the HUGGINGFACEHUB_API_TOKEN from the environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") 

# Initialize the text-to-speech pipeline globally or within the function.
# Initializing it globally outside the function is more efficient so the model is loaded only once.
print("Loading Text-to-Speech model: microsoft/speecht5_tts")
# This will download the model the first time you run it.
try:
    tts_pipe = pipeline("text-to-audio", model="microsoft/speecht5_tts")
    print("Text-to-Speech model loaded successfully.")
    # Load speaker embeddings for the TTS model
    print("Loading speaker embeddings for the TTS model...")
    # Dataset contains pre-computed speaker embeddings
    embeddings = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    global_speaker_embedding = np.array(embeddings[7306]["xvector"])
    print("Speaker embeddings loaded successfully.")
except Exception as e:
    print(f"Error loading Text-to-Speech model: {e}")
    # You might want to exit or handle this gracefully if the model can't be loaded
    tts_pipe = None # Set to None so subsequent calls will fail gracefully
    global_speaker_embedding = None

# Using huggingface transformers library to create a pipeline for image to text conversion
def img2Text(url):
    # Set up pipline for huggingface transformers with image to text model

    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=20, device="cpu")
    
    # Process the image URL, access the first element of the output and the value of 'generated_text' key
    text = pipe(url)[0]['generated_text']
    
    print("Initial text:", text)
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
    prompt = PromptTemplate(template=pre_prompt, input_variables=["scenario"])
    
    # Choose the OpenAI model to use for story generation
    llm = ChatOpenAI(model="gpt-4", temperature=1, max_tokens=1000, verbose=True)
    
    # Create a runnable to run the prompt with the LLM
    # Chain them together using the pipe operator
    model = prompt | llm
    
    # Generate the story using the model and the provided scenario
    final_story = model.invoke({"scenario": scenario})
    
    print("Final story:", final_story.content)
    return final_story.content
    
# Convert the generated story into speech using a text-to-speech model
def text2Speech(story):
    
    if tts_pipe is None or global_speaker_embedding is None:
        print("Text-to-Speech model or speaker embeddings not loaded, cannot convert text to speech. Exiting.")
        return
    
    print("Converting story to speech...")
    try:
        # Pass speaker embedding using forward_params
        tts_output = tts_pipe(story, forward_params={"speaker_embedding": global_speaker_embedding})
        
        # Access the audio data and sampling rate
        audio_array = tts_output["audio"]
        sampling_rate = tts_output["sampling_rate"]
        
        # Normalize audio if needed (soundfile expects float between -1 and 1)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array))
        
        # Save the audio to a file (WAV is often safer for raw audio arrays, then convert if needed)
        # We can directly save as MP3 if soundfile and its dependencies support it, but WAV is universal.
        output_filename = "story_audio.wav" # Recommended to save as WAV first
        sf.write(output_filename, audio_array, sampling_rate)
        print(f"Audio saved to {output_filename}")
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        print("Please ensure your Hugging Face token is valid and the model dependencies are installed.")

if __name__ == "__main__":
    # Select an image file
    print("Please select an image file to process:\n") 
    image_url = select_image()
    
    if image_url:
        inital_text = img2Text(image_url)
        if inital_text: # Ensure text is generated
            final_story = generate_story(inital_text)
            if final_story: # Ensure story is generated
                text2Speech(final_story)
            else:
                print("Story generation failed or returned empty.")
        else:
            print("Image to text conversion failed or returned empty.")
    else:
        print("Image selection cancelled or failed. Exiting.")