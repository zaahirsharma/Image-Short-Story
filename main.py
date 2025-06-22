# Use a pipeline as a high-level helper
# Can download hugging face model into local machine
from transformers import pipeline

from dotenv import load_dotenv, find_dotenv
import tkinter as tk
from tkinter import filedialog

# Load in environment variables from .env file
load_dotenv(find_dotenv())

def img2Text(url):
    # Set up pipline for huggingface transformers with image to text model

    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Process the image URL, access the first element of the output
    text = pipe(url)[0]['generated_text']
    
    print(text)
    return text

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
    
    if file_path:
        print(f"Sucessfully loaded image from filepath: {file_path}")
    else:
        print("No file selected.")
        return None
    
    return file_path

image_url = select_image()
img2Text(image_url)