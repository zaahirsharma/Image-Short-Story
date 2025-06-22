# Use a pipeline as a high-level helper
# Can download hugging face model into local machine
from transformers import pipeline

from dotenv import load_dotenv, find_dotenv

# Load in environment variables from .env file
load_dotenv(find_dotenv())


