# Image-to-Speech Short Story

This project is a Python application that takes an image, generates a descriptive text from it, crafts a short story based on that description using a large language model, and then converts the story into spoken audio.

## Features

* **Image to Text:** Utilizes the `Salesforce/blip-image-captioning-base` model to generate a textual description of an uploaded image.
* **Story Generation:** Employs `LangChain` with `OpenAI GPT-4` to create a concise, engaging story (max 20 words) from the image description.
* **Text-to-Speech (TTS):** Converts the generated story into natural-sounding speech using `microsoft/speecht5_tts` and `microsoft/speecht5_hifigan`, incorporating speaker embeddings for voice quality.
* **Local File Selection:** Provides a graphical interface (`tkinter`) to easily select image files from your computer.
* **Hardware Acceleration:** Configured to leverage available GPU hardware (Apple Silicon's MPS or NVIDIA's CUDA) for Text-to-Speech processing.

## Setup

### Prerequisites

Before running the application, ensure you have Python (3.9+) and `pip` installed.

### Installation

1.  **Clone this repository (if applicable) or create your project directory.**
2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On macOS/Linux
    source venv/bin/activate
    # On Windows
    venv\Scripts\activate
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install transformers torch torchaudio datasets soundfile python-dotenv "langchain_openai>=0.1.0" "langchain_core>=0.1.0" "openai>=1.0.0" numpy "sentencepiece>=0.1.99" tkinter
    ```
    * **Note on `tkinter`:** Tkinter is usually included with Python installations. If you encounter issues, you might need to install `python3-tk` on Linux (e.g., `sudo apt-get install python3-tk`) or ensure your Python installation includes it on macOS/Windows.

### Environment Variables

This project requires API keys for accessing external services.

1.  Create a file named `.env` in the root directory of your project.
2.  Add your API keys to this file:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token_here"
    ```
    * **`OPENAI_API_KEY`**: Replace `"your_openai_api_key_here"` with your actual API key from OpenAI.
    * **`HUGGINGFACEHUB_API_TOKEN`**: While the models used (`microsoft/speecht5_tts` and `Salesforce/blip-image-captioning-base`) are public, having a Hugging Face API token (`HUGGINGFACEHUB_API_TOKEN`) can help with rate limits and is good practice for accessing models from the Hugging Face Hub. You can obtain one from your [Hugging Face profile settings](https://huggingface.co/settings/tokens).

## Usage

1.  **Ensure your virtual environment is activated.**
2.  **Run the `main.py` script:**
    ```bash
    python main.py
    ```
3.  A file dialog will appear, prompting you to select an image file (JPEG, PNG, PDF).
4.  Once an image is selected, the application will:
    * Generate a description.
    * Create a short story.
    * Convert the story to speech.
5.  An audio file named `story_audio.wav` will be saved in the same directory as your `main.py` script.

## Troubleshooting and Notes

* **Hardware Considerations for Performance:**
    * **Apple Silicon Macs (M1/M2/M3):** For optimal Text-to-Speech performance, ensure your macOS is updated to **macOS 14.0 (Sonoma) or newer**. Older macOS versions might cause some PyTorch operations to fall back to CPU, resulting in slower generation.
    * **NVIDIA GPUs (CUDA):** If you have an NVIDIA GPU and PyTorch is installed with CUDA support, the Text-to-Speech model will automatically try to utilize it for significant speedups.
    * **CPU-only Devices:** On systems without compatible GPUs (or if GPU acceleration is not detected/available), all model operations will run on the CPU. This will work but will be notably slower, especially for Text-to-Speech.
* **First Run Downloads:** The first time you run the script, Hugging Face models and datasets will be downloaded. This can take some time depending on your internet connection. Subsequent runs will use the cached models.
* **Image-to-Text (`device="cpu"`):** The image-to-text model (`Salesforce/blip-image-captioning-base`) is explicitly set to run on the CPU (`device="cpu"`) for stability and consistent performance, regardless of GPU availability.
* **`story_audio.wav`:** The output audio will be a WAV file. You can play this file with any standard media player.

This project is open-source.

Contact: Zaahir Sharma - sharma.zaahir@gmail.com - https://www.linkedin.com/in/zaahir-sharma/

Project Link: [https://github.com/zaahirsharma/Image-Short-Story]
