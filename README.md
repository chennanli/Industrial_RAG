# Qwen2.5-VL Image and Video Analysis: Industrial RAG Implementation

This project provides multiple implementations for image and video analysis using the Qwen2.5-VL multimodal model. The main goal is to explore and demonstrate how multimodal large language models can be used for industrial Retrieval-Augmented Generation (RAG) applications involving visual content.

## Project Motivation

I created this project to study and explore the capabilities of multimodal AI models like Qwen2.5-VL. My primary interest was in understanding how these models can analyze visual content (both images and videos) for practical industrial applications. This implementation serves as a foundation for exploring more complex Industrial RAG systems that incorporate visual data alongside text.

By breaking videos into frames and analyzing them with the model, this project demonstrates how even video content can be processed by current multimodal LLMs, opening up new possibilities for video-based RAG systems in industrial contexts.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Implementation Approaches](#implementation-approaches)
  - [Hugging Face Transformers](#hugging-face-transformers)
  - [Gradio vs FastAPI+HTML](#gradio-vs-fastapi-html)
- [Running the Applications](#running-the-applications)
  - [Image Analysis Only](#image-analysis-only)
  - [Video Analysis with FastAPI](#video-analysis-with-fastapi)
  - [Video Analysis with Gradio](#video-analysis-with-gradio)
  - [Video Analysis with Time Frame Control](#video-analysis-with-time-frame-control)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)

## Overview

This project uses the Qwen2.5-VL multimodal model to analyze images and videos. The model can identify objects, describe scenes, and answer questions about visual content. The project includes several implementations:

1. **Image Analysis Only**: A simple Gradio interface for analyzing single images
2. **Video Analysis with FastAPI**: A web application using FastAPI and HTML for video analysis
3. **Video Analysis with Gradio**: A Gradio interface with tabs for both image and video analysis
4. **Video Analysis with Time Frame Control**: An advanced Gradio interface with controls for frame extraction rate

## Setup

1. Activate the virtual environment:
   ```bash
   source qwen25_env/bin/activate
   ```

2. Install required dependencies (if not already installed):
   ```bash
   pip install gradio transformers torch pillow
   pip install fastapi uvicorn python-multipart jinja2 aiofiles
   ```

3. Optional: Install OpenCV for better video processing (not required):
   ```bash
   pip install opencv-python
   ```

## Implementation Approaches

### Hugging Face Transformers

All implementations use the Hugging Face Transformers library to run the Qwen2.5-VL model locally on your Mac. This approach:

- Downloads the model weights from Hugging Face
- Runs inference directly on your local machine
- Does not require external APIs or services like vLLM or Ollama
- Uses the following key components:
  - `Qwen2_5_VLForConditionalGeneration`: The model class
  - `AutoProcessor`: Handles tokenization and image preprocessing
  - `process_vision_info`: Processes images for the model

The model is loaded with these parameters:
```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

### Gradio vs FastAPI+HTML

#### Gradio

**Pros:**
- Specifically designed for ML/AI demos with minimal code
- Built-in widgets for common AI inputs/outputs (images, audio, video)
- Automatic progress bars and real-time updates
- No frontend knowledge required (HTML/CSS/JavaScript)
- Quick prototyping with minimal boilerplate
- Handles file uploads and preprocessing automatically

**Cons:**
- Less customizable UI than custom HTML/CSS
- Not ideal for production-scale applications
- Limited control over the backend architecture
- May have performance limitations with very large files

#### FastAPI+HTML

**Pros:**
- Full control over frontend design with custom HTML/CSS/JS
- Better for production-ready applications
- More scalable architecture for complex applications
- Better performance for streaming responses
- More control over API endpoints and backend logic
- Can be integrated with any frontend framework (React, Vue, etc.)

**Cons:**
- Requires more code and knowledge of web development
- Need to implement your own progress indicators and UI components
- More complex to set up file handling and preprocessing
- Steeper learning curve

## Running the Applications

### Image Analysis Only

**File:** `image_analysis_only.py`

**Run Command:**
```bash
source qwen25_env/bin/activate && python image_analysis_only.py
```

**Usage:**
1. Open the Gradio interface at http://127.0.0.1:7860
2. Upload an image
3. Enter a prompt (or use the default "Describe this image in detail")
4. Click "Submit" to analyze the image

### Video Analysis with FastAPI

**File:** `video_analysis.py`

**Run Command:**
```bash
source qwen25_env/bin/activate && python video_analysis.py
```

**Usage:**
1. Open the web interface at http://localhost:8000 or http://0.0.0.0:8000
2. Upload a video
3. Enter an object to detect (e.g., "person", "car", "dog")
4. Click "Analyze Video" to process the video
5. Results will stream back as frames are analyzed
6. Frames are saved to the "frames" directory

### Video Analysis with Gradio

**File:** `video_analysis_gradio.py`

**Run Command:**
```bash
source qwen25_env/bin/activate && python video_analysis_gradio.py
```

**Usage:**
1. Open the Gradio interface at http://127.0.0.1:7860
2. Switch between "Image Analysis" and "Video Analysis" tabs
3. Upload a video
4. Enter an object to detect
5. Click "Analyze Video" to process the video
6. Results will appear in the text box

### Video Analysis with Time Frame Control

**File:** `video_analysis_gradio_time_frame.py`

**Run Command:**
```bash
source qwen25_env/bin/activate && python video_analysis_gradio_time_frame.py
```

**Usage:**
1. Open the Gradio interface at http://127.0.0.1:7860
2. Switch between "Image Analysis" and "Video Analysis" tabs
3. Upload a video
4. Enter an object to detect
5. Adjust the "Frame Interval" slider to control how many seconds between extracted frames (1-10 seconds)
6. Adjust the "Maximum Frames" slider to limit the total number of frames extracted (5-100 frames)
7. Optionally uncheck "Analyze Frames" to only extract frames without analysis (much faster)
8. Click "Process Video" to start
9. Extracted frames are saved to the "saved_frames" directory and displayed in the gallery

## Troubleshooting

### Blank Page in FastAPI Interface
- Make sure the "templates" directory exists and contains the "index.html" file
- Check that the "uploads" and "frames" directories exist
- Try accessing both http://localhost:8000 and http://0.0.0.0:8000

### OpenCV Not Found
- The applications will work without OpenCV, but with reduced functionality
- Install OpenCV with: `pip install opencv-python`

### Slow Processing
- For faster processing, use the time frame control version and:
  - Increase the frame interval (extract fewer frames)
  - Decrease the maximum frames
  - Uncheck "Analyze Frames" to only extract frames without analysis

### Port Already in Use
- If you get an error about the port being in use, stop the running application or use a different port
- For Gradio: Add `server_port=7861` to the `demo.launch()` call
- For FastAPI: Change the port in `uvicorn.run(app, host="0.0.0.0", port=8001)`

## File Structure

- `example.py`: Original example code
- `image_analysis_only.py`: Simplified version for image analysis only
- `video_analysis.py`: FastAPI implementation for video analysis
- `video_analysis_gradio.py`: Gradio implementation for video analysis
- `video_analysis_gradio_time_frame.py`: Advanced Gradio implementation with time frame control
- `templates/index.html`: HTML template for the FastAPI implementation
- `frames/`: Directory where video frames are saved (FastAPI version)
- `saved_frames/`: Directory where video frames are saved (Gradio time frame version)
- `uploads/`: Directory where uploaded videos are saved (FastAPI version)