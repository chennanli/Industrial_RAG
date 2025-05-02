# Multimodal Analysis and RAG System (Tabbed Interface)

This project provides a unified application for multimodal analysis and Retrieval-Augmented Generation (RAG) using the Qwen2.5-VL model. It combines text, image, and video analysis with knowledge base integration, presented through a user-friendly tabbed interface.

## Overview

The main application, `combined_tabbed.py`, integrates the functionalities of previous individual scripts into a single Gradio interface with three distinct tabs:

1.  **Concise RAG**: Perform RAG queries using a knowledge base built from PDF documents, supporting text, image, and video inputs.
2.  **Image Analysis Only**: Analyze single images for detailed descriptions or object identification.
3.  **Video Analysis & Sessions**: Extract and analyze frames from videos with time frame control, and manage saved video sessions.

The system uses the Qwen2.5-VL multimodal model for visual and text processing and can optionally use LangChain with FAISS for advanced knowledge retrieval (falls back to TF-IDF if LangChain dependencies are not met).

## Setup

1.  **Clone or download the project files.**
2.  **Activate the virtual environment:**
    ```bash
    source qwen25_env/bin/activate
    ```
    If you don't have the `qwen25_env` virtual environment, you'll need to create one and install the dependencies into it.
3.  **Install required dependencies:**
    Ensure your virtual environment is active, then run:
    ```bash
    pip3 install -r requirements.txt
    ```
    This will install all necessary packages, including `gradio`, `transformers`, `torch`, `opencv-python`, `qwen-vl-utils`, and the LangChain components (`langchain`, `langchain-community`, `faiss-cpu`, `sentence-transformers`) for enhanced RAG.

## Usage

1.  **Place PDF Files (for RAG):** Add your PDF documents to the `RAG_pdf` folder within the project directory.
2.  **Run the Application:**
    Activate your Python environment:
    ```bash
    source qwen25_env/bin/activate
    ```
    Then run the main script:
    ```bash
    python combined_tabbed.py
    ```
3.  **Access the Interface:** Open the local URL provided in the terminal output (e.g., `http://127.0.0.1:7860`) in your web browser.
4.  **Using the Tabs:**
    *   **Concise RAG Tab:**
        *   Click "Process PDF Knowledge Base" to load your documents.
        *   Enter your question and optionally upload an image or video.
        *   Click "Submit RAG Query" to get an answer based on the model's understanding and the knowledge base.
    *   **Image Analysis Only Tab:**
        *   Upload an image.
        *   Optionally enter an object to identify.
        *   Click "Analyze Image" to get a description or object analysis.
    *   **Video Analysis & Sessions Tab:**
        *   **Process New Video:** Upload a video, adjust frame extraction settings (interval, max frames), and click "Analyze Video" (or just "Process Video" if analysis is unchecked) to extract and analyze frames.
        *   **Use Existing Frames:** Select a previously processed session from the dropdown to view its frames in the gallery. You can refresh the list or delete sessions.

## How It Works

-   **Knowledge Base**: PDF documents are processed into text chunks, vectorized (using LangChain/FAISS or TF-IDF fallback), and stored for efficient retrieval.
-   **Multimodal Processing**: The Qwen2.5-VL model analyzes images and video frames (extracted from videos).
-   **RAG Integration**: For RAG queries, the system combines the user's question with relevant information retrieved from the knowledge base and the analysis of any provided image or video, feeding this enhanced context to the Qwen model for a comprehensive answer.
-   **Basic Analysis**: The Image Analysis and Video Analysis tabs provide direct analysis results from the Qwen model without necessarily querying the PDF knowledge base (unless explicitly part of the prompt).

## File Structure

-   `combined_tabbed.py`: The main application file containing all integrated code and the Gradio UI.
-   `requirements.txt`: Lists all project dependencies.
-   `LICENSE`: Specifies the personal, non-commercial use license.
-   `RAG_pdf/`: Folder to place your PDF files for the knowledge base.
-   `saved_frames/`: Directory where extracted video frames are saved.
-   `archieved/`: Contains older versions of the Python scripts.
-   `frame_cache/`: Cache directory for video frame processing.

## Troubleshooting

-   **Import Errors**: Ensure your `qwen25_env` virtual environment is active and you have run `pip3 install -r requirements.txt`.
-   **LangChain Fallback**: If you see messages about LangChain imports failing but the app still runs, it's using the TF-IDF fallback. Install the LangChain packages mentioned in `requirements.txt` for the advanced RAG.
-   **Slow Processing**: Reduce the number of frames extracted for video analysis or RAG.
-   **Port Already in Use**: Stop the previous process or modify the `demo.launch()` call in `combined_tabbed.py` to use a different port (e.g., `demo.launch(server_port=7861)`).
