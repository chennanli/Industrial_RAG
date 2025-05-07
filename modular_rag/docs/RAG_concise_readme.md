# Unified RAG System

A multimodal Retrieval-Augmented Generation (RAG) system that supports text, image, and video analysis with knowledge base integration.

## Overview

This system provides a unified interface for:

1. **Text RAG**: Query a knowledge base built from PDF documents
2. **Image Analysis + Knowledge Base**: Upload images and get answers that combine visual analysis with knowledge retrieval
3. **Video Analysis + Knowledge Base**: Upload videos and get answers that combine video frame analysis with knowledge retrieval

The system uses the Qwen2.5-VL multimodal model for all processing tasks and integrates FAISS vector search for knowledge retrieval.

## Key Features

- **Multimodal RAG**: Combines visual understanding with knowledge base retrieval
- **PDF Knowledge Base**: Automatically processes and indexes PDF files
- **Unified Interface**: Handles text, images, and videos in a single application
- **Clean UI**: Custom interface with no Gradio branding

## Requirements

- Python 3.8+
- Hugging Face Transformers
- PyTorch
- qwen_vl_utils
- LangChain, FAISS for vector storage
- Dependencies: gradio, PyMuPDF, langchain, FAISS, sentence-transformers, OpenCV, PIL

## Installation

This system has primary dependencies and fallback mechanisms to work even if some packages are missing. The recommended installation is:

```bash
pip install -r requirements.txt
```

### Dependency Flexibility

The system has several fallback mechanisms:

1. **PDF Processing**: Primary option is PyMuPDF, but will fall back to pdfplumber or basic text extraction
2. **Vector Store**: Primary option is LangChain with FAISS, but will fall back to scikit-learn's TF-IDF if LangChain is not available

This ensures the system can run even with minimal dependencies installed.

## Usage

1. **Place Documents**: Add your files to the `RAG_pdf` folder (supports PDF and other material types)

2. **Run the Application**:
   Activate your Python environment first:
   ```bash
   source qwen25_env/bin/activate
   ```
   Then run the main script:
   ```bash
   python launch_rag.py
   ```

3. **Using the Interface**:
   - Click "Process PDF Knowledge Base" to process documents
   - Enter your question in the text field
   - Choose your query mode:
     - **Text-only Query**: Uses the PDF knowledge base to answer questions
     - **Image Analysis + Knowledge Base**: Upload an image and get insights based on both visual analysis and knowledge retrieval
     - **Video Analysis + Knowledge Base**: Upload a video and get insights based on video content and knowledge retrieval
   - Click "Submit" to get your answer

## How It Works

### Text RAG System
- **PDF Processing**: Extracts text from PDFs in the RAG_pdf folder
- **Text Chunking**: Splits text into manageable chunks
- **Vectorization**: Creates vector embeddings using HuggingFace Embeddings
- **Retrieval**: Uses FAISS to find relevant content based on queries
- **Generation**: Constructs a prompt with relevant context and sends to Qwen2.5-VL

### Image Analysis + Knowledge Base
- **Image Description**: First analyzes the image to identify key elements
- **Context Retrieval**: Uses the image description plus query to search the knowledge base
- **Combined Analysis**: Provides answers that incorporate both visual information and retrieved knowledge
- **Enhanced Understanding**: Enables answering questions like "What might be wrong with this equipment?" with reference to technical documentation

### Video Analysis + Knowledge Base
- **Key Frame Extraction**: Extracts representative frames from the video
- **Frame Analysis**: Analyzes each key frame to understand video content
- **Knowledge Integration**: Combines frame descriptions with knowledge base search
- **Comprehensive Response**: Provides answers that reference both the video content and technical documentation

## Advanced Configuration

You can modify the following parameters in the code:

- `PDF_FOLDER`: Change the folder where PDFs are stored
- `chunk_size` and `chunk_overlap`: Adjust text chunking parameters
- `top_k`: Number of relevant chunks to retrieve for RAG

## Limitations

- Relies on the quality and capabilities of the Qwen2.5-VL model
- Video analysis is limited to a few key frames to maintain reasonable processing times
- PDF processing is done in-memory and may not handle extremely large documents efficiently
