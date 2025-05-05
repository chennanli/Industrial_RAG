# Modular RAG System

A modular, optimized Retrieval-Augmented Generation (RAG) system that supports multimodal queries with text, images, and videos. This project refactors the original `Concise_RAG_v5.py` into a structured, maintainable codebase.

## Key Features

- **Modular Architecture**: Clean separation of concerns with specialized modules
- **Multimodal Support**: Process text, image, and video queries with a unified interface
- **LM Studio Integration**: Dropdown selection of locally running LLM models through LM Studio
- **Hardware Acceleration**: Optimized for Apple M-series chips (MPS) and CUDA GPUs
- **Video Processing Optimizations**: Frame caching, parallel frame extraction
- **High-Contrast UI**: Improved source document contrast for better readability
- **Vector Store Backends**: FAISS with LangChain (primary) or TF-IDF (fallback)

## Required Files

The project requires only the following files and folders. You can safely remove any other files not listed here:

```
LLM_Project/Qwen2.5/
├── launch_rag.py              # Launch script for the application
├── qwen25_env/                # Python virtual environment (keep this)
├── RAG_pdf/                   # Folder for your PDF documents to be indexed
│   └── (your PDF files)       # Place your PDFs here
├── frame_cache/               # Cache for video frames (created automatically)
├── modular_rag/               # Main code directory
│   ├── __init__.py            # Package initialization
│   ├── app.py                 # Main application entry point
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py        # Package initialization
│   │   ├── config.py          # Configuration settings
│   │   └── helpers.py         # Helper functions
│   ├── models/                # Model backends
│   │   ├── __init__.py        # Package initialization
│   │   ├── huggingface.py     # Hugging Face model integration
│   │   └── lm_studio.py       # LM Studio integration
│   ├── rag_modules/           # RAG processing modules
│   │   ├── __init__.py        # Package initialization
│   │   ├── vector_store.py    # Vector store for document indexing
│   │   └── rag_processor.py   # RAG processing for text, image, video
│   └── ui/                    # User interface
│       ├── __init__.py        # Package initialization
│       ├── styles.py          # CSS styling definitions
│       └── components.py      # UI component definitions
```

### Documentation Files for GitHub

Consider keeping these documentation files as they provide useful background information:

- `RAG_concise_readme.md` - Original README with background information on the RAG system
- `OFFLOAD_FOLDER_EXPLANATION.md` - Explains memory management techniques
- `IMAGE_UPLOAD_FIX_GUIDE.md` - Contains troubleshooting for image upload issues
- `LM_STUDIO_FIX_GUIDE.md` - Contains guidance on LM Studio integration issues

You might want to place these in a `docs/` folder to keep your repository organized.



### Important Folders to Keep

Always maintain these folders:
- `qwen25_env/` - Contains your Python environment
- `RAG_pdf/` - Contains your PDF documents for RAG
- `frame_cache/` - Contains cached video frames (improves performance)

## Usage

1. Activate the Python environment:
   ```
   # On macOS/Linux
   source ../qwen25_env/bin/activate
   
   # On Windows
   ..\qwen25_env\Scripts\activate
   ```

2. Place PDF documents in the `RAG_pdf` folder in the working directory
3. Run the application:
   ```
   python app.py
   ```
4. Initialize the knowledge base with the "Process PDF Knowledge Base" button
5. Select model source (Hugging Face or LM Studio)
6. Submit queries with optional image or video

## Model Support

### Hugging Face
- Default: `Qwen/Qwen2.5-VL-7B-Instruct`
- Optimized for multimodal processing (text, image, video)

### LM Studio
- Connects to locally running LM Studio server (http://localhost:1234/v1)
- Dynamically fetches and displays available models
- Text-only processing (image/video descriptions processed by Hugging Face model)

## Performance Optimizations

- **Hardware Detection**: Automatically uses the best available hardware (MPS, CUDA, CPU)
- **Mixed Precision**: FP16 for GPU/MPS for faster processing
- **Video Frame Caching**: Avoids redundant processing of previously analyzed videos
- **Parallel Frame Extraction**: Multi-threaded video frame processing
- **Memory Management**: Efficient cleanup of temporary resources

## UI Improvements

- High-contrast source document display (dark background, light text)
- Model switching without restarting
- Clean visual design with responsive layout
- Proper error handling and status updates
