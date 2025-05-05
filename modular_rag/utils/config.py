"""
Configuration settings for the RAG system
"""
import os

# Path configurations
PDF_FOLDER = "RAG_pdf"
FRAME_CACHE_DIR = "frame_cache"

# Ensure folders exist
for folder in [PDF_FOLDER, FRAME_CACHE_DIR]:
    os.makedirs(folder, exist_ok=True)

# Model configuration dictionary - supports multiple model sources
MODEL_CONFIG = {
    "source": "HuggingFace",  # Options: "HuggingFace", "LMStudio"
    "huggingface_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "lm_studio_url": "http://localhost:1234/v1",
    "lm_studio_model": None,  # Will be populated dynamically
    "use_mps": True,  # Enable Metal Performance Shaders for M1/M2 Macs if available
    "use_fp16": True  # Use half-precision (float16) for faster inference
}
