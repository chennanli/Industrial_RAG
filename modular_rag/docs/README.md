# Documentation for Modular RAG System

This folder contains documentation files that provide useful background information on the original RAG system and various technical aspects. These are kept for reference and troubleshooting purposes.

## Available Documentation

### RAG_concise_readme.md
The original README file for the `Concise_RAG_v5.py` system. This provides context on the original implementation that was refactored into the current modular architecture.

### OFFLOAD_FOLDER_EXPLANATION.md
Explains the memory management techniques used in the system, particularly how model weights are offloaded to reduce memory usage during inference.

### IMAGE_UPLOAD_FIX_GUIDE.md
Contains troubleshooting guidance for image upload issues that may occur in the RAG interface. Useful for debugging if image processing problems occur.

### LM_STUDIO_FIX_GUIDE.md
Documentation on LM Studio integration, including tips for connecting to local LLM servers and handling vision models in LM Studio.

## Relationship to Modular System

The modular RAG system incorporates all the fixes and features described in these documents:

1. The enhanced memory management described in OFFLOAD_FOLDER_EXPLANATION.md is integrated into the `models/huggingface.py` module.

2. The image upload fixes from IMAGE_UPLOAD_FIX_GUIDE.md are integrated into the Gradio interface in the `ui/components.py` file.

3. The LM Studio integration described in LM_STUDIO_FIX_GUIDE.md is fully implemented in the `models/lm_studio.py` module, including the vision model support.

These documents are maintained for historical context and troubleshooting guidance, even though their content has been incorporated into the modular architecture.
