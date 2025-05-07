# Documentation for Modular RAG System

This folder contains documentation files that provide useful background information on the RAG system and various technical aspects. These are kept for reference and troubleshooting purposes.

## Available Documentation

### RAG_concise_readme.md
The original README file for the system. This provides context on the implementation that was refactored into the current modular architecture.

### CLEANUP_GUIDE.md
Provides guidance on how to clean up and organize the project folders for a more streamlined repository structure.

### FOLDER_USAGE.md
Explains the purpose and usage of various directories in the system, including essential directories like `RAG_pdf/` and utility directories.

## About This System

The modular RAG system provides flexible document processing with the following features:

1. **Multiple Input Types**: Beyond PDFs, the system can process various document types for the knowledge base.

2. **Dual Modes**: Works as both a RAG system (with knowledge base context) and a pure chatbot (direct LLM responses).

3. **Model Flexibility**: Uses Hugging Face's transformer library with Qwen2.5-VL-7B model by default, but can be easily switched to use any model in LM Studio.

4. **Multimodal Support**: Handles text, image, and video inputs, with text being mandatory while media inputs are optional.

## Performance Considerations

Model response quality and speed highly depend on your hardware configuration. The system was developed and tested on a Mac Mini with 64GB RAM and an M4 chip with good performance results. If using different hardware, you may need to adjust batch sizes or other parameters in the code.
