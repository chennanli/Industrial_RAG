# LM Studio Integration Guide

## Overview

This guide explains how to connect the RAG system with LM Studio for using local LLMs instead of Hugging Face models. LM Studio provides a way to run local LLM servers that you can integrate with this system.

## Setting Up LM Studio

1. **Download and Install LM Studio**
   - Get it from [LM Studio's website](https://lmstudio.ai/)
   - Install and launch the application

2. **Load a Compatible Model**
   - Recommended: Qwen2.5-7B, Gemma-3, Phi-3 or any model with similar capabilities
   - For multimodal support: use Qwen2.5-VL-7B-Instruct or a similar vision-capable model

3. **Start the Local Server**
   - Click on the "Local Server" tab in LM Studio
   - Click "Start Server" (usually runs on http://localhost:1234/v1)
   - Make sure a model is loaded before starting the server

## Using LM Studio with the Application

1. **Launch the RAG Application**
   ```bash
   python launch_rag.py
   ```

2. **Switch to LM Studio**
   - In the Model Source dropdown, select "LM Studio"
   - The application will automatically connect to the local server
   - Select the specific model from the dropdown (if multiple are available)

3. **Features with LM Studio**
   - Text queries: Fully supported
   - Image/video analysis: Requires a vision-capable model in LM Studio

## Multimodal Support with LM Studio

By default, when switching to LM Studio, image and video uploads may be disabled. To use multimodal features:

1. Make sure you have a vision-capable model loaded in LM Studio
2. The model must support the OpenAI vision API format
3. Recommended models: Qwen2.5-VL-7B-Instruct or compatible vision models

## Troubleshooting

### Connection Issues

If you see "LM Studio connection error" when switching:

1. **Check if LM Studio is running**
   - Make sure LM Studio application is open
   - Verify the server is started (clicked "Start Server")
   - Check a model is loaded

2. **Verify the Server URL**
   - Default: http://localhost:1234/v1
   - If using a different port, check the LM Studio interface

3. **Test the Connection**
   - Try accessing http://localhost:1234/v1/models in your browser
   - You should see a JSON response with model information

### Performance Considerations

- LM Studio performance depends on your local hardware
- For best results with larger models (7B+), use a system with at least 16GB RAM
- GPU acceleration is highly recommended

## Advanced Configuration

You can modify the LM Studio connection settings in the application:
- The default LM Studio URL is http://localhost:1234/v1
- Adjust batch sizes for video processing based on your hardware capabilities

## Switching Between Models

The application supports easy switching between Hugging Face and LM Studio:
- Switch to "HuggingFace" for cloud-based models
- Switch to "LM Studio" for locally running models
- No restart required when switching between modes
