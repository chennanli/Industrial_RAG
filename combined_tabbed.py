import os
import gradio as gr
import torch
import time
import tempfile
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import re
import traceback
import shutil
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import openai # Added for LM Studio interaction
import base64  # Added for image encoding

# ... (keep all your existing imports and initial setup code) ...

# --- Model Configuration ---
MODEL_CONFIG = {
    "source": "HuggingFace", # Default: "HuggingFace" or "LMStudio"
    "lm_studio_url": "http://localhost:1234/v1", # Default LM Studio URL
    "lm_studio_model": "gemma-3-12b-it", # Default model name based on available models
    "hf_model_name": "Qwen/Qwen2.5-VL-7B-Instruct"
}

# ... (keep all your global variables and initialization code) ...

# --- Model Initialization Function ---
def initialize_model():
    """Loads the appropriate model based on MODEL_CONFIG."""
    # ... (keep your existing initialization code unchanged) ...

# --- Response Generation Abstraction ---
def generate_response(messages, max_tokens=1024):
    """Generates a response using the currently configured model source."""
    source = MODEL_CONFIG["source"]
    print(f"Generating response using source: {source}") # Debug print

    if source == "HuggingFace":
        # ... (keep your existing HuggingFace logic unchanged) ...
            
    elif source == "LMStudio":
        if not openai_client:
            print("Error: LM Studio client not initialized.")
            return "Error: LM Studio client not initialized. Please check configuration and logs."
        
        try:
            # --- LM Studio Logic ---
            # Convert HF message format to OpenAI format
            openai_formatted_messages = []
            has_multimodal = False
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                
                if isinstance(content, list): # Multimodal format
                    openai_content = []
                    for item in content:
                        if item.get("type") == "text":
                            openai_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image":
                            has_multimodal = True
                            # Handle image data - convert to base64 if it's a file path
                            image_data = item.get("image")
                            if isinstance(image_data, str) and os.path.exists(image_data):
                                # Read image file and convert to base64
                                with open(image_data, 'rb') as img_file:
                                    image_bytes = img_file.read()
                                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                                # Determine image format
                                image_format = image_data.split('.')[-1]
                                if image_format == 'jpg':
                                    image_format = 'jpeg'
                                image_url = f"data:image/{image_format};base64,{base64_image}"
                            else:
                                image_url = image_data
                            
                            # Use OpenAI format for images
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            })
                        elif item.get("type") == "video":
                            has_multimodal = True
                            print(f"Warning: Video content found. LM Studio may not support video processing.")
                            # For videos, we'll append a text message instead
                            openai_content.append({
                                "type": "text",
                                "text": f"[Video content detected - LM Studio may not support video processing]"
                            })
                    
                    openai_formatted_messages.append({'role': role, 'content': openai_content})

                elif isinstance(content, str): # Simple text format
                    openai_formatted_messages.append({'role': role, 'content': content})
                else:
                    print(f"Warning: Skipping message with unexpected content format for role '{role}': {type(content)}")

            if not openai_formatted_messages:
                print("Error: No valid messages formatted for LM Studio.")
                return "Error: No valid messages found to send to LM Studio."

            print(f"DEBUG LM Studio Gen: Sending {len(openai_formatted_messages)} messages to model '{MODEL_CONFIG['lm_studio_model']}'. Has multimodal: {has_multimodal}")
            response = openai_client.chat.completions.create(
                model=MODEL_CONFIG["lm_studio_model"],
                messages=openai_formatted_messages,
                temperature=0.7,
                max_tokens=max_tokens, # OpenAI uses max_tokens for the completion length
            )
            # Check if usage information is available
            usage_info = response.usage if hasattr(response, 'usage') else 'Not available'
            print(f"DEBUG LM Studio Gen: Received response. Usage: {usage_info}")
            
            # Check if choices are available and non-empty
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                print("Error: LM Studio response did not contain expected choices.")
                return "Error: Received an empty or invalid response from LM Studio."
            # --- End LM Studio Logic ---
            
        except Exception as e:
            print(f"Error calling LM Studio API: {e}")
            print(traceback.format_exc())
            return f"Error communicating with LM Studio: {e}"
            
    else:
        print(f"Error: Invalid model source configured: {source}")
        return "Error: Invalid model source configured."

# --- Update the model selection handler to not disable vision inputs ---
def update_model_source_selection(choice):
    """Handles model source switching, initializes, and updates UI visibility/choices."""
    global hf_model, hf_processor, openai_client # Allow modification of globals
    print(f"Switching model source to: {choice}")
    MODEL_CONFIG["source"] = choice
    
    # Reset models/clients before initializing the new one
    hf_model = None
    hf_processor = None
    openai_client = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear GPU memory if applicable

    # Call the initialization function (which sets up hf_model/openai_client)
    # This needs to happen *before* trying to fetch models if LMStudio is chosen
    initialize_model()
    
    status_update = f"Model source set to {choice}."

    # Default UI updates (keep vision inputs enabled for both sources)
    vision_interactive = True  # Always enable vision inputs
    lm_dropdown_visible = False
    lm_dropdown_choices = []
    lm_dropdown_value = None # Reset dropdown value

    # Specific updates if LM Studio is chosen
    if choice == "LMStudio":
        print("LM Studio selected. Keeping vision inputs enabled - LM Studio will attempt to process images if the model supports it.")
        lm_dropdown_visible = True # Show LM Studio dropdown
        
        # Ensure client is initialized before fetching models
        if not openai_client:
            status_update += " Error: Failed to initialize LM Studio client. Check URL and server status."
            print("Error: openai_client is None after initialization attempt.")
        else:
            # Fetch models and update dropdown choices
            lm_dropdown_choices = get_lm_studio_models()
            if not lm_dropdown_choices:
                status_update += " Warning: Could not fetch models from LM Studio API. Is a model loaded?"
                print("Warning: get_lm_studio_models() returned empty list.")
            else:
                status_update += f" Found {len(lm_dropdown_choices)} model(s) in LM Studio."
                # Set default selection if possible
                current_lm_model_config = MODEL_CONFIG.get('lm_studio_model', None)
                if current_lm_model_config in lm_dropdown_choices:
                    lm_dropdown_value = current_lm_model_config
                elif lm_dropdown_choices: # Fallback to first available if config value not found
                    lm_dropdown_value = lm_dropdown_choices[0]
                    MODEL_CONFIG['lm_studio_model'] = lm_dropdown_value # Update config with fallback
                    print(f"LM Studio model reset to first available: {lm_dropdown_value}")
                    status_update += f" Defaulting to '{lm_dropdown_value}'."
                # If lm_dropdown_value is still None here, it means no models were found
    else:
        # For HuggingFace, make sure vision inputs are still enabled
        pass

    # Return updates for all relevant UI components
    return (
        status_update,
        gr.update(interactive=vision_interactive, value=None), # rag_image_input (always enabled)
        gr.update(interactive=vision_interactive, value=None), # rag_video_input (always enabled)
        gr.update(visible=lm_dropdown_visible, choices=lm_dropdown_choices, value=lm_dropdown_value) # lm_studio_model_dropdown
    )

# ... (keep all your other functions exactly as they are) ...

# The rest of the file should remain exactly the same, including all the RAG functions,
# image processing functions, video processing functions, UI definition, and event handlers.
