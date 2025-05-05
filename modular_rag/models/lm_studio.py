"""
LM Studio integration module for connecting to locally running LM Studio servers
"""
import os
import openai
import base64
from PIL import Image
import io
from modular_rag.utils.config import MODEL_CONFIG

def get_lm_studio_models():
    """Get list of available models from LM Studio"""
    lm_studio_url = MODEL_CONFIG["lm_studio_url"]
    try:
        client = openai.OpenAI(base_url=lm_studio_url, api_key="not-needed")
        models_list = client.models.list()
        # Extract model IDs
        model_names = [model.id for model in models_list.data]
        print(f"Found LM Studio models: {model_names}")
        return model_names
    except Exception as e:
        print(f"Error fetching models from LM Studio: {e}")
        return [f"Error: Could not connect to LM Studio ({e})"]

def get_current_lm_studio_model():
    """Get the currently loaded model in LM Studio"""
    lm_studio_url = MODEL_CONFIG["lm_studio_url"]
    try:
        client = openai.OpenAI(base_url=lm_studio_url, api_key="not-needed")
        models_list = client.models.list()
        if models_list.data:
            # Assuming the first model listed is the "current" one
            return models_list.data[0].id
        else:
            return None  # No models loaded
    except Exception as e:
        print(f"Error inferring current LM Studio model: {e}")
        return None

def encode_image_to_base64(image_path):
    """Encode image to base64 for sending to the API"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def process_image_query(text, image_path, max_tokens=4096, temperature=0.7):
    """Process a query with an image using LM Studio"""
    lm_studio_url = MODEL_CONFIG["lm_studio_url"]
    model_name = MODEL_CONFIG["lm_studio_model"]
    
    if not model_name:
        model_name = get_current_lm_studio_model()
        if not model_name:
            return "Error: No model currently loaded in LM Studio", False
    
    # Check if image exists
    if not os.path.exists(image_path):
        return f"Error: Image file not found: {image_path}", False
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "Error: Failed to encode image", False
        
        client = openai.OpenAI(base_url=lm_studio_url, api_key="not-needed")
        
        # Create content with image in OpenAI format
        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
        
        # Make the vision request
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Get the generated text
        content = response.choices[0].message.content.strip()
        
        # Simple heuristic to check if response might be truncated
        is_truncated = False
        
        return content, is_truncated
    except Exception as e:
        print(f"Error processing image with LM Studio: {e}")
        return f"Error processing image with LM Studio: {str(e)}", False

def query_lm_studio(prompt, max_tokens=4096, temperature=0.7):
    """Send a text-only query to LM Studio and get the response"""
    lm_studio_url = MODEL_CONFIG["lm_studio_url"]
    model_name = MODEL_CONFIG["lm_studio_model"]
    
    if not model_name:
        model_name = get_current_lm_studio_model()
        if not model_name:
            return "Error: No model currently loaded in LM Studio", False
        
    try:
        client = openai.OpenAI(base_url=lm_studio_url, api_key="not-needed")
        
        # Make the completion request
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Get the generated text
        content = response.choices[0].message.content.strip()
        
        # Simple heuristic to check if response might be truncated
        is_truncated = False
        
        # Return the generated text and truncation flag
        return content, is_truncated
    except Exception as e:
        print(f"Error querying LM Studio: {e}")
        return f"Error querying LM Studio: {str(e)}", False
