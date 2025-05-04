#!/usr/bin/env python3
"""
Test if your LM Studio installation supports image processing
"""

import openai
import base64
import json
from PIL import Image
import io
import requests

def create_test_image():
    """Create a simple test image"""
    # Create a simple image with text
    img = Image.new('RGB', (200, 100), color='red')
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Convert to base64
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
    return base64_image

def test_lm_studio_vision():
    """Test if LM Studio API supports image processing"""
    
    # Default LM Studio URL
    lm_studio_url = "http://localhost:1234/v1"
    
    try:
        # Check if LM Studio is running
        print("Checking if LM Studio is running...")
        response = requests.get(f"{lm_studio_url}/models")
        if response.status_code != 200:
            print(f"❌ LM Studio is not accessible at {lm_studio_url}")
            return False
        
        # Get available models
        models = response.json()['data']
        if not models:
            print("❌ No models loaded in LM Studio")
            return False
        
        model_id = models[0]['id']
        print(f"✅ Using model: {model_id}")
        
        # Initialize OpenAI client for LM Studio
        client = openai.OpenAI(
            base_url=lm_studio_url,
            api_key="lm-studio"
        )
        
        # Create test image
        base64_image = create_test_image()
        
        # Test 1: Try with image_url format (standard OpenAI format)
        print("\nTest 1: Testing with image_url format...")
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this image?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            print("✅ Success with image_url format!")
            print(f"Response: {response.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"❌ Failed with image_url format: {e}")
        
        # Test 2: Try with simple image object format
        print("\nTest 2: Testing with simple image format...")
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this image?"},
                            {"type": "image", "image": base64_image}
                        ]
                    }
                ],
                max_tokens=100
            )
            print("✅ Success with simple image format!")
            print(f"Response: {response.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"❌ Failed with simple image format: {e}")
        
        # Test 3: Check API capabilities
        print("\nTest 3: Checking API capabilities...")
        try:
            # Some APIs expose capabilities
            response = requests.get(f"{lm_studio_url}/capabilities")
            if response.status_code == 200:
                capabilities = response.json()
                print(f"API capabilities: {capabilities}")
            else:
                print("API capabilities endpoint not available")
        except:
            pass
        
        return False
        
    except Exception as e:
        print(f"❌ Error testing LM Studio vision: {e}")
        return False

def main():
    print("=== LM Studio Vision Support Test ===\n")
    
    if test_lm_studio_vision():
        print("\n🎉 Your LM Studio supports image processing!")
        print("Recommendation: Update your code to enable image support")
    else:
        print("\n❌ Your LM Studio doesn't support image processing")
        print("\nPossible reasons:")
        print("1. You're using an older version of LM Studio")
        print("2. The loaded model doesn't support vision")
        print("3. The API implementation needs specific configuration")
        print("\nRecommendations:")
        print("- Update LM Studio to the latest version")
        print("- Load a vision-capable model (e.g., Qwen2-VL)")
        print("- Check LM Studio documentation for vision API support")

if __name__ == "__main__":
    main()
