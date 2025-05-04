#!/usr/bin/env python3
"""
Enable image/video support for LM Studio in your application
This script updates the code to properly handle vision with LM Studio
"""

import re

def enable_lm_studio_vision():
    """Update the code to support vision through LM Studio API"""
    
    file_path = "/Users/chennanli/Desktop/LLM_Project/Qwen2.5/combined_tabbed.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Don't disable vision inputs for LM Studio
    content = content.replace(
        'vision_interactive = False # Disable vision inputs',
        'vision_interactive = True # Keep vision inputs enabled'
    )
    
    # 2. Update the generate_response function to properly handle images for LM Studio
    # Find the LM Studio error handling section
    pattern = r'if has_multimodal:.*?return.*?Error.*?not supported.*?"'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        old_code = match.group(0)
        
        # Replace with proper image handling
        new_code = '''if has_multimodal:
                        # Convert image to OpenAI API format
                        openai_formatted_messages = []
                        for msg in messages:
                            role = msg.get("role")
                            content = msg.get("content")
                            
                            if isinstance(content, list):
                                openai_content = []
                                for item in content:
                                    if item.get("type") == "text":
                                        openai_content.append({"type": "text", "text": item.get("text", "")})
                                    elif item.get("type") == "image":
                                        # Convert image to base64 if needed
                                        image_data = item.get("image")
                                        if isinstance(image_data, str) and image_data.startswith('/'):
                                            # Read image file and convert to base64
                                            import base64
                                            with open(image_data, 'rb') as img_file:
                                                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                                            image_url = f"data:image/jpeg;base64,{base64_image}"
                                        else:
                                            image_url = image_data
                                        
                                        openai_content.append({
                                            "type": "image_url",
                                            "image_url": {"url": image_url}
                                        })
                                
                                openai_formatted_messages.append({'role': role, 'content': openai_content})
                            elif isinstance(content, str):
                                openai_formatted_messages.append({'role': role, 'content': content})
                    else:
                        # No multimodal content, process normally'''
        
        content = content.replace(old_code, new_code)
    
    # 3. Update the status message
    content = content.replace(
        'Switching model source to: {choice}',
        'Switching model source to: {choice} (Supports vision if model has vision capabilities)'
    )
    
    # Create backup
    backup_path = file_path + '.vision_enabled.backup'
    with open(backup_path, 'w') as f:
        with open(file_path, 'r') as f2:
            f.write(f2.read())
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Enabled vision support for LM Studio")
    print(f"📝 Created backup at {backup_path}")
    print("\nChanges made:")
    print("1. Vision inputs remain enabled when LM Studio is selected")
    print("2. Images are properly converted to OpenAI API format")
    print("3. The system will attempt to process images through LM Studio")
    print("\n⚠️ Note: This will only work if:")
    print("- Your LM Studio version supports vision API")
    print("- You have loaded a vision-capable model")
    print("- The model supports the OpenAI vision API format")

if __name__ == "__main__":
    enable_lm_studio_vision()
