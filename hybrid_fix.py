import re

def create_hybrid_mode():
    """Creates a hybrid mode where image analysis uses HF but text generation uses LM Studio"""
    
    file_path = "/Users/chennanli/Desktop/LLM_Project/Qwen2.5/combined_tabbed.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Change the update_model_source_selection function to allow hybrid mode
    # Find the function
    pattern = r'def update_model_source_selection\(choice\):.*?return \('
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        old_func = match.group(0)
        
        # Create new function with hybrid support
        new_func = old_func.replace(
            'vision_interactive = False # Disable vision inputs',
            'vision_interactive = True # Enable vision inputs for hybrid mode'
        )
        
        # Update status message
        new_func = new_func.replace(
            'vision_interactive = False # Disable vision inputs',
            'vision_interactive = True # Keep vision inputs enabled\n             status_update += " Hybrid mode: Image analysis uses HuggingFace, text generation uses LM Studio."'
        )
        
        content = content.replace(old_func, new_func)
    
    # Modify the image processing functions to handle hybrid mode
    # Find generate_response function and add hybrid logic
    pattern = r'def generate_response\(messages, max_tokens=1024\):.*?else:'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        old_func = match.group(0)
        
        # Add hybrid mode check at the beginning
        new_func = old_func.replace(
            'source = MODEL_CONFIG["source"]',
            '''source = MODEL_CONFIG["source"]
    
    # Check if this is a multimodal request that needs hybrid processing
    has_images = False
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") in ["image", "video"]:
                    has_images = True
                    break
    
    # If LM Studio is selected but we have images, use HF for this request
    if source == "LMStudio" and has_images:
        print("Hybrid mode: Using HuggingFace for multimodal request")
        source = "HuggingFace"'''
        )
        
        content = content.replace(old_func, new_func)
    
    # Create backup
    backup_path = file_path + '.hybrid.backup'
    with open(backup_path, 'w') as f:
        with open(file_path, 'r') as f2:
            f.write(f2.read())
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Enabled hybrid mode for LM Studio + Image/Video processing")
    print(f"📝 Created backup at {backup_path}")
    print("\nChanges made:")
    print("1. Image/video upload fields remain enabled when LM Studio is selected")
    print("2. Image analysis will automatically use HuggingFace when images are uploaded")
    print("3. Text generation will use LM Studio as selected")
    print("4. This creates a best-of-both-worlds hybrid mode")

if __name__ == "__main__":
    create_hybrid_mode()
