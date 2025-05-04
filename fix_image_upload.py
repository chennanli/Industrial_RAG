import re

def add_helper_text():
    """Adds helper text to clarify image/video limitations when using LM Studio"""
    
    file_path = "/Users/chennanli/Desktop/LLM_Project/Qwen2.5/combined_tabbed.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the RAG image and video input declarations
    content = content.replace(
        'rag_image_input = gr.Image(label="Upload Image (Optional)", type="filepath")',
        'rag_image_input = gr.Image(label="Upload Image (Optional)", type="filepath")'
    )
    
    content = content.replace(
        'rag_video_input = gr.Video(label="Upload Video (Optional)")',
        'rag_video_input = gr.Video(label="Upload Video (Optional)")'
    )
    
    # Add a status message for when LM Studio is selected
    # Find the status update and add a clearer message
    content = content.replace(
        'vision_interactive = False # Disable vision inputs',
        'vision_interactive = False # Disable vision inputs\n             status_update += " Note: Image/video uploads are disabled for LM Studio (text-only model)."'
    )
    
    # Add visual feedback by changing placeholders when disabled
    content = content.replace(
        'gr.update(interactive=vision_interactive, value=None), # rag_image_input',
        'gr.update(interactive=vision_interactive, value=None, placeholder="Image upload disabled for LM Studio"), # rag_image_input'
    )
    
    content = content.replace(
        'gr.update(interactive=vision_interactive, value=None), # rag_video_input',
        'gr.update(interactive=vision_interactive, value=None, placeholder="Video upload disabled for LM Studio"), # rag_video_input'
    )
    
    # Create backup
    backup_path = file_path + '.imagefix.backup'
    with open(backup_path, 'w') as f:
        with open(file_path, 'r') as f2:
            f.write(f2.read())
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated combined_tabbed.py to make LM Studio limitations clearer")
    print(f"📝 Created backup at {backup_path}")
    print("\nChanges made:")
    print("1. Added status message explaining image/video are disabled for LM Studio")
    print("2. Added placeholders to show why upload fields are disabled")
    print("3. The fields will remain disabled when LM Studio is selected (as intended)")

if __name__ == "__main__":
    add_helper_text()
