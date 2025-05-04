def simple_enable_fields():
    """Simply re-enables image/video fields for LM Studio (without handling the errors)"""
    
    file_path = "/Users/chennanli/Desktop/LLM_Project/Qwen2.5/combined_tabbed.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace the vision_interactive setting
    content = content.replace(
        'vision_interactive = False # Disable vision inputs',
        'vision_interactive = True # Re-enable vision inputs'
    )
    
    # Remove the status message about disabling
    content = content.replace(
        'status_update += " Note: Image/video uploads are disabled for LM Studio (text-only model)."',
        'status_update += " Warning: LM Studio does not support image/video processing."'
    )
    
    # Create backup
    backup_path = file_path + '.simple.backup'
    with open(backup_path, 'w') as f:
        with open(file_path, 'r') as f2:
            f.write(f2.read())
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Re-enabled image/video fields for LM Studio")
    print(f"📝 Created backup at {backup_path}")
    print("\n⚠️ WARNING:")
    print("Image/video analysis will fail when using LM Studio")
    print("You'll need to handle errors if users try to upload media")

if __name__ == "__main__":
    simple_enable_fields()
