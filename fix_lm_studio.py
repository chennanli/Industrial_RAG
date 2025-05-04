#!/usr/bin/env python3
"""
Simple script to fix LM Studio connection issues by updating the configuration
"""

import json
import os
from pathlib import Path

# Path to your combined_tabbed.py file
SCRIPT_PATH = "/Users/chennanli/Desktop/LLM_Project/Qwen2.5/combined_tabbed.py"

# Default LM Studio URL (you can change this)
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"

def check_lm_studio_connection(url):
    """Check if LM Studio is accessible at the given URL"""
    try:
        import requests
        response = requests.get(f"{url}/models", timeout=5)
        if response.status_code == 200:
            print(f"✅ LM Studio is accessible at {url}")
            # Try to get list of models
            models = response.json()
            model_ids = [m.get('id', 'unknown') for m in models.get('data', [])]
            if model_ids:
                print(f"📋 Available models: {', '.join(model_ids)}")
            else:
                print("⚠️ No models found in LM Studio")
            return True
        else:
            print(f"❌ LM Studio returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio at {url}: {e}")
        return False

def update_lm_studio_url(url):
    """Update the LM Studio URL in the combined_tabbed.py file"""
    try:
        with open(SCRIPT_PATH, 'r') as f:
            content = f.read()
        
        # Replace the hardcoded URL
        updated_content = content.replace(
            '"lm_studio_url": "http://localhost:1234/v1"',
            f'"lm_studio_url": "{url}"'
        )
        
        # Create a backup
        backup_path = f"{SCRIPT_PATH}.backup"
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"📝 Created backup at {backup_path}")
        
        # Write the updated content
        with open(SCRIPT_PATH, 'w') as f:
            f.write(updated_content)
        print(f"✅ Updated LM Studio URL to {url}")
        return True
    except Exception as e:
        print(f"❌ Failed to update file: {e}")
        return False

def main():
    print("=== LM Studio Connection Fixer ===\n")
    
    # Check default URL
    print("🔍 Checking default LM Studio URL...")
    if check_lm_studio_connection(DEFAULT_LM_STUDIO_URL):
        print("\n✨ LM Studio is already accessible at the default URL!")
        print("To use a different URL, run this script again with your URL as argument.")
        return
    
    print("\n⚠️ LM Studio is not accessible at the default URL.")
    print("Please ensure that:")
    print("  1. LM Studio is running")
    print("  2. A model is loaded in LM Studio")
    print("  3. LM Studio server is enabled (check the LM Studio interface)")
    print("  4. LM Studio is using the correct port (usually 1234)")
    print()
    
    # Common solutions
    print("Common solutions:")
    print("  💡 Open LM Studio and click the 'Start Server' button")
    print("  💡 Check if LM Studio is running on a different port")
    print("  💡 Try accessing http://localhost:1234/v1/models in your browser")
    print()
    
    # Check if LM Studio is on a different port
    print("Checking common alternative ports...")
    for port in [1234, 8080, 3000, 5000, 8000]:
        url = f"http://localhost:{port}/v1"
        if check_lm_studio_connection(url):
            print(f"\n🎉 Found LM Studio running on port {port}!")
            response = input(f"Would you like to update the configuration to use this URL? (y/n): ")
            if response.lower() == 'y':
                if update_lm_studio_url(url):
                    print("\n✅ Configuration updated successfully!")
                    print("Please restart your application for the changes to take effect.")
            return
    
    print("\n❌ Could not find LM Studio on any common port.")
    print("\nIf you know the correct URL, you can update it manually:")
    print(f"1. Open {SCRIPT_PATH}")
    print('2. Find the line: "lm_studio_url": "http://localhost:1234/v1"')
    print("3. Replace with your LM Studio URL")
    print()
    print("You can also run this script with your URL as an argument:")
    print("  python fix_lm_studio.py http://your-lm-studio-url")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # URL provided as argument
        custom_url = sys.argv[1]
        print(f"🔄 Testing custom URL: {custom_url}")
        if check_lm_studio_connection(custom_url):
            update_lm_studio_url(custom_url)
        else:
            print(f"❌ Cannot connect to LM Studio at {custom_url}")
    else:
        main()
