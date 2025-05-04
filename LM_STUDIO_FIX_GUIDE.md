# LM Studio Connection Fix Guide

## Problem: "LM Studio connection error" when switching to LM Studio

This happens because LM Studio is not running or not accessible at the expected URL (http://localhost:1234/v1).

## Quick Solution:

### Option 1: Start LM Studio (Recommended)
1. Open LM Studio application
2. Click on the **"Local Server"** tab
3. Click the **"Start Server"** button
4. Make sure a model is loaded
5. Restart your application and try switching to LM Studio again

### Option 2: Diagnose the issue using the fix script
1. Open Terminal
2. Navigate to the project directory:
   ```bash
   cd /Users/chennanli/Desktop/LLM_Project/Qwen2.5
   ```
3. Run the diagnostic script:
   ```bash
   python fix_lm_studio.py
   ```
4. Follow the on-screen instructions

### Option 3: Check if LM Studio is running on a different port
1. Open LM Studio and check what port it's using (usually 1234)
2. If it's using a different port, run:
   ```bash
   python fix_lm_studio.py http://localhost:YOUR_PORT/v1
   ```
3. Restart your application

### Option 4: Manual fix
1. Open `combined_tabbed.py` in a text editor
2. Find this line:
   ```python
   "lm_studio_url": "http://localhost:1234/v1",
   ```
3. Replace with the correct URL (e.g., if LM Studio is on port 8080):
   ```python
   "lm_studio_url": "http://localhost:8080/v1",
   ```
4. Save the file
5. Restart your application

## Common LM Studio Issues:

1. **Server not started**: Make sure to click "Start Server" in the LM Studio interface
2. **No model loaded**: Load a model in LM Studio before starting the server
3. **Different port**: LM Studio might be using a different port than 1234
4. **API access not enabled**: In newer versions of LM Studio, you may need to enable API access

## Verify LM Studio is working:
Test the URL in your browser:
```
http://localhost:1234/v1/models
```
You should see a JSON response with model information.

## If nothing works:
1. Make sure LM Studio is properly installed
2. Check LM Studio logs for errors
3. Try reinstalling LM Studio
4. Use the HuggingFace option as an alternative

## Need more help?
The fix_lm_studio.py script will help diagnose the exact issue and provide specific solutions.
