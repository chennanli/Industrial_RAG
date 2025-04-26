import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import time
from pathlib import Path
import tempfile
import shutil
import datetime
import hashlib
import json

# Try to import OpenCV, but provide alternative if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
    print("OpenCV is available and will be used for video processing")
except ImportError:
    print("OpenCV (cv2) is not installed. Using PIL for image processing instead.")
    OPENCV_AVAILABLE = False
    from PIL import Image, ImageEnhance

# Import numpy (needed for frame generation)
import numpy as np

# Create necessary directories
FRAMES_DIR = Path("saved_frames")
FRAMES_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("frame_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Load model and processor
print("Loading model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("Model and processor loaded successfully!")

def preprocess_image(image_path):
    """Image preprocessing function"""
    if OPENCV_AVAILABLE:
        # Use OpenCV for preprocessing if available
        img = cv2.imread(image_path)
        if img is None:
            return False

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.imwrite(image_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        # Use PIL for basic preprocessing if OpenCV is not available
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure RGB mode
            
            # Apply some basic enhancement
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.2)  # Increase brightness
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # Increase sharpness
            
            img.save(image_path)
        except Exception as e:
            print(f"Error preprocessing image with PIL: {str(e)}")
            return False
    
    return True

def analyze_image(image, object_str):
    """Analyze image using Hugging Face Transformers"""
    if image is None:
        return "Please upload an image."
    
    prompt_str = f"""Please analyze the image and answer the following questions:
    1. Is there a {object_str} in the image?
    2. If yes, describe its appearance and location in the image in detail.
    3. If no, describe what you see in the image instead.
    4. On a scale of 1-10, how confident are you in your answer?

    Please structure your response as follows:
    Answer: [YES/NO]
    Description: [Your detailed description]
    Confidence: [1-10]"""
    
    try:
        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt_str},
                ],
            }
        ]
        
        # Prepare inference input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
        
        return response_text
    
    except Exception as e:
        return f"Error during processing: {str(e)}"

def get_video_hash(video_path, frame_interval, max_frames):
    """Generate a hash for the video based on its path and extraction parameters"""
    video_name = os.path.basename(video_path)
    # Create a unique identifier based on video path and extraction parameters
    hash_input = f"{video_path}_{frame_interval}_{max_frames}"
    
    # Add file modification time to the hash to detect changes to the video file
    try:
        mod_time = os.path.getmtime(video_path)
        hash_input += f"_{mod_time}"
        print(f"Including modification time in hash: {mod_time}")
    except Exception as e:
        print(f"Warning: Could not get modification time for {video_path}: {e}")
    
    video_hash = hashlib.md5(hash_input.encode()).hexdigest()
    print(f"Generated hash for video: {video_hash}")
    return video_hash

def check_frame_cache(video_path, frame_interval, max_frames):
    """Check if we have already extracted frames for this video with these parameters"""
    video_hash = get_video_hash(video_path, frame_interval, max_frames)
    cache_file = CACHE_DIR / f"{video_hash}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                session_dir = cache_data.get('session_dir')
                
                # Verify the session directory still exists
                if os.path.exists(session_dir):
                    print(f"Found cached frames at {session_dir}")
                    
                    # Get the frames from the cache
                    frames = []
                    missing_frames = 0
                    for frame_data in cache_data.get('frames', []):
                        second = frame_data[0]
                        frame_path = frame_data[1]
                        
                        # Verify the frame still exists
                        if os.path.exists(frame_path):
                            frames.append((second, frame_path))
                            print(f"Found cached frame: {frame_path}")
                        else:
                            missing_frames += 1
                            print(f"Warning: Cached frame does not exist: {frame_path}")
                    
                    print(f"Found {len(frames)} valid cached frames, {missing_frames} missing frames")
                    
                    # Only use cache if we found some frames
                    if frames:
                        print(f"Using cached frames from {session_dir}")
                        return frames, session_dir
                    else:
                        print(f"No valid frames found in cache, will extract new frames")
                else:
                    print(f"Cache session directory does not exist: {session_dir}")
        except Exception as e:
            print(f"Error reading cache: {str(e)}")
    
    return None, None

def save_to_frame_cache(video_path, frame_interval, max_frames, frames, session_dir):
    """Save frame extraction results to cache"""
    video_hash = get_video_hash(video_path, frame_interval, max_frames)
    cache_file = CACHE_DIR / f"{video_hash}.json"
    
    try:
        # Convert frames to serializable format
        serializable_frames = [(second, path) for second, path in frames]
        
        cache_data = {
            'video_path': video_path,
            'frame_interval': frame_interval,
            'max_frames': max_frames,
            'timestamp': datetime.datetime.now().isoformat(),
            'session_dir': session_dir,
            'frames': serializable_frames
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
            
        print(f"Saved frame extraction cache to {cache_file}")
    except Exception as e:
        print(f"Error saving cache: {str(e)}")

def extract_frames(video_path, output_dir, frame_interval=1, max_frames=None, progress=None):
    """Extract frames from video with configurable interval"""
    # Check if we have a cache hit
    cached_frames, cached_session_dir = check_frame_cache(video_path, frame_interval, max_frames)
    if cached_frames and cached_session_dir:
        print(f"Using {len(cached_frames)} cached frames from {cached_session_dir}")
        return cached_frames, cached_session_dir
    
    frames = []
    
    # Create a timestamp-based subfolder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.basename(video_path).split('.')[0]
    session_dir = os.path.join(output_dir, f"{video_name}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Created session directory: {session_dir}")
    
    if OPENCV_AVAILABLE:
        # Use OpenCV for frame extraction if available
        cap = cv2.VideoCapture(str(video_path))
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            # Create a placeholder frame
            import numpy as np  # Import numpy here to ensure it's available
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_path = os.path.join(session_dir, f"frame_0s.jpg")
            print(f"Creating placeholder frame at {frame_path}")
            cv2.imwrite(frame_path, dummy_img)
            frames.append((0, frame_path))
            
            # Save to cache even if there's an error
            save_to_frame_cache(video_path, frame_interval, max_frames, frames, session_dir)
            return frames, session_dir
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            print(f"Warning: Invalid FPS detected ({fps}), using default value of 25")
            fps = 25
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        extracted_count = 0
        
        # Calculate frames to skip based on interval
        frames_to_skip = fps * frame_interval
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # Extract frame at specified interval
                if frame_count % frames_to_skip == 0:
                    current_second = frame_count // fps
                    frame_path = os.path.join(session_dir, f"frame_{current_second}s.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append((current_second, frame_path))
                    extracted_count += 1
                    print(f"Extracted frame at {current_second}s: {frame_path}")
                    
                    # Update progress if provided
                    if progress is not None:
                        try:
                            progress((frame_count / total_frames), desc=f"Extracting frame at {current_second}s...")
                        except Exception as e:
                            print(f"Error updating progress: {str(e)}")
                    
                    # Stop if we've reached max_frames
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        except Exception as e:
            print(f"Error during OpenCV frame extraction: {str(e)}")
        finally:
            cap.release()
    else:
        # Use ffmpeg for frame extraction if OpenCV is not available
        try:
            import subprocess
            
            # Get video info using ffprobe
            cmd = [
                'ffprobe', 
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                fps_str = result.stdout.strip()
                
                # Check if fps_str is empty
                if not fps_str:
                    print("Error: ffprobe didn't return frame rate. Using default fps value.")
                    fps_str = "25"  # Use a default fps value
            except Exception as e:
                print(f"Error running ffprobe: {str(e)}. Using default fps value.")
                fps_str = "25"  # Use a default fps value
            
            # Parse fps (format is usually "num/den")
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den
            else:
                fps = float(fps_str)
            
            # Extract frames using ffmpeg with specified interval
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'fps=1/{frame_interval}',  # Extract 1 frame every frame_interval seconds
                '-q:v', '1',  # High quality
            ]
            
            # Add max frames limit if specified
            if max_frames:
                cmd.extend(['-frames:v', str(max_frames)])
                
            cmd.append(f'{session_dir}/frame_%04d.jpg')
            
            print(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg with output capture
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    print(f"ffmpeg error (code {result.returncode}):")
                    print(f"stderr: {result.stderr}")
                    print(f"stdout: {result.stdout}")
                    raise Exception(f"ffmpeg failed with code {result.returncode}")
                else:
                    print("ffmpeg completed successfully")
            except Exception as e:
                print(f"Exception running ffmpeg: {str(e)}")
                raise
            
            # Get list of extracted frames
            frame_files = sorted([f for f in os.listdir(session_dir) if f.startswith('frame_')])
            
            # Check if any frames were extracted
            if not frame_files:
                print("No frames were extracted by ffmpeg. Creating a placeholder frame.")
                # Create a placeholder frame
                from PIL import Image
                img = Image.new('RGB', (640, 480), color='black')
                img_path = os.path.join(session_dir, "frame_0s.jpg")
                img.save(img_path)
                frames.append((0, img_path))
                
                # Save to cache even if there's an error
                save_to_frame_cache(video_path, frame_interval, max_frames, frames, session_dir)
                return frames, session_dir
            
            # Process the extracted frames
            for i, frame in enumerate(frame_files):
                current_second = i * frame_interval
                old_path = os.path.join(session_dir, frame)
                new_path = os.path.join(session_dir, f"frame_{current_second}s.jpg")
                
                # Rename only if needed
                if old_path != new_path:
                    os.rename(old_path, new_path)
                
                frames.append((current_second, new_path))
                print(f"Renamed frame: {old_path} -> {new_path}")
                
                # Update progress if provided
                if progress is not None:
                    try:
                        progress((i + 1) / len(frame_files), desc=f"Processing frame at {current_second}s...")
                    except Exception as e:
                        print(f"Error updating progress: {str(e)}")
                
        except Exception as e:
            print(f"Error extracting frames with ffmpeg: {str(e)}")
            # Fallback to a very basic method if ffmpeg fails
            print("Attempting to use PIL for frame extraction (this will be slow)")
            
            try:
                from PIL import Image
                
                # Create a single frame as a placeholder
                img = Image.new('RGB', (640, 480), color='black')
                img_path = os.path.join(session_dir, "frame_0s.jpg")
                img.save(img_path)
                
                # Just add this single frame with a warning message
                frames.append((0, img_path))
                
            except Exception as e2:
                print(f"Error creating placeholder frame: {str(e2)}")
    
    print(f"Extracted {len(frames)} frames to {session_dir}")
    
    # Ensure we have at least one frame
    if not frames:
        print("No frames were extracted. Creating a placeholder frame.")
        try:
            # Try with OpenCV first
            if OPENCV_AVAILABLE:
                import numpy as np  # Import numpy here to ensure it's available
                dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
                frame_path = os.path.join(session_dir, "frame_0s.jpg")
                cv2.imwrite(frame_path, dummy_img)
            else:
                # Fall back to PIL
                from PIL import Image
                img = Image.new('RGB', (640, 480), color='black')
                frame_path = os.path.join(session_dir, "frame_0s.jpg")
                img.save(frame_path)
            
            frames.append((0, frame_path))
        except Exception as e:
            print(f"Error creating placeholder frame: {str(e)}")
    
    # Save the extracted frames to cache for future use
    print(f"Saving {len(frames)} frames to cache")
    save_to_frame_cache(video_path, frame_interval, max_frames, frames, session_dir)
    
    return frames, session_dir

# Create a simple cache for analysis results
analysis_cache = {}

def get_analysis_cache_key(frame_path, object_str):
    """Generate a cache key for analysis results"""
    return f"{frame_path}_{object_str}"

def process_video(video, object_str, frame_interval, max_frames, analyze_frames, progress=gr.Progress()):
    """Process video and analyze frames with configurable interval"""
    if video is None:
        return "Please upload a video.", None
    
    try:
        progress(0, desc="Starting video processing...")
        print(f"Video input: {video}")
        print(f"Type of video input: {type(video)}")
        print(f"Frame interval: {frame_interval}")
        print(f"Max frames: {max_frames}")
        print(f"Analyze frames: {analyze_frames}")
        
        # Extract frames with the specified interval
        frames, session_dir = extract_frames(
            video, 
            FRAMES_DIR, 
            frame_interval=frame_interval,
            max_frames=max_frames,
            progress=progress
        )
        
        if not frames:
            return "No frames could be extracted from the video.", session_dir
        
        # If analyze_frames is False, just return the extraction results
        if not analyze_frames:
            frame_paths = [f[1] for f in frames]
            return f"Extracted {len(frames)} frames to {session_dir}\n\nFrames are saved at: {session_dir}", session_dir
        
        # Analyze the extracted frames
        results = []
        consecutive_detections = 0
        first_detection_second = None
        
        # Force processing only a few frames if requested
        if len(frames) > max_frames:
            print(f"Limiting analysis to first {max_frames} frames")
            frames = frames[:max_frames]
        
        for i, (second, frame_path) in enumerate(frames):
            progress(0.5 + (i / len(frames)) * 0.5, desc=f"Analyzing frame at {second}s...")
            print(f"Processing frame {i+1}/{len(frames)}: {frame_path}")
            
            # Check if file exists
            if not os.path.exists(frame_path):
                print(f"Warning: Frame file does not exist: {frame_path}")
                continue
            
            # Check if we have a cached analysis for this frame and object
            cache_key = get_analysis_cache_key(frame_path, object_str)
            cached_response = analysis_cache.get(cache_key)
            
            if cached_response:
                print(f"Using cached analysis for {frame_path}")
                response = cached_response
            else:
                # Preprocess the frame
                if not preprocess_image(frame_path):
                    print(f"Warning: Could not preprocess frame: {frame_path}")
                    continue
                
                # Analyze the frame
                try:
                    response = analyze_image(frame_path, object_str)
                    print(f"Analysis response: {response[:100]}...")  # Print first 100 chars for debugging
                    
                    # Cache the response
                    analysis_cache[cache_key] = response
                    
                except Exception as e:
                    print(f"Error analyzing frame {frame_path}: {str(e)}")
                    results.append(f"Frame at {second}s: ERROR - {str(e)}\n\n")
                    continue
            
            # Parse the response
            is_match = False
            confidence = 0
            
            for line in response.strip().split('\n'):
                line = line.strip()
                if line.lower().startswith('answer:'):
                    answer = line.split(':', 1)[1].strip().upper()
                    is_match = answer == "YES"
                elif line.lower().startswith('confidence:'):
                    try:
                        confidence = int(line.split(':', 1)[1].strip())
                    except ValueError:
                        confidence = 0
            
            # Track consecutive detections
            if is_match and confidence >= 7:
                consecutive_detections += 1
                if consecutive_detections == 1:
                    first_detection_second = second
            else:
                consecutive_detections = 0
                
            # Add result
            results.append(f"Frame at {second}s: {'DETECTED' if is_match and confidence >= 7 else 'NOT DETECTED'} (Confidence: {confidence}/10)\n{response}\n\n")
            
            # Stop after 2 consecutive detections
            if consecutive_detections >= 2:
                results.append(f"\n\nObject detected consecutively, first detection at second {first_detection_second}")
                break
                
        
        if not results:
            return "No frames could be analyzed.", session_dir
            
        return "\n".join(results), session_dir
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error processing video: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return f"Error processing video: {str(e)}", None

def view_saved_frames(session_dir):
    """Return a gallery of saved frames"""
    if not session_dir or not os.path.exists(session_dir):
        return []
    
    frame_files = sorted([
        os.path.join(session_dir, f) for f in os.listdir(session_dir) 
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    
    print(f"Found {len(frame_files)} frames in {session_dir}")
    return frame_files

def clear_caches():
    """Clear both frame and analysis caches"""
    # Clear analysis cache in memory
    global analysis_cache
    old_size = len(analysis_cache)
    analysis_cache = {}
    
    # Clear frame cache files
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
        print(f"Found {len(cache_files)} cache files to delete")
        for cache_file in cache_files:
            cache_path = os.path.join(CACHE_DIR, cache_file)
            print(f"Removing cache file: {cache_path}")
            os.remove(cache_path)
        return f"Cleared {old_size} in-memory cache entries and {len(cache_files)} cache files"
    except Exception as e:
        print(f"Error clearing cache: {str(e)}")
        return f"Error clearing cache: {str(e)}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Qwen2.5-VL Video Analysis with Time Frame Control and Caching")
    
    with gr.Tab("Image Analysis"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Upload Image")
                image_object_input = gr.Textbox(
                    placeholder="Enter the object to detect (e.g., person, car, dog)", 
                    label="Object to Detect"
                )
                image_submit_btn = gr.Button("Analyze Image")
            
            with gr.Column():
                image_output = gr.Textbox(label="Analysis Result", lines=10)
        
        image_submit_btn.click(
            fn=analyze_image,
            inputs=[image_input, image_object_input],
            outputs=image_output
        )
    
    with gr.Tab("Video Analysis"):
        session_dir = gr.State(None)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                video_object_input = gr.Textbox(
                    placeholder="Enter the object to detect (e.g., person, car, dog)", 
                    label="Object to Detect",
                    value="person"
                )
                
                with gr.Row():
                    frame_interval = gr.Slider(
                        minimum=1, 
                        maximum=10, 
                        value=2, 
                        step=1, 
                        label="Frame Interval (seconds)", 
                        info="Extract one frame every X seconds"
                    )
                    max_frames = gr.Slider(
                        minimum=5, 
                        maximum=100, 
                        value=10, 
                        step=5, 
                        label="Maximum Frames", 
                        info="Maximum number of frames to extract and process"
                    )
                
                analyze_frames = gr.Checkbox(
                    label="Analyze Frames", 
                    value=True,
                    info="If unchecked, will only extract frames without analysis"
                )
                
                with gr.Row():
                    video_submit_btn = gr.Button("Process Video")
                    clear_cache_btn = gr.Button("Clear All Caches")
            
            with gr.Column():
                video_output = gr.Textbox(label="Analysis Results", lines=15)
                cache_status = gr.Textbox(label="Cache Status", lines=1)
                frames_gallery = gr.Gallery(label="Extracted Frames", show_label=True, columns=4, height="auto")
        
        video_submit_btn.click(
            fn=process_video,
            inputs=[video_input, video_object_input, frame_interval, max_frames, analyze_frames],
            outputs=[video_output, session_dir]
        )
        
        clear_cache_btn.click(
            fn=clear_caches,
            inputs=[],
            outputs=[cache_status]
        )
        
        # Update gallery when session_dir changes
        session_dir.change(
            fn=view_saved_frames,
            inputs=[session_dir],
            outputs=[frames_gallery]
        )

# Launch the application
if __name__ == "__main__":
    print("Starting Gradio interface with time frame control and caching...")
    demo.launch(share=False)  # Set share=False for local use only