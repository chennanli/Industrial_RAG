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
import numpy as np
import re

# Try to import OpenCV, but provide alternative if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
    print("OpenCV is available and will be used for video processing")
except ImportError:
    print("OpenCV (cv2) is not installed. Using PIL for image processing instead.")
    OPENCV_AVAILABLE = False
    from PIL import Image, ImageEnhance

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
def calculate_video_content_hash(video_path):
    """Calculate a hash based on video content (first few KB) and metadata"""
    try:
        # Get file size and modification time
        file_size = os.path.getsize(video_path)
        mod_time = os.path.getmtime(video_path)
        
        # Read first 1MB of the file for content hash
        with open(video_path, 'rb') as f:
            content = f.read(1024 * 1024)  # Read first 1MB
        
        # Create a hash combining content and metadata
        video_name = os.path.basename(video_path)
        hash_input = f"{video_name}_{file_size}_{mod_time}_{hashlib.md5(content).hexdigest()}"
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        print(f"Generated content hash for video {video_name}: {content_hash}")
        return content_hash
    except Exception as e:
        print(f"Error calculating video content hash: {e}")
        # Fallback to just the filename if we can't calculate a proper hash
        return hashlib.md5(os.path.basename(video_path).encode()).hexdigest()

def get_video_hash(video_path, frame_interval, max_frames):
    """Generate a hash for the video based on its content and extraction parameters"""
    # Get content-based hash instead of path-based hash
    content_hash = calculate_video_content_hash(video_path)
    
    # Create a unique identifier based on content hash and extraction parameters
    hash_input = f"{content_hash}_{frame_interval}_{max_frames}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def save_session_metadata(session_dir, video_path, frame_interval, max_frames, frames):
    """Save metadata about the session for future reference"""
    try:
        metadata = {
            'video_name': os.path.basename(video_path),
            'content_hash': calculate_video_content_hash(video_path),
            'frame_interval': frame_interval,
            'max_frames': max_frames,
            'timestamp': datetime.datetime.now().isoformat(),
            'frame_count': len(frames),
            'frames': [(second, os.path.basename(path)) for second, path in frames]
        }
        
        metadata_file = os.path.join(session_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved session metadata to {metadata_file}")
    except Exception as e:
        print(f"Error saving session metadata: {str(e)}")

def load_session_metadata(session_dir):
    """Load metadata about a session"""
    metadata_file = os.path.join(session_dir, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session metadata: {str(e)}")
    return None

def get_all_frame_sessions():
    """Get all available frame sessions with their metadata"""
    sessions = []
    
    if not os.path.exists(FRAMES_DIR):
        return sessions
    
    for session_name in os.listdir(FRAMES_DIR):
        session_dir = os.path.join(FRAMES_DIR, session_name)
        if os.path.isdir(session_dir):
            metadata = load_session_metadata(session_dir)
            if metadata:
                # Add the session directory to the metadata
                metadata['session_dir'] = session_dir
                sessions.append(metadata)
            else:
                # Create basic metadata if none exists
                frame_files = [f for f in os.listdir(session_dir) if f.endswith('.jpg') or f.endswith('.png')]
                if frame_files:
                    # Try to extract info from directory name
                    video_name = session_name.split('_')[0] if '_' in session_name else "Unknown"
                    timestamp = '_'.join(session_name.split('_')[1:]) if '_' in session_name else "Unknown"
                    
                    # Try to extract frame interval from filenames
                    frame_seconds = []
                    for f in frame_files:
                        match = re.search(r'frame_(\d+)s\.', f)
                        if match:
                            frame_seconds.append(int(match.group(1)))
                    
                    frame_interval = 0
                    if len(frame_seconds) >= 2:
                        frame_seconds.sort()
                        intervals = [frame_seconds[i+1] - frame_seconds[i] for i in range(len(frame_seconds)-1)]
                        if intervals:
                            frame_interval = min(intervals)
                    
                    sessions.append({
                        'session_dir': session_dir,
                        'video_name': video_name,
                        'timestamp': timestamp,
                        'frame_interval': frame_interval,
                        'max_frames': len(frame_files),
                        'frame_count': len(frame_files),
                        'frames': [(0, f) for f in frame_files]  # Placeholder
                    })
    
    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return sessions

def format_session_info(session):
    """Format session info for display"""
    video_name = session.get('video_name', 'Unknown')
    timestamp = session.get('timestamp', 'Unknown')
    frame_interval = session.get('frame_interval', 0)
    frame_count = session.get('frame_count', 0)
    max_frames = session.get('max_frames', 0)
    
    # Format timestamp for display
    if isinstance(timestamp, str) and 'T' in timestamp:
        # ISO format
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
    
    return f"{video_name} | Frames: {frame_count} | Interval: {frame_interval}s | Max: {max_frames} | {timestamp}"

def get_session_choices():
    """Get choices for session dropdown"""
    sessions = get_all_frame_sessions()
    return [format_session_info(session) for session in sessions], sessions
def extract_frames(video_path, output_dir, frame_interval=1, max_frames=None, progress=None):
    """Extract frames from video with configurable interval"""
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
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_path = os.path.join(session_dir, f"frame_0s.jpg")
            print(f"Creating placeholder frame at {frame_path}")
            cv2.imwrite(frame_path, dummy_img)
            frames.append((0, frame_path))
            
            # Save session metadata
            save_session_metadata(session_dir, video_path, frame_interval, max_frames, frames)
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
                
                # Save session metadata
                save_session_metadata(session_dir, video_path, frame_interval, max_frames, frames)
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
    
    # Save session metadata
    save_session_metadata(session_dir, video_path, frame_interval, max_frames, frames)
    
    return frames, session_dir
# Create a simple LRU cache for analysis results (max 100 entries)
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            # Move to the end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing entry
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # Add new entry
            if len(self.order) >= self.capacity:
                # Remove least recently used
                oldest = self.order.pop(0)
                del self.cache[oldest]
            self.cache[key] = value
            self.order.append(key)
    
    def clear(self):
        self.cache = {}
        self.order = []
    
    def __len__(self):
        return len(self.cache)

# Initialize the LRU cache for analysis results
analysis_cache = LRUCache(100)

def get_analysis_cache_key(frame_path, object_str):
    """Generate a cache key for analysis results"""
    return f"{frame_path}_{object_str}"

def get_frames_from_session(session_dir):
    """Get frames from an existing session directory"""
    if not os.path.exists(session_dir):
        return []
    
    frames = []
    frame_files = sorted([f for f in os.listdir(session_dir)
                         if (f.endswith('.jpg') or f.endswith('.png')) and f.startswith('frame_')])
    
    for frame_file in frame_files:
        # Extract second from filename (format: frame_Xs.jpg)
        match = re.search(r'frame_(\d+)s\.', frame_file)
        if match:
            second = int(match.group(1))
            frame_path = os.path.join(session_dir, frame_file)
            frames.append((second, frame_path))
    
    # Sort by second
    frames.sort(key=lambda x: x[0])
    print(f"Found {len(frames)} frames in session, with timestamps: {[f[0] for f in frames[:10]]}...")
    return frames

def process_video(video, object_str, frame_interval, max_frames, analyze_frames, progress=gr.Progress()):
    """Process video and analyze frames with configurable interval"""
    if video is None:
        return "Please upload a video.", None, "No video processed"
    
    try:
        start_time = time.time()  # Initialize start_time at the beginning
        progress(0, desc="Starting video processing...")
        print(f"Video input: {video}")
        print(f"Type of video input: {type(video)}")
        print(f"Frame interval: {frame_interval}")
        print(f"Max frames: {max_frames}")
        print(f"Analyze frames: {analyze_frames}")
        print(f"PROCESSING NEW VIDEO - THIS WILL TAKE TIME")
        
        # Extract frames with the specified interval
        frames, session_dir = extract_frames(
            video, 
            FRAMES_DIR, 
            frame_interval=frame_interval,
            max_frames=max_frames,
            progress=progress
        )
        
        if not frames:
            return "No frames could be extracted from the video.", session_dir, "No frames extracted"
        
        # If analyze_frames is False, just return the extraction results
        if not analyze_frames:
            frame_paths = [f[1] for f in frames]
            return f"Extracted {len(frames)} frames to {session_dir}\n\nFrames are saved at: {session_dir}", session_dir, f"Extracted {len(frames)} frames without analysis"
        
        # Analyze the extracted frames
        results = []
        detections = []
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
                    analysis_cache.put(cache_key, response)
                    
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
            
            # Track all detections with confidence info
            if is_match:
                detections.append((second, confidence))
                
            # Track consecutive high-confidence detections
            if is_match and confidence >= 7:
                consecutive_detections += 1
                if consecutive_detections == 1:
                    first_detection_second = second
            else:
                consecutive_detections = 0
                
            # Add result - show DETECTED if answer is YES, regardless of confidence
            # But note the confidence score separately
            detection_status = "DETECTED" if is_match else "NOT DETECTED"
            high_confidence = confidence >= 7
            confidence_note = f"(High Confidence: {confidence}/10)" if high_confidence else f"(Low Confidence: {confidence}/10)"
            
            results.append(f"Frame at {second}s: {detection_status} {confidence_note}\n{response}\n\n")
            
            # Stop after 2 consecutive detections
            if consecutive_detections >= 2:
                results.append(f"\n\nObject detected consecutively, first detection at second {first_detection_second}")
                break
        
        if not results:
            return "No frames could be analyzed.", session_dir, "No frames could be analyzed"
        
        # Create a summary of the analysis
        summary = ""
        if detections:
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x[1], reverse=True)
            top_detections = detections[:3]  # Top 3 detections
            
            detection_strs = [f"{object_str} at {second}s (Confidence: {confidence}/10)"
                             for second, confidence in top_detections]
            
            end_time = time.time()
            processing_time = end_time - start_time
            summary = f"ANSWER: YES\n"
            summary += f"CONFIDENCE: High confidence detections found\n"
            summary += f"PROCESSING TIME: {processing_time:.2f}s (NEW VIDEO PROCESSING)"
        else:
            end_time = time.time()
            processing_time = end_time - start_time
            summary = f"ANSWER: NO\n"
            summary += f"CONFIDENCE: No high confidence detections found\n"
            summary += f"PROCESSING TIME: {processing_time:.2f}s (NEW VIDEO PROCESSING)"
            
        return "\n".join(results), session_dir, summary
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error processing video: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return f"Error processing video: {str(e)}", None, f"Error: {str(e)}"
def use_existing_session(session_choice, object_str, analyze_frames, progress=gr.Progress()):
    """Use frames from an existing session"""
    if not session_choice:
        return "Please select a session.", None, "No session selected"
    
    try:
        start_time = time.time()
        # Get all sessions
        _, sessions = get_session_choices()
        
        # Find the selected session
        selected_index = -1
        for i, session_info in enumerate(sessions):
            if format_session_info(session_info) == session_choice:
                selected_index = i
                break
        
        if selected_index == -1:
            return "Session not found.", None, "Session not found"
        
        session = sessions[selected_index]
        session_dir = session['session_dir']
        
        progress(0.2, desc="Loading frames from session...")
        print(f"Using existing session: {session_dir} (NO VIDEO PROCESSING NEEDED)")
        
        # Get frames from the session
        frames = get_frames_from_session(session_dir)
        
        if not frames:
            return f"No frames found in session: {session_dir}", session_dir, "No frames found"
        
        print(f"Found {len(frames)} frames in session")
        
        # If analyze_frames is False, just return the frames
        if not analyze_frames:
            return f"Loaded {len(frames)} frames from {session_dir}", session_dir, f"Loaded {len(frames)} frames without analysis"
        
        # Analyze the frames
        results = []
        detections = []
        consecutive_detections = 0
        first_detection_second = None
        
        for i, (second, frame_path) in enumerate(frames):
            progress(0.3 + (i / len(frames)) * 0.7, desc=f"Analyzing frame at {second}s...")
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
                    analysis_cache.put(cache_key, response)
                    
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
            
            # Track all detections with confidence info
            if is_match:
                detections.append((second, confidence))
                
            # Track consecutive high-confidence detections
            if is_match and confidence >= 7:
                consecutive_detections += 1
                if consecutive_detections == 1:
                    first_detection_second = second
            else:
                consecutive_detections = 0
                
            # Add result - show DETECTED if answer is YES, regardless of confidence
            # But note the confidence score separately
            detection_status = "DETECTED" if is_match else "NOT DETECTED"
            high_confidence = confidence >= 7
            confidence_note = f"(High Confidence: {confidence}/10)" if high_confidence else f"(Low Confidence: {confidence}/10)"
            
            results.append(f"Frame at {second}s: {detection_status} {confidence_note}\n{response}\n\n")
            
            # Stop after 2 consecutive detections
            if consecutive_detections >= 2:
                results.append(f"\n\nObject detected consecutively, first detection at second {first_detection_second}")
                break
        
        if not results:
            return "No frames could be analyzed.", session_dir, "No frames could be analyzed"
        
        # Create a summary of the analysis
        summary = ""
        if detections:
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x[1], reverse=True)
            top_detections = detections[:3]  # Top 3 detections
            
            detection_strs = [f"{object_str} at {second}s (Confidence: {confidence}/10)"
                             for second, confidence in top_detections]
            
            end_time = time.time()
            processing_time = end_time - start_time
            summary = f"ANSWER: YES\n"
            summary += f"CONFIDENCE: High confidence detections found\n"
            summary += f"PROCESSING TIME: {processing_time:.2f}s (USING EXISTING FRAMES)"
        else:
            end_time = time.time()
            processing_time = end_time - start_time
            summary = f"ANSWER: NO\n"
            summary += f"CONFIDENCE: No high confidence detections found\n"
            summary += f"PROCESSING TIME: {processing_time:.2f}s (USING EXISTING FRAMES)"
            
        return "\n".join(results), session_dir, summary
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error using existing session: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return f"Error using existing session: {str(e)}", None, f"Error: {str(e)}"

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

def delete_session(session_dir):
    """Delete a session directory"""
    if not session_dir or not os.path.exists(session_dir):
        return f"Session directory does not exist: {session_dir}"
    
    try:
        shutil.rmtree(session_dir)
        return f"Deleted session directory: {session_dir}"
    except Exception as e:
        return f"Error deleting session directory: {str(e)}"

def refresh_session_list():
    """Refresh the session list"""
    choices, _ = get_session_choices()
    # For older versions of Gradio that don't support Dropdown.update
    return choices
# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Qwen2.5-VL Video Analysis with Time Frame Control")
    
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
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Process New Video")
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
                    
                    video_submit_btn = gr.Button("Process Video", variant="primary")
                
                with gr.Group():
                    gr.Markdown("### Use Existing Frames")
                    session_choices, _ = get_session_choices()
                    session_dropdown = gr.Dropdown(
                        choices=session_choices,
                        label="Select Saved Frame Session",
                        info="Choose a previously extracted frame set"
                    )
                    existing_object_input = gr.Textbox(
                        placeholder="Enter the object to detect (e.g., person, car, dog)", 
                        label="Object to Detect",
                        value="person"
                    )
                    existing_analyze_frames = gr.Checkbox(
                        label="Analyze Frames", 
                        value=True,
                        info="If unchecked, will only load frames without analysis"
                    )
                    
                    with gr.Row():
                        use_session_btn = gr.Button("Use Selected Session", variant="primary")
                        refresh_btn = gr.Button("Refresh Session List")
                        delete_session_btn = gr.Button("Delete Selected Session", variant="stop")
            
            with gr.Column(scale=2):
                with gr.Row():
                    video_output = gr.Textbox(label="Analysis Results", lines=15)
                
                with gr.Row():
                    analysis_summary = gr.Textbox(label="Analysis Summary", lines=3)
                
                frames_gallery = gr.Gallery(label="Extracted Frames", show_label=True, columns=4, height="auto")
        
        # Process new video
        video_submit_btn.click(
            fn=process_video,
            inputs=[video_input, video_object_input, frame_interval, max_frames, analyze_frames],
            outputs=[video_output, session_dir, analysis_summary]
        )
        
        # Use existing session
        use_session_btn.click(
            fn=use_existing_session,
            inputs=[session_dropdown, existing_object_input, existing_analyze_frames],
            outputs=[video_output, session_dir, analysis_summary]
        )
        
        # Refresh session list
        refresh_btn.click(
            fn=refresh_session_list,
            inputs=[],
            outputs=[session_dropdown]
        ).then(
            # After refreshing, clear the selection
            fn=lambda: None,
            inputs=[],
            outputs=[session_dropdown]
        )
        
        # Delete session
        delete_session_btn.click(
            fn=lambda choice: delete_session(sessions[session_choices.index(choice)]['session_dir']) if choice and choice in session_choices else "No session selected",
            inputs=[session_dropdown],
            outputs=[analysis_summary]
        ).then(
            fn=refresh_session_list,
            inputs=[],
            outputs=[session_dropdown]
        ).then(
            # After refreshing, clear the selection
            fn=lambda: None,
            inputs=[],
            outputs=[session_dropdown]
        )
        
        # Update gallery when session_dir changes
        session_dir.change(
            fn=view_saved_frames,
            inputs=[session_dir],
            outputs=[frames_gallery]
        )

# Launch the application
if __name__ == "__main__":
    # Clean old sessions on startup
    print("Starting Gradio interface with time frame control and session management...")
    demo.launch(share=False)  # Set share=False for local use only