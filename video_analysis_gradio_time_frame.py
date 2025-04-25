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

def extract_frames(video_path, output_dir, frame_interval=1, max_frames=None, progress=None):
    """Extract frames from video with configurable interval"""
    frames = []
    
    # Create a timestamp-based subfolder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.basename(video_path).split('.')[0]
    session_dir = os.path.join(output_dir, f"{video_name}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    if OPENCV_AVAILABLE:
        # Use OpenCV for frame extraction if available
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
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
                    
                    # Update progress if provided
                    if progress:
                        progress((frame_count / total_frames), desc=f"Extracting frame at {current_second}s...")
                    
                    # Stop if we've reached max_frames
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
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
                '-show_entries', 'stream=r_frame_rate,nb_frames', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            output_lines = result.stdout.strip().split('\n')
            fps_str = output_lines[0]
            
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
            
            subprocess.run(cmd, check=True)
            
            # Get list of extracted frames
            frame_files = sorted([f for f in os.listdir(session_dir) if f.startswith('frame_')])
            
            # Process the extracted frames
            for i, frame in enumerate(frame_files):
                current_second = i * frame_interval
                old_path = os.path.join(session_dir, frame)
                new_path = os.path.join(session_dir, f"frame_{current_second}s.jpg")
                
                # Rename only if needed
                if old_path != new_path:
                    os.rename(old_path, new_path)
                
                frames.append((current_second, new_path))
                
                # Update progress if provided
                if progress:
                    progress((i + 1) / len(frame_files), desc=f"Processing frame at {current_second}s...")
                
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
    return frames, session_dir

def process_video(video, object_str, frame_interval, max_frames, analyze_frames, progress=gr.Progress()):
    """Process video and analyze frames with configurable interval"""
    if video is None:
        return "Please upload a video.", None
    
    try:
        progress(0, desc="Starting video processing...")
        
        # Extract frames with the specified interval
        frames, session_dir = extract_frames(
            video, 
            FRAMES_DIR, 
            frame_interval=frame_interval,
            max_frames=max_frames,
            progress=progress
        )
        
        if not frames:
            return "No frames could be extracted from the video.", None
        
        # If analyze_frames is False, just return the extraction results
        if not analyze_frames:
            frame_paths = [f[1] for f in frames]
            return f"Extracted {len(frames)} frames to {session_dir}\n\nFrames are saved at: {session_dir}", session_dir
        
        # Analyze the extracted frames
        results = []
        consecutive_detections = 0
        first_detection_second = None
        
        for i, (second, frame_path) in enumerate(frames):
            progress(0.5 + (i / len(frames)) * 0.5, desc=f"Analyzing frame at {second}s...")
            
            # Preprocess the frame
            preprocess_image(frame_path)
            
            # Analyze the frame
            response = analyze_image(frame_path, object_str)
            
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
        
        return "\n".join(results), session_dir
    
    except Exception as e:
        return f"Error processing video: {str(e)}", None

def view_saved_frames(session_dir):
    """Return a gallery of saved frames"""
    if not session_dir or not os.path.exists(session_dir):
        return []
    
    frame_files = sorted([
        os.path.join(session_dir, f) for f in os.listdir(session_dir) 
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    
    return frame_files

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
                        value=20, 
                        step=5, 
                        label="Maximum Frames", 
                        info="Maximum number of frames to extract"
                    )
                
                analyze_frames = gr.Checkbox(
                    label="Analyze Frames", 
                    value=True,
                    info="If unchecked, will only extract frames without analysis"
                )
                
                video_submit_btn = gr.Button("Process Video")
            
            with gr.Column():
                video_output = gr.Textbox(label="Analysis Results", lines=15)
                frames_gallery = gr.Gallery(label="Extracted Frames", show_label=True, columns=4, height="auto")
        
        video_submit_btn.click(
            fn=process_video,
            inputs=[video_input, video_object_input, frame_interval, max_frames, analyze_frames],
            outputs=[video_output, session_dir]
        )
        
        # Update gallery when session_dir changes
        session_dir.change(
            fn=view_saved_frames,
            inputs=[session_dir],
            outputs=[frames_gallery]
        )

# Launch the application
if __name__ == "__main__":
    print("Starting Gradio interface with time frame control...")
    demo.launch(share=False)  # Set share=False for local use only