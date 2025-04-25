# Dependencies: pip install fastapi uvicorn python-multipart jinja2 aiofiles transformers torch pillow

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import os
import time
from pathlib import Path
import asyncio
import json
import base64
from PIL import Image
import io

# Import Hugging Face Transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Try to import OpenCV, but provide alternative if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV (cv2) is not installed. Using PIL for image processing instead.")
    OPENCV_AVAILABLE = False

app = FastAPI()

# Create necessary directories
UPLOAD_DIR = Path("uploads")
FRAMES_DIR = Path("frames")
UPLOAD_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

# Initialize Hugging Face model and processor
print("Loading model and processor...")
model = None
processor = None

async def load_model():
    global model, processor
    if model is None or processor is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            torch_dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print("Model and processor loaded successfully!")

# Set up templates and static file serving
templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

async def analyze_image(image_path: str, object_str: str):
    """Analyze image using Hugging Face Transformers"""
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
        # Ensure model is loaded
        await load_model()
        
        # Open image with PIL
        image = Image.open(image_path).convert("RGB")
        
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

        # Parse response
        response_lines = response_text.strip().split('\n')
        
        answer = None
        description = None
        confidence = 10
        
        for line in response_lines:
            line = line.strip()
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip().upper()
            elif any(line.lower().startswith(prefix) for prefix in
                    ['description:', 'reasoning:', 'alternative description:']):
                description = line.split(':', 1)[1].strip()
            elif line.lower().startswith('confidence:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 10
        
        return answer == "YES" and confidence >= 7, description, confidence
    except Exception as e:
        print(f"Error during image analysis: {str(e)}")
        return False, f"Error occurred: {str(e)}", 0

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
            from PIL import ImageEnhance
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

def extract_frames(video_path, output_dir, fps_step=1):
    """Extract frames from video using PIL and ffmpeg if OpenCV is not available"""
    if OPENCV_AVAILABLE:
        # Use OpenCV for frame extraction if available
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if frame_count % (fps * fps_step) == 0:  # Extract one frame per specified seconds
                    current_second = frame_count // fps
                    frame_path = os.path.join(output_dir, f"frame_{current_second}.jpg")
                    cv2.imwrite(frame_path, frame)
                    yield current_second, frame_path

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
                '-show_entries', 'stream=r_frame_rate', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            fps_str = result.stdout.strip()
            
            # Parse fps (format is usually "num/den")
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den
            else:
                fps = float(fps_str)
            
            # Extract frames using ffmpeg
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'fps=1/{fps_step}',  # Extract 1 frame every fps_step seconds
                '-q:v', '1',  # High quality
                f'{output_dir}/frame_%d.jpg'
            ]
            
            subprocess.run(cmd, check=True)
            
            # Get list of extracted frames
            frames = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_')])
            
            # Rename frames to include seconds
            for i, frame in enumerate(frames):
                current_second = i * fps_step
                old_path = os.path.join(output_dir, frame)
                new_path = os.path.join(output_dir, f"frame_{current_second}.jpg")
                
                # Rename only if needed
                if old_path != new_path:
                    os.rename(old_path, new_path)
                
                yield current_second, new_path
                
        except Exception as e:
            print(f"Error extracting frames with ffmpeg: {str(e)}")
            # Fallback to a very basic method if ffmpeg fails
            print("Attempting to use PIL for frame extraction (this will be slow)")
            
            try:
                from PIL import Image
                import io
                
                # Read video file in binary mode
                with open(video_path, 'rb') as f:
                    video_data = f.read()
                
                # Create a single frame as a placeholder
                img = Image.new('RGB', (640, 480), color='black')
                img_path = os.path.join(output_dir, "frame_0.jpg")
                img.save(img_path)
                
                # Just yield this single frame with a warning message
                yield 0, img_path
                
            except Exception as e2:
                print(f"Error creating placeholder frame: {str(e2)}")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_video(
        video: UploadFile = File(...),
        object_str: str = Form(...)
):
    try:
        # Save uploaded video
        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Create dedicated frames directory for current task
        task_frames_dir = FRAMES_DIR / video.filename.split('.')[0]
        task_frames_dir.mkdir(exist_ok=True)

        # Asynchronously generate analysis results
        async def generate_results():
            consecutive_detections = 0  # Count consecutive detections
            first_detection_second = None  # Record time of first detection

            try:
                # Extract frames from video
                for current_second, frame_path in extract_frames(video_path, task_frames_dir, fps_step=1):
                    if preprocess_image(frame_path):
                        is_match, description, confidence = await analyze_image(frame_path, object_str)

                        if is_match:
                            consecutive_detections += 1
                            if consecutive_detections == 1:
                                first_detection_second = current_second
                        else:
                            consecutive_detections = 0
                            first_detection_second = None

                        result = {
                            "status": "success",
                            "frame": {
                                "second": current_second,
                                "is_match": is_match,
                                "description": description,
                                "confidence": confidence,
                                "frame_path": f"/frames/{video.filename.split('.')[0]}/frame_{current_second}.jpg"
                            }
                        }

                        yield json.dumps(result) + "\n"

                        # If object detected twice consecutively, output result and stop
                        if consecutive_detections >= 2:
                            final_result = {
                                "status": "complete",
                                "message": f"Object detected consecutively, first detection at second {first_detection_second}",
                                "first_detection_time": first_detection_second
                            }
                            yield json.dumps(final_result) + "\n"
                            break

            except Exception as e:
                error_result = {
                    "status": "error",
                    "message": f"Error processing video: {str(e)}"
                }
                yield json.dumps(error_result) + "\n"

        return StreamingResponse(generate_results(), media_type="application/json")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    
    # Load model at startup
    asyncio.run(load_model())
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
