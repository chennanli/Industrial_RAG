"""
Helper functions for the RAG system
"""
import hashlib
import torch
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
import cv2

def get_device():
    """Get the appropriate device for tensor operations"""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_optimal_dtype(device):
    """Get optimal data type based on device"""
    return torch.float16 if device != "cpu" else "auto"

def split_text_into_chunks(text, source_filename, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks with source tracking."""
    chunks = []
    sources = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        sources.append(source_filename)  # Track the source filename for each chunk
        start = end - overlap
    return chunks, sources

def extract_video_frames(video_path, cache_dir, num_frames=5):
    """Extract frames from video with multithreading and caching"""
    # Check if we have cached frames
    cached_frames = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.jpg')]
    frames = []
    
    if cached_frames:
        print(f"Using {len(cached_frames)} cached frames for video")
        # Sort frames by name to maintain order
        cached_frames.sort()
        # Get timestamp from filename (assuming format frame_timestamp.jpg)
        frames = [(float(os.path.basename(f).split('_')[1].split('.')[0]), f) for f in cached_frames]
    else:
        print("Extracting new frames from video")
        # Extract key frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        def extract_frame(pos):
            cap = cv2.VideoCapture(video_path)  # Open a new capture for each thread
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            cap.release()
            if ret:
                time_pos = pos/fps
                frame_path = os.path.join(cache_dir, f"frame_{time_pos}.jpg")
                cv2.imwrite(frame_path, frame)
                return (time_pos, frame_path)
            return None
        
        # Extract evenly spaced frames
        frame_positions = [int(i * frame_count / num_frames) for i in range(num_frames)]
        
        # Use thread pool to extract frames in parallel
        with ThreadPoolExecutor(max_workers=num_frames) as executor:
            results = list(executor.map(extract_frame, frame_positions))
        
        # Filter out None results and store valid frames
        frames = [f for f in results if f is not None]
        
    return frames

def get_video_cache_dir(video_path, base_cache_dir):
    """Generate cache directory path for a specific video"""
    video_hash = hashlib.md5(video_path.encode()).hexdigest()
    cache_dir = os.path.join(base_cache_dir, video_hash)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def clean_temp_files(temp_dir):
    """Safely clean up temporary files"""
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as cleanup_e:
        print(f"Warning: Error during cleanup: {cleanup_e}")

def format_source_info(sources):
    """Format source information for display"""
    if isinstance(sources, list) and sources:
        # Sources are already formatted as "- filename.pdf"
        formatted_sources = "### Source Documents:\n\n" + "\n".join(sources)
    else:
        formatted_sources = "No specific source documents found."
    return formatted_sources

def print_debug(message, category=None):
    """Print debug messages with category prefix"""
    prefix = f"DEBUG - {category}: " if category else "DEBUG: "
    print(f"{prefix}{message}")

def handle_exception(e, operation=None):
    """Standard exception handler with detailed output"""
    error_trace = traceback.format_exc()
    operation_text = f" {operation}" if operation else ""
    print(f"Error{operation_text}: {str(e)}")
    print(f"Error trace: {error_trace}")
    return f"Error{operation_text}: {str(e)}"
