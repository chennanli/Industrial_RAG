"""
Video processing utilities for frame extraction and caching
"""
import os
import cv2
import tempfile
import datetime
import hashlib
import json
import re
from pathlib import Path

# Directory for frame caching
SAVED_FRAMES_DIR = Path("saved_frames")
SAVED_FRAMES_DIR.mkdir(exist_ok=True)

def calculate_video_content_hash(video_path):
    """Calculate a hash based on video content and metadata"""
    try:
        # Get file size and modification time
        file_size = os.path.getsize(video_path)
        mod_time = os.path.getmtime(video_path)
        
        # Read first 1MB of the file for content hash
        with open(video_path, 'rb') as f:
            content = f.read(1024 * 1024)
        
        # Create a hash combining content and metadata
        video_name = os.path.basename(video_path)
        hash_input = f"{video_name}_{file_size}_{mod_time}_{hashlib.md5(content).hexdigest()}"
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        print(f"Generated content hash for video {video_name}: {content_hash}")
        return content_hash
    except Exception as e:
        print(f"Error calculating video content hash: {e}")
        return hashlib.md5(os.path.basename(video_path).encode()).hexdigest()

def get_video_hash(video_path, frame_interval, max_frames):
    """Generate a hash for the video based on its content and extraction parameters"""
    content_hash = calculate_video_content_hash(video_path)
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
    
    if not os.path.exists(SAVED_FRAMES_DIR):
        return sessions
    
    for session_name in os.listdir(SAVED_FRAMES_DIR):
        session_dir = os.path.join(SAVED_FRAMES_DIR, session_name)
        if os.path.isdir(session_dir):
            metadata = load_session_metadata(session_dir)
            if metadata:
                # Add the session directory to the metadata
                metadata['session_dir'] = session_dir
                sessions.append(metadata)
            else:
                # Create basic metadata if none exists
                frame_files = [f for f in os.listdir(session_dir) 
                              if f.endswith('.jpg') or f.endswith('.png')]
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
                        intervals = [frame_seconds[i+1] - frame_seconds[i] 
                                    for i in range(len(frame_seconds)-1)]
                        if intervals:
                            frame_interval = min(intervals)
                    
                    sessions.append({
                        'session_dir': session_dir,
                        'video_name': video_name,
                        'timestamp': timestamp,
                        'frame_interval': frame_interval,
                        'max_frames': len(frame_files),
                        'frame_count': len(frame_files),
                        'frames': [(0, f) for f in frame_files]
                    })
    
    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return sessions

def get_session_dir_for_video(video_path, frame_interval, max_frames):
    """Check if frames for this video with these parameters are already cached"""
    sessions = get_all_frame_sessions()
    content_hash = calculate_video_content_hash(video_path)
    
    for session in sessions:
        if (session.get('content_hash') == content_hash and
            session.get('frame_interval') == frame_interval and
            session.get('max_frames') == max_frames):
            return session.get('session_dir')
    
    return None

def extract_frames(video_path, frame_interval=1, max_frames=10, progress=None):
    """Extract frames from video with configurable interval
    
    Args:
        video_path: Path to the video file
        frame_interval: Extract one frame every X seconds
        max_frames: Maximum number of frames to extract
        progress: Optional progress callback function
        
    Returns:
        tuple: (extracted_frames, session_dir)
            extracted_frames: List of tuples (second, frame_path)
            session_dir: Directory where frames are saved
    """
    frames = []
    
    # First check if we already have frames for this video with these parameters
    existing_session = get_session_dir_for_video(video_path, frame_interval, max_frames)
    if existing_session:
        print(f"Using cached frames from {existing_session}")
        # Load frames from existing session
        frames = get_frames_from_session(existing_session)
        if frames:
            return frames, existing_session
    
    # Create a timestamp-based subfolder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.basename(video_path).split('.')[0]
    session_dir = os.path.join(SAVED_FRAMES_DIR, f"{video_name}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Created session directory: {session_dir}")
    
    # Use OpenCV for frame extraction
    cap = cv2.VideoCapture(str(video_path))
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        # Create a placeholder frame
        dummy_img = 255 * cv2.UMat(cv2.eye(480, 640, cv2.CV_8UC3))  # White image
        frame_path = os.path.join(session_dir, f"frame_0s.jpg")
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
                
                # Update progress if provided
                if progress is not None:
                    try:
                        # Progress as percentage (0-1)
                        progress((frame_count / total_frames if total_frames > 0 else 0), 
                                desc=f"Extracting frame at {current_second}s...")
                    except Exception as e:
                        print(f"Error updating progress: {str(e)}")
                
                # Stop if we've reached max_frames
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
    except Exception as e:
        print(f"Error during frame extraction: {str(e)}")
    finally:
        cap.release()
    
    # Ensure we have at least one frame
    if not frames:
        print("No frames were extracted. Creating a placeholder frame.")
        dummy_img = 255 * cv2.UMat(cv2.eye(480, 640, cv2.CV_8UC3))  # White image
        frame_path = os.path.join(session_dir, "frame_0s.jpg")
        cv2.imwrite(frame_path, dummy_img)
        frames.append((0, frame_path))
    
    # Save session metadata
    save_session_metadata(session_dir, video_path, frame_interval, max_frames, frames)
    
    return frames, session_dir

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
    return frames

def get_frame_paths(session_dir):
    """Return paths to all frames in a session"""
    if not session_dir or not os.path.exists(session_dir):
        return []
    
    frames = get_frames_from_session(session_dir)
    return [frame[1] for frame in frames]

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
