"""
RAG processor module for text, image, and video processing
"""
import tempfile

# Use absolute imports instead of relative imports
from modular_rag.utils.config import MODEL_CONFIG, FRAME_CACHE_DIR
from modular_rag.utils.helpers import clean_temp_files, format_source_info, print_debug, handle_exception
from modular_rag.utils.video_utils import extract_frames, get_frame_paths
from modular_rag.rag_modules.vector_store import get_relevant_context

# Conditional imports based on model source
def get_model_interface():
    """Get the appropriate model interface based on configuration"""
    if MODEL_CONFIG["source"] == "LMStudio":
        from modular_rag.models.lm_studio import query_lm_studio, process_image_query
        return {
            "text_processor": query_lm_studio,
            "image_processor": process_image_query
        }
    else:  # Default to HuggingFace
        from modular_rag.models.huggingface import process_text_query, process_image_query
        return {
            "text_processor": process_text_query,
            "image_processor": process_image_query
        }

def process_text_rag(query, top_k=3):
    """Process a text-only RAG query"""
    try:
        # Get context from vector store
        context, sources = get_relevant_context(query, top_k)
        
        if isinstance(context, str) and (context.startswith("No knowledge base") or context.startswith("Error")):
            return context, sources
        
        # Build RAG prompt
        rag_prompt = f"""Answer the question based on the following information from technical documentation. If you cannot find the answer in the provided information, clearly state so.

Information from technical documents:
{context}

Question: {query}

Please provide a concise but complete answer and include references to the specific chunks (e.g., [Chunk 1], [Chunk 2]) that contain the information you used. At the end, provide brief suggestions based on the technical documentation. Be thorough but avoid unnecessary verbosity.

Answer:"""
        
        # Process the prompt through the appropriate model interface
        model_interface = get_model_interface()
        
        if "text_processor" in model_interface:
            answer, is_truncated = model_interface["text_processor"](rag_prompt)
            
            # Append truncation notice if needed
            if is_truncated:
                answer += "\n\n[Note: The response was truncated due to token limits. Please try a more specific question or break your query into smaller parts.]"
            
            return answer, sources
        else:
            return "Error: No text processor available for the selected model source.", sources
            
    except Exception as e:
        return handle_exception(e, "processing text RAG query"), []

def process_image_rag(image_path, query, top_k=3):
    """Process an image-based RAG query"""
    if not image_path:
        return "Please upload an image", "", []  # Empty string for answer and empty sources list
    
    try:
        model_interface = get_model_interface()
        
        # First check if we have an image processor
        if "image_processor" not in model_interface:
            return "Error: The selected model does not support image processing", "", []
        
        # Step 1: Generate image description
        description_prompt = "Describe this image in detail, focusing on any technical equipment, issues, or abnormalities visible."
        image_description, _ = model_interface["image_processor"](description_prompt, image_path, max_tokens=1024)
        
        # Step 2: Search knowledge base using image description
        search_query = f"{image_description} {query}"
        context, sources = get_relevant_context(search_query, top_k)
        
        if context.startswith("No knowledge base") or context.startswith("Error"):
            return image_description, "Error: Unable to retrieve knowledge base information.", sources
        
        # Step 3: Generate RAG response using image + context
        rag_prompt = f"""I'm looking at an image and need technical help based on documentation.

What I see in the image: {image_description}

Relevant information from technical documentation:
{context}

My question: {query}

Please provide a concise but complete answer based on both the image content and the technical documentation. Include specific recommendations and steps to address any issues identified. Be thorough but avoid unnecessary verbosity."""
        
        # Process the final RAG query with image
        final_response, is_truncated = model_interface["image_processor"](rag_prompt, image_path)
        
        # Append truncation notice if needed
        if is_truncated:
            final_response += "\n\n[Note: The response may have been truncated due to token limits. Consider asking a more specific question.]"
        
        return image_description, final_response, sources
        
    except Exception as e:
        return handle_exception(e, "processing image RAG query"), "", []

def process_video_rag(video_path, query, frame_interval=2, max_frames=10, top_k=3, progress=None):
    """Process a video-based RAG query with configurable frame extraction
    
    Args:
        video_path: Path to the video file
        query: The query to answer
        frame_interval: Extract one frame every X seconds
        max_frames: Maximum number of frames to extract
        top_k: Number of context chunks to retrieve
        progress: Optional progress callback function
        
    Returns:
        tuple: (frame_descriptions, final_response, sources)
    """
    if not video_path:
        return "Please upload a video", "", []  # Empty string for answer and empty sources list
    
    # Create a temporary directory to store intermediate results
    temp_dir = tempfile.mkdtemp()
    
    try:
        model_interface = get_model_interface()
        
        # First check if we have an image processor (needed for video frames)
        if "image_processor" not in model_interface:
            return "Error: The selected model does not support video processing", "", []
        
        # Extract frames from the video
        frames, session_dir = extract_frames(
            video_path, 
            frame_interval=frame_interval,
            max_frames=max_frames,
            progress=progress
        )
        
        if not frames:
            return "Could not extract frames from video", "", []
        
        # Process each frame with description
        frame_descriptions = []
        
        for i, (second, frame_path) in enumerate(frames):
            # Update progress if provided
            if progress:
                progress(0.5 + (i / len(frames)) * 0.5, desc=f"Analyzing frame at {second}s...")
                
            # Generate description for each frame
            description_prompt = "Describe this video frame in detail, focusing on any technical equipment, issues, or abnormalities visible."
            description, _ = model_interface["image_processor"](description_prompt, frame_path, max_tokens=1024)
            
            # Format timestamp and add to descriptions
            time_stamp = f"{int(second // 60)}min {int(second % 60)}sec"
            frame_descriptions.append(f"**Frame at {time_stamp}**: {description}")
        
        # Combine descriptions for search query
        combined_description = " ".join([desc.split("**")[2] for desc in frame_descriptions if "**" in desc and len(desc.split("**")) > 2])
        
        # Get relevant context from knowledge base
        search_query = f"{combined_description} {query}"
        context, sources = get_relevant_context(search_query, top_k)
        
        if context.startswith("No knowledge base") or context.startswith("Error"):
            return "\n\n".join(frame_descriptions), "Error: Unable to retrieve knowledge base information.", sources
        
        # Create the RAG prompt with all frame information
        rag_prompt = f"""I'm analyzing a video and need technical help based on documentation.

What I see in the video:
{''.join(frame_descriptions)}

Relevant information from technical documentation:
{context}

My question: {query}

Please provide a concise but complete answer based on both the video content and the technical documentation. Include specific recommendations and steps to address any issues identified. Be thorough but avoid unnecessary verbosity."""
        
        # Use a representative frame for the final query
        if frames:
            # Use middle frame as representative
            middle_frame = frames[len(frames)//2][1]
            
            # Process the final RAG query with representative frame
            final_response, is_truncated = model_interface["image_processor"](rag_prompt, middle_frame)
            
            # Append truncation notice if needed
            if is_truncated:
                final_response += "\n\n[Note: The response may have been truncated due to token limits. Consider asking a more specific question.]"
            
            return "\n\n".join(frame_descriptions), final_response, sources
        else:
            return "Error: No frames available for analysis", "", []
        
    except Exception as e:
        return handle_exception(e, "processing video RAG query"), "", []
    finally:
        # Clean up temporary files, but preserve cached frames
        clean_temp_files(temp_dir)

# Handler functions for UI integration
def handle_text_query(query, top_k=3):
    """Handle text query and format the results for UI"""
    result, sources = process_text_rag(query, top_k)
    formatted_sources = format_source_info(sources)
    print_debug(f"formatted_sources:\n{formatted_sources}", "text_query")
    return result, formatted_sources

def handle_image_query(query, image, top_k=3):
    """Handle image query and format the results for UI"""
    image_description, final_response, sources = process_image_rag(image, query, top_k)
    
    # Format the output
    description_output = f"## Image Description\n\n{image_description}"
    formatted_sources = format_source_info(sources)
    
    print_debug(f"formatted_sources:\n{formatted_sources}", "image_query")
    return description_output, final_response, formatted_sources

def handle_video_query(query, video, top_k=3):
    """Handle video query and format the results for UI"""
    frame_descriptions, final_response, sources = process_video_rag(video, query, top_k)
    
    # Format the output
    if isinstance(frame_descriptions, list):
        analysis_output = "## Analyzed Video Frames\n\n" + "\n\n".join(frame_descriptions)
    else:
        analysis_output = f"## Video Analysis\n\n{frame_descriptions}"
    
    formatted_sources = format_source_info(sources)
    
    print_debug(f"formatted_sources:\n{formatted_sources}", "video_query")
    return analysis_output, final_response, formatted_sources
