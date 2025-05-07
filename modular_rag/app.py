"""
Main application file for the modular RAG system
"""
import gradio as gr
import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules - using absolute imports
from modular_rag.utils.config import MODEL_CONFIG
from modular_rag.rag_modules.vector_store import process_pdfs
from modular_rag.rag_modules.rag_processor import handle_text_query, handle_image_query, handle_video_query
from modular_rag.models.huggingface import process_text_query, process_image_query
from modular_rag.models.lm_studio import query_lm_studio, process_image_query as lm_studio_process_image
from modular_rag.utils.video_utils import extract_frames, get_frame_paths
from modular_rag.ui.components import (
    create_model_selection_panel,
    create_pdf_processing_panel,
    create_query_panel,
    create_output_panels
)
from modular_rag.ui.styles import css

# Ensure necessary directories exist
os.makedirs("RAG_pdf", exist_ok=True)
os.makedirs("frame_cache", exist_ok=True)

# Global task cancel flag
processing_task = None

def get_model_interface():
    """Get model interface based on selected source"""
    if MODEL_CONFIG["source"] == "LMStudio":
        return {
            "text_processor": query_lm_studio,
            "image_processor": lm_studio_process_image
        }
    else:  # HuggingFace
        return {
            "text_processor": process_text_query,
            "image_processor": process_image_query
        }

def handle_direct_text_query(query):
    """Handle direct text query without RAG"""
    model_interface = get_model_interface()
    
    # Build a prompt without RAG context
    prompt = f"Please answer the following question as thoroughly as possible: {query}"
    
    # Process using appropriate model
    answer, _ = model_interface["text_processor"](prompt)
    return answer, "Direct analysis mode - no knowledge base used"
    
def handle_direct_image_query(query, image_path):
    """Handle direct image query without RAG"""
    model_interface = get_model_interface()
    
    if "image_processor" not in model_interface:
        return "Error: The selected model does not support image analysis", "", "Direct analysis mode - no knowledge base used"
        
    # First analyze the image
    description_prompt = "Describe this image in detail."
    image_description, _ = model_interface["image_processor"](description_prompt, image_path)
    
    # Then answer the question about the image
    answer_prompt = f"""
    I'm looking at an image that has been described as:
    {image_description}
    
    Based on this image, please answer the following: {query}
    """
    
    answer, _ = model_interface["image_processor"](answer_prompt, image_path)
    
    return image_description, answer, "Direct analysis mode - no knowledge base used"
    
def update_video_settings(video_path):
    """Show video settings when a video is uploaded, hide otherwise"""
    if video_path:
        # Video is uploaded, show settings
        return gr.update(visible=True)
    else:
        # No video, hide settings
        return gr.update(visible=False)

def process_direct_analysis(query, image_path, video_path, frame_interval, max_frames):
    """Process direct analysis without RAG"""
    model_interface = get_model_interface()
    
    # Simple text query without RAG
    if not image_path and not video_path:
        prompt = f"Please answer the following question as thoroughly as possible: {query}"
        answer, _ = model_interface["text_processor"](prompt)
        return "", answer
    
    # Image query without RAG
    if image_path:
        # First analyze the image
        description_prompt = "Describe this image in detail."
        image_description, _ = model_interface["image_processor"](description_prompt, image_path)
        
        # Answer question about image
        answer_prompt = f"""
        I'm looking at an image that has been described as:
        {image_description}
        
        Based on this image, please answer the following question: {query}
        
        Give a clear YES or NO answer first, then a very brief explanation. Focus ONLY on what's visible in the image.
        Do NOT mention wastewater, treatment plants, or any knowledge not directly visible in the image.
        Format your answer as:
        ANSWER: [YES/NO]
        EXPLANATION: [Brief explanation with specific details from the image]
        """
        answer, _ = model_interface["image_processor"](answer_prompt, image_path)
        
        return f"## Image Description\n\n{image_description}", answer
    
    # Video query without RAG
    if video_path:
        from modular_rag.utils.video_utils import get_frame_paths
        
        # Extract frames with specified settings
        frames, session_dir = extract_frames(
            video_path,
            frame_interval=frame_interval,
            max_frames=max_frames
        )
        
        if not frames:
            return "Could not extract frames from video", "Failed to process video"
        
        # Get paths for gallery display
        frame_paths = [path for _, path in frames]
        
        # Process each frame and create descriptions with more focused analysis
        frame_descriptions = []
        relevant_frames = []
        
        for second, frame_path in frames:
            # For each frame, first get a brief description
            description_prompt = f"Describe this video frame briefly, focusing specifically on whether: {query}"
            description, _ = model_interface["image_processor"](description_prompt, frame_path, max_tokens=256)
            
            # Check if the description indicates this frame is relevant to the query
            relevance_prompt = f"""
            Based on this description: "{description}"
            
            Is this frame relevant to answering the question: "{query}"?
            Answer with only YES or NO.
            """
            relevance, _ = model_interface["text_processor"](relevance_prompt)
            
            time_stamp = f"{int(second // 60)}min {int(second % 60)}sec"
            frame_descriptions.append(f"**Frame at {time_stamp}**: {description}")
            
            # If relevant, store this frame for detailed analysis
            if "YES" in relevance.upper():
                relevant_frames.append((second, frame_path, description))
        
        # Create a focused prompt based on only the relevant frames
        if relevant_frames:
            relevant_descriptions = [f"Frame at {sec}s: {desc}" for sec, _, desc in relevant_frames]
            relevant_text = "\n\n".join(relevant_descriptions)
            
            # Representative frame for analysis (pick the first relevant frame)
            rep_frame = relevant_frames[0][1]
            
            answer_prompt = f"""
            I'm analyzing specific video frames that might answer this question: {query}
            
            The relevant frames show:
            {relevant_text}
            
            Based ONLY on these video frames, answer the original question: {query}
            
            Give a clear YES or NO answer first, then a VERY brief explanation mentioning the specific timestamps.
            Focus ONLY on what's visible in the video frames.
            Do NOT mention wastewater, treatment plants, or any knowledge not directly visible in the frames.
            
            Format your answer as:
            ANSWER: [YES/NO]
            FOUND AT: [Specific timestamps where evidence was found]
            EXPLANATION: [Brief explanation with only specific details from the frames]
            """
            
            answer, _ = model_interface["image_processor"](answer_prompt, rep_frame)
        else:
            # No relevant frames found
            middle_frame = frames[len(frames)//2][1] if frames else None
            
            answer_prompt = f"""
            I've analyzed all frames from the video and none contain clear evidence regarding: {query}
            
            Based ONLY on the video frames, answer the original question.
            
            Give a clear NO answer, then a very brief explanation.
            
            Format your answer as:
            ANSWER: NO
            EXPLANATION: No evidence of this was found in any of the video frames.
            """
            
            if middle_frame:
                answer, _ = model_interface["image_processor"](answer_prompt, middle_frame)
            else:
                answer = "ANSWER: NO\nEXPLANATION: No video frames were available for analysis."
        
        # Just return the video descriptions for the analysis panel
        combined_description = "\n\n".join(frame_descriptions)
        return f"## Video Analysis\n\n{combined_description}", answer
    
    return "", "Error: No valid input provided"

# Global state for loaded session
loaded_session_dir = None

def master_query_handler(query, image_path, video_path, mode_selection, frame_interval, max_frames):
    """Master query handler that routes to the appropriate processor"""
    global processing_task, loaded_session_dir
    
    # First check if another process is already running
    if processing_task:
        return (gr.update(value="Busy", elem_classes="status-error"), 
                gr.update(value="", visible=False),
                gr.update(value=f"### Another Process Running"),
                "Another process is already running. Please cancel it or wait for it to complete.", 
                "Operation canceled - system busy",
                gr.update(visible=False, value=[]))
    
    processing_task = True  # Set to active
    
    image_desc_output = ""
    rag_answer_output = ""
    source_info_output = ""
    status_output = "Processing..."  # Initial status update
    use_rag = mode_selection == "Use Knowledge Base (RAG)"
    frame_paths = []
    
    # Check if we have a loaded session that should be used instead of extracting new frames
    using_loaded_frames = loaded_session_dir is not None and video_path is None

    try:
        # Check for cancel request
        if not processing_task:
            return (gr.update(value="Cancelled", elem_classes="status-error"), 
                    gr.update(value="", visible=False),
                    gr.update(value=f"### Operation Cancelled"),
                    "Processing was cancelled by user", 
                    "Operation cancelled",
                    [])
            
        if use_rag:
            # RAG Mode - use knowledge base
            if image_path is not None:
                print("Processing image query with RAG...")
                image_desc_output, rag_answer_output, source_info_output = handle_image_query(query, image_path)
            elif video_path is not None:
                print("Processing video query with RAG...")
                image_desc_output, rag_answer_output, source_info_output = handle_video_query(query, video_path)
            else:  # Text-only query
                print("Processing text-only query with RAG...")
                rag_answer_output, source_info_output = handle_text_query(query)
                image_desc_output = ""  # No image description for text query
        else:
            # Direct Analysis Mode - skip knowledge base
            if image_path is not None:
                print("Processing image query directly...")
                image_desc_output, rag_answer_output, source_info_output = handle_direct_image_query(query, image_path)
            elif video_path is not None or using_loaded_frames:
                print("Processing video query directly...")
                
                # If we have loaded frames instead of a new video, use those
                if using_loaded_frames:
                    print(f"Using pre-loaded frames from session: {loaded_session_dir}")
                    
                    # Get frames from the loaded session
                    from modular_rag.utils.video_utils import get_frames_from_session
                    frames = get_frames_from_session(loaded_session_dir)
                    
                    if not frames:
                        image_desc_output = "No frames found in the loaded session."
                        rag_answer_output = "Could not analyze video frames - none were found in the session."
                    else:
                        # Process these frames
                        model_interface = get_model_interface()
                        
                        # Process each frame and create descriptions
                        frame_descriptions = []
                        relevant_frames = []
                        
                        for second, frame_path in frames:
                            # For each frame, get a brief description
                            description_prompt = f"Describe this video frame briefly, focusing specifically on whether: {query}"
                            description, _ = model_interface["image_processor"](description_prompt, frame_path, max_tokens=256)
                            
                            # Check relevance
                            relevance_prompt = f"""
                            Based on this description: "{description}"
                            Is this frame relevant to answering the question: "{query}"?
                            Answer with only YES or NO.
                            """
                            relevance, _ = model_interface["text_processor"](relevance_prompt)
                            
                            time_stamp = f"{int(second // 60)}min {int(second % 60)}sec"
                            frame_descriptions.append(f"**Frame at {time_stamp}**: {description}")
                            
                            # Store relevant frames
                            if "YES" in relevance.upper():
                                relevant_frames.append((second, frame_path, description))
                        
                        # Create analysis summary
                        if relevant_frames:
                            # We found relevant frames
                            relevant_descriptions = [f"Frame at {sec}s: {desc}" for sec, _, desc in relevant_frames]
                            relevant_text = "\n\n".join(relevant_descriptions)
                            
                            rep_frame = relevant_frames[0][1]
                            
                            answer_prompt = f"""
                            I'm analyzing specific video frames that might answer this question: {query}
                            
                            The relevant frames show:
                            {relevant_text}
                            
                            Based ONLY on these video frames, answer the original question: {query}
                            
                            Give a clear YES or NO answer first, then a VERY brief explanation mentioning the specific timestamps.
                            Focus ONLY on what's visible in the video frames.
                            Do NOT mention wastewater, treatment plants, or any knowledge not directly visible in the frames.
                            
                            Format your answer as:
                            ANSWER: [YES/NO]
                            FOUND AT: [Specific timestamps where evidence was found]
                            EXPLANATION: [Brief explanation with only specific details from the frames]
                            """
                            
                            rag_answer_output, _ = model_interface["image_processor"](answer_prompt, rep_frame)
                        else:
                            # No relevant frames found
                            middle_frame = frames[len(frames)//2][1] if frames else None
                            
                            answer_prompt = f"""
                            I've analyzed all frames from the video and none contain clear evidence regarding: {query}
                            
                            Based ONLY on the video frames, answer the original question.
                            
                            Give a clear NO answer, then a very brief explanation.
                            
                            Format your answer as:
                            ANSWER: NO
                            EXPLANATION: No evidence of this was found in any of the video frames.
                            """
                            
                            if middle_frame:
                                rag_answer_output, _ = model_interface["image_processor"](answer_prompt, middle_frame)
                            else:
                                rag_answer_output = "ANSWER: NO\nEXPLANATION: No video frames were available for analysis."
                        
                        # Set the frame descriptions for display
                        image_desc_output = f"## Video Analysis (Using Saved Frames)\n\n" + "\n\n".join(frame_descriptions)
                        
                        # Update frame gallery with these loaded frames
                        frame_paths = [path for _, path in frames]
                        frame_gallery_update = gr.update(visible=True, value=frame_paths)
                else:
                    # Process a new video
                    image_desc_output, rag_answer_output = process_direct_analysis(query, None, video_path, frame_interval, max_frames)
                
                source_info_output = "Direct analysis mode - no knowledge base used"
            else:  # Text-only query
                print("Processing text-only query directly...")
                rag_answer_output, source_info_output = handle_direct_text_query(query)
                image_desc_output = ""  # No image description for text query

        status_output = "Ready"  # Final status update on success

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in master query handler: {str(e)}")
        print(f"Error trace: {error_trace}")
        status_output = "Error"  # Status update on error
        rag_answer_output = f"An error occurred during processing: {str(e)}"
        source_info_output = "Error in processing."
        # Keep image_desc_output as is if it was generated before the error
    finally:
        processing_task = None  # Reset task status

    # Update frame gallery display
    frame_paths = []
    if video_path is not None:
        # If we have a video, get the frame paths for the gallery
        if use_rag:
            # For RAG mode, we don't have direct access to frames, but might in the future
            frame_gallery_update = gr.update(visible=True, value=[])
        else:
            # For direct analysis, we extract frames with specified settings
            frames, session_dir = extract_frames(
                video_path,
                frame_interval=frame_interval,
                max_frames=max_frames
            )
            frame_paths = [path for _, path in frames]
            frame_gallery_update = gr.update(visible=True, value=frame_paths)
    else:
        # No video, hide gallery
        frame_gallery_update = gr.update(visible=False, value=[])

    # Return updates for all output components
    desc_update = gr.update(value=image_desc_output, visible=bool(image_path is not None or video_path is not None))
    status_update = gr.update(value=status_output, elem_classes=f"status-{status_output.lower()}")
    
    # Update result panel title based on mode
    result_title = "Direct Model Analysis" if not use_rag else "Answer from Knowledge Base"
    result_title_update = gr.update(value=f"### {result_title}")

    return status_update, desc_update, result_title_update, rag_answer_output, source_info_output, frame_gallery_update

def cancel_processing():
    """Cancel any ongoing processing task"""
    global processing_task
    
    # Force quit any current processing
    if processing_task:
        # Set flag to False to signal cancellation to any code checking it
        processing_task = False
        
        # Wait a moment to give the current process a chance to stop gracefully
        import time
        time.sleep(0.5)
        
        # If processing is still happening, we might need a more direct approach
        # in the future like thread termination, but for now we'll rely on the flag check
        
        return gr.update(value="Cancelled", elem_classes="status-error")
    
    return gr.update(value="Ready", elem_classes="status-ready")

def handle_pdf_processing():
    """Handle PDF processing with progress updates"""
    try:
        # Process PDFs without using progress callback for now
        result = process_pdfs(progress_callback=None)
        return result
    except Exception as e:
        import traceback
        print(f"Error in PDF processing: {e}")
        print(traceback.format_exc())
        return f"Error processing PDFs: {str(e)}"

def build_interface():
    """Build the Gradio interface"""
    with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as demo:
        gr.HTML("<h1 style='text-align: center; margin-bottom: 20px; color: #111827; font-size: 2.5em; font-weight: bold;'>Unified RAG System</h1>")
        gr.HTML("<p style='text-align: center; margin-bottom: 30px; color: #4b5563;'>Optimized Multimodal Retrieval-Augmented Generation with Text, Image, and Video Analysis</p>")

        # Create model selection panel with LM Studio integration
        model_panel, ui_load_fn, model_outputs = create_model_selection_panel()
        
        # Global status display
        with gr.Row():
            status = gr.Textbox(value="Ready", label="Status", elem_classes="status-ready", interactive=False)

        # Use percentage-based layout for better responsiveness
        with gr.Row(equal_height=True):
            # Left Panel (Inputs) - 40% of width
            with gr.Column(scale=2, min_width=400):
                # Add model selection panel
                model_panel
                
                # Knowledge Base Initialization
                pdf_panel, init_button, init_output = create_pdf_processing_panel()
                
                # Query Inputs
                query_panel, query_input, image_input, video_input, mode_selection, frame_interval, max_frames, video_settings, saved_frames_panel, session_dropdown, load_session_btn, refresh_btn, clear_frames_btn, frames_gallery, submit_button, cancel_button = create_query_panel()

            # Right Panel (Outputs) - 60% of width
            with gr.Column(scale=3, min_width=600):
                # Output panels
                status, analysis_panel, analysis_output, results_panel, results_title, results_output, sources_panel, sources_output = create_output_panels()

        # Event handlers for button clicks
        init_button.click(
            fn=handle_pdf_processing,
            inputs=[],
            outputs=init_output
        )

        # Show/hide video panels when video is uploaded or removed
        def update_video_panels(video_path):
            # Show both settings and saved frames panels if a video is uploaded
            if video_path:
                return gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False)
                
        video_input.change(
            fn=update_video_panels,
            inputs=[video_input],
            outputs=[video_settings, saved_frames_panel]
        )
        
        # Load saved frames functionality
        def load_saved_frames(session_choice):
            global loaded_session_dir
            
            if not session_choice:
                return gr.update(value=[], visible=False), gr.update(value="Please select a session first.")
                
            from modular_rag.utils.video_utils import get_session_choices, get_frame_paths
            
            # Get all sessions
            _, sessions = get_session_choices()
            
            # Find the selected session
            selected_index = -1
            for i, session_info in enumerate(sessions):
                formatted = f"{session_info.get('video_name', 'Unknown')} | Frames: {session_info.get('frame_count', 0)} | Interval: {session_info.get('frame_interval', 0)}s | Max: {session_info.get('max_frames', 0)} | {session_info.get('timestamp', 'Unknown')}"
                if formatted == session_choice:
                    selected_index = i
                    break
            
            if selected_index == -1:
                loaded_session_dir = None
                return gr.update(value=[], visible=False), gr.update(value="Session not found.")
                
            session = sessions[selected_index]
            session_dir = session.get('session_dir')
            
            # Store the session directory globally
            loaded_session_dir = session_dir
            
            # Get frame paths
            frame_paths = get_frame_paths(session_dir)
            
            if not frame_paths:
                return gr.update(value=[], visible=False), gr.update(value="No frames found in selected session.")
            
            # Update gallery with frames and make it visible
            return gr.update(value=frame_paths, visible=True), gr.update(value=f"Loaded {len(frame_paths)} frames from session: {session_choice}")
            
        load_session_btn.click(
            fn=load_saved_frames,
            inputs=[session_dropdown],
            outputs=[frames_gallery, status]
        )
        
        # Refresh session list functionality
        def refresh_session_list():
            from modular_rag.utils.video_utils import get_session_choices
            choices, _ = get_session_choices()
            return gr.update(choices=choices)
            
        refresh_btn.click(
            fn=refresh_session_list,
            outputs=[session_dropdown]
        )
        
        # Clear loaded frames functionality
        def clear_loaded_frames():
            global loaded_session_dir
            loaded_session_dir = None
            return [], gr.update(value="Loaded frames cleared. You can now upload a new video for analysis.")
            
        clear_frames_btn.click(
            fn=clear_loaded_frames,
            outputs=[frames_gallery, status]
        )
        
        # Master submit button handler
        submit_button.click(
            fn=master_query_handler,
            inputs=[query_input, image_input, video_input, mode_selection, frame_interval, max_frames],
            outputs=[status, analysis_output, results_title, results_output, sources_output, frames_gallery]
        )
        
        # Cancel button handler
        cancel_button.click(
            fn=cancel_processing,
            inputs=[],
            outputs=[status]
        )
        
        # Load model UI components on startup
        demo.load(
            ui_load_fn,
            outputs=model_outputs
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(show_api=False, share=False)
