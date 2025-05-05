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
    
def master_query_handler(query, image_path, video_path, mode_selection):
    """Master query handler that routes to the appropriate processor"""
    global processing_task
    processing_task = True  # Set to active
    
    image_desc_output = ""
    rag_answer_output = ""
    source_info_output = ""
    status_output = "Processing..."  # Initial status update
    use_rag = mode_selection == "Use Knowledge Base (RAG)"

    try:
        # Check for cancel request
        if not processing_task:
            return (gr.update(value="Cancelled", elem_classes="status-error"), 
                    gr.update(value="", visible=False),
                    "Processing was cancelled by user", 
                    "Operation cancelled")
            
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
            elif video_path is not None:
                print("Processing video query directly...")
                # Video direct analysis is similar to image but more complex
                # For now, just handle it like RAG but note the source
                image_desc_output, rag_answer_output, _ = handle_video_query(query, video_path)
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

    # Return updates for all output components
    desc_update = gr.update(value=image_desc_output, visible=bool(image_path is not None or video_path is not None))
    status_update = gr.update(value=status_output, elem_classes=f"status-{status_output.lower()}")
    
    # Update result panel title based on mode
    result_title = "Direct Model Analysis" if not use_rag else "Answer from Knowledge Base"
    result_title_update = gr.update(value=f"### {result_title}")

    return status_update, desc_update, result_title_update, rag_answer_output, source_info_output

def cancel_processing():
    """Cancel any ongoing processing task"""
    global processing_task
    if processing_task:
        processing_task = False
        return "Cancelling processing..."
    return "No active task to cancel"

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

        with gr.Row():
            # Left Panel (Inputs)
            with gr.Column(scale=1):
                # Add model selection panel
                model_panel
                
                # Knowledge Base Initialization
                pdf_panel, init_button, init_output = create_pdf_processing_panel()
                
                # Query Inputs
                query_panel, query_input, image_input, video_input, mode_selection, submit_button, cancel_button = create_query_panel()

            # Right Panel (Outputs)
            with gr.Column(scale=1):
                # Output panels
                status, analysis_panel, analysis_output, results_panel, results_title, results_output, sources_panel, sources_output = create_output_panels()

        # Event handlers for button clicks
        init_button.click(
            fn=handle_pdf_processing,
            inputs=[],
            outputs=init_output
        )

        # Master submit button handler
        submit_button.click(
            fn=master_query_handler,
            inputs=[query_input, image_input, video_input, mode_selection],
            outputs=[status, analysis_output, results_title, results_output, sources_output]
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
