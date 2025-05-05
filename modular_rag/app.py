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

def master_query_handler(query, image_path, video_path):
    """Master query handler that routes to the appropriate processor"""
    image_desc_output = ""
    rag_answer_output = ""
    source_info_output = ""
    status_output = "Processing..."  # Initial status update

    try:
        # Determine input type and call appropriate handler
        if image_path is not None:
            print("Processing image query...")
            image_desc_output, rag_answer_output, source_info_output = handle_image_query(query, image_path)
        elif video_path is not None:
            print("Processing video query...")
            image_desc_output, rag_answer_output, source_info_output = handle_video_query(query, video_path)
        else:  # Text-only query
            print("Processing text-only query...")
            rag_answer_output, source_info_output = handle_text_query(query)
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

    # Return updates for all output components
    desc_update = gr.update(value=image_desc_output, visible=bool(image_path is not None or video_path is not None))
    status_update = gr.update(value=status_output, elem_classes=f"status-{status_output.lower()}")

    return status_update, desc_update, rag_answer_output, source_info_output

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
                query_panel, query_input, image_input, video_input, submit_button = create_query_panel()

            # Right Panel (Outputs)
            with gr.Column(scale=1):
                # Output panels
                status, analysis_panel, analysis_output, results_panel, results_output, sources_panel, sources_output = create_output_panels()

        # Event handlers for button clicks
        init_button.click(
            fn=handle_pdf_processing,
            inputs=[],
            outputs=init_output
        )

        # Master submit button handler
        submit_button.click(
            fn=master_query_handler,
            inputs=[query_input, image_input, video_input],
            outputs=[status, analysis_output, results_output, sources_output]
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
