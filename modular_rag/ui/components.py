"""
UI components for the modular RAG system
"""
import gradio as gr
from modular_rag.utils.config import MODEL_CONFIG
from modular_rag.models.lm_studio import get_lm_studio_models, get_current_lm_studio_model
from modular_rag.ui.styles import css

def create_model_selection_panel():
    """Create model source selection panel with LM Studio integration"""
    
    with gr.Group(elem_classes="model-panel") as model_panel:
        gr.Markdown("### Model Selection", elem_classes="header-text")
        
        with gr.Row():
            model_choice = gr.Radio(
                choices=["HuggingFace", "LMStudio"],
                label="Model Source",
                value=MODEL_CONFIG["source"],
                interactive=True
            )
        
        with gr.Row():
            # For HuggingFace models (default visible)
            with gr.Column(visible=(MODEL_CONFIG["source"] == "HuggingFace")) as huggingface_panel:
                huggingface_model = gr.Textbox(
                    label="Hugging Face Model",
                    value=MODEL_CONFIG["huggingface_model"],
                    interactive=False
                )
            
            # For LM Studio models (initially hidden if not selected)
            with gr.Column(visible=(MODEL_CONFIG["source"] == "LMStudio")) as lm_studio_panel:
                lm_studio_dropdown = gr.Dropdown(
                    label="LM Studio Models",
                    choices=["Loading models..."],
                    value=None,
                    interactive=True,
                    allow_custom_value=True
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Models", elem_classes="refresh-btn")
                
                lm_studio_status = gr.Markdown("Status: Ready", elem_classes="model-status")
    
    # Event handlers for model selection
    
    # Function to refresh LM Studio models list
    def refresh_lm_studio_models():
        try:
            models = get_lm_studio_models()
            current_model = get_current_lm_studio_model()
            
            if current_model and current_model in models:
                return gr.update(choices=models, value=current_model), "Status: Models refreshed successfully"
            else:
                return gr.update(choices=models), "Status: Models refreshed (no model currently loaded)"
        except Exception as e:
            print(f"Error refreshing LM Studio models: {e}")
            return gr.update(choices=["Error: Could not connect to LM Studio"]), f"Status: Error: {str(e)}"
    
    # Handle LM Studio model selection
    def on_model_select(model_name):
        if model_name and model_name.strip():
            MODEL_CONFIG["lm_studio_model"] = model_name.strip()
            return f"Status: Selected model: {model_name}"
        return "Status: No model selected"
    
    # Handle model source change
    def update_model_source(choice):
        MODEL_CONFIG["source"] = choice
        
        # If switching to LM Studio, attempt to fetch available models
        if choice == "LMStudio":
            try:
                # Get available models from LM Studio
                models = get_lm_studio_models()
                current_model = get_current_lm_studio_model()
                
                # Update UI components
                return [
                    gr.update(visible=False),  # Hide HuggingFace panel
                    gr.update(visible=True),   # Show LM Studio panel
                    gr.update(choices=models, value=current_model if current_model else None),
                    f"Status: Connected to LM Studio" if current_model else "Status: Waiting for model selection"
                ]
            except Exception as e:
                print(f"Error fetching LM Studio models: {e}")
                return [
                    gr.update(visible=False),  # Hide HuggingFace panel
                    gr.update(visible=True),   # Show LM Studio panel
                    gr.update(choices=["Error fetching models"]),
                    f"Status: Error connecting to LM Studio: {str(e)}"
                ]
        else:
            # HuggingFace selected
            return [
                gr.update(visible=True),   # Show HuggingFace panel
                gr.update(visible=False),  # Hide LM Studio panel
                gr.update(),  # No change to dropdown
                "Status: Using Hugging Face model"
            ]
    
    # Function to initialize model UI on load
    def on_ui_load():
        # Initialize model interface based on current selection
        if MODEL_CONFIG["source"] == "LMStudio":
            try:
                models = get_lm_studio_models()
                current_model = get_current_lm_studio_model()
                MODEL_CONFIG["lm_studio_model"] = current_model
                
                if current_model:
                    return [
                        gr.update(visible=False),  # Hide HuggingFace panel
                        gr.update(visible=True),   # Show LM Studio panel
                        gr.update(choices=models, value=current_model),
                        f"Status: Connected to LM Studio, using model: {current_model}"
                    ]
                else:
                    return [
                        gr.update(visible=False),  # Hide HuggingFace panel
                        gr.update(visible=True),   # Show LM Studio panel
                        gr.update(choices=models),
                        "Status: No model detected in LM Studio"
                    ]
            except Exception as e:
                print(f"Error loading LM Studio models on startup: {e}")
                return [
                    gr.update(visible=True),   # Show HuggingFace panel (fallback)
                    gr.update(visible=False),  # Hide LM Studio panel
                    gr.update(choices=["Error connecting to LM Studio"]),
                    f"Status: Could not connect to LM Studio: {str(e)}"
                ]
        else:
            # HuggingFace is selected
            return [
                gr.update(visible=True),   # Show HuggingFace panel
                gr.update(visible=False),  # Hide LM Studio panel
                gr.update(),
                "Status: Using Hugging Face model"
            ]
    
    # Connect event handlers
    model_choice.change(
        update_model_source,
        inputs=[model_choice],
        outputs=[huggingface_panel, lm_studio_panel, lm_studio_dropdown, lm_studio_status]
    )
    
    refresh_btn.click(
        refresh_lm_studio_models,
        outputs=[lm_studio_dropdown, lm_studio_status]
    )
    
    lm_studio_dropdown.change(
        on_model_select,
        inputs=[lm_studio_dropdown],
        outputs=[lm_studio_status]
    )
    
    # Return the whole panel and the UI load function
    return model_panel, on_ui_load, [huggingface_panel, lm_studio_panel, lm_studio_dropdown, lm_studio_status]

def create_pdf_processing_panel():
    """Create PDF processing panel"""
    with gr.Group(elem_classes="panel") as pdf_panel:
        gr.Markdown("### Initialize Knowledge Base", elem_classes="header-text")
        init_button = gr.Button("Process PDF Knowledge Base", variant="primary", elem_classes="init-btn")
        init_output = gr.Textbox(label="Initialization Status", interactive=False)
    
    return pdf_panel, init_button, init_output

def create_query_panel():
    """Create query input panel"""
    with gr.Group(elem_classes="panel") as query_panel:
        gr.Markdown("### Enter Your Question", elem_classes="header-text")
        query_input = gr.Textbox(
            label="Question",
            placeholder="e.g., How to run the clarifier in a wastewater treatment plant?",
            lines=3,
            elem_classes="query-box"
        )
        image_input = gr.Image(label="Upload Image (Optional)", type="filepath")
        video_input = gr.Video(label="Upload Video (Optional)")
        
        submit_button = gr.Button("Submit Query", variant="primary", elem_classes="submit-btn")
    
    return query_panel, query_input, image_input, video_input, submit_button

def create_output_panels():
    """Create output display panels"""
    # Global status display
    status = gr.Textbox(value="Ready", label="Status", elem_classes="status-ready", interactive=False)
    
    # Image/Video Analysis Output
    with gr.Group(elem_classes="panel") as analysis_panel:
        gr.Markdown("### Image/Video Analysis", elem_classes="header-text")
        analysis_output = gr.Markdown(elem_classes="result-box", visible=False)  # Initially hidden
    
    # RAG Results Output
    with gr.Group(elem_classes="panel") as results_panel:
        gr.Markdown("### Answer from Knowledge Base", elem_classes="header-text")
        results_output = gr.Markdown(elem_classes="result-box")
    
    # Sources Display
    with gr.Group(elem_classes="panel") as sources_panel:
        gr.Markdown("### Sources", elem_classes="header-text")
        sources_output = gr.Markdown(elem_classes="source-box")
    
    return status, analysis_panel, analysis_output, results_panel, results_output, sources_panel, sources_output
