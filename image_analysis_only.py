import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os

# Activate the virtual environment (this is handled outside the script)
# For Mac, we'll use absolute paths for the images

# Load model and processor
print("Loading model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("Model and processor loaded successfully!")

def process_image_and_text(image, text_prompt):
    if image is None:
        return "Please upload an image."
    
    print(f"Processing image: {image}")
    
    # Build message format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # Gradio will handle the image path
                },
                {"type": "text", "text": text_prompt if text_prompt else "Describe this image in detail."},
            ],
        }
    ]
    
    try:
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
        print("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        
        return output_text[0]
    
    except Exception as e:
        return f"Error during processing: {str(e)}"

# Get absolute paths for the images in the image folder
current_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(current_dir, "image")
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Qwen2.5-VL Image Analysis Demo")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image")
            text_input = gr.Textbox(
                placeholder="Enter a prompt (if not provided, the model will describe the image)", 
                label="Prompt"
            )
            submit_btn = gr.Button("Submit")
        
        with gr.Column():
            output = gr.Textbox(label="Analysis Result")
    
    submit_btn.click(
        fn=process_image_and_text,
        inputs=[image_input, text_input],
        outputs=output
    )

    # Add examples using the actual images in the image folder
    if image_files:
        gr.Examples(
            examples=[
                [img, "What can you see in this image?"] for img in image_files
            ],
            inputs=[image_input, text_input],
        )

# Launch the application
if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch(share=False)  # Set share=False for local use only