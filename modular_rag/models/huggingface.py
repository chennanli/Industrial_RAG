"""
Hugging Face model integration for the multimodal RAG system
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from modular_rag.utils.config import MODEL_CONFIG
from modular_rag.utils.helpers import get_device, get_optimal_dtype, print_debug

# Global model and processor instances
model = None
processor = None

def initialize_model():
    """Initialize the Hugging Face model with optimizations"""
    global model, processor
    
    if model is not None and processor is not None:
        return model, processor
    
    print("Loading model and processor...")
    
    # Get optimal device and data type
    device = get_device()
    print(f"Using device: {device}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_CONFIG["huggingface_model"],
        torch_dtype=get_optimal_dtype(device),
        device_map=device if device == "cuda" else "auto"
    )

    # For MPS specifically
    if device == "mps":
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_CONFIG["huggingface_model"])
    print("Model and processor loaded successfully!")
    
    return model, processor

def process_text_query(text, max_tokens=8192, temperature=0.7):
    """Process a text-only query"""
    global model, processor
    
    if model is None or processor is None:
        model, processor = initialize_model()
    
    # Prepare message for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        }
    ]

    # Process with the model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Pass None instead of empty lists for images/videos
    inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=temperature,
            repetition_penalty=1.2
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    
    # Log token usage
    input_token_count = len(inputs.input_ids[0])
    output_token_count = len(generated_ids[0]) - input_token_count
    print_debug(f"Input tokens: {input_token_count}, Output tokens: {output_token_count}, Max allowed: {max_tokens}", "Text")
    
    return output_text[0], check_for_truncation(output_text[0], output_token_count, max_tokens)

def process_image_query(text, image_path, max_tokens=8192, temperature=0.7):
    """Process a query with an image"""
    global model, processor
    
    if model is None or processor is None:
        model, processor = initialize_model()
    
    # Prepare message for the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    # Process with the model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)  # Get image inputs
    
    device = model.device
    
    # Optimize for MPS/GPU: Use fp16 precision if available
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        with torch.autocast(device_type=device.type):
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            
            # Generate output
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=temperature,
                    repetition_penalty=1.2
                )
    else:
        # CPU fallback
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=temperature,
                repetition_penalty=1.2
            )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Log token usage
    input_token_count = len(inputs.input_ids[0])
    output_token_count = len(generated_ids[0]) - input_token_count
    print_debug(f"Input tokens: {input_token_count}, Output tokens: {output_token_count}, Max allowed: {max_tokens}", "Image")
    
    return output_text[0], check_for_truncation(output_text[0], output_token_count, max_tokens)

def check_for_truncation(text, output_token_count, max_tokens, threshold=0.975):
    """Check if the output was likely truncated"""
    # Calculate what percentage of max_tokens were used
    token_usage_percentage = output_token_count / max_tokens
    
    # Common ending phrases that might indicate truncation
    truncation_indicators = [
        "Suggestions", "Suggestions Based", "Suggestions Based on",
        "Suggestions Based on Technical", "Suggestions Based on Technical Documentation",
        "Suggestions Based on Technical Documentation:", "In conclusion", "To summarize",
        "In summary", "Finally,", "Therefore,"
    ]
    
    # Check if the answer appears to be cut off
    is_truncated = any(text.endswith(indicator) for indicator in truncation_indicators)
    is_truncated = is_truncated or token_usage_percentage >= threshold
    
    if is_truncated:
        print_debug(f"Truncation detected, output tokens: {output_token_count}", "Truncation")
        return True
    
    return False
