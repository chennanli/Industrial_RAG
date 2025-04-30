import os
import gradio as gr
import torch
import time
import tempfile
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import re
import traceback
import shutil
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

# Check if PyMuPDF is installed, if not use alternative approach
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    print("PyMuPDF not installed. Using alternative text extraction.")
    PYMUPDF_AVAILABLE = False

# Check if LangChain is installed
try:
    print("DEBUG: Attempting LangChain imports...") # DEBUG PRINT
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    # Updated import for HuggingFaceEmbeddings
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    print("DEBUG: LangChain imports successful.") # DEBUG PRINT
    LANGCHAIN_AVAILABLE = True
except ImportError as ie:
    # Specifically catch ImportError
    print(f"LangChain ImportError: {ie}. Please ensure langchain, langchain-community, langchain-huggingface, faiss-cpu (or faiss-gpu), sentence-transformers are installed.")
    LANGCHAIN_AVAILABLE = False
    Document = None
except Exception as e:
    # Catch any other unexpected error during import
    print(f"Unexpected error during LangChain import: {type(e).__name__}: {e}")
    print(traceback.format_exc()) # Print full traceback for other errors
    LANGCHAIN_AVAILABLE = False
    Document = None

print(f"DEBUG: LangChain available after import block? {LANGCHAIN_AVAILABLE}") # DEBUG PRINT

# Import Qwen model
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Required packages not found. Make sure transformers and qwen_vl_utils are installed.")
    exit(1)

# Try to import OpenCV, but provide alternative if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
    print("OpenCV is available and will be used for video processing")
except ImportError:
    print("OpenCV (cv2) is not installed. Using PIL for image processing instead.")
    OPENCV_AVAILABLE = False


# Initialize resources for RAG
if LANGCHAIN_AVAILABLE:
    # Use the updated HuggingFaceEmbeddings class
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = None
else:
    # Fallback to simple TF-IDF if LangChain is not available
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        document_chunks = []
        document_sources = []  # Store source filenames for each chunk
        vectorized_chunks = None
        SKLEARN_AVAILABLE = True
    except ImportError:
        print("Scikit-learn not found. TF-IDF fallback unavailable.")
        SKLEARN_AVAILABLE = False


# Load Qwen2.5-VL model and processor
print("Loading model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("Model and processor loaded successfully!")

# Default PDF folder
PDF_FOLDER = "RAG_pdf"

# Ensure folder exists
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

# Create necessary directories for video sessions
FRAMES_DIR = Path("saved_frames")
FRAMES_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("frame_cache")
CACHE_DIR.mkdir(exist_ok=True)

# --- RAG Functions (from Concise_RAG_v5.py) ---

# Simple text splitting function for non-LangChain fallback
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

# Function: Process PDF files and build vector store
def process_pdfs():
    global vector_store, document_chunks, document_sources, vectorized_chunks

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

    if not pdf_files:
        return "Error: No PDF files found. Please place PDF files in the RAG_pdf folder."

    texts_by_file = {} # Dictionary to store text per file: {filename: text}
    processed_files_count = 0

    # --- Step 1: Extract text from all PDFs ---
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        file_text = ""
        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(pdf_path)
                for page in doc:
                    file_text += page.get_text()
                doc.close()
            else: # Fallback methods
                try:
                    import pdfplumber
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            t = page.extract_text()
                            if t: file_text += t + "\n"
                except ImportError:
                    try:
                        with open(pdf_path, 'rb') as f:
                            # Very basic, might not work well
                            content = f.read().decode('utf-8', errors='ignore')
                            file_text += content
                    except Exception as basic_e:
                        print(f"Warning: Basic extraction failed for {pdf_file}: {basic_e}")
                        continue
                except Exception as plumber_e:
                    print(f"Warning: pdfplumber failed for {pdf_file}: {plumber_e}")
                    continue

            if file_text:
                texts_by_file[pdf_file] = file_text
                processed_files_count += 1
            else:
                 print(f"Warning: No text extracted from {pdf_file}")

        except Exception as e:
            print(f"Error processing PDF file {pdf_file}: {e}")
            # Continue processing other files

    if not texts_by_file:
        return "Error: No text could be extracted from any PDF files."

    # --- Step 2: Process extracted text and create Documents/Chunks ---
    all_docs = [] # For LangChain
    document_chunks = [] # For TF-IDF fallback
    document_sources = [] # For TF-IDF fallback source tracking
    total_chunks_count = 0

    print(f"DEBUG: Checking LANGCHAIN_AVAILABLE inside process_pdfs: {LANGCHAIN_AVAILABLE}") # DEBUG PRINT

    if LANGCHAIN_AVAILABLE:
        print("DEBUG: Attempting LangChain/FAISS path...") # DEBUG PRINT
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        for filename, text in texts_by_file.items():
            file_chunks = text_splitter.split_text(text)
            for chunk in file_chunks:
                # Create Document with source metadata
                all_docs.append(Document(page_content=chunk, metadata={"source": filename}))
            total_chunks_count += len(file_chunks)

        # --- Step 3: Create Vector Store (LangChain) ---
        if not all_docs:
             return "No text chunks generated. Knowledge base is empty."
        try:
            vector_store = FAISS.from_documents(all_docs, embeddings)
            print(f"FAISS vector store created successfully with {len(all_docs)} documents.")
        except Exception as faiss_e:
            return f"Error creating FAISS vector store: {faiss_e}"

    elif SKLEARN_AVAILABLE: # TF-IDF Fallback
        print("DEBUG: Entering TF-IDF Fallback path...") # DEBUG PRINT
        for filename, text in texts_by_file.items():
             # Simple chunking for TF-IDF with source tracking
             chunks, sources = split_text_into_chunks(text, filename)
             document_chunks.extend(chunks)
             document_sources.extend(sources)  # Track sources for each chunk
             total_chunks_count += len(chunks)

        # --- Step 3: Vectorize Chunks (TF-IDF) ---
        if document_chunks:
            try:
                vectorized_chunks = vectorizer.fit_transform(document_chunks)
                print(f"TF-IDF matrix created successfully with {len(document_chunks)} chunks.")
            except Exception as tfidf_e:
                 return f"Error creating TF-IDF matrix: {tfidf_e}"
        else:
            return "No text chunks generated for TF-IDF. Knowledge base is empty."
    else:
        return "Error: Neither LangChain nor Scikit-learn is available for vectorization."

    return f"Successfully processed {processed_files_count}/{len(pdf_files)} PDF files, generated {total_chunks_count} text chunks."

# Function: Get relevant context from vector store
def get_context(query, top_k=3):
    if LANGCHAIN_AVAILABLE:
        if vector_store is None:
            return "No knowledge base available. Please initialize the PDF knowledge base first.", []

        try:
            # Retrieve relevant documents with metadata
            docs = vector_store.similarity_search(query, k=top_k)
            
            # Create context with chunk identifiers
            context_chunks = []
            sources = []
            
            for i, doc in enumerate(docs):
                chunk_id = f"[Chunk {i+1}]"
                context_chunks.append(f"{chunk_id} {doc.page_content}")
                
                # Extract source information reliably from metadata
                source = doc.metadata.get('source', f'Unknown Source (Chunk {i+1})') # Use metadata 'source'
                sources.append(f"- {source}") # Append the actual source filename
            
            context = "\n\n".join(context_chunks)
            return context, sources
        except Exception as e:
            return f"Error retrieving context: {str(e)}", []
    elif SKLEARN_AVAILABLE:
        if not document_chunks or vectorized_chunks is None:
            return "No knowledge base available. Please initialize the PDF knowledge base first.", []

        try:
            # Vectorize the query
            query_vector = vectorizer.transform([query])

            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, vectorized_chunks)[0]

            # Get top-k most similar chunks
            top_indices = similarity_scores.argsort()[-top_k:][::-1]

            # Get the context with chunk identifiers
            context_chunks = []
            sources = []
            
            for i, idx in enumerate(top_indices):
                chunk_id = f"[Chunk {i+1}]"
                context_chunks.append(f"{chunk_id} {document_chunks[idx]}")
                
                # Use the tracked source filename for this chunk
                if idx < len(document_sources):
                    source = document_sources[idx]  # Get the actual source filename
                    sources.append(f"- {source}")
                else:
                    # Fallback if index is out of range (shouldn't happen)
                    sources.append(f"- Document Chunk {idx+1} (Source Unknown)")
            
            context = "\n\n".join(context_chunks)
            return context, sources
        except Exception as e:
            return f"Error retrieving context: {str(e)}", []
    else:
        return "Error: No vectorization method available.", []

# Function: RAG query for text
def rag_query(query, top_k=3):
    try:
        # Get relevant context and sources
        context, sources = get_context(query, top_k)

        if isinstance(context, str) and (context.startswith("No knowledge base") or context.startswith("Error")):
            return context, sources

        # Build prompt with reference to chunk identifiers
        prompt = f"""Answer the question based on the following information from technical documentation. If you cannot find the answer in the provided information, clearly state so.

Information from technical documents:
{context}

Question: {query}

Please provide a concise but complete answer and include references to the specific chunks (e.g., [Chunk 1], [Chunk 2]) that contain the information you used. At the end, provide brief suggestions based on the technical documentation. Be thorough but avoid unnecessary verbosity.

Answer:"""

        # Prepare message for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process with the model
        # Prepare inference input
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

        # Generate output with increased token limit
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=8192,  # Significantly increased token limit
                do_sample=False,      # Deterministic generation
                temperature=0.7,      # Lower temperature for more focused output
                repetition_penalty=1.2  # Discourage repetition
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        
        # Ensure the output is complete
        answer = output_text[0]
        
        # Log token usage for debugging
        input_token_count = len(inputs.input_ids[0])
        output_token_count = len(generated_ids[0]) - input_token_count
        print(f"DEBUG - Text RAG: Input tokens: {input_token_count}, Output tokens: {output_token_count}, Max allowed: 8192")
        
        # Enhanced check for truncated responses
        truncation_indicators = [
            "Suggestions", "Suggestions Based", "Suggestions Based on",
            "Suggestions Based on Technical", "Suggestions Based on Technical Documentation",
            "Suggestions Based on Technical Documentation:", "In conclusion", "To summarize",
            "In summary", "Finally,", "Therefore,"
        ]
        
        # Check if the answer appears to be cut off
        is_truncated = any(answer.endswith(indicator) for indicator in truncation_indicators)
        is_truncated = is_truncated or output_token_count >= 8000  # Close to the max limit
        
        if is_truncated:
            # If cut off, append a note
            print(f"DEBUG - Text RAG: Truncation detected, output tokens: {output_token_count}")
            answer += "\n\n[Note: The response was truncated due to token limits. Please try a more specific question or break your query into smaller parts.]"

        return answer, sources

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing query: {str(e)}")
        print(f"Error trace: {error_trace}")
        return f"Error processing query: {str(e)}", sources

# Function: Process image with RAG - MODIFIED RETURN
def process_image_with_rag(image, query, top_k=3):
    if image is None:
        return "Please upload an image", "", [] # Added empty string for answer and empty sources list

    try:
        # Step 1: Use the model to describe the image first
        description_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Describe this image in detail, focusing on any technical equipment, issues, or abnormalities visible."},
                ],
            }
        ]

        # Process the description request
        text = processor.apply_chat_template(
            description_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(description_messages) # Get image inputs
        inputs = processor(
            text=[text],
            images=image_inputs, # Pass image inputs
            videos=None,        # Pass None for videos
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate description
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024) # Increased max_new_tokens for description
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            image_description = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        # Step 2: Search knowledge base using image description
        search_query = f"{image_description} {query}"
        context, sources = get_context(search_query, top_k) # Get context and sources

        if context.startswith("No knowledge base") or context.startswith("Error"):
            return image_description, "Error: Unable to retrieve knowledge base information.", sources # Return description, error, and sources

        # Step 3: Use the context and image to answer the query
        rag_prompt = f"""I'm looking at an image and need technical help based on documentation.

What I see in the image: {image_description}

Relevant information from technical documentation:
{context}

My question: {query}

Please provide a concise but complete answer based on both the image content and the technical documentation. Include specific recommendations and steps to address any issues identified. Be thorough but avoid unnecessary verbosity."""

        # Process the RAG-enhanced query
        rag_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": rag_prompt},
                ],
            }
        ]

        # Process the final request
        text = processor.apply_chat_template(
            rag_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(rag_messages) # Get image inputs
        inputs = processor(
                text=[text],
                images=image_inputs, # Pass image inputs
                videos=None,        # Pass None for videos
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(model.device)

        # Generate final response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=8192) # Significantly increased max_new_tokens for RAG answer
            
            # Log token usage for debugging
            input_token_count = len(inputs.input_ids[0])
            output_token_count = len(generated_ids[0]) - input_token_count
            print(f"DEBUG - Image RAG: Input tokens: {input_token_count}, Output tokens: {output_token_count}, Max allowed: 8192")
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            final_response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
 
            # Check for truncation in image RAG
            if output_token_count >= 8000: # Check if very close to the 8192 limit
                 print(f"DEBUG - Image RAG: Truncation detected, output tokens: {output_token_count}")
                 final_response += "\n\n[Note: The response may have been truncated due to token limits. Consider asking a more specific question.]"

        # Return components separately for the new UI
        # Return: image_description, final_response, source_info
        return image_description, final_response, sources # Return description, answer, and sources

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing image with RAG: {str(e)}")
        print(f"Error trace: {error_trace}")
        return f"Error processing image with RAG: {str(e)}", "", [] # Added empty string for answer and empty sources list

# Function: Process video with RAG - MODIFIED RETURN
def process_video_with_rag(video_path, query, top_k=3):
    if video_path is None:
        return "Please upload a video", "", [] # Added empty string for answer and empty sources list

    # Create a temporary directory to store frames
    temp_dir = tempfile.mkdtemp()

    try:
        # Extract key frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Unable to open video file", "", [] # Added empty string for answer and empty sources list

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract 5 evenly spaced frames
        frames = []
        frame_positions = [int(i * frame_count / 5) for i in range(5)]

        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(temp_dir, f"frame_{pos}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append((pos/fps, frame_path))

        cap.release()

        if not frames:
            return "Could not extract frames from video", "", [] # Added empty string for answer and empty sources list

        # Process each frame with description
        frame_descriptions = []

        for time_pos, frame_path in frames:
            # Create description message
            description_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": frame_path,
                        },
                        {"type": "text", "text": "Describe this video frame in detail, focusing on any technical equipment, issues, or abnormalities visible."},
                    ],
                }
            ]

            # Process the description request
            text = processor.apply_chat_template(
                description_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(description_messages) # Get image inputs
            inputs = processor(
                text=[text],
                images=image_inputs, # Pass image inputs
                videos=None,        # Pass None for videos
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Generate description
            with torch.no_grad(): # Corrected indentation
                generated_ids = model.generate(**inputs, max_new_tokens=1024) # Increased max_new_tokens for frame description
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                description = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            # Add to frame descriptions
            time_stamp = f"{int(time_pos // 60)}min {int(time_pos % 60)}sec"
            frame_descriptions.append(f"**Frame at {time_stamp}**: {description}")

        # Combine descriptions for search query
        combined_description = " ".join([desc.split("**")[2] for desc in frame_descriptions if "**" in desc and len(desc.split("**")) > 2])

        # Get relevant context from knowledge base
        search_query = f"{combined_description} {query}"
        context, sources = get_context(search_query, top_k) # Get context and sources

        if context.startswith("No knowledge base") or context.startswith("Error"):
            # Return frame descriptions, error message, source info
            return "\n\n".join(frame_descriptions), "Error: Unable to retrieve knowledge base information.", sources # Return analysis, error, and sources

        # Create the RAG prompt with all frame information
        rag_prompt = f"""I'm analyzing a video and need technical help based on documentation.

What I see in the video:
{''.join(frame_descriptions)}

Relevant information from technical documentation:
{context}

My question: {query}

Please provide a concise but complete answer based on both the video content and the technical documentation. Include specific recommendations and steps to address any issues identified. Be thorough but avoid unnecessary verbosity."""

        # Use the most representative frame for the final query
        if frames and len(frames) > 0:
            middle_frame = frames[len(frames)//2][1]

            # Process the RAG-enhanced query
            rag_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": middle_frame,
                        },
                        {"type": "text", "text": rag_prompt},
                    ],
                }
            ]

            # Process the final request
            text = processor.apply_chat_template(
                rag_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(rag_messages) # Get image inputs
            inputs = processor(
                text=[text],
                images=image_inputs, # Pass image inputs
                videos=None,        # Pass None for videos
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Generate final response
            with torch.no_grad(): # Corrected indentation
                generated_ids = model.generate(**inputs, max_new_tokens=8192) # Significantly increased max_new_tokens for RAG answer
                
                # Log token usage for debugging
                input_token_count = len(inputs.input_ids[0])
                output_token_count = len(generated_ids[0]) - input_token_count
                print(f"DEBUG - Video RAG: Input tokens: {input_token_count}, Output tokens: {output_token_count}, Max allowed: 8192")
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                final_response = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
 
                # Check for truncation in video RAG
                if output_token_count >= 8000: # Check if very close to the 8192 limit
                    print(f"DEBUG - Video RAG: Truncation detected, output tokens: {output_token_count}")
                    final_response += "\n\n[Note: The response may have been truncated due to token limits. Consider asking a more specific question.]"

            # Return components separately for the new UI
            # Return: frame_descriptions (list), final_response, source_info
            return frame_descriptions, final_response, sources # Return analysis, answer, and sources
        else: # Corrected indentation
            return "Error: No frames available for analysis", "", [] # Added empty string for answer and empty sources list

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing video with RAG: {str(e)}")
        print(f"Error trace: {error_trace}")
        return f"Error processing video with RAG: {str(e)}", "", [] # Added empty string for answer and empty sources list
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

# Wrapper function for PDF processing
def handle_pdf_processing(progress=gr.Progress()):
    # Show progress during PDF processing
    progress(0.1, desc="Extracting text from PDFs...")
    result = process_pdfs()
    progress(1.0, desc="Complete")
    return result # Returns value for init_output

# Master query handler for RAG Tab
def master_rag_query_handler(query, image_path, video_path, progress=gr.Progress()):
    analysis_output = ""
    rag_answer_output = ""
    source_info_output = ""
    status_output = "Processing..." # Initial status update

    try:
        # Determine input type and call appropriate handler
        if image_path is not None:
            progress(0.1, desc="Analyzing image...")
            analysis_output, rag_answer_output, source_info_output = process_image_with_rag(image_path, query)
            progress(1.0, desc="Complete")
        elif video_path is not None:
            progress(0.1, desc="Extracting video frames...")
            # process_video_with_rag returns frame_descriptions, final_response, sources
            frame_descriptions, rag_answer_output, source_info_output = process_video_with_rag(video_path, query)
            # Combine frame descriptions into a single markdown string for output
            if isinstance(frame_descriptions, list):
                analysis_output = "## Analyzed Video Frames\n\n" + "\n\n".join(frame_descriptions)
            else:
                analysis_output = f"## Video Analysis\n\n{frame_descriptions}"
            progress(1.0, desc="Complete")
        else: # Text-only query
            progress(0.3, desc="Searching knowledge base...")
            rag_answer_output, source_info_output = rag_query(query)
            analysis_output = "" # No analysis for text query
            progress(1.0, desc="Complete")

        status_output = "Ready" # Final status update on success

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in master query handler: {str(e)}")
        print(f"Error trace: {error_trace}")
        status_output = "Error" # Status update on error
        rag_answer_output = f"An error occurred during processing: {str(e)}"
        source_info_output = "Error in processing."
        # Keep analysis_output as is if it was generated before the error

    # Format source info for display
    if isinstance(source_info_output, list) and source_info_output:
         formatted_sources = "### Source Documents:\n\n" + "\n".join(source_info_output)
    else:
         formatted_sources = "No specific source documents found."

    # Return updates for all output components
    analysis_update = gr.update(value=analysis_output, visible=bool(image_path is not None or video_path is not None))
    status_update = gr.update(value=status_output, elem_classes=f"status-{status_output.lower()}")

    return status_update, analysis_update, rag_answer_output, formatted_sources


# --- Basic Analysis Functions (from image_analysis_only.py / video_analysis_gradio_time_frame_enhanced.py) ---

def preprocess_image(image_path):
    """Image preprocessing function"""
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return False
    if OPENCV_AVAILABLE:
        # Use OpenCV for preprocessing if available
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: OpenCV could not read image: {image_path}")
            return False

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # Overwrite the original file with the preprocessed version
        success = cv2.imwrite(image_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not success:
            print(f"Error: OpenCV failed to write preprocessed image: {image_path}")
            return False
    else:
        # Use PIL for basic preprocessing if OpenCV is not available
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure RGB mode
            
            # Apply some basic enhancement
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.2)  # Increase brightness
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # Increase sharpness
            
            img.save(image_path) # Overwrite the original file
        except Exception as e:
            print(f"Error preprocessing image with PIL: {str(e)}")
            return False
    
    return True

def basic_image_analysis(image, object_str):
    """Analyze image using Hugging Face Transformers for basic description or object detection"""
    if image is None:
        return "Please upload an image."
    
    print(f"Performing basic image analysis on: {image}")
    
    # Determine the prompt based on whether an object is specified
    if object_str:
        prompt_str = f"""Please analyze the image and answer the following questions:
    1. Is there a {object_str} in the image?
    2. If yes, describe its appearance and location in the image in detail.
    3. If no, describe what you see in the image instead.
    4. On a scale of 1-10, how confident are you in your answer?

    Please structure your response as follows:
    Answer: [YES/NO]
    Description: [Your detailed description]
    Confidence: [1-10]"""
    else:
        prompt_str = "Describe this image in detail, focusing on any technical equipment, issues, or abnormalities visible."
    
    try:
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
        print("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024) # Limit tokens for basic analysis
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
        
        return response_text
    
    except Exception as e:
        return f"Error during basic image analysis: {str(e)}"

# --- Video Session Management and Basic Video Analysis Functions ---

def calculate_video_content_hash(video_path):
    """Calculate a hash based on video content (first few KB) and metadata"""
    try:
        # Get file size and modification time
        file_size = os.path.getsize(video_path)
        mod_time = os.path.getmtime(video_path)
        
        # Read first 1MB of the file for content hash
        with open(video_path, 'rb') as f:
            content = f.read(1024 * 1024)  # Read first 1MB
        
        # Create a hash combining content and metadata
        video_name = os.path.basename(video_path)
        hash_input = f"{video_name}_{file_size}_{mod_time}_{hashlib.md5(content).hexdigest()}"
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        print(f"Generated content hash for video {video_name}: {content_hash}")
        return content_hash
    except Exception as e:
        print(f"Error calculating video content hash: {e}")
        # Fallback to just the filename if we can't calculate a proper hash
        return hashlib.md5(os.path.basename(video_path).encode()).hexdigest()

def get_video_hash(video_path, frame_interval, max_frames):
    """Generate a hash for the video based on its content and extraction parameters"""
    # Get content-based hash instead of path-based hash
    content_hash = calculate_video_content_hash(video_path)
    
    # Create a unique identifier based on content hash and extraction parameters
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
    
    if not os.path.exists(FRAMES_DIR):
        return sessions
    
    for session_name in os.listdir(FRAMES_DIR):
        session_dir = os.path.join(FRAMES_DIR, session_name)
        if os.path.isdir(session_dir):
            metadata = load_session_metadata(session_dir)
            if metadata:
                # Add the session directory to the metadata
                metadata['session_dir'] = session_dir
                sessions.append(metadata)
            else:
                # Create basic metadata if none exists
                frame_files = [f for f in os.listdir(session_dir) if f.endswith('.jpg') or f.endswith('.png')]
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
                        intervals = [frame_seconds[i+1] - frame_seconds[i] for i in range(len(frame_seconds)-1)]
                        if intervals:
                            frame_interval = min(intervals)
                    
                    sessions.append({
                        'session_dir': session_dir,
                        'video_name': video_name,
                        'timestamp': timestamp,
                        'frame_interval': frame_interval,
                        'max_frames': len(frame_files),
                        'frame_count': len(frame_files),
                        'frames': [(0, f) for f in frame_files]  # Placeholder
                    })
    
    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return sessions

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

def extract_frames(video_path, output_dir, frame_interval=1, max_frames=None, progress=None):
    """Extract frames from video with configurable interval"""
    frames = []
    
    # Create a timestamp-based subfolder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.basename(video_path).split('.')[0]
    session_dir = os.path.join(output_dir, f"{video_name}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Created session directory: {session_dir}")
    
    if OPENCV_AVAILABLE:
        # Use OpenCV for frame extraction if available
        cap = cv2.VideoCapture(str(video_path))
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            # Create a placeholder frame
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_path = os.path.join(session_dir, f"frame_0s.jpg")
            print(f"Creating placeholder frame at {frame_path}")
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
                    print(f"Extracted frame at {current_second}s: {frame_path}")
                    
                    # Update progress if provided
                    if progress is not None and total_frames > 0: # Avoid division by zero
                        try:
                            progress((frame_count / total_frames), desc=f"Extracting frame at {current_second}s...")
                        except Exception as e:
                            print(f"Error updating progress: {str(e)}")
                    
                    # Stop if we've reached max_frames
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        except Exception as e:
            print(f"Error during OpenCV frame extraction: {str(e)}")
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
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                fps_str = result.stdout.strip()
                
                # Check if fps_str is empty
                if not fps_str:
                    print("Error: ffprobe didn't return frame rate. Using default fps value.")
                    fps_str = "25"  # Use a default fps value
            except Exception as e:
                print(f"Error running ffprobe: {str(e)}. Using default fps value.")
                fps_str = "25"  # Use a default fps value
            
            # Parse fps (format is usually "num/den")
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den
            else:
                fps = float(fps_str)
            
            # Extract frames using ffmpeg with specified interval
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'fps=1/{frame_interval}',  # Extract 1 frame every frame_interval seconds
                '-q:v', '1',  # High quality
            ]
            
            # Add max frames limit if specified
            if max_frames:
                cmd.extend(['-frames:v', str(max_frames)])
                
            cmd.append(f'{session_dir}/frame_%04d.jpg')
            
            print(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg with output capture
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    print(f"ffmpeg error (code {result.returncode}):")
                    print(f"stderr: {result.stderr}")
                    print(f"stdout: {result.stdout}")
                    raise Exception(f"ffmpeg failed with code {result.returncode}")
                else:
                    print("ffmpeg completed successfully")
            except Exception as e:
                print(f"Exception running ffmpeg: {str(e)}")
                raise
            
            # Get list of extracted frames
            frame_files = sorted([f for f in os.listdir(session_dir) if f.startswith('frame_')])
            
            # Check if any frames were extracted
            if not frame_files:
                print("No frames were extracted by ffmpeg. Creating a placeholder frame.")
                # Create a placeholder frame
                from PIL import Image
                img = Image.new('RGB', (640, 480), color='black')
                img_path = os.path.join(session_dir, "frame_0s.jpg")
                img.save(img_path)
                frames.append((0, img_path))
                
                # Save session metadata
                save_session_metadata(session_dir, video_path, frame_interval, max_frames, frames)
                return frames, session_dir
            
            # Process the extracted frames
            for i, frame in enumerate(frame_files):
                current_second = i * frame_interval
                old_path = os.path.join(session_dir, frame)
                new_path = os.path.join(session_dir, f"frame_{current_second}s.jpg")
                
                # Rename only if needed
                if old_path != new_path:
                    os.rename(old_path, new_path)
                
                frames.append((current_second, new_path))
                print(f"Renamed frame: {old_path} -> {new_path}")
                
                # Update progress if provided
                if progress is not None:
                    try:
                        progress((i + 1) / len(frame_files), desc=f"Processing frame at {current_second}s...")
                    except Exception as e:
                        print(f"Error updating progress: {str(e)}")
                
        except Exception as e:
            print(f"Error extracting frames with ffmpeg: {str(e)}")
            # Fallback to a very basic method if ffmpeg fails
            print("Attempting to use PIL for frame extraction (this will be slow)")
            
            try:
                from PIL import Image
                
                # Create a single frame as a placeholder
                img = Image.new('RGB', (640, 480), color='black')
                img_path = os.path.join(session_dir, "frame_0s.jpg")
                img.save(img_path)
                
                # Just add this single frame with a warning message
                frames.append((0, img_path))
                
            except Exception as e2:
                print(f"Error creating placeholder frame: {str(e2)}")
    
    print(f"Extracted {len(frames)} frames to {session_dir}")
    
    # Ensure we have at least one frame
    if not frames:
        print("No frames were extracted. Creating a placeholder frame.")
        try:
            # Try with OpenCV first
            if OPENCV_AVAILABLE:
                dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
                frame_path = os.path.join(session_dir, f"frame_0s.jpg")
                cv2.imwrite(frame_path, dummy_img)
            else:
                # Fall back to PIL
                from PIL import Image
                img = Image.new('RGB', (640, 480), color='black')
                frame_path = os.path.join(session_dir, "frame_0s.jpg")
                img.save(frame_path)
            
            frames.append((0, frame_path))
        except Exception as e:
            print(f"Error creating placeholder frame: {str(e)}")
    
    # Save session metadata
    save_session_metadata(session_dir, video_path, frame_interval, max_frames, frames)
    
    return frames, session_dir

def basic_video_analysis(video_path, object_str, frame_interval, max_frames, progress=gr.Progress()):
    """Process video and analyze frames for basic object detection"""
    if video_path is None:
        return "Please upload a video.", "No video processed"
    
    try:
        start_time = time.time()  # Initialize start_time at the beginning
        progress(0, desc="Starting basic video analysis...")
        print(f"Basic video analysis input: {video_path}")
        
        # Extract frames with the specified interval
        frames, session_dir = extract_frames(
            video_path, 
            FRAMES_DIR, 
            frame_interval=frame_interval,
            max_frames=max_frames,
            progress=progress
        )
        
        if not frames:
            return "No frames could be extracted from the video.", "No frames extracted"
        
        # Analyze the extracted frames
        results = []
        detections = []
        consecutive_detections = 0
        first_detection_second = None
        
        # Force processing only a few frames if requested
        if max_frames and len(frames) > max_frames:
            print(f"Limiting analysis to first {max_frames} frames")
            frames = frames[:max_frames]
        
        for i, (second, frame_path) in enumerate(frames):
            progress(0.5 + (i / len(frames)) * 0.5, desc=f"Analyzing frame at {second}s...")
            print(f"Processing frame {i+1}/{len(frames)}: {frame_path}")
            
            # Check if file exists
            if not os.path.exists(frame_path):
                print(f"Warning: Frame file does not exist: {frame_path}")
                continue
            
            # Preprocess the frame
            if not preprocess_image(frame_path):
                print(f"Warning: Could not preprocess frame: {frame_path}")
                continue
            
            # Analyze the frame
            try:
                response = basic_image_analysis(frame_path, object_str) # Use basic_image_analysis
                print(f"Analysis response: {response[:100]}...")  # Print first 100 chars for debugging
                
            except Exception as e:
                print(f"Error analyzing frame {frame_path}: {str(e)}")
                results.append(f"Frame at {second}s: ERROR - {str(e)}\n\n")
                continue
            
            # Parse the response (assuming basic_image_analysis returns the structured format if object_str is provided)
            is_match = False
            confidence = 0
            
            # Check if the response is in the structured format
            if object_str and "Answer:" in response and "Confidence:" in response:
                 for line in response.strip().split('\n'):
                    line = line.strip()
                    if line.lower().startswith('answer:'):
                        answer = line.split(':', 1)[1].strip().upper()
                        is_match = answer == "YES"
                    elif line.lower().startswith('confidence:'):
                        try:
                            confidence = int(line.split(':', 1)[1].strip())
                        except ValueError:
                            confidence = 0
            
                 # Track all detections with confidence info
                 if is_match:
                    detections.append((second, confidence))
                    
                 # Track consecutive high-confidence detections
                 if is_match and confidence >= 7:
                    consecutive_detections += 1
                    if consecutive_detections == 1:
                        first_detection_second = second
                 else:
                    consecutive_detections = 0
                    
                 # Add result - show DETECTED if answer is YES, regardless of confidence
                 # But note the confidence score separately
                 detection_status = "DETECTED" if is_match else "NOT DETECTED"
                 high_confidence = confidence >= 7
                 confidence_note = f"(High Confidence: {confidence}/10)" if high_confidence else f"(Low Confidence: {confidence}/10)"
                 
                 results.append(f"Frame at {second}s: {detection_status} {confidence_note}\n{response}\n\n")
                 
                 # Stop after 2 consecutive detections
                 if consecutive_detections >= 2:
                    results.append(f"\n\nObject detected consecutively, first detection at second {first_detection_second}")
                    break
            else:
                # If no object_str or not in structured format, just append the description
                results.append(f"Frame at {second}s:\n{response}\n\n")

        if not results:
            return "No frames could be analyzed.", "No frames could be analyzed"
        
        # Create a summary of the analysis
        summary = ""
        if object_str and detections:
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x[1], reverse=True)
            top_detections = detections[:3]  # Top 3 detections
            
            detection_strs = [f"{object_str} at {second}s (Confidence: {confidence}/10)"
                             for second, confidence in top_detections]
            
            end_time = time.time()
            processing_time = end_time - start_time
            summary = f"ANSWER: YES\n"
            summary += f"CONFIDENCE: High confidence detections found\n"
            summary += f"PROCESSING TIME: {processing_time:.2f}s"
        elif object_str and not detections:
             end_time = time.time()
             processing_time = end_time - start_time
             summary = f"ANSWER: NO\n"
             summary += f"CONFIDENCE: No high confidence detections found\n"
             summary += f"PROCESSING TIME: {processing_time:.2f}s"
        else:
             end_time = time.time()
             processing_time = end_time - start_time
             summary = f"Analysis Complete\nPROCESSING TIME: {processing_time:.2f}s"
            
        return "\n".join(results), summary
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error during basic video analysis: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return f"Error during basic video analysis: {str(e)}", f"Error: {str(e)}"

def get_frames_from_session(session_dir):
    """Get frames from an existing session directory"""
    if not session_dir or not os.path.exists(session_dir):
        return []
    
    frames = []
    frame_files = sorted([
        os.path.join(session_dir, f) for f in os.listdir(session_dir) 
        if (f.endswith('.jpg') or f.endswith('.png')) and f.startswith('frame_')
    ])
    
    for frame_file in frame_files:
        # Extract second from filename (format: frame_Xs.jpg)
        match = re.search(r'frame_(\d+)s\.', frame_file)
        if match:
            second = int(match.group(1))
            frame_path = os.path.join(session_dir, frame_file)
            frames.append((second, frame_path))
    
    # Sort by second
    frames.sort(key=lambda x: x[0])
    print(f"Found {len(frames)} frames in session, with timestamps: {[f[0] for f in frames[:10]]}...")
    return frames

def view_saved_frames(session_dir):
    """Return a gallery of saved frames"""
    if not session_dir or not os.path.exists(session_dir):
        return []
    
    frame_files = sorted([
        os.path.join(session_dir, f) for f in os.listdir(session_dir) 
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    
    print(f"Found {len(frame_files)} frames in {session_dir}")
    return frame_files

def delete_session(session_dir):
    """Delete a session directory"""
    if not session_dir or not os.path.exists(session_dir):
        return f"Session directory does not exist: {session_dir}"
    
    try:
        shutil.rmtree(session_dir)
        return f"Deleted session directory: {session_dir}"
    except Exception as e:
        return f"Error deleting session directory: {str(e)}"

def refresh_session_list():
    """Refresh the session list"""
    choices, _ = get_session_choices()
    return gr.update(choices=choices) # Use gr.update for dropdown

# --- Gradio UI Definition ---

css = """
.gradio-container {
    max-width: 1200px;
    margin: auto;
}
footer {
    display: none !important;
}
.footer-made-with-gradio {
    display: none !important;
}
.panel {
    background-color: #fffaf0; /* Light warm solar color */
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 2px solid #d1d5db; /* Darker, thicker border */
}
.status-ready {
    background-color: #10b981;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    display: inline-block;
    font-weight: bold;
}
.status-processing {
    background-color: #3b82f6;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    display: inline-block;
    font-weight: bold;
}
.status-error {
    background-color: #ef4444;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    display: inline-block;
    font-weight: bold;
}
.header-text {
    color: #111827;
    font-weight: bold;
    margin-bottom: 12px;
    font-size: 1.2em;
}
.query-box {
    min-height: 100px;
    border-radius: 8px;
    border: 1px solid #374151;
}
.result-box {
    min-height: 150px;
    background-color: #ffffff; /* White background */
    color: #111827 !important; /* Dark text color, forced */
    padding: 20px;
    border-radius: 8px;
    overflow-y: auto !important;
    display: block;
    border: 1px solid #e5e7eb;
    line-height: 1.6;
}
.result-box p, .result-box li, .result-box span, .result-box div {
    color: #111827 !important; /* Ensure child elements also inherit text color */
}
.result-box h1, .result-box h2, .result-box h3, .result-box h4, .result-box h5, .result-box h6 {
     color: #000000 !important; /* Black color for headings */
     margin-top: 16px;
     margin-bottom: 8px;
     font-weight: bold;
}
.source-box {
    background-color: #f9fafb;
    color: #111827 !important; /* Ensure dark text color for sources */
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
    border: 1px solid #e5e7eb;
}
.source-box p, .source-box li, .source-box span, .source-box div {
    color: #111827 !important; /* Ensure child elements also inherit text color */
}
.submit-btn {
    background-color: #3b82f6 !important;
    color: white !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.submit-btn:hover {
    background-color: #2563eb !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.result-box h2 { /* Target headings within the markdown output */
    margin-top: 16px;
    margin-bottom: 8px;
    font-weight: bold;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
}
.result-box p { /* Target paragraphs within the markdown output */
    margin-bottom: 12px;
}
.result-box ul, .result-box ol {
    margin-left: 20px;
    margin-bottom: 12px;
}
.result-box li {
    margin-bottom: 4px;
}
.result-box code {
    background-color: #f3f4f6;
    padding: 2px 4px;
    border-radius: 4px;
    font-family: monospace;
    color: #111827 !important;
}
.result-box pre {
    background-color: #f3f4f6;
    padding: 10px;
    border-radius: 8px;
    overflow-x: auto;
    margin-bottom: 12px;
}
.init-btn {
    background-color: #10b981 !important;
    color: white !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.init-btn:hover {
    background-color: #059669 !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
body {
    background-color: #fff8e1; /* Light warm background for the entire page */
}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as demo:
    gr.HTML("<h1 style='text-align: center; margin-bottom: 20px; color: #111827; font-size: 2.5em; font-weight: bold;'>Multimodal Analysis and RAG System</h1>")
    gr.HTML("<p style='text-align: center; margin-bottom: 30px; color: #4b5563;'>Combine RAG with Image/Video Analysis</p>")

    # Global status display
    with gr.Row():
        status = gr.Textbox(value="Ready", label="Status", elem_classes="status-ready", interactive=False)

    with gr.Tabs():
        # Tab 1: Concise RAG (Based on Concise_RAG_v5.py)
        with gr.TabItem("Concise RAG"):
            with gr.Row():
                # Left Panel (Inputs)
                with gr.Column(scale=1):
                    # Knowledge Base Initialization
                    with gr.Group(elem_classes="panel"):
                        gr.Markdown("### Initialize Knowledge Base", elem_classes="header-text")
                        rag_init_button = gr.Button("Process PDF Knowledge Base", variant="primary", elem_classes="init-btn")
                        rag_init_output = gr.Textbox(label="Initialization Status", interactive=False)

                    # Query Inputs
                    with gr.Group(elem_classes="panel"):
                        gr.Markdown("### Enter Your Question", elem_classes="header-text")
                        rag_query_input = gr.Textbox(
                            label="Question",
                            placeholder="e.g., How to run the clarifier in a wastewater treatment plant?",
                            lines=3,
                            elem_classes="query-box"
                        )
                        rag_image_input = gr.Image(label="Upload Image (Optional)", type="filepath")
                        rag_video_input = gr.Video(label="Upload Video (Optional)")

                        rag_submit_button = gr.Button("Submit RAG Query", variant="primary", elem_classes="submit-btn")

                # Right Panel (Outputs)
                with gr.Column(scale=1):
                    # Image/Video Description Output
                    with gr.Group(elem_classes="panel"):
                        gr.Markdown("### Image/Video Analysis", elem_classes="header-text")
                        # This will be updated with image description or video frame analysis
                        rag_analysis_output = gr.Markdown(elem_classes="result-box", visible=False) # Initially hidden

                    # RAG Results Output
                    with gr.Group(elem_classes="panel"):
                        gr.Markdown("### Answer from Knowledge Base", elem_classes="header-text")
                        # This will be updated with the RAG answer
                        rag_results_output = gr.Markdown(elem_classes="result-box")

                    # Sources Display
                    with gr.Group(elem_classes="panel"):
                        gr.Markdown("### Sources", elem_classes="header-text")
                        rag_sources_output = gr.Markdown(elem_classes="source-box")

        # Tab 2: Image Analysis Only
        with gr.TabItem("Image Analysis Only"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Image Analysis", elem_classes="header-text")
                    basic_image_input = gr.Image(type="filepath", label="Upload Image")
                    basic_image_object_input = gr.Textbox(
                        placeholder="Enter object to identify (optional)",
                        label="Object to Identify"
                    )
                    basic_image_analyze_btn = gr.Button("Analyze Image", variant="secondary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Result", elem_classes="header-text")
                    basic_image_output = gr.Textbox(label="Result", lines=20, interactive=False)

        # Tab 3: Video Analysis & Sessions
        with gr.TabItem("Video Analysis & Sessions"):
            video_session_dir_state = gr.State(None) # State to hold the current session directory
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Process New Video", elem_classes="header-text")
                        basic_video_input = gr.Video(label="Upload Video")
                        basic_video_object_input = gr.Textbox(
                            placeholder="Enter object to identify (optional)",
                            label="Object to Identify",
                            value="person"
                        )
                        
                        with gr.Row():
                            basic_video_frame_interval = gr.Slider(
                                minimum=1, 
                                maximum=10, 
                                value=2, 
                                step=1, 
                                label="Frame Interval (seconds)", 
                                info="Extract one frame every X seconds"
                            )
                            basic_video_max_frames = gr.Slider(
                                minimum=5, 
                                maximum=100, 
                                value=10, 
                                step=5, 
                                label="Maximum Frames", 
                                info="Maximum number of frames to extract and process"
                            )
                        
                        basic_video_analyze_btn = gr.Button("Analyze Video", variant="secondary")
                    
                    with gr.Group():
                        gr.Markdown("### Use Existing Frames", elem_classes="header-text")
                        session_choices, all_sessions_data = get_session_choices()
                        session_dropdown = gr.Dropdown(
                            choices=session_choices,
                            label="Select Saved Frame Session",
                            info="Choose a previously extracted frame set"
                        )
                        with gr.Row():
                            refresh_sessions_btn = gr.Button("Refresh List", variant="secondary")
                            delete_selected_session_btn = gr.Button("Delete Selected Session", variant="stop")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Results", elem_classes="header-text")
                    basic_video_output = gr.Textbox(label="Results", lines=15, interactive=False)
                    basic_video_summary = gr.Textbox(label="Summary", lines=3, interactive=False)
                    
                    gr.Markdown("### Frames Gallery", elem_classes="header-text")
                    session_frames_gallery = gr.Gallery(label="Extracted Frames", show_label=True, columns=4, height="auto")
                    session_info_display = gr.Textbox(label="Session Info", interactive=False)

    # --- Event Handlers ---

    # RAG Tab Handlers
    rag_init_button.click(
        fn=handle_pdf_processing,
        inputs=[],
        outputs=rag_init_output
    )
    rag_submit_button.click(
        fn=master_rag_query_handler,
        inputs=[rag_query_input, rag_image_input, rag_video_input],
        outputs=[status, rag_analysis_output, rag_results_output, rag_sources_output]
    )

    # Image Analysis Only Tab Handlers
    basic_image_analyze_btn.click(
        fn=basic_image_analysis,
        inputs=[basic_image_input, basic_image_object_input],
        outputs=basic_image_output
    )

    # Video Analysis & Sessions Tab Handlers
    basic_video_analyze_btn.click(
        fn=basic_video_analysis,
        inputs=[basic_video_input, basic_video_object_input, basic_video_frame_interval, basic_video_max_frames],
        outputs=[basic_video_output, basic_video_summary]
    )
    refresh_sessions_btn.click(
        fn=refresh_session_list,
        inputs=[],
        outputs=[session_dropdown]
    ).then(
        # After refreshing, clear the selection
        fn=lambda: gr.update(value=None), # Use gr.update to clear dropdown
        inputs=[],
        outputs=[session_dropdown]
    )
    delete_selected_session_btn.click(
        fn=lambda choice, sessions_data: delete_session(sessions_data[session_choices.index(choice)]['session_dir']) if choice and choice in session_choices else "No session selected",
        inputs=[session_dropdown, gr.State(all_sessions_data)], # Pass all_sessions_data as state
        outputs=[session_info_display] # Display deletion status/error
    ).then(
        fn=refresh_session_list,
        inputs=[],
        outputs=[session_dropdown]
    ).then(
        # After refreshing, clear the selection and gallery
        fn=lambda: (gr.update(value=None), [], "Session deleted. Select a new session."),
        inputs=[],
        outputs=[session_dropdown, session_frames_gallery, session_info_display]
    )
    session_dropdown.change(
        fn=lambda choice, sessions_data: (view_saved_frames(sessions_data[session_choices.index(choice)]['session_dir']), format_session_info(sessions_data[session_choices.index(choice)])) if choice and choice in session_choices else ([], "No session selected"),
        inputs=[session_dropdown, gr.State(all_sessions_data)], # Pass all_sessions_data as state
        outputs=[session_frames_gallery, session_info_display]
    )

# Launch the application
if __name__ == "__main__":
    print("Starting Tabbed Multimodal Analysis and RAG System...")
    demo.launch(show_api=False, share=False)
