import os
import gradio as gr
import torch
import time
import tempfile
import numpy as np
from PIL import Image
import cv2
import re
import traceback

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
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document # Import Document class
    print("DEBUG: LangChain imports successful.") # DEBUG PRINT
    LANGCHAIN_AVAILABLE = True
except ImportError as ie:
    # Specifically catch ImportError
    print(f"LangChain ImportError: {ie}. Please ensure langchain, faiss, sentence-transformers are installed.")
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

# Initialize resources
if LANGCHAIN_AVAILABLE:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = None
else:
    # Fallback to simple TF-IDF if LangChain is not available
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    document_chunks = []
    document_sources = []  # Store source filenames for each chunk
    vectorized_chunks = None

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

    else: # TF-IDF Fallback
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
    else:
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

# Wrapper function for text query processing - Returns components
def handle_text_query(query):
    # Status updates are handled implicitly
    result, sources = rag_query(query) # Get result and sources
    
    # Format source info to display only unique filenames
    if isinstance(sources, list) and sources:
        # Sources are already formatted as "- filename.pdf"
        formatted_sources = "### Source Documents:\n\n" + "\n".join(sources)
    else:
        formatted_sources = "No specific source documents found."
        
    print(f"DEBUG: handle_text_query - formatted_sources:\n{formatted_sources}") # DEBUG PRINT
    return result, formatted_sources # Returns values for results_output, sources_output

# Wrapper function for image query processing - Returns components
def handle_image_query(query, image):
    # Status updates are handled implicitly
    # process_image_with_rag returns: image_description, final_response, source_info
    image_description, final_response, sources = process_image_with_rag(image, query) # Get description, answer, and sources
    
    # Combine image description into a markdown string for output
    description_output = f"## Image Description\n\n{image_description}"
    
    # Format source info to display only unique filenames
    if isinstance(sources, list) and sources:
        # Sources are already formatted as "- filename.pdf"
        formatted_sources = "### Source Documents:\n\n" + "\n".join(sources)
    else:
        formatted_sources = "No specific source documents found."
        
    print(f"DEBUG: handle_image_query - formatted_sources:\n{formatted_sources}") # DEBUG PRINT
    return description_output, final_response, formatted_sources

# Wrapper function for video query processing - Returns components
def handle_video_query(query, video):
    # Status updates are handled implicitly
    # process_video_with_rag returns: frame_descriptions (list), final_response, source_info
    frame_descriptions, final_response, sources = process_video_with_rag(video, query) # Get analysis, answer, and sources
    
    # Combine frame descriptions into a single markdown string for output
    if isinstance(frame_descriptions, list):
        analysis_output = "## Analyzed Video Frames\n\n" + "\n\n".join(frame_descriptions)
    else:
        analysis_output = f"## Video Analysis\n\n{frame_descriptions}"
    
    # Format source info to display only unique filenames
    if isinstance(sources, list) and sources:
        # Sources are already formatted as "- filename.pdf"
        formatted_sources = "### Source Documents:\n\n" + "\n".join(sources)
    else:
        formatted_sources = "No specific source documents found."
        
    print(f"DEBUG: handle_video_query - formatted_sources:\n{formatted_sources}") # DEBUG PRINT
    return analysis_output, final_response, formatted_sources

# Master query handler to route based on input and update multiple outputs
def master_query_handler(query, image_path, video_path, progress=gr.Progress()):
    image_desc_output = ""
    rag_answer_output = ""
    source_info_output = ""
    status_output = "Processing..." # Initial status update

    try:
        # Determine input type and call appropriate handler
        if image_path is not None:
            progress(0.1, desc="Analyzing image...")
            image_desc_output, rag_answer_output, source_info_output = handle_image_query(query, image_path)
            progress(1.0, desc="Complete")
        elif video_path is not None:
            progress(0.1, desc="Extracting video frames...")
            image_desc_output, rag_answer_output, source_info_output = handle_video_query(query, video_path)
            progress(1.0, desc="Complete")
        else: # Text-only query
            progress(0.3, desc="Searching knowledge base...")
            rag_answer_output, source_info_output = handle_text_query(query)
            image_desc_output = "" # No image description for text query
            progress(1.0, desc="Complete")

        status_output = "Ready" # Final status update on success

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in master query handler: {str(e)}")
        print(f"Error trace: {error_trace}")
        status_output = "Error" # Status update on error
        rag_answer_output = f"An error occurred during processing: {str(e)}"
        source_info_output = "Error in processing."
        # Keep image_desc_output as is if it was generated before the error

    # Return updates for all output components
    # Use gr.update() to potentially hide/show the description box
    desc_update = gr.update(value=image_desc_output, visible=bool(image_path is not None or video_path is not None))
    status_update = gr.update(value=status_output, elem_classes=f"status-{status_output.lower()}")


    return status_update, desc_update, rag_answer_output, source_info_output

# Create Gradio interface
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
    border: 1px solid #e5e7eb;
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
.pure-text-tab, .image-tab, .video-tab {
    min-height: 40px;
    padding: 10px;
    margin-top: 10px;
}
.query-box {
    min-height: 100px;
    border-radius: 8px;
    border: 1px solid #374151;
}
.result-box {
    height: 350px;
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
    color: #111827;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
    border: 1px solid #e5e7eb;
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
    gr.HTML("<h1 style='text-align: center; margin-bottom: 20px; color: #111827; font-size: 2.5em; font-weight: bold;'>Unified RAG System</h1>")
    gr.HTML("<p style='text-align: center; margin-bottom: 30px; color: #4b5563;'>Multimodal Retrieval-Augmented Generation with Text, Image, and Video Analysis</p>")

    # Global status display
    with gr.Row():
        status = gr.Textbox(value="Ready", label="Status", elem_classes="status-ready", interactive=False)

    with gr.Row():
        # Left Panel (Inputs)
        with gr.Column(scale=1):
            # Knowledge Base Initialization
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Initialize Knowledge Base", elem_classes="header-text")
                init_button = gr.Button("Process PDF Knowledge Base", variant="primary", elem_classes="init-btn")
                init_output = gr.Textbox(label="Initialization Status", interactive=False)

            # Query Inputs
            with gr.Group(elem_classes="panel"):
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

        # Right Panel (Outputs)
        with gr.Column(scale=1):
            # Image/Video Description Output
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Image/Video Analysis", elem_classes="header-text")
                # This will be updated with image description or video frame analysis
                analysis_output = gr.Markdown(elem_classes="result-box", visible=False) # Initially hidden

            # RAG Results Output
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Answer from Knowledge Base", elem_classes="header-text")
                # This will be updated with the RAG answer
                results_output = gr.Markdown(elem_classes="result-box")

            # Sources Display
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Sources", elem_classes="header-text")
                sources_output = gr.Markdown(elem_classes="source-box")

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


if __name__ == "__main__":
    demo.launch(show_api=False, share=False)
