"""
Vector storage module for document indexing and retrieval
"""
import os
import traceback

# Use absolute imports instead of relative imports
from modular_rag.utils.config import PDF_FOLDER
from modular_rag.utils.helpers import split_text_into_chunks, print_debug, handle_exception

# Check if LangChain is installed, if not use TF-IDF fallback
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    print_debug("LangChain imports successful.")
    LANGCHAIN_AVAILABLE = True
except ImportError as ie:
    print(f"LangChain ImportError: {ie}. Using TF-IDF fallback.")
    LANGCHAIN_AVAILABLE = False
    Document = None
except Exception as e:
    print(f"Unexpected error during LangChain import: {type(e).__name__}: {e}")
    print(traceback.format_exc())
    LANGCHAIN_AVAILABLE = False
    Document = None

# Check if PyMuPDF is installed
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    print("PyMuPDF not installed. Using alternative text extraction.")
    PYMUPDF_AVAILABLE = False

# Initialize vector store resources
if LANGCHAIN_AVAILABLE:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = None
else:
    # Fallback to simple TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    document_chunks = []
    document_sources = []
    vectorized_chunks = None

def process_pdfs(progress_callback=None):
    """Process PDFs and build vector store with progress updates"""
    global vector_store, document_chunks, document_sources, vectorized_chunks
    
    if progress_callback:
        progress_callback(0.1, "Starting PDF processing...")
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    
    if not pdf_files:
        return "Error: No PDF files found. Please place PDF files in the RAG_pdf folder."
    
    texts_by_file = {}  # Dictionary to store text per file: {filename: text}
    processed_files_count = 0
    
    # --- Step 1: Extract text from all PDFs ---
    if progress_callback:
        progress_callback(0.2, "Extracting text from PDFs...")
        
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        file_text = extract_text_from_pdf(pdf_path, pdf_file)
        
        if file_text:
            texts_by_file[pdf_file] = file_text
            processed_files_count += 1
    
    if not texts_by_file:
        return "Error: No text could be extracted from any PDF files."
    
    # --- Step 2: Process extracted text and create vector store ---
    if progress_callback:
        progress_callback(0.5, "Building vector store...")
    
    if LANGCHAIN_AVAILABLE:
        result = build_faiss_vector_store(texts_by_file)
    else:
        result = build_tfidf_matrix(texts_by_file)
    
    if progress_callback:
        progress_callback(1.0, "PDF processing complete")
        
    return result

def extract_text_from_pdf(pdf_path, pdf_file):
    """Extract text from a PDF file with multiple fallback methods"""
    file_text = ""
    
    try:
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            for page in doc:
                file_text += page.get_text()
            doc.close()
        else:  # Fallback methods
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            file_text += t + "\n"
            except ImportError:
                try:
                    with open(pdf_path, 'rb') as f:
                        # Very basic fallback
                        content = f.read().decode('utf-8', errors='ignore')
                        file_text += content
                except Exception as basic_e:
                    print(f"Warning: Basic extraction failed for {pdf_file}: {basic_e}")
                    return ""
            except Exception as plumber_e:
                print(f"Warning: pdfplumber failed for {pdf_file}: {plumber_e}")
                return ""
        
        return file_text
    
    except Exception as e:
        print(f"Error processing PDF file {pdf_file}: {e}")
        return ""

def build_faiss_vector_store(texts_by_file):
    """Build FAISS vector store from extracted texts"""
    global vector_store
    
    all_docs = []  # For LangChain
    total_chunks_count = 0
    
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
    
    # Create Vector Store (LangChain)
    if not all_docs:
        return "No text chunks generated. Knowledge base is empty."
    
    try:
        vector_store = FAISS.from_documents(all_docs, embeddings)
        print(f"FAISS vector store created successfully with {len(all_docs)} documents.")
        return f"Successfully processed {len(texts_by_file)} PDF files, generated {total_chunks_count} text chunks."
    except Exception as faiss_e:
        return f"Error creating FAISS vector store: {faiss_e}"

def build_tfidf_matrix(texts_by_file):
    """Build TF-IDF matrix fallback when LangChain isn't available"""
    global document_chunks, document_sources, vectorized_chunks
    
    document_chunks = []  # For TF-IDF fallback
    document_sources = []  # For TF-IDF fallback source tracking
    total_chunks_count = 0
    
    for filename, text in texts_by_file.items():
        # Simple chunking for TF-IDF with source tracking
        chunks, sources = split_text_into_chunks(text, filename)
        document_chunks.extend(chunks)
        document_sources.extend(sources)
        total_chunks_count += len(chunks)
    
    # Vectorize Chunks (TF-IDF)
    if document_chunks:
        try:
            vectorized_chunks = vectorizer.fit_transform(document_chunks)
            print(f"TF-IDF matrix created successfully with {len(document_chunks)} chunks.")
            return f"Successfully processed {len(texts_by_file)} PDF files, generated {total_chunks_count} text chunks."
        except Exception as tfidf_e:
            return f"Error creating TF-IDF matrix: {tfidf_e}"
    else:
        return "No text chunks generated for TF-IDF. Knowledge base is empty."

def get_relevant_context(query, top_k=3):
    """Get relevant context from vector store based on query"""
    if LANGCHAIN_AVAILABLE:
        return get_context_langchain(query, top_k)
    else:
        return get_context_tfidf(query, top_k)

def get_context_langchain(query, top_k=3):
    """Retrieve context using LangChain's FAISS"""
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
            
            # Extract source information from metadata
            source = doc.metadata.get('source', f'Unknown Source (Chunk {i+1})')
            sources.append(f"- {source}")
        
        context = "\n\n".join(context_chunks)
        return context, sources
    except Exception as e:
        return f"Error retrieving context: {str(e)}", []

def get_context_tfidf(query, top_k=3):
    """Retrieve context using TF-IDF similarity (fallback)"""
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
                source = document_sources[idx]
                sources.append(f"- {source}")
            else:
                # Fallback if index is out of range (shouldn't happen)
                sources.append(f"- Document Chunk {idx+1} (Source Unknown)")
        
        context = "\n\n".join(context_chunks)
        return context, sources
    except Exception as e:
        return f"Error retrieving context: {str(e)}", []
