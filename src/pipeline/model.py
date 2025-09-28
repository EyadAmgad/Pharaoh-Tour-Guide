from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF for PDF processing
import logging

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store documents and embeddings
uploaded_documents = []
document_metadata = []  # Store metadata about each document chunk
model_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize empty FAISS index (will be populated when documents are loaded)
index = None

def load_documents_from_data_dir(data_dir: str = None) -> Dict[str, Any]:
    """
    Load and process all documents from the data directory
    
    Args:
        data_dir: Path to data directory (defaults to controllers/data)
        
    Returns:
        Dict with loading results
    """
    global uploaded_documents, document_metadata, index
    
    if data_dir is None:
        # Default to controllers/data directory
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "controllers" / "data"
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {"success": False, "error": f"Data directory not found: {data_dir}"}
    
    logger.info(f"Loading documents from: {data_dir}")
    
    # Find all supported files
    supported_extensions = ['.pdf', '.txt', '.md']
    files_to_process = []
    for ext in supported_extensions:
        files_to_process.extend(data_dir.rglob(f"*{ext}"))
    
    if not files_to_process:
        logger.warning(f"No supported files found in {data_dir}")
        return {"success": False, "error": "No supported files found"}
    
    # Process each file
    all_chunks = []
    processed_files = 0
    
    for file_path in files_to_process:
        try:
            logger.info(f"Processing: {file_path.name}")
            chunks = _extract_text_and_chunk(file_path)
            all_chunks.extend(chunks)
            processed_files += 1
            logger.info(f"✓ Processed {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"✗ Failed to process {file_path.name}: {e}")
    
    if not all_chunks:
        return {"success": False, "error": "No text chunks created from documents"}
    
    # Update global storage
    uploaded_documents = [chunk['content'] for chunk in all_chunks]
    document_metadata = all_chunks
    
    # Create FAISS index
    _build_faiss_index()
    
    logger.info(f"Successfully loaded {processed_files} files with {len(all_chunks)} total chunks")
    
    return {
        "success": True,
        "files_processed": processed_files,
        "total_chunks": len(all_chunks),
        "files": [f.name for f in files_to_process[:processed_files]]
    }

def _extract_text_and_chunk(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from a file and split into chunks
    
    Returns:
        List of chunk dictionaries with content and metadata
    """
    # Extract text based on file type
    text_content = _extract_text_from_file(file_path)
    
    if not text_content.strip():
        raise ValueError("No text content extracted")
    
    # Simple chunking (split by paragraphs or max length)
    chunks = _chunk_text(text_content, chunk_size=1000, overlap=200)
    
    # Create chunk objects with metadata
    chunk_objects = []
    for i, chunk_text in enumerate(chunks):
        if chunk_text.strip():
            chunk_objects.append({
                'content': chunk_text.strip(),
                'file_name': file_path.name,
                'file_path': str(file_path),
                'chunk_index': i,
                'chunk_id': f"{file_path.stem}_chunk_{i}"
            })
    
    return chunk_objects

def _extract_text_from_file(file_path: Path) -> str:
    """Extract text from different file types"""
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        return _extract_pdf_text(file_path)
    elif extension in ['.txt', '.md']:
        return _extract_txt_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        text_content = ""
        with fitz.open(str(file_path)) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text
        return text_content
    except Exception as e:
        # Fallback to basic PDF reading if PyMuPDF fails
        logger.warning(f"PyMuPDF failed for {file_path}, using fallback")
        return f"Content from {file_path.name} (PDF processing failed: {e})"

def _extract_txt_text(file_path: Path) -> str:
    """Extract text from TXT/MD files"""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not decode {file_path} with any encoding")

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple text chunking"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at sentence or paragraph boundaries
        chunk = text[start:end]
        for i in range(len(chunk) - 1, max(0, len(chunk) - 100), -1):
            if chunk[i] in '.!?\n':
                chunk = chunk[:i + 1]
                end = start + i + 1
                break
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c.strip()]

def _build_faiss_index():
    """Build FAISS index from uploaded documents"""
    global index
    
    if not uploaded_documents:
        logger.warning("No documents to index")
        return
    
    # Create embeddings
    logger.info("Creating embeddings for documents...")
    all_embeddings = model_embeddings.encode(uploaded_documents, convert_to_tensor=True)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(np.array(all_embeddings))
    
    logger.info(f"FAISS index built with {len(uploaded_documents)} documents")

def update_documents_with_chunks(chunks):
    """
    Update the document store and FAISS index with new chunks from uploaded files
    Args:
        chunks: List of Document objects from langchain text splitter
    """
    global uploaded_documents, index
    
    # Extract text content from chunks
    new_documents = [chunk.page_content for chunk in chunks]
    
    # Add to our uploaded documents store
    uploaded_documents.extend(new_documents)
    
    # Create/recreate FAISS index with all uploaded documents
    if uploaded_documents:
        all_embeddings = model_embeddings.encode(uploaded_documents, convert_to_tensor=True)
        
        # Create new FAISS index
        index = faiss.IndexFlatL2(all_embeddings.shape[1])
        index.add(np.array(all_embeddings))
    
    return len(new_documents)

def get_current_documents():
    """Get all current uploaded documents"""
    return uploaded_documents

def clear_documents():
    """Clear all uploaded documents (useful for testing or reset)"""
    global uploaded_documents, document_metadata, index
    uploaded_documents = []
    document_metadata = []
    index = None

def get_document_stats() -> Dict[str, Any]:
    """Get statistics about loaded documents"""
    if not document_metadata:
        return {"total_documents": 0, "total_chunks": 0, "files": []}
    
    # Count unique files
    unique_files = list(set(chunk['file_name'] for chunk in document_metadata))
    
    # Count chunks per file
    file_chunks = {}
    for chunk in document_metadata:
        filename = chunk['file_name']
        file_chunks[filename] = file_chunks.get(filename, 0) + 1
    
    return {
        "total_documents": len(unique_files),
        "total_chunks": len(document_metadata),
        "files": unique_files,
        "chunks_per_file": file_chunks,
        "has_index": index is not None
    }

# OpenRouter Setup
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
if(not OPENROUTER_API_KEY):
    raise ValueError("OPEN_ROUTER_API_KEY not found in environment variables.")
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# Chat History 
chat_history = []  # Will store conversation turns

def rag_query(user_query):
    # Auto-load documents if none are present
    if not uploaded_documents or index is None:
        logger.info("No documents loaded, attempting to load from data directory...")
        load_result = load_documents_from_data_dir()
        
        if not load_result["success"]:
            return f"No documents available to answer questions. Error: {load_result.get('error', 'Unknown error')}"
        else:
            logger.info(f"Auto-loaded {load_result['files_processed']} files with {load_result['total_chunks']} chunks")
    
    # 1. Retrieve relevant chunks
    query_embedding = model_embeddings.encode([user_query])
    D, I = index.search(query_embedding, k=3)
    
    # Get all current uploaded documents
    retrieved_chunks = [uploaded_documents[i] for i in I[0]]
    retrieved_context = "\n".join(retrieved_chunks)

    # 2. Build messages for OpenRouter
    messages = [{"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided context from uploaded documents."}]
    messages.extend(chat_history)  # add previous conversation
    messages.append({
        "role": "user",
        "content": f"Context from uploaded documents:\n{retrieved_context}\n\nQuestion: {user_query}"
    })

    # 3. Send request to OpenRouter
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",  # set your API key
            "Content-Type": "application/json",
   # optional, replace with your app name
        },
        data=json.dumps({
            "model": "x-ai/grok-4-fast:free",  # ⚠️ must be a valid model in your account
            "messages": messages,
        })
    )

    # 4. Parse response
    answer_json = response.json()
    try:
        answer = answer_json["choices"][0]["message"]["content"]
    except Exception:
        answer = f"Error: {answer_json}"

    # 5. Save this turn into history
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer



# ==== Step 4: Interactive Loop ====
def start_terminal_chat():
    """
    Start an interactive terminal chat session.
    This is now optional and won't run automatically.
    """
    print("🏺 Pharaoh Tour Guide - RAG Chatbot")
    print("=" * 40)
    
    # Try to load documents first
    print("📚 Loading documents from data directory...")
    load_result = load_documents_from_data_dir()
    
    if load_result["success"]:
        print(f"✅ Loaded {load_result['files_processed']} files:")
        for filename in load_result['files']:
            print(f"   • {filename}")
        print(f"✅ Total chunks: {load_result['total_chunks']}")
    else:
        print(f"⚠️  Could not load documents: {load_result.get('error', 'Unknown error')}")
        print("You can still chat, but responses will be limited.")
    
    print("\n💬 Chat started (type 'exit' to quit)")
    print("Ask me anything about Pharaoh history!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Ending chat. Goodbye!")
                break
                
            answer = rag_query(user_input)
            print(f"🤖 Bot: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Chat ended by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending chat.")
            break
        try:
            answer = rag_query(user_input)
            print(f"Bot: {answer}\n")
        except Exception as e:
            print("Error:", e)


# Only run interactive mode if this file is executed directly (not imported)
if __name__ == "__main__":
    start_terminal_chat()