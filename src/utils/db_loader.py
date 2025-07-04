"""
ChromaDB Vector Database Implementation for Lobe Agent
Properly handles sync/async operations
"""
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult
from autogen_core.model_context import ChatCompletionContext
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from .paths import paths, VECTORDB_PATH
import logging
import os
import gc

# PDF processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyPDF2 not available. PDF files will be skipped. Install with: pip install PyPDF2")

# Cryptography support for encrypted PDFs
try:
    import Crypto
    CRYPTO_AVAILABLE = True
except ImportError:
    try:
        import pycryptodome
        CRYPTO_AVAILABLE = True
    except ImportError:
        CRYPTO_AVAILABLE = False


logger = logging.getLogger(__name__)


class LobeVectorMemoryConfig(BaseModel):
    """Configuration for ChromaDB memory."""
    collection_name: str = Field(default="shared_memory", description="Name of the collection")
    persistence_path: str = Field(default=VECTORDB_PATH, description="Path for persistent storage")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    k: int = Field(default=10, description="Number of results to return")
    score_threshold: float = Field(default=0.3, description="Minimum similarity score")
    chunk_size: int = Field(default=1000, description="Maximum characters per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    enable_chunking: bool = Field(default=True, description="Enable text chunking for large documents")


class TextChunker:
    """Utility class for chunking large text documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Base metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata.update({
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_size": len(text)
            })
            return [{
                "content": text,
                "metadata": chunk_metadata
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence or paragraph
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start + self.chunk_size - 200, start)
                sentence_break = self._find_sentence_break(text, search_start, end)
                if sentence_break > start:
                    end = sentence_break
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_size": len(chunk_text)
                })
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
                
                chunk_index += 1
            
            # Move start position (with overlap)
            start = max(end - self.chunk_overlap, start + 1)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        # Update total_chunks in all metadata
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    
    def _find_sentence_break(self, text: str, start: int, end: int) -> int:
        """Find a good place to break text (sentence ending)"""
        # Look for sentence endings (., !, ?)
        for i in range(end - 1, start - 1, -1):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1].isspace():
                return i + 1
        
        # Look for paragraph breaks
        for i in range(end - 1, start - 1, -1):
            if text[i] == '\n' and i + 1 < len(text) and text[i + 1] in '\n\r':
                return i + 1
        
        # Look for any line break
        for i in range(end - 1, start - 1, -1):
            if text[i] == '\n':
                return i + 1
        
        # If no good break found, return original end
        return end


class LobeVectorMemory(Memory):
    """
    ChromaDB vector memory implementation with proper async handling.
    """
    
    def __init__(self, config: Optional[LobeVectorMemoryConfig] = None):
        self.config = config or LobeVectorMemoryConfig()
        self._embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Initialize text chunker
        self.chunker = TextChunker(self.config.chunk_size, self.config.chunk_overlap)
        
        # Initialize ChromaDB client synchronously
        self._client = chromadb.PersistentClient(path=self.config.persistence_path)
        
        # Get or create collection synchronously
        try:
            self._collection = self._client.get_collection(self.config.collection_name)
        except:
            self._collection = self._client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    async def add(self, content: MemoryContent) -> None:
        """Add content to ChromaDB using thread pool with optional chunking."""
        def _sync_add():
            # Convert content to string
            if content.mime_type == MemoryMimeType.TEXT:
                text_content = content.content
            else:
                text_content = str(content.content)
            
            # Check if chunking is enabled and content is large enough
            if self.config.enable_chunking and len(text_content) > self.config.chunk_size:
                # Use chunking for large documents
                chunks = self.chunker.chunk_text(text_content, content.metadata)
                
                documents = []
                embeddings = []
                metadatas = []
                ids = []
                
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    chunk_content = chunk["content"]
                    chunk_metadata = chunk["metadata"]
                    
                    # Generate embedding for chunk
                    embedding = self._embedding_model.encode(chunk_content).tolist()
                    
                    documents.append(chunk_content)
                    embeddings.append(embedding)
                    metadatas.append(chunk_metadata)
                    ids.append(chunk_id)
                
                # Add all chunks to collection
                self._collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                return f"Added {len(chunks)} chunks"
            else:
                # Add as single document (original behavior)
                doc_id = str(uuid.uuid4())
                
                # Generate embedding
                embedding = self._embedding_model.encode(text_content).tolist()
                
                # Add to collection
                self._collection.add(
                    documents=[text_content],
                    embeddings=[embedding],
                    metadatas=[content.metadata or {}],
                    ids=[doc_id]
                )
                
                return doc_id
        
        # Run in thread to avoid blocking
        result = await asyncio.to_thread(_sync_add)
        logger.info(f"Added content to collection {self.config.collection_name}: {result}")
    
    async def query(self, query: str, cancellation_token=None) -> List[MemoryQueryResult]:
        """Query ChromaDB using thread pool."""
        def _sync_query():
            # Generate query embedding
            query_embedding = self._embedding_model.encode(query).tolist()
            
            # Query collection
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=self.config.k
            )
            
            return results
        
        # Run in thread
        results = await asyncio.to_thread(_sync_query)
        
        if not results['documents'][0]:
            logger.warning("No documents found for query: %s", query)
            return []
        
        # Convert to MemoryQueryResult
        memory_results = []
        for i in range(len(results['documents'][0])):
            # Calculate similarity score (1 - distance for cosine)
            score = 1 - results['distances'][0][i]
            
            if score >= self.config.score_threshold:
                memory_results.append(
                    MemoryQueryResult(
                        results=[MemoryContent(
                            content=results['documents'][0][i],
                            mime_type=MemoryMimeType.TEXT,
                            metadata={
                                **results['metadatas'][0][i],
                                'score': score,
                                'id': results['ids'][0][i]
                            }
                        )]
                    )
                )
        
        return memory_results # is a list of memoryqueryresult, each containing a singleton list of memorycontent... this is hell
    
    async def search_by_keywords(self, keywords: List[str]) -> List[MemoryQueryResult]:
        """Search using multiple keywords."""
        all_results = {}
        seen_ids = set()
        
        # Query for each keyword
        for keyword in keywords:
            retrieved_results = await self.query(keyword)
            for result in retrieved_results:
                doc_id = result.results[0].metadata.get('id')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results[doc_id] = result
        
        # Sort by score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.results[0].metadata.get('score', 0),
            reverse=True
        )
        
        return sorted_results[:self.config.k]
    
    async def update_context(
        self,
        context: ChatCompletionContext,
        cancellation_token=None
    ) -> ChatCompletionContext:
        """Update the context with relevant memories."""
        # This is called automatically by AssistantAgent before inference
        # TODO: THIS IS A BUG IN AUTOGEN!
        # The AutoGen library expects the context to have a 'memories' attribute
        # with a 'results' property
        
        # Create a simple object to hold the memories results
        class MemoriesContainer:
            def __init__(self, results=None):
                self.results = results or []
        
        # Add the memories attribute to the context
        if not hasattr(context, 'memories'):
            context.memories = MemoriesContainer()
        
        return context
    
    async def clear(self) -> None:
        """Clear all memories."""
        def _sync_clear():
            try:
                self._client.delete_collection(self.config.collection_name)
                self._collection = self._client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logger.error(f"Error clearing collection: {e}")
        
        await asyncio.to_thread(_sync_clear)
    
    async def close(self) -> None:
        """Close the memory store."""
        # ChromaDB clients don't need explicit closing
        pass


def _extract_pdf_text(file_path: str) -> str:
    """
    Extract text content from a PDF file using PyPDF2.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content as string
    """
    pdf_reader = None
    file_handle = None
    
    try:
        file_handle = open(file_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(file_handle)
        text = ""
        
        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            if not CRYPTO_AVAILABLE:
                print(f"PDF {file_path} is encrypted but cryptography library not available")
                return ""
            # Try to decrypt with empty password
            try:
                pdf_reader.decrypt("")
            except:
                print(f"Could not decrypt PDF {file_path}")
                return ""
        
        # Extract text from all pages
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as page_error:
                print(f"Error extracting text from page {page_num} of {file_path}: {page_error}")
                continue
        
        return text.strip()
        
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""
    
    finally:
        # Ensure proper cleanup
        if file_handle:
            try:
                file_handle.close()
            except:
                pass
        # Force garbage collection to free memory
        gc.collect()


# Helper function for batch operations
async def batch_add_documents(memory: LobeVectorMemory, documents: List[Dict[str, Any]]):
    """Batch add multiple documents efficiently."""
    tasks = []
    for doc in documents:
        content = MemoryContent(
            content=doc["content"],
            mime_type=MemoryMimeType.TEXT,
            metadata=doc.get("metadata", {})
        )
        tasks.append(memory.add(content))
    
    await asyncio.gather(*tasks)

async def add_files_from_folder(memory: LobeVectorMemory, folder_path: str, file_extensions: List[str] = None):
    """
    Add all files from a folder to LobeVectorMemory.
    
    Args:
        memory: Initialized LobeVectorMemory instance
        folder_path: Path to the folder containing files
        file_extensions: Optional list of file extensions to filter by (e.g. ['.txt', '.md'])
    """
    documents = []
    
    # Walk through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Filter by extension if specified
            if file_extensions and not any(file.endswith(ext) for ext in file_extensions):
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                content = None
                file_type = "file"
                
                # Skip system files
                if file.startswith('.DS_Store') or file.startswith('.'):
                    print(f"Skipping system file: {file_path}")
                    continue
                
                # Handle PDF files
                if file.lower().endswith('.pdf'):
                    if PDF_AVAILABLE:
                        content = _extract_pdf_text(file_path)
                        file_type = "pdf"
                    else:
                        print(f"Skipping PDF file (PyPDF2 not available): {file_path}")
                        continue
                else:
                    # Try to read as text file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        # Try with different encodings
                        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    content = f.read()
                                print(f"Successfully read {file_path} with {encoding} encoding")
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if content is None:
                            print(f"Skipping binary file (cannot decode as text): {file_path}")
                            continue
                
                # Only add if we successfully extracted content
                if content and content.strip():
                    doc = {
                        "content": content,
                        "metadata": {
                            "source": file_path,
                            "filename": file,
                            "type": file_type
                        }
                    }
                    documents.append(doc)
                    print(f"Successfully processed: {file_path}")
                else:
                    print(f"Skipping empty file: {file_path}")
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    # Use the existing batch_add_documents function
    await batch_add_documents(memory, documents)
    
    return len(documents)