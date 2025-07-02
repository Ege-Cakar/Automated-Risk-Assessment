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
import logging

logger = logging.getLogger(__name__)


class LobeVectorMemoryConfig(BaseModel):
    """Configuration for ChromaDB memory."""
    collection_name: str = Field(default="shared_memory", description="Name of the collection")
    persistence_path: str = Field(default="../database/vectordb", description="Path for persistent storage")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    k: int = Field(default=5, description="Number of results to return")
    score_threshold: float = Field(default=0.3, description="Minimum similarity score")


class LobeVectorMemory(Memory):
    """
    ChromaDB vector memory implementation with proper async handling.
    """
    
    def __init__(self, config: Optional[LobeVectorMemoryConfig] = None):
        self.config = config or LobeVectorMemoryConfig()
        self._embedding_model = SentenceTransformer(self.config.embedding_model)
        
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
        """Add content to ChromaDB using thread pool."""
        def _sync_add():
            doc_id = str(uuid.uuid4())
            
            # Convert content to string
            if content.mime_type == MemoryMimeType.TEXT:
                text_content = content.content
            else:
                text_content = str(content.content)
            
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
        doc_id = await asyncio.to_thread(_sync_add)
        logger.info(f"Added document {doc_id} to collection {self.config.collection_name}")
    
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