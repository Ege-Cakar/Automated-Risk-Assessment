from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Any

class LobeVectorMemory:
    """Updated vector memory using current LangChain APIs"""
    
    def __init__(self, embeddings=None, persist_directory="./data/vectordb"):
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Use current Chroma API
        self.vectorstore = Chroma(
            collection_name="lobe_memory",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        self.retriever = self.vectorstore.as_retriever()
        self.config = type('Config', (), {'k': 5})()
    
    async def search_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search by keywords using current LangChain API"""
        query = " ".join(keywords)
        
        # Use invoke instead of deprecated get_relevant_documents
        docs = await self.retriever.ainvoke(query) if hasattr(self.retriever, 'ainvoke') else self.retriever.invoke(query)
        
        results = []
        for doc in docs:
            results.append({
                "results": [{
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }]
            })
        return results
    
    async def add(self, content: str, metadata: Dict[str, Any] = None):
        """Add content to vector store using current API"""
        doc = Document(page_content=content, metadata=metadata or {})
        self.vectorstore.add_documents([doc])
