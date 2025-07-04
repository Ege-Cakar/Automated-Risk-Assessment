from typing import List, Optional, Dict, Any, Callable
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_core.model_context import ChatCompletionContext
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.tools import FunctionTool, Tool
from autogen_core.models import ChatCompletionClient, SystemMessage
import json
import logging

logger = logging.getLogger(__name__)


class Lobe(AssistantAgent):
    """
    Custom agent that extends AssistantAgent with vector database capabilities.
    
    Features:
    - Accepts a list of keywords for initialization
    - Configurable temperature
    - Automatic context loading from vector database
    - Custom query_common_db tool
    """
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        vector_memory: 'LobeVectorMemory',
        keywords: List[str] = None,
        temperature: float = 0.7,
        system_message: str = None,
        tools: List[Tool] = None,
        **kwargs
    ):
        """
        Initialize Lobe agent.
        
        Args:
            name: Agent name
            model_client: The model client to use for inference
            vector_memory: The vector memory instance
            keywords: List of keywords for initial context
            temperature: Model temperature (0.0 to 1.0)
            system_message: System message for the agent
            tools: Additional tools for the agent
            **kwargs: Additional arguments passed to AssistantAgent
        """
        self.vector_memory = vector_memory
        self.keywords = keywords or []
        self._temperature = temperature
        self._base_system_message = system_message if system_message else "You are a helpful AI assistant with access to a knowledge base."
        
        # Create the query_common_db tool
        self.query_db_tool = self._create_query_db_tool()
        
        # Combine provided tools with our custom tool
        all_tools = (tools if tools else []) + [self.query_db_tool]
        
        # Initialize parent AssistantAgent
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=self._base_system_message,
            tools=all_tools,
            memory=[vector_memory] if vector_memory else [],
            **kwargs
        )
        
        # Override model client settings to use our temperature
        self._update_temperature()
    
    def _create_query_db_tool(self) -> FunctionTool:
        """Create the query_common_db tool."""
        async def query_common_db(keywords: List[str], top_k: int = 5) -> str:
            """
            Query the vector database with arbitrary keywords.
            
            Args:
                keywords: List of keywords to search for
                top_k: Number of results to return
                
            Returns:
                JSON string containing search results
            """
            try:
                # Temporarily update k value
                original_k = self.vector_memory.config.k
                self.vector_memory.config.k = top_k
                
                # Search using keywords
                results = await self.vector_memory.search_by_keywords(keywords)
                
                # Restore original k
                self.vector_memory.config.k = original_k
                
                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": result.results[0].content,
                        "score": result.results[0].metadata.get('score', 0),
                        "metadata": {k: v for k, v in result.results[0].metadata.items() if k not in ['score', 'id']}
                    })
                
                return json.dumps({
                    "query": keywords,
                    "results": formatted_results,
                    "count": len(formatted_results)
                }, indent=2)
                
            except Exception as e:
                logger.error(f"Error querying database: {e}")
                return json.dumps({
                    "error": str(e),
                    "query": keywords,
                    "results": []
                })
        
        return FunctionTool(
            query_common_db,
            description="Query the shared knowledge base with a list of keywords to retrieve relevant information."
        )
    
    async def initialize_context(self):
        """Initialize context from keywords - call after creation."""
        if not self.keywords:
            return
            
        results = await self.vector_memory.search_by_keywords(self.keywords)
        if not results:
            context = f"Initial keywords: {', '.join(self.keywords)}"
        else:
            context_parts = [f"Relevant context for keywords [{', '.join(self.keywords)}]:"]
            for i, result in enumerate(results[:3], 1):
                context_parts.append(f"{i}. {result.results[0].content}")
            context = "\n".join(context_parts)
        
        # Update system message
        self._system_messages[0] = SystemMessage(
            content=f"{self._base_system_message}\n\n{context}"
        )
    
    def _update_temperature(self):
        """Update the temperature in model client configuration."""
        # This depends on the specific model client implementation
        # For OpenAI clients, it would be in the model parameters
        if hasattr(self._model_client, '_model_args'):
            self._model_client._model_args['temperature'] = self._temperature
        # TODO: determine/finalize model / make it work for many different clients
    
    @property
    def temperature(self) -> float:
        """Get current temperature."""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float):
        """Set temperature (0.0 to 1.0)."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self._temperature = value
        self._update_temperature()
    
    async def update_keywords(self, keywords: List[str]):
        """Update keywords and refresh context."""
        self.keywords = keywords
        await self.initialize_context()
    
    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Add new knowledge to the vector database."""
        await self.vector_memory.add(
            MemoryContent(
                content=content,
                mime_type=MemoryMimeType.TEXT,
                metadata=metadata or {}
            )
        )
        logger.info(f"Added new knowledge to {self.name}'s database")
        # TODO: This might be broken...