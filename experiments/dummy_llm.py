"""
Dummy LLM implementation for testing routing logic without API calls.
"""
from typing import Any, List, Dict, Optional, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
import json
import logging

logger = logging.getLogger(__name__)


class DummyLLM(BaseChatModel):
    """
    A dummy LLM that returns predictable responses for testing routing logic.
    """
    
    name: str = "DummyLLM"
    response_templates: Dict[str, str] = {}
    call_count: int = 0
    
    def __init__(self, name: str = "DummyLLM", response_templates: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.response_templates = response_templates or {}
        self.call_count = 0
    
    @property
    def _llm_type(self) -> str:
        return "dummy"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a dummy response based on message content."""
        self.call_count += 1
        
        # Get the last message content
        last_message = messages[-1] if messages else None
        content = last_message.content if last_message else ""
        
        # Determine response based on context
        response = self._generate_response(content, messages)
        
        logger.info(f"{self.name} called (count: {self.call_count})")
        logger.debug(f"Input: {content[:100]}...")
        logger.debug(f"Response: {response[:100]}...")
        
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _generate_response(self, content: str, messages: List[BaseMessage]) -> str:
        """Generate appropriate response based on content."""
        content_lower = content.lower()
        
        # Check for coordinator decision making
        if "decide" in content_lower or "next action" in content_lower or "which expert" in content_lower:
            return self._coordinator_response(content, messages)
        
        # Check for expert deliberation
        if "lobe" in content_lower or "deliberate" in content_lower:
            return self._expert_response(content, messages)
        
        # Check for summary
        if "summary" in content_lower or "synthesize" in content_lower:
            return self._summary_response(content, messages)
        
        # Check for tool calls
        if "create_section" in content_lower:
            return self._tool_response(content, messages)
        
        # Default response
        return f"Dummy response from {self.name}. Call count: {self.call_count}"
    
    def _coordinator_response(self, content: str, messages: List[BaseMessage]) -> str:
        """Generate coordinator decision response."""
        # Extract expert names from content if available
        response = {
            "reasoning": f"Dummy coordinator reasoning (call {self.call_count})",
            "decision": "expert_1" if self.call_count % 3 == 1 else "expert_2" if self.call_count % 3 == 2 else "summarize",
            "keywords": ["test", "dummy", "routing"],
            "instructions": "Test instructions for the expert"
        }
        return json.dumps(response, indent=2)
    
    def _expert_response(self, content: str, messages: List[BaseMessage]) -> str:
        """Generate expert response."""
        return f"Expert analysis from {self.name}. This is a dummy response for testing routing. CONCLUDED"
    
    def _summary_response(self, content: str, messages: List[BaseMessage]) -> str:
        """Generate summary response."""
        return f"Summary from {self.name}: This is a dummy summary for testing. All work completed."
    
    def _tool_response(self, content: str, messages: List[BaseMessage]) -> str:
        """Generate tool call response."""
        return json.dumps({
            "status": "success",
            "message": f"Dummy tool call from {self.name}",
            "section_id": f"section_{self.call_count}"
        })
    
    def bind_tools(self, tools: List[Any]) -> "DummyLLM":
        """Mock bind_tools for compatibility."""
        return self


class DummyCoordinatorLLM(DummyLLM):
    """Specialized dummy LLM for coordinator behavior."""
    
    expert_sequence: List[str] = []
    current_index: int = 0
    
    def __init__(self, expert_sequence: Optional[List[str]] = None, **kwargs):
        super().__init__(name="DummyCoordinator", **kwargs)
        self.expert_sequence = expert_sequence or ["expert_1", "expert_2", "summarize"]
        self.current_index = 0
    
    def _coordinator_response(self, content: str, messages: List[BaseMessage]) -> str:
        """Generate coordinator decision following a sequence."""
        decision = self.expert_sequence[self.current_index % len(self.expert_sequence)]
        self.current_index += 1
        
        response = {
            "reasoning": f"Selecting {decision} in sequence (step {self.current_index})",
            "decision": decision,
            "keywords": ["test", "sequence", f"step_{self.current_index}"],
            "instructions": f"Please provide your analysis as {decision}"
        }
        return json.dumps(response, indent=2)


class DummyExpertLLM(DummyLLM):
    """Specialized dummy LLM for expert behavior."""
    
    def __init__(self, expert_name: str, **kwargs):
        super().__init__(name=f"Dummy{expert_name}", **kwargs)
        self.expert_name = expert_name
    
    def _expert_response(self, content: str, messages: List[BaseMessage]) -> str:
        """Generate expert response with proper conclusion."""
        return f"""
{self.expert_name} Analysis (Dummy Response {self.call_count}):

This is a test response to verify routing logic.
- Point 1: Test routing is working
- Point 2: Dummy LLM is responding correctly
- Point 3: System architecture is functioning

CONCLUDED
"""
