from typing import List, Dict, Any
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import TextMessage, ChatMessage, StopMessage, HandoffMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import FunctionTool
import asyncio
import logging
from .lobe import Lobe
from ..utils.db_loader import LobeVectorMemory

logger = logging.getLogger(__name__)

class Expert(BaseChatAgent):
    """
    Expert agent that internally manages a team of two Lobe agents.
    Appears as a single agent externally but runs an internal deliberation process.
    Supports handoffs.
    """
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        vector_memory: 'LobeVectorMemory',
        system_message: str = None,
        lobe1_config: Dict[str, Any] = None,
        lobe2_config: Dict[str, Any] = None,
        handoffs: List[BaseChatAgent] = None,
        max_rounds: int = 100,
        description: str = "An expert agent that internally deliberates using two specialized lobes, one for creativity and one for reasoning.",
        **kwargs
    ):
        """
        Initialize Expert agent with two internal Lobe agents.
        
        Args:
            name: Expert agent name
            model_client: The model client for both lobes
            vector_memory: Shared vector memory for both lobes
            lobe1_config: Configuration for Lobe 1 (keywords, temperature, system_message, tools)
            lobe2_config: Configuration for Lobe 2 (keywords, temperature, system_message, tools)
            max_rounds: Maximum rounds of internal discussion
            description: Agent description
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description)
        
        self._model_client = model_client
        self._vector_memory = vector_memory
        self._max_rounds = max_rounds
        self._base_system_message = system_message if system_message else (
            "You are an expert assistant with deep knowledge in your domain. "
            "You think carefully and provide well-reasoned responses."
        )
        self._handoffs = handoffs or []

        handoff_tools = self._create_handoff_tools()
        
        # Default configurations
        lobe1_config = lobe1_config or {}
        lobe2_config = lobe2_config or {}

        lobe1_specific = """You are the creative counterpart implementing SWIFT methodology. Your role:
        1. Generate what-if scenarios by systematically applying guide words (Wrong: Person/Place/Thing/Idea/Time/Process/Amount, Failure: Control/Detection/Equipment) to system components
        2. For each scenario, imagine creative but plausible deviations and their cascading consequences
        3. Consider both obvious and non-obvious failure modes
        4. Propose innovative safeguards beyond standard controls
        Stay focused on SWIFT risk assessment - every idea must connect to a specific guide word + component combination.
        IMPORTANT: You are in a dialogue with your analytical counterpart. Present scenarios one at a time for discussion. Do NOT try to cover everything in one message. Start with your most critical scenario.
        """

        lobe2_specific = """You are the analytical counterpart implementing SWIFT methodology. Your role:
        1. Validate what-if scenarios for realism and criticality
        2. Assess likelihood and impact using consistent criteria
        3. Evaluate existing safeguards' effectiveness quantitatively
        4. Prioritize risks using a risk matrix (likelihood × impact)
        5. Synthesize actionable recommendations with implementation details
        IMPORTANT: You are in a dialogue with your creative counterpart. Respond to each scenario they present, then ask for the next one. Only after you've covered multiple scenarios across different guide words should you synthesize findings. When ready to conclude (after substantial analysis), start your message with 'CONCLUDE:' followed by structured findings"""
            
        if self._handoffs:
            handoff_instruction = self._create_handoff_instruction()
            lobe1_specific += f"\n\n{handoff_instruction}"
            lobe2_specific += f"\n\n{handoff_instruction}"

        lobe1_full_message = f"{self._base_system_message}\n\n{lobe1_specific}"
        lobe2_full_message = f"{self._base_system_message}\n\n{lobe2_specific}"

        lobe1_tools = lobe1_config.get('tools', []) + handoff_tools
        lobe2_tools = lobe2_config.get('tools', []) + handoff_tools

        # Create Lobe 1 - Analytical specialist
        self._lobe1 = Lobe(
            name=f"{name}_Creative",
            model_client=model_client,
            vector_memory=vector_memory,
            keywords=lobe1_config.get('keywords', []),
            temperature=lobe1_config.get('temperature', 0.8),
            system_message=lobe1_full_message,
            tools=lobe1_tools
        )
        
        self._lobe2 = Lobe(
            name=f"{name}_VoReason",
            model_client=model_client,
            vector_memory=vector_memory,
            keywords=lobe2_config.get('keywords', []),
            temperature=lobe2_config.get('temperature', 0.4),  # Lower temperature for checking
            system_message=lobe2_full_message,
            tools=lobe2_tools
        )
        
        # Initialize contexts for both lobes
        self._initialized = False
        
        # Setup internal team
        self._setup_internal_team()
    
    def _create_handoff_tools(self) -> List[FunctionTool]:
        """Create function tools for each handoff agent."""
        tools = []
        for target_name in self._handoffs:
            def create_handoff_function(target: str):
                async def handoff_to_agent(reason: str = "") -> str:
                    """Mark that a handoff should occur after providing response."""
                    # Store handoff info for later processing
                    self._pending_handoff = {
                        'target': target,
                        'reason': reason,
                    }
                    return f"[Handoff to {target} noted - will complete after providing response]"
                
                return handoff_to_agent
            
            tool = FunctionTool(
                create_handoff_function(target_name),
                name=f"transfer_to_{target_name}",
                description=f"Mark for transfer to {target_name} after responding."
            )
            tools.append(tool)
        
        return tools

    
    def _create_handoff_instruction(self) -> str:
        """Create instruction text about available handoff agents."""
        if not self._handoffs:
            return ""
        
        handoff_list = [f"- {name}" for name in self._handoffs]
        
        return (
            "When you determine that another agent could provide additional value after "
            "your response, you can mark the conversation for handoff. You should ALWAYS "
            "provide your complete analysis first, then use the transfer function to "
            "indicate which agent should provide supplementary expertise.\n\n"
            "Available agents for handoff:\n" + 
            "\n".join(handoff_list) + 
            "\n\nIMPORTANT: Always complete your full response before marking for handoff."
        )
        
    async def _initialize_lobes(self):
        """Initialize context for both lobes."""
        if not self._initialized:
            await self._lobe1.initialize_context()
            await self._lobe2.initialize_context()
            self._initialized = True
            logger.info(f"Initialized both lobes for Expert {self.name}")
    
    def _setup_internal_team(self):
        """Set up the internal round-robin team for the two lobes."""
        # Use TextMentionTermination to stop when Lobe2 says "CONCLUDE:"
        conclude_condition = TextMentionTermination("CONCLUDE:")

        handoff_condition = TextMentionTermination("transfer_to_")
        
        # Fallback: stop after max rounds
        max_messages_condition = MaxMessageTermination(max_messages=self._max_rounds * 2)
        
        # Create internal team
        self._internal_team = RoundRobinGroupChat(
            participants=[self._lobe1, self._lobe2],
            termination_condition=conclude_condition | max_messages_condition | handoff_condition
        )
        
        logger.info(f"Setup internal team for Expert {self.name}")
    
    async def on_messages(
        self, 
        messages: List[ChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """
        Process incoming messages by running internal team discussion.
        Can provide a full response AND hand off to another agent.
        
        Args:
            messages: List of chat messages
            cancellation_token: Cancellation token
            
        Returns:
            Response containing the expert's synthesized conclusion and/or handoff
        """
        if not messages:
            logger.warning(f"Expert {self.name} received empty message list")
            msg = f"Expert {self.name} called. Messages count: {len(messages)}. "
            return Response(
                chat_message=TextMessage(
                    content=msg,
                    source=self.name
                )
            )
    
        # Reset pending handoff
        self._pending_handoff = None
        
        # Ensure lobes are initialized
        await self._initialize_lobes()
        
        # Extract the query from the last message
        last_message = messages[-1]
        if isinstance(last_message, TextMessage):
            query = last_message.content
        else:
            query = str(last_message)
        
        # Log the incoming query
        logger.info(f"Expert {self.name} received a message.")
        
        # Prepare initial task for internal team
        internal_task = query
        
        try:
            # Run internal team discussion
            logger.info(f"Starting internal deliberation for Expert {self.name}")
            result = await self._internal_team.run(
                task=internal_task,
                cancellation_token=cancellation_token
            )
            
            # Extract and format the conclusion FIRST
            final_response = self._extract_conclusion(result)
            
            # Check if a handoff was triggered during internal discussion
            if self._pending_handoff:
                logger.info(f"Expert {self.name} providing response and initiating handoff to {self._pending_handoff['target']}")
                
                # Combine the response with handoff indication
                combined_content = (
                    f"{final_response}\n\n"
                    f"---\n"
                    f"For additional expertise, I'm handing this off to {self._pending_handoff['target']}: "
                    f"{self._pending_handoff['reason']}"
                )
                
                # Return a HandoffMessage that includes the full response
                handoff_msg = HandoffMessage(
                    content=combined_content,
                    target=self._pending_handoff['target'],
                    source=self.name
                )
                return Response(chat_message=handoff_msg)
            
            logger.info(f"Expert {self.name} completed deliberation")
            
            # Return normal response if no handoff
            return Response(
                chat_message=TextMessage(
                    content=final_response,
                    source=self.name
                )
            )
            
        except asyncio.CancelledError:
            logger.warning(f"Expert {self.name} deliberation was cancelled")
            raise
            
        except Exception as e:
            logger.error(f"Error in Expert {self.name} deliberation: {str(e)}", exc_info=True)
            error_message = (
                f"I encountered an error during internal deliberation. "
                f"Let me provide a direct response instead: Based on the query about "
                f"'{query}', I need more specific information to provide a detailed answer."
            )
            return Response(
                chat_message=TextMessage(
                    content=error_message,
                    source=self.name
                )
            )

    
    def _extract_conclusion(self, task_result: TaskResult) -> str:
        """
        Extract the conclusion from the internal team's discussion.
        
        Args:
            task_result: Result from the internal team discussion
            
        Returns:
            The synthesized conclusion string
        """
        # Look for explicit conclusion from Lobe 2
        for message in reversed(task_result.messages):
            if isinstance(message, TextMessage) and message.source == self._lobe2.name:
                content = message.content
                if "CONCLUDE:" in content:
                    # Extract everything after "CONCLUDE:"
                    conclusion_start = content.find("CONCLUDE:") + len("CONCLUDE:")
                    conclusion = content[conclusion_start:].strip()
                    return conclusion
        
        # Fallback: If no explicit conclusion, synthesize from last few messages
        logger.warning(f"No explicit conclusion found for Expert {self.name}, synthesizing...")
        
        synthesis_parts = ["Based on internal analysis:"]
        
        # Get last 3 messages for synthesis
        recent_messages = task_result.messages
        
        for msg in recent_messages:
            if isinstance(msg, TextMessage):
                content = msg.content
                synthesis_parts.append(f"- {content}")
        
        return "\n".join(synthesis_parts)
    
    async def update_keywords(self, lobe1_keywords: List[str] = None, lobe2_keywords: List[str] = None):
        """
        Update keywords for one or both lobes.
        
        Args:
            lobe1_keywords: New keywords for Lobe 1
            lobe2_keywords: New keywords for Lobe 2
        """
        if lobe1_keywords is not None:
            await self._lobe1.update_keywords(lobe1_keywords)
            logger.info(f"Updated Lobe 1 keywords for Expert {self.name}")
            
        if lobe2_keywords is not None:
            await self._lobe2.update_keywords(lobe2_keywords)
            logger.info(f"Updated Lobe 2 keywords for Expert {self.name}")
    
    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """
        Add new knowledge to the shared vector database.
        
        Args:
            content: Knowledge content to add
            metadata: Optional metadata
        """
        # Add through one of the lobes (they share the same vector memory)
        await self._lobe1.add_knowledge(content, metadata)
        logger.info(f"Added knowledge to Expert {self.name}'s shared database")
    
    @property
    def lobe1(self) -> Lobe:
        """Access to Lobe 1 for direct configuration if needed."""
        return self._lobe1
    
    @property
    def lobe2(self) -> Lobe:
        """Access to Lobe 2 for direct configuration if needed."""
        return self._lobe2
        
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent's state."""
        # Reset both internal lobes
        if hasattr(self, '_lobe1') and self._lobe1:
            await self._lobe1.on_reset(cancellation_token)
        if hasattr(self, '_lobe2') and self._lobe2:
            await self._lobe2.on_reset(cancellation_token)
        
        # Reset initialization flag
        self._initialized = False
        
        # Re-setup the internal team to ensure clean state
        if hasattr(self, '_internal_team'):
            self._setup_internal_team()
        
        logger.info(f"Reset Expert {self.name}")
    
    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        """Message types that this agent can produce."""
        # Expert can produce text messages and potentially stop messages
        return [TextMessage, StopMessage, HandoffMessage]