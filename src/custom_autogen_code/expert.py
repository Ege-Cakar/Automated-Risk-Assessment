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
        max_rounds: int = 3,
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
        # Default configurations
        lobe1_config = lobe1_config or {}
        lobe2_config = lobe2_config or {}

        lobe1_general = """You are the CREATIVE LOBE in an internal expert deliberation.

        Your job: Generate creative scenarios and ideas for your reasoning counterpart to validate.

        Internal conversation pattern:
        1. You receive a request/question from the main expert
        2. You generate 2-3 creative scenarios or ideas related to the request
        3. You present ONE scenario at a time to your reasoning counterpart
        4. You wait for their assessment before presenting the next scenario
        5. After 2-3 exchanges, your reasoning counterpart will CONCLUDE

        Example flow:
        You: "I'll analyze [request]. Scenario 1: What if [creative idea]..."
        Reasoning: "I assess this as [analysis]. Next scenario?"
        You: "Scenario 2: Consider if [different angle]..."
        Reasoning: "This has [assessment]. Continue..."
        You: "Final scenario: What about [third perspective]..."
        Reasoning: "CONCLUDE: Based on our analysis..."

        IMPORTANT: 
        - You're talking to your internal reasoning counterpart, not external experts
        - Present scenarios one at a time
        - Wait for assessment before continuing
        - Be creative but realistic
        """

        lobe2_general = """You are the REASONING LOBE in an internal expert deliberation.

        Your job: Validate and analyze scenarios from your creative counterpart, then synthesize conclusions.

        Internal conversation pattern:
        1. Your creative counterpart presents scenarios one at a time
        2. You assess each scenario for validity, likelihood, impact
        3. You ask for the next scenario until you have enough to analyze
        4. You end with "CONCLUDE:" followed by your final synthesized analysis

        Example flow:
        Creative: "Scenario 1: What if [idea]..."
        You: "I assess this as [risk level] because [reasoning]. Next scenario?"
        Creative: "Scenario 2: Consider if [idea 2]..."
        You: "This has [likelihood/impact] due to [analysis]. Continue..."
        Creative: "Final scenario: What about [idea 3]..."
        You: "CONCLUDE: Based on analyzing these scenarios, [final synthesis with recommendations]"

        CRITICAL RULES:
        - Always assess each scenario before asking for next
        - Always end with "CONCLUDE:" after analyzing multiple scenarios
        - Your CONCLUDE response becomes the expert's final answer
        - Be thorough but concise in your final conclusion
        """            
        
        domain_specific_prompt = f"""{self._base_system_message}

        DOMAIN EXPERTISE: Apply your specialized knowledge to the internal deliberation.
        
        When your creative lobe presents scenarios, evaluate them using your domain expertise.
        When your reasoning lobe asks for assessments, provide domain-specific risk analysis.
        
        The final CONCLUDE statement should reflect your expert domain knowledge and provide
        actionable recommendations specific to your area of expertise.
        """
        # If not coordinator use domain specific
        lobe1_full_message = f"{domain_specific_prompt}\n\n{lobe1_general}"
        lobe2_full_message = f"{domain_specific_prompt}\n\n{lobe2_general}"

        lobe1_tools = lobe1_config.get('tools', [])
        lobe2_tools = lobe2_config.get('tools', [])
        
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
        # Handle agent selection/probing phase (what Swarm does internally)
        if not messages:
            logger.info(f"Agent selection probe for {self.name}")
            # Don't start internal deliberation for probes - just indicate readiness
            return Response(
                chat_message=TextMessage(
                    content="",  # Empty response for selection phase
                    source=self.name
                )
            )
        
        # Filter for actual content messages (like AssistantAgent does)
        meaningful_messages = []
        for msg in messages:
            if isinstance(msg, TextMessage) and msg.content and msg.content.strip():
                meaningful_messages.append(msg)
        
        if not meaningful_messages:
            logger.info(f"No meaningful content for {self.name}")
            return Response(
                chat_message=TextMessage(
                    content="Please provide a specific query to analyze.",
                    source=self.name
                )
            )
        
        # Now proceed with actual processing
        logger.info(f"Processing actual content for {self.name}")
        
        # Reset pending handoff
        self._pending_handoff = None
        
        # Ensure lobes are initialized
        await self._initialize_lobes()
        
        # Use the last meaningful message
        last_message = meaningful_messages[-1]
        
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
            
            hit_max_rounds = not any(
                isinstance(msg, TextMessage) and 
                msg.source == self._lobe2.name and 
                "CONCLUDE:" in msg.content
                for msg in result.messages
            )
        
            if hit_max_rounds:
                # Force a summary from Voice of Reason
                logger.info(f"Max rounds reached, requesting summary from Voice of Reason")
                summary_prompt = f"""You've reached the maximum deliberation rounds. 
                            
                Based on the entire discussion you've had with your creative counterpart, provide a final summary.

                Start with "CONCLUDE: After extensive internal deliberation," and then:
                1. Summarize the main ideas explored
                2. Highlight key assessments made
                3. Provide actionable insights despite not reaching a natural conclusion
                4. Keep it professional and valuable for the end user"""
                            
                # Get summary from Voice of Reason
                summary_messages = [TextMessage(content=summary_prompt, source="system")]
                summary_response = await self._lobe2.on_messages(
                    messages=summary_messages,
                    cancellation_token=cancellation_token
                )
                
                # Add the summary to our results
                result.messages.append(summary_response.chat_message)
            
            # Extract and format the conclusion
            final_response = self._extract_conclusion(result)
            
            logger.info(f"Expert {self.name} completed deliberation")
            
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
        # Look for CONCLUDE: from Lobe 2 (should always be there now)
        for message in reversed(task_result.messages):
            if isinstance(message, TextMessage) and message.source == self._lobe2.name:
                content = message.content   
                if "CONCLUDE:" in content:
                    # Extract everything after "CONCLUDE:"
                    conclusion_start = content.find("CONCLUDE:") + len("CONCLUDE:")
                    conclusion = content[conclusion_start:].strip()
                    return conclusion
        
        # This should rarely happen now
        logger.error(f"No conclusion found for Expert {self.name}")
        return "Unable to complete the analysis at this time."
    
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
        return [TextMessage, StopMessage]