from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from src.utils.memory import LobeVectorMemory   
from langgraph.graph import StateGraph, START, END
from src.utils.schemas import ExpertState
from src.custom_code.lobe import Lobe
import logging

logger = logging.getLogger(__name__)

class Expert:
    """Updated Expert class using current LangGraph patterns"""
    
    def __init__(
        self,
        name: str,
        model_client: ChatOpenAI,
        vector_memory: LobeVectorMemory,
        system_message: str = None,
        lobe1_config: Dict[str, Any] = None,
        lobe2_config: Dict[str, Any] = None,
        max_rounds: int = 3,
        description: str = "An expert agent that internally deliberates using two specialized lobes.",
        debug: bool = False,  # Toggleable debug output
        **kwargs
    ):
        self.name = name
        self._model_client = model_client
        self._vector_memory = vector_memory
        self._max_rounds = max_rounds
        self.description = description
        self.debug = debug  # Store debug flag
        
        self._base_system_message = system_message if system_message else (
            "You are an expert assistant with deep knowledge in your domain. "
            "You think carefully and provide well-reasoned responses."
        )
        
        # Default configurations - same prompts as AutoGen
        lobe1_config = lobe1_config or {}
        lobe2_config = lobe2_config or {}
        
        lobe1_general = """You are the CREATIVE LOBE in an internal expert deliberation.

        Your job: Generate creative scenarios and ideas for your reasoning counterpart to validate.

        Internal conversation pattern:
        1. You receive a request/question from the main expert
        2. You generate creative scenarios or ideas related to the request
        3. You present one scenario at a time to your reasoning counterpart
        4. You wait for their assessment before presenting the next scenario
        5. After a few exchanges, your reasoning counterpart will CONCLUDE

        Example flow:
        You: "I'll analyze [request]. Scenario 1: What if [creative idea]..."
        Reasoning: "I assess this as [analysis]. Next scenario?"
        You: "Scenario 2: Consider if [different angle]..."
        Reasoning: "This has [assessment]. Continue..."
        You: "What about [third perspective]..."

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
        2. You assess each scenario for validity, likelihood, impact, risk. Try not to overestimate or underestimate. 
        3. You ask for the next scenario until you have enough to analyze
        4. You end with "CONCLUDE:" followed by your final synthesized analysis

        Example flow:
        Creative: "Scenario 1: What if [idea]..."
        You: "I assess this as [risk level] because [reasoning]. Next scenario?"
        Creative: "Scenario 2: Consider if [idea 2]..."
        You: "This has [likelihood/impact] due to [analysis]. Continue..."
        Creative: "What about [idea 3]..."
        You: "CONCLUDE: Based on analyzing these scenarios, [final synthesis with recommendations]"

        CRITICAL RULES:
        - Always assess each scenario before asking for next
        - Always end with "CONCLUDE:" after analyzing multiple scenarios
        - Your CONCLUDE response becomes the expert's final answer, so answer as if the deliberation between you and the creative counterpart was done by one person.
        - Be thorough but concise in your final conclusion
        - You must always finish with handing things back to the swift_coordinator.
        """
        
        domain_specific_prompt = f"""{self._base_system_message}

        DOMAIN EXPERTISE: Apply your specialized knowledge to the internal deliberation.
        
        When your creative lobe presents scenarios, evaluate them using your domain expertise.
        When your reasoning lobe asks for assessments, provide domain-specific risk analysis.
        
        The final CONCLUDE statement should reflect your expert domain knowledge and provide
        actionable recommendations specific to your area of expertise.
        """
        
        lobe1_full_message = f"{domain_specific_prompt}\n\n{lobe1_general}"
        lobe2_full_message = f"{domain_specific_prompt}\n\n{lobe2_general}"
        
        # Create lobes using current APIs
        self._lobe1 = Lobe(
            name=f"{name}_Creative",
            model_client=model_client,
            vector_memory=vector_memory,
            keywords=lobe1_config.get('keywords', []),
            temperature=lobe1_config.get('temperature', 0.8),
            system_message=lobe1_full_message,
            tools=lobe1_config.get('tools', [])
        )
        
        self._lobe2 = Lobe(
            name=f"{name}_VoReason",
            model_client=model_client,
            vector_memory=vector_memory,
            keywords=lobe2_config.get('keywords', []),
            temperature=lobe2_config.get('temperature', 0.4),
            system_message=lobe2_full_message,
            tools=lobe2_config.get('tools', [])
        )
        
        # Build the internal deliberation graph using current LangGraph patterns
        self._internal_graph = self._build_internal_graph()
        self._initialized = False
    
    def _build_internal_graph(self) -> StateGraph:
        workflow = StateGraph(ExpertState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_deliberation)
        workflow.add_node("lobe1_respond", self._lobe1_respond)
        workflow.add_node("lobe2_respond", self._lobe2_respond)
        workflow.add_node("extract_conclusion", self._extract_conclusion)
        
        # Set entry point using current API
        workflow.add_edge(START, "initialize")
        
        # Define edges using current patterns
        workflow.add_edge("initialize", "lobe1_respond")
        workflow.add_conditional_edges(
            "lobe1_respond",
            self._should_continue_after_lobe1,
            {
                "lobe2": "lobe2_respond",
                "conclude": "extract_conclusion"
            }
        )
        workflow.add_conditional_edges(
            "lobe2_respond", 
            self._should_continue_after_lobe2,
            {
                "lobe1": "lobe1_respond",
                "conclude": "extract_conclusion"
            }
        )
        workflow.add_edge("extract_conclusion", END)
        
        return workflow.compile()
    
    async def _initialize_deliberation(self, state: ExpertState) -> ExpertState:
        """Initialize the internal deliberation"""
        if not self._initialized:
            await self._lobe1.initialize_context()
            await self._lobe2.initialize_context()
            self._initialized = True
            logger.info(f"Initialized both lobes for Expert {self.name}")
        
        if self.debug:
            print(f"\n🔄 Starting internal deliberation for Expert {self.name}")
            print(f"📋 Query: {state['query']}")
            print(f"🎯 Max rounds: {state.get('max_rounds', 3)}")
            print("=" * 60)
        
        return {
            **state,
            "messages": state.get("messages", []) + ["Starting internal deliberation..."],
            "iteration_count": 0,
            "concluded": False
        }
    
    async def _lobe1_respond(self, state: ExpertState) -> ExpertState:
        """Creative lobe responds"""
        context = f"Previous discussion: {state.get('lobe2_response', '')}" if state.get("lobe2_response") else ""
        
        response = await self._lobe1.respond(state["query"], context)
        
        # Show the internal deliberation only if debug is enabled
        if self.debug:
            print(f"\n🎨 Creative Lobe ({self.name}): {response}")
        logger.info(f"Lobe 1 (Creative) responded for Expert {self.name}")
        
        return {
            **state,
            "lobe1_response": response,
            "messages": state.get("messages", []) + [f"Creative: {response}"],
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    
    async def _lobe2_respond(self, state: ExpertState) -> ExpertState:
        """Reasoning lobe responds"""
        context = f"Creative lobe said: {state.get('lobe1_response', '')}"
        
        response = await self._lobe2.respond(state["query"], context)
        
        # Check for CONCLUDE: - mirrors AutoGen termination condition
        concluded = "CONCLUDE:" in response
        
        # Show the internal deliberation only if debug is enabled
        if self.debug:
            if concluded:
                print(f"\n🧠 Reasoning Lobe ({self.name}): {response}")
                print(f"\n✅ Expert {self.name} deliberation CONCLUDED")
            else:
                print(f"\n🧠 Reasoning Lobe ({self.name}): {response}")
        
        logger.info(f"Lobe 2 (Reasoning) responded for Expert {self.name}")
        
        return {
            **state,
            "lobe2_response": response,
            "messages": state.get("messages", []) + [f"Reasoning: {response}"],
            "concluded": concluded
        }
    
    def _should_continue_after_lobe1(self, state: ExpertState) -> str:
        """Decide next step after lobe1"""
        if state.get("iteration_count", 0) >= state.get("max_rounds", 3) * 2:
            return "conclude"
        return "lobe2"
    
    def _should_continue_after_lobe2(self, state: ExpertState) -> str:
        """Decide next step after lobe2"""
        if state.get("concluded", False) or "CONCLUDE:" in state.get("lobe2_response", ""):
            return "conclude"
        if state.get("iteration_count", 0) >= state.get("max_rounds", 3) * 2:
            return "conclude"
        return "lobe1"
    
    async def _extract_conclusion(self, state: ExpertState) -> ExpertState:
        """Extract final conclusion"""
        conclusion = ""
        
        if self.debug:
            print(f"\n🎯 Extracting final conclusion for Expert {self.name}")
        
        # Look for CONCLUDE: in lobe2 response
        lobe2_response = state.get("lobe2_response", "")
        if "CONCLUDE:" in lobe2_response:
            conclusion_start = lobe2_response.find("CONCLUDE:") + len("CONCLUDE:")
            conclusion = lobe2_response[conclusion_start:].strip()
            if self.debug:
                print("✅ Found natural conclusion from Reasoning Lobe")
        else:
            # Force summary if max rounds reached
            if self.debug:
                print("⚠️  Max rounds reached, forcing summary from Voice of Reason")
            logger.info(f"Max rounds reached, requesting summary from Voice of Reason")
            summary_prompt = """You've reached the maximum deliberation rounds. 
                        
            Based on the entire discussion you've had with your creative counterpart, provide a final summary.

            Start with "CONCLUDE: After extensive internal deliberation," and then:
            1. Summarize the main ideas explored
            2. Highlight key assessments made
            3. Provide actionable insights despite not reaching a natural conclusion
            4. Keep it professional and valuable for the end user"""
            
            summary_response = await self._lobe2.respond(summary_prompt, "")
            if self.debug:
                print(f"\n🧠 Reasoning Lobe (Forced Summary): {summary_response}")
            
            if "CONCLUDE:" in summary_response:
                conclusion_start = summary_response.find("CONCLUDE:") + len("CONCLUDE:")
                conclusion = summary_response[conclusion_start:].strip()
            else:
                conclusion = "Unable to complete the analysis at this time."
        
        if self.debug:
            print("=" * 60)
            print(f"🏁 Final Expert Response: {conclusion}")
        logger.info(f"Expert {self.name} completed deliberation")
        
        return {
            **state,
            "final_conclusion": conclusion,
            "concluded": True
        }
    
    async def process_message(self, query: str) -> str:
        """Process a message using current LangGraph API"""
        if self.debug:
            print(f"\n🚀 Expert {self.name} received message: '{query}'")
        logger.info(f"Expert {self.name} received a message")
        
        # Create initial state using current pattern
        initial_state: ExpertState = {
            "messages": [],
            "query": query,
            "lobe1_response": "",
            "lobe2_response": "",
            "final_conclusion": "",
            "iteration_count": 0,
            "max_rounds": self._max_rounds,
            "concluded": False,
            "vector_context": ""
        }
        
        try:
            # Run internal deliberation graph using current API
            logger.info(f"Starting internal deliberation for Expert {self.name}")
            final_state = await self._internal_graph.ainvoke(initial_state)
            
            conclusion = final_state.get("final_conclusion", "No conclusion reached")
            if self.debug:
                print(f"\n🎉 Expert {self.name} completed processing!")
            return conclusion
            
        except Exception as e:
            logger.error(f"Error in Expert {self.name} deliberation: {str(e)}", exc_info=True)
            error_msg = (
                f"I encountered an error during internal deliberation. "
                f"Let me provide a direct response instead: Based on the query about "
                f"'{query}', I need more specific information to provide a detailed answer."
            )
            if self.debug:
                print(f"\n❌ Error in Expert {self.name}: {error_msg}")
            return error_msg
    
    async def update_keywords(self, lobe1_keywords: List[str] = None, lobe2_keywords: List[str] = None):
        """Update keywords for lobes"""
        if lobe1_keywords is not None:
            await self._lobe1.update_keywords(lobe1_keywords)
            logger.info(f"Updated Lobe 1 keywords for Expert {self.name}")
            
        if lobe2_keywords is not None:
            await self._lobe2.update_keywords(lobe2_keywords)
            logger.info(f"Updated Lobe 2 keywords for Expert {self.name}")
    
    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Add knowledge to vector database"""
        await self._vector_memory.add(content, metadata)
        logger.info(f"Added knowledge to Expert {self.name}'s shared database")
    
    @property
    def lobe1(self) -> Lobe:
        """Access to Lobe 1"""
        return self._lobe1
    
    @property
    def lobe2(self) -> Lobe:
        """Access to Lobe 2"""
        return self._lobe2
