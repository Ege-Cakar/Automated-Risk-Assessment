from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from src.utils.memory import LobeVectorMemory   
from langgraph.graph import StateGraph, START, END
from src.utils.schemas import ExpertState
from src.custom_code.lobe import Lobe
from src.utils.report import create_section, read_current_document, list_sections, propose_edit
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
        lobe3_config: Dict[str, Any] = None,
        max_rounds: int = 4,
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

        self._internal_conversation = []
        self._team_conversation_context = ""
        
        self._base_system_message = system_message if system_message else (
            "You are an expert assistant with deep knowledge in your domain. "
            "You think carefully and provide well-reasoned responses."
        )
        
        default_tools = [create_section, read_current_document, list_sections, propose_edit]
        # Default configurations - same prompts as AutoGen
        lobe1_config = lobe1_config or {}
        lobe2_config = lobe2_config or {}
        lobe3_config = lobe3_config or {}

        lobe1_tools = lobe1_config.get('tools', []) + [read_current_document, list_sections]
        lobe2_tools = lobe2_config.get('tools', []) + [create_section]
        lobe3_tools = lobe3_config.get('tools', []) + []

        lobe1_general = """You are the CREATIVE LOBE in an internal expert deliberation.

        Your role: Generate innovative risk perspectives through structured dialogue with the reasoning lobe.

        DELIBERATION PROCESS:
        1. Start with creative proposals for the SPECIFIC task
        2. Present your ideas with initial reasoning
        3. Explicitly invite critique: "What gaps do you see?" "How would this fail?"
        4. Build on feedback iteratively
        5. Continue until you both agree the analysis is comprehensive

        ARGUMENTATION APPROACH:
        Even in creative mode, structure your thinking:
        - "Given [observation], I hypothesize [risk scenario] because [reasoning]"
        - "This could lead to [consequence] through [mechanism]"

        Example opening for keyword generation:
        "For authentication keywords, I propose:
        - 'BYPASS' - Premise: Attackers seek path of least resistance. Inference: They'll target recovery flows. Conclusion: Password reset is a critical vector.
        - 'SPOOF' - Premise: Users trust familiar interfaces...
        What other attack patterns should we consider? Are there systemic vulnerabilities I'm missing?"

        Tools:
        - read_current_document: Review existing assessment
        - list_sections: Check coverage"""


        lobe2_general = """You are the REASONING LOBE in an internal expert deliberation.

        Your role: Analyze, structure, and synthesize the creative input into a comprehensive response.

        DELIBERATION PROCESS:
        1. Critically examine each creative proposal
        2. Add systematic analysis and structure
        3. Identify gaps and expand coverage
        4. Continue dialogue until truly comprehensive
        5. Synthesize the COMPLETE analysis for the coordinator

        ARGUMENTATION RIGOR:
        Transform creative insights into structured arguments:
        - Validate premises: "Your bypass scenario assumes X, which is valid because..."
        - Strengthen inferences: "Additionally, this connects to Y through mechanism Z"
        - Expand conclusions: "This implies we also need to consider..."

        SYNTHESIS REQUIREMENTS:
        When ready to conclude (after thorough deliberation):
        1. Use create_section ONCE with the FULL collaborative analysis
        2. Write "CONCLUDED" to let the summarizer lobe, who will take over from there, know that you are finished
        3. Include ALL content requested (actual keywords, scenarios, etc.)
        4. Present clear argument chains for each item

        Remember: The coordinator needs the ACTUAL deliverables with full reasoning. Also remember to always include CONCLUDED in the final response. 

        IF YOU DON'T INCLUDE CONCLUDED IN YOUR FINAL RESPONSE, THEN THE COORDINATOR WILL NOT SEE YOUR FINAL RESPONSE AND WILL NOT BE ABLE TO USE IT. YOU WILL BE MESSING THINGS UP!

        Tools:
        - read_current_document: Review context
        - list_sections: Check existing work
        - create_section: Document FINAL synthesized analysis"""


        lobe3_general = """

        You are the Summarizer.

        • You read the *entire* internal deliberation that has already reached the
        CONCLUDED signal that was given to you.

        • Your only goal is to turn the material you see into a cohesive, polished response from the voice of one expert, in first person. 
        – Preserve all factual / argumentative content.  
        – Fix ordering, markdown, headings, spacing.  
        – Speak in **first-person singular** (“I …”).  
        – Do NOT add new analysis or reopen discussion.

        Return ONLY the polished text that should be shown to the coordinator, who you are responding to –
        no commentary, no wrappers.

        """

        
        domain_specific_prompt = f"""{self._base_system_message}

        Apply your specialized knowledge to identify and assess risks in your domain. 

        Your analysis should:
        - Draw on technical expertise to identify vulnerabilities
        - Connect risks within your domain to broader system impacts
        - Provide specific, actionable recommendations
        - Use clear reasoning to justify risk ratings and priorities

        Write your assessment as a professional - thorough, well-reasoned, and focused on helping the organization understand and address real vulnerabilities.
        """
        
        lobe1_full_message = f"{domain_specific_prompt}\n\n{lobe1_general}"
        lobe2_full_message = f"{domain_specific_prompt}\n\n{lobe2_general}"
        lobe3_full_message = f"{lobe3_general}"
        
        # Create lobes using current APIs
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
            temperature=lobe2_config.get('temperature', 0.4),
            system_message=lobe2_full_message,
            tools=lobe2_tools
        )

        self._lobe3 = Lobe(
            name=f"{name}_Reporter",
            model_client=model_client,
            vector_memory=vector_memory,
            system_message=lobe3_full_message,
            tools=lobe3_tools
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
        workflow.add_node("lobe3_respond", self._lobe3_respond)

        
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

        workflow.add_edge("extract_conclusion", "lobe3_respond")
        workflow.add_edge("lobe3_respond", END)
        
        return workflow.compile()
    
    async def _initialize_deliberation(self, state: ExpertState) -> ExpertState:
        """Initialize the internal deliberation"""
        if not self._initialized:
            await self._lobe1.initialize_context()
            await self._lobe2.initialize_context()
            self._initialized = True
            logger.info(f"Initialized both lobes for Expert {self.name}")
        
        self._internal_conversation = []

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
        # Build context for creative lobe
        context = state.get("team_context", "")
        
        # Add internal deliberation history
        for msg in self._internal_conversation:
            context += f"\n--{msg['speaker']}{' (YOU)' if msg['speaker'].endswith('Creative') else ''}: {msg['content']}"
        
        response = await self._lobe1.respond(state["query"], context)
        
        # Add to internal conversation
        self._internal_conversation.append({
            "speaker": f"{self.name}_Creative", 
            "content": response
        })
        
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
        """Reasoning lobe responds - can speak after tool use"""
        # Build context for reasoning lobe
        context = state.get("team_context", "")
        
        # Add internal deliberation history
        for msg in self._internal_conversation:
            context += f"\n--{msg['speaker']}{' (YOU)' if msg['speaker'].endswith('VoReason') else ''}: {msg['content']}"
        
        response = await self._lobe2.respond(state["query"], context)
        
        # Check if there was a tool call in the response
        tool_used = "Tool" in response# and "result:" in response
        
        # If a tool was used, extract the result and add follow-up
        if tool_used:
            # Check if there's meaningful content after the tool result
            lines = response.strip().split('\n')
            tool_section_ended = False
            follow_up_lines = []
            
            for line in lines:
                if tool_section_ended:
                    follow_up_lines.append(line)
                elif line.strip() == "" and "Result:" in '\n'.join(lines[:lines.index(line)]):
                    tool_section_ended = True
            
            follow_up_text = '\n'.join(follow_up_lines).strip()
            
            if len(follow_up_text) < 50:  # Not enough follow-up
                # Add the tool response to conversation first
                self._internal_conversation.append({
                    "speaker": f"{self.name}_VoReason",
                    "content": response
                })
                
                # Now ask for analysis
                analysis_context = context + f"\n--{self.name}_VoReason: {response}"
                analysis_prompt = "Based on the tool result above, please provide your analysis and either continue the discussion or conclude with your findings."
                
                follow_up_response = await self._lobe2.respond(analysis_prompt, analysis_context)
                response = f"{response}\n\n{follow_up_response}"
                
                # Don't add to conversation again, will be added below
                self._internal_conversation.pop()  # Remove the duplicate
        
        # Add complete response to internal conversation
        self._internal_conversation.append({
            "speaker": f"{self.name}_VoReason",
            "content": response
        })
        
        # Check for RESPONSE (without colon requirement)
        concluded = "RESPONSE" in response.upper()
        
        if self.debug:
            if concluded:
                print(f"\n🧠 Reasoning Lobe ({self.name}): {response}")
                print(f"\n✅ Expert {self.name} deliberation concluded.")
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
        lobe2_response = state.get("lobe2_response", "")

        # ───── early-exit hooks ─────
        done = (
            "CONCLUDED" in lobe2_response.upper()              # plain token
            or "CONCLUDE" in lobe2_response.upper()
            or "CONCLUDE:" in lobe2_response.upper()  
            or "CONCLUDE\n" in lobe2_response.upper()
            or "CONCLUDE " in lobe2_response.upper()
            or "CONCLUDE." in lobe2_response.upper()
        )
        if done:
            return "conclude"
        # ────────────────────────────────

        if state.get("iteration_count", 0) >= state.get("max_rounds", 3) * 2:
            return "conclude"
        return "lobe1"
    
    async def _extract_conclusion(self, state: ExpertState) -> ExpertState:
        return {
            **state,
            "conversation": self._internal_conversation,
        }

    async def _lobe3_respond(self, state: ExpertState) -> ExpertState:
        conversation = state.get("conversation", "")
        deliberation_log = "\n".join(
            f"{m['speaker']}: {m['content']}" for m in conversation
        )

        prompt = (
            "You are the REPORTER-LOBE.\n"
            "Your teammates finished their discussion and signalled CONCLUDED.\n\n"
            "─── FULL DELIBERATION (do NOT quote verbatim) ───\n"
            f"{deliberation_log}\n\n"
            "Task: Write a single, polished answer in first-person SINGULAR that\n"
            "captures every substantive point, arranges them logically(you must have a single voice), and\n"
            "meets the Coordinator's deliverable requirements.\n\n"
            "Return ONLY the finished section (no preamble like 'Here is the …')."
        )

        final_conclusion = await self._lobe3.respond(prompt)

        return {
            **state,
            "final_conclusion": final_conclusion,
            "concluded": True
        }
        
    async def process_message(self, query: str, team_context: str = "") -> str:
        """Process a message with team conversation context"""
        if self.debug:
            print(f"\n🚀 Expert {self.name} received message")
        logger.info(f"Expert {self.name} received a message")
        
        # Store team context for lobes to access
        self._team_conversation_context = team_context
        
        # Create initial state
        initial_state: ExpertState = {
            "messages": [],
            "query": query,  # This is the current instruction
            "team_context": team_context,  # Full conversation history
            "lobe1_response": "",
            "lobe2_response": "",
            "final_conclusion": "",
            "iteration_count": 0,
            "max_rounds": self._max_rounds,
            "concluded": False,
            "vector_context": ""
        }
        
        try:
            # Run internal deliberation
            logger.info(f"Starting internal deliberation for Expert {self.name}")
            final_state = await self._internal_graph.ainvoke(initial_state)
            
            conclusion = final_state.get("final_conclusion", "No conclusion reached")
            if self.debug:
                print(f"\n🎉 Expert {self.name} completed processing!")
            return conclusion
            
        except Exception as e:
            logger.error(f"Error in Expert {self.name} deliberation: {str(e)}", exc_info=True)
            return f"I encountered an error during internal deliberation."
    
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
