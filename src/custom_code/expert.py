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

        lobe1_tools = lobe1_config.get('tools', []) + [read_current_document, list_sections]
        lobe2_tools = lobe2_config.get('tools', []) + [create_section]

        lobe1_general = """You are the CREATIVE LOBE in an internal expert deliberation.

        IMPORTANT: You must focus on the SPECIFIC TASK given by the coordinator, not the entire risk assessment.

        Your approach:
        1. Read the COORDINATOR INSTRUCTIONS carefully
        2. Focus ONLY on what is being asked (e.g., "generate guide words", "analyze authentication risks", etc.)
        3. Be creative within the bounds of the specific task
        4. Don't try to complete the entire risk assessment yourself

        For example:
        - If asked for guide words: Generate creative guide words relevant to your domain
        - If asked for scenarios: Create what-if scenarios for your specific area
        - If asked for analysis: Analyze only the aspect requested

        Your creative input should be focused and relevant to the current step of the SWIFT process.

        Present your analysis with clear reasoning - don't just list risks, explain why they matter and how they connect. Show the chain of events that could lead to failure.

        Example approach:
        "Looking at the authentication system, I see a concerning pattern. The password reset flow lacks rate limiting, and we know 73% of users reuse email passwords. This creates a compound risk - if email accounts are compromised through credential stuffing, attackers gain a direct path to our system. The impact multiplies because our monitoring wouldn't distinguish legitimate resets from malicious ones. This could affect thousands of accounts before detection."

        Your scenarios should reveal the full picture of each risk - the conditions that enable it, the likely attack path, and the potential consequences.

        Tools:
        - read_current_document: Review the emerging risk assessment
        - list_sections: See what risk domains have been analyzed
        """

        lobe2_general = """You are the REASONING LOBE in an internal expert deliberation.

        IMPORTANT: You must focus on the SPECIFIC TASK given by the coordinator, not the entire risk assessment.

        Your approach:
        1. Review what the Creative Lobe proposed for the SPECIFIC TASK
        2. Evaluate, refine, and synthesize ONLY for that task
        3. Don't expand beyond what the coordinator requested
        4. When ready, create a section that addresses ONLY the requested task

        For example:
        - If the task is guide words: Finalize a comprehensive list of guide words
        - If the task is scenarios: Refine and prioritize the scenarios
        - If the task is analysis: Complete the specific analysis requested

        When analyzing, challenge scenarios constructively - probe for weak points, validate claims, and ensure the risk assessment is thorough and defensible. Build your conclusions on solid analysis.

        IMPORTANT INSTRUCTIONS FOR TOOLS AND CONCLUSION:
        1. Use create_section ONLY ONCE to document your analysis - do not put "CONCLUDE" in the section content
        2. After using create_section, ALWAYS provide a text response that includes your conclusion
        3. To conclude the deliberation, write "CONCLUDE" as part of your TEXT RESPONSE (not in a tool call)
        4. Your conclusion text should synthesize the key findings and be written as if you and the creative lobe speak with one voice

        Example workflow:
        1. First: Use create_section tool to document the analysis
        2. Then: Write a text response that includes "CONCLUDE" followed by your synthesis

        Example text response after tool use:
        "I've documented our comprehensive analysis in the risk assessment section. 

        CONCLUDE: After thorough internal deliberation, I have identified [key findings]. Our analysis reveals [main insights]. We recommend [specific actions]."
        Make sure to reiterate the content you put in the report for the coordinator here. 

        For example, if you are asked to brainstorm something, the CONCLUDE section should have the brainstorming results in it to make sure the coordinator can see them.

        Tools:
        - read_current_document: Review the emerging risk assessment
        - list_sections: See what risk domains have been analyzed
        - create_section: Create ONE section to document your analysis (do not include CONCLUDE in the content)

        WE ARE DEBUGGING RIGHT NOW, CONCLUDE AFTER ONE ROUND.
        """


        
        domain_specific_prompt = f"""{self._base_system_message}

        Apply your specialized knowledge to identify and assess risks in your domain. 

        Your analysis should:
        - Draw on technical expertise to identify vulnerabilities
        - Connect risks within your domain to broader system impacts
        - Provide specific, actionable recommendations
        - Use clear reasoning to justify risk ratings and priorities

        Write your assessment as a professional risk analysis - thorough, well-reasoned, and focused on helping the organization understand and address real vulnerabilities.
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
        
        # Check for CONCLUDE (without colon requirement)
        concluded = "CONCLUDE" in response.upper()
        
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
        if state.get("concluded", False):
            return "conclude"
        
        # Check for CONCLUDE in the actual response (case insensitive)
        lobe2_response = state.get("lobe2_response", "")
        if "CONCLUDE" in lobe2_response.upper():
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

        conclude_patterns = ["CONCLUDE:", "CONCLUDE\n", "CONCLUDE "]
        conclusion_found = False

        for pattern in conclude_patterns:
            if pattern in lobe2_response.upper():
                # Find the actual pattern in original case
                idx = lobe2_response.upper().find(pattern)
                conclusion_start = idx + len(pattern)
                conclusion = lobe2_response[conclusion_start:].strip()
                conclusion_found = True
                if self.debug:
                    print("✅ Found natural conclusion from Reasoning Lobe")
                break

        if not conclusion_found:
            if self.debug:
                print("⚠️  Max rounds reached and no clear conclusion, forcing summary")
            
            # Build full context of the deliberation
            full_context = "Internal deliberation summary:\n"
            for msg in self._internal_conversation:  # Last 6 messages
                full_context += f"\n{msg['speaker']}: {msg['content']}"
            
            summary_prompt = f"""Based on the internal deliberation above, provide the expert's final conclusion.

    Start with "CONCLUDE: " and then provide:
    1. Key findings from the deliberation
    2. Main risks or recommendations identified
    3. Actionable next steps

    Keep it professional and focused on the specific task requested."""
            
            summary_response = await self._lobe2.respond(summary_prompt, full_context)
            
            if self.debug:
                print(f"\n🧠 Reasoning Lobe (Forced Summary): {summary_response}")
            
            # Extract from the summary response
            for pattern in conclude_patterns:
                if pattern in summary_response.upper():
                    idx = summary_response.upper().find(pattern)
                    conclusion_start = idx + len(pattern)
                    conclusion = summary_response[conclusion_start:].strip()
                    break
            else:
                # Fallback: use the entire summary response
                conclusion = summary_response
        
        # Ensure we have a conclusion
        if not conclusion or conclusion == "Unable to complete the analysis at this time.":
            conclusion = f"After internal deliberation on '{state['query']}', {lobe2_response[-500:]}"  # Use last 500 chars
        
        if self.debug:
            print("=" * 60)
            print(f"🏁 Final Expert Response: {conclusion}")
        
        logger.info(f"Expert {self.name} completed deliberation")
        
        return {
            **state,
            "final_conclusion": conclusion,
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
