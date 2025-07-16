"""
Hub-and-spoke architecture where Coordinator controls all expert interactions:
- Coordinator decides which expert speaks next
- Expert does internal deliberation, returns to Coordinator  
- Coordinator updates keywords and manages conversation flow, sends to another expert
- Summary Agent synthesizes final report when needed
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import logging
from src.utils.schemas import TeamState
from src.custom_code.summarizer import SummaryAgent
from src.custom_code.coordinator import Coordinator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

class ExpertTeam:
    """
    Main team orchestrator using LangGraph for state management.
    Implements hub-and-spoke architecture with coordinator control.
    """
    
    def __init__(
        self,
        coordinator: Coordinator,
        experts: Dict[str, Any],  # Expert instances
        summary_agent: SummaryAgent,
        max_messages: int = 20,
        recursion_limit: int = 75,
        debug: bool = False
    ):
        self.coordinator = coordinator
        self.experts = experts
        self.summary_agent = summary_agent
        self.max_messages = max_messages
        self.debug = debug
        self.recursion_limit = recursion_limit

        # Build the team graph
        self.team_graph = self._build_team_graph()
    
    def _build_team_graph(self) -> StateGraph:
        """Build the LangGraph state machine for team coordination"""

        workflow = StateGraph(TeamState)

        # Add core nodes
        workflow.add_node("coordinator", self._coordinator_decide)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("finalize", self._finalize)

        # One node per expert
        for expert_name in self.experts:
            workflow.add_node(expert_name, self._expert_deliberate)
            # Each expert returns to coordinator after speaking
            workflow.add_edge(expert_name, "coordinator")

        # Entry point
        workflow.add_edge(START, "coordinator")

        # Build routing map dynamically: each expert name maps to itself
        route_map = {name: name for name in self.experts}
        route_map["summarize"] = "generate_summary"

        workflow.add_conditional_edges(
            "coordinator",
            self._route_after_coordinator,
            route_map,
        )

        # Summary → finalize → END
        workflow.add_edge("generate_summary", "finalize")
        workflow.add_edge("finalize", END)
    
        return workflow.compile()
    
    async def _coordinator_decide(self, state: TeamState) -> TeamState:
        """Coordinator decides next action"""
        decision_data = await self.coordinator.decide_next_action(state)
        
        return {
            **state,
            "coordinator_decision": decision_data["decision"],
            "coordinator_instructions": decision_data["instructions"],
            "conversation_keywords": decision_data.get("keywords", state.get("conversation_keywords", [])),
            "messages": state["messages"] + [{
                "speaker": "Coordinator",
                "content": f"Decision: {decision_data['decision']} | Reasoning: {decision_data['reasoning']}"
            }]
        }
    
    async def _expert_deliberate(self, state: TeamState) -> TeamState:
        """Run expert deliberation and return to coordinator"""
        expert_name = state["coordinator_decision"]
        expert = self.experts[expert_name]
        
        # Update expert keywords if provided
        if state.get("conversation_keywords"):
            await expert.update_keywords(
                lobe1_keywords=state["conversation_keywords"],
                lobe2_keywords=state["conversation_keywords"]
            )
        
        if self.debug:
            print(f"\n🔄 {expert_name} starting deliberation...")
        
        # Build team conversation context (without internal deliberations)
        team_context = f"User Query: {state['query']}\n\n"
        
        for msg in state["messages"]:
            speaker = msg["speaker"]
            content = msg["content"]
            
            if speaker == "Coordinator":
                # Clean up coordinator messages
                if "Decision:" in content and "Reasoning:" in content:
                    reasoning_part = content.split("Reasoning:")[1].strip()
                    team_context += f"Coordinator: {reasoning_part}\n\n"
                else:
                    team_context += f"Coordinator: {content}\n\n"
            else:
                # Expert final responses only
                team_context += f"{speaker}: {content}\n\n"
        
        # Get the current instruction from coordinator
        current_instruction = ""
        for msg in reversed(state["messages"]):
            if msg["speaker"] == "Coordinator" and "Reasoning:" in msg["content"]:
                current_instruction = msg["content"].split("Reasoning:")[1].strip()
                break
        
        # Get expert response with team context
        expert_response = await expert.process_message(current_instruction, team_context)
        
        return {
            **state,
            "expert_responses": {**state["expert_responses"], expert_name: expert_response},
            "message_count": state["message_count"] + 1,
            "messages": state["messages"] + [{
                "speaker": expert_name,
                "content": expert_response
            }],
            "current_speaker": "Coordinator"
        }

        
    async def _generate_summary(self, state: TeamState) -> TeamState:
        """Generate final summary"""
        final_report = await self.summary_agent.generate_summary(state)
        
        return {
            **state,
            "final_report": final_report,
            "concluded": True,
            "messages": state["messages"] + [{
                "speaker": "SummaryAgent", 
                "content": final_report
            }]
        }
    
    async def _finalize(self, state: TeamState) -> TeamState:
        """Finalize the conversation"""
        if self.debug:
            print(f"\n🏁 Team consultation completed!")
            print(f"📊 Total messages: {state['message_count']}")
            print(f"👥 Experts consulted: {list(state['expert_responses'].keys())}")
        
        return {**state, "concluded": True}
    
    def _route_after_coordinator(self, state: TeamState) -> str:
        """Return the next node key based on coordinator decision"""
        decision = state["coordinator_decision"]
        if decision == "summarize":
            return "summarize"
        elif decision == "end":
            return "finalize"  # triggers END via finalize node
        else:
            # coordinator returns the expert name directly; fallback to first expert
            return decision if decision in self.experts else next(iter(self.experts))
    
    async def consult(self, query: str) -> str:
        """Main method to run team consultation"""
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"🖋️ SWIFT RISK ASSESSMENT STARTING")
            print(f"{'='*80}")
            print(f"📋 Query: {query}")
            print(f"👥 Available Experts: {list(self.experts.keys())}")
            print(f"⏱️  Max Messages: {self.max_messages}")
        
        # Initialize team state
        initial_state: TeamState = {
            "messages": [],
            "query": query,
            "current_speaker": "Coordinator",
            "conversation_keywords": [],
            "expert_responses": {},
            "message_count": 0,
            "max_messages": self.max_messages,
            "concluded": False,
            "coordinator_decision": "",
            "final_report": "",
            "debug": self.debug
        }
        
        try:
            # Run the team consultation
            final_state = await self.team_graph.ainvoke(initial_state, {"recursion_limit": self.recursion_limit})
            
            # Return final report or last expert response
            result = final_state.get("final_report", "No summary generated")
            
            if self.debug:
                print(f"\n{'='*80}")
                print(f"📋 FINAL TEAM RESPONSE:")
                print(f"{'='*80}")
            
            return result
            
        except Exception as e:
            logger.error(f"Team consultation error: {e}", exc_info=True)
            error_msg = f"Team consultation encountered an error: {str(e)}"
            if self.debug:
                print(f"\n❌ {error_msg}")
            return error_msg