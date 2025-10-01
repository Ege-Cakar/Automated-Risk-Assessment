"""
Different team topology implementations for testing routing logic.

Topologies:
1. Star/Hub-and-Spoke: All communication goes through a central coordinator
2. Deep/Sequential: Experts communicate in a chain (Expert1 -> Expert2 -> Expert3 -> Summary)
"""
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from src.utils.schemas import TeamState
from src.custom_code.expert import Expert
from src.custom_code.coordinator import Coordinator
from src.custom_code.summarizer import SummaryAgent
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DeepTopologyTeam:
    """
    Deep/Sequential topology where experts communicate in a linear chain.
    Expert1 -> Expert2 -> Expert3 -> Summary -> End
    No central coordinator; each expert passes directly to the next.
    """
    
    def __init__(
        self,
        experts: Dict[str, Expert],
        summary_agent: SummaryAgent,
        expert_order: Optional[List[str]] = None,
        max_messages: int = 10,
        debug: bool = False,
        conversation_path: str = "experiments/results",
    ):
        self.experts = experts
        self.summary_agent = summary_agent
        self.expert_order = expert_order or list(experts.keys())
        self.max_messages = max_messages
        self.debug = debug
        self.conversation_path = conversation_path
        
        Path(self.conversation_path).mkdir(parents=True, exist_ok=True)
        self.conversation_id = f"deep_topology_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build the team graph
        self.team_graph = self._build_deep_topology_graph()
    
    def _build_deep_topology_graph(self) -> StateGraph:
        """Build a linear/sequential graph where experts communicate in order."""
        workflow = StateGraph(TeamState)
        
        # Add initialization node
        workflow.add_node("initialize", self._initialize_state)
        
        # Add nodes for each expert in order
        for i, expert_name in enumerate(self.expert_order):
            workflow.add_node(expert_name, self._create_expert_node(expert_name))
        
        # Add summary node
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("finalize", self._finalize)
        
        # Build linear chain: START -> initialize -> expert1 -> expert2 -> ... -> summary -> finalize -> END
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", self.expert_order[0])
        
        # Connect experts in sequence
        for i in range(len(self.expert_order) - 1):
            workflow.add_edge(self.expert_order[i], self.expert_order[i + 1])
        
        # Last expert connects to summary
        workflow.add_edge(self.expert_order[-1], "generate_summary")
        workflow.add_edge("generate_summary", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _initialize_state(self, state: TeamState) -> TeamState:
        """Initialize the state for deep topology."""
        if self.debug:
            print(f"\n{'='*80}")
            print(f"🔗 DEEP TOPOLOGY STARTING")
            print(f"{'='*80}")
            print(f"📋 Query: {state.get('query', 'N/A')}")
            print(f"👥 Expert Order: {self.expert_order}")
            print(f"{'='*80}\n")
        
        return {
            **state,
            "current_speaker": self.expert_order[0],
            "message_count": 0,
        }
    
    def _create_expert_node(self, expert_name: str):
        """Create an async function for an expert node."""
        async def expert_node(state: TeamState) -> TeamState:
            """Run expert deliberation in the chain."""
            expert = self.experts[expert_name]
            
            if self.debug:
                print(f"\n{'='*60}")
                print(f"🧠 {expert_name} deliberating...")
                print(f"{'='*60}")
            
            # Get expert's response
            query = state.get("query", "")
            response = await expert.deliberate(query)
            
            # Update state
            new_messages = state.get("messages", []) + [{
                "speaker": expert_name,
                "content": response
            }]
            
            # Determine next speaker (next in chain or summary)
            current_idx = self.expert_order.index(expert_name)
            next_speaker = (
                self.expert_order[current_idx + 1] 
                if current_idx + 1 < len(self.expert_order) 
                else "Summary"
            )
            
            if self.debug:
                print(f"✅ {expert_name} completed")
                print(f"➡️  Next: {next_speaker}\n")
            
            return {
                **state,
                "messages": new_messages,
                "current_speaker": next_speaker,
                "message_count": state.get("message_count", 0) + 1,
                "expert_responses": {
                    **state.get("expert_responses", {}),
                    expert_name: response
                }
            }
        
        return expert_node
    
    async def _generate_summary(self, state: TeamState) -> TeamState:
        """Generate final summary from all expert responses."""
        if self.debug:
            print(f"\n{'='*60}")
            print(f"📝 Generating Summary...")
            print(f"{'='*60}")
        
        # Collect all expert responses
        expert_responses = state.get("expert_responses", {})
        conversation = "\n\n".join([
            f"{expert}: {response}" 
            for expert, response in expert_responses.items()
        ])
        
        # Generate summary
        summary = await self.summary_agent.generate_summary(conversation)
        
        if self.debug:
            print(f"✅ Summary generated")
            print(f"{'='*60}\n")
        
        return {
            **state,
            "final_report": summary,
            "concluded": True,
            "current_speaker": "Summary"
        }
    
    async def _finalize(self, state: TeamState) -> TeamState:
        """Finalize the conversation."""
        if self.debug:
            print(f"\n{'='*80}")
            print(f"✅ DEEP TOPOLOGY COMPLETED")
            print(f"{'='*80}\n")
        
        return state
    
    async def consult(self, query: str) -> str:
        """Main method to run team consultation."""
        initial_state: TeamState = {
            "messages": [],
            "query": query,
            "current_speaker": "Initialize",
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
            final_state = await self.team_graph.ainvoke(initial_state, {"recursion_limit": 50})
            result = final_state.get("final_report", "No summary generated")
            return result
        except Exception as e:
            logger.error(f"Error in deep topology: {e}", exc_info=True)
            return f"Error: {str(e)}"


class StarTopologyTeam:
    """
    Star/Hub-and-Spoke topology (existing architecture).
    All communication goes through a central coordinator.
    Coordinator -> Expert1 -> Coordinator -> Expert2 -> Coordinator -> Summary
    
    This is essentially a wrapper around the existing ExpertTeam for comparison.
    """
    
    def __init__(
        self,
        coordinator: Coordinator,
        experts: Dict[str, Expert],
        summary_agent: SummaryAgent,
        max_messages: int = 10,
        debug: bool = False,
        conversation_path: str = "experiments/results",
    ):
        self.coordinator = coordinator
        self.experts = experts
        self.summary_agent = summary_agent
        self.max_messages = max_messages
        self.debug = debug
        self.conversation_path = conversation_path
        
        Path(self.conversation_path).mkdir(parents=True, exist_ok=True)
        self.conversation_id = f"star_topology_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build the team graph
        self.team_graph = self._build_star_topology_graph()
    
    def _build_star_topology_graph(self) -> StateGraph:
        """Build hub-and-spoke graph where all communication goes through coordinator."""
        workflow = StateGraph(TeamState)
        
        # Add core nodes
        workflow.add_node("coordinator", self._coordinator_decide)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("finalize", self._finalize)
        
        # Add expert nodes
        for expert_name in self.experts:
            workflow.add_node(expert_name, self._expert_deliberate)
            # Each expert returns to coordinator
            workflow.add_edge(expert_name, "coordinator")
        
        # Entry point
        workflow.add_edge(START, "coordinator")
        
        # Build routing map
        route_map = {name: name for name in self.experts}
        route_map["summarize"] = "generate_summary"
        route_map["continue_coordinator"] = "coordinator"
        
        workflow.add_conditional_edges(
            "coordinator",
            self._route_after_coordinator,
            route_map,
        )
        
        # Summary -> finalize -> END
        workflow.add_edge("generate_summary", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _coordinator_decide(self, state: TeamState) -> TeamState:
        """Coordinator decides next action."""
        if self.debug:
            print(f"\n{'='*60}")
            print(f"🎯 Coordinator deciding...")
            print(f"{'='*60}")
        
        # Use coordinator to decide
        decision_dict = await self.coordinator.decide_next_action(state)
        
        if self.debug:
            print(f"📋 Decision: {decision_dict.get('decision', 'N/A')}")
            print(f"💭 Reasoning: {decision_dict.get('reasoning', 'N/A')}")
            print(f"{'='*60}\n")
        
        return {
            **state,
            "coordinator_decision": decision_dict.get("decision", "summarize"),
            "conversation_keywords": decision_dict.get("keywords", []),
            "current_speaker": "Coordinator",
            "messages": state.get("messages", []) + [{
                "speaker": "Coordinator",
                "content": decision_dict.get("reasoning", "")
            }]
        }
    
    async def _expert_deliberate(self, state: TeamState) -> TeamState:
        """Run expert deliberation."""
        expert_name = state.get("current_speaker", "")
        expert = self.experts.get(expert_name)
        
        if not expert:
            logger.error(f"Expert {expert_name} not found")
            return state
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"🧠 {expert_name} deliberating...")
            print(f"{'='*60}")
        
        query = state.get("query", "")
        response = await expert.deliberate(query)
        
        if self.debug:
            print(f"✅ {expert_name} completed")
            print(f"{'='*60}\n")
        
        return {
            **state,
            "messages": state.get("messages", []) + [{
                "speaker": expert_name,
                "content": response
            }],
            "message_count": state.get("message_count", 0) + 1,
            "expert_responses": {
                **state.get("expert_responses", {}),
                expert_name: response
            }
        }
    
    def _route_after_coordinator(self, state: TeamState) -> str:
        """Route based on coordinator decision."""
        decision = state.get("coordinator_decision", "summarize")
        
        if decision == "continue_coordinator":
            return "continue_coordinator"
        elif decision == "summarize":
            return "summarize"
        else:
            # Assume it's an expert name
            return decision if decision in self.experts else "summarize"
    
    async def _generate_summary(self, state: TeamState) -> TeamState:
        """Generate final summary."""
        if self.debug:
            print(f"\n{'='*60}")
            print(f"📝 Generating Summary...")
            print(f"{'='*60}")
        
        expert_responses = state.get("expert_responses", {})
        conversation = "\n\n".join([
            f"{expert}: {response}" 
            for expert, response in expert_responses.items()
        ])
        
        summary = await self.summary_agent.generate_summary(conversation)
        
        if self.debug:
            print(f"✅ Summary generated")
            print(f"{'='*60}\n")
        
        return {
            **state,
            "final_report": summary,
            "concluded": True
        }
    
    async def _finalize(self, state: TeamState) -> TeamState:
        """Finalize the conversation."""
        if self.debug:
            print(f"\n{'='*80}")
            print(f"✅ STAR TOPOLOGY COMPLETED")
            print(f"{'='*80}\n")
        
        return state
    
    async def consult(self, query: str) -> str:
        """Main method to run team consultation."""
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
            final_state = await self.team_graph.ainvoke(initial_state, {"recursion_limit": 50})
            result = final_state.get("final_report", "No summary generated")
            return result
        except Exception as e:
            logger.error(f"Error in star topology: {e}", exc_info=True)
            return f"Error: {str(e)}"
