from typing import Dict, Any
from langchain_openai import ChatOpenAI
from src.utils.schemas import TeamState
from src.utils.system_prompts import SWIFT_COORDINATOR_PROMPT
import json
import logging

logger = logging.getLogger(__name__)

class Coordinator:
    """
    Central coordinator that manages expert selection and conversation flow.
    Acts as the hub in hub-and-spoke architecture.
    """
    
    def __init__(
        self,
        model_client: ChatOpenAI,
        experts: Dict[str, Any],  # Will be Expert instances
        debug: bool = False
    ):
        self.model_client = model_client
        self.experts = experts
        self.debug = debug
        
        self.system_message = SWIFT_COORDINATOR_PROMPT
    
    async def decide_next_action(self, state: TeamState) -> Dict[str, Any]:
        """Coordinator decides the next action"""
        
        if self.debug:
            print(f"\n🎯 Coordinator analyzing conversation (Message {state['message_count']}/{state['max_messages']})")
        
        # Check message limit
        if state["message_count"] >= state["max_messages"]:
            if self.debug:
                print("⏰ Message limit reached - forcing summarize")
            return {
                "reasoning": "Message limit reached",
                "decision": "summarize", 
                "keywords": ["summary", "conclusion"],
                "instructions": "Create final comprehensive summary of the risk assessment according to your instructions."
            }
        
        # Build conversation context
        conversation_summary = ""
        if state["messages"]:
            conversation_summary = "\n".join([
                f"{msg['speaker']}: {msg['content']}" # Do not truncate messages, at least for now
                for msg in state["messages"][-10:]  # Last 10 messages for now, I'm sure current frontier models can handle more
            ])
        
        expert_status = ""
        for expert_name in self.experts.keys():
            contribution_count = sum(1 for msg in state["messages"] if msg["speaker"] == expert_name)
            status = f"Contributed {contribution_count} time(s)" if contribution_count > 0 else "Not consulted"
            expert_status += f"- {expert_name}: {status}\n"
        
        prompt = f"""Original Query: {state['query']}

Current Keywords: {state.get('conversation_keywords', [])}

Expert Status:
{expert_status}

Recent Conversation:
{conversation_summary}

Available Experts: {list(self.experts.keys())}

Decide what to do next. Respond with valid JSON only."""
        
        # Format system message with expert list
        formatted_system = self.system_message.format(
            expert_list=", ".join(self.experts.keys())
        )
        
        messages = [
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.model_client.ainvoke(messages)
            decision_data = json.loads(response.content.strip())
            
            if self.debug:
                print(f"🧠 Coordinator Decision: {decision_data['decision']}")
                print(f"💭 Reasoning: {decision_data['reasoning']}")
                if decision_data.get('keywords'):
                    print(f"🔑 Updated Keywords: {decision_data['keywords']}")
            
            return decision_data
            
        except Exception as e:
            logger.error(f"Coordinator decision error: {e}")
            # Fallback decision
            if len(state["expert_responses"]) < len(self.experts):
                unused_experts = [name for name in self.experts.keys() 
                                if name not in state["expert_responses"]]
                return {
                    "reasoning": "Fallback - consulting unused expert",
                    "decision": unused_experts[0],
                    "keywords": state.get("conversation_keywords", []),
                    "instructions": "Please analyze the query"
                }
            else:
                return {
                    "reasoning": "Fallback - ready to summarize",
                    "decision": "summarize",
                    "keywords": ["summary"],
                    "instructions": "Create final summary"
                }

