from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from src.utils.schemas import TeamState
from src.utils.system_prompts import SWIFT_COORDINATOR_PROMPT
import json
import logging
from src.utils.report import read_current_document, list_sections, merge_section

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
        debug: bool = False,
        tools: List[Any] = None
    ):
        self.base_model_client = model_client
        self.experts = experts
        self.debug = debug
        self.tools = tools or [read_current_document, list_sections, merge_section]
        self.system_message = SWIFT_COORDINATOR_PROMPT

        # Bind tools to create a new model client
        self.model_client = model_client.bind_tools(self.tools)
    
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
                f"{msg['speaker']}: {msg['content']}"
                for msg in state["messages"][-20:]
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

You may use the available tools (read_current_document, list_sections, merge_section) to track progress and merge different sections if desired.
Then decide what to do next. Your final response must be valid JSON only."""
        
        # Format system message with expert list
        formatted_system = self.system_message.format(
            expert_list=", ".join(self.experts.keys())
        )
        
        messages = [
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # First invocation - might include tool calls
            response = await self.model_client.ainvoke(messages)
            
            # Handle tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute each tool call
                tool_messages = messages.copy()
                tool_messages.append(response)  # Add AI message with tool calls
                
                for tool_call in response.tool_calls:
                    tool_func = None
                    for tool in self.tools:
                        if tool.name == tool_call['name']:
                            tool_func = tool
                            break
                    
                    if tool_func:
                        try:
                            # Execute the tool
                            result = await tool_func.ainvoke(tool_call['args'])
                            if self.debug:
                                print(f"🔧 Coordinator used {tool_call['name']}: {result}")
                            
                            # Add tool result to messages
                            tool_messages.append({
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": tool_call['id']
                            })
                        except Exception as e:
                            if self.debug:
                                print(f"❌ Coordinator tool error: {e}")
                            tool_messages.append({
                                "role": "tool",
                                "content": f"Error: {str(e)}",
                                "tool_call_id": tool_call['id']
                            })
                
                # Ask for the decision after tool use
                tool_messages.append({
                    "role": "user",
                    "content": "Based on the tool results, please provide your decision in JSON format."
                })
                
                # Get the final decision
                follow_up = await self.model_client.ainvoke(tool_messages)
                
                # Extract JSON from the follow-up response
                if follow_up.content:
                    content = follow_up.content.strip()
                    # Try to find JSON in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        decision_data = json.loads(json_content)
                    else:
                        decision_data = json.loads(content)
                else:
                    raise ValueError("No content in follow-up response")
            else:
                # No tool calls, parse JSON response normally
                content = response.content.strip()
                
                # Try to find JSON in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    decision_data = json.loads(json_content)
                else:
                    decision_data = json.loads(content)
            
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
