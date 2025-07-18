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
        tools: List[Any] = None,
        swift_info: str = ""
    ):
        self.base_model_client = model_client
        self.experts = experts
        self.debug = debug
        self.tools = tools or [read_current_document, list_sections, merge_section]
        self.system_message = SWIFT_COORDINATOR_PROMPT
        self.swift_info = swift_info

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

    You may use the available tools (read_current_document, list_sections, merge_section) to review progress.
    Remember: You CANNOT create content - only direct experts to create it.

    What content needs to be created next? Which expert should create it?

    Your response must be valid JSON only."""
        
        # Format system message with expert list
        formatted_system = self.system_message.format(
            expert_list=", ".join(self.experts.keys()),
            swift_info=self.swift_info
        )
        
        # DEFINE messages HERE - this was missing!
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
                            
                            # Better formatted debug output
                            if self.debug:
                                result_preview = str(result)
                                if len(result_preview) > 200:
                                    result_preview = result_preview[:200] + "..."
                                print(f"🔧 Coordinator used {tool_call['name']}")
                                if tool_call.get('args'):
                                    print(f"   Args: {tool_call['args']}")
                                print(f"   Result: {result_preview}")
                            
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
                
                # Ask for the decision after tool use with clearer instructions
                tool_messages.append({
                    "role": "user",
                    "content": """Based on the tool results above, provide your decision in JSON format.

    You MUST respond with valid JSON containing these fields:
    {{
        "reasoning": "Your analysis of what you learned from the tools and what should happen next",
        "decision": "continue_coordinator" OR "expert_name" OR "summarize",
        "keywords": ["relevant", "keywords", "here"],  // required if handing off to expert
        "instructions": "Clear instructions for the expert"  // required if handing off to expert
    }}

    Example responses:
    - To continue coordinating: {{"reasoning": "I need to merge the approved sections", "decision": "continue_coordinator"}}
    - To hand off: {{"reasoning": "We need guide words created", "decision": "Fire Safety Systems Expert", "keywords": ["guide", "words", "swift"], "instructions": "Please create a comprehensive guide word list for the SWIFT assessment"}}
    - To summarize: {{"reasoning": "All experts have contributed", "decision": "summarize", "keywords": ["summary"], "instructions": "Create final report"}}

    RESPOND ONLY WITH JSON."""
                })
                
                # Get the final decision with retry logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        follow_up = await self.model_client.ainvoke(tool_messages)
                        
                        if not follow_up.content:
                            if retry < max_retries - 1:
                                if self.debug:
                                    print(f"⚠️  Empty response from coordinator, retry {retry + 1}/{max_retries}")
                                continue
                            else:
                                raise ValueError("No content in follow-up response after retries")
                        
                        # Extract JSON from the follow-up response
                        content = follow_up.content.strip()
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        
                        if json_start != -1 and json_end > json_start:
                            json_content = content[json_start:json_end]
                            decision_data = json.loads(json_content)
                            break  # Success, exit retry loop
                        else:
                            decision_data = json.loads(content)
                            break
                            
                    except json.JSONDecodeError as e:
                        if retry < max_retries - 1:
                            if self.debug:
                                print(f"⚠️  JSON parse error, retry {retry + 1}/{max_retries}: {e}")
                            continue
                        else:
                            raise
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
            
            # Validate and fix decision_data
            if decision_data.get("decision") == "continue_coordinator":
                # For continuation, we don't need keywords or instructions
                if "reasoning" not in decision_data:
                    decision_data["reasoning"] = "Continuing coordinator analysis"
            else:
                # For expert handoff or summary, ensure all fields
                if "keywords" not in decision_data:
                    decision_data["keywords"] = state.get("conversation_keywords", [])
                if "instructions" not in decision_data:
                    if decision_data["decision"] == "summarize":
                        decision_data["instructions"] = "Create final comprehensive summary"
                    else:
                        # Better default instructions based on conversation state
                        try:
                            sections = await list_sections.ainvoke({})
                            if not sections or sections == "[]":
                                decision_data["instructions"] = "Please create the initial guide word list for our SWIFT assessment following Step 1 of the methodology"
                            else:
                                decision_data["instructions"] = "Please analyze the risks in your domain and create appropriate content for the SWIFT assessment"
                        except:
                            decision_data["instructions"] = "Please analyze the query and create appropriate content for the risk assessment"
            
            if self.debug:
                print(f"🧠 Coordinator Decision: {decision_data['decision']}")
                print(f"💭 Reasoning: {decision_data['reasoning']}")
                if decision_data.get('keywords'):
                    print(f"🔑 Updated Keywords: {decision_data['keywords']}")
            
            return decision_data
            
        except Exception as e:
            logger.error(f"Coordinator decision error: {e}")
            
            # Better fallback decision with context
            sections_result = "[]"
            try:
                sections_result = await list_sections.ainvoke({})
            except:
                pass
            
            # Determine what stage we're at
            if sections_result == "[]" or not sections_result:
                # No sections yet, need to start with guide words
                return {
                    "reasoning": "No sections created yet - starting with guide word generation",
                    "decision": "Regulatory Compliance & Standards Expert",
                    "keywords": ["guide", "words", "swift", "hazard", "deviation"],
                    "instructions": "Please create a comprehensive guide word list for the SWIFT assessment following Step 1 of the methodology"
                }
            else:
                # We have some sections, continue with next expert
                if len(state["expert_responses"]) < len(self.experts):
                    unused_experts = [name for name in self.experts.keys() 
                                    if name not in state["expert_responses"]]
                    return {
                        "reasoning": f"Error recovery - consulting next expert ({e})",
                        "decision": unused_experts[0],
                        "keywords": state.get("conversation_keywords", ["risk", "assessment", "hazard"]),
                        "instructions": "Please analyze the risks in your domain and contribute to the SWIFT assessment"
                    }
                else:
                    return {
                        "reasoning": "Error recovery - ready to summarize",
                        "decision": "summarize",
                        "keywords": ["summary"],
                        "instructions": "Create final summary"
                    }
