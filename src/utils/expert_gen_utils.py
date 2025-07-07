import json
from pathlib import Path
import os
from typing import List, Dict, Any, Sequence, Annotated
from autogen_agentchat.conditions import FunctionalTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from .response_types import Expert, OrganizationResponse
from autogen_core.tools import FunctionTool


class CombinedTerminationCondition(FunctionalTermination):
    """
    A termination condition class that checks for organizer's "DONE"
    and critic's "APPROVED" message. It also handles resetting state.
    Inherits from FunctionalTermination to integrate with autogen's termination logic.
    """
    def __init__(self, expert_tracker: 'ExpertTracker'):
        self.expert_tracker = expert_tracker
        super().__init__(func=self.is_terminated)

    def is_terminated(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> bool:
        # Process any new messages for expert tracking
        self.expert_tracker.process_new_messages(messages)
        
        if len(messages) < 2:
            return False
        
        # Look for the most recent organizer message containing "DONE"
        latest_done_index = -1
        
        for i, message in enumerate(messages):
            if (hasattr(message, 'source') and message.source == "organizer" and 
                hasattr(message, 'content')):
                
                content = str(message.content).upper()
                
                # Check if message contains "DONE" (not a tool response)
                if "DONE" in content and not hasattr(message.content, 'done'):
                    latest_done_index = i
        
        # If organizer never said "DONE", we can't terminate  
        if latest_done_index == -1:
            return False
        
        # Look for critic "APPROVED" AFTER the most recent "DONE"
        for i in range(latest_done_index + 1, len(messages)):
            message = messages[i]
            if (hasattr(message, 'source') and message.source == "organizer_critic" and
                hasattr(message, 'content')):
                
                critic_content = str(message.content).upper()
                if "APPROVED" in critic_content:
                    print(f"TERMINATION: Found DONE at index {latest_done_index}, APPROVED at index {i}")
                    return True
        
        print(f"TERMINATION: Found DONE at index {latest_done_index}, but no APPROVED after it")
        return False


    async def reset(self):
        """Resets the expert_tracker state."""
        #await super().reset()
        self.expert_tracker.reset()

class ExpertTracker:
    """Tracks experts throughout the conversation and captures approved ones."""
    
    def __init__(self):
        self.pending_experts = {}  # Maps message_id -> expert_data
        self.approved_experts = []
        self.output_file = "approved_experts.json"
        self.processed_message_ids = set()  # Track which messages we've already processed

    def reset(self):
        """Resets the tracker's state for a new run."""
        self.pending_experts = {}
        self.processed_message_ids = set()
        print("ExpertTracker state has been reset.")
    
    def process_new_messages(self, all_messages):
        """Process any new messages since last call."""
        for i, message in enumerate(all_messages):
            message_id = f"{i}_{getattr(message, 'source', 'unknown')}"
            
            if message_id in self.processed_message_ids:
                continue
                
            self.processed_message_ids.add(message_id)
            
            # Check if this is an organizer message with an expert
            if (hasattr(message, 'source') and message.source == "organizer" and 
                hasattr(message, 'content')):
                
                expert_data = self._extract_expert_from_message(message)
                if expert_data:
                    self.pending_experts[message_id] = expert_data
                    print(f"Found new expert proposal: {expert_data.name}")
            
            # Check if this is a critic approval
            elif (hasattr(message, 'source') and message.source == "organizer critic" and
                  hasattr(message, 'content') and "APPROVED" in message.content.upper()):
                
                # Find the most recent pending expert to approve
                if self.pending_experts:
                    # Get the most recent expert (highest message_id)
                    recent_expert_id = max(self.pending_experts.keys(), 
                                         key=lambda x: int(x.split('_')[0]))
                    expert_data = self.pending_experts.pop(recent_expert_id)
                    
                    self._save_expert(expert_data)
                    print(f"Approved and saved expert: {expert_data.name}")
    
    def _extract_expert_from_message(self, message):
        """Extract expert data from organizer message."""
        try:
            content = message.content
            
            # Since organizer uses OrganizationResponse model, content should have 'response' field
            if hasattr(content, 'response'):
                return content.response
            elif isinstance(content, dict) and 'response' in content:
                response_data = content['response']
                if isinstance(response_data, dict):
                    return Expert(**response_data)
                else:
                    return response_data
            
            return None
            
        except Exception as e:
            print(f"Error extracting expert from message: {e}")
            return None

    def _save_expert(self, expert_data):
        """Save expert to the approved experts list and file."""
        # Check for duplicates
        existing_names = [exp["name"] for exp in self.approved_experts]
        if expert_data["name"] in existing_names:
            print(f"Skipping duplicate expert: {expert_data['name']}")
            return
        
        # Add to approved list
        self.approved_experts.append(expert_data)
        
        # Write all experts to file
        with open(self.output_file, "w") as f:
            json.dump(self.approved_experts, f, indent=4)
        
        print(f"Saved expert {expert_data['name']} to {self.output_file}")


def create_expert_response(
    thoughts: Annotated[str, "Your reasoning about the expert"],
    expert_name: Annotated[str, "Name of the expert"],
    expert_system_prompt: Annotated[str, "Detailed system prompt for the expert"],
    expert_keywords: Annotated[list, "List of keywords relevant to the expert"]
    ) -> dict:
        """Create a structured expert response"""
        return {
            "thoughts": thoughts,
            "response": {
                "name": expert_name,
                "system_prompt": expert_system_prompt,
                "keywords": expert_keywords
            }
        }
    
create_expert_tool = FunctionTool(
    name="create_expert_response",
    func=create_expert_response,
    description="Create a structured expert object."
)
    
def func_save_expert(
    expert_name: Annotated[str, "Name of the expert"],
    expert_system_prompt: Annotated[str, "Detailed system prompt for the expert"],
    expert_keywords: Annotated[list, "List of keywords relevant to the expert"]
) -> dict:
    """Save an expert to the structured document"""
    output_file = "src/text_files/approved_experts.json"
    
    # Load existing experts or create new list
    existing_experts = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_experts = json.load(f)
                # Handle case where file contains single expert dict instead of list
                if isinstance(existing_experts, dict):
                    existing_experts = [existing_experts]
        except (json.JSONDecodeError, FileNotFoundError):
            existing_experts = []
    
    # Check for duplicates
    existing_names = [exp.get("name") for exp in existing_experts if isinstance(exp, dict)]
    if expert_name in existing_names:
        print(f"Expert {expert_name} already exists, skipping...")
        return {"status": "duplicate", "name": expert_name}
    
    # Add new expert
    new_expert = {
        "name": expert_name,
        "system_prompt": expert_system_prompt,
        "keywords": expert_keywords
    }
    
    existing_experts.append(new_expert)
    
    # Write back to file
    with open(output_file, "w") as f:
        json.dump(existing_experts, f, indent=4)
    
    print(f"Saved expert {expert_name} to {output_file}")
    return new_expert

save_expert_tool = FunctionTool(
    name="save_expert",
    func=func_save_expert,
    description="Save an expert to file."
    )