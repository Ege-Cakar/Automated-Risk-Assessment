import autogen
import json
from dotenv import load_dotenv
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.memory.chroma import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from pydantic import BaseModel
from typing import List, Dict, Any

load_dotenv()

# Configuration for the LLM (replace with your actual configuration)
config_list = [
    {
        "model": "gpt-4",
    }
]

# 1. Define the Agents
proposer = autogen.ConversableAgent(
    name="Proposer",
    llm_config={"config_list": config_list},
    system_message="You are a creative agent. Propose a short, interesting fact.",
)

approver = autogen.ConversableAgent(
    name="Approver",
    llm_config={"config_list": config_list},
    system_message="You are a meticulous editor. If the proposed fact is good, respond with only the word 'APPROVE'. Otherwise, provide a reason for rejection.",
)

# A place to store the approved facts
structured_document = []
output_file = "approved_facts.json"

# 2. Define a Custom Reply Function
def add_to_document_if_approved(recipient, messages, sender, config):
    """
    This function is called when the 'approver' is about to reply.
    It checks if the last message from the 'proposer' was approved and, if so,
    adds it to our structured document.
    """
    # Get the last message sent
    last_message = messages[-1]

    # Check if the approver's response will be "APPROVE"
    if "APPROVE" in last_message.get("content", "").upper():
        print(f"\n--- Approval received. Writing to document. ---")

        # The message to be approved is the one before the approver's message
        message_to_approve = messages[-2]
        
        # Add the content of the proposer's message to our list
        structured_document.append({"fact": message_to_approve.get("content", "")})

        # Write the updated list to a JSON file
        with open(output_file, "w") as f:
            json.dump(structured_document, f, indent=4)
        
        print(f"--- Successfully wrote to {output_file} ---")

    return False, None  # Continue the normal chat flow

# 3. Register the Reply Function
approver.register_reply(
    trigger=proposer,  # Trigger this function when the approver is replying to the proposer
    reply_func=add_to_document_if_approved,
    config={},
)

# 4. Set up the Group Chat
groupchat = autogen.GroupChat(agents=[proposer, approver], messages=[], max_round=4)
# 5. Initiate the Chat
proposer.initiate_chat(
    groupchat,
    message="Propose a fascinating fact about ancient Rome.",
)

# Display the final structured document
print("\nFinal content of the structured document:")
print(structured_document)