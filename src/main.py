import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Set
from utils.expert_gen_utils import CombinedTerminationCondition, ExpertTracker, create_expert_tool, save_expert_tool
from text_files.system_prompts import ORGANIZER_PROMPT, CRITIC_PROMPT

load_dotenv()

# Storage
approved_experts = []
captured_expert_names: Set[str] = set()
output_file = "text_files/approved_experts.json"

organization_client = OpenAIChatCompletionClient(
    model="o4-mini",
)

base_client = OpenAIChatCompletionClient(
    model="o4-mini",
)

organizer_agent = AssistantAgent(
    "organizer",
    model_client=organization_client,
    tools=[create_expert_tool],
    system_message=ORGANIZER_PROMPT,
)

organizer_critic_agent = AssistantAgent(
    "organizer_critic",
    model_client=base_client,
    tools=[save_expert_tool],
    system_message=CRITIC_PROMPT,
)

# Initialize ChromaDB memory with custom config
relevant_file_database = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="Submitted Files",
        persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
        k=10,  # Return top  k results
        score_threshold=0.4,  # Minimum similarity score
    )
)
# Global expert tracker
expert_tracker = ExpertTracker()


team = RoundRobinGroupChat(
    [organizer_agent, organizer_critic_agent], 
    termination_condition=CombinedTerminationCondition(expert_tracker=expert_tracker)
)

async def main():
    await team.reset()  # Reset the team for a new task.
    # async for message in team.run_stream(task="Generate a team of experts for risk assessment."):  # type: ignore
    #     if isinstance(message, TaskResult):
    #         print("Stop Reason:", message.stop_reason)
    #     else:
    #         print(message)
    # or:
    # Read the requirement and SWIFT method info files
    with open("text_files/dummy_req.txt", "r") as f:
        dummy_req = f.read()
    
    with open("text_files/swift_info.txt", "r") as f:
        swift_info = f.read()
    
    # Create the task request string
    task = f"""Generate a team of experts for risk assessment based on the following:

User Request: {dummy_req}

Information on SWIFT steps: {swift_info}

You will have access to relevant data to help with keyword generation and expert identification.
"""
    await Console(team.run_stream(task=task))  # Stream the messages to the console.

    await team.close()

asyncio.run(main())