import asyncio
import json
import multiprocessing
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from dotenv import load_dotenv
from typing import Set
from utils.expert_gen_utils import CombinedTerminationCondition, ExpertTracker, create_expert_tool, save_expert_tool
from text_files.system_prompts import ORGANIZER_PROMPT, CRITIC_PROMPT, SWIFT_COORDINATOR_PROMPT
from utils.db_loader import LobeVectorMemory, add_files_from_folder
from custom_autogen_code.expert import Expert
from utils.paths import paths
from utils.db_loader import LobeVectorMemoryConfig

# Fix multiprocessing issues
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

load_dotenv()

# Storage
approved_experts = []
captured_expert_names: Set[str] = set()

output_file = "./text_files/approved_experts.json"
# Reset output file to empty JSON array
with open(output_file, "w") as f:
    f.write("[]")


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
memory = LobeVectorMemory(
    config=LobeVectorMemoryConfig(
        collection_name="expert_test",
        persistence_path=paths.VECTORDB_PATH
    )
)


# Global expert tracker
expert_tracker = ExpertTracker()


organizing_team = RoundRobinGroupChat(
    [organizer_agent, organizer_critic_agent], 
    termination_condition=TextMentionTermination(text="EXPERT GENERATION DONE", sources=["organizer"])
)

async def main():
    try:
        await organizing_team.reset()  # Reset the team for a new task.
        # async for message in team.run_stream(task="Generate a team of experts for risk assessment."):  # type: ignore
        #     if isinstance(message, TaskResult):
        #         print("Stop Reason:", message.stop_reason)
        #     else:
        #         print(message)
        # or:
        # Read the requirement and SWIFT method info files
        with open("./text_files/dummy_req.txt", "r") as f:
            dummy_req = f.read()
        
        with open("./text_files/swift_info.txt", "r") as f:
            swift_info = f.read()
        
        # Add files to memory
        db_file_path = "../database"
        await add_files_from_folder(memory, db_file_path)

        print("Files added to memory")
        
        # Create the task request string
        task = f"""Generate a team of experts for risk assessment based on the following:

        User Request: {dummy_req}

        Information on SWIFT steps: {swift_info}

        You will have access to relevant data to help with keyword generation and expert identification.
        """
        await Console(organizing_team.run_stream(task=task))  # Stream the messages to the console.

        # await team.close() # No need when using Console wrapper

        # Read text_files/approved_experts.json
        with open("./text_files/approved_experts.json", "r") as f:
            approved_experts = json.load(f)

        swift_agents = []

        swift_coordinator = Expert(
            "swift_coordinator",
            model_client=base_client,
            system_message=SWIFT_COORDINATOR_PROMPT,
            
        )

        swift_agents.append(swift_coordinator)

        # Loop over approved experts to create an agent for each of them, then assemble a team
        for expert in approved_experts:
            expert_agent = AssistantAgent(
                expert["name"],
                model_client=base_client,
                system_message=expert["system_prompt"],
            )
            print(expert_agent.name)
            swift_agents.append(expert_agent)
        
        # Create the team
        SwiftTeam = SelectorGroupChat(participants=swift_agents, model_client=base_client, termination_condition=TextMentionTermination(text="SWIFT TEAM DONE"))
        
        await Console(SwiftTeam.run_stream(task="Generate a team of experts for risk assessment."))  # Stream the messages to the console.
        
    except Exception as e:
        print(f"Error in main function: {e}")
        raise
    finally:
        # Clean up resources
        try:
            if 'memory' in globals():
                # Force cleanup of memory resources
                import gc
                gc.collect()
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
    

asyncio.run(main())