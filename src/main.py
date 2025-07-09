import asyncio
import json
import multiprocessing
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from dotenv import load_dotenv
from typing import Set
from src.utils.expert_gen_utils import CombinedTerminationCondition, ExpertTracker, create_expert_tool, save_expert_tool
from src.text_files.system_prompts import ORGANIZER_PROMPT, CRITIC_PROMPT, SWIFT_COORDINATOR_PROMPT, SUMMARY_AGENT_PROMPT, KEYWORD_GENERATOR_PROMPT, SWIFT_SELECTOR_PROMPT
from src.utils.db_loader import LobeVectorMemory, add_files_from_folder
from src.custom_autogen_code.expert import Expert
from src.utils.paths import VECTORDB_PATH, paths, DOCUMENTS_DIR
from src.utils.db_loader import LobeVectorMemoryConfig
from src.utils.keyword_save import save_keywords, save_keywords_tool
from src.utils.filter import SWIFTStatusFormatter, setup_swift_logging, PeriodicStatusLogger
from autogen_core.models import ModelInfo
from src.utils.save_report import save_report_tool
from autogen_agentchat.conditions import FunctionCallTermination

# Fix multiprocessing issues
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

load_dotenv()

# Storage
approved_experts = []
captured_expert_names: Set[str] = set()

output_file = "src/text_files/approved_experts.json"
generate_from_scratch = False
# Reset output file to empty JSON array
if generate_from_scratch:
    with open(output_file, "w") as f:
        f.write("[]")

base_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash-preview-04-17",
    model_info=ModelInfo(vision=False, function_calling=True, json_output=False, structured_output=False, family="gemini-2.5-flash", reasoning_effort="low")
)
deterministic_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash-preview-04-17",
    model_info=ModelInfo(vision=False, function_calling=True, json_output=False, structured_output=False, family="gemini-2.5-flash", reasoning_effort="low")
)

# base_client = OpenAIChatCompletionClient(
#     model="gpt-4.1",
# )
# deterministic_client = OpenAIChatCompletionClient(
#     model="gpt-4.1",
# )


organizer_agent = AssistantAgent(
    "organizer",
    model_client=base_client,
    tools=[create_expert_tool],
    system_message=ORGANIZER_PROMPT,
)

organizer_critic_agent = AssistantAgent(
    "organizer_critic",
    model_client=base_client,
    tools=[save_expert_tool],
    system_message=CRITIC_PROMPT,
)

keyword_generator_agent = AssistantAgent(
    "keyword_generator_agent",
    model_client=base_client,
    tools=[save_keywords_tool],
    system_message=KEYWORD_GENERATOR_PROMPT,
)

# Initialize ChromaDB memory with custom config     
memory = LobeVectorMemory(
    config=LobeVectorMemoryConfig(
        collection_name="expert_test",
        persistence_path=str(VECTORDB_PATH)  
    )
)


# Global expert tracker
expert_tracker = ExpertTracker()


organizing_team = RoundRobinGroupChat(
    [organizer_agent, organizer_critic_agent], 
    termination_condition=TextMentionTermination(text="EXPERT GENERATION DONE", sources=["organizer"])
)

async def main():
    status_tracker = setup_swift_logging()
    periodic_logger = PeriodicStatusLogger(interval=120) 
    periodic_logger.start(status_tracker)
    try:
        # Add files to memory
        await add_files_from_folder(memory, str(DOCUMENTS_DIR))
        print("Files added to memory")
        
        await organizing_team.reset()  # Reset the team for a new task.
        # async for message in team.run_stream(task="Generate a team of experts for risk assessment."):  # type: ignore
        #     if isinstance(message, TaskResult):
        #         print("Stop Reason:", message.stop_reason)
        #     else:
        #         print(message)
        # or:
        # Read the requirement and SWIFT method info files
        with open("src/text_files/dummy_req.txt", "r") as f:
            dummy_req = f.read()
        
        with open("src/text_files/swift_info.txt", "r") as f:
            swift_info = f.read()
        
        
        if generate_from_scratch:
            # Create the task request string
            expert_gen_task = f"""Generate a team of experts for risk assessment based on the following:

            User Request: {dummy_req}

            Information on SWIFT steps: {swift_info}

            You will have access to relevant data to help with keyword generation and expert identification. 
            """#TODO: Feed data from the database here?
            await Console(organizing_team.run_stream(task=expert_gen_task))  # Stream the messages to the console.


        # await team.close() # No need when using Console wrapper

        # Read text_files/approved_experts.json
        with open("src/text_files/approved_experts.json", "r") as f:
            approved_experts = json.load(f)


        if generate_from_scratch:
            # Generate the list of guide/key words: 
            keyword_gen_task = f"""Generate a list of guide words for SWIFT based on the following:

            User Request: {dummy_req}

            Information on SWIFT steps: {swift_info}
            """
            await Console(keyword_generator_agent.run_stream(task=keyword_gen_task))  # Stream the messages to the console. 

        # Read the src/text_files/keywords.txt file
        with open("src/text_files/keywords.txt", "r") as f:
            swift_guide_words = f.read().splitlines()

        swift_agents = []

        swift_expert_keywords = [
            "SWIFT",
            "Risk Assessment",
            "Risk Management",
            "Financial Crime Compliance",
            "Sanctions Screening",
            "Anti-Money Laundering (AML)",
            "Counter-Terrorist Financing (CTF)",
            "Know Your Customer (KYC)",
            "Cybersecurity",
            "Fraud Detection",
            "Operational Risk",
            "Third-Party Risk Management",
            "Compliance Framework",
            "Regulatory Compliance",
            "Financial Messaging",
            "Secure Financial Communications",
            "Payment Systems",
            "Cross-Border Payments",
            "Securities Settlement",
            "Trade Finance",
            "Data Analytics",
            "Risk Mitigation",
            "Threat Analysis",
            "Vulnerability Assessment",
            "Incident Response",
            "Business Continuity",
            "SWIFT Customer Security Programme (CSP)",
            "SWIFT gpi",
            "ISO 20022",
            "Financial Instrument",
            "Correspondent Banking",
            "Payment Tracking",
            "Transaction Monitoring"
        ]

        swift_lobe1_config = {
            'keywords': swift_expert_keywords,
            'temperature': 0.6,
        }

        swift_lobe2_config = {
            'keywords': swift_expert_keywords,
            'temperature': 0.4,
        }

        swift_termination_condition = FunctionCallTermination(function_name="save_report")

        summary_agent = AssistantAgent(
            name="summary_agent",
            model_client=base_client,
            system_message=SUMMARY_AGENT_PROMPT,
            tools=[save_report_tool]
        )

        experts_to_provide_to_coordinator = []
        for expert in approved_experts:
            experts_to_provide_to_coordinator.append(expert["name"])

        swift_coordinator = AssistantAgent(
            "swift_coordinator",
            model_client=base_client,
            system_message=SWIFT_COORDINATOR_PROMPT.format(guide_words=swift_guide_words, experts=experts_to_provide_to_coordinator),
        )

        swift_agents.append(swift_coordinator)

        swift_agents.append(summary_agent)

        # Loop over approved experts to create an agent for each of them, then assemble a team
        for expert in approved_experts:
            expert_agent = Expert(
                name=expert["name"].lower().replace(" ", "_").replace("-","_"),
                model_client=base_client,
                system_message=f"{expert['system_prompt']}\n\nKeep responses focused and concise (2-3 paragraphs). Always relate back to the specific risk assessment question.",
                vector_memory=memory,
                lobe1_config={
                    'keywords': expert["keywords"],
                    'temperature': 0.6,
                },
                lobe2_config={
                    'keywords': expert["keywords"],
                    'temperature': 0.4,
                }
            )
            swift_agents.append(expert_agent)
            print(f"Added {expert['name']} to team")
        
        # Create the team
        SwiftTeam = SelectorGroupChat(
            participants=swift_agents, 
            model_client=deterministic_client,
            selector_prompt=SWIFT_SELECTOR_PROMPT,
            termination_condition=swift_termination_condition,
            max_turns=60  # Safety net
        )
        
        await Console(SwiftTeam.run_stream(task=dummy_req))  # Stream the messages to the console.
        
    except Exception as e:
        print(f"Error in main function: {e}")
        raise
    finally:
        try:
            if 'memory' in globals():
                import gc
                gc.collect()
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")


asyncio.run(main())