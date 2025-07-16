from typing import List, Dict, Any, Optional, Annotated  
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
import json
import logging
import asyncio
from dotenv import load_dotenv
from src.custom_code.expert import Expert
from src.custom_code.lobe import LobeVectorMemory   
from src.custom_code.coordinator import Coordinator
from src.custom_code.summarizer import SummaryAgent
from src.custom_code.ra_team import ExpertTeam
from src.custom_code.expert_generator import ExpertGenerator
from src.utils.memory import initialize_database
from pathlib import Path


load_dotenv()

logger = logging.getLogger(__name__)

# Debug flag - Set to True to see internal deliberation, False for quiet mode
DEBUG_INTERNAL_DELIBERATION = True

generate_from_scratch = False


# Usage example with current APIs
async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    Path("data/report").mkdir(parents=True, exist_ok=True)
    Path("data/conversations").mkdir(parents=True, exist_ok=True)
    
    report = "data/text_files/report.md"
    
    # Clear report
    with open(report, "w") as f:
        f.write("")


    # Read the risk assessment request file
    with open("data/text_files/dummy_req.txt", "r", encoding="utf-8") as file:
        risk_assessment_request = file.read()
        logger.info("Risk assessment request file loaded successfully")

    # Read the risk assessment request file
    with open("data/text_files/swift_info.md", "r", encoding="utf-8") as file:
        swift_info = file.read()
        logger.info("Swift info file loaded successfully")

    # Read the database info file
    with open("data/text_files/database_info.txt", "r", encoding="utf-8") as file:
        database_info = file.read()
        logger.info("Database info file loaded successfully")

    # Setup vector memory using current API
    vector_memory = await initialize_database()
    
    # Create model client using current API
    model_client = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.7
    )

    thinking_client = ChatOpenAI(
        model="o4-mini",
        temperature=0.7
    )
    

    
    if generate_from_scratch:
        with open("data/text_files/approved_experts.json", "w") as f:
            f.write("[]")
        
        # Create the task request string
        expert_gen_task = f"""Generate a team of experts for risk assessment based on the following:

        User Request: {risk_assessment_request}

        Information on SWIFT steps: {swift_info}

        You will have access to relevant data to help with keyword generation and expert identification. 
        """

        expert_generator = ExpertGenerator(
            model="o4-mini",
            provider="openai",
            min_experts=5,
            max_experts=12
        )

        _ = expert_generator.run_expert_generator(
            user_request=risk_assessment_request,
            swift_details=swift_info, 
            database_info=database_info
        )
    
    with open("data/text_files/approved_experts.json", "r") as f:
        approved_experts = json.load(f)

    # Create experts
    experts = {}
    for expert in approved_experts:
        keywords = expert["keywords"]
        lobe1_config = {
            "keywords": keywords,
            "temperature": 0.8
        }
        lobe2_config = {
            "keywords": keywords,
            "temperature": 0.4
        }
        expert_agent = Expert(
            name=expert["name"].lower().replace(" ", "_").replace("-","_"),
            model_client=model_client,
            vector_memory=vector_memory,
            system_message=expert["system_prompt"],
            lobe1_config=lobe1_config,
            lobe2_config=lobe2_config,
            debug=DEBUG_INTERNAL_DELIBERATION  # Let team handle debug output
        )
        experts[expert["name"]] = expert_agent
    
    # Create team components
    coordinator = Coordinator(model_client, experts, debug=DEBUG_INTERNAL_DELIBERATION, swift_info=swift_info)
    summary_agent = SummaryAgent(model_client, debug=DEBUG_INTERNAL_DELIBERATION)
    
    # Create team
    team = ExpertTeam(
        coordinator=coordinator,
        experts=experts,
        summary_agent=summary_agent,
        max_messages=30,
        debug=DEBUG_INTERNAL_DELIBERATION,
        conversation_path="data/conversations",
    )
    
    png = team.team_graph.get_graph().draw_mermaid_png()
    
    with open("team_graph.png", "wb") as f:
        f.write(png)
    
    if DEBUG_INTERNAL_DELIBERATION:
        print("✅ Expert system initialized!")
        print("\n" + "="*80)
    
    # Process a query
    query = risk_assessment_request 
    response = await team.consult(query)
    
    if DEBUG_INTERNAL_DELIBERATION:
        print("\n" + "="*80)
        print(f"📋 FINAL EXPERT RESPONSE:")
        print("="*80)
    
    print(f"{response}")

if __name__ == "__main__":
    asyncio.run(main())