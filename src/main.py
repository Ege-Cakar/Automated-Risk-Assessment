from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
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

load_dotenv()

logger = logging.getLogger(__name__)

# Debug flag - Set to True to see internal deliberation, False for quiet mode
DEBUG_INTERNAL_DELIBERATION = False

# Usage example with current APIs
async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Read the risk assessment request file
    with open("data/text_files/dummy_req.txt", "r", encoding="utf-8") as file:
        risk_assessment_request = file.read()
        logger.info("Risk assessment request file loaded successfully")

    # Setup vector memory using current API
    vector_memory = LobeVectorMemory(persist_directory="./data/vectordb")
    
    # Create model client using current API
    model_client = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.7
    )
    
    security_expert = Expert(
        name="SecurityExpert",
        model_client=model_client,
        vector_memory=vector_memory,
        system_message="You are a cybersecurity expert specializing in threat analysis.",
        debug=False  # Let team handle debug output
    )
    
    compliance_expert = Expert(
        name="ComplianceExpert", 
        model_client=model_client,
        vector_memory=vector_memory,
        system_message="You are a compliance expert specializing in regulatory requirements.",
        debug=False
    )
    
    architecture_expert = Expert(
        name="ArchitectureExpert",
        model_client=model_client, 
        vector_memory=vector_memory,
        system_message="You are a cloud architecture expert specializing in scalable systems.",
        debug=False
    )
    
    experts = {
        "SecurityExpert": security_expert,
        "ComplianceExpert": compliance_expert,
        "ArchitectureExpert": architecture_expert
    }
    
    # Create team components
    coordinator = Coordinator(model_client, experts, debug=True)
    summary_agent = SummaryAgent(model_client, debug=True)
    
    # Create team
    team = ExpertTeam(
        coordinator=coordinator,
        experts=experts,
        summary_agent=summary_agent,
        max_messages=15,
        debug=True
    )
    
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