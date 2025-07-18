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
from langchain_google_genai import ChatGoogleGenerativeAI
import inquirer
from src.utils.system_prompts import EXPERT_EXTRAS


load_dotenv()

logger = logging.getLogger(__name__)

# Debug flag - Set to True to see internal deliberation, False for quiet mode
DEBUG_INTERNAL_DELIBERATION = True

generate_from_scratch = False

async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    Path("data/report").mkdir(parents=True, exist_ok=True)
    Path("data/conversations").mkdir(parents=True, exist_ok=True)
    
    report = "data/text_files/report.md"
    
    # ADDED: Interactive menu
    print("\n" + "="*60)
    print("SWIFT RISK ASSESSMENT SYSTEM")
    print("="*60)
    print("\nSelect operation mode:")
    print("1) Generate from scratch with NEW experts")
    print("2) Generate from scratch with SAVED experts")
    print("3) Generate summary from saved sections.json")
    print("="*60)
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # ADDED: Process choice
    generate_new_experts = (choice == '1')
    run_full_assessment = (choice in ['1', '2'])
    summary_only = (choice == '3')
    
    print(f"\nSelected: Option {choice}")
    print("="*60 + "\n")

    # Clear report
    with open(report, "w") as f:
        f.write("")

    if run_full_assessment:
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
            model="o3"
        )

        thinking_client = ChatOpenAI(
            model="o3"
        )
    
    if generate_new_experts:
        print("🔄 Generating new expert team...")
        with open("data/text_files/approved_experts.json", "w") as f:
            f.write("[]")
        
        # Create the task request string
        expert_gen_task = f"""Generate a team of experts for risk assessment based on the following:

        User Request: {risk_assessment_request}

        Information on SWIFT steps: {swift_info}

        You will have access to relevant data to help with keyword generation and expert identification. 
        """

        expert_generator = ExpertGenerator(
            model="o3",
            provider="openai",
            min_experts=5,
            max_experts=12
        )

        _ = expert_generator.run_expert_generator(
            user_request=risk_assessment_request,
            swift_details=swift_info, 
            database_info=database_info
        )
        print("✅ New expert team generated!")
    
    # main.py - UPDATED OPTION 3 SECTION ONLY

    # ADDED: Option 3 - Summary only from saved sections
    if summary_only:
        print("📊 Generating summary from saved sections...")

        swift_info = ""

        with open("data/text_files/swift_info.md", "r", encoding="utf-8") as file:
            swift_info = file.read()
        
        # Import DocumentManager
        from src.utils.document_manager import DocumentManager
        
        # Load existing document manager
        doc_manager = DocumentManager(base_path="data/report")
        
        if not doc_manager.sections:
            print("❌ No saved sections found in data/report/sections.json")
            print("Please run option 1 or 2 first to generate sections.")
            return
        
        # Create model client without tools binding for direct summary
        model_client = ChatOpenAI(model="o3")
        
        # Build content from saved sections
        messages = []
        expert_responses = {}
        all_sections_content = []
        
        # Process each section
        print("\n📄 Found sections:")
        for section_id, section in doc_manager.sections.items():
            print(f"  - {section.domain} by {section.author} (status: {section.status.value})")
            if section.status.value == "draft":
                expert_name = section.author
                expert_responses[expert_name] = section.content
                messages.append({
                    "speaker": expert_name,
                    "content": section.content
                })
                all_sections_content.append(f"=== {section.domain} (by {expert_name}) ===\n{section.content}")
        
        if not expert_responses:
            print("❌ No merged sections found. Please complete a full assessment first.")
            return
        
        print(all_sections_content)
        
        # Create a direct summary prompt
        summary_prompt = f"""You are the SWIFT Risk Assessment Summary Agent. Based on the expert analyses provided below, as well as information on SWIFT, generate a comprehensive final report.

        Expert Contributions:

        {chr(10).join(all_sections_content)}

        Information on SWIFT:
        {swift_info}
        
        Make sure all details and arguments are preserved. Be comprehensive. Be specific. And be clear in your arguments.

        I want you to have arguments clearly laid out. I want you to be thorough. 

        What you are saying must be logically and argumentatively complete with premises, inferences and conclusions. EVERYTHING YOU SAY MUST BE WELL SUPPORTED, EITHER THROUGH ARGUMENTATION OR EVIDENCE.

        """

        # Generate summary directly
        print("\n🔄 Generating comprehensive summary...")
        
        try:
            response = model_client.invoke([
                {"role": "system", "content": "You are a professional risk assessment summarizer. Create clear, actionable reports."},
                {"role": "user", "content": summary_prompt}
            ])
            
            final_report = response.content
            
            if final_report:
                print("\n" + "="*80)
                print("📋 FINAL SUMMARY REPORT:")
                print("="*80)
                print(final_report)
                
                # Save the summary
                with open("data/text_files/summary_report.md", "w") as f:
                    f.write(final_report)
                print(f"\n✅ Summary saved to data/text_files/summary_report.md")
                
                # Also update the main report
                with open("data/text_files/report.md", "w") as f:
                    f.write(final_report)
                print(f"✅ Main report updated at data/text_files/report.md")
            else:
                print("❌ Failed to generate summary")
                
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            logger.error(f"Summary generation error: {e}", exc_info=True)
        
        return  # Exit early for summary-only mode
    
    if run_full_assessment:
        with open("data/text_files/approved_experts.json", "r") as f:
            approved_experts = json.load(f)

        if not approved_experts:
            print("❌ No approved experts found. Please run option 1 first.")
            return

        print(f"✅ Loaded {len(approved_experts)} experts")

        # Create experts
        experts = {}
        for expert in approved_experts:
            keywords = expert["keywords"]
            lobe1_config = {
                "keywords": keywords,
                "temperature": 1
            }
            lobe2_config = {
                "keywords": keywords,
                "temperature": 1
            }
            expert_agent = Expert(
                name=expert["name"].lower().replace(" ", "_").replace("-","_"),
                model_client=model_client,
                vector_memory=vector_memory,
                system_message=expert["system_prompt"]+"\n\n"+EXPERT_EXTRAS,
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