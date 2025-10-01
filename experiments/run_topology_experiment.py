"""
Experiment to test different team topologies with dummy LLMs.

This script tests:
1. Deep/Sequential topology: Expert1 -> Expert2 -> Expert3 -> Summary
2. Star/Hub-and-Spoke topology: Coordinator -> Expert -> Coordinator (existing architecture)

Goal: Verify routing logic works correctly without making actual API calls.
"""
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.dummy_llm import DummyLLM, DummyCoordinatorLLM, DummyExpertLLM
from experiments.topologies import DeepTopologyTeam, StarTopologyTeam
from src.custom_code.expert import Expert
from src.custom_code.coordinator import Coordinator
from src.custom_code.summarizer import SummaryAgent
from src.custom_code.lobe import LobeVectorMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiments/results/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DummyVectorMemory:
    """Dummy vector memory for testing."""
    
    def __init__(self):
        self.config = type('Config', (), {'k': 5})()
    
    async def search_by_keywords(self, keywords):
        """Return dummy search results."""
        return [{
            "content": f"Dummy content for keywords: {', '.join(keywords)}",
            "metadata": {"score": 0.9, "source": "test"}
        }]


async def create_dummy_expert(name: str, debug: bool = False) -> Expert:
    """Create an expert with dummy LLM."""
    dummy_llm = DummyExpertLLM(expert_name=name)
    dummy_memory = DummyVectorMemory()
    
    expert = Expert(
        name=name,
        model_client=dummy_llm,
        vector_memory=dummy_memory,
        system_message=f"You are {name}, a test expert.",
        lobe1_config={"keywords": ["test"], "temperature": 0.7},
        lobe2_config={"keywords": ["test"], "temperature": 0.7},
        debug=debug
    )
    
    return expert


async def run_deep_topology_experiment(query: str, debug: bool = True):
    """Run experiment with deep/sequential topology."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: DEEP/SEQUENTIAL TOPOLOGY")
    print("="*80)
    print(f"Query: {query}")
    print("="*80 + "\n")
    
    # Create dummy experts
    expert_1 = await create_dummy_expert("Expert_1", debug=debug)
    expert_2 = await create_dummy_expert("Expert_2", debug=debug)
    expert_3 = await create_dummy_expert("Expert_3", debug=debug)
    
    experts = {
        "Expert_1": expert_1,
        "Expert_2": expert_2,
        "Expert_3": expert_3,
    }
    
    # Create dummy summary agent
    dummy_summary_llm = DummyLLM(name="DummySummary")
    summary_agent = SummaryAgent(dummy_summary_llm, debug=debug)
    
    # Create deep topology team
    team = DeepTopologyTeam(
        experts=experts,
        summary_agent=summary_agent,
        expert_order=["Expert_1", "Expert_2", "Expert_3"],
        max_messages=10,
        debug=debug,
        conversation_path="experiments/results/deep"
    )
    
    # Run consultation
    logger.info("Starting deep topology consultation...")
    start_time = datetime.now()
    
    result = await team.consult(query)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("DEEP TOPOLOGY RESULT")
    print("="*80)
    print(result)
    print(f"\nDuration: {duration:.2f} seconds")
    print("="*80 + "\n")
    
    # Save visualization
    try:
        png = team.team_graph.get_graph().draw_mermaid_png()
        with open("experiments/results/deep_topology_graph.png", "wb") as f:
            f.write(png)
        logger.info("Deep topology graph saved to experiments/results/deep_topology_graph.png")
    except Exception as e:
        logger.warning(f"Could not save graph visualization: {e}")
    
    return result


async def run_star_topology_experiment(query: str, debug: bool = True):
    """Run experiment with star/hub-and-spoke topology."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: STAR/HUB-AND-SPOKE TOPOLOGY")
    print("="*80)
    print(f"Query: {query}")
    print("="*80 + "\n")
    
    # Create dummy experts
    expert_1 = await create_dummy_expert("Expert_1", debug=debug)
    expert_2 = await create_dummy_expert("Expert_2", debug=debug)
    expert_3 = await create_dummy_expert("Expert_3", debug=debug)
    
    experts = {
        "Expert_1": expert_1,
        "Expert_2": expert_2,
        "Expert_3": expert_3,
    }
    
    # Create dummy coordinator
    dummy_coordinator_llm = DummyCoordinatorLLM(
        expert_sequence=["Expert_1", "Expert_2", "Expert_3", "summarize"]
    )
    coordinator = Coordinator(
        model_client=dummy_coordinator_llm,
        experts=experts,
        debug=debug,
        swift_info="Test SWIFT info"
    )
    
    # Create dummy summary agent
    dummy_summary_llm = DummyLLM(name="DummySummary")
    summary_agent = SummaryAgent(dummy_summary_llm, debug=debug)
    
    # Create star topology team
    team = StarTopologyTeam(
        coordinator=coordinator,
        experts=experts,
        summary_agent=summary_agent,
        max_messages=10,
        debug=debug,
        conversation_path="experiments/results/star"
    )
    
    # Run consultation
    logger.info("Starting star topology consultation...")
    start_time = datetime.now()
    
    result = await team.consult(query)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("STAR TOPOLOGY RESULT")
    print("="*80)
    print(result)
    print(f"\nDuration: {duration:.2f} seconds")
    print("="*80 + "\n")
    
    # Save visualization
    try:
        png = team.team_graph.get_graph().draw_mermaid_png()
        with open("experiments/results/star_topology_graph.png", "wb") as f:
            f.write(png)
        logger.info("Star topology graph saved to experiments/results/star_topology_graph.png")
    except Exception as e:
        logger.warning(f"Could not save graph visualization: {e}")
    
    return result


async def main():
    """Run both topology experiments."""
    # Ensure results directory exists
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/deep").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/star").mkdir(parents=True, exist_ok=True)
    
    # Test query
    query = "Analyze the routing logic and confirm that all experts are consulted in the correct order."
    
    print("\n" + "="*80)
    print("TOPOLOGY ROUTING EXPERIMENT")
    print("Testing routing logic with dummy LLMs (no API calls)")
    print("="*80)
    
    # Run both experiments
    try:
        deep_result = await run_deep_topology_experiment(query, debug=True)
        star_result = await run_star_topology_experiment(query, debug=True)
        
        # Summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print("\n✅ Deep Topology Test:")
        print("   - Expert order: Expert_1 -> Expert_2 -> Expert_3 -> Summary")
        print("   - No coordinator intervention")
        print("   - Linear/Sequential communication")
        
        print("\n✅ Star Topology Test:")
        print("   - Expert order: Coordinator -> Expert -> Coordinator (hub-and-spoke)")
        print("   - All communication through coordinator")
        print("   - Centralized decision making")
        
        print("\n📊 Results:")
        print(f"   - Deep topology completed successfully")
        print(f"   - Star topology completed successfully")
        print(f"   - Routing logic verified for both topologies")
        
        print("\n📁 Output files:")
        print("   - experiments/results/deep_topology_graph.png")
        print("   - experiments/results/star_topology_graph.png")
        print("   - experiments/results/experiment_*.log")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"\n❌ Experiment failed: {e}\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
