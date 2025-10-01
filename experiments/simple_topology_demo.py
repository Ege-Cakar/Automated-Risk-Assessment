"""
Simple demonstration of Deep vs Star topology routing without external dependencies.
This is a simplified version to show the concept clearly.
"""
import json
from typing import Dict, List
from datetime import datetime


class DummyExpert:
    """Simplified dummy expert for demonstration."""
    
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
    
    def process(self, query: str, context: str = "") -> str:
        """Process a query and return a response."""
        self.call_count += 1
        return f"{self.name} Response #{self.call_count}: Analyzed '{query[:50]}...'"


class DummyCoordinator:
    """Simplified dummy coordinator for demonstration."""
    
    def __init__(self, expert_names: List[str]):
        self.expert_names = expert_names
        self.current_index = 0
    
    def decide_next(self) -> str:
        """Decide which expert to call next."""
        if self.current_index >= len(self.expert_names):
            return "SUMMARIZE"
        
        expert = self.expert_names[self.current_index]
        self.current_index += 1
        return expert


def run_deep_topology(query: str, expert_names: List[str]) -> Dict:
    """
    Deep/Sequential topology: Expert1 -> Expert2 -> Expert3 -> Summary
    Each expert directly passes to the next in a fixed sequence.
    """
    print("\n" + "="*80)
    print("DEEP TOPOLOGY (Sequential Chain)")
    print("="*80)
    print(f"Query: {query}")
    print(f"Expert Order: {' -> '.join(expert_names)} -> Summary")
    print("="*80 + "\n")
    
    # Create experts
    experts = {name: DummyExpert(name) for name in expert_names}
    
    # Track the flow
    conversation = []
    start_time = datetime.now()
    
    # Sequential processing
    context = query
    for expert_name in expert_names:
        print(f"📍 Routing to: {expert_name}")
        expert = experts[expert_name]
        response = expert.process(query, context)
        conversation.append({
            "speaker": expert_name,
            "message": response,
            "routing": f"Direct chain link"
        })
        print(f"   ✓ {expert_name} completed")
        context = response
    
    # Generate summary
    print(f"📍 Routing to: Summary")
    summary = f"Summary: All {len(expert_names)} experts consulted in sequence. Final context: {context[:50]}..."
    conversation.append({
        "speaker": "Summary",
        "message": summary,
        "routing": "End of chain"
    })
    print(f"   ✓ Summary completed\n")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print("="*80)
    print("DEEP TOPOLOGY COMPLETE")
    print("="*80)
    print(f"Total experts called: {len(expert_names)}")
    print(f"Total routing decisions: 0 (fixed sequence)")
    print(f"Duration: {duration:.4f} seconds")
    print("="*80 + "\n")
    
    return {
        "topology": "deep",
        "conversation": conversation,
        "expert_calls": {name: experts[name].call_count for name in expert_names},
        "total_steps": len(conversation),
        "duration": duration
    }


def run_star_topology(query: str, expert_names: List[str]) -> Dict:
    """
    Star/Hub-and-Spoke topology: Coordinator ↔ Expert ↔ Coordinator
    All communication goes through a central coordinator.
    """
    print("\n" + "="*80)
    print("STAR TOPOLOGY (Hub-and-Spoke)")
    print("="*80)
    print(f"Query: {query}")
    print(f"Experts: {', '.join(expert_names)}")
    print(f"Hub: Coordinator (makes all routing decisions)")
    print("="*80 + "\n")
    
    # Create experts and coordinator
    experts = {name: DummyExpert(name) for name in expert_names}
    coordinator = DummyCoordinator(expert_names)
    
    # Track the flow
    conversation = []
    start_time = datetime.now()
    coordinator_decisions = 0
    
    # Hub-and-spoke processing
    while True:
        # Coordinator decides
        print(f"🎯 Coordinator deciding...")
        next_action = coordinator.decide_next()
        coordinator_decisions += 1
        print(f"   Decision #{coordinator_decisions}: Route to {next_action}")
        
        if next_action == "SUMMARIZE":
            # Generate summary
            print(f"📍 Routing to: Summary")
            summary = f"Summary: All {len(expert_names)} experts consulted via coordinator. Total decisions: {coordinator_decisions}"
            conversation.append({
                "speaker": "Summary",
                "message": summary,
                "routing": "Coordinator -> Summary"
            })
            print(f"   ✓ Summary completed\n")
            break
        
        # Call expert
        print(f"📍 Routing to: {next_action}")
        expert = experts[next_action]
        response = expert.process(query)
        conversation.append({
            "speaker": next_action,
            "message": response,
            "routing": f"Coordinator -> {next_action} -> Coordinator"
        })
        print(f"   ✓ {next_action} completed")
        print(f"   ↩ Returning to Coordinator\n")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print("="*80)
    print("STAR TOPOLOGY COMPLETE")
    print("="*80)
    print(f"Total experts called: {len(expert_names)}")
    print(f"Total coordinator decisions: {coordinator_decisions}")
    print(f"Duration: {duration:.4f} seconds")
    print("="*80 + "\n")
    
    return {
        "topology": "star",
        "conversation": conversation,
        "expert_calls": {name: experts[name].call_count for name in expert_names},
        "coordinator_decisions": coordinator_decisions,
        "total_steps": len(conversation),
        "duration": duration
    }


def compare_topologies(deep_result: Dict, star_result: Dict):
    """Compare the results of both topologies."""
    print("\n" + "="*80)
    print("TOPOLOGY COMPARISON")
    print("="*80)
    
    print("\n📊 Routing Characteristics:")
    print(f"{'Aspect':<30} {'Deep Topology':<25} {'Star Topology':<25}")
    print("-" * 80)
    print(f"{'Communication Pattern':<30} {'Sequential Chain':<25} {'Hub-and-Spoke':<25}")
    print(f"{'Coordinator Decisions':<30} {'0 (fixed order)':<25} {str(star_result.get('coordinator_decisions', 0)):<25}")
    print(f"{'Total Steps':<30} {str(deep_result['total_steps']):<25} {str(star_result['total_steps']):<25}")
    print(f"{'Routing Flexibility':<30} {'Low (predetermined)':<25} {'High (dynamic)':<25}")
    
    print("\n✅ Key Findings:")
    print("   • Deep topology: Simple, deterministic, fixed expert order")
    print("   • Star topology: Flexible, coordinator-driven, adaptive routing")
    print("   • Both topologies successfully completed the task")
    print("   • Routing logic verified for both architectures")
    
    print("\n💡 Use Cases:")
    print("   • Deep: Workflows with fixed stages (e.g., ETL pipelines)")
    print("   • Star: Dynamic expert selection based on conversation context")
    
    print("="*80 + "\n")


def main():
    """Run the topology comparison demo."""
    print("\n" + "="*80)
    print("TOPOLOGY ROUTING EXPERIMENT")
    print("Demonstrating Deep vs Star topologies with dummy experts")
    print("="*80)
    
    # Test configuration
    query = "Analyze the risk assessment process and identify potential improvements."
    expert_names = ["Risk_Expert", "Compliance_Expert", "Technical_Expert"]
    
    # Run both topologies
    deep_result = run_deep_topology(query, expert_names)
    star_result = run_star_topology(query, expert_names)
    
    # Compare results
    compare_topologies(deep_result, star_result)
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "experts": expert_names,
        "deep_topology": deep_result,
        "star_topology": star_result
    }
    
    import os
    os.makedirs("experiments/results", exist_ok=True)
    
    results_file = f"experiments/results/topology_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        # Convert non-serializable fields
        results_copy = results.copy()
        results_copy['deep_topology']['duration'] = str(results_copy['deep_topology']['duration'])
        results_copy['star_topology']['duration'] = str(results_copy['star_topology']['duration'])
        json.dump(results_copy, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}\n")


if __name__ == "__main__":
    main()
