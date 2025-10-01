# Topology Routing Experiments

This directory contains experiments to test different team communication topologies using dummy LLMs (no API calls).

## Overview

Based on concepts from the [Recursive Self-Improvement Suite](https://github.com/keskival/recursive-self-improvement-suite), this experiment tests the routing logic of different team architectures:

### 1. Deep/Sequential Topology
- **Structure**: Expert1 → Expert2 → Expert3 → Summary
- **Characteristics**:
  - Linear chain of experts
  - Each expert passes output directly to the next
  - No central coordinator
  - Deterministic routing order
  
### 2. Star/Hub-and-Spoke Topology
- **Structure**: Coordinator ↔ Expert ↔ Coordinator (hub-and-spoke)
- **Characteristics**:
  - Central coordinator manages all communication
  - Coordinator decides which expert speaks next
  - Allows dynamic expert selection
  - Current implementation in the main codebase

## Files

- `dummy_llm.py` - Dummy LLM implementations that return predictable responses without API calls
- `topologies.py` - Implementation of Deep and Star topology team architectures
- `run_topology_experiment.py` - Main experiment script
- `README.md` - This file

## Running the Experiments

```bash
# From the repository root
python experiments/run_topology_experiment.py
```

## What It Tests

1. **Routing Logic**: Verifies that messages flow correctly between agents
2. **State Management**: Ensures team state is properly maintained across transitions
3. **Expert Invocation**: Confirms each expert is called in the expected order
4. **Summary Generation**: Validates that final summaries are produced correctly

## Expected Output

The experiment will:
1. Create dummy LLMs that simulate expert and coordinator behavior
2. Run both topology experiments with the same query
3. Generate visualization graphs (PNG files) showing the routing structure
4. Log all routing decisions and state transitions
5. Compare the behavior of both topologies

## Results

Results are saved to:
- `experiments/results/deep/` - Deep topology results
- `experiments/results/star/` - Star topology results
- `experiments/results/deep_topology_graph.png` - Deep topology visualization
- `experiments/results/star_topology_graph.png` - Star topology visualization
- `experiments/results/experiment_*.log` - Detailed execution logs

## Key Differences

| Aspect | Deep Topology | Star Topology |
|--------|--------------|---------------|
| Communication | Sequential | Through coordinator |
| Routing | Deterministic | Dynamic |
| Flexibility | Low | High |
| Coordinator | None | Central |
| Expert Order | Fixed | Coordinator decides |

## Benefits of Each Topology

### Deep Topology
- ✅ Simpler routing logic
- ✅ Predictable execution flow
- ✅ Lower complexity
- ❌ Less flexible expert selection
- ❌ Cannot adapt to conversation needs

### Star Topology
- ✅ Dynamic expert selection
- ✅ Coordinator can adapt based on conversation
- ✅ Better for complex, multi-expert scenarios
- ❌ More complex routing logic
- ❌ Coordinator becomes bottleneck

## Related Concepts

This experiment draws inspiration from:
- **Recursive Self-Improvement Suite**: Testing open-ended tasks with LLM agents
- **Multi-Agent Systems**: Different communication patterns in distributed systems
- **Graph Neural Networks**: Information flow in different graph topologies
