# Quick Start Guide - Topology Experiments

## What This Is

This experiment demonstrates different team communication architectures (topologies) using **dummy LLMs** that don't make actual API calls. This allows you to test and verify routing logic without incurring costs or delays.

## Inspired By

The [Recursive Self-Improvement Suite](https://github.com/keskival/recursive-self-improvement-suite) repository explores how AI systems can improve through open-ended tasks and different architectural patterns. Our topology experiments test how information flows through different expert team structures.

## Quick Run

### Option 1: Simple Demo (Recommended - No Dependencies)

```bash
cd /home/runner/work/Automated-Risk-Assessment/Automated-Risk-Assessment
python3 experiments/simple_topology_demo.py
```

This runs a standalone demonstration that shows:
- Deep topology (sequential chain of experts)
- Star topology (hub-and-spoke with coordinator)
- Side-by-side comparison
- JSON results file

**Runtime**: < 1 second
**Dependencies**: None (pure Python)

### Option 2: Full Integration (Requires Dependencies)

```bash
# Install dependencies
pip install langgraph langchain langchain-openai langchain-core

# Run full experiment
python3 experiments/run_topology_experiment.py
```

This integrates with the actual system architecture but uses dummy LLMs instead of real ones.

## What You'll See

### Deep Topology Output
```
📍 Routing to: Risk_Expert
   ✓ Risk_Expert completed
📍 Routing to: Compliance_Expert
   ✓ Compliance_Expert completed
📍 Routing to: Technical_Expert
   ✓ Technical_Expert completed
📍 Routing to: Summary
   ✓ Summary completed
```

### Star Topology Output
```
🎯 Coordinator deciding...
   Decision #1: Route to Risk_Expert
📍 Routing to: Risk_Expert
   ✓ Risk_Expert completed
   ↩ Returning to Coordinator

🎯 Coordinator deciding...
   Decision #2: Route to Compliance_Expert
...
```

## Understanding the Output

### Key Metrics

1. **Total Steps**: Number of processing stages
2. **Coordinator Decisions**: How many routing decisions were made
3. **Expert Calls**: How many times each expert was invoked
4. **Duration**: Time to complete (very fast with dummy LLMs)

### Interpretation

- **Deep topology** has 0 coordinator decisions because routing is predetermined
- **Star topology** has N+1 coordinator decisions (one per expert plus summary)
- Both should call each expert exactly once
- Both should produce a final summary

## Files Created

After running, you'll find:

```
experiments/results/
├── topology_comparison_YYYYMMDD_HHMMSS.json  # Detailed results
├── deep_topology_graph.png                    # Visual graph (full version only)
└── star_topology_graph.png                    # Visual graph (full version only)
```

## Customization

### Change Expert Names
Edit `simple_topology_demo.py`:
```python
expert_names = ["Your_Expert_1", "Your_Expert_2", "Your_Expert_3"]
```

### Change Query
Edit `simple_topology_demo.py`:
```python
query = "Your custom query here"
```

### Add More Experts
Just add more names to the `expert_names` list - the topologies will automatically adapt.

## Verification Checklist

Use this to verify the experiment worked correctly:

- [ ] Deep topology called all experts in sequence
- [ ] Star topology returned to coordinator after each expert
- [ ] Both topologies generated a summary
- [ ] No actual API calls were made (confirmed by instant execution)
- [ ] Results JSON file was created
- [ ] All experts were called exactly once
- [ ] Routing patterns match the expected topology

## Troubleshooting

### Import Errors (Full Version)
If you get import errors with `run_topology_experiment.py`:
- Use `simple_topology_demo.py` instead (no dependencies)
- Or install missing packages: `pip install langgraph langchain`

### No Output
- Make sure you're in the repository root directory
- Check that the experiments folder exists
- Verify Python 3.7+ is installed

### Permission Errors
- Make sure experiments/results directory is writable
- Run: `mkdir -p experiments/results`

## Next Steps

1. **Compare topologies**: Look at the JSON output to see the differences
2. **Read the diagrams**: Check `TOPOLOGY_DIAGRAMS.md` for visual explanations
3. **Modify the code**: Try adding your own routing logic
4. **Integrate with real LLMs**: Replace dummy LLMs with actual API clients

## Key Takeaways

✅ **Deep Topology**: Best for fixed workflows with predetermined stages
✅ **Star Topology**: Best for dynamic conversations requiring adaptive routing
✅ **Dummy LLMs**: Great for testing routing logic without API costs
✅ **Routing Verification**: Both topologies can be validated independently

## Related Files

- `README.md` - Detailed documentation
- `TOPOLOGY_DIAGRAMS.md` - Visual architecture diagrams
- `dummy_llm.py` - Dummy LLM implementations
- `topologies.py` - Topology class implementations
- `simple_topology_demo.py` - Standalone demonstration
- `run_topology_experiment.py` - Full integration experiment
