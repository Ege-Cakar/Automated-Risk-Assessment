# Experiments Directory Index

## 📚 Documentation Files

### 1. [QUICKSTART.md](QUICKSTART.md) - Start Here! ⭐
**Quick start guide for running the experiments**
- How to run the demos
- What to expect
- Troubleshooting tips
- Minimal dependencies

**Best for**: First-time users who want to see results quickly

### 2. [README.md](README.md) - Overview
**Detailed overview of the experiments**
- Topology concepts
- Architecture explanation
- Benefits of each approach
- Related concepts

**Best for**: Understanding the theory and motivation

### 3. [TOPOLOGY_DIAGRAMS.md](TOPOLOGY_DIAGRAMS.md) - Visual Guide
**Visual architecture diagrams and flow charts**
- ASCII art diagrams
- Flow comparisons
- Decision trees
- Trade-offs table

**Best for**: Visual learners who want to see the architecture

### 4. [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) - Results
**Complete experimental results and analysis**
- Test configuration
- Quantitative results
- Verification checklist
- Lessons learned

**Best for**: Reviewing what was tested and the outcomes

## 🐍 Python Files

### Core Implementation

1. **[dummy_llm.py](dummy_llm.py)** - Dummy LLM Classes
   - `DummyLLM`: Base dummy LLM
   - `DummyCoordinatorLLM`: Coordinator simulation
   - `DummyExpertLLM`: Expert simulation
   - No actual API calls

2. **[topologies.py](topologies.py)** - Topology Implementations
   - `DeepTopologyTeam`: Sequential chain architecture
   - `StarTopologyTeam`: Hub-and-spoke architecture
   - LangGraph integration

### Experiment Scripts

3. **[simple_topology_demo.py](simple_topology_demo.py)** - Standalone Demo ⭐
   - **Recommended starting point**
   - No external dependencies
   - Complete working example
   - < 1 second runtime
   - Outputs JSON results

4. **[run_topology_experiment.py](run_topology_experiment.py)** - Full Integration
   - Integrates with actual system
   - Requires LangGraph/LangChain
   - Uses dummy LLMs
   - Generates graph visualizations

## 📊 Results Files

Located in `results/` directory:
- `topology_comparison_*.json` - Experimental results
- Graph visualizations (when generated)
- Execution logs

## 🚀 Quick Start Commands

### Run Standalone Demo (No Dependencies)
```bash
python3 experiments/simple_topology_demo.py
```

### Run Full Integration (Requires Dependencies)
```bash
pip install langgraph langchain langchain-openai langchain-core
python3 experiments/run_topology_experiment.py
```

## 📖 Reading Order Recommendations

### For Beginners
1. [QUICKSTART.md](QUICKSTART.md) - Learn how to run
2. Run [simple_topology_demo.py](simple_topology_demo.py) - See it work
3. [TOPOLOGY_DIAGRAMS.md](TOPOLOGY_DIAGRAMS.md) - Understand visually
4. [README.md](README.md) - Learn the concepts

### For Developers
1. [README.md](README.md) - Understand the concepts
2. [dummy_llm.py](dummy_llm.py) - Review implementation
3. [topologies.py](topologies.py) - Study architectures
4. [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) - See results

### For Researchers
1. [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) - Review methodology
2. [TOPOLOGY_DIAGRAMS.md](TOPOLOGY_DIAGRAMS.md) - Analyze architectures
3. [README.md](README.md) - Understand context
4. Results files - Examine data

## 🎯 Key Concepts

### Deep Topology
**Sequential chain**: Expert1 → Expert2 → Expert3 → Summary
- Fixed routing
- No coordinator
- Simple and deterministic

### Star Topology
**Hub-and-spoke**: Coordinator ↔ Expert ↔ Coordinator
- Dynamic routing
- Central coordinator
- Flexible and adaptive

## 🔍 What Was Tested

✅ Routing logic for both topologies
✅ State management across transitions
✅ Expert invocation in correct order
✅ Summary generation
✅ Error handling

## 📈 Results Summary

| Topology | Decisions | Steps | Success |
|----------|-----------|-------|---------|
| Deep | 0 | 4 | ✅ 100% |
| Star | 4 | 4 | ✅ 100% |

Both topologies successfully completed all test scenarios.

## 🔗 Related Resources

- [Recursive Self-Improvement Suite](https://github.com/keskival/recursive-self-improvement-suite) - Inspiration
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/) - Graph framework
- Main repository - Parent implementation

## 💡 Use Cases

### Deep Topology Best For:
- ETL pipelines
- Data transformation workflows
- Fixed processing stages
- Simple linear processes

### Star Topology Best For:
- Dynamic conversations
- Multi-domain problems
- Adaptive expert selection
- Complex decision flows

## 🛠️ Customization

All scripts can be modified to:
- Change expert names
- Modify queries
- Add more experts
- Adjust routing logic
- Change output formats

See individual files for specific customization points.

## ✨ Highlights

- 🚀 Runs in < 1 second
- 💰 No API costs (dummy LLMs)
- 📦 Standalone version available
- 📊 JSON output for analysis
- 📚 Comprehensive documentation
- ✅ Fully verified routing logic

---

**Last Updated**: October 1, 2025  
**Status**: Complete and Verified  
**Maintainer**: Automated Risk Assessment Team
