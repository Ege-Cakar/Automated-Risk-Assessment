# Topology Architecture Diagrams

## Deep/Sequential Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    DEEP TOPOLOGY                             │
│              (Sequential/Linear Chain)                       │
└─────────────────────────────────────────────────────────────┘

    Query
      ↓
┌──────────┐
│ Expert 1 │ ──→ Processes query
└──────────┘      Generates response
      ↓
┌──────────┐
│ Expert 2 │ ──→ Receives Expert 1's output
└──────────┘      Adds own analysis
      ↓
┌──────────┐
│ Expert 3 │ ──→ Receives Expert 2's output
└──────────┘      Adds final analysis
      ↓
┌──────────┐
│ Summary  │ ──→ Synthesizes all expert outputs
└──────────┘
      ↓
   Result

Characteristics:
• Fixed, deterministic routing
• Each expert passes directly to next
• No central coordinator
• Simple, linear flow
• Lower complexity
```

## Star/Hub-and-Spoke Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    STAR TOPOLOGY                             │
│                 (Hub-and-Spoke)                              │
└─────────────────────────────────────────────────────────────┘

                    Query
                      ↓
              ┌──────────────┐
              │ Coordinator  │ ◄──── Central Hub
              │   (Hub)      │       Makes all routing decisions
              └──────────────┘
                /    |    \
               /     |     \
              ↓      ↓      ↓
        ┌────────┐ ┌────────┐ ┌────────┐
        │Expert 1│ │Expert 2│ │Expert 3│  ◄──── Spokes
        └────────┘ └────────┘ └────────┘       Communicate only with hub
              \      |      /
               \     |     /
                ↓    ↓    ↓
              ┌──────────────┐
              │ Coordinator  │ ◄──── Decides next action
              │   (Hub)      │       (another expert or summary)
              └──────────────┘
                      ↓
                ┌──────────┐
                │ Summary  │
                └──────────┘
                      ↓
                   Result

Characteristics:
• Dynamic routing based on conversation context
• Central coordinator manages all communication
• Experts never communicate directly
• Higher flexibility
• More complex routing logic
```

## Flow Comparison

### Deep Topology Message Flow
```
Start → Expert1 → Expert2 → Expert3 → Summary → End

Total hops: 4
Coordinator decisions: 0 (predetermined)
Communication pattern: Linear chain
```

### Star Topology Message Flow
```
Start → Coordinator → Expert1 → Coordinator → 
        Coordinator → Expert2 → Coordinator →
        Coordinator → Expert3 → Coordinator →
        Coordinator → Summary → End

Total hops: 10 (includes coordinator transitions)
Coordinator decisions: 4
Communication pattern: Hub-and-spoke
```

## Routing Decision Trees

### Deep Topology (No Decisions)
```
Query received
    ↓ [automatic]
Expert 1
    ↓ [automatic]
Expert 2
    ↓ [automatic]
Expert 3
    ↓ [automatic]
Summary
```

### Star Topology (Dynamic Decisions)
```
Query received
    ↓
Coordinator decides ──→ [Which expert? What context?]
    ├─→ Expert 1? ✓
    ├─→ Expert 2? 
    └─→ Expert 3?
    ↓
Expert 1 responds
    ↓
Coordinator decides ──→ [Continue? Which expert next?]
    ├─→ Expert 2? ✓
    ├─→ Expert 3?
    └─→ Summarize?
    ↓
Expert 2 responds
    ↓
Coordinator decides ──→ [Continue? Which expert next?]
    ├─→ Expert 3? ✓
    └─→ Summarize?
    ↓
Expert 3 responds
    ↓
Coordinator decides ──→ [Continue? Summarize?]
    └─→ Summarize ✓
    ↓
Summary
```

## Trade-offs Table

| Aspect | Deep Topology | Star Topology |
|--------|---------------|---------------|
| **Complexity** | Low | High |
| **Flexibility** | Low (fixed order) | High (adaptive) |
| **Coordinator Role** | None | Central hub |
| **Message Hops** | N+1 (N experts) | 2N+2 (coordinator transitions) |
| **Decision Points** | 0 | N+1 |
| **Expert Autonomy** | High | Low |
| **Context Awareness** | Sequential only | Global (via coordinator) |
| **Failure Handling** | Difficult | Easier (coordinator can adapt) |
| **Scalability** | Limited | Better |
| **Use Case** | Fixed workflows | Dynamic conversations |

## Example Scenarios

### When to Use Deep Topology
1. **Fixed Processing Pipeline**: ETL, data transformation
2. **Sequential Dependencies**: Each stage requires previous output
3. **Simple Workflows**: Known steps, no branching
4. **Low Complexity**: Minimal coordination needed

### When to Use Star Topology
1. **Dynamic Expert Selection**: Context-dependent routing
2. **Complex Conversations**: Multiple back-and-forth exchanges
3. **Adaptive Workflows**: Decision-based routing
4. **Quality Control**: Coordinator can review and redirect
5. **Multi-domain Problems**: Experts from different domains

## Verification Results

Based on our dummy LLM experiments:

### Deep Topology ✅
- All experts called in correct order
- No coordinator overhead
- Deterministic execution path
- Suitable for workflows with known stages

### Star Topology ✅
- Coordinator successfully routes to all experts
- Dynamic decision making verified
- All experts return to coordinator
- Suitable for adaptive conversations

Both topologies successfully completed the test query and demonstrated correct routing behavior.
