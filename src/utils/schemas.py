from typing import List, TypedDict, Dict

class ExpertState(TypedDict):
    """State for the Expert agent's internal deliberation"""
    messages: List[str]
    query: str
    lobe1_response: str
    lobe2_response: str
    final_conclusion: str
    iteration_count: int
    max_rounds: int
    concluded: bool
    vector_context: str

class TeamState(TypedDict):
    """State for the entire team conversation"""
    messages: List[Dict[str, str]]              # All conversation messages
    query: str                                  # Original user query
    current_speaker: str                        # Current agent name
    conversation_keywords: List[str]            # Current conversation keywords
    expert_responses: Dict[str, str]            # expert_name -> last_response
    message_count: int                          # Track message limit
    max_messages: int                           # Maximum allowed messages
    concluded: bool                             # Whether conversation is done
    coordinator_decision: str                   # "expert_name", "summarize", "end"
    final_report: str                          # Summary agent's final output
    debug: bool                                # Debug mode flag