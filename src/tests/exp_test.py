"""
Quick Start: Minimal working example of Expert Agent with internal Lobe team
"""
import asyncio
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

# Import your implementations
from ..utils.db_loader import LobeVectorMemory, LobeVectorMemoryConfig
from ..custom_autogen_code.lobe import Lobe
from ..custom_autogen_code.expert import Expert  # Your Expert implementation
from dotenv import load_dotenv

load_dotenv()


async def test_expert_with_shared_messages():
    """Test Expert agent with shared base system message"""
    
    print("="*60)
    print("Testing Expert Agent with Shared System Messages")
    print("="*60)
    
    # 1. Initialize shared vector memory
    memory = LobeVectorMemory(
        config=LobeVectorMemoryConfig(
            collection_name="expert_test",
            persistence_path="./expert_test_db"
        )
    )
    
    # 2. Add some test knowledge
    test_knowledge = [
        {
            "content": "Machine learning models require careful validation to avoid overfitting.",
            "metadata": {"topic": "ml", "aspect": "validation"}
        },
        {
            "content": "Cross-validation is essential for robust model evaluation.",
            "metadata": {"topic": "ml", "aspect": "evaluation"}
        },
        {
            "content": "Feature engineering can significantly improve model performance.",
            "metadata": {"topic": "ml", "aspect": "features"}
        }
    ]
    
    for item in test_knowledge:
        await memory.add(
            MemoryContent(
                content=item["content"],
                mime_type=MemoryMimeType.TEXT,
                metadata=item["metadata"]
            )
        )

    # 3. Create model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
    )
    
    # 4. Create Expert with shared base system message
    ml_expert = Expert(
        name="MLExpert",
        model_client=model_client,
        vector_memory=memory,
        system_message="You are a machine learning expert with deep knowledge of model development, validation, and deployment. Always provide practical, evidence-based advice.",
        max_rounds=6
    )
    
    # 5. Test 1: Direct query to Expert
    print("\nTest 1: Direct Expert Query")
    print("-" * 40)
    
    response = await ml_expert.on_messages(
        [TextMessage(
            content="What are the key considerations when building a production ML model?",
            source="user"
        )],
        CancellationToken()
    )
    
    print(f"Expert response:\n{response.chat_message.content}\n")
    
    # 6. Test 2: Expert using vector DB through internal lobes
    print("\nTest 2: Expert with Vector DB Access")
    print("-" * 40)
    
    response = await ml_expert.on_messages(
        [TextMessage(
            content="Based on the knowledge base, what are the best practices for model validation?",
            source="user"
        )],
        CancellationToken()
    )
    
    print(f"Expert response:\n{response.chat_message.content}\n")
    
    # 7. Test 3: Expert in a team setting
    print("\nTest 3: Expert in a Team")
    print("-" * 40)
    
    # Create a simple coordinator agent
    from autogen_agentchat.agents import AssistantAgent
    coordinator = AssistantAgent(
        name="Coordinator",
        model_client=model_client,
        system_message="You coordinate technical discussions and ensure all aspects are covered."
    )
    
    # Create a team with the Expert
    team = RoundRobinGroupChat(
        participants=[coordinator, ml_expert],
        termination_condition=MaxMessageTermination(max_messages=4)
    )
    
    # Run a team task
    result = await team.run(
        task="Let's design a fraud detection system. Coordinator, please gather requirements. Expert, provide ML architecture recommendations."
    )
    
    print("Team discussion completed!")
    print(f"Number of messages: {len(result.messages)}")
    print(f"Last message: {result.messages[-1].content}\n")
    
    # 8. Verify system message inheritance
    print("\nVerifying System Message Inheritance:")
    print("-" * 40)
    print(f"Base system message: '{ml_expert._base_system_message[:80]}...'\n")
    print(f"Lobe 1 keywords: {ml_expert._lobe1.keywords}")
    print(f"Lobe 2 keywords: {ml_expert._lobe2.keywords}")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_expert_with_shared_messages())