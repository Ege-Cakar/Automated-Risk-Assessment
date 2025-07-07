"""
Quick Start: Minimal working example of Lobe Agent with ChromaDB
"""
import asyncio
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# Assuming you've saved the implementations in these files
from src.utils.db_loader import LobeVectorMemory, LobeVectorMemoryConfig
from src.custom_autogen_code.lobe import Lobe
from dotenv import load_dotenv

load_dotenv()


async def quick_start():
    # 1. Initialize ChromaDB vector memory
    memory = LobeVectorMemory(
        config=LobeVectorMemoryConfig(
            collection_name="quickstart",
            persistence_path="./quickstart_db"
        )
    )
    
    # 2. Add some knowledge
    await memory.add(
        MemoryContent(
            content="The Lobe agent is a custom AutoGen agent with ChromaDB vector database integration.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"topic": "lobe"}
        )
    )
    
    # 3. Create model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # 4. Create Lobe agent
    agent = Lobe(
        name="Assistant",
        model_client=model_client,
        vector_memory=memory,
        keywords=["lobe", "agent"],
        temperature=0.7
    )
    
    # 5. Initialize context (IMPORTANT!)
    await agent.initialize_context()
    
    # 6. Use the agent
    response = await agent.on_messages(
        [TextMessage(content="What is a Lobe agent?", source="user")],
        CancellationToken()
    )
    
    print(f"Agent response: {response.chat_message.content}")
    
    # 7. Use the query tool
    response = await agent.on_messages(
        [TextMessage(
            content="Use query_common_db to search for 'lobe' and tell me what you find.",
            source="user"
        )],
        CancellationToken()
    )
    
    print(f"Tool response: {response.chat_message.content}")


if __name__ == "__main__":

    
    # Run the example
    asyncio.run(quick_start())