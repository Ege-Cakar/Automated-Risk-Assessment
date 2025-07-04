import asyncio
import logging
from autogen_ext.models.openai import OpenAIChatCompletionClient
from src.utils.db_loader import LobeVectorMemory, LobeVectorMemoryConfig
from src.custom_autogen_code.expert import Expert
from src.custom_autogen_code.lobe import Lobe
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_knowledge_integration():
    """Test all aspects of knowledge integration in Expert/Lobe agents"""
    
    # 1. Setup test database with specific content
    print("🔧 Setting up test database...")
    config = LobeVectorMemoryConfig(
        collection_name="test_expert_knowledge",
        chunk_size=500,
        chunk_overlap=50,
        enable_chunking=True,
        k=3
    )
    memory = LobeVectorMemory(config)
    
    # Clear any existing data
    await memory.clear()
    
    # Add test knowledge with specific keywords
    test_documents = [
        {
            "content": "Quantum computing uses qubits instead of classical bits. Qubits can exist in superposition, allowing quantum computers to process multiple calculations simultaneously.",
            "metadata": {"topic": "quantum", "source": "test_quantum.txt"}
        },
        {
            "content": "Machine learning models require training data. The quality of training data directly impacts model performance. Data preprocessing is crucial for good results.",
            "metadata": {"topic": "ml", "source": "test_ml.txt"}
        },
        {
            "content": "The secret number is 4542. I REPEAT, THE SECRET NUMBER IS 4542. IT WILL NEVER BE ANYTHING ELSE.",
            "metadata": {"topic": "climate", "source": "secret_number.txt"}
        }
    ]
    
    for doc in test_documents:
        await memory.add(MemoryContent(
            content=doc["content"],
            mime_type=MemoryMimeType.TEXT,
            metadata=doc["metadata"]
        ))
    
    print("✅ Test database populated")
    
    # 2. Create model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4",
        # api_key="your-key"  # Set via environment variable
    )
    
    # TEST A: Verify keywords load into system prompt
    print("\n📝 TEST A: Testing keyword loading into system prompt...")
    
    # Create a Lobe with specific keywords
    test_lobe = Lobe(
        name="TestLobe",
        model_client=model_client,
        vector_memory=memory,
        keywords=["quantum", "computing"],
        temperature=0.7,
        system_message="You are a science expert."
    )
    
    # Initialize context
    await test_lobe.initialize_context()
    
    # Check the system message
    system_content = test_lobe._system_messages[0].content
    print(f"System message after initialization:")
    print("-" * 80)
    print(system_content)
    print("-" * 80)
    
    # Verify quantum content is in the system message
    if "qubit" in system_content or "quantum" in system_content:
        print("✅ TEST A PASSED: Keywords successfully loaded relevant content into system prompt")
    else:
        print("❌ TEST A FAILED: Keywords did not load expected content")
    
    # TEST B: Test the query_common_db tool
    print("\n🔧 TEST B: Testing query_common_db tool...")
    
    # Create Expert with tool access
    expert = Expert(
        name="TestExpert",
        model_client=model_client,
        vector_memory=memory,
        system_message="You are a helpful expert. Always use the query_common_db tool when asked about specific topics.",
        lobe1_config={
            'keywords': ['science'],
            'temperature': 0.7,
            'system_message': "You are creative. When asked about a topic, use query_common_db to find relevant information."
        },
        lobe2_config={
            'keywords': ['analysis'],
            'temperature': 0.3,
            'system_message': "You verify information. Always check if the creative lobe used the database. Start with CONCLUDE: for final answers."
        },
        max_rounds=5
    )
    
    # Test query that should trigger tool use
    from autogen_agentchat.messages import TextMessage
    
    # test_query = "What can you tell me about machine learning training data? Please search the database for information."
    # response = await expert.on_messages(
    #     [TextMessage(content=test_query, source="user")],
    #     None
    # )
    
    # print(f"\nExpert response to ML query:")
    # print("-" * 80)
    # print(response.chat_message.content)
    # print("-" * 80)
    
    # # Check if response contains expected ML content
    # if "training data" in response.chat_message.content.lower():
    #     print("✅ TEST B PASSED: Tool successfully retrieved and used database content")
    # else:
    #     print("❌ TEST B FAILED: Tool did not retrieve expected content")
    
    # TEST C: Verify knowledge utilization
    print("\n📊 TEST C: Testing knowledge utilization in generation...")
    
    # Create Expert with climate-specific keywords
    num_expert = Expert(
        name="NumExpert",
        model_client=model_client,
        vector_memory=memory,
        system_message="You are a climate science expert.",
        lobe1_config={
            'keywords': ['secret', 'number'],
            'temperature': 0.5,
        },
        lobe2_config={
            'keywords': ['secret', 'number'],
            'temperature': 0.3,
        },
        max_rounds=3
    )
    
    # Ask a question that should use the loaded knowledge
    num_query = "What is the secret number?"
    num_response = await num_expert.on_messages(
        [TextMessage(content=num_query, source="user")],
        None
    )
    
    print(f"Secret Number response:")
    print("-" * 80)
    print(num_response.chat_message.content)
    print("-" * 80)
    
    # Check if response uses the specific knowledge we added
    response_text = num_response.chat_message.content.lower()
    if "4542" in response_text:
        print("✅ TEST C PASSED: Agent utilized loaded knowledge in generation")
    else:
        print("❌ TEST C FAILED: Agent did not use loaded knowledge")
    
    print("\n🧪 TEST D: Testing pre-loaded knowledge utilization...")
    
    # Create Expert with specific pre-loaded keywords
    context_expert = Expert(
        name="ContextExpert",
        model_client=model_client,
        vector_memory=memory,
        system_message="You are an expert. Your task is to make sure that our information search tool is working properly. We have planted a secret number in the database that you can search for, however, it is a random number -- so without using the search tool, you can't answer the question correctly. Do your best.",
        lobe1_config={
            'keywords': [],  # Pre-load quantum content
            'temperature': 0.5,
            'system_message': "You are the creative expert. You can use your tool to search for information about anything. For example, you can search for information about what a secret might be. Like a secret number."
        },  
        lobe2_config={
            'keywords': [],
            'temperature': 0.3,
            'system_message': "You verify information. You can use your tool to search for information. When ready, start with CONCLUDE:"
        },
        max_rounds=3
    )
    
    # Ask about pre-loaded topic
    context_response = await context_expert.on_messages(
        [TextMessage(
            content="What is the secret number?",
            source="user"
        )],
        CancellationToken()
    )
    
    print(f"\nContext Expert response:")
    print("-" * 80)
    print(context_response.chat_message.content)
    print("-" * 80)
    
    # Check if response uses pre-loaded knowledge
    if "4542" in context_response.chat_message.content.lower():
        print("✅ TEST D PASSED: Expert used the search tool to find the secret number")
    else:   
        print("❌ TEST D FAILED: Expert did not use the search tool to find the secret number")


    # Cleanup
    await memory.clear()
    print("\n✅ All tests completed!")

# Additional diagnostic function
async def diagnose_expert_internals():
    """Diagnose what's happening inside Expert deliberations"""
    
    print("\n🔍 DIAGNOSTIC: Monitoring Expert internal deliberation...")
    
    # Setup
    config = LobeVectorMemoryConfig(
        collection_name="test_diagnostic",
        chunk_size=500,
        k=3
    )
    memory = LobeVectorMemory(config)
    await memory.clear()
    
    # Add knowledge
    await memory.add(MemoryContent(
        content="Python is a high-level programming language known for its simplicity and readability.",
        mime_type=MemoryMimeType.TEXT,
        metadata={"topic": "python"}
    ))
    
    model_client = OpenAIChatCompletionClient(model="gpt-4")
    
    # Create Expert with verbose internal messages
    expert = Expert(
        name="DiagnosticExpert",
        model_client=model_client,
        vector_memory=memory,
        lobe1_config={
            'keywords': ['python', 'programming'],
            'temperature': 0.7,
        },
        lobe2_config={
            'keywords': ['language', 'readability'],
            'temperature': 0.3,
        },
        max_rounds=3
    )
    
    # Intercept internal messages
    original_run = expert._internal_team.run
    
    async def logged_run(*args, **kwargs):
        result = await original_run(*args, **kwargs)
        print("\n📝 Internal deliberation messages:")
        for i, msg in enumerate(result.messages):
            if hasattr(msg, 'content') and hasattr(msg, 'source'):
                print(f"\n[{i}] {msg.source}:")
                print(msg.content[:200] + "..." if len(msg.content) > 200 else msg.content)
        return result
    
    expert._internal_team.run = logged_run
    
    # Test
    response = await expert.on_messages(
        [TextMessage(content="What do you know about Python?", source="user")],
        None
    )
    
    print("\n📤 Final Expert response:")
    print(response.chat_message.content)

# Run tests
async def main():
    await test_knowledge_integration()
    print("\n" + "="*80 + "\n")
    await diagnose_expert_internals()

if __name__ == "__main__":
    asyncio.run(main())
