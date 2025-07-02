"""
Test ChromaDB integration step by step
Run this to verify ChromaDB is working correctly
"""
import asyncio
import chromadb
from sentence_transformers import SentenceTransformer


async def test_chromadb():
    print("1. Testing basic ChromaDB...")
    
    # Test 1: Basic ChromaDB
    try:
        client = chromadb.PersistentClient(path="./test_db")
        print("✓ ChromaDB client created")
    except Exception as e:
        print(f"✗ ChromaDB client failed: {e}")
        return
    
    # Test 2: Collection operations
    try:
        # Delete if exists
        try:
            client.delete_collection("test")
        except:
            pass
        
        collection = client.create_collection("test")
        print("✓ Collection created")
    except Exception as e:
        print(f"✗ Collection creation failed: {e}")
        return
    
    # Test 3: Embeddings
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_text = "Hello world"
        embedding = model.encode(test_text).tolist()
        print(f"✓ Embedding created (dim: {len(embedding)})")
    except Exception as e:
        print(f"✗ Embedding failed: {e}")
        return
    
    # Test 4: Add document
    try:
        collection.add(
            documents=["Test document"],
            embeddings=[embedding],
            metadatas=[{"test": "data"}],
            ids=["test1"]
        )
        print("✓ Document added")
    except Exception as e:
        print(f"✗ Document add failed: {e}")
        return
    
    # Test 5: Query
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1
        )
        print("✓ Query successful")
        print(f"  Found: {results['documents'][0][0]}")
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return
    
    # Test 6: Async wrapper
    try:
        async def async_query():
            return await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=1
            )
        
        results = await async_query()
        print("✓ Async wrapper working")
    except Exception as e:
        print(f"✗ Async wrapper failed: {e}")
        return
    
    print("\n✓ All tests passed! ChromaDB is working correctly.")
    
    # Cleanup
    client.delete_collection("test")
    print("✓ Cleanup complete")


async def test_lobe_memory():
    print("\n2. Testing LobeVectorMemory...")
    
    try:
        from utils.db_loader import LobeVectorMemory, LobeVectorMemoryConfig
        from autogen_core.memory import MemoryContent, MemoryMimeType
        
        # Create memory
        memory = LobeVectorMemory(
            config=LobeVectorMemoryConfig(
                collection_name="test_lobe",
                persistence_path="./test_lobe_db"
            )
        )
        print("✓ LobeVectorMemory created")
        
        # Add content
        await memory.add(
            MemoryContent(
                content="Test content for Lobe",
                mime_type=MemoryMimeType.TEXT,
                metadata={"test": "true"}
            )
        )
        print("✓ Content added")
        
        # Query
        results = await memory.query("test")
        print(f"✓ Query successful: {len(results)} results")
        
        # Cleanup
        await memory.clear()
        print("✓ Memory cleared")
        
        print("\n✓ LobeVectorMemory is working correctly!")
        
    except Exception as e:
        print(f"✗ LobeVectorMemory test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing ChromaDB Integration for Lobe Agent\n")
    asyncio.run(test_chromadb())
    asyncio.run(test_lobe_memory())