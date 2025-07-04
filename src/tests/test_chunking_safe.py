"""
Safe test script to verify chunking functionality without memory issues
"""
import asyncio
import gc
from dotenv import load_dotenv
from ..utils.db_loader import LobeVectorMemory, LobeVectorMemoryConfig, add_files_from_folder
from autogen_core.memory import MemoryContent, MemoryMimeType
from ..utils.paths import paths, DOCUMENTS_DIR
load_dotenv()

async def test_chunking_safe():
    """
    Safely test the chunking functionality with a controlled example.

    This test initializes a LobeVectorMemory instance with chunking enabled and 
    adds a large document that is expected to be chunked into smaller parts. It verifies
    that the document is added successfully and checks the chunking behavior by performing
    keyword searches. The test ensures that the results contain unique chunks and evaluates
    the effectiveness of the chunking process by analyzing duplicate content in the search results.
    """
    print("🔧 Testing chunking functionality safely...")
    
    # Initialize memory with chunking enabled
    config = LobeVectorMemoryConfig(
        collection_name="safe_chunking_test",
        chunk_size=1000,    
        chunk_overlap=50,
        enable_chunking=True,
        k=5
    )
    memory = LobeVectorMemory(config)
    await memory.clear()
    
    try:
        await add_files_from_folder(memory, DOCUMENTS_DIR)
        
        # Get collection info
        collection = memory._collection
        total_chunks = collection.count()
        print(f"📊 Total chunks in collection: {total_chunks}")
        
        # Test search for 'risk management'
        print(f"\n🔍 Testing search for 'risk management'...")
        results = await memory.search_by_keywords(["risk", "management"])
        
        if results:
            print(f"📄 Search returned {len(results)} results")
            
            seen_content_hashes = set()
            unique_chunks = 0
            
            for i, result in enumerate(results):
                content = result.results[0].content
                metadata = result.results[0].metadata
                
                # Create a content hash for uniqueness check
                content_hash = hash(content[:50])
                chunk_index = metadata.get('chunk_index', 'N/A')
                total_chunks = metadata.get('total_chunks', 'N/A')
                score = metadata.get('score', 'N/A')
                
                print(f"\n{'='*80}")
                print(f"📄 Result {i+1}:")
                print(f"  Source file: {metadata.get('filename', 'unknown')}")
                print(f"  Chunk: {chunk_index}/{total_chunks}")
                print(f"  Score: {score:.4f}" if isinstance(score, float) else f"  Score: {score}")
                print(f"  Content hash: {content_hash}")
                print(f"  Chunk ID: {metadata.get('id', 'N/A')}")
                print(f"  Content length: {len(content)} chars")
                print(f"\n  FULL CONTENT:")
                print(f"  {'-'*76}")
                print(f"  {repr(content)}")  # This shows the exact content with escape chars
                print(f"  {'-'*76}")
                
                if content_hash not in seen_content_hashes:
                    unique_chunks += 1
                    seen_content_hashes.add(content_hash)
                else:
                    print(f"  ⚠️  DUPLICATE content detected!")
            
            print(f"\n📊 Chunking Analysis:")
            print(f"  Total results: {len(results)}")
            print(f"  Unique chunks: {unique_chunks}")
            print(f"  Duplicate ratio: {(len(results) - unique_chunks) / len(results):.2%}")
            
            if unique_chunks == len(results):
                print("✅ PERFECT: All search results are unique chunks!")
            elif unique_chunks > len(results) * 0.7:
                print("✅ GOOD: Most search results are unique chunks!")
            else:
                print("⚠️  Still seeing some duplication")
                
        else:
            print("📄 No search results returned")
            
        # Test search for 'security assessment'
        print(f"\n🔍 Testing search for 'security assessment'...")
        results = await memory.search_by_keywords(["security", "assessment"])
        
        if results:
            print(f"📄 Search returned {len(results)} results")
            for i, result in enumerate(results[:2]):  # Show first 2
                content = result.results[0].content
                metadata = result.results[0].metadata
                chunk_index = metadata.get('chunk_index', 'N/A')
                score = metadata.get('score', 'N/A')
                
                print(f"\n📄 Result {i+1}:")
                print(f"  Chunk: {chunk_index}")
                print(f"  Score: {score:.4f}" if isinstance(score, float) else f"  Score: {score}")
                print(f"  Preview: {content[:100]}...")
        
        print(f"\n✅ Safe chunking test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("🔍 DIAGNOSTIC: Checking for duplicate content in database...")

    # Query specifically for "management"
    diagnostic_results = await memory.query("management")
    print(f"Found {len(diagnostic_results)} results for 'management'")

    # Group by content
    content_groups = {}
    for result in diagnostic_results[:20]:  # Check first 20
        content = result.results[0].content
        metadata = result.results[0].metadata
        
        if content not in content_groups:
            content_groups[content] = []
        
        content_groups[content].append({
            'filename': metadata.get('filename', 'unknown'),
            'chunk_index': metadata.get('chunk_index', 'N/A'),
            'total_chunks': metadata.get('total_chunks', 'N/A'),
            'id': metadata.get('id', 'N/A')
        })

    # Show duplicate content
    for content, occurrences in content_groups.items():
        if len(occurrences) > 1:
            print(f"\n⚠️  Found {len(occurrences)} chunks with identical content:")
            print(f"Content: {repr(content[:100])}...")
            for occ in occurrences:
                print(f"  - {occ['filename']}: chunk {occ['chunk_index']}/{occ['total_chunks']} (ID: {occ['id'][:8]}...)")

    gc.collect()

async def main():
    try:
        await test_chunking_safe()
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()

if __name__ == "__main__":
    asyncio.run(main())
