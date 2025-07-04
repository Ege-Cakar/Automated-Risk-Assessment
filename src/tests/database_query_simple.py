"""
Simple and robust database query example that avoids memory corruption issues
"""
import asyncio
import gc
from dotenv import load_dotenv
from utils.db_loader import LobeVectorMemory, add_files_from_folder
from utils.db_summarizer import get_database_stats
from utils.paths import paths
from utils.db_loader import LobeVectorMemoryConfig

load_dotenv()

async def main():
    """
    Simple example showing database loading and basic statistics
    """
    print("🔄 Initializing database and loading files...")
    
    # Initialize memory
    memory = LobeVectorMemory(
        config=LobeVectorMemoryConfig(
            collection_name="expert_test",
            persistence_path=paths.VECTORDB_PATH
        )
    )
    
    # Load files from database folder
    db_file_path = "../database"
    await add_files_from_folder(memory, db_file_path)
    
    # Force garbage collection after file loading
    gc.collect()
    
    print("✅ Files loaded successfully!")
    
    print("\n" + "="*60)
    print("📊 DATABASE STATISTICS")
    print("="*60)
    
    try:
        # Get database statistics
        stats = await get_database_stats(memory)
        print(f"Total Documents: {stats.get('total_documents', 0)}")
        print(f"Total Characters: {stats.get('total_characters', 0):,}")
        print(f"Average Document Size: {stats.get('average_document_size', 0):,} characters")
        print("\nDocument Types:")
        for doc_type, count in stats.get('document_types', {}).items():
            print(f"  - {doc_type}: {count}")
            
        print("\n" + "="*60)
        print("📋 DATABASE CONTENT OVERVIEW")
        print("="*60)
        
        # Get a sample of document sources to show what's loaded
        all_docs = await get_sample_documents(memory, limit=10)
        if all_docs:
            print("Sample of loaded documents:")
            for i, doc in enumerate(all_docs, 1):
                filename = doc.get("metadata", {}).get("filename", "Unknown")
                content_length = len(doc.get("content", ""))
                print(f"{i:2d}. {filename} ({content_length:,} characters)")
        
        print("\n" + "="*60)
        print("🎯 BASIC SEARCH FUNCTIONALITY")
        print("="*60)
        
        # Demonstrate basic search without complex AI querying
        search_terms = ["risk assessment", "compliance", "cybersecurity"]
        
        for term in search_terms:
            print(f"\n🔍 Searching for: '{term}'")
            matches = await simple_text_search(memory, term, limit=3)
            if matches:
                print(f"Found {len(matches)} documents containing '{term}':")
                for match in matches:
                    filename = match.get("metadata", {}).get("filename", "Unknown")
                    print(f"  - {filename}")
            else:
                print(f"No documents found containing '{term}'")
        
        print("\n🎉 Database query tests completed successfully!")
        print("\n💡 The database is now loaded and ready for use!")
        print("   You can query it using the LLM integration or other tools.")
        
    except Exception as e:
        print(f"Error during database operations: {e}")
        return

async def get_sample_documents(memory: LobeVectorMemory, limit: int = 10):
    """Get a sample of documents from the database"""
    try:
        collection = memory._collection
        results = collection.get(limit=limit)
        
        documents = []
        if results and 'documents' in results:
            for i, doc_content in enumerate(results['documents']):
                metadata = results.get('metadatas', [{}])[i] if i < len(results.get('metadatas', [])) else {}
                documents.append({
                    "content": doc_content,
                    "metadata": metadata
                })
        
        return documents
    except Exception as e:
        print(f"Error retrieving sample documents: {e}")
        return []

async def simple_text_search(memory: LobeVectorMemory, search_term: str, limit: int = 5):
    """Perform simple text search without complex vector operations"""
    try:
        collection = memory._collection
        results = collection.get()
        
        matches = []
        if results and 'documents' in results:
            for i, doc_content in enumerate(results['documents']):
                if search_term.lower() in doc_content.lower():
                    metadata = results.get('metadatas', [{}])[i] if i < len(results.get('metadatas', [])) else {}
                    matches.append({
                        "content": doc_content,
                        "metadata": metadata
                    })
                    if len(matches) >= limit:
                        break
        
        return matches
    except Exception as e:
        print(f"Error during text search: {e}")
        return []

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
    finally:
        # Clean up resources
        gc.collect()
