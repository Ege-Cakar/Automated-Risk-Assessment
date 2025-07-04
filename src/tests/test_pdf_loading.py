"""
Test script to diagnose PDF loading issues
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.db_loader import LobeVectorMemory, add_files_from_folder


async def test_pdf_loading():
    """Test PDF loading with different path approaches"""
    
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Test 1: Using relative path
    print("\n=== Test 1: Relative path '../database' ===")
    relative_path = "../database"
    print(f"Checking if path exists: {os.path.exists(relative_path)}")
    print(f"Absolute path resolves to: {os.path.abspath(relative_path)}")
    
    # Test 2: Using absolute path
    print("\n=== Test 2: Absolute path ===")
    # Get the project root directory (2 levels up from this test file)
    project_root = Path(__file__).parent.parent.parent
    absolute_path = project_root / "database"
    print(f"Project root: {project_root}")
    print(f"Database path: {absolute_path}")
    print(f"Path exists: {absolute_path.exists()}")
    
    if absolute_path.exists():
        # List PDF files
        pdf_files = list(absolute_path.glob("*.pdf"))
        print(f"\nFound {len(pdf_files)} PDF files:")
        for i, pdf in enumerate(pdf_files[:5]):  # Show first 5
            print(f"  {i+1}. {pdf.name}")
        if len(pdf_files) > 5:
            print(f"  ... and {len(pdf_files) - 5} more")
    
    # Test 3: Try to load PDFs with absolute path
    print("\n=== Test 3: Loading PDFs with absolute path ===")
    try:
        memory = LobeVectorMemory()
        await add_files_from_folder(memory, str(absolute_path), file_extensions=['.pdf'])
        print("✓ PDF loading completed successfully!")

        
        
    except Exception as e:
        print(f"✗ Error loading PDFs: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Try with relative path from different working directory
    print("\n=== Test 4: Testing relative path resolution ===")
    # Save current directory
    original_cwd = os.getcwd()
    
    try:
        # Change to src directory
        src_dir = Path(__file__).parent.parent
        os.chdir(src_dir)
        print(f"Changed working directory to: {os.getcwd()}")
        
        relative_path = "../database"
        print(f"Does '../database' exist from src/: {os.path.exists(relative_path)}")
        
    finally:
        # Restore original directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    asyncio.run(test_pdf_loading())
