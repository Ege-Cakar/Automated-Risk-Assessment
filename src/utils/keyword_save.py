from autogen_core.tools import FunctionTool
from typing import Annotated
import os

def save_keywords(
    keywords: Annotated[list, "List of keywords relevant to SWIFT."],
    ) -> dict:
        """Save keywords to src/text_files/keywords.txt file"""
        # Ensure the directory exists
        os.makedirs("src/text_files", exist_ok=True)
        
        # Write keywords to file, one per line
        with open("src/text_files/keywords.txt", "w") as f:
            for keyword in keywords:
                f.write(f"{keyword},")
        
        return {
            "status": "success",
            "message": f"Saved {len(keywords)} keywords to src/text_files/keywords.txt",
            "keywords_count": len(keywords)
        }
    
save_keywords_tool = FunctionTool(
    name="save_keywords",
    func=save_keywords,
    description="Save keywords to a structured document."
)