from langchain_core.tools import tool
from typing import Annotated

@tool 
def write_to_whiteboard(to_write: Annotated[str, "Content to write to report"]):
    """Write to whiteboard"""
    whiteboard = "data/text_files/whiteboard.md"
    
    with open(whiteboard, "a") as f:
        f.write(to_write)
    
    return "Report updated"
    
@tool 
def read_whiteboard() -> Annotated[str, "Content of report"]:
    """Read whiteboard"""
    whiteboard = "data/text_files/whiteboard.md"
    
    with open(whiteboard, "r") as f:
        return f.read() 


    