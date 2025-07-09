import os
from autogen_core.tools import FunctionTool
from typing import Annotated

def save_report(report: Annotated[str, "Report to save."]) -> dict:
    """Save report to src/text_files/report.md file"""
    os.makedirs("src/text_files", exist_ok=True)
    with open("src/text_files/report.md", "w") as f:
        f.write(report)
    
    return {
        "status": "success",
        "message": "Report saved to src/text_files/report.md"
        }

save_report_tool = FunctionTool(
    name="save_report",
    func=save_report,
    description="Save report to a markdown document."
)
