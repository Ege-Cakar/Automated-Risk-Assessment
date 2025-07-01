from pydantic import BaseModel
from typing import List, Literal

class Expert(BaseModel):
    name: str
    system_prompt: str 
    keywords: List[str]

class OrganizationResponse(BaseModel):
    thoughts: str
    done: Literal["DONE", "NOT_DONE"]
    response: Expert

class KeywordBrainstormingResponse(BaseModel):
    thoughts: str
    response: List[str]