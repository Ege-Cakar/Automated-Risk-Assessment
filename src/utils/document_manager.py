from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import json
import uuid
from enum import Enum

class SectionStatus(Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    MERGED = "merged"

@dataclass
class Section:
    section_id: str
    domain: str
    author: str
    content: str
    version: int = 1
    status: SectionStatus = SectionStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentChange:
    change_id: str
    section_id: str
    author: str
    change_type: Literal["create", "edit", "merge", "approve"]
    content_before: Optional[str] = None
    content_after: Optional[str] = None
    rationale: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class DocumentManager:
    def __init__(self, base_path: str = "data/report"):
        self.base_path = base_path
        self.sections: Dict[str, Section] = {}
        self.history: List[DocumentChange] = []
        self.current_document: Dict[str, str] = {}  # section_id -> content
        
    def create_section(self, domain: str, author: str, content: str) -> str:
        """Create a new draft section"""
        section_id = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        section = Section(
            section_id=section_id,
            domain=domain,
            author=author,
            content=content
        )
        self.sections[section_id] = section
        
        change = DocumentChange(
            change_id=uuid.uuid4().hex,
            section_id=section_id,
            author=author,
            change_type="create",
            content_after=content
        )
        self.history.append(change)
        
        return section_id
    
    def propose_edit(self, section_id: str, author: str, new_content: str, rationale: str) -> str:
        """Propose an edit to existing section (creates new version)"""
        if section_id not in self.sections:
            raise ValueError(f"Section {section_id} not found")
            
        original = self.sections[section_id]
        new_version_id = f"{original.domain}_v{original.version + 1}_{datetime.now().strftime('%H%M%S')}"
        
        new_section = Section(
            section_id=new_version_id,
            domain=original.domain,
            author=author,
            content=new_content,
            version=original.version + 1,
            parent_version=section_id,
            metadata={"rationale": rationale}
        )
        self.sections[new_version_id] = new_section
        
        change = DocumentChange(
            change_id=uuid.uuid4().hex,
            section_id=new_version_id,
            author=author,
            change_type="edit",
            content_before=original.content,
            content_after=new_content,
            rationale=rationale
        )
        self.history.append(change)
        
        return new_version_id
    
    def merge_to_document(self, section_id: str, coordinator_notes: str = "") -> bool:
        """Merge approved section into main document"""
        if section_id not in self.sections:
            return False
            
        section = self.sections[section_id]
        section.status = SectionStatus.MERGED
        section.updated_at = datetime.now()
        
        self.current_document[section.domain] = section.content
        
        change = DocumentChange(
            change_id=uuid.uuid4().hex,
            section_id=section_id,
            author="coordinator",
            change_type="merge",
            rationale=coordinator_notes
        )
        self.history.append(change)
        
        return True
    
    def get_current_document_markdown(self) -> str:
        """Generate clean markdown from current document state"""
        md_parts = ["# Risk Assessment Report\n"]
        md_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for domain, content in sorted(self.current_document.items()):
            md_parts.append(f"## {domain.replace('_', ' ').title()}\n\n")
            md_parts.append(content)
            md_parts.append("\n\n")
            
        return "".join(md_parts)
