from pathlib import Path
import os
from functools import lru_cache
from typing import List, Optional

@lru_cache(maxsize=1)
def find_project_root(start_path: Optional[Path] = None, markers: Optional[List[str]] = None) -> Path:
    """
    Find project root by looking for marker files.
    
    Args:
        start_path: Path to start searching from (defaults to current file's directory)
        markers: List of files/directories that indicate project root
    
    Returns:
        Path: Project root directory
    """
    if markers is None:
        markers = [
            'requirements.txt',
            '.git',
            'README.md',
            '.gitignore'
        ]
    
    if start_path is None:
        start_path = Path(__file__).parent
    else:
        start_path = Path(start_path)
    
    # Search upwards from start_path
    for path in [start_path] + list(start_path.parents):
        if any((path / marker).exists() for marker in markers):
            return path.resolve()
    
    # Fallback to start_path if no markers found
    return start_path.resolve()

class ProjectPaths:
    """Centralized path management for the project."""
    
    def __init__(self, custom_root: Optional[str] = None):
        self._root = Path(custom_root).resolve() if custom_root else find_project_root()
        self._ensure_directories()
    
    @property
    def root(self) -> Path:
        """Project root directory."""
        return self._root

    @property
    def database(self) -> Path:
        """Database directory."""
        return self._root / "database"
    
    @property
    def documents(self) -> Path:
        """Documents directory."""
        return self.database / "documents"
    
    @property
    def vectordb(self) -> Path:
        """Vector database directory."""
        return self.database / "vectordb"
    
    # @property
    # def config(self) -> Path:
    #     """Configuration directory."""
    #     return self._root / "config"
    
    @property
    def logs(self) -> Path:
        """Logs directory."""
        return self._root / "logs"
    
    @property
    def src(self) -> Path:
        """Source directory."""
        return self._root / "src"

    @property
    def utils(self) -> Path:
        """Utils directory."""
        return self.src / "src/utils"

    @property
    def custom_autogen_code(self) -> Path:
        """Custom autogen code directory."""
        return self.src / "src/custom_autogen_code"

    @property
    def temp(self) -> Path:
        """Temporary files directory."""
        return self._root / "temp"
    
    def get_path(self, relative_path: str) -> Path:
        """Get path relative to project root."""
        return self._root / relative_path
    
    def get_database_path(self, db_name: str) -> Path:
        """Get path for a specific database."""
        return self.database / db_name
    
    def _ensure_directories(self):
        """Create common directories if they don't exist."""
        common_dirs = [
            self.documents,
            self.logs,
            self.src,
            self.utils,
            self.custom_autogen_code,
            self.temp
        ]
        
        for directory in common_dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        return f"ProjectPaths(root={self._root})"
    
    def __repr__(self) -> str:
        return self.__str__()

# Global instance - import this from anywhere
paths = ProjectPaths()

# Convenience functions
def get_project_path(relative_path: str) -> Path:
    """Get path relative to project root."""
    return paths.get_path(relative_path)

def get_database_path(db_name: str = "vectordb") -> Path:
    """Get database path."""
    return paths.get_database_path(db_name)

# Pre-defined common paths for easy import
PROJECT_ROOT = paths.root
DATABASE_DIR = paths.database
DOCUMENTS_DIR = paths.documents
VECTORDB_PATH = paths.vectordb
LOGS_DIR = paths.logs
TEMP_DIR = paths.temp
CUSTOM_AUTOGEN_CODE_DIR = paths.custom_autogen_code
SRC_DIR = paths.src
UTILS_DIR = paths.utils

