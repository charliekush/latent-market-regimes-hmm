from pathlib import Path
import os 

def get_project_root() -> Path:
    """
    Finds the project root
    """
    current_path = Path(os.path.abspath(__file__)).resolve()
    for parent in current_path.parents:
        if (parent / ".git").is_dir():
            return parent
    return current_path.parent