"""Create project archive excluding unnecessary files."""

import zipfile
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

archive_path = output_dir / "causalbench.zip"

# Directories and patterns to exclude
exclude_dirs = {'.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints', 
                'output', '.venv', 'env', 'venv', '.mypy_cache', 'node_modules'}
exclude_extensions = {'.pyc', '.zip', '.png', '.jpg', '.jpeg', '.gif'}
exclude_patterns = ['/results/', '/data/']

def should_exclude(file_path):
    """Check if file should be excluded."""
    path_str = str(file_path)
    
    # Check for excluded directories
    for part in file_path.parts:
        if part in exclude_dirs:
            return True
    
    # Check extensions (but only for results/data large files)
    if file_path.suffix in exclude_extensions:
        if 'results' in path_str or 'data' in path_str:
            return True
    
    # Check patterns
    for pattern in exclude_patterns:
        if pattern in path_str and file_path.suffix in {'.csv', '.png', '.pdf'}:
            return True
    
    return False

print(f"Creating archive: {archive_path}")
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(project_root):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = Path(root) / file
            
            # Skip if in exclude list
            if should_exclude(file_path):
                continue
            
            # Get relative path for archive
            arcname = file_path.relative_to(project_root)
            zf.write(file_path, arcname)

print(f"Archive created: {archive_path}")
print(f"Archive size: {archive_path.stat().st_size / 1024 / 1024:.2f} MB")

