"""
Run all notebooks programmatically to ensure proper path handling.
"""

import sys
from pathlib import Path
import subprocess
import os

# Set working directory to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)

print(f"Running notebooks from: {project_root}")
print("=" * 50)

notebooks = [
    "notebook_ihdp.ipynb",
    "notebook_twins.ipynb",
    "notebook_sachs.ipynb",
    "notebook_acic.ipynb",
    "notebook_lalonde.ipynb",
]

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

success_count = 0
for notebook in notebooks:
    notebook_path = project_root / "notebooks" / notebook
    print(f"\nProcessing {notebook}...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nbconvert", 
             "--to", "notebook",
             "--execute",
             "--ExecutePreprocessor.timeout=600",
             "--output-dir=str(results_dir)",
             str(notebook_path)],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print(f"[OK] {notebook} completed")
            success_count += 1
        else:
            print(f"[FAIL] {notebook} failed")
            print(f"Error: {result.stderr[:200]}")
    except Exception as e:
        print(f"✗ {notebook} failed with exception: {e}")

print("\n" + "=" * 50)
print(f"Completed: {success_count}/{len(notebooks)} notebooks")

