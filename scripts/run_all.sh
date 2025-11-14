#!/bin/bash
# Run all notebooks headlessly and save outputs

set -e  # Exit on error

echo "=========================================="
echo "Running CausalBench notebooks headlessly"
echo "=========================================="

# Create results directory
mkdir -p results
mkdir -p output

# Log file
LOG_FILE="results/logs.txt"
echo "$(date): Starting notebook execution" > "$LOG_FILE"

# List of notebooks
NOTEBOOKS=(
    "notebook_ihdp.ipynb"
    "notebook_twins.ipynb"
    "notebook_sachs.ipynb"
    "notebook_acic.ipynb"
    "notebook_lalonde.ipynb"
)

# Run each notebook
for notebook in "${NOTEBOOKS[@]}"; do
    echo "Processing $notebook..."
    echo "$(date): Processing $notebook" >> "$LOG_FILE"
    
    if python -m nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=600 \
        --output-dir=results \
        "notebooks/$notebook" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ $notebook completed"
        echo "$(date): ✓ $notebook completed" >> "$LOG_FILE"
    else
        echo "✗ $notebook failed"
        echo "$(date): ✗ $notebook failed" >> "$LOG_FILE"
        # Continue with other notebooks
    fi
done

# Generate summary PDF
echo "Generating summary PDF..."
python scripts/generate_summary.py

# Create zip archive
echo "Creating archive..."
cd ..
mkdir -p output
# Note: zip syntax varies by OS, using basic exclusion
zip -r output/causalbench.zip . \
    -x "*.git/*" \
    -x "*__pycache__/*" \
    -x "*.pyc" \
    -x "*/.pytest_cache/*" \
    -x "*/.ipynb_checkpoints/*" \
    -x "output/*" \
    2>&1 | grep -v "zip warning" || true

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Results saved to: results/"
echo "Archive created: output/causalbench.zip"

# Print SHA256 if available
if command -v sha256sum &> /dev/null; then
    echo "SHA256: $(sha256sum output/causalbench.zip | cut -d' ' -f1)"
elif command -v shasum &> /dev/null; then
    echo "SHA256: $(shasum -a 256 output/causalbench.zip | cut -d' ' -f1)"
fi

echo "$(date): All notebooks processed" >> "$LOG_FILE"
