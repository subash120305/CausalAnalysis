# CausalBench

A full end-to-end reproducible Python project demonstrating causal identification, estimation, discovery, and sensitivity analysis using DoWhy and EconML on standard causal ML benchmarks.

## Features

- **Multiple Datasets**: IHDP, Twins, Sachs, ACIC, Lalonde
- **Estimation Methods**: IPW, Propensity Score Matching, Doubly Robust, DML, DRLearner
- **Discovery Methods**: PC, FCI, NOTEARS
- **Reproducible**: Fixed random seeds, comprehensive logging
- **Docker Support**: Fully containerized execution
- **Headless Execution**: Notebooks can run without Jupyter interface

## Requirements

- Python 3.10+ (3.10-3.12 recommended; 3.13 may have compatibility issues with some packages)
- 8-16 GB RAM (no GPU required)
- Internet connection for dataset downloads
- Graphviz (optional, for DAG visualization): Download from https://graphviz.org/download/

## Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate causalbench
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

**Note**: If you encounter installation errors on Python 3.13, try using Python 3.10-3.12, or install packages individually to identify problematic ones.

### Option 3: Docker

```bash
docker build -t causalbench .
docker run -v $(pwd)/results:/app/results causalbench
```

## Quick Start

### Run a Single Dataset

```bash
# Run IHDP with default estimators
python -m src.dowhy_pipeline --dataset ihdp --estimator ipw psm dr dml

# Run with sampling for quick tests (1000 rows)
python -m src.dowhy_pipeline --dataset twins --sample 1000 --estimator ipw psm

# Run all estimators on Lalonde
python -m src.dowhy_pipeline --dataset lalonde --estimator ipw psm dr dml drlearner
```

### Run All Notebooks

```bash
# Execute all notebooks headlessly
bash scripts/run_all.sh
```

This will:
1. Execute all notebooks in `notebooks/`
2. Save results to `results/`
3. Generate summary PDF at `results/summary.pdf`
4. Create archive at `output/causalbench.zip`

## Project Structure

```
CausalBench/
├── src/
│   ├── data_loader.py      # Dataset download & validation
│   ├── dag_builder.py       # DAG visualization utilities
│   ├── dowhy_pipeline.py    # Main DoWhy pipeline
│   ├── econml_estimators.py # EconML wrappers (DML, DRLearner)
│   ├── discovery.py         # Causal discovery (PC, FCI, NOTEARS)
│   ├── metrics.py           # Evaluation metrics (ATE-RMSE, PEHE)
│   └── viz.py               # Plotting utilities
├── notebooks/
│   ├── notebook_ihdp.ipynb
│   ├── notebook_twins.ipynb
│   ├── notebook_sachs.ipynb
│   ├── notebook_acic.ipynb
│   └── notebook_lalonde.ipynb
├── scripts/
│   ├── run_all.sh           # Headless notebook execution
│   └── generate_summary.py  # Summary PDF generation
├── tests/
│   ├── test_data_loader.py
│   └── test_dowhy_pipeline.py
├── data/                     # Downloaded datasets
├── results/                  # Output directory
├── Dockerfile
├── environment.yml
├── requirements.txt
└── README.md
```

## Datasets

### IHDP (Infant Health and Development Program)
- **Type**: Semi-synthetic observational study
- **Features**: 25 confounders, binary treatment, continuous outcome
- **Ground Truth**: Available (semi-synthetic)

### Twins
- **Type**: Processed twins benchmark
- **Source**: DoWhy built-in dataset
- **Features**: Multiple confounders, treatment effect on mortality

### Sachs
- **Type**: Protein signaling network
- **Purpose**: Causal discovery (DAG inference)
- **Ground Truth**: Known protein interaction network

### ACIC (Atlantic Causal Inference Conference)
- **Type**: Synthetic benchmark datasets
- **Year**: 2019 challenge data
- **Note**: May require manual download from official challenge page

### Lalonde
- **Type**: Policy evaluation (NSW job training program)
- **Source**: Dehejia-Wahba (1999)
- **Features**: Observational data with treatment/control groups

## Methods

### Estimation
- **IPW (Inverse Propensity Weighting)**: `backdoor.propensity_score_weighting`
- **PSM (Propensity Score Matching)**: `backdoor.propensity_score_matching`
- **Doubly Robust**: `backdoor.econometric.doubly_robust`
- **DML (Double Machine Learning)**: EconML `LinearDML`
- **DRLearner**: EconML `DRLearner`

### Discovery
- **PC Algorithm**: Constraint-based (causal-learn)
- **FCI**: Fast Causal Inference with latent confounders
- **NOTEARS**: Continuous optimization for DAG learning

## Usage Examples

### Command-Line Interface

```bash
# Basic usage
python -m src.dowhy_pipeline --dataset ihdp

# With specific estimators
python -m src.dowhy_pipeline --dataset twins --estimator ipw psm dr

# Quick run with sampling
python -m src.dowhy_pipeline --dataset lalonde --sample 500

# Custom output directory
python -m src.dowhy_pipeline --dataset ihdp --output-dir ./my_results
```

### Python API

```python
from src.dowhy_pipeline import run_full_pipeline
from pathlib import Path

results = run_full_pipeline(
    dataset_name="ihdp",
    estimators=["ipw", "psm", "dr", "dml"],
    sample=1000,  # Optional: reduce dataset size
    output_dir=Path("results"),
    random_state=42
)

print(results)
```

### Causal Discovery (Sachs)

```python
from src.discovery import compare_discovery_methods
import pandas as pd

data = pd.read_csv("data/sachs/sachs_data.csv")
results = compare_discovery_methods(
    data,
    methods=["pc", "fci", "notears"],
    output_dir=Path("results/sachs")
)
```

## Output

After running `scripts/run_all.sh`, you'll find:

- `results/<dataset>/estimators_summary.csv` - ATE estimates and runtimes
- `results/<dataset>/sample_plots/` - Visualizations (ATE comparison, ITE scatter)
- `results/sachs/discovery_precision_recall.csv` - Discovery metrics
- `results/sachs/sample_plots/*.png` - Discovered DAG visualizations
- `results/summary.pdf` - Auto-generated summary report
- `results/logs.txt` - Execution logs with timestamps
- `data/link_checks.json` - URL status for all dataset links

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest -v tests/

# Run specific test
python -m pytest tests/test_data_loader.py -v

# With coverage
python -m pytest --cov=src tests/
```

## Reproducibility

- **Random Seed**: Fixed to 42 throughout
- **Environment**: Locked Python 3.10 + requirements versions
- **Logging**: All operations logged with timestamps
- **Validation**: Schema checks for downloaded datasets
- **Provenance**: README.txt saved in each dataset directory with URL and timestamp

## Troubleshooting

### Dataset Download Fails

1. Check `data/link_checks.json` for URL status
2. Check `results/logs.txt` for detailed error messages
3. For Kaggle datasets, set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables
4. Some datasets (e.g., ACIC) may require manual download from official sources

### Memory Issues

Use the `--sample` flag to reduce dataset size:

```bash
python -m src.dowhy_pipeline --dataset ihdp --sample 1000
```

### Graphviz Errors

On Linux:
```bash
sudo apt-get install graphviz
```

On macOS:
```bash
brew install graphviz
```

On Windows: Install Graphviz from https://graphviz.org/

## Performance Notes

- **CPU-only**: No GPU required
- **Sampling**: Use `--sample` for quick runs on large datasets
- **Parallel Execution**: Notebooks run sequentially to avoid conflicts
- **Caching**: Datasets are cached in `data/` after first download

## Citation

If you use CausalBench in your research, please cite:

```bibtex
@software{causalbench2024,
  title={CausalBench: Reproducible Causal Inference Pipelines},
  author={},
  year={2024},
  url={https://github.com/subash120305/CausalBench}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Acknowledgments

- DoWhy (https://www.pywhy.org/)
- EconML (https://econml.azurewebsites.net/)
- Causal-learn (https://github.com/py-why/causal-learn)

## Contact

For issues or questions, please open an issue on GitHub.
