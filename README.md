# CausalBench

A full end-to-end reproducible Python project demonstrating causal identification, estimation, discovery, and sensitivity analysis using DoWhy and EconML on standard causal ML benchmarks.

## ⚡ Quick Start (60 Seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup all datasets (run once)
python setup_all_datasets.py

# 3. Launch interactive app
streamlit run app.py

# 4. Open browser at http://localhost:8501
```

**Done!** All datasets ready. Try analyzing the LaLonde job training dataset to see if training increases earnings!

## Features

### Core Capabilities

- **Interactive Streamlit Dashboard**: User-friendly web interface for causal analysis
- **Custom Dataset Upload**: Upload your own CSV files with automatic validation
- **Multiple Built-in Datasets**: IHDP, Twins, Sachs, ACIC, Lalonde + 25+ sample datasets
- **Estimation Methods**: IPW, Propensity Score Matching, Doubly Robust, DML, DRLearner
- **Discovery Methods**: PC, FCI, NOTEARS (structure learning)
- **Intelligent Insights**: Automated interpretation of causal effects with statistical confidence

### Developer Features

- **Reproducible**: Fixed random seeds, comprehensive logging
- **Docker Support**: Fully containerized execution
- **Headless Execution**: Notebooks can run without Jupyter interface
- **Automated Testing**: Pytest suite for data loaders and pipelines
- **Dataset Validation**: Automatic checks for treatment/outcome/confounders

## Requirements

- Python 3.10+ (see [Python Version Compatibility](#python-version-compatibility) below)
- 8-16 GB RAM (no GPU required)
- Internet connection for dataset downloads
- Graphviz (optional, for DAG visualization): Download from https://graphviz.org/download/

### Python Version Compatibility

This project supports **Python 3.10-3.13** with conditional dependencies:

| Python Version | DoWhy | EconML | NOTEARS | Status |
|----------------|-------|--------|---------|--------|
| 3.10-3.12 | 0.11.0+ | 0.14.0+ | ✅ Available | ✅ **Recommended** |
| 3.13 | 0.8.x | 0.13.0+ | ❌ Not available | ⚠️ Supported (with limitations) |

**Feature Availability on Python 3.13:**

- ✅ DoWhy estimation methods (IPW, PSM, DR)
- ✅ EconML estimators (DML, DRLearner)
- ✅ PC and FCI discovery algorithms
- ❌ NOTEARS discovery (will be skipped gracefully)

**Recommendation**: Use Python 3.10, 3.11, or 3.12 for full feature availability.

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

### Option A: Interactive Web App (Recommended for Beginners)

```bash
# 1. Setup datasets (one-time)
python setup_all_datasets.py

# 2. Launch dashboard
streamlit run app.py

# 3. Open http://localhost:8501 in browser
```

**Try it now:**

1. Select "Built-in Datasets" → "lalonde" (job training study)
2. Choose estimators: IPW, PSM, DML
3. Set sample size: 300
4. Click "Run Causal Analysis"
5. View results: ATE ≈ $900-1800 (training increases earnings!)

**Upload your own data:**

1. Click "Upload Custom Data"
2. Upload a CSV with treatment/outcome columns
3. System validates automatically and suggests column mappings
4. Run analysis and get causal estimates!

### Option B: Command-Line Interface

```bash
# Run IHDP with default estimators
python -m src.dowhy_pipeline --dataset ihdp --estimator ipw psm dr dml

# Run with sampling for quick tests (1000 rows)
python -m src.dowhy_pipeline --dataset twins --sample 1000 --estimator ipw psm

# Run all estimators on Lalonde
python -m src.dowhy_pipeline --dataset lalonde --estimator ipw psm dr dml drlearner
```

### Option C: Run All Notebooks

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
├── app.py                           # 🎨 Streamlit interactive dashboard
├── setup_all_datasets.py            # 📦 One-click dataset downloader
├── src/
│   ├── data_loader.py               # Dataset download & validation
│   ├── dataset_validator.py         # Custom dataset validation
│   ├── custom_dataset_manager.py    # Upload & manage custom datasets
│   ├── column_analyzer.py           # Auto-detect treatment/outcome columns
│   ├── intelligent_insights.py      # AI-powered result interpretation
│   ├── dowhy_pipeline.py            # Main DoWhy pipeline
│   ├── econml_estimators.py         # EconML wrappers (DML, DRLearner)
│   ├── discovery.py                 # Causal discovery (PC, FCI, NOTEARS)
│   ├── dag_builder.py               # DAG visualization utilities
│   ├── metrics.py                   # Evaluation metrics (ATE-RMSE, PEHE)
│   └── viz.py                       # Plotting utilities
├── notebooks/
│   ├── notebook_ihdp.ipynb          # Example: Healthcare study
│   ├── notebook_twins.ipynb         # Example: Mortality study
│   ├── notebook_sachs.ipynb         # Example: Protein networks
│   ├── notebook_acic.ipynb          # Example: ACIC benchmark
│   └── notebook_lalonde.ipynb       # Example: Job training
├── data/
│   ├── ihdp/                        # Built-in: IHDP dataset
│   ├── twins/                       # Built-in: Twins dataset
│   ├── lalonde/                     # Built-in: LaLonde dataset
│   ├── sachs/                       # Built-in: Sachs protein data
│   ├── acic2019/                    # Built-in: ACIC 2019 benchmark
│   ├── sample/                      # 🆕 Sample datasets for testing uploads
│   │   ├── marketing_campaign.csv
│   │   ├── medical_trial.csv
│   │   ├── education_study.csv
│   │   └── job_training.csv
│   └── custom_uploads/              # 🆕 User uploaded datasets
├── scripts/
│   ├── run_all.sh                   # Headless notebook execution
│   └── generate_summary.py          # Summary PDF generation
├── tests/
│   ├── test_data_loader.py
│   └── test_dowhy_pipeline.py
├── results/                         # Output directory for analysis
├── Dockerfile
├── environment.yml
├── requirements.txt
└── README.md
```

## Datasets

### Built-in Benchmark Datasets (Auto-downloaded)

#### IHDP (Infant Health and Development Program)

- **Type**: Semi-synthetic observational study
- **Size**: 747 rows, 25 confounders
- **Treatment**: Binary (early intervention program)
- **Outcome**: Continuous (cognitive test scores)
- **Ground Truth**: Available (semi-synthetic)

#### Twins

- **Type**: Processed twins benchmark
- **Size**: 11,400 rows
- **Source**: DoWhy built-in dataset
- **Treatment**: Binary (lighter vs heavier twin)
- **Outcome**: Binary (mortality)

#### Lalonde

- **Type**: Policy evaluation (NSW job training program)
- **Size**: 614 rows
- **Source**: Dehejia-Wahba (1999)
- **Treatment**: Binary (job training participation)
- **Outcome**: Continuous (earnings in 1978)
- **Use Case**: Classic example of treatment effect estimation

#### Sachs

- **Type**: Protein signaling network
- **Size**: 853 rows, 11 proteins
- **Purpose**: Causal discovery (DAG inference)
- **Ground Truth**: Known protein interaction network
- **Use Case**: Structure learning validation

#### ACIC 2019 (Atlantic Causal Inference Conference)

- **Type**: Synthetic benchmark datasets
- **Size**: 10 test datasets (low-dimensional)
- **Features**: Y (outcome), A (treatment), V1-Vp (covariates)
- **Ground Truth**: Population ATE available for evaluation
- **Download**: Auto-downloaded (3200 challenge datasets available separately)

### Sample Datasets for Testing Uploads

Located in `data/sample/`:

- **marketing_campaign.csv** - Email campaign effectiveness (ATE ≈ $15)
- **medical_trial.csv** - Drug trial outcome (ATE ≈ -3 days hospital stay)
- **education_study.csv** - Tutoring program (ATE ≈ +8 test points)
- **job_training.csv** - Skills training (ATE ≈ +$1800 income)

### 25+ Additional Datasets

CausalBench includes diverse real-world datasets across domains:

- **Healthcare**: Heart disease, hypertension treatment
- **Finance**: Credit card offers, loan programs
- **Education**: Online learning, student performance
- **Retail**: Store layout, product recommendations
- **Energy**: Smart thermostats, efficiency programs
- **Public Health**: Vaccination campaigns
- **And many more...**

See [data/](data/) folder for complete list.

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

### Common Issues

#### "Command not found" when running streamlit

```bash
python -m streamlit run app.py
```

#### "Module not found" errors

```bash
pip install -r requirements.txt
```

#### Datasets show "N/A" or not loading

```bash
python setup_all_datasets.py
```

#### Port already in use (8501)

```bash
streamlit run app.py --server.port 8502
```

#### Dataset Download Fails

1. Check `data/link_checks.json` for URL status
2. Check `results/logs.txt` for detailed error messages
3. For Kaggle datasets, set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables
4. ACIC 2019 test datasets are auto-downloaded (full challenge datasets require manual download)

#### Memory Issues

Use the `--sample` flag to reduce dataset size:

```bash
python -m src.dowhy_pipeline --dataset ihdp --sample 1000
```

Or adjust sample size in the Streamlit interface.

#### Graphviz Errors (for DAG visualization)

On Linux:

```bash
sudo apt-get install graphviz
```

On macOS:

```bash
brew install graphviz
```

On Windows: Install Graphviz from https://graphviz.org/

#### Python 3.13 Issues

If you encounter package installation errors on Python 3.13:

1. Try Python 3.10-3.12 instead (recommended)
2. Install packages individually to identify problematic ones
3. NOTEARS will be skipped on Python 3.13 (expected behavior)

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
