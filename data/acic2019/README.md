# ACIC 2019 Dataset

## Overview
The Atlantic Causal Inference Conference (ACIC) 2019 Data Challenge dataset is a benchmark for evaluating causal inference methods. The challenge focused on estimating the population average treatment effect (PATE).

## Dataset Structure

The dataset files are in CSV format with the following structure:
- **Y**: Outcome (binary or continuous)
- **A**: Binary treatment indicator (0 or 1)
- **V1, V2, ..., Vp**: Covariates (number varies by dataset)

### File Types

1. **Main datasets** (`testdatasetX.csv`): Contains observed data
   - Columns: Y, A, V1, V2, ..., Vp

2. **Counterfactual files** (`testdatasetX_cf.csv`): Contains ground truth for evaluation
   - ATE: Population average treatment effect E(Y(1) - Y(0))
   - EY1_i: Expected counterfactual outcome under treatment
   - EY0_i: Expected counterfactual outcome under control

### Outcome Generation

- **Continuous Y**: Y = A × EY1_i + (1-A) × EY0_i + ε, where ε ~ N(0, σ²)
- **Binary Y**: Y ~ Bernoulli(A × EY1_i + (1-A) × EY0_i)

## Available Files

- `acic_2019_sample.csv` - Sample dataset (testdataset1 from low-dimensional test set)
- `TestDatasets_lowD/` - Low-dimensional test datasets (10 datasets)
- `acic_test_lowD.zip` - Original download archive
- `ACIC_README.txt` - Original challenge documentation

## Download Sources

The ACIC 2019 challenge provides several dataset collections:

### Test Datasets (Small, for development)
- **Low-dimensional**: https://www.dropbox.com/s/qaj6fjbzorzmwpp/TestDatasets_lowD_Dec28.zip?dl=1
- **High-dimensional**: https://www.dropbox.com/s/7st5ttdihk6dzfz/TestDatasets_highD_Dec28.zip?dl=1

### Challenge Datasets (Large, 3200 datasets each)
- **Low-dimensional**: https://www.dropbox.com/s/g0elnbfmhbf7rr3/low_dimensional_datasets.zip?dl=0
- **High-dimensional**: https://www.dropbox.com/s/k2k1cs42i3pzkuu/high_dimensional_datasets.zip?dl=0

## Usage Example

```python
from src.data_loader import download_acic
import pandas as pd

# Load sample dataset
data = pd.read_csv('data/acic2019/acic_2019_sample.csv')

# Run causal inference
from src.dowhy_pipeline import run_full_pipeline
results = run_full_pipeline(
    dataset_name="acic",
    estimators=["ipw", "psm", "dr", "dml"],
    random_state=42
)
```

## References

- **Official Challenge Page**: https://sites.google.com/view/acic2019datachallenge/data-challenge
- **Challenge Year**: 2019
- **Number of Datasets**: 3,200 per track (low-dim and high-dim)
- **Task**: Estimate population average treatment effect (PATE)

## Citation

If you use this dataset, please cite the ACIC 2019 Data Challenge:

```
Atlantic Causal Inference Conference (ACIC) 2019 Data Challenge
https://sites.google.com/view/acic2019datachallenge/home
```

## Notes

- The population-level effect (PATE) is NOT identical to the sample average treatment effect (SATE)
- Covariates were drawn from publicly available data and simulations
- The test datasets contain ground truth for method evaluation
- Challenge datasets may not include ground truth (competition format)
