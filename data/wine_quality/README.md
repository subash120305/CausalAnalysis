# Wine Quality Dataset

**Domain:** Food Science

**Description:** Portuguese red wine - effect of sulphate levels on wine quality rating

**Source:** UCI Machine Learning Repository

**Size:** 1,599 rows × 12 columns

---

## Dataset Structure

- **Treatment Variable:** `treatment`
- **Outcome Variable:** `outcome`
- **Confounders:** 10 variables

---

## Column Descriptions

### `treatment`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.48

### `outcome`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.53

### `fixed acidity`

**Type:** Numeric
**Range:** 4.60 to 15.90
**Mean:** 8.32

### `volatile acidity`

**Type:** Numeric
**Range:** 0.12 to 1.58
**Mean:** 0.53

### `citric acid`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.27

### `residual sugar`

**Type:** Numeric
**Range:** 0.90 to 15.50
**Mean:** 2.54

### `chlorides`

**Type:** Numeric
**Range:** 0.01 to 0.61
**Mean:** 0.09

### `free sulfur dioxide`

**Type:** Numeric
**Range:** 1.00 to 72.00
**Mean:** 15.87

### `total sulfur dioxide`

**Type:** Numeric
**Range:** 6.00 to 289.00
**Mean:** 46.47

### `density`

**Type:** Numeric
**Range:** 0.99 to 1.00
**Mean:** 1.00

### `pH`

**Type:** Numeric
**Range:** 2.74 to 4.01
**Mean:** 3.31

### `alcohol`

**Type:** Numeric
**Range:** 8.40 to 14.90
**Mean:** 10.42

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/wine_quality/wine_quality.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('wine_quality', estimators=['ipw', 'psm', 'dml'])
print(results)
```
