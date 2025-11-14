# Communities and Crime Dataset

**Domain:** Social Policy

**Description:** US communities data - relationship between immigrant population and crime rates

**Source:** UCI Machine Learning Repository

**Size:** 1,993 rows × 8 columns

---

## Dataset Structure

- **Treatment Variable:** `treatment`
- **Outcome Variable:** `outcome`
- **Confounders:** 6 variables

---

## Column Descriptions

### `treatment`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.49

### `outcome`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.50

### `population`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.06

### `householdsize`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.46

### `pct_young`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.18

### `pct_divorced`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.50

### `pct_unemployed`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.35

### `median_income`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.28

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/communities_crime/communities_crime.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('communities_crime', estimators=['ipw', 'psm', 'dml'])
print(results)
```
