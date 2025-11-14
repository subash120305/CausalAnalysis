# Heart Disease Dataset

**Domain:** Healthcare

**Description:** Cleveland Clinic data - association between exercise-induced angina and heart disease

**Source:** UCI Machine Learning Repository

**Size:** 297 rows × 9 columns

---

## Dataset Structure

- **Treatment Variable:** `treatment`
- **Outcome Variable:** `outcome`
- **Confounders:** 7 variables

---

## Column Descriptions

### `treatment`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.33

### `outcome`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.46

### `age`

**Type:** Numeric
**Range:** 29.00 to 77.00
**Mean:** 54.54

### `sex`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.68

### `cp`

**Type:** Numeric
**Range:** 1.00 to 4.00
**Mean:** 3.16

### `trestbps`

**Type:** Numeric
**Range:** 94.00 to 200.00
**Mean:** 131.69

### `chol`

**Type:** Numeric
**Range:** 126.00 to 564.00
**Mean:** 247.35

### `thalach`

**Type:** Numeric
**Range:** 71.00 to 202.00
**Mean:** 149.60

### `oldpeak`

**Type:** Numeric
**Range:** 0.00 to 6.20
**Mean:** 1.06

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/heart_disease/heart_disease.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('heart_disease', estimators=['ipw', 'psm', 'dml'])
print(results)
```
