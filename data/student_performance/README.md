# Student Performance Dataset

**Domain:** Education

**Description:** Portuguese secondary school data - effect of paid tutoring on math grades

**Source:** UCI Machine Learning Repository

**Size:** 395 rows × 10 columns

---

## Dataset Structure

- **Treatment Variable:** `treatment`
- **Outcome Variable:** `outcome`
- **Confounders:** 8 variables

---

## Column Descriptions

### `treatment`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.46

### `outcome`

**Type:** Numeric
**Range:** 0.00 to 20.00
**Mean:** 10.42

### `age`

**Type:** Numeric
**Range:** 15.00 to 22.00
**Mean:** 16.70

### `Medu`

**Type:** Numeric
**Range:** 0.00 to 4.00
**Mean:** 2.75

### `Fedu`

**Type:** Numeric
**Range:** 0.00 to 4.00
**Mean:** 2.52

### `studytime`

**Type:** Numeric
**Range:** 1.00 to 4.00
**Mean:** 2.04

### `failures`

**Type:** Numeric
**Range:** 0.00 to 3.00
**Mean:** 0.33

### `absences`

**Type:** Numeric
**Range:** 0.00 to 75.00
**Mean:** 5.71

### `G1`

**Type:** Numeric
**Range:** 3.00 to 19.00
**Mean:** 10.91

### `G2`

**Type:** Numeric
**Range:** 0.00 to 19.00
**Mean:** 10.71

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/student_performance/student_performance.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('student_performance', estimators=['ipw', 'psm', 'dml'])
print(results)
```
