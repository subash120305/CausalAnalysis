# Energy Efficiency Dataset

**Domain:** Energy

**Description:** Building simulations - effect of glazing area on heating load

**Source:** UCI Machine Learning Repository

**Size:** 768 rows × 9 columns

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
**Mean:** 0.50

### `outcome`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.50

### `X1`

**Type:** Numeric
**Range:** 0.62 to 0.98
**Mean:** 0.76

### `X2`

**Type:** Numeric
**Range:** 514.50 to 808.50
**Mean:** 671.71

### `X3`

**Type:** Numeric
**Range:** 245.00 to 416.50
**Mean:** 318.50

### `X4`

**Type:** Numeric
**Range:** 110.25 to 220.50
**Mean:** 176.60

### `X5`

**Type:** Numeric
**Range:** 3.50 to 7.00
**Mean:** 5.25

### `X7`

**Type:** Numeric
**Range:** 0.00 to 0.40
**Mean:** 0.23

### `X8`

**Type:** Numeric
**Range:** 0.00 to 5.00
**Mean:** 2.81

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/energy_efficiency/energy_efficiency.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('energy_efficiency', estimators=['ipw', 'psm', 'dml'])
print(results)
```
