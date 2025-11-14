# Bank Marketing Campaign

**Domain:** Marketing

**Description:** Portuguese bank telemarketing - effect of cellular contact vs landline on subscription

**Source:** UCI Machine Learning Repository

**Size:** 5,000 rows × 9 columns

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
**Mean:** 0.63

### `outcome`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.11

### `age`

**Type:** Numeric
**Range:** 17.00 to 88.00
**Mean:** 39.99

### `duration`

**Type:** Numeric
**Range:** 2.00 to 2420.00
**Mean:** 260.17

### `campaign`

**Type:** Numeric
**Range:** 1.00 to 42.00
**Mean:** 2.53

### `pdays`

**Type:** Numeric
**Range:** 0.00 to 999.00
**Mean:** 960.47

### `previous`

**Type:** Numeric
**Range:** 0.00 to 6.00
**Mean:** 0.17

### `emp.var.rate`

**Type:** Numeric
**Range:** -3.40 to 1.40
**Mean:** 0.09

### `cons.price.idx`

**Type:** Numeric
**Range:** 92.20 to 94.77
**Mean:** 93.58

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/bank_marketing/bank_marketing.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('bank_marketing', estimators=['ipw', 'psm', 'dml'])
print(results)
```
