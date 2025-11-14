# IHDP - Infant Health and Development Program

**Domain:** Healthcare

**Description:** Randomized trial studying effect of specialist home visits on cognitive development in premature infants

**Source:** https://github.com/AMLab-Amsterdam/CEVAE

**Size:** 747 rows × 30 columns

---

## Dataset Structure

- **Treatment Variable:** `treatment`
- **Outcome Variable:** `y_factual`
- **Confounders:** 25 variables

---

## Column Descriptions

### `treatment`

**Description:** Received specialist home visits

**Type:** binary
**Range:** 0.00 to 1.00
**Mean:** 0.19

### `y_factual`

**Description:** Cognitive test score at age 3

**Type:** continuous
**Range:** -1.54 to 11.27
**Mean:** 3.16

### `x1`

**Type:** Numeric
**Range:** -1.04 to 10.17
**Mean:** 5.70

### `x2`

**Type:** Numeric
**Range:** 0.92 to 9.82
**Mean:** 2.43

### `x3`

**Type:** Numeric
**Range:** 5.59 to 7.95
**Mean:** 6.45

### `x4`

**Type:** Numeric
**Range:** -2.73 to 1.51
**Mean:** 0.00

### `x5`

**Type:** Numeric
**Range:** -3.80 to 2.60
**Mean:** -0.00

### `x6`

**Type:** Numeric
**Range:** -1.85 to 2.99
**Mean:** -0.00

### `x7`

**Type:** Numeric
**Range:** -0.88 to 2.24
**Mean:** -0.00

### `x8`

**Type:** Numeric
**Range:** -5.13 to 2.37
**Mean:** 0.00

### `x9`

**Type:** Numeric
**Range:** -1.85 to 2.95
**Mean:** -0.00

### `x10`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.51

### `x11`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.09

### `x12`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.52

### `x13`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.36

### `x14`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.27

### `x15`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.22

### `x16`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.36

### `x17`

**Type:** Numeric
**Range:** 1.00 to 2.00
**Mean:** 1.46

### `x18`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.14

### `x19`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.96

### `x20`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.59

### `x21`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.96

### `x22`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.14

### `x23`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.14

### `x24`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.16

### `x25`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.08

### `x26`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.07

### `x27`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.13

### `x28`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.16

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/ihdp/ihdp.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('ihdp', estimators=['ipw', 'psm', 'dml'])
print(results)
```
