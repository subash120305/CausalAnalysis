# LaLonde Job Training Study

**Domain:** Economics

**Description:** National Supported Work Demonstration - evaluating effect of job training on earnings

**Source:** Dehejia & Wahba (1999)

**Size:** 445 rows × 11 columns

---

## Dataset Structure

- **Treatment Variable:** `treat`
- **Outcome Variable:** `re78`
- **Confounders:** 8 variables

---

## Column Descriptions

### `data_id`

**Type:** Categorical
**Unique values:** 1

### `treat`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.42

### `age`

**Type:** Numeric
**Range:** 17.00 to 55.00
**Mean:** 25.37

### `education`

**Type:** Numeric
**Range:** 3.00 to 16.00
**Mean:** 10.20

### `black`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.83

### `hispanic`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.09

### `married`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.17

### `nodegree`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.78

### `re74`

**Type:** Numeric
**Range:** 0.00 to 39570.68
**Mean:** 2102.27

### `re75`

**Type:** Numeric
**Range:** 0.00 to 25142.24
**Mean:** 1377.14

### `re78`

**Type:** Numeric
**Range:** 0.00 to 60307.93
**Mean:** 5300.76

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/lalonde/lalonde.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('lalonde', estimators=['ipw', 'psm', 'dml'])
print(results)
```
