# Bike Sharing Dataset

**Domain:** Transportation

**Description:** Capital Bikeshare (DC) - effect of working days on bike rental demand

**Source:** UCI Machine Learning Repository

**Size:** 731 rows × 12 columns

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
**Mean:** 0.68

### `outcome`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.50

### `season`

**Type:** Numeric
**Range:** 1.00 to 4.00
**Mean:** 2.50

### `yr`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.50

### `mnth`

**Type:** Numeric
**Range:** 1.00 to 12.00
**Mean:** 6.52

### `holiday`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.03

### `weekday`

**Type:** Numeric
**Range:** 0.00 to 6.00
**Mean:** 3.00

### `weathersit`

**Type:** Numeric
**Range:** 1.00 to 3.00
**Mean:** 1.40

### `temp`

**Type:** Numeric
**Range:** 0.06 to 0.86
**Mean:** 0.50

### `atemp`

**Type:** Numeric
**Range:** 0.08 to 0.84
**Mean:** 0.47

### `hum`

**Type:** Numeric
**Range:** 0.00 to 0.97
**Mean:** 0.63

### `windspeed`

**Type:** Numeric
**Range:** 0.02 to 0.51
**Mean:** 0.19

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/bike_sharing/bike_sharing.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('bike_sharing', estimators=['ipw', 'psm', 'dml'])
print(results)
```
