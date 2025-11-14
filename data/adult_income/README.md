# Adult Income Dataset

**Domain:** Social Science

**Description:** Census data studying effect of college education on high income (>$50K)

**Source:** UCI Machine Learning Repository

**Size:** 32,561 rows × 30 columns

---

## Dataset Structure

- **Treatment Variable:** `treatment`
- **Outcome Variable:** `outcome`
- **Confounders:** 4 variables

---

## Column Descriptions

### `treatment`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.25

### `age`

**Type:** Numeric
**Range:** 17.00 to 90.00
**Mean:** 38.58

### `education_num`

**Type:** Numeric
**Range:** 1.00 to 16.00
**Mean:** 10.08

### `hours_per_week`

**Type:** Numeric
**Range:** 1.00 to 99.00
**Mean:** 40.44

### `sex`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.67

### `outcome`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.24

### `race_Asian-Pac-Islander`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.03

### `race_Black`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.10

### `race_Other`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.01

### `race_White`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.85

### `marital_status_Married-AF-spouse`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.00

### `marital_status_Married-civ-spouse`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.46

### `marital_status_Married-spouse-absent`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.01

### `marital_status_Never-married`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.33

### `marital_status_Separated`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.03

### `marital_status_Widowed`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.03

### `occupation_Adm-clerical`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.12

### `occupation_Armed-Forces`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.00

### `occupation_Craft-repair`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.13

### `occupation_Exec-managerial`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.12

### `occupation_Farming-fishing`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.03

### `occupation_Handlers-cleaners`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.04

### `occupation_Machine-op-inspct`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.06

### `occupation_Other-service`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.10

### `occupation_Priv-house-serv`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.00

### `occupation_Prof-specialty`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.13

### `occupation_Protective-serv`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.02

### `occupation_Sales`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.11

### `occupation_Tech-support`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.03

### `occupation_Transport-moving`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.05

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/adult_income/adult_income.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

### Causal Analysis

```python
from src.dowhy_pipeline import run_full_pipeline

# Run causal inference
results = run_full_pipeline('adult_income', estimators=['ipw', 'psm', 'dml'])
print(results)
```
