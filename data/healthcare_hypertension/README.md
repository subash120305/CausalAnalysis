# healthcare_hypertension

**Size:** 800 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `patient_id`

**Type:** Numeric
**Range:** 1.00 to 800.00
**Mean:** 400.50

### `drug_treatment`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.41

### `age`

**Type:** Numeric
**Range:** 25.00 to 85.00
**Mean:** 54.62

### `bmi`

**Type:** Numeric
**Range:** 18.00 to 42.60
**Mean:** 28.31

### `smoking`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.28

### `baseline_bp`

**Type:** Numeric
**Range:** 120.00 to 180.00
**Mean:** 146.21

### `exercise_hrs_per_week`

**Type:** Numeric
**Range:** 0.00 to 7.00
**Mean:** 2.03

### `bp_reduction`

**Type:** Numeric
**Range:** -24.00 to 41.70
**Mean:** 9.66

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/healthcare_hypertension/healthcare_hypertension.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

