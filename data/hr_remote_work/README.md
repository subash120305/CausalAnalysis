# hr_remote_work

**Size:** 650 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `employee_id`

**Type:** Numeric
**Range:** 1.00 to 650.00
**Mean:** 325.50

### `remote_work_option`

**Type:** Numeric
**Range:** 1.00 to 1.00
**Mean:** 1.00

### `tenure_years`

**Type:** Numeric
**Range:** 0.50 to 25.00
**Mean:** 4.79

### `performance_rating`

**Type:** Numeric
**Range:** 1.00 to 5.00
**Mean:** 3.50

### `commute_time_min`

**Type:** Numeric
**Range:** 5.00 to 120.00
**Mean:** 29.35

### `team_size`

**Type:** Numeric
**Range:** 2.00 to 16.00
**Mean:** 7.96

### `has_children`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.40

### `job_satisfaction_score`

**Type:** Numeric
**Range:** 2.21 to 5.00
**Mean:** 4.50

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/hr_remote_work/hr_remote_work.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

