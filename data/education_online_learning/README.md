# education_online_learning

**Size:** 950 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `student_id`

**Type:** Numeric
**Range:** 1.00 to 950.00
**Mean:** 475.50

### `interactive_videos`

**Type:** Numeric
**Range:** 0.00 to 0.00
**Mean:** 0.00

### `prior_gpa`

**Type:** Numeric
**Range:** 1.00 to 4.00
**Mean:** 2.80

### `study_hours_per_week`

**Type:** Numeric
**Range:** 1.00 to 24.10
**Mean:** 5.99

### `attendance_rate`

**Type:** Numeric
**Range:** 23.40 to 98.70
**Mean:** 71.56

### `parent_education_level`

**Type:** Numeric
**Range:** 1.00 to 4.00
**Mean:** 2.49

### `internet_quality`

**Type:** Numeric
**Range:** 1.00 to 3.00
**Mean:** 2.10

### `final_exam_score`

**Type:** Numeric
**Range:** 69.90 to 100.00
**Mean:** 96.34

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/education_online_learning/education_online_learning.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

