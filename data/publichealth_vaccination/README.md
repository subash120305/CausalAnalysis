# publichealth_vaccination

**Size:** 1,200 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `person_id`

**Type:** Numeric
**Range:** 1.00 to 1200.00
**Mean:** 600.50

### `sms_reminders`

**Type:** Numeric
**Range:** 0.00 to 0.00
**Mean:** 0.00

### `age`

**Type:** Numeric
**Range:** 18.00 to 90.00
**Mean:** 45.55

### `chronic_conditions`

**Type:** Numeric
**Range:** 0.00 to 5.00
**Mean:** 0.99

### `health_literacy_score`

**Type:** Numeric
**Range:** 1.00 to 5.00
**Mean:** 3.02

### `insurance_status`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.86

### `distance_to_clinic_km`

**Type:** Numeric
**Range:** 0.50 to 31.30
**Mean:** 6.00

### `vaccinated`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.85

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/publichealth_vaccination/publichealth_vaccination.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

