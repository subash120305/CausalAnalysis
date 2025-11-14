# energy_smart_thermostat

**Size:** 850 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `household_id`

**Type:** Numeric
**Range:** 1.00 to 850.00
**Mean:** 425.50

### `smart_thermostat`

**Type:** Numeric
**Range:** 1.00 to 1.00
**Mean:** 1.00

### `home_size_sqft`

**Type:** Numeric
**Range:** 800.00 to 3665.00
**Mean:** 1995.03

### `occupants`

**Type:** Numeric
**Range:** 1.00 to 8.00
**Mean:** 3.06

### `home_age_years`

**Type:** Numeric
**Range:** 1.00 to 100.00
**Mean:** 18.95

### `insulation_rating`

**Type:** Numeric
**Range:** 1.00 to 5.00
**Mean:** 2.97

### `baseline_usage_kwh`

**Type:** Numeric
**Range:** 300.00 to 2558.00
**Mean:** 1206.88

### `energy_savings_kwh`

**Type:** Numeric
**Range:** 243.00 to 857.00
**Mean:** 534.11

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/energy_smart_thermostat/energy_smart_thermostat.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

