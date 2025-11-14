# telecom_5g_upgrade

**Size:** 1,000 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `customer_id`

**Type:** Numeric
**Range:** 1.00 to 1000.00
**Mean:** 500.50

### `five_g_access`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.54

### `customer_tenure_months`

**Type:** Numeric
**Range:** 1.00 to 120.00
**Mean:** 24.18

### `data_usage_gb`

**Type:** Numeric
**Range:** 1.70 to 200.00
**Mean:** 48.23

### `plan_price`

**Type:** Numeric
**Range:** 20.00 to 125.68
**Mean:** 60.15

### `support_calls_last_year`

**Type:** Numeric
**Range:** 0.00 to 8.00
**Mean:** 2.09

### `device_age_months`

**Type:** Numeric
**Range:** 1.00 to 60.00
**Mean:** 17.25

### `satisfaction_score`

**Type:** Numeric
**Range:** 3.20 to 10.00
**Mean:** 7.14

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/telecom_5g_upgrade/telecom_5g_upgrade.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

