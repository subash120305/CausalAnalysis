# retail_store_layout

**Size:** 380 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `store_id`

**Type:** Numeric
**Range:** 1.00 to 380.00
**Mean:** 190.50

### `optimized_layout`

**Type:** Numeric
**Range:** 1.00 to 1.00
**Mean:** 1.00

### `store_size_sqm`

**Type:** Numeric
**Range:** 500.00 to 3202.00
**Mean:** 1537.13

### `daily_foot_traffic`

**Type:** Numeric
**Range:** 253.00 to 364.00
**Mean:** 301.46

### `competitor_distance_km`

**Type:** Numeric
**Range:** 0.50 to 17.90
**Mean:** 4.23

### `avg_transaction_value`

**Type:** Numeric
**Range:** 10.00 to 95.84
**Mean:** 46.78

### `parking_spaces`

**Type:** Numeric
**Range:** 31.00 to 70.00
**Mean:** 49.87

### `sales_increase_pct`

**Type:** Numeric
**Range:** 11.10 to 40.00
**Mean:** 25.88

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/retail_store_layout/retail_store_layout.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

