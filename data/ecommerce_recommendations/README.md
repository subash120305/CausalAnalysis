# ecommerce_recommendations

**Size:** 1,200 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `user_id`

**Type:** Numeric
**Range:** 1.00 to 1200.00
**Mean:** 600.50

### `ai_recommendations`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.55

### `user_tenure_days`

**Type:** Numeric
**Range:** 1.00 to 2000.00
**Mean:** 368.12

### `past_purchases`

**Type:** Numeric
**Range:** 1.00 to 16.00
**Mean:** 7.88

### `avg_order_value`

**Type:** Numeric
**Range:** 10.00 to 152.43
**Mean:** 75.81

### `browse_time_min`

**Type:** Numeric
**Range:** 2.20 to 104.10
**Mean:** 29.40

### `email_engagement_score`

**Type:** Numeric
**Range:** 0.01 to 1.00
**Mean:** 0.51

### `purchase_amount_30d`

**Type:** Numeric
**Range:** 61.20 to 233.64
**Mean:** 121.21

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/ecommerce_recommendations/ecommerce_recommendations.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

