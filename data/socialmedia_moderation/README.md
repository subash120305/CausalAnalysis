# socialmedia_moderation

**Size:** 1,500 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `user_id`

**Type:** Numeric
**Range:** 1.00 to 1500.00
**Mean:** 750.50

### `ai_moderation`

**Type:** Numeric
**Range:** 0.00 to 0.00
**Mean:** 0.00

### `account_age_days`

**Type:** Numeric
**Range:** 1.00 to 3000.00
**Mean:** 507.68

### `follower_count`

**Type:** Numeric
**Range:** 10.00 to 100000.00
**Mean:** 2936.73

### `engagement_rate`

**Type:** Numeric
**Range:** 0.37 to 71.56
**Mean:** 20.51

### `post_frequency_per_week`

**Type:** Numeric
**Range:** 0.00 to 14.00
**Mean:** 4.95

### `verified_account`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.15

### `user_retained`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.79

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/socialmedia_moderation/socialmedia_moderation.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

