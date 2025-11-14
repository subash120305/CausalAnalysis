# finance_creditcard

**Size:** 1,100 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `customer_id`

**Type:** Numeric
**Range:** 1.00 to 1100.00
**Mean:** 550.50

### `premium_offer`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.29

### `credit_score`

**Type:** Numeric
**Range:** 426.00 to 850.00
**Mean:** 683.40

### `annual_income`

**Type:** Numeric
**Range:** 20000.00 to 447223.00
**Mean:** 73906.14

### `age`

**Type:** Numeric
**Range:** 21.00 to 75.00
**Mean:** 42.60

### `existing_cards`

**Type:** Numeric
**Range:** 0.00 to 8.00
**Mean:** 2.52

### `debt_to_income_ratio`

**Type:** Numeric
**Range:** 0.01 to 0.77
**Mean:** 0.28

### `card_activated`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.71

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/finance_creditcard/finance_creditcard.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

