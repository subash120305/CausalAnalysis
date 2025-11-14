# transportation_rideshare

**Size:** 2,000 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `ride_id`

**Type:** Numeric
**Range:** 1.00 to 2000.00
**Mean:** 1000.50

### `dynamic_pricing`

**Type:** Numeric
**Range:** 0.00 to 0.00
**Mean:** 0.00

### `hour_of_day`

**Type:** Numeric
**Range:** 0.00 to 23.00
**Mean:** 11.54

### `day_of_week`

**Type:** Numeric
**Range:** 0.00 to 6.00
**Mean:** 3.00

### `trip_distance_km`

**Type:** Numeric
**Range:** 1.00 to 26.10
**Mean:** 6.06

### `driver_rating`

**Type:** Numeric
**Range:** 3.62 to 5.00
**Mean:** 4.68

### `surge_multiplier`

**Type:** Numeric
**Range:** 1.00 to 2.00
**Mean:** 1.31

### `ride_accepted`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.65

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/transportation_rideshare/transportation_rideshare.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

