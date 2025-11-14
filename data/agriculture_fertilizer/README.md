# agriculture_fertilizer

**Size:** 500 rows × 8 columns

---

## Dataset Structure


---

## Column Descriptions

### `farm_id`

**Type:** Numeric
**Range:** 1.00 to 500.00
**Mean:** 250.50

### `organic_fertilizer`

**Type:** Numeric
**Range:** 0.00 to 0.00
**Mean:** 0.00

### `soil_quality_score`

**Type:** Numeric
**Range:** 3.10 to 9.70
**Mean:** 6.56

### `rainfall_mm`

**Type:** Numeric
**Range:** 300.00 to 1308.00
**Mean:** 744.24

### `avg_temperature_c`

**Type:** Numeric
**Range:** 10.10 to 33.40
**Mean:** 21.95

### `farm_size_hectares`

**Type:** Numeric
**Range:** 1.00 to 19.00
**Mean:** 5.79

### `irrigation_system`

**Type:** Numeric
**Range:** 0.00 to 1.00
**Mean:** 0.40

### `crop_yield_kg_per_hectare`

**Type:** Numeric
**Range:** 3796.00 to 8000.00
**Mean:** 5955.06

---

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/agriculture_fertilizer/agriculture_fertilizer.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())
```

