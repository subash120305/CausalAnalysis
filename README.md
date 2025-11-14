# CausalBench - Interactive Causal Inference Platform

**Production-ready tool for causal inference with 22 diverse datasets, intelligent validation, and automated insights.**

Analyze cause-and-effect relationships using state-of-the-art methods (DoWhy, EconML) with an intuitive Streamlit interface.

---

## 🚀 Quick Start (60 Seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch app
streamlit run app.py

# 3. Browse to http://localhost:8501
```

**That's it!** All 22 datasets are ready to use. Upload your own data or explore built-in datasets.

---

## ✨ Key Features

- **📊 22 Diverse Datasets** - Healthcare, Finance, Education, E-Commerce, and more
- **🆕 Upload Any Dataset** - CSV, Excel with intelligent validation
- **🤖 Auto-Detects Columns** - Finds treatment, outcome, confounders automatically
- **💾 Smart Caching** - Analyzed datasets saved for instant recall
- **🔬 5 Estimation Methods** - IPW, PSM, Doubly Robust, DML, DRLearner
- **📈 Real-Time Visualization** - ATE charts, distributions, comparisons
- **🧠 Intelligent Insights** - Identifies key factors, target groups, actionable recommendations
- **📝 Rich Metadata** - Every dataset has detailed column descriptions
- **🎯 Production-Ready** - Reproducible, tested, documented

---

## 📊 Built-in Datasets (22 Total)

### 🌐 Real Datasets from Online Sources (10)

1. **IHDP** (Healthcare) - 747 rows
   - Infant Health and Development Program
   - Treatment: Specialist home visits → Outcome: Cognitive scores

2. **LaLonde** (Economics) - 445 rows
   - Job Training Study (NSW)
   - Treatment: Job training → Outcome: 1978 earnings

3. **Adult Income** (Social Science) - 32,561 rows
   - Census data
   - Treatment: College education → Outcome: High income (>$50K)

4. **Student Performance** (Education) - 395 rows
   - Portuguese secondary school
   - Treatment: Paid tutoring → Outcome: Math grades

5. **Heart Disease** (Healthcare) - 297 rows
   - Cleveland Clinic data
   - Treatment: Exercise-induced angina → Outcome: Heart disease

6. **Bank Marketing** (Marketing) - 5,000 rows
   - Portuguese bank telemarketing
   - Treatment: Cellular contact → Outcome: Term deposit subscription

7. **Wine Quality** (Food Science) - 1,599 rows
   - Portuguese red wine
   - Treatment: High sulphates → Outcome: Quality rating

8. **Communities & Crime** (Social Policy) - 1,993 rows
   - US communities data
   - Treatment: Immigration levels → Outcome: Crime rates

9. **Bike Sharing** (Transportation) - 731 rows
   - Capital Bikeshare (DC)
   - Treatment: Working day → Outcome: Rental demand

10. **Energy Efficiency** (Energy) - 768 rows
    - Building simulations
    - Treatment: Glazing area → Outcome: Heating load

### 🎲 High-Quality Synthetic Datasets (12)

Generated with realistic relationships and known treatment effects:

11. **Healthcare Hypertension** - 800 patients | ATE: -12 mmHg
12. **E-Commerce Recommendations** - 1,200 users | ATE: +$28
13. **Education Online Learning** - 950 students | ATE: +7.5 points
14. **Finance Credit Card** - 1,100 customers | ATE: +15%
15. **HR Remote Work** - 650 employees | ATE: +0.8 satisfaction
16. **Agriculture Fertilizer** - 500 farms | ATE: +450 kg/hectare
17. **Transportation Ride Share** - 2,000 rides | ATE: +18%
18. **Social Media Moderation** - 1,500 users | ATE: +8% retention
19. **Energy Smart Thermostat** - 850 households | ATE: -180 kWh
20. **Retail Store Layout** - 380 stores | ATE: +12% sales
21. **Telecom 5G Upgrade** - 1,000 customers | ATE: +1.2 satisfaction
22. **Public Health Vaccination** - 1,200 participants | ATE: +22% vaccination

---

## 📁 Dataset Structure

Every dataset has its own folder with full documentation:

```
data/
  ihdp/
    - ihdp.csv           # The actual data
    - README.md          # Column descriptions, usage examples
  lalonde/
    - lalonde.csv
    - README.md
  ... (20 more folders)
```

**Each README includes:**
- Description and source
- Column explanations (what each variable means)
- Value ranges, types, and units
- Impact descriptions (how columns affect outcome)
- Copy-paste usage examples

---

## 📋 Dataset Requirements (Upload Your Own)

Your data needs:

1. **Treatment** (REQUIRED) - Binary column (0/1, Yes/No)
2. **Outcome** (REQUIRED) - Continuous or binary measurement
3. **Confounders** (RECOMMENDED) - Variables affecting both treatment and outcome
4. **Sample Size** (RECOMMENDED) - 100+ observations, 30+ per treatment group

**Example:**
```csv
customer_id,email_sent,age,prior_purchases,purchase_amount
1,1,25,5,120.50
2,0,30,3,0.00
3,1,35,2,89.00
```

System auto-detects: Treatment=`email_sent`, Outcome=`purchase_amount`, Confounders=`age`,`prior_purchases`

---

## 🎯 Common Use Cases

| Domain | Question | Treatment | Outcome |
|--------|----------|-----------|---------|
| **Healthcare** | Does Drug X work? | `received_drug` | `recovery_days` |
| **Marketing** | Do emails increase sales? | `email_sent` | `purchase_amount` |
| **Education** | Does tutoring help? | `tutoring` | `final_score` |
| **Policy** | Does training increase earnings? | `job_training` | `annual_salary` |
| **E-Commerce** | Do recommendations work? | `ai_recommendations` | `purchases` |
| **HR** | Does remote work improve satisfaction? | `remote_work` | `job_satisfaction` |

---

## 💡 How It Works

### 1. Upload → 2. Validate → 3. Analyze → 4. Get Insights

```
Upload CSV
    ↓
System checks: "Is this suitable for causal inference?"
    ↓ YES (Confidence: 90%)
Auto-detects: Treatment, Outcome, Confounders
    ↓
You choose methods: IPW, PSM, DML
    ↓
Run Analysis (30-60s)
    ↓
Results: "Email campaign increases purchases by $15.17"
    ↓
Intelligent Insights: Key factors, target groups, recommendations
    ↓
Cached for instant recall next time!
```

---

## 📊 Understanding Results

**ATE (Average Treatment Effect)**

- **Positive (+$1,800)** → Treatment INCREASES outcome by $1,800
- **Negative (-3.2 days)** → Treatment DECREASES outcome by 3.2 days
- **Near-zero (+$0.50)** → Minimal effect, may not be cost-effective

**Confidence**

- **All methods agree** (IPW≈PSM≈DML) → ✅ High confidence, robust
- **Methods disagree** (IPW=$20, DML=$5) → ⚠️ Trust most robust (DML)

---

## 🧠 Intelligent Insights (NEW!)

After analysis, CausalBench automatically provides:

### 📈 Treatment Effect Interpretation
Plain English explanation:
> "The treatment 'job_training' increases 'annual_salary' by $1,800 on average. This is a moderate positive effect."

### 🔑 Key Factors Affecting Outcome
Identifies which variables matter most:
- **education** (correlation: 0.58) - Higher education predicts better outcomes
- **prior_income** (correlation: 0.39) - Prior earnings influence results
- Shows treated vs control group differences

### ⚠️ Risk & Success Factors
- **Success Factors**: Variables that promote positive outcomes (↑)
- **Risk Factors**: Variables associated with worse outcomes (↓)

### 🎯 Who Benefits Most?
Identifies target groups with heterogeneous treatment effects:
> "Treatment is most effective for: Higher education (effect: $2,500 vs $900 for lower)"

### 💡 Actionable Recommendations
Data-driven suggestions:
- ✓ **IMPLEMENT**: "Treatment shows positive impact. Consider scaling up."
- ✓ **PRIORITIZE**: "Focus on higher education candidates."
- ✓ **LEVERAGE**: "Screen for success factors."
- ✓ **MEASURE ROI**: "Effect size is $1,800. Compare against cost."

**Example Output:**
```
📊 EXECUTIVE SUMMARY
Decision: RECOMMEND
Reason: job_training increases earnings by $1,077

Confidence: HIGH - All methods agree closely (±15%)

Top Actions:
  1. ✓ IMPLEMENT: Training shows positive impact.
  2. ✓ PRIORITIZE: Focus on higher education candidates.
```

---

## 🐛 Troubleshooting

### "Streamlit command not found"
```bash
python -m streamlit run app.py
```

### "No module named 'dowhy'"
```bash
pip install -r requirements.txt
```

### "Dataset not suitable"
- **Missing treatment?** Add binary column (0/1)
- **Too small?** Need 100+ rows
- **Multi-valued treatment?** Recode as binary

### "Analysis failed"
- Remove missing values
- Ensure treatment is 0/1 (not strings)
- Check group sizes (need 30+ per group)
- Try different estimator (DML is most robust)

---

## 💻 Command Line Usage

```bash
# Run single dataset
python -m src.dowhy_pipeline --dataset ihdp --estimator ipw psm dml

# Quick test with sampling
python -m src.dowhy_pipeline --dataset ihdp --sample 500
```

**Python API:**
```python
from src.dowhy_pipeline import run_full_pipeline

results = run_full_pipeline("ihdp", estimators=["ipw", "dml"])
print(results)
```

---

## 📚 API Reference

### Load Datasets
```python
from src.data_loader_new import get_available_datasets, load_dataset, get_dataset_config

# List all datasets
datasets = get_available_datasets()
for name, info in datasets.items():
    print(f"{name}: {info['display_name']} ({info['size']})")

# Load a dataset
df = load_dataset('ihdp')  # Full dataset
df = load_dataset('ihdp', sample=500)  # Sample 500 rows

# Get configuration
config = get_dataset_config('ihdp')
print(f"Treatment: {config['treatment']}")
print(f"Outcome: {config['outcome']}")
print(f"Confounders: {config['confounders']}")
```

### Run Causal Analysis
```python
from src.dowhy_pipeline import run_full_pipeline

# Run multiple estimators
results = run_full_pipeline(
    'ihdp',
    estimators=['ipw', 'psm', 'dml'],
    sample=500
)
print(results)
```

### Generate Insights
```python
from src.intelligent_insights import generate_insights, format_insights_for_display

insights = generate_insights(
    data=df,
    treatment_col='treatment',
    outcome_col='outcome',
    confounders=['age', 'education'],
    ate_result=1234.56,
    estimator_results={'ipw': 1200, 'psm': 1250, 'dml': 1240}
)

# Display formatted insights
print(format_insights_for_display(insights))
```

### Analyze Custom Dataset
```python
from src.column_analyzer import analyze_all_columns, format_column_analysis_for_display

# Analyze uploaded data
column_analysis = analyze_all_columns(df, dataset_name='my_data')

# Display analysis
print(format_column_analysis_for_display(column_analysis))
```

---

## 🔬 Estimation Methods Explained

### 1. IPW (Inverse Propensity Weighting)
- Reweights observations by probability of treatment
- Fast but sensitive to extreme propensity scores
- **Use when**: Large sample, good overlap

### 2. PSM (Propensity Score Matching)
- Matches treated units with similar controls
- Intuitive "apples to apples" comparison
- **Use when**: Want matched pairs, medium sample

### 3. Doubly Robust (DR)
- Combines propensity scores with outcome regression
- Correct if either model is right
- **Use when**: Want extra robustness

### 4. DML (Double Machine Learning)
- Uses ML for both propensity and outcome
- Reduces bias, handles complex relationships
- **Use when**: Large sample, complex confounding

### 5. DRLearner
- Combines doubly robust with ML
- Most flexible, handles heterogeneity
- **Use when**: Expect treatment effects vary by subgroup

---

## 📖 Learn More

- **Individual Dataset READMEs**: See `data/*/README.md` for specific dataset details
- **PYTHON_VERSION_NOTES.md**: Python compatibility information
- **START_HERE.txt**: 60-second quick start guide

---

## 🤝 Contributing

Want to add a dataset?

1. Create folder: `data/your_dataset/`
2. Add CSV: `data/your_dataset/your_dataset.csv`
3. Create README with column descriptions
4. Update `data/dataset_metadata.json`

---

## 📄 License

This project is for educational and research purposes. Individual datasets retain their original licenses.

---

## 🙏 Acknowledgments

**Real Datasets from:**
- UCI Machine Learning Repository
- NBER / Academic Research (LaLonde, IHDP)
- Public Health Studies

**Built with:**
- [DoWhy](https://microsoft.github.io/dowhy/) - Microsoft's causal inference library
- [EconML](https://econml.azurewebsites.net/) - Econometric ML library
- [Streamlit](https://streamlit.io/) - Interactive web interface
- [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/)

---

## 📞 Support

- **Documentation**: This README + individual dataset READMEs
- **Issues**: GitHub Issues
- **Questions**: Check troubleshooting section above

---

**You're ready! Launch the app and explore causal inference with 22 diverse datasets! 🚀**
