"""
Streamlit interactive dashboard for CausalBench with custom dataset upload.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dowhy_pipeline import load_dataset, get_dataset_config, run_full_pipeline
from src.data_loader import RANDOM_SEED
from src.dataset_validator import validate_for_causal_inference, suggest_column_mapping
from src.custom_dataset_manager import CustomDatasetManager, load_kaggle_dataset

st.set_page_config(
    page_title="CausalBench - Interactive Causal Inference",
    page_icon="🔬",
    layout="wide"
)

# Initialize dataset manager
if 'dataset_manager' not in st.session_state:
    st.session_state.dataset_manager = CustomDatasetManager()

# Title and description
st.title("🔬 CausalBench: Interactive Causal Inference Demo")
st.markdown("""
This interactive dashboard demonstrates **causal inference** - discovering cause-and-effect relationships from data.

### What is Causal Inference?
Unlike correlation, causal inference answers: *"What happens if we intervene?"*
- **Example**: Does a training program actually improve job outcomes? Or is it just selection bias?
- **Application**: Policy evaluation, medical treatments, marketing campaigns

### How it Works:
1. **Select or upload a dataset** with treatment and outcome variables
2. **System validates** if data is suitable for causal analysis
3. **Choose estimators** (different statistical methods)
4. **View results**: Average Treatment Effect (ATE) - the causal impact
""")

st.divider()

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Dataset source selection
dataset_source = st.sidebar.radio(
    "Choose Data Source",
    ["Built-in Datasets", "Upload Custom Data", "Load from Kaggle"],
    help="Select whether to use benchmark datasets or your own data"
)

# Built-in datasets info
dataset_info = {
    "ihdp": {
        "name": "IHDP (Infant Health Development Program)",
        "description": "Semi-synthetic data studying effect of specialist home visits on children's cognitive test scores.",
        "treatment": "Specialist home visits",
        "outcome": "Cognitive test score",
        "sample_size": "~700 infants"
    },
    "twins": {
        "name": "Twins Mortality Study",
        "description": "Study of treatment effect on twin mortality rates.",
        "treatment": "Medical intervention",
        "outcome": "Mortality rate",
        "sample_size": "~11,000 twins"
    },
    "lalonde": {
        "name": "LaLonde Job Training",
        "description": "Famous economics study: Does job training increase earnings?",
        "treatment": "NSW job training program",
        "outcome": "1978 earnings (re78)",
        "sample_size": "~600 individuals"
    }
}

dataset_name = None
uploaded_data = None
column_mapping = None
dataset_id = None

# Handle dataset source
if dataset_source == "Built-in Datasets":
    # Get list of previously analyzed custom datasets
    custom_datasets = st.session_state.dataset_manager.list_datasets()
    custom_dataset_options = {d['name']: d['id'] for d in custom_datasets if d['is_valid']}

    all_options = ["ihdp", "twins", "lalonde"] + list(custom_dataset_options.keys())

    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        all_options,
        help="Choose a causal inference benchmark dataset or previously uploaded data"
    )

    # Check if custom dataset
    if dataset_name in custom_dataset_options:
        dataset_id = custom_dataset_options[dataset_name]
        uploaded_data = st.session_state.dataset_manager.get_dataset(dataset_id)
        metadata = st.session_state.dataset_manager.get_metadata(dataset_id)
        column_mapping = metadata["column_mapping"]
        st.sidebar.success(f"✅ Using cached dataset: {dataset_name}")

elif dataset_source == "Upload Custom Data":
    st.sidebar.subheader("📁 Upload Your Data")

    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your dataset for causal analysis"
    )

    if uploaded_file:
        try:
            # Load file
            if uploaded_file.name.endswith('.csv'):
                uploaded_data = pd.read_csv(uploaded_file)
            else:
                uploaded_data = pd.read_excel(uploaded_file)

            st.sidebar.success(f"✅ Loaded {len(uploaded_data)} rows, {len(uploaded_data.columns)} columns")

            # Validate dataset
            with st.spinner("🔍 Validating dataset for causal inference..."):
                validation_result = validate_for_causal_inference(uploaded_data)

            if not validation_result.is_valid:
                st.error("❌ Dataset Not Suitable for Causal Inference")
                st.markdown(f"**Reason:** {validation_result.reason}")

                if validation_result.issues:
                    st.markdown("**Issues Found:**")
                    for issue in validation_result.issues:
                        st.markdown(f"- {issue}")

                if validation_result.suggestions:
                    st.markdown("**Suggestions:**")
                    for key, suggestion in validation_result.suggestions.items():
                        st.markdown(f"- **{key.title()}**: {suggestion}")

                st.info("💡 **What makes a dataset suitable for causal inference?**\n"
                       "- A binary treatment variable (0/1, Yes/No)\n"
                       "- An outcome variable to measure (continuous or binary)\n"
                       "- Confounder variables (factors affecting both treatment and outcome)\n"
                       "- At least 100+ observations")

                st.stop()
            else:
                st.success(f"✅ Dataset Validated (Confidence: {validation_result.confidence:.0%})")

                # Show detected columns
                st.sidebar.subheader("🎯 Auto-Detected Columns")

                suggested_mapping = suggest_column_mapping(validation_result)

                # Treatment selection
                treatment_options = [c["name"] for c in validation_result.detected_columns["treatment_candidates"]]
                if treatment_options:
                    treatment_col = st.sidebar.selectbox(
                        "Treatment Variable",
                        treatment_options,
                        help="Binary variable indicating who received treatment"
                    )
                    # Show why it was detected
                    treat_info = next(c for c in validation_result.detected_columns["treatment_candidates"] if c["name"] == treatment_col)
                    st.sidebar.caption(f"ℹ️ {treat_info['reason']}")
                else:
                    st.error("No treatment variable detected")
                    st.stop()

                # Outcome selection
                outcome_options = [c["name"] for c in validation_result.detected_columns["outcome_candidates"]]
                if outcome_options:
                    outcome_col = st.sidebar.selectbox(
                        "Outcome Variable",
                        outcome_options,
                        help="Variable measuring the effect you want to estimate"
                    )
                    outcome_info = next(c for c in validation_result.detected_columns["outcome_candidates"] if c["name"] == outcome_col)
                    st.sidebar.caption(f"ℹ️ {outcome_info['reason']}")
                else:
                    st.error("No outcome variable detected")
                    st.stop()

                # Confounders selection
                confounder_options = validation_result.detected_columns["confounder_candidates"]
                # Remove treatment and outcome from confounders
                confounder_options = [c for c in confounder_options if c not in [treatment_col, outcome_col]]

                confounders = st.sidebar.multiselect(
                    "Confounder Variables",
                    confounder_options,
                    default=confounder_options[:5] if len(confounder_options) >= 5 else confounder_options,
                    help="Variables that affect both treatment and outcome"
                )

                column_mapping = {
                    "treatment": treatment_col,
                    "outcome": outcome_col,
                    "confounders": confounders
                }

                # Dataset name for caching
                dataset_name = st.sidebar.text_input(
                    "Name this dataset (for future use)",
                    value=uploaded_file.name.replace('.csv', '').replace('.xlsx', ''),
                    help="Give this dataset a memorable name"
                )

        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
            st.stop()

elif dataset_source == "Load from Kaggle":
    st.sidebar.subheader("📊 Load from Kaggle")

    kaggle_url = st.sidebar.text_input(
        "Kaggle Dataset URL or ID",
        placeholder="e.g., username/dataset-name or full URL",
        help="Enter Kaggle dataset URL or identifier (owner/dataset-name)"
    )

    if st.sidebar.button("📥 Download from Kaggle"):
        if not kaggle_url:
            st.sidebar.error("Please enter a Kaggle URL")
        else:
            with st.spinner("⬇️ Downloading from Kaggle..."):
                uploaded_data = load_kaggle_dataset(kaggle_url)

            if uploaded_data is None:
                st.error("❌ Failed to download Kaggle dataset. Check:\n"
                        "- Kaggle API credentials are configured (~/.kaggle/kaggle.json)\n"
                        "- Dataset URL is correct\n"
                        "- Dataset is public or you have access")
                st.info("💡 Setup Kaggle API:\n"
                       "1. Go to kaggle.com → Account → Create New API Token\n"
                       "2. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)\n"
                       "3. Install: pip install kaggle")
                st.stop()
            else:
                st.sidebar.success(f"✅ Downloaded {len(uploaded_data)} rows")

                # Same validation flow as upload
                with st.spinner("🔍 Validating dataset..."):
                    validation_result = validate_for_causal_inference(uploaded_data)

                if not validation_result.is_valid:
                    st.error("❌ Dataset Not Suitable for Causal Inference")
                    st.markdown(f"**Reason:** {validation_result.reason}")
                    if validation_result.issues:
                        for issue in validation_result.issues:
                            st.markdown(f"- {issue}")
                    st.stop()
                else:
                    st.success(f"✅ Validated (Confidence: {validation_result.confidence:.0%})")

                    # Column selection UI (same as upload)
                    suggested_mapping = suggest_column_mapping(validation_result)

                    treatment_options = [c["name"] for c in validation_result.detected_columns["treatment_candidates"]]
                    outcome_options = [c["name"] for c in validation_result.detected_columns["outcome_candidates"]]
                    confounder_options = validation_result.detected_columns["confounder_candidates"]

                    if not treatment_options or not outcome_options:
                        st.error("Missing required variables")
                        st.stop()

                    treatment_col = st.sidebar.selectbox("Treatment Variable", treatment_options)
                    outcome_col = st.sidebar.selectbox("Outcome Variable", outcome_options)

                    confounder_options = [c for c in confounder_options if c not in [treatment_col, outcome_col]]
                    confounders = st.sidebar.multiselect(
                        "Confounder Variables",
                        confounder_options,
                        default=confounder_options[:5] if len(confounder_options) >= 5 else confounder_options
                    )

                    column_mapping = {
                        "treatment": treatment_col,
                        "outcome": outcome_col,
                        "confounders": confounders
                    }

                    dataset_name = st.sidebar.text_input(
                        "Name this dataset",
                        value=kaggle_url.split('/')[-1] if '/' in kaggle_url else kaggle_url
                    )

# Estimator selection
estimators = st.sidebar.multiselect(
    "Select Estimators",
    ["ipw", "psm", "dr", "dml", "drlearner"],
    default=["ipw", "psm", "dml"],
    help="Choose causal inference methods to compare"
)

estimator_info = {
    "ipw": "Inverse Propensity Weighting - reweights samples by likelihood of treatment",
    "psm": "Propensity Score Matching - matches similar treated/control units",
    "dr": "Doubly Robust - combines outcome regression and propensity scores",
    "dml": "Double Machine Learning - uses ML for nuisance estimation",
    "drlearner": "Doubly Robust Learner - EconML's advanced DR method"
}

# Sample size (only for built-in datasets)
if dataset_source == "Built-in Datasets" and dataset_name in ["ihdp", "twins", "lalonde"]:
    sample_size = st.sidebar.slider(
        "Sample Size (for quick demo)",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Reduce dataset size for faster computation"
    )
else:
    sample_size = None

# Display dataset info
if dataset_name and dataset_source == "Built-in Datasets" and dataset_name in dataset_info:
    st.header(f"📊 Dataset: {dataset_info[dataset_name]['name']}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        **Description**: {dataset_info[dataset_name]['description']}

        **Treatment**: {dataset_info[dataset_name]['treatment']}

        **Outcome**: {dataset_info[dataset_name]['outcome']}

        **Sample Size**: {dataset_info[dataset_name]['sample_size']}
        """)

    with col2:
        st.markdown("""
        **Goal**: Estimate the Average Treatment Effect (ATE)

        **ATE** = Average outcome if everyone got treatment - Average outcome if nobody got treatment

        This tells us the *causal impact* of the treatment.
        """)

elif uploaded_data is not None:
    st.header(f"📊 Dataset: {dataset_name}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        **Source**: {dataset_source}

        **Rows**: {len(uploaded_data):,}

        **Columns**: {len(uploaded_data.columns)}

        **Treatment**: {column_mapping['treatment']}

        **Outcome**: {column_mapping['outcome']}

        **Confounders**: {len(column_mapping['confounders'])} variables
        """)

    with col2:
        st.markdown("""
        **Goal**: Estimate the Average Treatment Effect (ATE)

        **ATE** = Average outcome if everyone got treatment - Average outcome if nobody got treatment

        This tells us the *causal impact* of the treatment.
        """)

st.divider()

# Run analysis button
run_button_label = "🚀 Run Causal Analysis"

# Check if we have cached results
if dataset_id and st.session_state.dataset_manager.has_cached_results(dataset_id):
    if st.button("📊 Show Cached Results (Instant)", type="secondary"):
        cached = st.session_state.dataset_manager.get_cached_results(dataset_id)
        results_df = pd.DataFrame(cached["results"])

        st.success("✅ Showing cached analysis results")

        # Display results (reuse visualization code below)
        st.header("📈 Results: Average Treatment Effect (ATE)")

        if len(results_df) > 0:
            display_df = results_df.copy()
            if "ate" in display_df.columns:
                display_df["ATE Estimate"] = display_df["ate"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            if "runtime_seconds" in display_df.columns:
                display_df["Runtime (s)"] = display_df["runtime_seconds"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

            display_df["Description"] = display_df["estimator"].map(estimator_info)

            st.dataframe(
                display_df[["estimator", "ATE Estimate", "Runtime (s)", "Description"]],
                use_container_width=True,
                hide_index=True
            )

            # Visualization
            valid_results = results_df[results_df["ate"].notna()]

            if len(valid_results) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))

                bars = ax.bar(valid_results["estimator"], valid_results["ate"], alpha=0.7, color="steelblue")
                ax.set_xlabel("Estimator", fontsize=12)
                ax.set_ylabel("ATE Estimate", fontsize=12)
                ax.set_title("Average Treatment Effect by Estimator", fontsize=14)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10)

                plt.tight_layout()
                st.pyplot(fig)

                mean_ate = valid_results["ate"].mean()
                st.markdown(f"**Mean ATE**: {mean_ate:.4f}")

    st.markdown("---")
    run_button_label = "🔄 Re-run Analysis (takes 30-60s)"

if st.button(run_button_label, type="primary"):
    # Load data
    with st.spinner("Loading dataset..."):
        try:
            if uploaded_data is not None:
                data = uploaded_data.copy()
                config = {
                    "treatment": column_mapping["treatment"],
                    "outcome": column_mapping["outcome"],
                    "confounders": column_mapping["confounders"]
                }
            else:
                data = load_dataset(dataset_name, sample=sample_size)
                config = get_dataset_config(dataset_name)

            st.success(f"✅ Loaded {len(data)} samples")

            # Show data preview
            with st.expander("📋 View Data Sample (first 10 rows)"):
                st.dataframe(data.head(10))

                # Show treatment distribution
                if config["treatment"] in data.columns:
                    treatment_counts = data[config["treatment"]].value_counts()
                    st.write("**Treatment Distribution:**")
                    st.write(f"- Treated: {treatment_counts.get(1, 0)} ({treatment_counts.get(1, 0)/len(data)*100:.1f}%)")
                    st.write(f"- Control: {treatment_counts.get(0, 0)} ({treatment_counts.get(0, 0)/len(data)*100:.1f}%)")

        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")
            st.exception(e)
            st.stop()

    # Run pipeline
    with st.spinner("Running causal inference pipeline... This may take 30-60 seconds"):
        try:
            # Custom dataset handling
            if uploaded_data is not None:
                from src.dowhy_pipeline import run_dowhy_pipeline
                from src.econml_estimators import run_econml_pipeline

                treatment = config["treatment"]
                outcome = config["outcome"]
                confounders = config["confounders"]

                results_list = []

                # DoWhy estimators
                dowhy_methods = {
                    "ipw": "backdoor.propensity_score_weighting",
                    "psm": "backdoor.propensity_score_matching",
                    "dr": "backdoor.econometric.doubly_robust"
                }

                for est_name in estimators:
                    if est_name in dowhy_methods:
                        try:
                            result = run_dowhy_pipeline(
                                data,
                                treatment,
                                outcome,
                                confounders,
                                estimator_method=dowhy_methods[est_name],
                                output_dir=Path("results/streamlit_custom"),
                                random_state=RANDOM_SEED
                            )
                            results_list.append({
                                "estimator": est_name,
                                "ate": result["estimated_ate"],
                                "runtime_seconds": result.get("runtime_seconds", 0)
                            })
                        except Exception as e:
                            st.warning(f"⚠️ {est_name} failed: {str(e)}")

                # EconML estimators
                econml_ests = [e for e in estimators if e in ["dml", "drlearner"]]
                if econml_ests:
                    try:
                        econml_results = run_econml_pipeline(
                            data,
                            treatment,
                            outcome,
                            confounders,
                            estimators=econml_ests,
                            random_state=RANDOM_SEED
                        )
                        results_list.extend(econml_results.to_dict('records'))
                    except Exception as e:
                        st.warning(f"⚠️ EconML failed: {str(e)}")

                results_df = pd.DataFrame(results_list)

                # Save to cache
                if dataset_name and column_mapping:
                    if not dataset_id:
                        validation_result = validate_for_causal_inference(data)
                        dataset_id = st.session_state.dataset_manager.save_dataset(
                            data,
                            dataset_name,
                            f"Uploaded via {dataset_source}",
                            column_mapping,
                            validation_result
                        )

                    st.session_state.dataset_manager.save_analysis_results(
                        dataset_id,
                        results_df,
                        estimators,
                        column_mapping
                    )

            else:
                # Built-in dataset
                output_dir = Path("results/streamlit_demo")
                results_df = run_full_pipeline(
                    dataset_name,
                    estimators=estimators,
                    sample=sample_size,
                    output_dir=output_dir,
                    random_state=RANDOM_SEED
                )

            st.success("✅ Analysis complete!")

        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
            st.exception(e)
            st.stop()

    # Display results
    st.header("📈 Results: Average Treatment Effect (ATE)")

    if len(results_df) > 0:
        st.subheader("Estimator Comparison")

        display_df = results_df.copy()
        if "ate" in display_df.columns:
            display_df["ATE Estimate"] = display_df["ate"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        if "runtime_seconds" in display_df.columns:
            display_df["Runtime (s)"] = display_df["runtime_seconds"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        display_df["Description"] = display_df["estimator"].map(estimator_info)

        st.dataframe(
            display_df[["estimator", "ATE Estimate", "Runtime (s)", "Description"]],
            use_container_width=True,
            hide_index=True
        )

        # Visualization
        st.subheader("📊 ATE Comparison Chart")

        valid_results = results_df[results_df["ate"].notna()]

        if len(valid_results) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))

            bars = ax.bar(valid_results["estimator"], valid_results["ate"], alpha=0.7, color="steelblue")
            ax.set_xlabel("Estimator", fontsize=12)
            ax.set_ylabel("ATE Estimate", fontsize=12)
            ax.set_title("Average Treatment Effect by Estimator", fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)

            # Interpretation
            st.subheader("💡 Interpretation")
            mean_ate = valid_results["ate"].mean()
            std_ate = valid_results["ate"].std()

            st.markdown(f"""
            **Mean ATE across methods**: {mean_ate:.4f} (±{std_ate:.4f})

            **What does this mean?**
            - The treatment effect is approximately **{mean_ate:.4f}** units on the outcome variable
            - {'Positive' if mean_ate > 0 else 'Negative'} ATE suggests the treatment {'increases' if mean_ate > 0 else 'decreases'} the outcome
            - Different estimators show {'similar' if std_ate < abs(mean_ate) * 0.2 else 'varying'} results

            {'**For ' + dataset_info[dataset_name]['name'] + '**:' if dataset_name in dataset_info else '**For your dataset:**'}
            - Effect: {'Beneficial' if mean_ate > 0 else 'Harmful'} on average
            """)

            # Generate intelligent insights
            try:
                from src.intelligent_insights import generate_insights, format_insights_for_display

                # Prepare estimator results dict
                estimator_results = dict(zip(results_df["estimator"], results_df["ate"]))

                # Generate insights
                insights = generate_insights(
                    data=data,
                    treatment_col=config["treatment"],
                    outcome_col=config["outcome"],
                    confounders=config["confounders"],
                    ate_result=mean_ate,
                    estimator_results=estimator_results
                )

                # Display insights
                st.divider()
                formatted_insights = format_insights_for_display(insights)
                st.markdown(formatted_insights)

            except Exception as e:
                st.warning(f"Could not generate intelligent insights: {e}")

        else:
            st.warning("No valid results to display")
    else:
        st.warning("No results generated")

# Footer
st.divider()
st.header("📚 About the Methods")

with st.expander("🔍 What are these estimators?"):
    st.markdown("""
    ### Causal Inference Methods

    **1. IPW (Inverse Propensity Weighting)**
    - Reweights observations by the inverse probability of receiving treatment
    - Corrects for confounding by making treated/control groups comparable
    - Fast but sensitive to extreme propensity scores

    **2. PSM (Propensity Score Matching)**
    - Matches each treated unit with similar control units based on propensity score
    - Intuitive: compares "apples to apples"
    - May discard data if no good matches found

    **3. DR (Doubly Robust)**
    - Combines outcome regression and propensity scores
    - Robust: correct if either model is correct
    - More stable than IPW or matching alone

    **4. DML (Double Machine Learning)**
    - Uses machine learning for nuisance parameters (propensity, outcome)
    - Reduces bias from model misspecification
    - State-of-the-art method from economics/statistics

    **5. DRLearner (Doubly Robust Learner)**
    - EconML's implementation combining ML with doubly robust estimation
    - Flexible and powerful for heterogeneous effects
    - Can estimate individualized treatment effects
    """)

with st.expander("🎯 Use Cases"):
    st.markdown("""
    ### Real-World Applications

    **Healthcare**
    - Does a drug reduce mortality? (treatment = drug, outcome = survival)
    - Impact of surgery vs. medication

    **Economics/Policy**
    - Does job training increase earnings? (LaLonde dataset)
    - Effect of minimum wage on employment

    **Marketing**
    - Does an email campaign increase purchases?
    - Impact of discount offers on customer lifetime value

    **Education**
    - Does tutoring improve test scores?
    - Effect of class size on student outcomes

    **Technology**
    - Does a new feature increase user engagement?
    - Impact of recommendations on clicks/purchases
    """)

st.divider()
st.markdown("""
---
**CausalBench** | Built with DoWhy, EconML, Streamlit | [GitHub](#) | [Docs](README.md)
""")
