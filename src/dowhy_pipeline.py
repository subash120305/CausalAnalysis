"""
DoWhy pipeline for causal inference: identification, estimation, refutation.
"""

import argparse
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys

from .data_loader import (
    download_ihdp, download_twins, download_sachs, download_acic, download_lalonde,
    RANDOM_SEED
)
from .metrics import evaluate_estimator
from .viz import plot_ate_comparison, plot_ite_scatter
from .econml_estimators import run_econml_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix NetworkX compatibility BEFORE importing DoWhy
# DoWhy 0.8 uses d_separated which is deprecated in NetworkX 3.4+
# We wrap it to suppress deprecation warnings
try:
    import networkx as nx
    import warnings
    
    # NetworkX 3.4 has d_separated but shows deprecation warnings
    # Wrap it to suppress the warning
    if hasattr(nx.algorithms, 'd_separated'):
        _original_d_separated = nx.algorithms.d_separated
        
        def d_separated_wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return _original_d_separated(*args, **kwargs)
        
        nx.algorithms.d_separated = d_separated_wrapper
    elif hasattr(nx.algorithms, 'd_separation'):
        # NetworkX 3.5+: d_separated was removed, use is_d_separator
        from networkx.algorithms.d_separation import is_d_separator
        
        def d_separated_wrapper(G, x, y, z):
            return is_d_separator(G, x, y, z)
        
        nx.algorithms.d_separated = d_separated_wrapper
except Exception:
    pass  # NetworkX not available, will fail at DoWhy import

# Fix scipy deprecation warnings from sklearn
# sklearn's LogisticRegression uses scipy.optimize with deprecated options
try:
    import warnings
    import scipy.optimize
    
    _original_minimize = scipy.optimize.minimize
    
    def minimize_wrapper(*args, **kwargs):
        # Remove deprecated options if present
        kwargs.pop('disp', None)
        kwargs.pop('iprint', None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return _original_minimize(*args, **kwargs)
    
    scipy.optimize.minimize = minimize_wrapper
except Exception:
    pass

# Fix sklearn DataConversionWarning
# DoWhy internally passes column vectors to sklearn, which expects 1D arrays
try:
    from sklearn.utils.validation import column_or_1d
    import warnings
    
    _original_column_or_1d = column_or_1d
    
    def column_or_1d_wrapper(y, *, dtype=None, warn=False, device=None):
        # Convert column vectors to 1D arrays before sklearn sees them
        y_arr = np.asarray(y)
        if y_arr.ndim > 1 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
            warn = False  # No need to warn, we fixed it
        # Call original with fixed array
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Suppress DataConversionWarning
            return _original_column_or_1d(y_arr, dtype=dtype, warn=warn, device=device)
    
    # Patch it before DoWhy imports sklearn
    import sklearn.utils.validation
    sklearn.utils.validation.column_or_1d = column_or_1d_wrapper
except Exception:
    pass

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
        
except ImportError:
    logger.error("DoWhy not available. Install with: pip install dowhy")
    DOWHY_AVAILABLE = False


def load_dataset(dataset_name: str, sample: Optional[int] = None) -> pd.DataFrame:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of dataset (ihdp, twins, sachs, acic, lalonde)
        sample: Optional number of rows to sample (for quick runs)
    
    Returns:
        DataFrame with loaded data
    """
    download_funcs = {
        "ihdp": download_ihdp,
        "twins": download_twins,
        "sachs": download_sachs,
        "acic": download_acic,
        "lalonde": download_lalonde
    }
    
    if dataset_name not in download_funcs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(download_funcs.keys())}")
    
    file_path = download_funcs[dataset_name]()
    
    if file_path is None:
        raise FileNotFoundError(f"Failed to download {dataset_name} dataset")
    
    df = pd.read_csv(file_path)
    
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=RANDOM_SEED).reset_index(drop=True)
        logger.info(f"Sampled {sample} rows from {dataset_name}")
    
    return df


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get dataset-specific configuration (treatment, outcome, confounders)."""
    
    configs = {
        "ihdp": {
            "treatment": "treatment",
            "outcome": "y_factual",
            "confounders": [f"x{i}" for i in range(1, 26)]  # x1 to x25
        },
        "twins": {
            "treatment": "T",
            "outcome": "y",
            "confounders": ["x1", "x2", "x3", "x4", "x5"]
        },
        "sachs": {
            "treatment": None,  # Discovery dataset, not RCT
            "outcome": None,
            "confounders": None
        },
        "acic": {
            "treatment": "z",
            "outcome": "y",
            "confounders": ["x1", "x2", "x3", "x4"]
        },
        "lalonde": {
            "treatment": "treat",
            "outcome": "re78",
            "confounders": ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
        }
    }
    
    return configs.get(dataset_name, {
        "treatment": "treatment",
        "outcome": "outcome",
        "confounders": []
    })


def run_dowhy_pipeline(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    estimator_method: str = "backdoor.propensity_score_weighting",
    output_dir: Path = Path("results"),
    true_ate: Optional[float] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run full DoWhy pipeline: model, identify, estimate, refute.
    
    Args:
        data: DataFrame with treatment, outcome, confounders
        treatment: Treatment column name
        outcome: Outcome column name
        confounders: List of confounder column names
        estimator_method: DoWhy estimator method string
        output_dir: Directory to save results
        true_ate: Optional true ATE for evaluation
        random_state: Random seed
    
    Returns:
        Dictionary with results
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy not installed")
    
    np.random.seed(random_state)
    results = {}
    
    # Use common_causes parameter (more compatible across DoWhy versions)
    # This avoids graph parsing issues and NetworkX compatibility problems
    graph_str = f"common_causes: {confounders}"
    
    # Ensure data columns are properly shaped (1D arrays) to avoid sklearn warnings
    # DoWhy uses sklearn internally for propensity score estimation
    # Pandas Series can sometimes be treated as column vectors by sklearn
    data_clean = data.copy()
    # Convert to numpy arrays and ensure 1D shape
    for col in [treatment, outcome] + (confounders if confounders else []):
        if col in data_clean.columns:
            values = data_clean[col].values
            if values.ndim > 1 and values.shape[1] == 1:
                data_clean[col] = values.ravel()
            elif values.ndim == 0:
                data_clean[col] = values.flatten()
    
    # Create CausalModel using common_causes (DoWhy will build graph internally)
    model = CausalModel(
        data=data_clean,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders if confounders else None
    )
    
    results["graph"] = graph_str
    
    # Identify estimand
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    results["identified_estimand"] = str(identified_estimand)
    
    # Estimate effect
    start_time = time.time()
    # DoWhy 0.8 doesn't support random_state in estimate_effect, using numpy seed instead
    np.random.seed(random_state)
    causal_estimate = model.estimate_effect(
        identified_estimand,
        method_name=estimator_method
    )
    runtime = time.time() - start_time
    
    estimated_ate = causal_estimate.value
    results["estimated_ate"] = float(estimated_ate)
    results["runtime_seconds"] = runtime
    results["estimator_method"] = estimator_method
    
    logger.info(f"Estimated ATE: {estimated_ate:.4f}")
    
    # Refutation (optional, can be slow)
    try:
        refute_result = model.refute_estimate(
            identified_estimand,
            causal_estimate,
            method_name="random_common_cause",
            random_state=random_state
        )
        results["refute_p_value"] = refute_result.new_effect
        logger.info(f"Refutation result: {refute_result}")
    except Exception as e:
        logger.warning(f"Refutation failed: {e}")
        results["refute_p_value"] = None
    
    # Evaluate if true ATE available
    if true_ate is not None:
        metrics = evaluate_estimator(
            estimated_ate,
            true_ate=true_ate,
            runtime_seconds=runtime
        )
        results.update(metrics)
    
    return results


def run_full_pipeline(
    dataset_name: str,
    estimators: List[str] = ["ipw", "psm", "dr", "dml"],
    sample: Optional[int] = None,
    output_dir: Path = Path("results"),
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run full causal inference pipeline for a dataset.
    
    Args:
        dataset_name: Name of dataset
        estimators: List of estimator names
        sample: Optional sample size
        output_dir: Output directory
        random_state: Random seed
    
    Returns:
        DataFrame with estimator comparison
    """
    logger.info(f"Running pipeline for {dataset_name}")
    
    # Load dataset
    data = load_dataset(dataset_name, sample=sample)
    config = get_dataset_config(dataset_name)
    
    if config["treatment"] is None:
        logger.warning(f"{dataset_name} is a discovery dataset, skipping ATE estimation")
        return pd.DataFrame()
    
    treatment = config["treatment"]
    outcome = config["outcome"]
    confounders = [c for c in config["confounders"] if c in data.columns]
    
    # Prepare output directory
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    results_list = []
    
    # Run DoWhy estimators
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
                    output_dir=dataset_output_dir,
                    random_state=random_state
                )
                results_list.append({
                    "estimator": est_name,
                    "ate": result["estimated_ate"],
                    "runtime_seconds": result.get("runtime_seconds", 0)
                })
            except Exception as e:
                logger.error(f"Failed to run {est_name}: {e}")
                results_list.append({
                    "estimator": est_name,
                    "ate": np.nan,
                    "runtime_seconds": 0,
                    "error": str(e)
                })
    
    # Run EconML estimators
    econml_ests = [e for e in estimators if e in ["dml", "drlearner", "tmle"]]
    if econml_ests:
        try:
            econml_results = run_econml_pipeline(
                data,
                treatment,
                outcome,
                confounders,
                estimators=econml_ests,
                random_state=random_state
            )
            results_list.extend(econml_results.to_dict('records'))
        except Exception as e:
            logger.error(f"EconML pipeline failed: {e}")
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_path = dataset_output_dir / "estimators_summary.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Create visualizations
    if len(results_df) > 0 and not results_df["ate"].isna().all():
        plot_ate_comparison(
            dict(zip(results_df["estimator"], results_df["ate"])),
            output_path=dataset_output_dir / "sample_plots" / "ate_comparison.png"
        )
    
    return results_df


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run DoWhy causal inference pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ihdp", "twins", "sachs", "acic", "lalonde"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--estimator",
        type=str,
        nargs="+",
        default=["ipw", "psm", "dr", "dml"],
        help="Estimator methods to run"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for quick runs (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    try:
        results_df = run_full_pipeline(
            args.dataset,
            estimators=args.estimator,
            sample=args.sample,
            output_dir=output_dir,
            random_state=args.random_seed
        )
        
        print("\n" + "="*50)
        print("Pipeline completed successfully")
        print("="*50)
        print(results_df.to_string())
        print(f"\nResults saved to {output_dir / args.dataset}/")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
