"""
EconML estimator wrappers for causal inference.
"""

import numpy as np
import pandas as pd
import time
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from econml.dml import LinearDML
    from econml.dr import DRLearner
    from econml.metalearners import TLearner, SLearner
    ECONML_AVAILABLE = True
except ImportError:
    logger.warning("EconML not available. Some estimators will be unavailable.")
    ECONML_AVAILABLE = False


def estimate_dml(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    random_state: int = 42
) -> Tuple[float, Dict[str, Any]]:
    """
    Estimate ATE using Doubly Robust Learning (DML).
    
    Args:
        X: Covariates (n_samples, n_features)
        T: Treatment (n_samples,)
        Y: Outcome (n_samples,)
        random_state: Random seed
    
    Returns:
        (ate_estimate, metadata_dict)
    """
    if not ECONML_AVAILABLE:
        raise ImportError("EconML not installed. Install with: pip install econml")
    
    start_time = time.time()
    
    # Fit DML
    dml_model = LinearDML(
        model_y=None,  # Will use default (linear model)
        model_t=None,
        random_state=random_state
    )
    
    dml_model.fit(Y, T, X=X)
    
    # Estimate ATE
    ate = dml_model.ate()
    
    runtime = time.time() - start_time
    
    metadata = {
        "method": "DML",
        "runtime_seconds": runtime,
        "ate_estimate": float(ate[0]) if isinstance(ate, np.ndarray) else float(ate)
    }
    
    return float(ate[0]) if isinstance(ate, np.ndarray) else float(ate), metadata


def estimate_drlearner(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    random_state: int = 42
) -> Tuple[float, Dict[str, Any]]:
    """
    Estimate ATE using Doubly Robust Learner.
    
    Args:
        X: Covariates (n_samples, n_features)
        T: Treatment (n_samples,)
        Y: Outcome (n_samples,)
        random_state: Random seed
    
    Returns:
        (ate_estimate, metadata_dict)
    """
    if not ECONML_AVAILABLE:
        raise ImportError("EconML not installed. Install with: pip install econml")
    
    start_time = time.time()
    
    # Fit DRLearner
    dr_model = DRLearner(
        model_propensity=None,
        model_regression=None,
        random_state=random_state
    )
    
    dr_model.fit(Y, T, X=X)
    
    # Estimate ATE
    ate = dr_model.ate()
    
    runtime = time.time() - start_time
    
    metadata = {
        "method": "DRLearner",
        "runtime_seconds": runtime,
        "ate_estimate": float(ate[0]) if isinstance(ate, np.ndarray) else float(ate)
    }
    
    return float(ate[0]) if isinstance(ate, np.ndarray) else float(ate), metadata


def estimate_tmle(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    random_state: int = 42
) -> Tuple[float, Dict[str, Any]]:
    """
    Estimate ATE using Targeted Maximum Likelihood Estimation (TMLE).
    
    Note: This is a simplified wrapper. Full TMLE may require additional setup.
    
    Args:
        X: Covariates (n_samples, n_features)
        T: Treatment (n_samples,)
        Y: Outcome (n_samples,)
        random_state: Random seed
    
    Returns:
        (ate_estimate, metadata_dict)
    """
    if not ECONML_AVAILABLE:
        raise ImportError("EconML not installed. Install with: pip install econml")
    
    # For now, use DRLearner as TMLE alternative
    # Full TMLE implementation would require additional dependencies
    logger.warning("Using DRLearner as TMLE proxy. Full TMLE requires additional setup.")
    return estimate_drlearner(X, T, Y, random_state)


def run_econml_pipeline(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounders: list,
    estimators: list = ["dml", "drlearner"],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run multiple EconML estimators and return comparison DataFrame.
    
    Args:
        data: DataFrame with treatment, outcome, and confounders
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounders: List of confounder column names
        estimators: List of estimator names to run
        random_state: Random seed
    
    Returns:
        DataFrame with estimator results
    """
    X = data[confounders].values
    T = data[treatment_col].values
    Y = data[outcome_col].values
    
    # Ensure 1D arrays for sklearn compatibility (avoid DataConversionWarning)
    if T.ndim > 1:
        T = T.ravel()
    if Y.ndim > 1:
        Y = Y.ravel()
    
    results = []
    
    for est_name in estimators:
        try:
            if est_name.lower() == "dml":
                ate, metadata = estimate_dml(X, T, Y, random_state)
            elif est_name.lower() == "drlearner":
                ate, metadata = estimate_drlearner(X, T, Y, random_state)
            elif est_name.lower() == "tmle":
                ate, metadata = estimate_tmle(X, T, Y, random_state)
            else:
                logger.warning(f"Unknown estimator: {est_name}")
                continue
            
            results.append({
                "estimator": est_name,
                "ate": ate,
                "runtime_seconds": metadata.get("runtime_seconds", 0)
            })
            
        except Exception as e:
            logger.error(f"Failed to run {est_name}: {e}")
            results.append({
                "estimator": est_name,
                "ate": np.nan,
                "runtime_seconds": 0,
                "error": str(e)
            })
    
    return pd.DataFrame(results)
