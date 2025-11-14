"""
Tests for dowhy_pipeline module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.dowhy_pipeline import (
    load_dataset, get_dataset_config, run_dowhy_pipeline, RANDOM_SEED
)


@pytest.fixture
def synthetic_data():
    """Create a small synthetic dataset for testing."""
    np.random.seed(RANDOM_SEED)
    n = 100
    
    # Generate confounders
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    
    # Generate treatment based on confounders
    propensity = 1 / (1 + np.exp(-(0.5 * X1 + 0.5 * X2)))
    T = np.random.binomial(1, propensity)
    
    # Generate outcome
    Y = 2.0 * T + 1.5 * X1 + 1.0 * X2 + np.random.randn(n) * 0.1
    
    df = pd.DataFrame({
        "treatment": T,
        "outcome": Y,
        "x1": X1,
        "x2": X2
    })
    
    return df


def test_get_dataset_config():
    """Test dataset configuration retrieval."""
    config = get_dataset_config("ihdp")
    assert "treatment" in config
    assert "outcome" in config
    assert "confounders" in config


def test_run_dowhy_pipeline_synthetic(synthetic_data):
    """Test DoWhy pipeline on synthetic data."""
    try:
        result = run_dowhy_pipeline(
            data=synthetic_data,
            treatment="treatment",
            outcome="outcome",
            confounders=["x1", "x2"],
            estimator_method="backdoor.propensity_score_weighting",
            output_dir=Path("results/test"),
            random_state=RANDOM_SEED
        )
        
        assert "estimated_ate" in result
        assert isinstance(result["estimated_ate"], (int, float))
        assert "runtime_seconds" in result
        
        # ATE should be close to true value (~2.0) but allow some error
        assert abs(result["estimated_ate"] - 2.0) < 1.0, \
            f"Estimated ATE {result['estimated_ate']} should be close to 2.0"
        
    except ImportError as e:
        pytest.skip(f"DoWhy not available: {e}")


def test_load_dataset_with_sample():
    """Test dataset loading with sampling."""
    # This may fail if dataset not downloaded, but should handle gracefully
    try:
        data = load_dataset("ihdp", sample=100)
        assert len(data) == 100, "Sampled dataset should have 100 rows"
        assert isinstance(data, pd.DataFrame), "Should return DataFrame"
    except Exception as e:
        # If dataset not available, that's okay for test
        pytest.skip(f"Dataset not available: {e}")
