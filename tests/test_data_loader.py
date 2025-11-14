"""
Tests for data_loader module.
"""

import pytest
import json
from pathlib import Path
from src.data_loader import (
    download_ihdp, download_twins, download_sachs, download_acic, download_lalonde,
    check_links, LINK_CHECKS_FILE, DATA_DIR
)


def test_link_checks_exist():
    """Test that link_checks.json is created."""
    check_links()
    assert LINK_CHECKS_FILE.exists(), "link_checks.json should exist after check_links()"
    
    with open(LINK_CHECKS_FILE, 'r') as f:
        checks = json.load(f)
    
    assert isinstance(checks, dict), "link_checks.json should contain a dictionary"
    assert len(checks) > 0, "link_checks should contain at least one URL"


def test_data_dir_structure():
    """Test that data directory structure is created."""
    assert DATA_DIR.exists(), "data directory should exist"


def test_ihdp_download():
    """Test IHDP download (may fail due to network, but should handle gracefully)."""
    result = download_ihdp()
    # If download succeeds, file should exist
    # If it fails, result is None but no exception
    if result:
        assert Path(result).exists(), f"IHDP file should exist at {result}"


def test_twins_download():
    """Test Twins download."""
    result = download_twins()
    if result:
        assert Path(result).exists(), f"Twins file should exist at {result}"


def test_sachs_download():
    """Test Sachs download."""
    result = download_sachs()
    if result:
        assert Path(result).exists(), f"Sachs file should exist at {result}"


def test_lalonde_download():
    """Test Lalonde download."""
    result = download_lalonde()
    if result:
        assert Path(result).exists(), f"Lalonde file should exist at {result}"


def test_dataset_directories():
    """Test that dataset directories are created."""
    datasets = ["ihdp", "twins", "sachs", "acic2019", "lalonde"]
    
    # Try to create directories (via download attempts)
    download_ihdp()
    download_twins()
    download_sachs()
    download_acic(2019)
    download_lalonde()
    
    # Check that at least some directories exist
    existing_dirs = [d for d in datasets if (DATA_DIR / d.replace("2019", "")).exists()]
    assert len(existing_dirs) > 0, "At least one dataset directory should exist"
