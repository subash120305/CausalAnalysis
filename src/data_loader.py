"""
Data loader module for downloading and validating causal datasets.

Handles IHDP, Twins, Sachs, ACIC, and Lalonde datasets with fallback mirrors
and comprehensive error handling.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from urllib.parse import urlparse
import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
DATA_DIR = Path("./data")
LINK_CHECKS_FILE = DATA_DIR / "link_checks.json"


def _ensure_data_dir(dataset_name: str) -> Path:
    """Ensure dataset directory exists."""
    dataset_dir = DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _download_file(url: str, dest_path: Path, timeout: int = 10) -> Tuple[bool, int, str]:
    """
    Download a file from URL with HEAD check first.
    
    Returns:
        (success, status_code, error_message)
    """
    try:
        # First try HEAD request
        head_response = requests.head(url, timeout=timeout, allow_redirects=True)
        status_code = head_response.status_code
        
        if status_code in [200, 301, 302, 303, 307, 308]:
            # HEAD succeeded, now download
            response = requests.get(url, timeout=timeout * 3, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {url} to {dest_path}")
            return True, response.status_code, ""
        else:
            # HEAD failed, try GET directly
            logger.warning(f"HEAD failed with {status_code}, trying GET for {url}")
            response = requests.get(url, timeout=timeout * 3, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {url} to {dest_path} via GET")
            return True, response.status_code, ""
            
    except requests.exceptions.RequestException as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, 0, error_msg


def _check_kaggle_api(dataset_path: str, dest_path: Path) -> Tuple[bool, str]:
    """Attempt to download from Kaggle API if credentials are available."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Extract dataset identifier from path
        # Format: kaggle/datasets/{username}/{dataset}
        parts = dataset_path.split('/')
        if len(parts) >= 4 and parts[-2] == 'datasets':
            username = parts[-1].split('-')[0] if '-' in parts[-1] else parts[-1]
            dataset = parts[-1]
            api.dataset_download_files(dataset, path=str(dest_path.parent), unzip=True)
            
            # Find the downloaded file
            for file in dest_path.parent.glob("*"):
                if file.is_file() and file.suffix in ['.csv', '.txt', '.tsv']:
                    if file != dest_path:
                        file.rename(dest_path)
                    logger.info(f"Downloaded from Kaggle to {dest_path}")
                    return True, ""
        
        return False, "Kaggle dataset path format not recognized"
        
    except ImportError:
        return False, "kaggle package not installed"
    except Exception as e:
        error_msg = f"Kaggle API error: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def _save_provenance(dataset_dir: Path, url: str, timestamp: str, status_code: int):
    """Save provenance information to README.txt."""
    readme_path = dataset_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"Dataset: {dataset_dir.name}\n")
        f.write(f"Source URL: {url}\n")
        f.write(f"Download timestamp: {timestamp}\n")
        f.write(f"HTTP Status Code: {status_code}\n")
        f.write(f"Provenance: Downloaded via CausalBench data_loader.py\n")


def _update_link_checks(url: str, status_code: int, success: bool):
    """Update link_checks.json with URL status."""
    LINK_CHECKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if LINK_CHECKS_FILE.exists():
        with open(LINK_CHECKS_FILE, 'r') as f:
            checks = json.load(f)
    else:
        checks = {}
    
    checks[url] = {
        "status_code": status_code if success else 0,
        "success": success,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(LINK_CHECKS_FILE, 'w') as f:
        json.dump(checks, f, indent=2)


def download_ihdp() -> Optional[str]:
    """
    Download IHDP dataset.
    
    Returns:
        Path to downloaded file or None if failed
    """
    dataset_dir = _ensure_data_dir("ihdp")
    dest_path = dataset_dir / "ihdp_npci_1.csv"
    
    if dest_path.exists():
        logger.info(f"IHDP dataset already exists at {dest_path}")
        return str(dest_path)
    
    urls = [
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
        "https://www.kaggle.com/datasets/konradb/ihdp-data",
        "https://www.pywhy.org/dowhy/v0.11/example_notebooks/dowhy_ihdp_data_example.html"
    ]
    
    for url in urls:
        if "kaggle.com" in url:
            # Try Kaggle API
            success, error = _check_kaggle_api(url, dest_path)
            if success:
                _update_link_checks(url, 200, True)
                _save_provenance(dataset_dir, url, time.strftime("%Y-%m-%d %H:%M:%S"), 200)
                return str(dest_path)
            else:
                logger.warning(f"Kaggle download failed: {error}")
                _update_link_checks(url, 0, False)
                print(f"\nKaggle download failed. To download manually, run:\n"
                      f"  curl -O {url}\n"
                      f"Or authenticate Kaggle API with credentials from ~/.kaggle/kaggle.json")
                continue
        
        if url.endswith('.html'):
            # Skip HTML pages, these are documentation
            continue
        
        success, status_code, error = _download_file(url, dest_path)
        _update_link_checks(url, status_code, success)
        
        if success:
            # Validate schema
            try:
                df = pd.read_csv(dest_path)
                # IHDP should have specific columns (adjust based on actual schema)
                if len(df.columns) >= 5:  # Basic validation
                    _save_provenance(dataset_dir, url, time.strftime("%Y-%m-%d %H:%M:%S"), status_code)
                    logger.info(f"IHDP downloaded and validated: {len(df)} rows, {len(df.columns)} cols")
                    return str(dest_path)
            except Exception as e:
                logger.error(f"Schema validation failed: {e}")
    
    logger.error("Failed to download IHDP from all sources")
    return None


def download_twins() -> Optional[str]:
    """Download Twins dataset."""
    dataset_dir = _ensure_data_dir("twins")
    dest_path = dataset_dir / "twins_data.csv"
    
    if dest_path.exists():
        logger.info(f"Twins dataset already exists at {dest_path}")
        return str(dest_path)
    
    # Try to use DoWhy's built-in loader first
    try:
        from dowhy.datasets import twins_dataset
        df = twins_dataset()
        df.to_csv(dest_path, index=False)
        logger.info(f"Twins dataset downloaded via DoWhy to {dest_path}")
        _save_provenance(dataset_dir, "DoWhy built-in loader", time.strftime("%Y-%m-%d %H:%M:%S"), 200)
        return str(dest_path)
    except Exception as e:
        logger.warning(f"DoWhy loader failed: {e}, trying URLs...")
    
    urls = [
        "https://www.pywhy.org/dowhy/v0.12/example_notebooks/dowhy_twins_example.html",
        "https://www.kaggle.com/datasets",
    ]
    
    for url in urls:
        if "kaggle.com" in url and not url.endswith('.csv'):
            # Kaggle search - provide instructions
            logger.warning("Kaggle search required for twins dataset")
            print(f"\nFor Twins dataset, search Kaggle or use DoWhy's built-in loader.")
            _update_link_checks(url, 0, False)
            continue
        
        if url.endswith('.html'):
            continue
        
        success, status_code, error = _download_file(url, dest_path)
        _update_link_checks(url, status_code, success)
        
        if success:
            try:
                df = pd.read_csv(dest_path)
                _save_provenance(dataset_dir, url, time.strftime("%Y-%m-%d %H:%M:%S"), status_code)
                return str(dest_path)
            except Exception as e:
                logger.error(f"Schema validation failed: {e}")
    
    logger.error("Failed to download Twins from all sources")
    return None


def download_sachs() -> Optional[str]:
    """Download Sachs protein signaling dataset."""
    dataset_dir = _ensure_data_dir("sachs")
    dest_path = dataset_dir / "sachs_data.csv"
    
    if dest_path.exists():
        logger.info(f"Sachs dataset already exists at {dest_path}")
        return str(dest_path)
    
    urls = [
        "https://zenodo.org/records/7681811",  # Zenodo record (may need to find actual file URL)
        "https://www.bnlearn.com/research/sachs05/",
        "https://www.science.org/doi/10.1126/science.1105809"
    ]
    
    # Try causal-learn's Sachs dataset loader
    try:
        from causallearn.dataset import load_sachs
        data = load_sachs()
        df = pd.DataFrame(data)
        df.to_csv(dest_path, index=False)
        logger.info(f"Sachs dataset loaded via causal-learn to {dest_path}")
        _save_provenance(dataset_dir, "causal-learn built-in loader", time.strftime("%Y-%m-%d %H:%M:%S"), 200)
        return str(dest_path)
    except Exception as e:
        logger.warning(f"causal-learn loader failed: {e}, trying URLs...")
    
    # For Zenodo, try to construct actual file URL
    zenodo_base = "https://zenodo.org/api/records/7681811"
    try:
        response = requests.get(zenodo_base, timeout=10)
        if response.status_code == 200:
            record = response.json()
            files = record.get('files', [])
            for file_info in files:
                file_url = file_info.get('links', {}).get('self', '')
                if file_url.endswith('.csv') or file_url.endswith('.txt'):
                    success, status_code, error = _download_file(file_url, dest_path)
                    _update_link_checks(file_url, status_code, success)
                    if success:
                        _save_provenance(dataset_dir, file_url, time.strftime("%Y-%m-%d %H:%M:%S"), status_code)
                        return str(dest_path)
    except Exception as e:
        logger.warning(f"Zenodo API failed: {e}")
    
    logger.error("Failed to download Sachs from all sources")
    return None


def download_acic(year: int = 2019) -> Optional[str]:
    """Download ACIC dataset for specified year."""
    dataset_dir = _ensure_data_dir(f"acic{year}")
    dest_path = dataset_dir / f"acic_{year}_data.csv"
    
    if dest_path.exists():
        logger.info(f"ACIC {year} dataset already exists at {dest_path}")
        return str(dest_path)
    
    urls = [
        f"https://sites.google.com/view/acic{year}datachallenge/home",
        f"https://github.com/zalandoresearch/ACIC23-competition",
        f"https://sites.google.com/view/acic{year}datachallenge/data-challenge"
    ]
    
    # ACIC datasets are complex - provide instructions for manual download
    logger.warning(f"ACIC {year} requires manual setup. Please download from official challenge page.")
    print(f"\nACIC {year} dataset requires manual download.")
    print(f"Visit: https://sites.google.com/view/acic{year}datachallenge/data-challenge")
    print(f"Download and place files in {dataset_dir}/")
    
    for url in urls:
        _update_link_checks(url, 0, False)
    
    return None


def download_lalonde() -> Optional[str]:
    """Download Lalonde dataset."""
    dataset_dir = _ensure_data_dir("lalonde")
    dest_path = dataset_dir / "lalonde_data.csv"
    
    if dest_path.exists():
        logger.info(f"Lalonde dataset already exists at {dest_path}")
        return str(dest_path)
    
    urls = [
        "https://www.kaggle.com/datasets/samuelzakouri/lalonde",
        "https://raw.githubusercontent.com/Maluuba/gflownet/main/data/lalonde.csv",  # Example mirror
    ]
    
    for url in urls:
        if "kaggle.com" in url:
            success, error = _check_kaggle_api(url, dest_path)
            if success:
                _update_link_checks(url, 200, True)
                _save_provenance(dataset_dir, url, time.strftime("%Y-%m-%d %H:%M:%S"), 200)
                return str(dest_path)
            else:
                _update_link_checks(url, 0, False)
                continue
        
        success, status_code, error = _download_file(url, dest_path)
        _update_link_checks(url, status_code, success)
        
        if success:
            try:
                df = pd.read_csv(dest_path)
                _save_provenance(dataset_dir, url, time.strftime("%Y-%m-%d %H:%M:%S"), status_code)
                return str(dest_path)
            except Exception as e:
                logger.error(f"Schema validation failed: {e}")
    
    logger.error("Failed to download Lalonde from all sources")
    return None


def check_links() -> Dict[str, Dict]:
    """
    Check all dataset links and return status codes.
    
    Returns:
        Dictionary mapping URLs to status information
    """
    logger.info("Checking all dataset links...")
    
    all_urls = [
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
        "https://www.kaggle.com/datasets/konradb/ihdp-data",
        "https://www.pywhy.org/dowhy/v0.11/example_notebooks/dowhy_ihdp_data_example.html",
        "https://www.pywhy.org/dowhy/v0.12/example_notebooks/dowhy_twins_example.html",
        "https://zenodo.org/records/7681811",
        "https://www.bnlearn.com/research/sachs05/",
        "https://sites.google.com/view/acic2019datachallenge/home",
        "https://www.kaggle.com/datasets/samuelzakouri/lalonde",
    ]
    
    results = {}
    for url in all_urls:
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            status_code = response.status_code
            results[url] = {
                "status_code": status_code,
                "success": status_code in [200, 301, 302, 303, 307, 308],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            _update_link_checks(url, status_code, status_code in [200, 301, 302, 303, 307, 308])
        except Exception as e:
            results[url] = {
                "status_code": 0,
                "success": False,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            _update_link_checks(url, 0, False)
    
    return results
