"""
New simplified data loader for the 12 built-in datasets.
All datasets are generated locally and stored in data/ directory.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
METADATA_FILE = DATA_DIR / "dataset_metadata.json"


# Dataset registry - maps short name to filename
DATASET_REGISTRY = {
    # Healthcare
    'healthcare_hypertension': {
        'file': 'healthcare_hypertension.csv',
        'display_name': 'Healthcare - Hypertension Drug Trial',
        'description': 'Clinical trial of hypertension medication',
        'domain': 'Healthcare',
        'size': '800 patients'
    },

    # E-Commerce
    'ecommerce_recommendations': {
        'file': 'ecommerce_recommendations.csv',
        'display_name': 'E-Commerce - AI Recommendations',
        'description': 'AI-powered product recommendation A/B test',
        'domain': 'E-Commerce',
        'size': '1,200 users'
    },

    # Education
    'education_online_learning': {
        'file': 'education_online_learning.csv',
        'display_name': 'Education - Online Learning Platform',
        'description': 'Interactive vs traditional video lessons',
        'domain': 'Education',
        'size': '950 students'
    },

    # Finance
    'finance_creditcard': {
        'file': 'finance_creditcard.csv',
        'display_name': 'Finance - Credit Card Marketing',
        'description': 'Premium vs standard card offer campaign',
        'domain': 'Finance',
        'size': '1,100 customers'
    },

    # HR
    'hr_remote_work': {
        'file': 'hr_remote_work.csv',
        'display_name': 'HR - Remote Work Policy',
        'description': 'Impact of remote work on job satisfaction',
        'domain': 'Human Resources',
        'size': '650 employees'
    },

    # Agriculture
    'agriculture_fertilizer': {
        'file': 'agriculture_fertilizer.csv',
        'display_name': 'Agriculture - Fertilizer Treatment',
        'description': 'Organic vs synthetic fertilizer effects',
        'domain': 'Agriculture',
        'size': '500 farms'
    },

    # Transportation
    'transportation_rideshare': {
        'file': 'transportation_rideshare.csv',
        'display_name': 'Transportation - Surge Pricing',
        'description': 'Dynamic pricing impact on ride acceptance',
        'domain': 'Transportation',
        'size': '2,000 rides'
    },

    # Social Media
    'socialmedia_moderation': {
        'file': 'socialmedia_moderation.csv',
        'display_name': 'Social Media - Content Moderation',
        'description': 'AI vs manual content moderation',
        'domain': 'Social Media',
        'size': '1,500 users'
    },

    # Energy
    'energy_smart_thermostat': {
        'file': 'energy_smart_thermostat.csv',
        'display_name': 'Energy - Smart Thermostat',
        'description': 'Energy savings from smart thermostats',
        'domain': 'Energy & Utilities',
        'size': '850 households'
    },

    # Retail
    'retail_store_layout': {
        'file': 'retail_store_layout.csv',
        'display_name': 'Retail - Store Layout Optimization',
        'description': 'Data-driven vs traditional store layouts',
        'domain': 'Retail',
        'size': '380 stores'
    },

    # Telecom
    'telecom_5g_upgrade': {
        'file': 'telecom_5g_upgrade.csv',
        'display_name': 'Telecom - 5G Network Upgrade',
        'description': '5G impact on customer satisfaction',
        'domain': 'Telecommunications',
        'size': '1,000 customers'
    },

    # Public Health
    'publichealth_vaccination': {
        'file': 'publichealth_vaccination.csv',
        'display_name': 'Public Health - Vaccination Campaign',
        'description': 'SMS reminders impact on vaccination rates',
        'domain': 'Public Health',
        'size': '1,200 participants'
    }
}


def get_available_datasets() -> Dict[str, Dict[str, str]]:
    """
    Get list of all available datasets with metadata.
    Scans data/ directory for all dataset folders.

    Returns:
        Dictionary mapping dataset keys to their display info
    """
    available = {}

    # Scan data directory for dataset folders
    if DATA_DIR.exists():
        for folder in DATA_DIR.iterdir():
            if folder.is_dir() and folder.name not in ['sample', '__pycache__', '.git']:
                # Check if it has a CSV file
                csv_file = folder / f"{folder.name}.csv"
                if csv_file.exists():
                    # Load metadata if available
                    metadata = get_dataset_metadata(folder.name)

                    # Get dataset info
                    try:
                        df = pd.read_csv(csv_file, nrows=1)  # Just read header
                        size = f"{len(pd.read_csv(csv_file)):,} rows"
                    except:
                        size = "Unknown"

                    available[folder.name] = {
                        'file': str(csv_file),
                        'display_name': metadata.get('name', folder.name.replace('_', ' ').title()),
                        'description': metadata.get('description', 'No description available'),
                        'domain': metadata.get('domain', 'General'),
                        'size': size
                    }

    return available


def load_dataset(dataset_name: str, sample: Optional[int] = None) -> pd.DataFrame:
    """
    Load a dataset by name.

    Args:
        dataset_name: Key from DATASET_REGISTRY or dataset folder name
        sample: Optional number of rows to sample

    Returns:
        DataFrame with the dataset

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset_name is invalid
    """
    # Try new folder structure first: data/dataset_name/dataset_name.csv
    folder_path = DATA_DIR / dataset_name
    file_in_folder = folder_path / f"{dataset_name}.csv"

    if file_in_folder.exists():
        file_path = file_in_folder
    elif dataset_name in DATASET_REGISTRY:
        # Fallback to old structure
        file_path = DATA_DIR / DATASET_REGISTRY[dataset_name]['file']
    else:
        # Try to find the CSV file anywhere in data directory
        possible_paths = list(DATA_DIR.rglob(f"{dataset_name}.csv"))
        if possible_paths:
            file_path = possible_paths[0]
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {', '.join(get_available_datasets().keys())}"
            )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            f"Run 'python download_real_datasets.py' or 'python create_12_datasets.py' to generate datasets."
        )

    logger.info(f"Loading dataset: {dataset_name} from {file_path}")
    df = pd.read_csv(file_path)

    # Sample if requested
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)
        logger.info(f"Sampled {sample} rows from {dataset_name}")

    return df


def get_dataset_config(dataset_name: str) -> Dict[str, any]:
    """
    Get configuration for a dataset (treatment, outcome, confounders).

    Args:
        dataset_name: Key from DATASET_REGISTRY

    Returns:
        Dictionary with 'treatment', 'outcome', 'confounders' keys
    """
    # Load metadata
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}. "
            f"Run 'python create_12_datasets.py' to generate metadata."
        )

    with open(METADATA_FILE, 'r') as f:
        all_metadata = json.load(f)

    if dataset_name not in all_metadata:
        raise ValueError(f"No metadata found for dataset: {dataset_name}")

    metadata = all_metadata[dataset_name]

    return {
        'treatment': metadata['treatment'],
        'outcome': metadata['outcome'],
        'confounders': metadata['confounders']
    }


def get_dataset_metadata(dataset_name: str) -> Dict[str, any]:
    """
    Get full metadata for a dataset including column descriptions.

    Args:
        dataset_name: Key from DATASET_REGISTRY

    Returns:
        Full metadata dictionary
    """
    if not METADATA_FILE.exists():
        return {}

    with open(METADATA_FILE, 'r') as f:
        all_metadata = json.load(f)

    return all_metadata.get(dataset_name, {})


def get_datasets_by_domain() -> Dict[str, List[str]]:
    """
    Group datasets by domain for easier navigation.

    Returns:
        Dictionary mapping domain name to list of dataset keys
    """
    by_domain = {}

    for key, info in DATASET_REGISTRY.items():
        domain = info['domain']
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(key)

    return by_domain


def validate_dataset(df: pd.DataFrame, config: Dict[str, any]) -> Tuple[bool, List[str]]:
    """
    Validate that a dataset has required columns and structure.

    Args:
        df: DataFrame to validate
        config: Configuration with treatment, outcome, confounders

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check treatment column
    if config['treatment'] not in df.columns:
        issues.append(f"Treatment column '{config['treatment']}' not found")
    elif df[config['treatment']].nunique() != 2:
        issues.append(f"Treatment must be binary (found {df[config['treatment']].nunique()} unique values)")

    # Check outcome column
    if config['outcome'] not in df.columns:
        issues.append(f"Outcome column '{config['outcome']}' not found")

    # Check confounders
    for conf in config['confounders']:
        if conf not in df.columns:
            issues.append(f"Confounder '{conf}' not found")

    # Check sample size
    if len(df) < 50:
        issues.append(f"Sample size too small ({len(df)} rows, need 50+)")

    is_valid = len(issues) == 0
    return is_valid, issues


# Backwards compatibility - keep old function names
def list_available_datasets() -> List[str]:
    """Legacy function - returns list of dataset keys."""
    return list(get_available_datasets().keys())
