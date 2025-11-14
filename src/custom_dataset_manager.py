"""
Manager for custom uploaded datasets with caching and persistence.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

CUSTOM_DATA_DIR = Path("data/custom_uploads")
CACHE_FILE = CUSTOM_DATA_DIR / "dataset_cache.json"
RESULTS_CACHE_FILE = CUSTOM_DATA_DIR / "results_cache.json"


class CustomDatasetManager:
    """Manages custom uploaded datasets and their analysis results."""

    def __init__(self):
        self.cache = self._load_cache()
        self.results_cache = self._load_results_cache()
        CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> Dict:
        """Load dataset metadata cache."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                return {}
        return {}

    def _load_results_cache(self) -> Dict:
        """Load analysis results cache."""
        if RESULTS_CACHE_FILE.exists():
            try:
                with open(RESULTS_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load results cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save dataset metadata cache."""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _save_results_cache(self):
        """Save analysis results cache."""
        try:
            with open(RESULTS_CACHE_FILE, 'w') as f:
                json.dump(self.results_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save results cache: {e}")

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for caching."""
        # Use first/last rows + shape as fingerprint
        fingerprint = f"{df.shape}_{df.columns.tolist()}"
        if len(df) > 0:
            fingerprint += f"_{df.iloc[0].to_dict()}_{df.iloc[-1].to_dict()}"
        return hashlib.md5(fingerprint.encode()).hexdigest()[:12]

    def save_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        source: str,
        column_mapping: Dict[str, Any],
        validation_result: Any
    ) -> str:
        """
        Save custom dataset and metadata.

        Args:
            df: DataFrame to save
            name: User-provided name
            source: Source description (uploaded file, kaggle link, etc.)
            column_mapping: Detected/confirmed column mapping
            validation_result: Validation result object

        Returns:
            Dataset ID (hash)
        """
        dataset_id = self._compute_hash(df)

        # Save dataframe
        data_path = CUSTOM_DATA_DIR / f"{dataset_id}.csv"
        df.to_csv(data_path, index=False)

        # Save metadata
        metadata = {
            "id": dataset_id,
            "name": name,
            "source": source,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "column_mapping": column_mapping,
            "is_valid": validation_result.is_valid,
            "confidence": validation_result.confidence,
            "data_path": str(data_path)
        }

        self.cache[dataset_id] = metadata
        self._save_cache()

        logger.info(f"Saved dataset '{name}' with ID {dataset_id}")
        return dataset_id

    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset by ID."""
        if dataset_id not in self.cache:
            return None

        data_path = Path(self.cache[dataset_id]["data_path"])
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return None

        return pd.read_csv(data_path)

    def get_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset metadata."""
        return self.cache.get(dataset_id)

    def list_datasets(self) -> List[Dict]:
        """List all cached datasets."""
        return [
            {
                "id": dataset_id,
                "name": meta["name"],
                "uploaded_at": meta["uploaded_at"],
                "n_rows": meta["n_rows"],
                "is_valid": meta["is_valid"],
                "confidence": meta["confidence"]
            }
            for dataset_id, meta in self.cache.items()
        ]

    def save_analysis_results(
        self,
        dataset_id: str,
        results_df: pd.DataFrame,
        estimators: List[str],
        column_mapping: Dict[str, Any]
    ):
        """Cache analysis results."""
        self.results_cache[dataset_id] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "estimators": estimators,
            "column_mapping": column_mapping,
            "results": results_df.to_dict('records')
        }
        self._save_results_cache()

    def get_cached_results(self, dataset_id: str) -> Optional[Dict]:
        """Get cached analysis results."""
        return self.results_cache.get(dataset_id)

    def has_cached_results(self, dataset_id: str) -> bool:
        """Check if results are cached for dataset."""
        return dataset_id in self.results_cache

    def delete_dataset(self, dataset_id: str):
        """Delete dataset and its results."""
        if dataset_id in self.cache:
            # Delete file
            data_path = Path(self.cache[dataset_id]["data_path"])
            if data_path.exists():
                data_path.unlink()

            # Remove from caches
            del self.cache[dataset_id]
            self._save_cache()

            if dataset_id in self.results_cache:
                del self.results_cache[dataset_id]
                self._save_results_cache()

            logger.info(f"Deleted dataset {dataset_id}")


def load_kaggle_dataset(kaggle_url: str) -> Optional[pd.DataFrame]:
    """
    Download dataset from Kaggle URL.

    Args:
        kaggle_url: Kaggle dataset URL or identifier

    Returns:
        DataFrame or None if failed
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        # Parse URL to get dataset identifier
        # Example: https://www.kaggle.com/datasets/username/dataset-name
        if "kaggle.com/datasets/" in kaggle_url:
            parts = kaggle_url.split("datasets/")[-1].rstrip("/").split("/")
            if len(parts) >= 2:
                owner = parts[0]
                dataset_name = parts[1]
                dataset_id = f"{owner}/{dataset_name}"
            else:
                dataset_id = kaggle_url.split("datasets/")[-1].rstrip("/")
        else:
            # Assume it's already in format: owner/dataset-name
            dataset_id = kaggle_url

        # Download to temp directory
        download_path = CUSTOM_DATA_DIR / "temp_kaggle"
        download_path.mkdir(exist_ok=True)

        logger.info(f"Downloading Kaggle dataset: {dataset_id}")
        api.dataset_download_files(dataset_id, path=str(download_path), unzip=True)

        # Find CSV file
        csv_files = list(download_path.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in Kaggle dataset")
            return None

        # Load first CSV
        df = pd.read_csv(csv_files[0])

        # Cleanup temp files
        for file in download_path.glob("*"):
            file.unlink()

        logger.info(f"Loaded Kaggle dataset: {len(df)} rows, {len(df.columns)} cols")
        return df

    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        return None
    except Exception as e:
        logger.error(f"Failed to load Kaggle dataset: {e}")
        return None
