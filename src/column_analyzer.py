"""
Intelligent column analyzer that understands dataset context.
Provides ChatGPT-like descriptions of columns with their significance and impact.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    """Load metadata for a dataset from the metadata JSON file."""
    metadata_path = Path("data/dataset_metadata.json")

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    try:
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        return all_metadata.get(dataset_name, {})
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return {}


def analyze_column_statistics(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for a column.

    Returns:
        Dictionary with statistics like min, max, mean, unique values, etc.
    """
    stats = {
        'name': column,
        'dtype': str(df[column].dtype),
        'missing_count': int(df[column].isna().sum()),
        'missing_pct': float(df[column].isna().mean() * 100),
        'non_null_count': int(df[column].notna().sum())
    }

    # Numeric columns
    if pd.api.types.is_numeric_dtype(df[column]):
        stats['type'] = 'numeric'
        stats['min'] = float(df[column].min())
        stats['max'] = float(df[column].max())
        stats['mean'] = float(df[column].mean())
        stats['median'] = float(df[column].median())
        stats['std'] = float(df[column].std())
        stats['unique_values'] = int(df[column].nunique())

        # Check if binary
        unique_vals = df[column].dropna().unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            stats['is_binary'] = True
            stats['distribution'] = {
                '0': int((df[column] == 0).sum()),
                '1': int((df[column] == 1).sum())
            }
        else:
            stats['is_binary'] = False

    # Categorical columns
    else:
        stats['type'] = 'categorical'
        stats['unique_values'] = int(df[column].nunique())
        value_counts = df[column].value_counts()
        stats['top_values'] = {
            str(k): int(v) for k, v in value_counts.head(10).items()
        }

        # Check if binary
        if stats['unique_values'] == 2:
            stats['is_binary'] = True
            stats['distribution'] = stats['top_values']
        else:
            stats['is_binary'] = False

    return stats


def generate_column_description(
    column_stats: Dict[str, Any],
    metadata: Dict[str, Any],
    column_name: str
) -> str:
    """
    Generate a ChatGPT-like natural language description of a column.

    Uses both statistics and metadata to create comprehensive explanation.
    """

    # Get metadata for this specific column if available
    col_metadata = metadata.get('columns', {}).get(column_name, {})

    description_parts = []

    # 1. What is this column?
    if 'description' in col_metadata:
        description_parts.append(f"**What it is:** {col_metadata['description']}")
    else:
        description_parts.append(f"**Column:** {column_name}")

    # 2. Type and values
    if column_stats.get('is_binary'):
        if 'values' in col_metadata:
            val_0 = col_metadata['values'].get('0', '0')
            val_1 = col_metadata['values'].get('1', '1')
            description_parts.append(f"**Type:** Binary (Yes/No)")
            description_parts.append(f"**Values:** `{val_0}` (0) vs `{val_1}` (1)")
        else:
            description_parts.append(f"**Type:** Binary variable (0 or 1)")

        # Distribution
        if 'distribution' in column_stats:
            dist = column_stats['distribution']
            total = sum(dist.values()) if isinstance(dist, dict) else column_stats['non_null_count']
            if isinstance(dist, dict) and '0' in dist and '1' in dist:
                pct_0 = (dist['0'] / total * 100) if total > 0 else 0
                pct_1 = (dist['1'] / total * 100) if total > 0 else 0
                description_parts.append(f"**Distribution:** {dist['0']} ({pct_0:.1f}%) vs {dist['1']} ({pct_1:.1f}%)")

    elif column_stats['type'] == 'numeric':
        # Numeric column
        unit = col_metadata.get('unit', 'units')
        description_parts.append(f"**Type:** Continuous numerical")
        description_parts.append(
            f"**Range:** {column_stats['min']:.2f} to {column_stats['max']:.2f} {unit}"
        )
        description_parts.append(
            f"**Average:** {column_stats['mean']:.2f} {unit} (median: {column_stats['median']:.2f})"
        )

        # Interpret the range
        if 'range' in col_metadata:
            expected_range = col_metadata['range']
            description_parts.append(
                f"**Expected range:** {expected_range[0]} - {expected_range[1]} {unit}"
            )

    elif column_stats['type'] == 'categorical':
        description_parts.append(f"**Type:** Categorical")
        description_parts.append(f"**Unique values:** {column_stats['unique_values']}")

        if 'values' in col_metadata:
            # Show what each value means
            val_meanings = col_metadata['values']
            description_parts.append("**Value meanings:**")
            for key, meaning in val_meanings.items():
                count = column_stats.get('top_values', {}).get(key, 0)
                description_parts.append(f"  - `{key}`: {meaning} ({count} observations)")

    # 3. How does it impact the outcome?
    if 'impact' in col_metadata:
        description_parts.append(f"\n**💡 Impact on outcome:** {col_metadata['impact']}")

    # 4. Data quality
    if column_stats['missing_pct'] > 0:
        description_parts.append(
            f"\n⚠️ **Missing data:** {column_stats['missing_count']} "
            f"({column_stats['missing_pct']:.1f}%) values are missing"
        )

    return "\n".join(description_parts)


def analyze_all_columns(
    df: pd.DataFrame,
    dataset_name: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all columns in a dataset and generate comprehensive descriptions.

    Args:
        df: DataFrame to analyze
        dataset_name: Name of dataset to load metadata for

    Returns:
        Dictionary mapping column names to their full analysis
    """

    # Load metadata if dataset name provided
    metadata = {}
    if dataset_name:
        metadata = load_dataset_metadata(dataset_name)

    analysis = {}

    for column in df.columns:
        try:
            # Get statistics
            stats = analyze_column_statistics(df, column)

            # Generate description
            description = generate_column_description(stats, metadata, column)

            analysis[column] = {
                'statistics': stats,
                'description': description,
                'metadata': metadata.get('columns', {}).get(column, {})
            }

        except Exception as e:
            logger.error(f"Failed to analyze column {column}: {e}")
            analysis[column] = {
                'statistics': {'name': column, 'error': str(e)},
                'description': f"**Column:** {column}\n⚠️ Analysis failed: {e}",
                'metadata': {}
            }

    return analysis


def format_column_analysis_for_display(column_analysis: Dict[str, Any]) -> str:
    """
    Format column analysis as markdown for Streamlit display.

    Args:
        column_analysis: Dictionary from analyze_all_columns()

    Returns:
        Formatted markdown string
    """

    output = "## 📊 Dataset Column Analysis\n\n"
    output += "_Understanding what each variable means and how it affects the outcome_\n\n"
    output += "---\n\n"

    for column_name, analysis in column_analysis.items():
        output += f"### 📌 {column_name}\n\n"
        output += analysis['description']
        output += "\n\n---\n\n"

    return output


def get_dataset_overview(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """
    Generate an overview of the entire dataset.

    Returns:
        Markdown-formatted overview string
    """

    overview = "## 📋 Dataset Overview\n\n"

    # Basic info
    if 'name' in metadata:
        overview += f"**Name:** {metadata['name']}\n\n"
    if 'domain' in metadata:
        overview += f"**Domain:** {metadata['domain']}\n\n"
    if 'description' in metadata:
        overview += f"**Description:** {metadata['description']}\n\n"

    overview += f"**Size:** {len(df):,} rows × {len(df.columns)} columns\n\n"

    # Study design
    if 'treatment' in metadata and 'outcome' in metadata:
        overview += "### 🎯 Causal Question\n\n"
        treatment = metadata['treatment']
        outcome = metadata['outcome']

        # Get treatment description
        treat_desc = metadata.get('columns', {}).get(treatment, {}).get('description', treatment)
        outcome_desc = metadata.get('columns', {}).get(outcome, {}).get('description', outcome)

        overview += f"**Research Question:** Does **{treat_desc}** affect **{outcome_desc}**?\n\n"

        # Treatment distribution
        if treatment in df.columns:
            treat_counts = df[treatment].value_counts()
            if 1 in treat_counts.index and 0 in treat_counts.index:
                treated = treat_counts[1]
                control = treat_counts[0]
                overview += f"- **Treated group:** {treated:,} ({treated/len(df)*100:.1f}%)\n"
                overview += f"- **Control group:** {control:,} ({control/len(df)*100:.1f}%)\n\n"

        # Confounders
        if 'confounders' in metadata:
            confounders = metadata['confounders']
            overview += f"**Confounding variables** (factors to control for): {len(confounders)}\n"
            overview += "- " + "\n- ".join(confounders) + "\n\n"

    # Expected effect
    if 'true_ate' in metadata:
        true_ate = metadata['true_ate']
        overview += f"**Known treatment effect** (for validation): {true_ate}\n\n"

    overview += "---\n\n"

    return overview


def analyze_uploaded_dataset(
    df: pd.DataFrame,
    treatment_col: str = None,
    outcome_col: str = None,
    confounders: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze an uploaded dataset without metadata.
    Intelligently infers column meanings from names and data.

    Args:
        df: DataFrame to analyze
        treatment_col: Name of treatment column (if known)
        outcome_col: Name of outcome column (if known)
        confounders: List of confounder columns (if known)

    Returns:
        Complete analysis dictionary
    """

    analysis = {
        'overview': {},
        'columns': {},
        'recommendations': []
    }

    # Basic overview
    analysis['overview'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_data': df.isna().sum().sum(),
        'missing_pct': (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
    }

    # Analyze each column
    for column in df.columns:
        try:
            stats = analyze_column_statistics(df, column)

            # Infer meaning from column name
            inferred_meaning = _infer_column_meaning(column, stats)

            # Determine role in causal analysis
            role = "Unknown"
            if treatment_col and column == treatment_col:
                role = "TREATMENT (Independent Variable)"
            elif outcome_col and column == outcome_col:
                role = "OUTCOME (Dependent Variable)"
            elif confounders and column in confounders:
                role = "CONFOUNDER (Control Variable)"

            description = f"**Role:** {role}\n\n"
            description += f"**Inferred meaning:** {inferred_meaning}\n\n"

            # Add statistics
            if stats['type'] == 'numeric':
                description += f"**Type:** Numerical\n"
                description += f"**Range:** {stats['min']:.2f} to {stats['max']:.2f}\n"
                description += f"**Mean:** {stats['mean']:.2f} (std: {stats['std']:.2f})\n"
                description += f"**Median:** {stats['median']:.2f}\n"
            else:
                description += f"**Type:** Categorical\n"
                description += f"**Unique values:** {stats['unique_values']}\n"
                if 'top_values' in stats:
                    description += "**Most common:** " + ", ".join([f"{k} ({v})" for k, v in list(stats['top_values'].items())[:3]]) + "\n"

            if stats['missing_pct'] > 0:
                description += f"\n⚠️ **Missing:** {stats['missing_pct']:.1f}%\n"

            analysis['columns'][column] = {
                'statistics': stats,
                'description': description,
                'role': role
            }

        except Exception as e:
            logger.error(f"Failed to analyze column {column}: {e}")

    # Generate recommendations
    analysis['recommendations'] = _generate_analysis_recommendations(df, analysis)

    return analysis


def _infer_column_meaning(column_name: str, stats: Dict[str, Any]) -> str:
    """Infer what a column likely represents based on its name and statistics."""

    name_lower = column_name.lower()

    # Common patterns
    if any(word in name_lower for word in ['id', 'identifier', 'key']):
        return "Likely an identifier/ID column (not used in analysis)"

    if any(word in name_lower for word in ['age', 'years', 'year']):
        return "Age or time-related variable"

    if any(word in name_lower for word in ['price', 'cost', 'amount', 'salary', 'income', 'revenue']):
        return "Financial/monetary variable"

    if any(word in name_lower for word in ['score', 'rating', 'grade']):
        return "Performance or quality metric"

    if any(word in name_lower for word in ['count', 'number', 'total']):
        return "Count or quantity variable"

    if any(word in name_lower for word in ['rate', 'percentage', 'pct', 'ratio']):
        return "Proportion or rate variable"

    if any(word in name_lower for word in ['treatment', 'intervention', 'exposed', 'drug', 'trained']):
        return "Possible treatment/intervention variable"

    if any(word in name_lower for word in ['outcome', 'result', 'success', 'failure', 'target']):
        return "Possible outcome/target variable"

    # Binary variables
    if stats.get('is_binary'):
        return "Binary yes/no variable"

    # Default
    if stats['type'] == 'numeric':
        return "Numerical variable (continuous measurement)"
    else:
        return "Categorical variable (grouping factor)"


def _generate_analysis_recommendations(df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations for improving the analysis."""

    recommendations = []

    # Check for missing data
    if analysis['overview']['missing_pct'] > 5:
        recommendations.append(
            f"⚠️ Dataset has {analysis['overview']['missing_pct']:.1f}% missing data. "
            "Consider imputation or removal of incomplete rows."
        )

    # Check sample size
    if analysis['overview']['rows'] < 100:
        recommendations.append(
            "⚠️ Small sample size (<100). Results may be unreliable. Consider collecting more data."
        )

    # Check for constant columns
    constant_cols = [
        col for col, info in analysis['columns'].items()
        if info['statistics'].get('unique_values', 0) == 1
    ]
    if constant_cols:
        recommendations.append(
            f"⚠️ Columns with no variation: {', '.join(constant_cols)}. These should be removed."
        )

    return recommendations
