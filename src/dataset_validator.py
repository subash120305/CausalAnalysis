"""
Intelligent dataset validator for causal inference.
Detects if uploaded data is suitable for causal analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DatasetValidationResult:
    """Result of dataset validation."""

    def __init__(self):
        self.is_valid = False
        self.confidence = 0.0  # 0-1 score
        self.issues = []
        self.warnings = []
        self.suggestions = {}
        self.detected_columns = {
            "treatment_candidates": [],
            "outcome_candidates": [],
            "confounder_candidates": []
        }
        self.reason = ""


def validate_for_causal_inference(df: pd.DataFrame) -> DatasetValidationResult:
    """
    Intelligently validate if dataset is suitable for causal inference.

    Args:
        df: Input DataFrame

    Returns:
        DatasetValidationResult with validation details
    """
    result = DatasetValidationResult()

    # Basic checks
    if len(df) < 50:
        result.issues.append("Dataset too small (< 50 rows). Need at least 100+ for reliable inference.")
        result.reason = "Insufficient data for causal analysis"
        return result

    if len(df.columns) < 3:
        result.issues.append("Too few columns (< 3). Need: treatment, outcome, and confounders.")
        result.reason = "Missing required variables (need treatment, outcome, confounders)"
        return result

    # Detect potential treatment columns (binary or categorical)
    treatment_candidates = _detect_treatment_columns(df)
    result.detected_columns["treatment_candidates"] = treatment_candidates

    # Detect potential outcome columns (continuous or binary)
    outcome_candidates = _detect_outcome_columns(df)
    result.detected_columns["outcome_candidates"] = outcome_candidates

    # Detect potential confounders
    confounder_candidates = _detect_confounders(df, treatment_candidates, outcome_candidates)
    result.detected_columns["confounder_candidates"] = confounder_candidates

    # Validation logic
    if len(treatment_candidates) == 0:
        result.issues.append("No binary treatment variable detected. Causal inference needs a treatment/intervention column (0/1 or Yes/No).")
        result.suggestions["treatment"] = "Add a column indicating who received treatment/intervention (binary: 0/1, True/False)"
        result.reason = "No treatment variable found"
        return result

    if len(outcome_candidates) == 0:
        result.issues.append("No suitable outcome variable detected. Need a measurable outcome (continuous or binary).")
        result.suggestions["outcome"] = "Add a column with the outcome you want to measure (e.g., sales, recovery time, test scores)"
        result.reason = "No outcome variable found"
        return result

    if len(confounder_candidates) == 0:
        result.warnings.append("No confounders detected. Results may be biased if treatment assignment isn't random.")
        result.suggestions["confounders"] = "Consider adding variables that affect both treatment and outcome (e.g., age, prior history)"

    # Check for common data quality issues
    _check_data_quality(df, result)

    # Calculate confidence score
    confidence_score = _calculate_confidence(
        len(df),
        len(treatment_candidates),
        len(outcome_candidates),
        len(confounder_candidates),
        len(result.issues)
    )
    result.confidence = confidence_score

    # Final decision
    if len(result.issues) == 0:
        result.is_valid = True
        result.reason = f"Dataset appears suitable for causal inference (confidence: {confidence_score:.0%})"
    else:
        result.is_valid = False
        if not result.reason:
            result.reason = "Dataset structure not suitable for causal inference"

    return result


def _detect_treatment_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect potential treatment columns (binary or low-cardinality categorical)."""
    candidates = []

    # Common treatment column name patterns
    treatment_keywords = [
        'treatment', 'treat', 'intervention', 'exposed', 'exposure',
        'assigned', 'group', 'condition', 'therapy', 'drug', 'medication',
        'training', 'program', 'campaign', 'test', 'experimental'
    ]

    for col in df.columns:
        col_lower = col.lower()

        # Skip ID columns
        if 'id' in col_lower or col_lower in ['index', 'row']:
            continue

        # Check if binary
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)

        if n_unique == 2:
            # Binary column - strong candidate
            score = 0.8

            # Boost score if name suggests treatment
            if any(kw in col_lower for kw in treatment_keywords):
                score = 0.95

            # Check if values are 0/1, True/False, Yes/No
            vals_set = set(str(v).lower() for v in unique_vals)
            if vals_set in [{'0', '1'}, {'true', 'false'}, {'yes', 'no'},
                           {'0.0', '1.0'}, {'treated', 'control'}]:
                score = max(score, 0.9)

            candidates.append({
                "name": col,
                "score": score,
                "type": "binary",
                "values": list(unique_vals),
                "reason": f"Binary column with values {list(unique_vals)}"
            })

        elif 2 < n_unique <= 5:
            # Low cardinality - possible treatment groups
            score = 0.5
            if any(kw in col_lower for kw in treatment_keywords):
                score = 0.7

            candidates.append({
                "name": col,
                "score": score,
                "type": "categorical",
                "values": list(unique_vals),
                "reason": f"Categorical with {n_unique} groups: {list(unique_vals)[:3]}"
            })

    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _detect_outcome_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect potential outcome columns."""
    candidates = []

    outcome_keywords = [
        'outcome', 'result', 'response', 'effect', 'score', 'value',
        'revenue', 'sales', 'profit', 'earnings', 'income', 'salary',
        'time', 'duration', 'days', 'rate', 'percentage', 'success',
        'recovery', 'survival', 'health', 'performance', 'grade',
        'conversion', 'click', 'purchase', 'engagement', 'satisfaction'
    ]

    for col in df.columns:
        col_lower = col.lower()

        # Skip ID columns
        if 'id' in col_lower or col_lower in ['index', 'row']:
            continue

        # Check data type
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()

            # Continuous outcome (many unique values)
            if unique_count > 10:
                score = 0.6

                # Boost if name suggests outcome
                if any(kw in col_lower for kw in outcome_keywords):
                    score = 0.85

                candidates.append({
                    "name": col,
                    "score": score,
                    "type": "continuous",
                    "unique_count": unique_count,
                    "reason": f"Continuous numeric variable ({unique_count} unique values)"
                })

            # Binary outcome
            elif unique_count == 2:
                score = 0.7
                if any(kw in col_lower for kw in outcome_keywords):
                    score = 0.9

                candidates.append({
                    "name": col,
                    "score": score,
                    "type": "binary",
                    "unique_count": unique_count,
                    "reason": "Binary outcome variable"
                })

    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _detect_confounders(
    df: pd.DataFrame,
    treatment_candidates: List[Dict],
    outcome_candidates: List[Dict]
) -> List[str]:
    """Detect potential confounder columns."""
    confounders = []

    # Exclude treatment and outcome columns
    excluded_cols = set()
    if treatment_candidates:
        excluded_cols.add(treatment_candidates[0]["name"])
    if outcome_candidates:
        excluded_cols.add(outcome_candidates[0]["name"])

    for col in df.columns:
        col_lower = col.lower()

        # Skip IDs, treatment, outcome
        if col in excluded_cols:
            continue
        if 'id' in col_lower or col_lower in ['index', 'row']:
            continue

        # Numeric or reasonable categorical columns can be confounders
        if pd.api.types.is_numeric_dtype(df[col]):
            confounders.append(col)
        elif df[col].nunique() <= 20:  # Low-cardinality categorical
            confounders.append(col)

    return confounders


def _check_data_quality(df: pd.DataFrame, result: DatasetValidationResult):
    """Check for common data quality issues."""

    # Missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).max()
    if missing_pct > 50:
        result.warnings.append(f"Some columns have >50% missing values. Consider imputation or removal.")
    elif missing_pct > 20:
        result.warnings.append(f"Some columns have >20% missing values.")

    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        result.warnings.append(f"Constant columns detected: {constant_cols}. These won't help analysis.")

    # Check sample size per treatment group
    if result.detected_columns["treatment_candidates"]:
        treat_col = result.detected_columns["treatment_candidates"][0]["name"]
        group_sizes = df[treat_col].value_counts()

        if group_sizes.min() < 30:
            result.warnings.append(f"Small treatment group detected ({group_sizes.min()} samples). Need 30+ per group for reliable estimates.")


def _calculate_confidence(
    n_rows: int,
    n_treatment: int,
    n_outcome: int,
    n_confounders: int,
    n_issues: int
) -> float:
    """Calculate confidence score for dataset suitability."""

    if n_issues > 0:
        return 0.0

    score = 0.0

    # Sample size
    if n_rows >= 500:
        score += 0.3
    elif n_rows >= 200:
        score += 0.2
    elif n_rows >= 100:
        score += 0.1

    # Treatment detection
    if n_treatment > 0:
        score += 0.3

    # Outcome detection
    if n_outcome > 0:
        score += 0.3

    # Confounders
    if n_confounders >= 3:
        score += 0.1
    elif n_confounders >= 1:
        score += 0.05

    return min(score, 1.0)


def suggest_column_mapping(validation_result: DatasetValidationResult) -> Dict[str, str]:
    """
    Suggest best column mapping for causal analysis.

    Returns:
        Dictionary with 'treatment', 'outcome', 'confounders'
    """
    mapping = {}

    if validation_result.detected_columns["treatment_candidates"]:
        mapping["treatment"] = validation_result.detected_columns["treatment_candidates"][0]["name"]

    if validation_result.detected_columns["outcome_candidates"]:
        mapping["outcome"] = validation_result.detected_columns["outcome_candidates"][0]["name"]

    mapping["confounders"] = validation_result.detected_columns["confounder_candidates"]

    return mapping
