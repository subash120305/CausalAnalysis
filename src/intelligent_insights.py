"""
Intelligent insights generator for causal analysis results.
Provides interpretations, identifies key factors, and suggests actionable recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_insights(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounders: List[str],
    ate_result: float,
    estimator_results: Dict[str, float]
) -> Dict[str, any]:
    """
    Generate intelligent insights from causal analysis.

    Args:
        data: The dataset
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        confounders: List of confounder columns
        ate_result: Average Treatment Effect
        estimator_results: Dict of estimator -> ATE

    Returns:
        Dictionary with insights and recommendations
    """

    insights = {
        "summary": "",
        "key_factors": [],
        "treatment_effect_interpretation": "",
        "recommendations": [],
        "risk_factors": [],
        "success_factors": [],
        "confidence_level": "",
        "target_groups": []
    }

    # 1. Treatment Effect Interpretation
    insights["treatment_effect_interpretation"] = _interpret_treatment_effect(
        ate_result, treatment_col, outcome_col
    )

    # 2. Identify Key Factors
    insights["key_factors"] = _identify_key_factors(
        data, treatment_col, outcome_col, confounders
    )

    # 3. Risk & Success Factors
    insights["risk_factors"], insights["success_factors"] = _identify_risk_success_factors(
        data, treatment_col, outcome_col, confounders, ate_result
    )

    # 4. Target Groups (who benefits most)
    insights["target_groups"] = _identify_target_groups(
        data, treatment_col, outcome_col, confounders, ate_result
    )

    # 5. Actionable Recommendations
    insights["recommendations"] = _generate_recommendations(
        ate_result, treatment_col, outcome_col,
        insights["key_factors"],
        insights["target_groups"]
    )

    # 6. Confidence Assessment
    insights["confidence_level"] = _assess_confidence(estimator_results)

    # 7. Executive Summary
    insights["summary"] = _generate_summary(
        ate_result, treatment_col, outcome_col,
        insights["confidence_level"],
        insights["recommendations"][:2]
    )

    return insights


def _interpret_treatment_effect(ate: float, treatment: str, outcome: str) -> str:
    """Generate plain English interpretation of ATE."""

    # Determine direction
    if ate > 0:
        direction = "increases"
        impact = "positive"
    elif ate < 0:
        direction = "decreases"
        impact = "negative"
    else:
        direction = "has no effect on"
        impact = "neutral"

    # Determine magnitude
    abs_ate = abs(ate)
    if abs_ate < 0.1:
        magnitude = "very small"
    elif abs_ate < 1:
        magnitude = "small"
    elif abs_ate < 5:
        magnitude = "moderate"
    elif abs_ate < 20:
        magnitude = "large"
    else:
        magnitude = "very large"

    interpretation = f"""
The treatment '{treatment}' {direction} '{outcome}' by {abs(ate):.2f} units on average.

This is a {magnitude} {impact} effect. """

    if impact == "positive":
        interpretation += "The treatment appears beneficial."
    elif impact == "negative":
        interpretation += "The treatment may have adverse effects."
    else:
        interpretation += "The treatment has minimal impact."

    return interpretation.strip()


def _identify_key_factors(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str]
) -> List[Dict]:
    """Identify which confounders most strongly affect the outcome."""

    key_factors = []

    for conf in confounders:
        if conf not in data.columns:
            continue

        try:
            # Calculate correlation with outcome
            if pd.api.types.is_numeric_dtype(data[conf]):
                corr = data[conf].corr(data[outcome])

                # Calculate difference between treated/control groups
                treated_mean = data[data[treatment] == 1][conf].mean()
                control_mean = data[data[treatment] == 0][conf].mean()
                difference = treated_mean - control_mean

                # Importance score
                importance = abs(corr) * abs(difference)

                if abs(corr) > 0.1:  # Only include meaningful correlations
                    key_factors.append({
                        "factor": conf,
                        "correlation": corr,
                        "treated_avg": treated_mean,
                        "control_avg": control_mean,
                        "difference": difference,
                        "importance": importance
                    })
        except:
            continue

    # Sort by importance
    key_factors.sort(key=lambda x: x["importance"], reverse=True)

    return key_factors[:5]  # Top 5


def _identify_risk_success_factors(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    ate: float
) -> Tuple[List[str], List[str]]:
    """Identify factors that increase risk vs promote success."""

    risk_factors = []
    success_factors = []

    for conf in confounders:
        if conf not in data.columns or not pd.api.types.is_numeric_dtype(data[conf]):
            continue

        try:
            corr = data[conf].corr(data[outcome])

            # Negative correlation = risk factor (lowers outcome)
            # Positive correlation = success factor (increases outcome)
            if corr < -0.15:
                risk_factors.append(f"{conf} (↓ outcome by {abs(corr):.0%})")
            elif corr > 0.15:
                success_factors.append(f"{conf} (↑ outcome by {corr:.0%})")
        except:
            continue

    return risk_factors[:3], success_factors[:3]


def _identify_target_groups(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    ate: float
) -> List[str]:
    """Identify which groups benefit most from treatment."""

    target_groups = []

    # Analyze treatment heterogeneity
    for conf in confounders:
        if conf not in data.columns or not pd.api.types.is_numeric_dtype(data[conf]):
            continue

        try:
            # Split by median
            median_val = data[conf].median()

            # Effect in high group
            high_group = data[data[conf] >= median_val]
            if len(high_group) > 30:
                treated_high = high_group[high_group[treatment] == 1][outcome].mean()
                control_high = high_group[high_group[treatment] == 0][outcome].mean()
                effect_high = treated_high - control_high

                # Effect in low group
                low_group = data[data[conf] < median_val]
                if len(low_group) > 30:
                    treated_low = low_group[low_group[treatment] == 1][outcome].mean()
                    control_low = low_group[low_group[treatment] == 0][outcome].mean()
                    effect_low = treated_low - control_low

                    # If effect differs substantially
                    if abs(effect_high - effect_low) > abs(ate) * 0.3:
                        if effect_high > effect_low:
                            target_groups.append(
                                f"Higher {conf} (effect: {effect_high:.2f} vs {effect_low:.2f} for lower)"
                            )
                        else:
                            target_groups.append(
                                f"Lower {conf} (effect: {effect_low:.2f} vs {effect_high:.2f} for higher)"
                            )
        except:
            continue

    return target_groups[:3]


def _generate_recommendations(
    ate: float,
    treatment: str,
    outcome: str,
    key_factors: List[Dict],
    target_groups: List[str]
) -> List[str]:
    """Generate actionable recommendations."""

    recommendations = []

    # 1. Primary recommendation based on ATE
    if ate > 0:
        recommendations.append(
            f"✓ IMPLEMENT: '{treatment}' shows positive impact. Consider scaling up to maximize '{outcome}'."
        )

        if target_groups:
            recommendations.append(
                f"✓ PRIORITIZE: Focus on {target_groups[0].split('(')[0].strip()} - they show the strongest response."
            )
    elif ate < 0:
        recommendations.append(
            f"✗ RECONSIDER: '{treatment}' may harm '{outcome}'. Evaluate alternatives or discontinue."
        )
    else:
        recommendations.append(
            f"⚠ NEUTRAL: '{treatment}' has minimal effect on '{outcome}'. Consider cost-effectiveness."
        )

    # 2. Recommendations based on key factors
    if key_factors:
        top_factor = key_factors[0]
        factor_name = top_factor["factor"]
        corr = top_factor["correlation"]

        if corr > 0:
            recommendations.append(
                f"✓ LEVERAGE: '{factor_name}' strongly predicts success. Screen for higher values when selecting candidates."
            )
        else:
            recommendations.append(
                f"⚠ MITIGATE: '{factor_name}' is a risk factor. Provide additional support for those with high values."
            )

    # 3. Data-driven optimization
    if len(key_factors) >= 2:
        second_factor = key_factors[1]["factor"]
        recommendations.append(
            f"✓ MONITOR: Track '{second_factor}' alongside '{key_factors[0]['factor']}' for early intervention."
        )

    # 4. ROI consideration
    if ate > 0:
        recommendations.append(
            f"✓ MEASURE ROI: Effect size is {abs(ate):.2f}. Compare against implementation cost to ensure profitability."
        )

    return recommendations


def _assess_confidence(estimator_results: Dict[str, float]) -> str:
    """Assess confidence level based on agreement between estimators."""

    if len(estimator_results) < 2:
        return "Medium (single estimator - consider running multiple methods)"

    values = [v for v in estimator_results.values() if v is not None and not np.isnan(v)]

    if len(values) < 2:
        return "Low (insufficient valid results)"

    mean_ate = np.mean(values)
    std_ate = np.std(values)

    # Coefficient of variation
    if mean_ate != 0:
        cv = abs(std_ate / mean_ate)
    else:
        cv = std_ate

    if cv < 0.15:
        return "HIGH - All methods agree closely (±15%)"
    elif cv < 0.30:
        return "MEDIUM - Methods show moderate agreement (±30%)"
    else:
        return "LOW - Methods disagree significantly. Results may be sensitive to assumptions."


def _generate_summary(
    ate: float,
    treatment: str,
    outcome: str,
    confidence: str,
    top_recommendations: List[str]
) -> str:
    """Generate executive summary."""

    if ate > 0:
        decision = "RECOMMEND"
        reason = f"increases {outcome} by {ate:.2f}"
    elif ate < 0:
        decision = "DO NOT RECOMMEND"
        reason = f"decreases {outcome} by {abs(ate):.2f}"
    else:
        decision = "NEUTRAL"
        reason = f"has negligible effect on {outcome}"

    summary = f"""
📊 EXECUTIVE SUMMARY

Decision: {decision}
Reason: {treatment} {reason}

Confidence: {confidence}

Top Actions:
{chr(10).join(f'  {i+1}. {rec}' for i, rec in enumerate(top_recommendations))}
"""

    return summary.strip()


def format_insights_for_display(insights: Dict) -> str:
    """Format insights as markdown for Streamlit."""

    output = f"""
## 🎯 Intelligent Analysis

{insights['summary']}

---

### 📈 Treatment Effect

{insights['treatment_effect_interpretation']}

---

### 🔑 Key Factors Affecting Outcome

"""

    if insights['key_factors']:
        for i, factor in enumerate(insights['key_factors'], 1):
            corr_dir = "positively" if factor['correlation'] > 0 else "negatively"
            output += f"""
**{i}. {factor['factor']}**
- Correlation with outcome: {factor['correlation']:.3f} ({corr_dir} related)
- Treated group average: {factor['treated_avg']:.2f}
- Control group average: {factor['control_avg']:.2f}
- Difference: {factor['difference']:.2f}
"""
    else:
        output += "No significant factors identified.\n"

    output += "\n---\n\n"

    # Risk & Success Factors
    if insights['risk_factors'] or insights['success_factors']:
        output += "### ⚠️ Risk & Success Factors\n\n"

        if insights['success_factors']:
            output += "**Success Factors** (promote positive outcomes):\n"
            for factor in insights['success_factors']:
                output += f"- ✓ {factor}\n"
            output += "\n"

        if insights['risk_factors']:
            output += "**Risk Factors** (associated with worse outcomes):\n"
            for factor in insights['risk_factors']:
                output += f"- ⚠️ {factor}\n"
            output += "\n"

        output += "---\n\n"

    # Target Groups
    if insights['target_groups']:
        output += "### 🎯 Who Benefits Most?\n\n"
        output += "Treatment is most effective for:\n"
        for group in insights['target_groups']:
            output += f"- {group}\n"
        output += "\n---\n\n"

    # Recommendations
    output += "### 💡 Actionable Recommendations\n\n"
    for rec in insights['recommendations']:
        output += f"{rec}\n\n"

    return output
