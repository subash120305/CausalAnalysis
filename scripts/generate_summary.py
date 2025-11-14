"""
Generate summary PDF from results.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Determine results directory relative to script location
script_dir = Path(__file__).parent
project_root = script_dir.parent
results_dir = project_root / "results"
output_pdf = results_dir / "summary.pdf"
output_md = results_dir / "summary.md"

# Create results directory if it doesn't exist
results_dir.mkdir(parents=True, exist_ok=True)


def collect_results() -> dict:
    """Collect results from all datasets."""
    results = {}
    
    datasets = ["ihdp", "twins", "sachs", "acic", "lalonde"]
    
    for dataset in datasets:
        dataset_dir = results_dir / dataset
        
        # Try to load estimator summary
        summary_file = dataset_dir / "estimators_summary.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            results[dataset] = {
                "type": "estimation",
                "data": df
            }
        
        # Try to load discovery results
        discovery_file = dataset_dir / "discovery_precision_recall.csv"
        if discovery_file.exists():
            df = pd.read_csv(discovery_file)
            results[dataset] = {
                "type": "discovery",
                "data": df
            }
        
        # Try ACIC RMSE
        rmse_file = dataset_dir / "ATE_rmse_table.csv"
        if rmse_file.exists():
            df = pd.read_csv(rmse_file)
            results[dataset] = {
                "type": "rmse",
                "data": df
            }
    
    return results


def generate_markdown_summary(results: dict) -> str:
    """Generate markdown summary."""
    md = """# CausalBench: Reproducible Causal Inference Pipelines

## Objective

CausalBench provides a comprehensive, reproducible framework for causal inference including identification, estimation, discovery, and sensitivity analysis using DoWhy and EconML.

## Methods

We implement multiple causal inference methods:

### Estimation Methods
- **IPW (Inverse Propensity Weighting)**: Propensity score weighting via DoWhy
- **PSM (Propensity Score Matching)**: Matching-based estimation
- **DR (Doubly Robust)**: Doubly robust estimation combining outcome and propensity models
- **DML (Double Machine Learning)**: Via EconML
- **DRLearner**: Doubly robust learner from EconML

### Discovery Methods
- **PC Algorithm**: Constraint-based causal discovery
- **FCI Algorithm**: Fast Causal Inference for latent confounders
- **NOTEARS**: Continuous optimization for DAG learning

## Datasets

1. **IHDP**: Semi-synthetic dataset from Infant Health and Development Program
2. **Twins**: Processed twins benchmark dataset
3. **Sachs**: Protein signaling network with known ground-truth DAG
4. **ACIC**: Synthetic benchmark from Atlantic Causal Inference Conference
5. **Lalonde**: Policy evaluation dataset (Dehejia-Wahba)

## Results

"""
    
    # Add results tables
    for dataset, result_info in results.items():
        md += f"\n### {dataset.upper()}\n\n"
        
        if result_info["type"] == "estimation":
            df = result_info["data"]
            try:
                md += df.to_markdown(index=False)
            except AttributeError:
                # Fallback if to_markdown not available
                md += df.to_string(index=False)
        elif result_info["type"] == "discovery":
            df = result_info["data"]
            try:
                md += df.to_markdown(index=False)
            except AttributeError:
                md += df.to_string(index=False)
        elif result_info["type"] == "rmse":
            df = result_info["data"]
            try:
                md += df.to_markdown(index=False)
            except AttributeError:
                md += df.to_string(index=False)
        
        md += "\n"
    
    md += """
## Discussion

This framework demonstrates reproducible causal inference pipelines across multiple estimators and datasets. Results vary by dataset characteristics and estimator assumptions.

## Next Steps

- Extend to additional datasets (Jobs, News, etc.)
- Implement more advanced estimators (Causal Forest, Neural IV)
- Add sensitivity analysis for unobserved confounding
- Benchmark against published results

---
*Generated automatically by CausalBench*
"""
    
    return md


def generate_pdf_summary(results: dict):
    """Generate PDF summary."""
    doc = SimpleDocTemplate(str(output_pdf), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
    )
    
    story.append(Paragraph("CausalBench: Reproducible Causal Inference", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Objective
    story.append(Paragraph("<b>Objective</b>", styles['Heading2']))
    story.append(Paragraph(
        "CausalBench provides a comprehensive, reproducible framework for causal inference "
        "including identification, estimation, discovery, and sensitivity analysis using DoWhy and EconML.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    # Methods
    story.append(Paragraph("<b>Methods</b>", styles['Heading2']))
    methods_text = """
    <b>Estimation:</b> IPW, PSM, Doubly Robust, DML, DRLearner<br/>
    <b>Discovery:</b> PC, FCI, NOTEARS
    """
    story.append(Paragraph(methods_text, styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    # Datasets
    story.append(Paragraph("<b>Datasets</b>", styles['Heading2']))
    datasets_text = "IHDP, Twins, Sachs, ACIC, Lalonde"
    story.append(Paragraph(datasets_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Results
    story.append(Paragraph("<b>Results</b>", styles['Heading2']))
    
    for dataset, result_info in results.items():
        story.append(Paragraph(f"<b>{dataset.upper()}</b>", styles['Heading3']))
        
        df = result_info["data"]
        
        # Create table
        table_data = [df.columns.tolist()]
        for _, row in df.iterrows():
            table_data.append([str(val) for val in row.values])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
    
    # Discussion
    story.append(Paragraph("<b>Discussion</b>", styles['Heading2']))
    story.append(Paragraph(
        "This framework demonstrates reproducible causal inference pipelines across multiple "
        "estimators and datasets. Results vary by dataset characteristics and estimator assumptions.",
        styles['Normal']
    ))
    
    doc.build(story)
    print(f"Generated PDF summary: {output_pdf}")


def main():
    """Main function."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = collect_results()
    
    # Generate markdown
    md_content = generate_markdown_summary(results)
    with open(output_md, 'w') as f:
        f.write(md_content)
    print(f"Generated markdown summary: {output_md}")
    
    # Generate PDF
    if len(results) > 0:
        try:
            generate_pdf_summary(results)
        except Exception as e:
            print(f"Failed to generate PDF: {e}")
            print("Markdown summary still available.")
    else:
        print(f"No results found. Run notebooks first.")
        print(f"Expected results in: {results_dir}")
        print(f"Looking for files like:")
        print(f"  - {results_dir}/ihdp/estimators_summary.csv")
        print(f"  - {results_dir}/twins/estimators_summary.csv")
        print(f"  - {results_dir}/sachs/discovery_precision_recall.csv")
        # Create a placeholder summary anyway
        with open(output_md, 'w') as f:
            f.write("# CausalBench: Reproducible Causal Inference Pipelines\n\n")
            f.write("No results found. Please run the notebooks first.\n\n")
            f.write(f"Run: `python -m pytest tests/` or `bash scripts/run_all.sh`\n")


if __name__ == "__main__":
    main()
