#!/usr/bin/env python3
"""
Results Table Generator for Nutrition5k Project
QA-Agent | Phase 5

Generates formatted tables for paper/milestone report:
- Table 1: Classification Results (EfficientNet-B0, EfficientNet-B4, ViT-B/16)
- Table 2: Segmentation Results (Mask R-CNN)
- Table 3: Regression Results (Multi-output: calories, protein, carb, fat, mass)
- Table 4: End-to-End Pipeline Results

Usage:
    python scripts/generate_results_table.py --results-dir experiments/evaluation
    python scripts/generate_results_table.py --results-dir experiments/evaluation --format latex
    python scripts/generate_results_table.py --results-dir experiments/evaluation --format markdown
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


# ============================================================================
# Table Formatters
# ============================================================================

class TableFormatter:
    """Base class for table formatting."""

    @staticmethod
    def format_value(value: float, precision: int = 2) -> str:
        """Format a numeric value."""
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)


class MarkdownFormatter(TableFormatter):
    """Format tables as Markdown."""

    @staticmethod
    def format_table(headers: List[str], rows: List[List[Any]],
                    title: str = "") -> str:
        """Format as markdown table."""
        lines = []

        if title:
            lines.append(f"### {title}")
            lines.append("")

        # Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Rows
        for row in rows:
            formatted = [str(cell) for cell in row]
            lines.append("| " + " | ".join(formatted) + " |")

        lines.append("")
        return "\n".join(lines)


class LaTeXFormatter(TableFormatter):
    """Format tables as LaTeX."""

    @staticmethod
    def format_table(headers: List[str], rows: List[List[Any]],
                    title: str = "", label: str = "") -> str:
        """Format as LaTeX table."""
        lines = []

        # Table environment
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")

        if title:
            lines.append(f"\\caption{{{title}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

        # Tabular
        col_spec = "|" + "c|" * len(headers)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\hline")

        # Header
        lines.append(" & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\")
        lines.append("\\hline")

        # Rows
        for row in rows:
            formatted = [str(cell) for cell in row]
            lines.append(" & ".join(formatted) + " \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)


class CSVFormatter(TableFormatter):
    """Format tables as CSV."""

    @staticmethod
    def format_table(headers: List[str], rows: List[List[Any]],
                    title: str = "") -> str:
        """Format as CSV."""
        lines = []

        if title:
            lines.append(f"# {title}")

        lines.append(",".join(headers))

        for row in rows:
            lines.append(",".join([str(cell) for cell in row]))

        return "\n".join(lines)


# ============================================================================
# Table Generators
# ============================================================================

def generate_table1_classification(
    results: Dict[str, Dict[str, Any]],
    formatter: TableFormatter
) -> str:
    """
    Generate Table 1: Classification Results.

    Compares different backbones: EfficientNet-B0, EfficientNet-B4, ViT-B/16
    """
    headers = ["Model", "Backbone", "Top-1 Acc (%)", "Top-5 Acc (%)", "F1 Score", "Params (M)"]

    rows = []

    # Expected model configurations
    configs = [
        ("Classifier", "EfficientNet-B0", "efficientnet_b0", 5.0),
        ("Classifier", "EfficientNet-B4", "efficientnet_b4", 19.0),
        ("Classifier", "ViT-B/16", "vit_b_16", 86.0),
    ]

    for model_name, backbone, key, params in configs:
        if key in results:
            r = results[key]
            rows.append([
                model_name,
                backbone,
                f"{r.get('top1_accuracy', 0):.2f}",
                f"{r.get('top5_accuracy', 0):.2f}",
                f"{r.get('f1_score', 0):.4f}",
                f"{params:.1f}"
            ])
        else:
            # Placeholder for missing results
            rows.append([
                model_name,
                backbone,
                "-",
                "-",
                "-",
                f"{params:.1f}"
            ])

    return formatter.format_table(headers, rows, title="Table 1: Classification Results")


def generate_table2_segmentation(
    results: Dict[str, Any],
    formatter: TableFormatter
) -> str:
    """
    Generate Table 2: Segmentation Results.

    Mask R-CNN with ResNet-50-FPN backbone.
    """
    headers = ["Model", "Backbone", "mAP@0.5", "mAP@0.5:0.95", "Mean IoU", "Params (M)"]

    rows = []

    if results:
        rows.append([
            "Mask R-CNN",
            "ResNet-50-FPN",
            f"{results.get('mAP_50', results.get('map_50', 0)):.4f}",
            f"{results.get('mAP_50_95', results.get('map', 0)):.4f}",
            f"{results.get('mean_iou', 0):.4f}",
            "44.4"
        ])
    else:
        rows.append([
            "Mask R-CNN",
            "ResNet-50-FPN",
            "-",
            "-",
            "-",
            "44.4"
        ])

    return formatter.format_table(headers, rows, title="Table 2: Segmentation Results")


def generate_table3_regression(
    results: Dict[str, Any],
    formatter: TableFormatter
) -> str:
    """
    Generate Table 3: Regression Results.

    Multi-output regression for calories, protein, carb, fat, mass.
    """
    headers = ["Output", "MAE", "RMSE", "MAPE (%)", "R²"]

    outputs = ['calories', 'protein', 'carb', 'fat', 'mass']
    rows = []

    for output in outputs:
        if results and f'{output}_mae' in results:
            rows.append([
                output.capitalize(),
                f"{results[f'{output}_mae']:.4f}",
                f"{results[f'{output}_rmse']:.4f}",
                f"{results[f'{output}_mape']:.2f}",
                f"{results.get(f'{output}_r2', 0):.4f}"
            ])
        else:
            rows.append([
                output.capitalize(),
                "-",
                "-",
                "-",
                "-"
            ])

    # Overall row
    if results and 'overall_mae' in results:
        rows.append([
            "**Overall**",
            f"{results['overall_mae']:.4f}",
            f"{results['overall_rmse']:.4f}",
            f"{results['overall_mape']:.2f}",
            f"{results.get('overall_r2', 0):.4f}"
        ])

    return formatter.format_table(headers, rows, title="Table 3: Regression Results")


def generate_table4_end_to_end(
    results: Dict[str, Any],
    formatter: TableFormatter
) -> str:
    """
    Generate Table 4: End-to-End Pipeline Results.

    Combined metrics from all three tasks.
    """
    headers = ["Task", "Metric", "Value"]

    rows = []

    if results:
        # Classification metrics
        if 'classification' in results:
            cls = results['classification']
            rows.append(["Classification", "Top-1 Accuracy (%)", f"{cls.get('top1_accuracy', 0):.2f}"])
            rows.append(["", "Top-5 Accuracy (%)", f"{cls.get('top5_accuracy', 0):.2f}"])
            rows.append(["", "F1 Score", f"{cls.get('f1_score', 0):.4f}"])

        # Segmentation metrics
        if 'segmentation' in results:
            seg = results['segmentation']
            rows.append(["Segmentation", "Avg Detections/Image", f"{seg.get('avg_detections_per_image', 0):.2f}"])

        # Regression metrics
        if 'regression' in results:
            reg = results['regression']
            rows.append(["Regression", "Calories MAE", f"{reg.get('calories_mae', 0):.4f}"])
            rows.append(["", "Calories MAPE (%)", f"{reg.get('calories_mape', 0):.2f}"])
            rows.append(["", "Protein MAE", f"{reg.get('protein_mae', 0):.4f}"])
            rows.append(["", "Mass MAE", f"{reg.get('mass_mae', 0):.4f}"])
    else:
        rows.append(["Classification", "Top-1 Accuracy (%)", "-"])
        rows.append(["Segmentation", "mAP@0.5", "-"])
        rows.append(["Regression", "Calories MAE", "-"])

    return formatter.format_table(headers, rows, title="Table 4: End-to-End Pipeline Results")


# ============================================================================
# Report Generator
# ============================================================================

def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all evaluation results from directory."""
    results = {
        'classification': {},
        'segmentation': {},
        'regression': {},
        'end_to_end': {}
    }

    # Find all JSON result files
    for json_file in results_dir.glob('*_metrics.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)

        filename = json_file.stem

        if 'classifier' in filename:
            # Try to identify backbone from filename or config
            results['classification']['default'] = data
        elif 'segmentation' in filename:
            results['segmentation'] = data
        elif 'regression' in filename:
            results['regression'] = data
        elif 'end_to_end' in filename:
            results['end_to_end'] = data

    return results


def generate_all_tables(
    results_dir: str,
    output_format: str = 'markdown',
    output_file: Optional[str] = None
) -> str:
    """
    Generate all paper tables.

    Args:
        results_dir: Directory containing evaluation results
        output_format: Output format (markdown, latex, csv)
        output_file: Optional output file path

    Returns:
        Formatted tables string
    """
    results_path = Path(results_dir)

    # Select formatter
    if output_format == 'latex':
        formatter = LaTeXFormatter()
    elif output_format == 'csv':
        formatter = CSVFormatter()
    else:
        formatter = MarkdownFormatter()

    # Load results
    results = load_results(results_path)

    # Generate tables
    tables = []

    tables.append("# Nutrition5k Experiment Results")
    tables.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    tables.append("")

    # Table 1: Classification
    tables.append(generate_table1_classification(
        results.get('classification', {}),
        formatter
    ))

    # Table 2: Segmentation
    tables.append(generate_table2_segmentation(
        results.get('segmentation', {}),
        formatter
    ))

    # Table 3: Regression
    tables.append(generate_table3_regression(
        results.get('regression', {}),
        formatter
    ))

    # Table 4: End-to-End
    tables.append(generate_table4_end_to_end(
        results.get('end_to_end', {}),
        formatter
    ))

    output = "\n".join(tables)

    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output)
        print(f"Tables saved to: {output_file}")

    return output


def generate_placeholder_tables(output_format: str = 'markdown') -> str:
    """Generate placeholder tables when no results are available."""

    # Select formatter
    if output_format == 'latex':
        formatter = LaTeXFormatter()
    elif output_format == 'csv':
        formatter = CSVFormatter()
    else:
        formatter = MarkdownFormatter()

    tables = []

    tables.append("# Nutrition5k Experiment Results")
    tables.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    tables.append("")
    tables.append("*Note: Results pending. Run training experiments to populate tables.*")
    tables.append("")

    # Table 1: Classification (placeholder)
    headers = ["Model", "Backbone", "Top-1 Acc (%)", "Top-5 Acc (%)", "F1 Score", "Params (M)"]
    rows = [
        ["Classifier", "EfficientNet-B0", "-", "-", "-", "5.0"],
        ["Classifier", "EfficientNet-B4", "-", "-", "-", "19.0"],
        ["Classifier", "ViT-B/16", "-", "-", "-", "86.0"],
    ]
    tables.append(formatter.format_table(headers, rows, title="Table 1: Classification Results"))

    # Table 2: Segmentation (placeholder)
    headers = ["Model", "Backbone", "mAP@0.5", "mAP@0.5:0.95", "Mean IoU", "Params (M)"]
    rows = [["Mask R-CNN", "ResNet-50-FPN", "-", "-", "-", "44.4"]]
    tables.append(formatter.format_table(headers, rows, title="Table 2: Segmentation Results"))

    # Table 3: Regression (placeholder)
    headers = ["Output", "MAE", "RMSE", "MAPE (%)", "R²"]
    rows = [
        ["Calories", "-", "-", "-", "-"],
        ["Protein", "-", "-", "-", "-"],
        ["Carb", "-", "-", "-", "-"],
        ["Fat", "-", "-", "-", "-"],
        ["Mass", "-", "-", "-", "-"],
        ["**Overall**", "-", "-", "-", "-"],
    ]
    tables.append(formatter.format_table(headers, rows, title="Table 3: Regression Results"))

    # Table 4: End-to-End (placeholder)
    headers = ["Task", "Metric", "Value"]
    rows = [
        ["Classification", "Top-1 Accuracy (%)", "-"],
        ["Segmentation", "mAP@0.5", "-"],
        ["Regression", "Calories MAE", "-"],
    ]
    tables.append(formatter.format_table(headers, rows, title="Table 4: End-to-End Pipeline Results"))

    return "\n".join(tables)


# ============================================================================
# Main CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate results tables for Nutrition5k paper',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--results-dir', type=str, default='experiments/evaluation',
        help='Directory containing evaluation results'
    )

    parser.add_argument(
        '--format', type=str, default='markdown',
        choices=['markdown', 'latex', 'csv'],
        help='Output format'
    )

    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (prints to stdout if not specified)'
    )

    parser.add_argument(
        '--placeholder', action='store_true',
        help='Generate placeholder tables (no results needed)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.placeholder:
        output = generate_placeholder_tables(args.format)
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Warning: Results directory '{results_dir}' not found.")
            print("Generating placeholder tables instead.")
            output = generate_placeholder_tables(args.format)
        else:
            output = generate_all_tables(
                results_dir=args.results_dir,
                output_format=args.format,
                output_file=args.output
            )

    if not args.output:
        print(output)


if __name__ == '__main__':
    main()
