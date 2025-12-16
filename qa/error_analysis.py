#!/usr/bin/env python3
"""
Error Analysis Module for Nutrition5k Project
QA-Agent | Phase 5

Provides comprehensive error analysis for all model types:
- Classification: per-class errors, confusion analysis, misclassification patterns
- Regression: per-calorie-range errors, outlier detection, error distribution
- Segmentation: detection failures, false positives/negatives
- End-to-End: combined analysis across all tasks

Usage:
    python qa/error_analysis.py --model classifier --predictions experiments/evaluation/classifier_test_predictions.json
    python qa/error_analysis.py --model regression --predictions experiments/evaluation/regression_test_predictions.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict

import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not available. Visualizations disabled.")


# ============================================================================
# Classification Error Analysis
# ============================================================================

class ClassificationErrorAnalyzer:
    """Analyze classification errors and misclassification patterns."""

    def __init__(self, predictions: List[int], targets: List[int],
                 class_names: Optional[List[str]] = None):
        """
        Initialize analyzer.

        Args:
            predictions: List of predicted class indices
            targets: List of ground truth class indices
            class_names: Optional list of class names
        """
        self.predictions = np.array(predictions)
        self.targets = np.array(targets)
        self.num_classes = max(max(predictions), max(targets)) + 1
        self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]

    def get_per_class_accuracy(self) -> Dict[str, float]:
        """Compute accuracy for each class."""
        per_class_acc = {}

        for cls_idx in range(self.num_classes):
            mask = self.targets == cls_idx
            if mask.sum() == 0:
                continue
            correct = (self.predictions[mask] == cls_idx).sum()
            total = mask.sum()
            per_class_acc[self.class_names[cls_idx]] = float(correct / total) * 100

        return per_class_acc

    def get_worst_classes(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get classes with lowest accuracy."""
        per_class = self.get_per_class_accuracy()
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
        return sorted_classes[:n]

    def get_best_classes(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get classes with highest accuracy."""
        per_class = self.get_per_class_accuracy()
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        return sorted_classes[:n]

    def get_confusion_pairs(self, n: int = 10) -> List[Tuple[str, str, int]]:
        """Get most common confusion pairs (predicted, actual, count)."""
        confusion_counts = Counter()

        for pred, target in zip(self.predictions, self.targets):
            if pred != target:
                confusion_counts[(self.class_names[pred], self.class_names[target])] += 1

        return confusion_counts.most_common(n)

    def get_misclassification_analysis(self) -> Dict[str, Any]:
        """Get comprehensive misclassification analysis."""
        incorrect_mask = self.predictions != self.targets

        analysis = {
            'total_samples': len(self.predictions),
            'correct': int((~incorrect_mask).sum()),
            'incorrect': int(incorrect_mask.sum()),
            'accuracy': float((~incorrect_mask).mean() * 100),
            'worst_classes': self.get_worst_classes(10),
            'best_classes': self.get_best_classes(10),
            'top_confusion_pairs': self.get_confusion_pairs(10)
        }

        return analysis

    def plot_confusion_matrix(self, save_path: Optional[str] = None, top_n: int = 20):
        """Plot confusion matrix for top N classes."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping confusion matrix plot.")
            return

        from sklearn.metrics import confusion_matrix

        # Get top N classes by frequency
        class_counts = Counter(self.targets)
        top_classes = [cls for cls, _ in class_counts.most_common(top_n)]

        # Filter to only include top classes
        mask = np.isin(self.targets, top_classes)
        filtered_targets = self.targets[mask]
        filtered_preds = self.predictions[mask]

        # Compute confusion matrix
        cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=False, cmap='Blues',
                   xticklabels=[self.class_names[i][:10] for i in top_classes],
                   yticklabels=[self.class_names[i][:10] for i in top_classes])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Top {top_n} Classes)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()
        plt.close()


# ============================================================================
# Regression Error Analysis
# ============================================================================

class RegressionErrorAnalyzer:
    """Analyze regression errors by calorie range and identify outliers."""

    def __init__(self, predictions: np.ndarray, targets: np.ndarray,
                 output_names: List[str] = None):
        """
        Initialize analyzer.

        Args:
            predictions: [N, D] array of predictions
            targets: [N, D] array of ground truth
            output_names: Names of output variables
        """
        self.predictions = np.array(predictions)
        self.targets = np.array(targets)

        if self.predictions.ndim == 1:
            self.predictions = self.predictions.reshape(-1, 1)
            self.targets = self.targets.reshape(-1, 1)

        self.output_names = output_names or ['calories', 'protein', 'carb', 'fat', 'mass']

    def get_errors(self) -> np.ndarray:
        """Get absolute errors."""
        return np.abs(self.predictions - self.targets)

    def get_percentage_errors(self) -> np.ndarray:
        """Get absolute percentage errors."""
        return np.abs((self.predictions - self.targets) / (self.targets + 1e-8)) * 100

    def get_per_range_errors(self, output_idx: int = 0,
                            ranges: List[Tuple[float, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze errors by target value range.

        Args:
            output_idx: Index of output to analyze (default: calories)
            ranges: List of (min, max) tuples for ranges

        Returns:
            Dict mapping range to error statistics
        """
        if ranges is None:
            # Default calorie ranges
            ranges = [
                (0, 200), (200, 400), (400, 600),
                (600, 800), (800, 1000), (1000, float('inf'))
            ]

        targets = self.targets[:, output_idx]
        errors = self.get_errors()[:, output_idx]
        pct_errors = self.get_percentage_errors()[:, output_idx]

        range_errors = {}

        for min_val, max_val in ranges:
            mask = (targets >= min_val) & (targets < max_val)
            if mask.sum() == 0:
                continue

            range_name = f"{int(min_val)}-{int(max_val) if max_val != float('inf') else 'inf'}"
            range_errors[range_name] = {
                'count': int(mask.sum()),
                'mae': float(errors[mask].mean()),
                'std': float(errors[mask].std()),
                'mape': float(pct_errors[mask].mean()),
                'max_error': float(errors[mask].max()),
                'median_error': float(np.median(errors[mask]))
            }

        return range_errors

    def get_outliers(self, threshold_percentile: float = 95) -> Dict[str, Any]:
        """
        Identify outlier predictions (errors above threshold percentile).

        Args:
            threshold_percentile: Percentile for outlier threshold

        Returns:
            Dict with outlier statistics per output
        """
        errors = self.get_errors()
        outliers = {}

        for i, name in enumerate(self.output_names):
            if i >= errors.shape[1]:
                break

            threshold = np.percentile(errors[:, i], threshold_percentile)
            outlier_mask = errors[:, i] > threshold

            outliers[name] = {
                'threshold': float(threshold),
                'count': int(outlier_mask.sum()),
                'percentage': float(outlier_mask.mean() * 100),
                'max_error': float(errors[:, i].max()),
                'outlier_indices': np.where(outlier_mask)[0].tolist()[:20]  # First 20
            }

        return outliers

    def get_overall_analysis(self) -> Dict[str, Any]:
        """Get comprehensive regression error analysis."""
        errors = self.get_errors()

        analysis = {
            'total_samples': len(self.predictions),
            'per_output': {}
        }

        for i, name in enumerate(self.output_names):
            if i >= errors.shape[1]:
                break

            analysis['per_output'][name] = {
                'mae': float(errors[:, i].mean()),
                'rmse': float(np.sqrt((errors[:, i] ** 2).mean())),
                'mape': float(self.get_percentage_errors()[:, i].mean()),
                'std': float(errors[:, i].std()),
                'min_error': float(errors[:, i].min()),
                'max_error': float(errors[:, i].max()),
                'median_error': float(np.median(errors[:, i]))
            }

        # Add calorie-range analysis
        if 'calories' in self.output_names:
            cal_idx = self.output_names.index('calories')
            analysis['calorie_range_errors'] = self.get_per_range_errors(cal_idx)

        # Add outlier analysis
        analysis['outliers'] = self.get_outliers()

        return analysis

    def plot_error_distribution(self, save_path: Optional[str] = None):
        """Plot error distribution for each output."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping error distribution plot.")
            return

        errors = self.get_errors()
        n_outputs = min(len(self.output_names), errors.shape[1])

        fig, axes = plt.subplots(1, n_outputs, figsize=(4*n_outputs, 4))
        if n_outputs == 1:
            axes = [axes]

        for i, (ax, name) in enumerate(zip(axes, self.output_names[:n_outputs])):
            ax.hist(errors[:, i], bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(errors[:, i].mean(), color='red', linestyle='--',
                      label=f'Mean: {errors[:, i].mean():.2f}')
            ax.axvline(np.median(errors[:, i]), color='green', linestyle='--',
                      label=f'Median: {np.median(errors[:, i]):.2f}')
            ax.set_xlabel(f'{name} Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Error Distribution')
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Error distribution saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_prediction_vs_actual(self, output_idx: int = 0, save_path: Optional[str] = None):
        """Plot prediction vs actual scatter plot."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping scatter plot.")
            return

        preds = self.predictions[:, output_idx]
        targets = self.targets[:, output_idx]
        name = self.output_names[output_idx]

        plt.figure(figsize=(8, 8))
        plt.scatter(targets, preds, alpha=0.5, s=10)

        # Perfect prediction line
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.xlabel(f'Actual {name}')
        plt.ylabel(f'Predicted {name}')
        plt.title(f'{name}: Prediction vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Scatter plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()


# ============================================================================
# Combined Error Analysis Report
# ============================================================================

def generate_error_report(
    model_type: str,
    predictions_path: str,
    output_dir: str,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive error analysis report.

    Args:
        model_type: Type of model (classifier, regression, end_to_end)
        predictions_path: Path to predictions JSON file
        output_dir: Directory to save analysis results
        class_names: Optional class names for classification

    Returns:
        Analysis results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    with open(predictions_path, 'r') as f:
        data = json.load(f)

    results = {'model_type': model_type}

    if model_type == 'classifier':
        analyzer = ClassificationErrorAnalyzer(
            predictions=data['predictions'],
            targets=data['targets'],
            class_names=class_names
        )

        results['classification'] = analyzer.get_misclassification_analysis()

        # Generate plots
        if HAS_MATPLOTLIB:
            analyzer.plot_confusion_matrix(
                save_path=str(output_dir / 'confusion_matrix.png')
            )

    elif model_type == 'regression':
        analyzer = RegressionErrorAnalyzer(
            predictions=np.array(data['predictions']),
            targets=np.array(data['targets'])
        )

        results['regression'] = analyzer.get_overall_analysis()

        # Generate plots
        if HAS_MATPLOTLIB:
            analyzer.plot_error_distribution(
                save_path=str(output_dir / 'error_distribution.png')
            )
            analyzer.plot_prediction_vs_actual(
                output_idx=0,
                save_path=str(output_dir / 'calories_scatter.png')
            )

    elif model_type == 'end_to_end':
        # Classification analysis
        if 'classification' in data:
            cls_analyzer = ClassificationErrorAnalyzer(
                predictions=data['classification']['predictions'],
                targets=data['classification']['targets'],
                class_names=class_names
            )
            results['classification'] = cls_analyzer.get_misclassification_analysis()

            if HAS_MATPLOTLIB:
                cls_analyzer.plot_confusion_matrix(
                    save_path=str(output_dir / 'e2e_confusion_matrix.png')
                )

        # Regression analysis
        if 'regression' in data:
            reg_analyzer = RegressionErrorAnalyzer(
                predictions=np.array(data['regression']['predictions']),
                targets=np.array(data['regression']['targets'])
            )
            results['regression'] = reg_analyzer.get_overall_analysis()

            if HAS_MATPLOTLIB:
                reg_analyzer.plot_error_distribution(
                    save_path=str(output_dir / 'e2e_error_distribution.png')
                )

    # Save results
    results_path = output_dir / f'{model_type}_error_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Error analysis saved to: {results_path}")

    return results


def print_error_report(results: Dict[str, Any]) -> None:
    """Print formatted error analysis report."""
    print("\n" + "="*80)
    print("ERROR ANALYSIS REPORT")
    print("="*80)

    model_type = results.get('model_type', 'unknown')
    print(f"\nModel Type: {model_type.upper()}")

    if 'classification' in results:
        cls = results['classification']
        print("\n" + "-"*40)
        print("CLASSIFICATION ERROR ANALYSIS")
        print("-"*40)
        print(f"Total Samples: {cls['total_samples']}")
        print(f"Correct: {cls['correct']} | Incorrect: {cls['incorrect']}")
        print(f"Accuracy: {cls['accuracy']:.2f}%")

        print("\nWorst Performing Classes:")
        for name, acc in cls['worst_classes'][:5]:
            print(f"  {name}: {acc:.2f}%")

        print("\nTop Confusion Pairs (Predicted → Actual):")
        for (pred, actual), count in cls['top_confusion_pairs'][:5]:
            print(f"  {pred} → {actual}: {count} times")

    if 'regression' in results:
        reg = results['regression']
        print("\n" + "-"*40)
        print("REGRESSION ERROR ANALYSIS")
        print("-"*40)
        print(f"Total Samples: {reg['total_samples']}")

        print("\nPer-Output Metrics:")
        for name, metrics in reg.get('per_output', {}).items():
            print(f"\n  {name.upper()}:")
            print(f"    MAE: {metrics['mae']:.4f}")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    MAPE: {metrics['mape']:.2f}%")
            print(f"    Max Error: {metrics['max_error']:.4f}")

        if 'calorie_range_errors' in reg:
            print("\nCalorie Range Analysis:")
            for range_name, metrics in reg['calorie_range_errors'].items():
                print(f"  {range_name} cal: n={metrics['count']}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")

        if 'outliers' in reg:
            print("\nOutlier Analysis (>95th percentile):")
            for name, info in reg['outliers'].items():
                print(f"  {name}: {info['count']} outliers ({info['percentage']:.1f}%), threshold={info['threshold']:.2f}")

    print("\n" + "="*80)


# ============================================================================
# Main CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Error Analysis for Nutrition5k Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model', type=str, required=True,
        choices=['classifier', 'regression', 'segmentation', 'end_to_end'],
        help='Model type to analyze'
    )

    parser.add_argument(
        '--predictions', type=str, required=True,
        help='Path to predictions JSON file'
    )

    parser.add_argument(
        '--output-dir', type=str, default='qa/analysis',
        help='Directory to save analysis results'
    )

    parser.add_argument(
        '--class-names', type=str, default=None,
        help='Path to JSON file with class names'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)

    # Generate report
    results = generate_error_report(
        model_type=args.model,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        class_names=class_names
    )

    # Print report
    print_error_report(results)


if __name__ == '__main__':
    main()
