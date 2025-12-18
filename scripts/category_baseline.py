"""
Category-wise Baseline for Calorie Estimation.

This script implements a simple baseline that predicts calories based on
the mean calories of each food category from the training set.

This serves as a comparison point for the learned regression model.

Phase 5.5.1 - Supplementary Experiments
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import Nutrition5kDataset


def compute_category_statistics(dataset: Nutrition5kDataset) -> dict:
    """
    Compute mean calories per food category from training set.

    Args:
        dataset: Training dataset

    Returns:
        Dictionary mapping category name to mean calories
    """
    category_calories = defaultdict(list)

    for item in dataset.metadata:
        category = item["food_category"]
        calories = item["calories"]
        category_calories[category].append(calories)

    # Compute mean for each category
    category_means = {}
    for category, calories_list in category_calories.items():
        category_means[category] = {
            "mean": np.mean(calories_list),
            "std": np.std(calories_list),
            "count": len(calories_list),
            "min": np.min(calories_list),
            "max": np.max(calories_list),
        }

    return category_means


def evaluate_baseline(test_dataset: Nutrition5kDataset,
                      category_means: dict,
                      global_mean: float) -> dict:
    """
    Evaluate category-wise baseline on test set.

    Uses ground-truth category (oracle) to predict calories.
    For unknown categories, falls back to global mean.

    Args:
        test_dataset: Test dataset
        category_means: Dictionary of mean calories per category
        global_mean: Global mean calories (fallback)

    Returns:
        Dictionary of evaluation metrics
    """
    predictions = []
    targets = []
    categories_used = []
    unknown_count = 0

    for item in test_dataset.metadata:
        category = item["food_category"]
        true_calories = item["calories"]

        if category in category_means:
            pred_calories = category_means[category]["mean"]
        else:
            pred_calories = global_mean
            unknown_count += 1

        predictions.append(pred_calories)
        targets.append(true_calories)
        categories_used.append(category)

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    errors = predictions - targets
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE - handle zero targets
    non_zero_mask = targets > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(abs_errors[non_zero_mask] / targets[non_zero_mask]) * 100
    else:
        mape = float('inf')

    # R^2 score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Per-category analysis
    category_errors = defaultdict(list)
    for i, category in enumerate(categories_used):
        category_errors[category].append(abs_errors[i])

    per_category_mae = {
        cat: np.mean(errs) for cat, errs in category_errors.items()
    }

    return {
        "num_samples": len(targets),
        "num_unknown_categories": unknown_count,
        "calories_mae": float(mae),
        "calories_rmse": float(rmse),
        "calories_mape": float(mape),
        "calories_r2": float(r2),
        "mean_prediction": float(np.mean(predictions)),
        "mean_target": float(np.mean(targets)),
        "per_category_mae": per_category_mae,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Category-wise baseline for calorie estimation"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nutrition5k",
        help="Path to Nutrition5k dataset root"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/phase5.5_baseline",
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CATEGORY-WISE BASELINE FOR CALORIE ESTIMATION")
    print("=" * 60)
    print(f"\nDataset root: {args.data_root}")
    print(f"Output directory: {output_dir}")

    # Load training set to compute category statistics
    print("\n[1/4] Loading training set...")
    train_dataset = Nutrition5kDataset(
        root_dir=args.data_root,
        split="train",
        mode="regression",
    )

    # Compute category statistics
    print("\n[2/4] Computing category statistics...")
    category_means = compute_category_statistics(train_dataset)

    # Compute global mean (fallback for unknown categories)
    all_calories = [item["calories"] for item in train_dataset.metadata]
    global_mean = np.mean(all_calories)

    print(f"  - Number of categories: {len(category_means)}")
    print(f"  - Global mean calories: {global_mean:.2f}")
    print(f"  - Training samples: {len(train_dataset)}")

    # Save category statistics
    stats_file = output_dir / "category_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "global_mean": float(global_mean),
            "global_std": float(np.std(all_calories)),
            "num_categories": len(category_means),
            "categories": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                              for kk, vv in v.items()}
                          for k, v in category_means.items()},
        }, f, indent=2)
    print(f"  - Saved statistics to {stats_file}")

    # Load test set
    print("\n[3/4] Loading test set...")
    # Use same global classes as training for consistency
    test_dataset = Nutrition5kDataset(
        root_dir=args.data_root,
        split="test",
        mode="regression",
        global_classes=train_dataset.classes,
    )

    # Evaluate baseline
    print("\n[4/4] Evaluating baseline on test set...")
    results = evaluate_baseline(test_dataset, category_means, global_mean)

    # Print results
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"\nTest samples: {results['num_samples']}")
    print(f"Unknown categories: {results['num_unknown_categories']}")
    print(f"\nCalorie Estimation Metrics:")
    print(f"  MAE:  {results['calories_mae']:.2f} kcal")
    print(f"  RMSE: {results['calories_rmse']:.2f} kcal")
    print(f"  MAPE: {results['calories_mape']:.2f}%")
    print(f"  R^2:  {results['calories_r2']:.4f}")

    # Compare with regression model results
    print("\n" + "-" * 60)
    print("COMPARISON WITH REGRESSION MODEL")
    print("-" * 60)

    # Load regression results if available
    regression_results_file = Path("./experiments/evaluation_fixed/regression_test_metrics.json")
    if regression_results_file.exists():
        with open(regression_results_file) as f:
            reg_results = json.load(f)

        print(f"\n{'Method':<25} {'MAE (kcal)':<15} {'MAPE (%)':<15} {'R^2':<10}")
        print("-" * 65)
        print(f"{'Category Baseline':<25} {results['calories_mae']:<15.2f} {results['calories_mape']:<15.2f} {results['calories_r2']:<10.4f}")
        print(f"{'Regression Model':<25} {reg_results['calories_mae']:<15.2f} {reg_results['calories_mape']:<15.2f} {reg_results['calories_r2']:<10.4f}")

        # Calculate improvement
        mae_improvement = (results['calories_mae'] - reg_results['calories_mae']) / results['calories_mae'] * 100
        print(f"\nRegression model improves MAE by {mae_improvement:.1f}% over baseline")
    else:
        print("(Regression results not found for comparison)")

    # Save results
    results_file = output_dir / "baseline_results.json"
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {
        k: (float(v) if isinstance(v, (np.floating, float)) else
            ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v))
        for k, v in results.items()
    }
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate report
    report_file = output_dir / "baseline_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CATEGORY-WISE BASELINE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test samples: {results['num_samples']}\n")
        f.write(f"Number of categories: {len(category_means)}\n")
        f.write(f"Unknown categories in test: {results['num_unknown_categories']}\n\n")
        f.write("Metrics:\n")
        f.write(f"  MAE:  {results['calories_mae']:.2f} kcal\n")
        f.write(f"  RMSE: {results['calories_rmse']:.2f} kcal\n")
        f.write(f"  MAPE: {results['calories_mape']:.2f}%\n")
        f.write(f"  R^2:  {results['calories_r2']:.4f}\n\n")
        f.write("Top 10 categories by MAE:\n")
        sorted_categories = sorted(results['per_category_mae'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]
        for cat, mae in sorted_categories:
            f.write(f"  {cat}: {mae:.2f} kcal\n")
    print(f"Report saved to {report_file}")

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
