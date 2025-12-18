"""
Two-Stage Baseline for Calorie Estimation.

This script implements a two-stage approach that:
1. Stage 1: Predict food category (using trained classifier)
2. Stage 2: Predict food mass (using trained regression model)
3. Calculate: calories = predicted_mass × calorie_density[predicted_category]

Key insight:
    卡路里估算的本质是: 卡路里 = 食物重量(g) × 单位热量密度(kcal/g)

Comparison:
    - Category Baseline: Has category info, no weight info (uses mean calories per category)
    - Two-Stage Baseline: Has category info AND weight info (predicts mass, calculates calories)
    - Regression-Only: Implicitly learns both via CNN features
    - End-to-End: Explicit segmentation + classification + regression

Phase 5.5.4 - Supplementary Experiments
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import Nutrition5kDataset
from utils.config_loader import load_config
from models.classifier import FoodClassifier


class MassRegressor(nn.Module):
    """
    Simple mass regressor that uses EfficientNet backbone.
    Predicts food mass in grams from image features.
    """

    def __init__(self, backbone='efficientnet_b0', pretrained=True, dropout=0.3):
        super().__init__()

        # Use EfficientNet backbone
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained)

        # Get feature dimension
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            in_features = 1280  # EfficientNet-B0 default

        # Mass regression head (output = 1 for mass only)
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Mass output only
        )

    def forward(self, x):
        features = self.backbone(x)
        mass = self.regressor(features)
        return mass


def compute_calorie_density_table(dataset: Nutrition5kDataset) -> dict:
    """
    Compute calorie density (kcal/g) for each food category from training set.

    calorie_density[category] = mean(calories) / mean(mass)

    Args:
        dataset: Training dataset

    Returns:
        Dictionary mapping category name to calorie density (kcal/g)
    """
    category_data = defaultdict(lambda: {'calories': [], 'mass': []})

    for item in dataset.metadata:
        category = item["food_category"]
        calories = item["calories"]
        mass = item["mass_g"]  # Field is 'mass_g' in metadata

        # Skip invalid entries
        if mass <= 0 or calories < 0:
            continue

        category_data[category]['calories'].append(calories)
        category_data[category]['mass'].append(mass)

    # Compute calorie density for each category
    calorie_density = {}
    for category, data in category_data.items():
        total_calories = np.sum(data['calories'])
        total_mass = np.sum(data['mass'])

        if total_mass > 0:
            # density = total_calories / total_mass
            calorie_density[category] = total_calories / total_mass
        else:
            calorie_density[category] = 1.0  # Default fallback

    # Compute global average density as fallback
    all_calories = sum(sum(d['calories']) for d in category_data.values())
    all_mass = sum(sum(d['mass']) for d in category_data.values())
    global_density = all_calories / all_mass if all_mass > 0 else 1.0

    return calorie_density, global_density


def evaluate_two_stage_gt_category(
    test_dataset: Nutrition5kDataset,
    mass_predictions: np.ndarray,
    calorie_density: dict,
    global_density: float
) -> dict:
    """
    Evaluate Two-Stage Baseline using GROUND TRUTH category.

    This is an oracle experiment to measure the upper bound when
    category prediction is perfect.

    Args:
        test_dataset: Test dataset
        mass_predictions: Predicted mass values [N,]
        calorie_density: Calorie density table
        global_density: Global average density (fallback)

    Returns:
        Dictionary of evaluation metrics
    """
    predictions = []
    targets = []
    unknown_count = 0

    for i, item in enumerate(test_dataset.metadata):
        category = item["food_category"]  # Ground truth category
        true_calories = item["calories"]
        pred_mass = mass_predictions[i]

        # Look up calorie density
        if category in calorie_density:
            density = calorie_density[category]
        else:
            density = global_density
            unknown_count += 1

        # Calculate predicted calories
        pred_calories = pred_mass * density

        predictions.append(pred_calories)
        targets.append(true_calories)

    predictions = np.array(predictions)
    targets = np.array(targets)

    return compute_metrics(predictions, targets, unknown_count)


def evaluate_two_stage_pred_category(
    test_dataset: Nutrition5kDataset,
    mass_predictions: np.ndarray,
    category_predictions: np.ndarray,
    class_names: list,
    calorie_density: dict,
    global_density: float
) -> dict:
    """
    Evaluate Two-Stage Baseline using PREDICTED category.

    This is the realistic scenario where both category and mass
    are predicted by the models.

    Args:
        test_dataset: Test dataset
        mass_predictions: Predicted mass values [N,]
        category_predictions: Predicted category indices [N,]
        class_names: List of category names
        calorie_density: Calorie density table
        global_density: Global average density (fallback)

    Returns:
        Dictionary of evaluation metrics
    """
    predictions = []
    targets = []
    unknown_count = 0
    category_correct = 0

    for i, item in enumerate(test_dataset.metadata):
        true_category = item["food_category"]
        true_calories = item["calories"]
        pred_mass = mass_predictions[i]
        pred_category_idx = category_predictions[i]

        # Get predicted category name
        if pred_category_idx < len(class_names):
            pred_category = class_names[pred_category_idx]
        else:
            pred_category = None

        # Track category accuracy
        if pred_category == true_category:
            category_correct += 1

        # Look up calorie density
        if pred_category and pred_category in calorie_density:
            density = calorie_density[pred_category]
        else:
            density = global_density
            unknown_count += 1

        # Calculate predicted calories
        pred_calories = pred_mass * density

        predictions.append(pred_calories)
        targets.append(true_calories)

    predictions = np.array(predictions)
    targets = np.array(targets)

    metrics = compute_metrics(predictions, targets, unknown_count)
    metrics['category_accuracy'] = category_correct / len(test_dataset)

    return metrics


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, unknown_count: int) -> dict:
    """Compute calorie estimation metrics."""
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

    return {
        "num_samples": len(targets),
        "num_unknown_categories": int(unknown_count),
        "calories_mae": float(mae),
        "calories_rmse": float(rmse),
        "calories_mape": float(mape),
        "calories_r2": float(r2),
        "mean_prediction": float(np.mean(predictions)),
        "mean_target": float(np.mean(targets)),
    }


def load_trained_classifier(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load trained EfficientNet classifier."""
    print(f"Loading classifier from: {checkpoint_path}")

    model = FoodClassifier(
        backbone='efficientnet_b0',
        num_classes=num_classes,
        pretrained=False,  # Will load trained weights
        freeze_backbone=False,
        dropout=0.3
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def predict_categories(model, dataloader, device):
    """Run classifier inference on dataset."""
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting categories"):
            images = batch['image'].to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())

    return np.array(all_predictions)


def predict_mass(test_dataset, device):
    """
    Predict mass using trained regression model.

    Note: We use the trained regression model's mass predictions.
    The regression model outputs [calories, protein, carb, fat, mass].
    We extract the mass predictions (index 4).
    """
    # Try multiple prediction file locations
    prediction_files = [
        Path("./experiments/evaluation_fixed/regression_test_predictions.json"),
        Path("./experiments/evaluation_improved/regression_test_predictions.json"),
        Path("./experiments/evaluation/regression_test_predictions.json"),
    ]

    for regression_results_file in prediction_files:
        if regression_results_file.exists():
            print(f"Loading regression predictions from: {regression_results_file}")
            with open(regression_results_file) as f:
                reg_data = json.load(f)

            # Extract mass predictions - format is [[cal, prot, carb, fat, mass], ...]
            predictions = reg_data['predictions']
            if len(predictions) > 0:
                if isinstance(predictions[0], list):
                    # Array format [calories, protein, carb, fat, mass]
                    mass_predictions = np.array([p[4] for p in predictions])
                elif isinstance(predictions[0], dict):
                    # Dictionary format {'mass': value, ...}
                    mass_predictions = np.array([p['mass'] for p in predictions])
                else:
                    raise ValueError(f"Unknown prediction format: {type(predictions[0])}")
                return mass_predictions

    # If predictions not available, use ground truth mass for oracle experiment
    print("WARNING: Regression predictions not found, using ground truth mass for oracle")
    mass_predictions = np.array([item['mass_g'] for item in test_dataset.metadata])
    return mass_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Baseline for Calorie Estimation"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nutrition5k",
        help="Path to Nutrition5k dataset root"
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default="./experiments/FoodClassifier_classification/checkpoints/best.pth",
        help="Path to trained classifier checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/phase5.5_two_stage",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("TWO-STAGE BASELINE FOR CALORIE ESTIMATION")
    print("=" * 60)
    print(f"\nDataset root: {args.data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")

    # Load training set to compute calorie density table
    print("\n[1/6] Loading training set...")
    train_dataset = Nutrition5kDataset(
        root_dir=args.data_root,
        split="train",
        mode="regression",
    )
    print(f"  - Training samples: {len(train_dataset)}")

    # Compute calorie density table
    print("\n[2/6] Computing calorie density table...")
    calorie_density, global_density = compute_calorie_density_table(train_dataset)
    print(f"  - Number of categories: {len(calorie_density)}")
    print(f"  - Global average density: {global_density:.4f} kcal/g")

    # Show top 10 categories by density
    sorted_density = sorted(calorie_density.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 categories by calorie density:")
    for cat, density in sorted_density[:10]:
        print(f"    {cat}: {density:.4f} kcal/g")

    # Save calorie density table
    density_file = output_dir / "calorie_density_table.json"
    with open(density_file, 'w') as f:
        json.dump({
            "global_density": global_density,
            "num_categories": len(calorie_density),
            "categories": calorie_density
        }, f, indent=2)
    print(f"\n  - Saved density table to {density_file}")

    # Load test set
    print("\n[3/6] Loading test set...")
    test_dataset = Nutrition5kDataset(
        root_dir=args.data_root,
        split="test",
        mode="regression",
        global_classes=train_dataset.classes,
    )
    print(f"  - Test samples: {len(test_dataset)}")

    # Get mass predictions (from regression model or ground truth)
    print("\n[4/6] Getting mass predictions...")
    mass_predictions = predict_mass(test_dataset, device)

    # Compute ground truth mass statistics
    gt_mass = np.array([item['mass_g'] for item in test_dataset.metadata])
    mass_mae = np.mean(np.abs(mass_predictions - gt_mass))
    print(f"  - Mass prediction MAE: {mass_mae:.2f} g")

    # Evaluate Two-Stage with GT category (oracle)
    print("\n[5/6] Evaluating Two-Stage (GT Category)...")
    results_gt = evaluate_two_stage_gt_category(
        test_dataset,
        mass_predictions,
        calorie_density,
        global_density
    )

    print(f"\n  Two-Stage (GT Category) Results:")
    print(f"    MAE:  {results_gt['calories_mae']:.2f} kcal")
    print(f"    RMSE: {results_gt['calories_rmse']:.2f} kcal")
    print(f"    MAPE: {results_gt['calories_mape']:.2f}%")
    print(f"    R^2:  {results_gt['calories_r2']:.4f}")

    # Load classifier and get category predictions
    print("\n[6/6] Evaluating Two-Stage (Predicted Category)...")

    # Check if classifier checkpoint exists
    classifier_path = Path(args.classifier_checkpoint)
    results_pred = None
    if classifier_path.exists():
        try:
            # Load classifier - check checkpoint for num_classes
            checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
            # Determine num_classes from checkpoint
            if 'config' in checkpoint and 'model' in checkpoint['config']:
                ckpt_num_classes = checkpoint['config']['model'].get('num_classes', 154)
            else:
                # Try to infer from model state dict
                ckpt_num_classes = 154  # Default
                for key in checkpoint.get('model_state_dict', {}).keys():
                    if 'classifier' in key and 'weight' in key:
                        ckpt_num_classes = checkpoint['model_state_dict'][key].shape[0]
                        break

            print(f"  Classifier checkpoint has {ckpt_num_classes} classes")

            # Load classifier with correct num_classes from checkpoint
            classifier = load_trained_classifier(
                str(classifier_path),
                num_classes=ckpt_num_classes,
                device=device
            )

            # Create dataloader for classification
            test_dataset_cls = Nutrition5kDataset(
                root_dir=args.data_root,
                split="test",
                mode="classification",
                global_classes=train_dataset.classes,
            )
            test_loader = DataLoader(
                test_dataset_cls,
                batch_size=32,
                shuffle=False,
                num_workers=0
            )

            # Get category predictions
            category_predictions = predict_categories(classifier, test_loader, device)

            # Evaluate Two-Stage with predicted category
            results_pred = evaluate_two_stage_pred_category(
                test_dataset,
                mass_predictions,
                category_predictions,
                train_dataset.classes,
                calorie_density,
                global_density
            )

            print(f"\n  Two-Stage (Predicted Category) Results:")
            print(f"    Category Accuracy: {results_pred['category_accuracy']:.4f}")
            print(f"    MAE:  {results_pred['calories_mae']:.2f} kcal")
            print(f"    RMSE: {results_pred['calories_rmse']:.2f} kcal")
            print(f"    MAPE: {results_pred['calories_mape']:.2f}%")
            print(f"    R^2:  {results_pred['calories_r2']:.4f}")

        except Exception as e:
            print(f"  WARNING: Failed to load classifier: {e}")
            print("  Skipping predicted category evaluation")
            results_pred = None
    else:
        print(f"  WARNING: Classifier checkpoint not found at {classifier_path}")
        print("  Skipping predicted category evaluation")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    # Load other baseline results
    print(f"\n{'Method':<30} {'MAE (kcal)':<15} {'MAPE (%)':<15} {'R^2':<10}")
    print("-" * 70)

    # Category baseline
    category_baseline_file = Path("./experiments/phase5.5_baseline/baseline_results.json")
    if category_baseline_file.exists():
        with open(category_baseline_file) as f:
            cat_results = json.load(f)
        print(f"{'Category Baseline (GT)':<30} {cat_results['calories_mae']:<15.2f} {cat_results['calories_mape']:<15.2f} {cat_results['calories_r2']:<10.4f}")

    # Two-Stage GT
    print(f"{'Two-Stage (GT Category)':<30} {results_gt['calories_mae']:<15.2f} {results_gt['calories_mape']:<15.2f} {results_gt['calories_r2']:<10.4f}")

    # Two-Stage Predicted
    if results_pred:
        print(f"{'Two-Stage (Pred Category)':<30} {results_pred['calories_mae']:<15.2f} {results_pred['calories_mape']:<15.2f} {results_pred['calories_r2']:<10.4f}")

    # Regression model
    regression_results_file = Path("./experiments/evaluation_fixed/regression_test_metrics.json")
    if regression_results_file.exists():
        with open(regression_results_file) as f:
            reg_results = json.load(f)
        print(f"{'Regression-Only (CNN)':<30} {reg_results['calories_mae']:<15.2f} {reg_results['calories_mape']:<15.2f} {reg_results['calories_r2']:<10.4f}")

    # Save results
    all_results = {
        "two_stage_gt_category": results_gt,
        "two_stage_pred_category": results_pred,
        "mass_mae": float(mass_mae),
    }

    results_file = output_dir / "two_stage_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("""
Key Findings:
1. Two-Stage (GT Category) shows the upper bound when category is known
2. Two-Stage (Pred Category) reflects realistic performance
3. Comparison with Category Baseline shows the value of mass prediction
4. Comparison with Regression-Only shows whether explicit decomposition helps

The Two-Stage approach explicitly models:
  calories = mass × calorie_density(category)

This is interpretable and allows us to analyze:
  - How much does mass prediction error contribute?
  - How much does category error contribute?
""")

    print("\n" + "=" * 60)
    print("TWO-STAGE BASELINE EVALUATION COMPLETE")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
