#!/usr/bin/env python3
"""
End-to-End Model Evaluation Script
Phase 5.5 - Evaluates the trained end-to-end multi-task model on test set.

This script evaluates the End-to-End model (Segmentation + Classification + Regression)
and computes calorie MAE for comparison with other baselines.

Key fix: The model outputs values in normalized space, so we need to denormalize
using statistics computed from the training set.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import get_datasets
from utils.config_loader import load_config
from utils.checkpoint import load_checkpoint
from models.end_to_end import EndToEndFoodRecognition


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate End-to-End Model')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/end_to_end/checkpoints/best.pth',
                        help='Path to checkpoint')
    parser.add_argument('--config', type=str,
                        default='configs/end_to_end.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/phase5.5_end_to_end_eval',
                        help='Output directory')
    return parser.parse_args()


def compute_normalization_stats(train_dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for nutrition targets from training data.

    Returns:
        Tuple of (mean, std) arrays with shape (5,) for [calories, protein, carb, fat, mass]
    """
    print("\nComputing normalization statistics from training data...")

    all_targets = []
    for idx in tqdm(range(len(train_dataset)), desc="Computing stats"):
        sample = train_dataset[idx]
        # Get nutrition values
        nutrition = np.array([
            sample['calories'].item() if hasattr(sample['calories'], 'item') else sample['calories'],
            sample['protein_g'].item() if hasattr(sample['protein_g'], 'item') else sample['protein_g'],
            sample['carb_g'].item() if hasattr(sample['carb_g'], 'item') else sample['carb_g'],
            sample['fat_g'].item() if hasattr(sample['fat_g'], 'item') else sample['fat_g'],
            sample['mass_g'].item() if hasattr(sample['mass_g'], 'item') else sample['mass_g'],
        ])
        all_targets.append(nutrition)

    all_targets = np.array(all_targets)
    mean = np.mean(all_targets, axis=0)
    std = np.std(all_targets, axis=0)

    # Avoid division by zero
    std = np.where(std < 1e-6, 1.0, std)

    print(f"\nNormalization statistics:")
    labels = ['calories', 'protein', 'carb', 'fat', 'mass']
    for i, label in enumerate(labels):
        print(f"  {label}: mean={mean[i]:.2f}, std={std[i]:.2f}")

    return mean, std


def load_model(checkpoint_path: str, config: Dict, device: torch.device) -> nn.Module:
    """Load End-to-End model from checkpoint."""
    print(f"\nLoading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Get model config from checkpoint or use provided config
    ckpt_config = checkpoint.get('config', {})
    model_config = ckpt_config.get('model', config.get('model', {}))

    # Initialize model
    model = EndToEndFoodRecognition(
        num_classes=model_config.get('num_classes', 155),
        classifier_config={
            'backbone': model_config.get('classifier_backbone', 'efficientnet_b0'),
            'pretrained': False
        },
        regressor_config={
            'hidden_dims': model_config.get('hidden_dims', [512, 256, 128]),
            'output_dim': model_config.get('output_dim', 5),
            'dropout': model_config.get('dropout', 0.3)
        },
        use_geometric_features=model_config.get('use_geometric_features', True),
        aggregate_method=model_config.get('aggregate_method', 'sum'),
        min_detection_score=model_config.get('min_detection_score', 0.1)  # Lower threshold for better detection
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Override detection score threshold for evaluation
    model.min_detection_score = 0.05  # Very low threshold to see what's detected
    print(f"Detection score threshold: {model.min_detection_score}")

    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    params = model.get_num_params()
    print(f"Model parameters: {params['total']:,}")

    return model


def evaluate_model(model: nn.Module, test_dataset, device: torch.device,
                   mean: np.ndarray = None, std: np.ndarray = None,
                   use_denorm: bool = False) -> Dict[str, Any]:
    """
    Evaluate End-to-End model on test set.

    Args:
        model: End-to-End model
        test_dataset: Test dataset
        device: Device to use
        mean: Normalization mean (5,) for [calories, protein, carb, fat, mass] - NOT USED in fixed version
        std: Normalization std (5,) for [calories, protein, carb, fat, mass] - NOT USED in fixed version
        use_denorm: Whether to apply denormalization - NOT USED in fixed version

    Returns:
        Dictionary with evaluation metrics

    NOTE: With the fixed training code, model outputs are directly in real calorie space,
    so no denormalization is needed. The mean/std/use_denorm params are kept for backward
    compatibility but are not used.
    """
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    print("NOTE: Fixed model outputs real calorie values directly (no denormalization needed)")

    # Store predictions and targets
    all_cal_predictions = []
    all_cal_targets = []
    all_mass_predictions = []
    all_mass_targets = []
    all_class_predictions = []
    all_class_targets = []

    # Store raw (normalized) predictions for analysis
    all_cal_predictions_raw = []
    all_mass_predictions_raw = []

    num_empty_detections = 0

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset[idx]

            # Get image and move to device
            image = sample['image'].to(device)

            # Get targets
            cal_target = sample['calories'].item()
            mass_target = sample['mass_g'].item()
            class_target = sample['label']

            # Forward pass (model expects list of images)
            results = model([image])
            result = results[0]

            # Extract predictions
            if len(result['instances']) > 0:
                # KEY INSIGHT: Training uses GT boxes to predict TOTAL nutrition for the dish
                # So at inference, we should use the TOP-1 instance (highest confidence)
                # rather than aggregating multiple instances
                top_inst = result['instances'][0]  # Already sorted by confidence

                cal_pred_raw = top_inst['calories']
                mass_pred_raw = top_inst.get('mass_g', 0.0)

                if use_denorm:
                    # Denormalize the single instance prediction
                    cal_pred = cal_pred_raw * std[0] + mean[0]
                    mass_pred = mass_pred_raw * std[4] + mean[4]
                else:
                    cal_pred = cal_pred_raw
                    mass_pred = mass_pred_raw

                # For classification, use the first instance's category
                class_pred = top_inst['category']
            else:
                # No detection - use mean as fallback
                cal_pred_raw = 0.0
                mass_pred_raw = 0.0
                if use_denorm:
                    cal_pred = mean[0]  # Use mean for empty detections
                    mass_pred = mean[4]
                else:
                    cal_pred = 0.0
                    mass_pred = 0.0
                class_pred = -1
                num_empty_detections += 1

            all_cal_predictions_raw.append(cal_pred_raw)
            all_mass_predictions_raw.append(mass_pred_raw)
            all_cal_predictions.append(cal_pred)
            all_cal_targets.append(cal_target)
            all_mass_predictions.append(mass_pred)
            all_mass_targets.append(mass_target)
            all_class_predictions.append(class_pred)
            all_class_targets.append(class_target)

    # Convert to numpy
    cal_preds = np.array(all_cal_predictions)
    cal_targets = np.array(all_cal_targets)
    mass_preds = np.array(all_mass_predictions)
    mass_targets = np.array(all_mass_targets)
    class_preds = np.array(all_class_predictions)
    class_targets = np.array(all_class_targets)

    # Compute metrics
    # Calorie metrics
    cal_mae = np.mean(np.abs(cal_preds - cal_targets))
    cal_rmse = np.sqrt(np.mean((cal_preds - cal_targets) ** 2))

    # MAPE (avoid division by zero)
    nonzero_mask = cal_targets > 1e-6
    if nonzero_mask.sum() > 0:
        cal_mape = np.mean(np.abs((cal_preds[nonzero_mask] - cal_targets[nonzero_mask]) / cal_targets[nonzero_mask])) * 100
    else:
        cal_mape = 0.0

    # R2 score
    ss_res = np.sum((cal_targets - cal_preds) ** 2)
    ss_tot = np.sum((cal_targets - np.mean(cal_targets)) ** 2)
    cal_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Mass metrics
    mass_mae = np.mean(np.abs(mass_preds - mass_targets))

    # Classification accuracy (excluding empty detections)
    valid_mask = class_preds >= 0
    if valid_mask.sum() > 0:
        class_acc = np.mean(class_preds[valid_mask] == class_targets[valid_mask]) * 100
    else:
        class_acc = 0.0

    metrics = {
        'num_samples': len(test_dataset),
        'num_empty_detections': num_empty_detections,
        'empty_detection_rate': num_empty_detections / len(test_dataset) * 100,
        'calories_mae': float(cal_mae),
        'calories_rmse': float(cal_rmse),
        'calories_mape': float(cal_mape),
        'calories_r2': float(cal_r2),
        'mass_mae': float(mass_mae),
        'classification_accuracy': float(class_acc),
        'mean_prediction': float(np.mean(cal_preds)),
        'mean_target': float(np.mean(cal_targets))
    }

    return metrics


def main():
    args = parse_args()

    print("=" * 80)
    print("END-TO-END MODEL EVALUATION")
    print("=" * 80)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load config
    config = load_config(args.config)

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load datasets (need training set for normalization stats)
    print("\nLoading datasets...")
    dataset_root = config.get('data', {}).get('dataset_root', 'data/nutrition5k')
    train_dataset, _, test_dataset = get_datasets(
        root=dataset_root,
        task='regression'  # Use regression mode to get calories/mass targets
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Compute normalization statistics from training data
    # This is needed because the model outputs values in normalized space
    mean, std = compute_normalization_stats(train_dataset)

    # First, evaluate WITHOUT denormalization to see raw model output behavior
    print("\n" + "-" * 40)
    print("Evaluation WITHOUT denormalization (raw outputs)")
    print("-" * 40)
    metrics_raw = evaluate_model(model, test_dataset, device, mean=mean, std=std, use_denorm=False)
    print(f"  Raw Mean Prediction: {metrics_raw['mean_prediction']:.2f}")
    print(f"  Raw Mean Target: {metrics_raw['mean_target']:.2f}")
    print(f"  Raw MAE: {metrics_raw['calories_mae']:.2f}")

    # Then evaluate WITH denormalization
    print("\n" + "-" * 40)
    print("Evaluation WITH denormalization")
    print("-" * 40)
    metrics = evaluate_model(model, test_dataset, device, mean=mean, std=std, use_denorm=True)

    # Store both results
    metrics['raw_metrics'] = {
        'mean_prediction': metrics_raw['mean_prediction'],
        'calories_mae': metrics_raw['calories_mae']
    }

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTest Samples: {metrics['num_samples']}")
    print(f"Empty Detections: {metrics['num_empty_detections']} ({metrics['empty_detection_rate']:.2f}%)")
    print(f"\nCalorie Estimation:")
    print(f"  MAE:  {metrics['calories_mae']:.2f} kcal")
    print(f"  RMSE: {metrics['calories_rmse']:.2f} kcal")
    print(f"  MAPE: {metrics['calories_mape']:.2f}%")
    print(f"  R2:   {metrics['calories_r2']:.4f}")
    print(f"\nMass Estimation:")
    print(f"  MAE:  {metrics['mass_mae']:.2f} g")
    print(f"\nClassification Accuracy: {metrics['classification_accuracy']:.2f}%")
    print(f"\nMean Prediction: {metrics['mean_prediction']:.2f} kcal")
    print(f"Mean Target:     {metrics['mean_target']:.2f} kcal")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add normalization stats to metrics for reference
    metrics['normalization'] = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'labels': ['calories', 'protein', 'carb', 'fat', 'mass']
    }

    results_file = output_dir / 'end_to_end_test_metrics.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
