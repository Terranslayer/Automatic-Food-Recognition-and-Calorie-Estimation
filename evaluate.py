#!/usr/bin/env python3
"""
Unified Evaluation Script for Nutrition5k Project
Training-Agent | Phase 4

Evaluates trained models on test/validation sets and generates comprehensive reports.

Usage:
    python evaluate.py --model classifier --checkpoint checkpoints/best.pth --split test
    python evaluate.py --model segmentation --checkpoint checkpoints/best.pth --config configs/mask_rcnn.yaml
    python evaluate.py --model regression --checkpoint checkpoints/best.pth
    python evaluate.py --model end_to_end --checkpoint checkpoints/best.pth --save-predictions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import project modules
from utils.config_loader import load_config
from utils.dataset import Nutrition5kDataset, get_datasets
from utils.metrics import (
    accuracy,
    mae,
    rmse,
    mape,
    r2_score,
    MetricAggregator
)
from utils.checkpoint import load_checkpoint
from models.classifier import FoodClassifier
from models.segmentation import FoodSegmentation
from models.calorie_regressor import CalorieRegressor
from models.end_to_end import EndToEndFoodRecognition


def collate_fn(batch):
    """Custom collate function for segmentation/detection datasets."""
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on Nutrition5k dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['classifier', 'segmentation', 'regression', 'end_to_end'],
        help='Model type to evaluate'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to JSON file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/evaluation',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save sample visualizations'
    )

    parser.add_argument(
        '--num-visualize',
        type=int,
        default=20,
        help='Number of samples to visualize'
    )

    return parser.parse_args()


def load_model_from_checkpoint(
    model_type: str,
    checkpoint_path: str,
    device: torch.device,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Load model from checkpoint.

    Args:
        model_type: Type of model (classifier, segmentation, regression, end_to_end)
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        config: Configuration dictionary

    Returns:
        Loaded model in eval mode
    """
    print(f"\nLoading {model_type} model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Initialize model based on type
    if model_type == 'classifier':
        model = FoodClassifier(
            num_classes=config.get('model', {}).get('num_classes', 132),
            backbone=config.get('model', {}).get('backbone', 'efficientnet_b0'),
            pretrained=False
        )
    elif model_type == 'segmentation':
        model = FoodSegmentation(
            num_classes=config.get('model', {}).get('num_classes', 132),
            pretrained=False
        )
    elif model_type == 'regression':
        model = CalorieRegressor(
            backbone=config.get('model', {}).get('backbone', 'efficientnet_b0'),
            pretrained=False
        )
    elif model_type == 'end_to_end':
        model = EndToEndFoodRecognition(
            num_classes=config.get('model', {}).get('num_classes', 132),
            classifier_backbone=config.get('model', {}).get('classifier_backbone', 'efficientnet_b0'),
            regression_backbone=config.get('model', {}).get('regression_backbone', 'efficientnet_b0')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best metrics: {checkpoint.get('best_metrics', {})}")

    return model


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate classification model.

    Args:
        model: Classification model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating classifier...")

    all_predictions = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            targets = batch['dish_id'].to(device)

            # Forward pass
            outputs = model(images)

            # Store results
            all_logits.append(outputs.cpu())
            all_predictions.append(outputs.argmax(dim=1).cpu())
            all_targets.append(targets.cpu())

    # Concatenate all results
    all_logits = torch.cat(all_logits, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    top1_acc = accuracy(all_logits, all_targets, topk=1)
    top5_acc = accuracy(all_logits, all_targets, topk=5)

    # Compute per-class metrics
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets.numpy(),
        all_predictions.numpy(),
        average='macro',
        zero_division=0
    )

    conf_matrix = confusion_matrix(all_targets.numpy(), all_predictions.numpy())

    metrics = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_samples': len(all_targets),
        'confusion_matrix': conf_matrix.tolist()
    }

    return metrics, {
        'predictions': all_predictions.numpy().tolist(),
        'targets': all_targets.numpy().tolist(),
        'logits': all_logits.numpy().tolist()
    }


def evaluate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate segmentation model.

    Note: Full mAP computation requires complex IoU matching. This provides simplified metrics.

    Args:
        model: Segmentation model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating segmentation model...")
    warnings.warn(
        "Simplified segmentation metrics. Full mAP computation would require "
        "proper IoU-based matching of predicted and ground-truth boxes/masks."
    )

    all_predictions = []
    total_detections = 0
    total_gt_boxes = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)

            # Forward pass
            outputs = model(images)

            # Count detections
            for output in outputs:
                num_detections = len(output['boxes'])
                total_detections += num_detections
                all_predictions.append({
                    'num_boxes': num_detections,
                    'scores': output['scores'].cpu().numpy().tolist() if num_detections > 0 else [],
                    'labels': output['labels'].cpu().numpy().tolist() if num_detections > 0 else []
                })

            # Count ground truth boxes (if available)
            if 'boxes' in batch:
                for boxes in batch['boxes']:
                    total_gt_boxes += len(boxes)

    # Compute simplified metrics
    avg_detections_per_image = total_detections / len(dataloader.dataset)

    metrics = {
        'total_detections': total_detections,
        'total_gt_boxes': total_gt_boxes,
        'avg_detections_per_image': float(avg_detections_per_image),
        'num_samples': len(dataloader.dataset),
        'note': 'Simplified metrics. Full mAP computation requires IoU-based matching.'
    }

    return metrics, {'predictions': all_predictions}


def evaluate_regression(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate regression model.

    Args:
        model: Regression model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating regression model...")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)

            # Get targets (calories, protein, carb, fat, mass)
            targets = torch.stack([
                batch['total_calories'],
                batch['total_protein'],
                batch['total_carb'],
                batch['total_fat'],
                batch['total_mass']
            ], dim=1).to(device)

            # Forward pass
            outputs = model(images)

            # Store results
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics for each output
    output_names = ['calories', 'protein', 'carb', 'fat', 'mass']
    metrics = {'num_samples': len(dataloader.dataset)}

    for i, name in enumerate(output_names):
        preds = all_predictions[:, i]
        targets = all_targets[:, i]

        metrics[f'{name}_mae'] = float(mae(preds, targets))
        metrics[f'{name}_rmse'] = float(rmse(preds, targets))
        metrics[f'{name}_mape'] = float(mape(preds, targets))
        metrics[f'{name}_r2'] = float(r2_score(preds, targets))

    # Compute overall metrics (averaged across outputs)
    metrics['overall_mae'] = np.mean([metrics[f'{name}_mae'] for name in output_names])
    metrics['overall_rmse'] = np.mean([metrics[f'{name}_rmse'] for name in output_names])
    metrics['overall_mape'] = np.mean([metrics[f'{name}_mape'] for name in output_names])
    metrics['overall_r2'] = np.mean([metrics[f'{name}_r2'] for name in output_names])

    return metrics, {
        'predictions': all_predictions.numpy().tolist(),
        'targets': all_targets.numpy().tolist()
    }


def evaluate_end_to_end(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate end-to-end model (classification + segmentation + regression).

    Args:
        model: End-to-end model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating end-to-end model...")

    # Classification outputs
    all_class_logits = []
    all_class_targets = []

    # Segmentation outputs
    all_seg_predictions = []

    # Regression outputs
    all_reg_predictions = []
    all_reg_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            class_targets = batch['dish_id'].to(device)

            # Get regression targets
            reg_targets = torch.stack([
                batch['total_calories'],
                batch['total_protein'],
                batch['total_carb'],
                batch['total_fat'],
                batch['total_mass']
            ], dim=1).to(device)

            # Forward pass
            outputs = model(images)

            # Store classification results
            all_class_logits.append(outputs['classification'].cpu())
            all_class_targets.append(class_targets.cpu())

            # Store segmentation results
            for seg_output in outputs['segmentation']:
                all_seg_predictions.append({
                    'num_boxes': len(seg_output['boxes']),
                    'scores': seg_output['scores'].cpu().numpy().tolist() if len(seg_output['boxes']) > 0 else [],
                    'labels': seg_output['labels'].cpu().numpy().tolist() if len(seg_output['boxes']) > 0 else []
                })

            # Store regression results
            all_reg_predictions.append(outputs['regression'].cpu())
            all_reg_targets.append(reg_targets.cpu())

    # Concatenate results
    all_class_logits = torch.cat(all_class_logits, dim=0)
    all_class_targets = torch.cat(all_class_targets, dim=0)
    all_reg_predictions = torch.cat(all_reg_predictions, dim=0)
    all_reg_targets = torch.cat(all_reg_targets, dim=0)

    # Compute classification metrics
    top1_acc = accuracy(all_class_logits, all_class_targets, topk=1)
    top5_acc = accuracy(all_class_logits, all_class_targets, topk=5)

    from sklearn.metrics import precision_recall_fscore_support
    predictions = all_class_logits.argmax(dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_class_targets.numpy(),
        predictions.numpy(),
        average='macro',
        zero_division=0
    )

    # Compute regression metrics
    output_names = ['calories', 'protein', 'carb', 'fat', 'mass']
    regression_metrics = {}

    for i, name in enumerate(output_names):
        preds = all_reg_predictions[:, i]
        targets = all_reg_targets[:, i]

        regression_metrics[f'{name}_mae'] = float(mae(preds, targets))
        regression_metrics[f'{name}_rmse'] = float(rmse(preds, targets))
        regression_metrics[f'{name}_mape'] = float(mape(preds, targets))
        regression_metrics[f'{name}_r2'] = float(r2_score(preds, targets))

    # Segmentation metrics (simplified)
    total_detections = sum(pred['num_boxes'] for pred in all_seg_predictions)
    avg_detections = total_detections / len(all_seg_predictions)

    metrics = {
        'num_samples': len(dataloader.dataset),
        # Classification
        'classification': {
            'top1_accuracy': float(top1_acc),
            'top5_accuracy': float(top5_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        # Segmentation
        'segmentation': {
            'total_detections': total_detections,
            'avg_detections_per_image': float(avg_detections),
            'note': 'Simplified metrics'
        },
        # Regression
        'regression': regression_metrics
    }

    predictions_data = {
        'classification': {
            'predictions': predictions.numpy().tolist(),
            'targets': all_class_targets.numpy().tolist()
        },
        'segmentation': all_seg_predictions,
        'regression': {
            'predictions': all_reg_predictions.numpy().tolist(),
            'targets': all_reg_targets.numpy().tolist()
        }
    }

    return metrics, predictions_data


def print_evaluation_report(metrics: Dict[str, Any], model_type: str) -> None:
    """
    Print formatted evaluation report.

    Args:
        metrics: Dictionary of evaluation metrics
        model_type: Type of model evaluated
    """
    print("\n" + "="*80)
    print(f"EVALUATION REPORT - {model_type.upper()} MODEL")
    print("="*80)
    print(f"\nNumber of samples: {metrics['num_samples']}")
    print()

    if model_type == 'classifier':
        print("Classification Metrics:")
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  F1 Score:       {metrics['f1_score']:.4f}")

    elif model_type == 'segmentation':
        print("Segmentation Metrics:")
        print(f"  Total Detections:           {metrics['total_detections']}")
        print(f"  Avg Detections per Image:   {metrics['avg_detections_per_image']:.2f}")
        if 'total_gt_boxes' in metrics:
            print(f"  Total GT Boxes:             {metrics['total_gt_boxes']}")
        print(f"\n  Note: {metrics['note']}")

    elif model_type == 'regression':
        print("Regression Metrics:")
        print("\n  Per-Output Metrics:")
        output_names = ['calories', 'protein', 'carb', 'fat', 'mass']
        for name in output_names:
            print(f"\n  {name.upper()}:")
            print(f"    MAE:  {metrics[f'{name}_mae']:.4f}")
            print(f"    RMSE: {metrics[f'{name}_rmse']:.4f}")
            print(f"    MAPE: {metrics[f'{name}_mape']:.2f}%")
            print(f"    R²:   {metrics[f'{name}_r2']:.4f}")

        print("\n  Overall (averaged):")
        print(f"    MAE:  {metrics['overall_mae']:.4f}")
        print(f"    RMSE: {metrics['overall_rmse']:.4f}")
        print(f"    MAPE: {metrics['overall_mape']:.2f}%")
        print(f"    R²:   {metrics['overall_r2']:.4f}")

    elif model_type == 'end_to_end':
        print("End-to-End Model Metrics:")

        print("\nClassification:")
        print(f"  Top-1 Accuracy: {metrics['classification']['top1_accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {metrics['classification']['top5_accuracy']:.2f}%")
        print(f"  F1 Score:       {metrics['classification']['f1_score']:.4f}")

        print("\nSegmentation:")
        print(f"  Total Detections:         {metrics['segmentation']['total_detections']}")
        print(f"  Avg Detections per Image: {metrics['segmentation']['avg_detections_per_image']:.2f}")

        print("\nRegression (Key Metrics):")
        print(f"  Calories MAE:  {metrics['regression']['calories_mae']:.4f}")
        print(f"  Calories MAPE: {metrics['regression']['calories_mape']:.2f}%")
        print(f"  Protein MAE:   {metrics['regression']['protein_mae']:.4f}")
        print(f"  Mass MAE:      {metrics['regression']['mass_mae']:.4f}")

    print("\n" + "="*80)


def save_results(
    metrics: Dict[str, Any],
    predictions: Optional[Dict[str, Any]],
    output_dir: Path,
    model_type: str,
    split: str,
    save_predictions: bool
) -> None:
    """
    Save evaluation results to files.

    Args:
        metrics: Evaluation metrics
        predictions: Model predictions
        output_dir: Directory to save results
        model_type: Type of model
        split: Dataset split
        save_predictions: Whether to save predictions
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_dir / f'{model_type}_{split}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save predictions if requested
    if save_predictions and predictions is not None:
        predictions_file = output_dir / f'{model_type}_{split}_predictions.json'
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to: {predictions_file}")

    # Save text report
    report_file = output_dir / f'{model_type}_{split}_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"EVALUATION REPORT - {model_type.upper()} MODEL\n")
        f.write("="*80 + "\n\n")
        f.write(json.dumps(metrics, indent=2))
    print(f"Report saved to: {report_file}")


def main():
    """Main evaluation function."""
    args = parse_args()

    print("="*80)
    print("NUTRITION5K MODEL EVALUATION")
    print("="*80)
    print(f"\nModel type:     {args.model}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Config:         {args.config}")
    print(f"Split:          {args.split}")
    print(f"Device:         {args.device}")
    print(f"Batch size:     {args.batch_size}")

    # Set device
    device = torch.device(args.device)

    # Load config
    config = load_config(args.config)

    # Determine dataset mode
    mode_mapping = {
        'classifier': 'classification',
        'segmentation': 'segmentation',
        'regression': 'regression',
        'end_to_end': 'end_to_end'
    }
    mode = mode_mapping[args.model]

    # Create dataset and dataloader
    print(f"\nLoading {args.split} dataset in {mode} mode...")
    dataset = Nutrition5kDataset(
        root_dir=config.get('data', {}).get('dataset_root', 'data/nutrition5k'),
        split=args.split,
        mode=mode,
        transform=None  # Use default transforms
    )

    # Create dataloader with appropriate collate function
    custom_collate = collate_fn if mode in ['segmentation', 'end_to_end'] else None
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True if args.device == 'cuda' else False
    )

    print(f"Loaded {len(dataset)} samples")

    # Load model
    model = load_model_from_checkpoint(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
        config=config
    )

    # Run evaluation
    if args.model == 'classifier':
        metrics, predictions = evaluate_classifier(model, dataloader, device)
    elif args.model == 'segmentation':
        metrics, predictions = evaluate_segmentation(model, dataloader, device)
    elif args.model == 'regression':
        metrics, predictions = evaluate_regression(model, dataloader, device)
    elif args.model == 'end_to_end':
        metrics, predictions = evaluate_end_to_end(model, dataloader, device)

    # Print report
    print_evaluation_report(metrics, args.model)

    # Save results
    output_dir = Path(args.output_dir)
    save_results(
        metrics=metrics,
        predictions=predictions if args.save_predictions else None,
        output_dir=output_dir,
        model_type=args.model,
        split=args.split,
        save_predictions=args.save_predictions
    )

    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
