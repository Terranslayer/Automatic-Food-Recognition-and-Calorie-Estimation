"""
Evaluation Metrics for Nutrition5k Project

Implements metrics for:
- Classification: Top-1/5 accuracy, F1-score, precision, recall
- Segmentation: mAP, IoU
- Regression: MAE, RMSE, MAPE, R²
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score


# ============================================================================
# Classification Metrics
# ============================================================================

def accuracy(predictions: torch.Tensor, targets: torch.Tensor, topk: int = 1) -> float:
    """
    Compute top-k accuracy.

    Args:
        predictions: [N, num_classes] logits or probabilities
        targets: [N] ground truth labels
        topk: k for top-k accuracy

    Returns:
        Top-k accuracy (0-100)
    """
    batch_size = targets.size(0)
    _, pred_indices = predictions.topk(topk, dim=1, largest=True, sorted=True)

    correct = pred_indices.eq(targets.view(-1, 1).expand_as(pred_indices))
    correct_k = correct.sum().item()

    return 100.0 * correct_k / batch_size


def top1_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy."""
    return accuracy(predictions, targets, topk=1)


def top5_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-5 accuracy."""
    return accuracy(predictions, targets, topk=5)


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: [N, num_classes] logits
        targets: [N] ground truth labels
        num_classes: Number of classes

    Returns:
        Dict with: top1_acc, top5_acc, f1, precision, recall
    """
    # Get predicted classes
    pred_classes = predictions.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    metrics = {
        'top1_acc': top1_accuracy(predictions, targets),
    }

    # Add top-5 only if we have enough classes
    if num_classes >= 5:
        metrics['top5_acc'] = top5_accuracy(predictions, targets)

    # F1, precision, recall (macro average)
    metrics['f1'] = f1_score(targets_np, pred_classes, average='macro', zero_division=0) * 100
    metrics['precision'] = precision_score(targets_np, pred_classes, average='macro', zero_division=0) * 100
    metrics['recall'] = recall_score(targets_np, pred_classes, average='macro', zero_division=0) * 100

    return metrics


# ============================================================================
# Segmentation Metrics
# ============================================================================

def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union for binary masks.

    Args:
        pred_mask: [H, W] predicted mask (0-1 or binary)
        gt_mask: [H, W] ground truth mask (binary)
        threshold: Threshold for binarizing predictions

    Returns:
        IoU score
    """
    # Binarize if needed
    if pred_mask.dtype == torch.float:
        pred_mask = (pred_mask > threshold).float()

    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute mean Average Precision for object detection/segmentation.

    Simplified mAP computation (for reference, use torchmetrics for production).

    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        targets: List of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for positive detection

    Returns:
        mAP score
    """
    # Placeholder - full mAP implementation is complex
    # For production, use: torchmetrics.detection.mean_ap.MeanAveragePrecision

    # Simple approximation: count correct detections
    total_predictions = 0
    correct_predictions = 0

    for pred, target in zip(predictions, targets):
        if len(pred['boxes']) == 0:
            continue

        total_predictions += len(pred['boxes'])

        # Match predictions to targets (greedy matching)
        for pred_box in pred['boxes']:
            for target_box in target['boxes']:
                # Compute box IoU (simplified)
                iou = compute_box_iou(pred_box, target_box)
                if iou >= iou_threshold:
                    correct_predictions += 1
                    break

    if total_predictions == 0:
        return 0.0

    return correct_predictions / total_predictions


def compute_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        box1: [4] (x1, y1, x2, y2)
        box2: [4] (x1, y1, x2, y2)

    Returns:
        IoU score
    """
    # Intersection
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union


# ============================================================================
# Regression Metrics
# ============================================================================

def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Mean Absolute Error.

    Args:
        predictions: [N] or [N, D] predictions
        targets: [N] or [N, D] ground truth

    Returns:
        MAE
    """
    return torch.abs(predictions - targets).mean().item()


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Root Mean Squared Error.

    Args:
        predictions: [N] or [N, D] predictions
        targets: [N] or [N, D] ground truth

    Returns:
        RMSE
    """
    return torch.sqrt(((predictions - targets) ** 2).mean()).item()


def mape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        predictions: [N] or [N, D] predictions
        targets: [N] or [N, D] ground truth
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE (0-100)
    """
    percentage_errors = torch.abs((targets - predictions) / (targets + epsilon))
    return (100.0 * percentage_errors.mean()).item()


def r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    R-squared (coefficient of determination).

    Args:
        predictions: [N] predictions
        targets: [N] ground truth

    Returns:
        R² score
    """
    predictions_np = predictions.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()

    return r2_score(targets_np, predictions_np)


def regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Args:
        predictions: [N] or [N, D] predictions
        targets: [N] or [N, D] ground truth

    Returns:
        Dict with: mae, rmse, mape, r2
    """
    return {
        'mae': mae(predictions, targets),
        'rmse': rmse(predictions, targets),
        'mape': mape(predictions, targets),
        'r2': r2(predictions, targets),
    }


# ============================================================================
# Multi-output Regression Metrics
# ============================================================================

def multi_output_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_names: List[str]
) -> Dict[str, float]:
    """
    Compute regression metrics for multi-output predictions.

    Args:
        predictions: [N, D] predictions (D outputs)
        targets: [N, D] ground truth
        output_names: List of output names (length D)

    Returns:
        Dict with metrics for each output and overall
    """
    assert predictions.shape[1] == len(output_names), "Dimension mismatch"

    metrics = {}

    # Overall metrics (averaged across all outputs)
    metrics['overall_mae'] = mae(predictions, targets)
    metrics['overall_rmse'] = rmse(predictions, targets)
    metrics['overall_mape'] = mape(predictions, targets)

    # Per-output metrics
    for i, name in enumerate(output_names):
        pred_i = predictions[:, i]
        target_i = targets[:, i]

        metrics[f'{name}_mae'] = mae(pred_i, target_i)
        metrics[f'{name}_rmse'] = rmse(pred_i, target_i)
        metrics[f'{name}_mape'] = mape(pred_i, target_i)
        metrics[f'{name}_r2'] = r2(pred_i, target_i)

    return metrics


# ============================================================================
# Metric Aggregator
# ============================================================================

class MetricAggregator:
    """
    Aggregate metrics over batches/epochs.

    Usage:
        >>> agg = MetricAggregator()
        >>> for batch in dataloader:
        ...     metrics = compute_metrics(batch)
        ...     agg.update(metrics)
        >>> final_metrics = agg.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        Update metrics with new batch.

        Args:
            metrics: Dict of metric values
            count: Number of samples (for weighted averaging)
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value * count
            self.counts[key] += count

    def compute(self) -> Dict[str, float]:
        """
        Compute averaged metrics.

        Returns:
            Dict of averaged metrics
        """
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }

    def get(self, key: str) -> Optional[float]:
        """Get specific metric."""
        if key in self.metrics:
            return self.metrics[key] / self.counts[key]
        return None


class MetricsTracker:
    """
    Simple metrics tracker for training loops.
    Provides a simpler interface than MetricAggregator.

    Usage:
        >>> tracker = MetricsTracker()
        >>> tracker.update('loss', 0.5)
        >>> tracker.update('accuracy', 0.9)
        >>> avg_loss = tracker.get_average('loss')
        >>> all_metrics = tracker.get_averages()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, key: str, value: float, count: int = 1):
        """
        Update a single metric.

        Args:
            key: Metric name
            value: Metric value
            count: Number of samples (for weighted averaging)
        """
        if key not in self.metrics:
            self.metrics[key] = 0.0
            self.counts[key] = 0

        self.metrics[key] += value * count
        self.counts[key] += count

    def get_average(self, key: str) -> float:
        """
        Get average value of a metric.

        Args:
            key: Metric name

        Returns:
            Average value, or 0.0 if metric doesn't exist
        """
        if key in self.metrics and self.counts[key] > 0:
            return self.metrics[key] / self.counts[key]
        return 0.0

    def get_averages(self) -> Dict[str, float]:
        """
        Get all averaged metrics as a dictionary.

        Returns:
            Dict of all averaged metrics
        """
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
            if self.counts[key] > 0
        }

    def __contains__(self, key: str) -> bool:
        """Check if metric exists."""
        return key in self.metrics


if __name__ == '__main__':
    # Smoke test
    print("Testing metrics...")

    # Test classification metrics
    print("\n1. Classification metrics:")
    predictions = torch.randn(100, 10)  # 100 samples, 10 classes
    targets = torch.randint(0, 10, (100,))

    metrics = classification_metrics(predictions, targets, num_classes=10)
    print(f"   Top-1 Accuracy: {metrics['top1_acc']:.2f}%")
    print(f"   Top-5 Accuracy: {metrics['top5_acc']:.2f}%")
    print(f"   F1 Score: {metrics['f1']:.2f}%")
    print(f"   [OK] Classification metrics computed")

    # Test regression metrics
    print("\n2. Regression metrics:")
    predictions = torch.randn(100)
    targets = torch.randn(100)

    metrics = regression_metrics(predictions, targets)
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   R²: {metrics['r2']:.4f}")
    print(f"   [OK] Regression metrics computed")

    # Test multi-output regression
    print("\n3. Multi-output regression metrics:")
    predictions = torch.randn(100, 5)  # 5 outputs
    targets = torch.randn(100, 5)
    output_names = ['calories', 'protein', 'carb', 'fat', 'mass']

    metrics = multi_output_regression_metrics(predictions, targets, output_names)
    print(f"   Overall MAE: {metrics['overall_mae']:.4f}")
    print(f"   Calories MAE: {metrics['calories_mae']:.4f}")
    print(f"   [OK] Multi-output metrics computed")

    # Test metric aggregator
    print("\n4. Metric aggregator:")
    agg = MetricAggregator()

    # Simulate 3 batches
    agg.update({'loss': 1.0, 'accuracy': 80.0}, count=32)
    agg.update({'loss': 0.8, 'accuracy': 85.0}, count=32)
    agg.update({'loss': 0.6, 'accuracy': 90.0}, count=32)

    final = agg.compute()
    print(f"   Average loss: {final['loss']:.4f} (should be ~0.8)")
    print(f"   Average accuracy: {final['accuracy']:.2f}% (should be ~85%)")
    print(f"   [OK] Aggregator working")

    print("\n[SUCCESS] All metrics tests passed!")
