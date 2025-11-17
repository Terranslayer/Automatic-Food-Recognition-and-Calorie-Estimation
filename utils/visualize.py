"""
Visualization utilities for Nutrition5k dataset and model outputs.

Provides functions to visualize images, predictions, and dataset statistics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
import seaborn as sns
from PIL import Image


def denormalize_image(image: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406],
                     std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Denormalize an image tensor for visualization.

    Args:
        image: Normalized image tensor (C, H, W)
        mean: Normalization mean values
        std: Normalization std values

    Returns:
        Denormalized image as numpy array (H, W, C) in range [0, 1]
    """
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = torch.clamp(image, 0, 1)
    return image.permute(1, 2, 0).cpu().numpy()


def visualize_batch(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    predictions: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    calories: Optional[torch.Tensor] = None,
    pred_calories: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    max_images: int = 16,
):
    """
    Visualize a batch of images with labels and predictions.

    Args:
        images: Batch of images (B, C, H, W)
        labels: Ground truth labels (B,)
        predictions: Predicted labels (B,)
        class_names: List of class names
        calories: Ground truth calories (B,)
        pred_calories: Predicted calories (B,)
        save_path: If specified, save visualization to this path
        max_images: Maximum number of images to display
    """
    batch_size = min(len(images), max_images)
    n_cols = 4
    n_rows = (batch_size + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if batch_size > 1 else [axes]

    for i in range(batch_size):
        ax = axes[i]

        # Denormalize and display image
        img = denormalize_image(images[i])
        ax.imshow(img)

        # Build title
        title_parts = []

        if labels is not None:
            label_str = class_names[labels[i]] if class_names else f"Label: {labels[i]}"
            title_parts.append(f"GT: {label_str}")

        if predictions is not None:
            pred_str = class_names[predictions[i]] if class_names else f"Pred: {predictions[i]}"
            correct = (labels[i] == predictions[i]) if labels is not None else None
            color = "green" if correct else "red"
            title_parts.append(f"Pred: {pred_str}")

        if calories is not None:
            title_parts.append(f"Cal: {calories[i]:.0f} kcal")

        if pred_calories is not None:
            title_parts.append(f"Pred Cal: {pred_calories[i]:.0f} kcal")

        ax.set_title("\n".join(title_parts), fontsize=10)
        ax.axis("off")

    # Hide unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_dataset_samples(
    dataset,
    num_samples: int = 9,
    save_path: Optional[str] = None,
):
    """
    Visualize random samples from a dataset.

    Args:
        dataset: PyTorch Dataset object
        num_samples: Number of samples to visualize
        save_path: If specified, save visualization to this path
    """
    n_cols = 3
    n_rows = (num_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    # Sample random indices
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i]

        sample = dataset[idx]
        image = sample["image"]
        category = sample.get("category", "Unknown")

        # Denormalize and display
        img = denormalize_image(image)
        ax.imshow(img)

        # Build title
        title = f"{category}"
        if "calories" in sample:
            calories = sample["calories"].item() if torch.is_tensor(sample["calories"]) else sample["calories"]
            title += f"\n{calories:.0f} kcal"

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_class_distribution(
    class_distribution: Dict[str, int],
    save_path: Optional[str] = None,
    title: str = "Class Distribution",
):
    """
    Plot class distribution as a bar chart.

    Args:
        class_distribution: Dictionary mapping class names to counts
        save_path: If specified, save plot to this path
        title: Plot title
    """
    # Sort by count (descending)
    sorted_items = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(classes)), counts)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_calorie_distribution(
    calories: List[float],
    save_path: Optional[str] = None,
    title: str = "Calorie Distribution",
    bins: int = 50,
):
    """
    Plot calorie distribution as a histogram.

    Args:
        calories: List of calorie values
        save_path: If specified, save plot to this path
        title: Plot title
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(calories, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(calories), color="red", linestyle="--",
               label=f"Mean: {np.mean(calories):.1f} kcal")
    ax.axvline(np.median(calories), color="green", linestyle="--",
               label=f"Median: {np.median(calories):.1f} kcal")

    ax.set_xlabel("Calories (kcal)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
):
    """
    Plot confusion matrix as a heatmap.

    Args:
        confusion_matrix: Confusion matrix (N_classes, N_classes)
        class_names: List of class names
        save_path: If specified, save plot to this path
        title: Plot title
        normalize: If True, normalize confusion matrix by row
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (
            confusion_matrix.sum(axis=1, keepdims=True) + 1e-10
        )

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"}
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Optional dict of training metrics
        val_metrics: Optional dict of validation metrics
        save_path: If specified, save plot to this path
    """
    n_plots = 1
    if train_metrics:
        n_plots += len(train_metrics)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label="Train Loss")
    axes[0].plot(epochs, val_losses, 'r-', label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot additional metrics
    if train_metrics:
        for idx, (metric_name, train_vals) in enumerate(train_metrics.items(), 1):
            val_vals = val_metrics.get(metric_name, []) if val_metrics else []

            axes[idx].plot(epochs, train_vals, 'b-', label=f"Train {metric_name}")
            if val_vals:
                axes[idx].plot(epochs, val_vals, 'r-', label=f"Val {metric_name}")

            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f"Training and Validation {metric_name}")
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()
