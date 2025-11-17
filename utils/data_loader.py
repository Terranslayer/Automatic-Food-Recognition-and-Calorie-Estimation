"""
DataLoader utilities for Nutrition5k dataset.

Provides convenient functions to create train/val/test DataLoaders
with appropriate configurations.
"""

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from utils.dataset import Nutrition5kDataset, get_default_transforms


def create_dataloaders(
    root_dir: str,
    mode: str = "classification",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: Tuple[int, int] = (224, 224),
    subset_size: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        root_dir: Path to dataset root directory
        mode: Dataset mode ('classification', 'regression', 'segmentation', 'end_to_end')
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        image_size: Target image size (height, width)
        subset_size: If specified, use only first N samples per split (for debugging)

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Create datasets for each split
    train_dataset = Nutrition5kDataset(
        root_dir=root_dir,
        split="train",
        mode=mode,
        transform=get_default_transforms("train", image_size),
        subset_size=subset_size,
        image_size=image_size,
    )

    val_dataset = Nutrition5kDataset(
        root_dir=root_dir,
        split="val",
        mode=mode,
        transform=get_default_transforms("val", image_size),
        subset_size=subset_size,
        image_size=image_size,
    )

    test_dataset = Nutrition5kDataset(
        root_dir=root_dir,
        split="test",
        mode=mode,
        transform=get_default_transforms("test", image_size),
        subset_size=subset_size,
        image_size=image_size,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def create_single_dataloader(
    root_dir: str,
    split: str,
    mode: str = "classification",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: Tuple[int, int] = (224, 224),
    subset_size: Optional[int] = None,
    shuffle: bool = None,
) -> DataLoader:
    """
    Create a single DataLoader for specified split.

    Args:
        root_dir: Path to dataset root directory
        split: One of 'train', 'val', 'test'
        mode: Dataset mode
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        image_size: Target image size
        subset_size: If specified, use only first N samples
        shuffle: Whether to shuffle (default: True for train, False otherwise)

    Returns:
        DataLoader for the specified split
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = Nutrition5kDataset(
        root_dir=root_dir,
        split=split,
        mode=mode,
        transform=get_default_transforms(split, image_size),
        subset_size=subset_size,
        image_size=image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )

    return dataloader


def get_dataset_info(root_dir: str, mode: str = "classification") -> Dict:
    """
    Get dataset information without creating full DataLoaders.

    Args:
        root_dir: Path to dataset root directory
        mode: Dataset mode

    Returns:
        Dictionary with dataset statistics
    """
    train_dataset = Nutrition5kDataset(
        root_dir=root_dir,
        split="train",
        mode=mode,
        subset_size=None,
    )

    val_dataset = Nutrition5kDataset(
        root_dir=root_dir,
        split="val",
        mode=mode,
        subset_size=None,
    )

    test_dataset = Nutrition5kDataset(
        root_dir=root_dir,
        split="test",
        mode=mode,
        subset_size=None,
    )

    return {
        "num_classes": train_dataset.num_classes,
        "classes": train_dataset.classes,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "total_size": len(train_dataset) + len(val_dataset) + len(test_dataset),
        "train_class_distribution": train_dataset.get_class_distribution(),
        "train_calorie_stats": train_dataset.get_calorie_statistics(),
    }
