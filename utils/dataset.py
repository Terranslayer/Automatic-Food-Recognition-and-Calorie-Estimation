"""
Nutrition5k Dataset Implementation.

This module provides PyTorch Dataset classes for loading and processing
the Nutrition5k dataset for food recognition and calorie estimation.

Supports:
- Classification mode: food category prediction
- Segmentation mode: instance segmentation with Mask R-CNN
- Regression mode: calorie and nutrient prediction
- End-to-end mode: combined pipeline
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class Nutrition5kDataset(Dataset):
    """
    Nutrition5k Dataset for food recognition and calorie estimation.

    Args:
        root_dir: Path to dataset root directory
        split: One of 'train', 'val', 'test'
        mode: One of 'classification', 'segmentation', 'regression', 'end_to_end'
        transform: Optional image transformations
        subset_size: If specified, use only first N samples (for debugging)
        image_size: Target image size (height, width)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        mode: str = "classification",
        transform: Optional[Any] = None,
        subset_size: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.mode = mode
        self.transform = transform
        self.image_size = image_size

        # Validate inputs
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert mode in ["classification", "segmentation", "regression", "end_to_end"], \
            f"Invalid mode: {mode}"

        # Load metadata
        self.metadata = self._load_metadata()

        # Apply subset if specified
        if subset_size is not None:
            self.metadata = self.metadata[:subset_size]

        # Build class mapping
        self.classes = self._build_class_mapping()
        self.num_classes = len(self.classes)

        print(f"Loaded {split} split: {len(self.metadata)} samples, "
              f"{self.num_classes} classes, mode={mode}")

    def _load_metadata(self) -> List[Dict]:
        """
        Load and parse metadata from Nutrition5k dataset.

        Returns:
            List of metadata dictionaries for each sample
        """
        metadata_dir = self.root_dir / "metadata"

        # Try to find metadata CSV/JSON files
        # Nutrition5k typically has dish_metadata_cafe*.csv files
        metadata_files = list(metadata_dir.glob("dish_metadata*.csv"))

        if not metadata_files:
            # Fallback: try to find any CSV files
            metadata_files = list(metadata_dir.glob("*.csv"))

        if not metadata_files:
            raise FileNotFoundError(
                f"No metadata files found in {metadata_dir}. "
                "Please ensure the dataset is downloaded correctly."
            )

        # Load all metadata files
        metadata_list = []
        for metadata_file in metadata_files:
            df = pd.read_csv(metadata_file)
            metadata_list.append(df)

        # Combine all metadata
        if metadata_list:
            combined_df = pd.concat(metadata_list, ignore_index=True)
        else:
            raise ValueError("No metadata loaded")

        # Filter by split
        # First, try to load split IDs from txt files (official Nutrition5k format)
        split_ids = self._load_split_ids()

        if split_ids is not None:
            # Filter metadata by split IDs
            combined_df = combined_df[combined_df["dish_id"].isin(split_ids)]
        elif "split" in combined_df.columns:
            # Fallback: use split column if exists
            combined_df = combined_df[combined_df["split"] == self.split]
        else:
            # Fallback: create manual split
            # Use 70% train, 15% val, 15% test
            np.random.seed(42)
            n = len(combined_df)
            indices = np.random.permutation(n)

            if self.split == "train":
                split_indices = indices[:int(0.7 * n)]
            elif self.split == "val":
                split_indices = indices[int(0.7 * n):int(0.85 * n)]
            else:  # test
                split_indices = indices[int(0.85 * n):]

            combined_df = combined_df.iloc[split_indices].reset_index(drop=True)

        # Convert to list of dictionaries
        metadata = []
        for idx, row in combined_df.iterrows():
            item = {
                "dish_id": row.get("dish_id", f"dish_{idx}"),
                "image_path": self._get_image_path(row),
                "calories": float(row.get("total_calories", row.get("calories", 0))),
                "mass_g": float(row.get("total_mass", row.get("mass", 0))),
                "food_category": str(row.get("food_name", row.get("category", "unknown"))),
                # Additional fields
                "protein_g": float(row.get("total_protein", 0)),
                "carb_g": float(row.get("total_carb", 0)),
                "fat_g": float(row.get("total_fat", 0)),
            }

            # Only add if image exists
            if item["image_path"] and os.path.exists(item["image_path"]):
                metadata.append(item)

        if len(metadata) == 0:
            raise ValueError(f"No valid samples found for split '{self.split}'")

        return metadata

    def _load_split_ids(self) -> Optional[set]:
        """
        Load split IDs from txt files (official Nutrition5k format).

        Tries to load from:
        - root_dir/dish_ids/splits/rgb_train_ids.txt
        - root_dir/dish_ids/splits/rgb_test_ids.txt
        - Uses 'test' for both 'val' and 'test' splits

        Returns:
            Set of dish IDs for this split, or None if files not found
        """
        splits_dir = self.root_dir / "dish_ids" / "splits"

        if not splits_dir.exists():
            return None

        # Map split names to file names
        split_file_map = {
            "train": "rgb_train_ids.txt",
            "val": "rgb_test_ids.txt",  # Use test set for validation
            "test": "rgb_test_ids.txt",
        }

        split_file = splits_dir / split_file_map.get(self.split)

        if not split_file.exists():
            return None

        # Read dish IDs from file
        with open(split_file, 'r') as f:
            dish_ids = set(line.strip() for line in f if line.strip())

        return dish_ids

    def _get_image_path(self, row: pd.Series) -> Optional[str]:
        """
        Extract image path from metadata row.

        Supports both official and Kaggle dataset structures:
        - Official: root_dir/imagery/dish_XXXXX.jpg
        - Kaggle: root_dir/dish_XXXXX/frames_sampled30/*.jpeg

        Args:
            row: Pandas Series containing metadata

        Returns:
            Full path to image file, or None if not found
        """
        # Try different possible column names
        possible_columns = ["image_path", "image", "id", "dish_id"]

        for col in possible_columns:
            if col in row and pd.notna(row[col]):
                dish_id = str(row[col])

                # Try official structure: root_dir/imagery/dish_XXXXX.*
                if (self.root_dir / "imagery").exists():
                    # Handle different path formats
                    if dish_id.endswith(('.jpg', '.png', '.jpeg')):
                        image_path = self.root_dir / "imagery" / dish_id
                    else:
                        # Try adding common extensions
                        for ext in ['.jpg', '.png', '.jpeg']:
                            image_path = self.root_dir / "imagery" / f"{dish_id}{ext}"
                            if image_path.exists():
                                return str(image_path)

                    if image_path.exists():
                        return str(image_path)

                # Try Kaggle structure: root_dir/dish_XXXXX/frames_sampled30/*.jpeg
                dish_dir = self.root_dir / dish_id
                if dish_dir.exists():
                    frames_dir = dish_dir / "frames_sampled30"
                    if frames_dir.exists():
                        # Find first available image (any camera, any frame)
                        image_files = list(frames_dir.glob("*.jpeg")) + list(frames_dir.glob("*.jpg"))
                        if image_files:
                            return str(image_files[0])  # Return first image

        return None

    def _build_class_mapping(self) -> List[str]:
        """
        Build mapping from class names to indices.

        Returns:
            Sorted list of unique class names
        """
        categories = set(item["food_category"] for item in self.metadata)
        return sorted(list(categories))

    def _get_class_index(self, category: str) -> int:
        """Get class index for a category name."""
        return self.classes.index(category)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing image and labels (format depends on mode)
        """
        item = self.metadata[idx]

        # Load image
        image = Image.open(item["image_path"]).convert("RGB")

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default: resize and convert to tensor
            image = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])(image)

        # Prepare output based on mode
        if self.mode == "classification":
            return {
                "image": image,
                "label": self._get_class_index(item["food_category"]),
                "category": item["food_category"],
                "dish_id": item["dish_id"],
            }

        elif self.mode == "regression":
            return {
                "image": image,
                "calories": torch.tensor(item["calories"], dtype=torch.float32),
                "mass_g": torch.tensor(item["mass_g"], dtype=torch.float32),
                "protein_g": torch.tensor(item["protein_g"], dtype=torch.float32),
                "carb_g": torch.tensor(item["carb_g"], dtype=torch.float32),
                "fat_g": torch.tensor(item["fat_g"], dtype=torch.float32),
                "label": self._get_class_index(item["food_category"]),
                "category": item["food_category"],
                "dish_id": item["dish_id"],
            }

        elif self.mode == "segmentation":
            # For segmentation, we would need mask annotations
            # This is a placeholder - actual implementation depends on
            # availability of segmentation masks in Nutrition5k
            return {
                "image": image,
                "label": self._get_class_index(item["food_category"]),
                "category": item["food_category"],
                "dish_id": item["dish_id"],
                # TODO: Add masks when available
                # "masks": ...,
                # "boxes": ...,
            }

        elif self.mode == "end_to_end":
            # Combined mode: all information
            return {
                "image": image,
                "label": self._get_class_index(item["food_category"]),
                "category": item["food_category"],
                "calories": torch.tensor(item["calories"], dtype=torch.float32),
                "mass_g": torch.tensor(item["mass_g"], dtype=torch.float32),
                "protein_g": torch.tensor(item["protein_g"], dtype=torch.float32),
                "carb_g": torch.tensor(item["carb_g"], dtype=torch.float32),
                "fat_g": torch.tensor(item["fat_g"], dtype=torch.float32),
                "dish_id": item["dish_id"],
            }

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of samples across classes.

        Returns:
            Dictionary mapping class names to sample counts
        """
        distribution = {}
        for item in self.metadata:
            category = item["food_category"]
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def get_calorie_statistics(self) -> Dict[str, float]:
        """
        Get calorie statistics for the dataset.

        Returns:
            Dictionary with min, max, mean, median calories
        """
        calories = [item["calories"] for item in self.metadata]
        return {
            "min": float(np.min(calories)),
            "max": float(np.max(calories)),
            "mean": float(np.mean(calories)),
            "median": float(np.median(calories)),
            "std": float(np.std(calories)),
        }


def get_default_transforms(split: str, image_size: Tuple[int, int] = (224, 224)):
    """
    Get default image transformations for training/validation.

    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size

    Returns:
        torchvision transforms composition
    """
    if split == "train":
        # Training augmentations
        return T.Compose([
            T.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Validation/test: no augmentation
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])


def get_datasets(root: str, task: str = 'classification', image_size: Tuple[int, int] = (224, 224)):
    """
    Create train, val, and test datasets for different tasks.

    Args:
        root: Path to nutrition5k dataset root
        task: One of 'classification', 'segmentation', 'regression', 'end_to_end'
        image_size: Target image size

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Get transforms
    train_transforms = get_default_transforms('train', image_size)
    val_transforms = get_default_transforms('val', image_size)
    test_transforms = get_default_transforms('test', image_size)

    # Create datasets based on task
    train_dataset = Nutrition5kDataset(
        root=root,
        split='train',
        task=task,
        transform=train_transforms
    )

    val_dataset = Nutrition5kDataset(
        root=root,
        split='val',
        task=task,
        transform=val_transforms
    )

    test_dataset = Nutrition5kDataset(
        root=root,
        split='test',
        task=task,
        transform=test_transforms
    )

    return train_dataset, val_dataset, test_dataset
