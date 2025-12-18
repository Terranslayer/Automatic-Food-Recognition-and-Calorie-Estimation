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
        global_classes: Optional list of class names (for consistent indices across splits)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        mode: str = "classification",
        transform: Optional[Any] = None,
        subset_size: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        global_classes: Optional[List[str]] = None,
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

        # Build class mapping - use global classes if provided for consistent indices
        if global_classes is not None:
            self.classes = global_classes
        else:
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

        # Priority 1: Try dish_metadata_with_labels.csv (contains food categories)
        labeled_metadata = metadata_dir / "dish_metadata_with_labels.csv"
        if labeled_metadata.exists():
            metadata_files = [labeled_metadata]
        else:
            # Priority 2: Try dish_metadata_cafe*.csv files
            metadata_files = list(metadata_dir.glob("dish_metadata_cafe*.csv"))

            if not metadata_files:
                # Priority 3: Try dish_metadata*.csv files
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
            # Get food category - try multiple possible column names
            food_category = "unknown"
            for col_name in ["food_category", "food_name", "category", "primary_ingredient"]:
                if col_name in row and pd.notna(row[col_name]):
                    food_category = str(row[col_name])
                    break

            item = {
                "dish_id": row.get("dish_id", f"dish_{idx}"),
                "image_path": self._get_image_path(row),
                "calories": float(row.get("total_calories", row.get("calories", 0))),
                "mass_g": float(row.get("total_mass", row.get("mass", 0))),
                "food_category": food_category,
                # Additional fields - try multiple column names
                "protein_g": float(row.get("total_protein", row.get("protein", 0))),
                "carb_g": float(row.get("total_carb", row.get("carb", 0))),
                "fat_g": float(row.get("total_fat", row.get("fat", 0))),
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
            # For segmentation with Mask R-CNN
            # Since Nutrition5k doesn't have per-pixel masks, we create a
            # pseudo-mask covering the entire image (single food item per image)
            label_idx = self._get_class_index(item["food_category"])

            # Create pseudo bounding box (full image) and mask
            # Image is already a tensor after transform: [C, H, W]
            _, h, w = image.shape

            # Single box covering most of image (with small margin)
            margin = 0.05
            boxes = torch.tensor([[
                w * margin,      # x1
                h * margin,      # y1
                w * (1-margin),  # x2
                h * (1-margin)   # y2
            ]], dtype=torch.float32)

            # Labels for each box (1-indexed for Mask R-CNN, 0 is background)
            labels = torch.tensor([label_idx + 1], dtype=torch.int64)

            # Pseudo mask: elliptical mask centered in image
            masks = torch.zeros((1, h, w), dtype=torch.uint8)
            center_y, center_x = h // 2, w // 2
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing='ij'
            )
            # Elliptical mask
            ellipse = ((x_coords - center_x) / (w * 0.4))**2 + \
                      ((y_coords - center_y) / (h * 0.4))**2
            masks[0] = (ellipse <= 1).to(torch.uint8)

            # Target dict for Mask R-CNN
            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([hash(item["dish_id"]) % (2**31)]),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((1,), dtype=torch.int64),
            }

            return {
                "image": image,
                "target": target,
                "category": item["food_category"],
                "dish_id": item["dish_id"],
            }

        elif self.mode == "end_to_end":
            # Combined mode: segmentation + classification + regression
            label_idx = self._get_class_index(item["food_category"])

            # Create pseudo bounding box and mask (same as segmentation mode)
            _, h, w = image.shape
            margin = 0.05
            boxes = torch.tensor([[
                w * margin, h * margin,
                w * (1-margin), h * (1-margin)
            ]], dtype=torch.float32)

            labels = torch.tensor([label_idx + 1], dtype=torch.int64)

            # Pseudo elliptical mask
            masks = torch.zeros((1, h, w), dtype=torch.uint8)
            center_y, center_x = h // 2, w // 2
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing='ij'
            )
            ellipse = ((x_coords - center_x) / (w * 0.4))**2 + \
                      ((y_coords - center_y) / (h * 0.4))**2
            masks[0] = (ellipse <= 1).to(torch.uint8)

            # Target dict for Mask R-CNN
            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([hash(item["dish_id"]) % (2**31)]),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((1,), dtype=torch.int64),
            }

            # Stacked nutrition tensor for regression
            nutrition = torch.tensor([
                item["calories"],
                item["protein_g"],
                item["carb_g"],
                item["fat_g"],
                item["mass_g"]
            ], dtype=torch.float32)

            return {
                "image": image,
                "target": target,
                "label": torch.tensor(label_idx, dtype=torch.long),
                "nutrition": nutrition,
                "category": item["food_category"],
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


def get_segmentation_transforms(split: str, image_size: Tuple[int, int] = (224, 224)):
    """
    Get transforms for segmentation tasks (no normalization - Mask R-CNN handles it).

    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size

    Returns:
        torchvision transforms composition
    """
    if split == "train":
        return T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),  # Just convert to tensor, no normalization
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])


def _get_global_classes(root: str) -> List[str]:
    """
    Build a global class list from ALL splits to ensure consistent label indices.

    This is critical for training - without consistent indices, 'beef' might be
    class 11 in train but class 8 in val, causing all predictions to be wrong.

    Args:
        root: Path to nutrition5k dataset root

    Returns:
        Sorted list of all unique food categories across all splits
    """
    root_path = Path(root)
    metadata_dir = root_path / "metadata"

    # Load the labeled metadata file (contains all samples)
    labeled_metadata = metadata_dir / "dish_metadata_with_labels.csv"
    if labeled_metadata.exists():
        df = pd.read_csv(labeled_metadata)
    else:
        # Fallback to other metadata files
        metadata_files = list(metadata_dir.glob("dish_metadata*.csv"))
        if metadata_files:
            df = pd.concat([pd.read_csv(f) for f in metadata_files], ignore_index=True)
        else:
            raise FileNotFoundError(f"No metadata files found in {metadata_dir}")

    # Extract all unique food categories
    all_categories = set()
    for col_name in ["food_category", "food_name", "category", "primary_ingredient"]:
        if col_name in df.columns:
            categories = df[col_name].dropna().astype(str).unique()
            all_categories.update(categories)
            break

    return sorted(list(all_categories))


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
    # Get transforms based on task
    if task in ('segmentation', 'end_to_end'):
        # Segmentation/end-to-end uses special transforms (no normalization)
        train_transforms = get_segmentation_transforms('train', image_size)
        val_transforms = get_segmentation_transforms('val', image_size)
        test_transforms = get_segmentation_transforms('test', image_size)
    else:
        train_transforms = get_default_transforms('train', image_size)
        val_transforms = get_default_transforms('val', image_size)
        test_transforms = get_default_transforms('test', image_size)

    # Build global class mapping FIRST to ensure consistent indices across all splits
    global_classes = _get_global_classes(root)

    # Create datasets with shared class mapping
    train_dataset = Nutrition5kDataset(
        root_dir=root,
        split='train',
        mode=task,
        transform=train_transforms,
        global_classes=global_classes
    )

    val_dataset = Nutrition5kDataset(
        root_dir=root,
        split='val',
        mode=task,
        transform=val_transforms,
        global_classes=global_classes
    )

    test_dataset = Nutrition5kDataset(
        root_dir=root,
        split='test',
        mode=task,
        transform=test_transforms,
        global_classes=global_classes
    )

    return train_dataset, val_dataset, test_dataset
