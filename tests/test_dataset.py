"""
Unit tests for Nutrition5k dataset implementation.

Tests cover:
- Dataset initialization
- Data loading
- Transforms
- DataLoader creation
- Error handling
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import Nutrition5kDataset, get_default_transforms
from utils.data_loader import create_dataloaders, create_single_dataloader, get_dataset_info


# Fixture for dataset path
@pytest.fixture
def dataset_path():
    """Get dataset path from environment or use default."""
    default_path = Path("./data/nutrition5k")
    path = os.getenv("NUTRITION5K_PATH", str(default_path))
    return path


@pytest.fixture
def skip_if_no_dataset(dataset_path):
    """Skip test if dataset is not available."""
    if not Path(dataset_path).exists():
        pytest.skip(f"Dataset not found at {dataset_path}. Run scripts/download_data.py first.")


# Dataset initialization tests
class TestDatasetInitialization:
    """Test dataset initialization and configuration."""

    def test_dataset_init_train(self, dataset_path, skip_if_no_dataset):
        """Test initializing training dataset."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="classification",
            subset_size=10,
        )
        assert len(dataset) > 0
        assert dataset.num_classes > 0

    def test_dataset_init_val(self, dataset_path, skip_if_no_dataset):
        """Test initializing validation dataset."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="val",
            mode="classification",
            subset_size=10,
        )
        assert len(dataset) > 0

    def test_dataset_init_test(self, dataset_path, skip_if_no_dataset):
        """Test initializing test dataset."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="test",
            mode="classification",
            subset_size=10,
        )
        assert len(dataset) > 0

    def test_invalid_split(self, dataset_path):
        """Test that invalid split raises error."""
        with pytest.raises(AssertionError):
            Nutrition5kDataset(
                root_dir=dataset_path,
                split="invalid",
                mode="classification",
            )

    def test_invalid_mode(self, dataset_path):
        """Test that invalid mode raises error."""
        with pytest.raises(AssertionError):
            Nutrition5kDataset(
                root_dir=dataset_path,
                split="train",
                mode="invalid",
            )

    def test_subset_size(self, dataset_path, skip_if_no_dataset):
        """Test subset size limiting."""
        subset_size = 5
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="classification",
            subset_size=subset_size,
        )
        assert len(dataset) == subset_size


# Dataset output tests
class TestDatasetOutput:
    """Test dataset __getitem__ output format."""

    def test_classification_mode_output(self, dataset_path, skip_if_no_dataset):
        """Test classification mode output format."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="classification",
            subset_size=1,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "label" in sample
        assert "category" in sample
        assert "dish_id" in sample

        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape[0] == 3  # RGB channels
        assert isinstance(sample["label"], int)
        assert isinstance(sample["category"], str)

    def test_regression_mode_output(self, dataset_path, skip_if_no_dataset):
        """Test regression mode output format."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="regression",
            subset_size=1,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "calories" in sample
        assert "mass_g" in sample
        assert "label" in sample

        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["calories"], torch.Tensor)
        assert isinstance(sample["mass_g"], torch.Tensor)

    def test_end_to_end_mode_output(self, dataset_path, skip_if_no_dataset):
        """Test end-to-end mode output format."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="end_to_end",
            subset_size=1,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "label" in sample
        assert "calories" in sample
        assert "category" in sample

    def test_image_shape(self, dataset_path, skip_if_no_dataset):
        """Test image shape after transforms."""
        image_size = (224, 224)
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="classification",
            subset_size=1,
            image_size=image_size,
        )

        sample = dataset[0]
        image = sample["image"]

        assert image.shape == (3, image_size[0], image_size[1])


# Transform tests
class TestTransforms:
    """Test data transforms."""

    def test_train_transforms(self):
        """Test training transforms creation."""
        transforms = get_default_transforms("train", image_size=(224, 224))
        assert transforms is not None

    def test_val_transforms(self):
        """Test validation transforms creation."""
        transforms = get_default_transforms("val", image_size=(224, 224))
        assert transforms is not None

    def test_test_transforms(self):
        """Test test transforms creation."""
        transforms = get_default_transforms("test", image_size=(224, 224))
        assert transforms is not None


# DataLoader tests
class TestDataLoader:
    """Test DataLoader creation."""

    def test_create_single_dataloader(self, dataset_path, skip_if_no_dataset):
        """Test creating a single DataLoader."""
        dataloader = create_single_dataloader(
            root_dir=dataset_path,
            split="train",
            mode="classification",
            batch_size=4,
            num_workers=0,  # Use 0 workers for testing
            subset_size=10,
        )

        assert dataloader is not None
        assert len(dataloader) > 0

        # Test iteration
        batch = next(iter(dataloader))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape[0] == 4  # batch size

    def test_create_dataloaders(self, dataset_path, skip_if_no_dataset):
        """Test creating train/val/test DataLoaders."""
        dataloaders = create_dataloaders(
            root_dir=dataset_path,
            mode="classification",
            batch_size=4,
            num_workers=0,
            subset_size=10,
        )

        assert "train" in dataloaders
        assert "val" in dataloaders
        assert "test" in dataloaders

        # Test train loader
        train_batch = next(iter(dataloaders["train"]))
        assert train_batch["image"].shape[0] <= 4

    def test_get_dataset_info(self, dataset_path, skip_if_no_dataset):
        """Test getting dataset information."""
        info = get_dataset_info(dataset_path, mode="classification")

        assert "num_classes" in info
        assert "classes" in info
        assert "train_size" in info
        assert "val_size" in info
        assert "test_size" in info
        assert "train_class_distribution" in info
        assert "train_calorie_stats" in info

        assert info["num_classes"] > 0
        assert info["train_size"] > 0


# Dataset statistics tests
class TestDatasetStatistics:
    """Test dataset statistics methods."""

    def test_class_distribution(self, dataset_path, skip_if_no_dataset):
        """Test class distribution calculation."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="classification",
            subset_size=100,
        )

        distribution = dataset.get_class_distribution()

        assert isinstance(distribution, dict)
        assert len(distribution) > 0
        assert all(isinstance(k, str) for k in distribution.keys())
        assert all(isinstance(v, int) for v in distribution.values())
        assert sum(distribution.values()) == len(dataset)

    def test_calorie_statistics(self, dataset_path, skip_if_no_dataset):
        """Test calorie statistics calculation."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="regression",
            subset_size=100,
        )

        stats = dataset.get_calorie_statistics()

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats

        assert stats["min"] >= 0
        assert stats["max"] > stats["min"]
        assert stats["mean"] > 0


# Edge case tests
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_dataset_directory(self):
        """Test that missing directory raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            dataset = Nutrition5kDataset(
                root_dir="/nonexistent/path",
                split="train",
                mode="classification",
            )

    def test_empty_subset(self, dataset_path, skip_if_no_dataset):
        """Test behavior with subset_size=0."""
        dataset = Nutrition5kDataset(
            root_dir=dataset_path,
            split="train",
            mode="classification",
            subset_size=0,
        )
        assert len(dataset) == 0


# Integration tests
class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_training_workflow(self, dataset_path, skip_if_no_dataset):
        """Test a complete training workflow simulation."""
        # Create DataLoaders
        dataloaders = create_dataloaders(
            root_dir=dataset_path,
            mode="classification",
            batch_size=4,
            num_workers=0,
            subset_size=10,
        )

        # Simulate training loop
        train_loader = dataloaders["train"]
        for i, batch in enumerate(train_loader):
            images = batch["image"]
            labels = batch["label"]

            assert images.shape[0] <= 4
            assert len(labels) <= 4

            if i >= 2:  # Test just a few batches
                break

    def test_multiple_modes(self, dataset_path, skip_if_no_dataset):
        """Test that all modes work correctly."""
        modes = ["classification", "regression", "end_to_end"]

        for mode in modes:
            dataset = Nutrition5kDataset(
                root_dir=dataset_path,
                split="train",
                mode=mode,
                subset_size=5,
            )

            sample = dataset[0]
            assert "image" in sample
            assert isinstance(sample["image"], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
