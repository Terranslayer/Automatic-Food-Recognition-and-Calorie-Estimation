"""
Integration tests for the Nutrition5k project.

Tests the complete pipeline: data loading -> model creation -> training -> evaluation.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Import project modules
from utils.dataset import get_datasets, Nutrition5kDataset
from utils.config_loader import load_config
from utils.metrics import accuracy, mae, rmse
from utils.logger import Logger
from utils.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
from models.classifier import FoodClassifier
from models.segmentation import FoodSegmentation
from models.calorie_regressor import CalorieRegressor
from models.end_to_end import EndToEndFoodRecognition


# Skip if dataset not available
DATASET_PATH = Path("data/nutrition5k")
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="Dataset not found"
)


class TestFullPipeline:
    """Test complete training pipeline."""

    @SKIP_IF_NO_DATA
    def test_classifier_pipeline(self):
        """Test classifier: data -> model -> forward -> metrics."""
        # Load data
        train_ds, val_ds, _ = get_datasets(
            str(DATASET_PATH), task='classification'
        )

        # Create model
        model = FoodClassifier(
            num_classes=132,
            backbone='efficientnet_b0',
            pretrained=False
        )
        model.eval()

        # Get batch
        sample = train_ds[0]
        images = sample['image'].unsqueeze(0)
        labels = torch.tensor([sample['label']])

        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        # Check output
        assert outputs.shape == (1, 132)

        # Compute metrics
        acc = accuracy(outputs, labels, topk=1)
        assert 0 <= acc <= 100

    @SKIP_IF_NO_DATA
    def test_regression_pipeline(self):
        """Test regression: data -> feature extractor -> regressor -> metrics."""
        # Load data
        train_ds, _, _ = get_datasets(
            str(DATASET_PATH), task='regression'
        )

        # Create models
        feature_extractor = FoodClassifier(
            num_classes=132,
            backbone='efficientnet_b0',
            pretrained=False
        )
        regressor = CalorieRegressor(
            input_dim=1280,
            output_dim=5
        )
        feature_extractor.eval()
        regressor.eval()

        # Get batch
        sample = train_ds[0]
        images = sample['image'].unsqueeze(0)
        targets = torch.tensor([[
            sample['calories'].item(),
            sample['protein_g'].item(),
            sample['carb_g'].item(),
            sample['fat_g'].item(),
            sample['mass_g'].item()
        ]])

        # Forward pass
        with torch.no_grad():
            features = feature_extractor.extract_features(images)
            outputs = regressor(features)

        # Check output
        assert outputs.shape == (1, 5)

        # Compute metrics
        mae_val = mae(outputs[:, 0], targets[:, 0])
        assert mae_val >= 0

    @SKIP_IF_NO_DATA
    def test_segmentation_pipeline(self):
        """Test segmentation: data -> model -> forward."""
        # Load data
        train_ds, _, _ = get_datasets(
            str(DATASET_PATH), task='segmentation'
        )

        # Create model
        model = FoodSegmentation(
            num_classes=132,
            pretrained=False
        )
        model.eval()

        # Get batch
        sample = train_ds[0]
        images = [sample['image']]

        # Forward pass (inference mode)
        with torch.no_grad():
            outputs = model(images)

        # Check output structure
        assert len(outputs) == 1
        assert 'boxes' in outputs[0]
        assert 'labels' in outputs[0]
        assert 'scores' in outputs[0]
        assert 'masks' in outputs[0]

    @SKIP_IF_NO_DATA
    def test_end_to_end_pipeline(self):
        """Test end-to-end: data -> model -> forward."""
        # Load data
        train_ds, _, _ = get_datasets(
            str(DATASET_PATH), task='end_to_end'
        )

        # Create model
        model = EndToEndFoodRecognition(
            num_classes=132,
            classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False},
            regressor_config={'output_dim': 5}
        )
        model.eval()

        # Get batch
        sample = train_ds[0]
        images = [sample['image']]

        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        # Check output structure
        assert len(outputs) == 1
        assert 'instances' in outputs[0]
        assert 'total_calories' in outputs[0]


class TestReproducibility:
    """Test reproducibility with fixed seeds."""

    def test_seed_consistency(self):
        """Test that same seed produces same results."""
        def run_with_seed(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = FoodClassifier(
                num_classes=132,
                backbone='efficientnet_b0',
                pretrained=False
            )

            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            return out

        # Run twice with same seed
        out1 = run_with_seed(42)
        out2 = run_with_seed(42)

        # Should be identical
        assert torch.allclose(out1, out2)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        def run_with_seed(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = FoodClassifier(
                num_classes=132,
                backbone='efficientnet_b0',
                pretrained=False
            )

            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            return out

        out1 = run_with_seed(42)
        out2 = run_with_seed(123)

        # Should be different
        assert not torch.allclose(out1, out2)


class TestCheckpointIntegration:
    """Test checkpoint save/load integration."""

    def test_checkpoint_roundtrip(self):
        """Test saving and loading checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model
            model = FoodClassifier(
                num_classes=132,
                backbone='efficientnet_b0',
                pretrained=False
            )
            optimizer = torch.optim.Adam(model.parameters())

            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "test.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                metrics={'loss': 0.5},
                config={'test': True},
                filepath=checkpoint_path
            )

            # Load checkpoint
            checkpoint = load_checkpoint(checkpoint_path)

            # Verify
            assert checkpoint['epoch'] == 5
            assert checkpoint['metrics']['loss'] == 0.5
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint

    def test_checkpoint_manager(self):
        """Test CheckpointManager best model tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                max_checkpoints=3
            )

            # Create model
            model = FoodClassifier(
                num_classes=132,
                backbone='efficientnet_b0',
                pretrained=False
            )
            optimizer = torch.optim.Adam(model.parameters())

            # Save multiple checkpoints
            for epoch in range(5):
                manager.save(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={'val_loss': 1.0 - epoch * 0.1},
                    config={'test': True}
                )

            # Check that best is tracked
            assert manager.best_metric is not None


class TestConfigIntegration:
    """Test configuration loading and inheritance."""

    def test_base_config_loading(self):
        """Test loading base config."""
        config = load_config('configs/base.yaml')

        assert 'experiment' in config
        assert 'data' in config
        assert 'training' in config
        assert 'optimizer' in config

    def test_debug_config_inheritance(self):
        """Test debug config inherits from base."""
        config = load_config('configs/debug.yaml')

        # Should have base config values
        assert 'experiment' in config

        # Should have debug overrides
        assert config['data'].get('subset_size') == 10 or \
               config['training'].get('num_epochs') == 1

    def test_model_configs_exist(self):
        """Test all model configs exist and load."""
        config_files = [
            'configs/efficientnet.yaml',
            'configs/vit.yaml',
            'configs/mask_rcnn.yaml',
            'configs/end_to_end.yaml'
        ]

        for config_file in config_files:
            if Path(config_file).exists():
                config = load_config(config_file)
                assert 'model' in config


class TestMetricsIntegration:
    """Test metrics computation."""

    def test_classification_metrics(self):
        """Test classification metrics computation."""
        # Simulated predictions
        logits = torch.tensor([
            [2.0, 1.0, 0.5],
            [0.5, 2.0, 1.0],
            [1.0, 0.5, 2.0]
        ])
        targets = torch.tensor([0, 1, 2])

        acc = accuracy(logits, targets, topk=1)
        assert acc == 100.0  # All correct

    def test_regression_metrics(self):
        """Test regression metrics computation."""
        predictions = torch.tensor([100.0, 200.0, 300.0])
        targets = torch.tensor([110.0, 190.0, 310.0])

        mae_val = mae(predictions, targets)
        rmse_val = rmse(predictions, targets)

        assert mae_val == 10.0
        assert rmse_val == 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
