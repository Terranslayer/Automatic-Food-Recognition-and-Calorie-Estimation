#!/usr/bin/env python3
"""
Training Scripts Tests
Training-Agent | Phase 4

Pytest smoke tests for training scripts and utilities.
Ensures all training scripts can be imported and basic functions work.

Usage:
    pytest tests/test_training.py -v
    python tests/test_training.py
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTrainingScriptImports:
    """Test that all training scripts can be imported."""

    def test_import_train_classifier(self):
        """Test that train_classifier.py can be imported."""
        import train_classifier
        assert hasattr(train_classifier, 'main')
        assert hasattr(train_classifier, 'train_epoch')
        assert hasattr(train_classifier, 'validate')
        print("✓ train_classifier.py imported successfully")

    def test_import_train_segmentation(self):
        """Test that train_segmentation.py can be imported."""
        import train_segmentation
        assert hasattr(train_segmentation, 'main')
        assert hasattr(train_segmentation, 'train_epoch')
        assert hasattr(train_segmentation, 'validate')
        print("✓ train_segmentation.py imported successfully")

    def test_import_train_regression(self):
        """Test that train_regression.py can be imported."""
        import train_regression
        assert hasattr(train_regression, 'main')
        assert hasattr(train_regression, 'train_epoch')
        assert hasattr(train_regression, 'validate')
        print("✓ train_regression.py imported successfully")

    def test_import_train_end_to_end(self):
        """Test that train_end_to_end.py can be imported."""
        import train_end_to_end
        assert hasattr(train_end_to_end, 'main')
        assert hasattr(train_end_to_end, 'train_epoch')
        assert hasattr(train_end_to_end, 'validate')
        print("✓ train_end_to_end.py imported successfully")

    def test_import_evaluate(self):
        """Test that evaluate.py can be imported."""
        import evaluate
        assert hasattr(evaluate, 'main')
        assert hasattr(evaluate, 'evaluate_classifier')
        assert hasattr(evaluate, 'evaluate_segmentation')
        assert hasattr(evaluate, 'evaluate_regression')
        assert hasattr(evaluate, 'evaluate_end_to_end')
        print("✓ evaluate.py imported successfully")


class TestUtilities:
    """Test utility modules."""

    def test_config_loader(self):
        """Test config loader can load base config."""
        from utils.config_loader import load_config
        config = load_config('configs/base.yaml')

        # Check expected keys
        assert 'experiment' in config
        assert 'data' in config
        assert 'training' in config
        assert 'optimizer' in config
        assert 'scheduler' in config

        # Check experiment settings
        assert 'random_seed' in config['experiment']
        assert 'deterministic' in config['experiment']

        # Check training settings
        assert 'batch_size' in config['training']
        assert 'num_epochs' in config['training']

        print(f"✓ Config loaded with {len(config)} top-level keys")

    def test_config_debug_mode(self):
        """Test debug config inherits from base config."""
        from utils.config_loader import load_config
        config = load_config('configs/debug.yaml')

        # Should have all base config keys
        assert 'experiment' in config
        assert 'data' in config
        assert 'training' in config

        # Debug-specific overrides
        assert config['data']['subset_size'] == 10
        assert config['training']['batch_size'] == 4
        assert config['training']['num_epochs'] == 1

        print("✓ Debug config loaded with correct overrides")

    def test_metrics_accuracy(self):
        """Test accuracy metric computation."""
        from utils.metrics import accuracy
        import torch

        # Create dummy predictions and targets
        predictions = torch.tensor([
            [0.1, 0.2, 0.7],  # predicts class 2
            [0.8, 0.1, 0.1],  # predicts class 0
            [0.2, 0.6, 0.2],  # predicts class 1
            [0.3, 0.3, 0.4],  # predicts class 2
        ])
        targets = torch.tensor([2, 0, 1, 2])

        acc = accuracy(predictions, targets, topk=1)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
        assert acc == 100.0  # All predictions correct

        print(f"✓ Accuracy computed: {acc:.2f}%")

    def test_metrics_mae(self):
        """Test MAE metric computation."""
        from utils.metrics import mae
        import torch

        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.2, 2.8, 4.1])

        mae_value = mae(predictions, targets)
        assert isinstance(mae_value, float)
        assert mae_value >= 0

        # Manual check
        expected_mae = torch.mean(torch.abs(predictions - targets)).item()
        assert abs(mae_value - expected_mae) < 1e-5

        print(f"✓ MAE computed: {mae_value:.4f}")

    def test_metrics_rmse(self):
        """Test RMSE metric computation."""
        from utils.metrics import rmse
        import torch

        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.2, 2.8, 4.1])

        rmse_value = rmse(predictions, targets)
        assert isinstance(rmse_value, float)
        assert rmse_value >= 0

        print(f"✓ RMSE computed: {rmse_value:.4f}")

    def test_metrics_mape(self):
        """Test MAPE metric computation."""
        from utils.metrics import mape
        import torch

        predictions = torch.tensor([100.0, 200.0, 300.0])
        targets = torch.tensor([110.0, 190.0, 310.0])

        mape_value = mape(predictions, targets)
        assert isinstance(mape_value, float)
        assert mape_value >= 0

        print(f"✓ MAPE computed: {mape_value:.2f}%")

    def test_metrics_r2_score(self):
        """Test R² score computation."""
        from utils.metrics import r2_score
        import torch

        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = r2_score(predictions, targets)
        assert isinstance(r2, float)
        assert abs(r2 - 1.0) < 1e-5  # Perfect predictions

        print(f"✓ R² score computed: {r2:.4f}")

    def test_metric_aggregator(self):
        """Test MetricAggregator utility."""
        from utils.metrics import MetricAggregator
        import torch

        aggregator = MetricAggregator()

        # Add some metrics
        aggregator.update({'loss': 1.5})
        aggregator.update({'loss': 1.3})
        aggregator.update({'loss': 1.2})
        aggregator.update({'accuracy': 85.0})
        aggregator.update({'accuracy': 87.0})

        # Get averages
        avg_loss = aggregator.get('loss')
        avg_acc = aggregator.get('accuracy')

        assert isinstance(avg_loss, float)
        assert isinstance(avg_acc, float)
        assert abs(avg_loss - 1.333) < 0.01
        assert abs(avg_acc - 86.0) < 0.01

        # Get all metrics
        all_metrics = aggregator.compute()
        assert 'loss' in all_metrics
        assert 'accuracy' in all_metrics

        # Reset
        aggregator.reset()
        assert len(aggregator.compute()) == 0

        print("✓ MetricAggregator works correctly")

    def test_logger_creation(self):
        """Test Logger can be created and used."""
        from utils.logger import Logger
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Logger(tmpdir, experiment_name='test', use_tensorboard=True, use_wandb=False)

            # Log some metrics
            logger.log_scalar('train/loss', 1.5, step=0)
            logger.log_scalar('train/loss', 1.3, step=1)
            logger.log_scalar('val/accuracy', 85.0, step=0)

            # Check log directory created
            log_dir = Path(tmpdir)
            assert log_dir.exists()

            logger.close()

        print("✓ Logger created and used successfully")

    def test_checkpoint_save_load(self):
        """Test checkpoint save and load functionality."""
        from utils.checkpoint import save_checkpoint, load_checkpoint
        import torch
        import torch.nn as nn

        # Create dummy model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_checkpoint.pth'

            # Save checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                metrics={'loss': 1.5, 'accuracy': 85.0},
                config={'test': True},
                filepath=filepath,
                scheduler=None
            )

            # Check file exists
            assert filepath.exists()

            # Load checkpoint
            checkpoint = load_checkpoint(filepath)

            # Verify contents
            assert checkpoint['epoch'] == 5
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'metrics' in checkpoint
            assert checkpoint['metrics']['loss'] == 1.5
            assert checkpoint['metrics']['accuracy'] == 85.0

        print("✓ Checkpoint save/load works correctly")

    def test_checkpoint_best_model_save(self):
        """Test CheckpointManager for best model tracking."""
        from utils.checkpoint import CheckpointManager
        import torch.nn as nn
        import torch

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            # Create checkpoint manager
            manager = CheckpointManager(
                checkpoint_dir=save_dir,
                max_checkpoints=2,
                metric_name='val_loss',
                mode='min'
            )

            # Save checkpoint with manager
            manager.save(
                model=model,
                optimizer=optimizer,
                epoch=10,
                metrics={'val_loss': 0.8},
                config={'test': True}
            )

            # Check checkpoint was saved
            assert len(list(save_dir.glob('*.pth'))) > 0

        print("✓ Best model checkpoint saved correctly")


class TestDatasetLoading:
    """Test dataset loading (skipped if dataset not available)."""

    def test_dataset_classification_mode(self):
        """Test dataset loading in classification mode."""
        from utils.dataset import Nutrition5kDataset

        try:
            dataset = Nutrition5kDataset(
                root_dir='data/nutrition5k',
                split='train',
                mode='classification',
                subset_size=5
            )
            assert len(dataset) <= 5

            # Try to get first sample
            sample = dataset[0]
            assert 'image' in sample
            assert 'dish_id' in sample

            print(f"✓ Classification dataset loaded: {len(dataset)} samples")

        except (FileNotFoundError, ValueError) as e:
            pytest.skip(f"Dataset not available: {e}")

    def test_dataset_regression_mode(self):
        """Test dataset loading in regression mode."""
        from utils.dataset import Nutrition5kDataset

        try:
            dataset = Nutrition5kDataset(
                root_dir='data/nutrition5k',
                split='train',
                mode='regression',
                subset_size=5
            )
            assert len(dataset) <= 5

            # Try to get first sample
            sample = dataset[0]
            assert 'image' in sample
            assert 'calories' in sample
            assert 'protein_g' in sample

            print(f"✓ Regression dataset loaded: {len(dataset)} samples")

        except (FileNotFoundError, ValueError) as e:
            pytest.skip(f"Dataset not available: {e}")

    def test_dataset_segmentation_mode(self):
        """Test dataset loading in segmentation mode."""
        from utils.dataset import Nutrition5kDataset

        try:
            dataset = Nutrition5kDataset(
                root_dir='data/nutrition5k',
                split='train',
                mode='segmentation',
                subset_size=5
            )
            assert len(dataset) <= 5

            print(f"✓ Segmentation dataset loaded: {len(dataset)} samples")

        except (FileNotFoundError, ValueError) as e:
            pytest.skip(f"Dataset not available: {e}")

    def test_dataloader_creation(self):
        """Test dataloader creation."""
        from utils.dataset import Nutrition5kDataset
        from torch.utils.data import DataLoader

        try:
            dataset = Nutrition5kDataset(
                root_dir='data/nutrition5k',
                split='train',
                mode='classification',
                subset_size=10
            )

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0
            )

            # Try to get one batch
            batch = next(iter(dataloader))
            assert 'image' in batch
            assert 'dish_id' in batch

            print(f"✓ DataLoader created: {len(dataloader)} batches")

        except (FileNotFoundError, ValueError) as e:
            pytest.skip(f"Dataset not available: {e}")


class TestModelInstantiation:
    """Test that models can be instantiated."""

    def test_classifier_instantiation(self):
        """Test FoodClassifier can be instantiated."""
        from models.classifier import FoodClassifier
        import torch

        model = FoodClassifier(
            num_classes=132,
            backbone='efficientnet_b0',
            pretrained=False
        )

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 132)

        print("✓ FoodClassifier instantiated and forward pass successful")

    def test_segmentation_instantiation(self):
        """Test FoodSegmentation can be instantiated."""
        from models.segmentation import FoodSegmentation
        import torch

        model = FoodSegmentation(
            num_classes=132,
            pretrained=False
        )

        # Set to eval mode for inference
        model.eval()

        # Test forward pass
        x = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]
        with torch.no_grad():
            output = model(x)

        assert isinstance(output, list)
        assert len(output) == 2

        print("✓ FoodSegmentation instantiated and forward pass successful")

    def test_regression_instantiation(self):
        """Test CalorieRegressor can be instantiated."""
        from models.calorie_regressor import CalorieRegressor
        import torch

        model = CalorieRegressor(
            input_dim=1280,  # EfficientNet-B0 feature dimension
            output_dim=5
        )

        # Test forward pass with features (not images)
        x = torch.randn(2, 1280)  # Features, not images
        output = model(x)
        assert output.shape == (2, 5)  # calories, protein, carb, fat, mass

        print("✓ CalorieRegressor instantiated and forward pass successful")

    def test_end_to_end_instantiation(self):
        """Test EndToEndFoodRecognition can be instantiated."""
        from models.end_to_end import EndToEndFoodRecognition
        import torch

        model = EndToEndFoodRecognition(
            num_classes=132,
            classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False},
            regressor_config={'output_dim': 5}  # visual_dim is set automatically
        )

        # Set to eval mode
        model.eval()

        # Test forward pass
        x = [torch.randn(3, 800, 800), torch.randn(3, 800, 800)]
        with torch.no_grad():
            output = model(x)

        # End-to-end model returns list of results, one per image
        assert len(output) == 2
        assert 'instances' in output[0]

        print("EndToEndFoodRecognition instantiated and forward pass successful")


def run_all_tests():
    """Run all tests with pytest."""
    print("\n" + "="*80)
    print("Running Training Scripts Tests")
    print("="*80 + "\n")

    # Run pytest
    exit_code = pytest.main([__file__, '-v', '--tb=short'])

    print("\n" + "="*80)
    if exit_code == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80 + "\n")

    return exit_code


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
