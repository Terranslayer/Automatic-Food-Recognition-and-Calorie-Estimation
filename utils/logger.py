"""
TensorBoard Logger with Optional WandB Integration

Provides unified logging interface for:
- Scalar metrics (loss, accuracy, etc.)
- Images (predictions, visualizations)
- Histograms (weights, gradients)
- Hyperparameters and metadata

Supports:
- TensorBoard (torch.utils.tensorboard)
- Weights & Biases (wandb) - optional
- Experiment metadata tracking
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# WandB is optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """
    Unified experiment logger with TensorBoard and optional WandB support.

    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        config: Configuration dict to save as metadata
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable Weights & Biases logging
        wandb_project: WandB project name
        wandb_entity: WandB entity/username

    Example:
        >>> logger = Logger(
        ...     log_dir='experiments/run1',
        ...     experiment_name='efficientnet_b0',
        ...     config={'lr': 0.001, 'batch_size': 32},
        ...     use_tensorboard=True
        ... )
        >>> logger.log_scalar('loss/train', 0.5, step=100)
        >>> logger.log_image('predictions', image_tensor, step=100)
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.config = config or {}
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard
        self.tb_writer = None
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"[Logger] TensorBoard logging to: {self.log_dir}")

        # Initialize WandB
        self.wandb_run = None
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("[Logger] WARNING: wandb not installed, disabling WandB logging")
                self.use_wandb = False
            else:
                self.wandb_run = wandb.init(
                    project=wandb_project or "nutrition5k",
                    entity=wandb_entity,
                    name=experiment_name,
                    config=self.config,
                    dir=str(self.log_dir)
                )
                print(f"[Logger] WandB logging initialized")

        # Save experiment metadata
        self._save_metadata()

    def _save_metadata(self):
        """Save experiment metadata to JSON file."""
        metadata = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'log_dir': str(self.log_dir),
        }

        metadata_path = self.log_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def log_scalar(
        self,
        tag: str,
        value: Union[float, int, torch.Tensor],
        step: int
    ):
        """
        Log scalar value.

        Args:
            tag: Metric name (e.g., 'loss/train', 'accuracy/val')
            value: Scalar value
            step: Global step/iteration
        """
        # Convert tensor to float
        if isinstance(value, torch.Tensor):
            value = value.item()

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

        # WandB
        if self.wandb_run is not None:
            wandb.log({tag: value}, step=step)

    def log_scalars(
        self,
        tag_value_dict: Dict[str, Union[float, int, torch.Tensor]],
        step: int
    ):
        """
        Log multiple scalars at once.

        Args:
            tag_value_dict: Dict of {tag: value}
            step: Global step/iteration
        """
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)

    def log_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.ndarray],
        step: int,
        dataformats: str = 'CHW'
    ):
        """
        Log image.

        Args:
            tag: Image tag
            image: Image tensor [C, H, W] or [H, W, C] or numpy array
            step: Global step
            dataformats: Format of image ('CHW', 'HWC', 'HW')
        """
        # Convert to tensor if numpy
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_image(tag, image, step, dataformats=dataformats)

        # WandB
        if self.wandb_run is not None:
            # WandB expects HWC format
            if dataformats == 'CHW':
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
            wandb.log({tag: wandb.Image(image_np)}, step=step)

    def log_images(
        self,
        tag: str,
        images: Union[torch.Tensor, np.ndarray],
        step: int,
        dataformats: str = 'NCHW'
    ):
        """
        Log multiple images as a grid.

        Args:
            tag: Image tag
            images: Image tensor [N, C, H, W] or numpy array
            step: Global step
            dataformats: Format of images
        """
        # Convert to tensor if numpy
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        # TensorBoard (uses grid)
        if self.tb_writer is not None:
            from torchvision.utils import make_grid
            grid = make_grid(images, normalize=True)
            self.tb_writer.add_image(tag, grid, step)

        # WandB (logs multiple images)
        if self.wandb_run is not None:
            images_list = [wandb.Image(img) for img in images]
            wandb.log({tag: images_list}, step=step)

    def log_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        step: int,
        bins: str = 'tensorflow'
    ):
        """
        Log histogram of values (e.g., weights, gradients).

        Args:
            tag: Histogram tag
            values: Tensor or array of values
            step: Global step
            bins: Binning method ('tensorflow', 'auto', 'fd', etc.)
        """
        # Convert to tensor if numpy
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)

        # Flatten if multi-dimensional
        values = values.flatten()

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, step, bins=bins)

        # WandB
        if self.wandb_run is not None:
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)

    def log_model_parameters(
        self,
        model: torch.nn.Module,
        step: int,
        log_weights: bool = True,
        log_gradients: bool = True
    ):
        """
        Log model parameters and gradients.

        Args:
            model: PyTorch model
            step: Global step
            log_weights: Whether to log parameter histograms
            log_gradients: Whether to log gradient histograms
        """
        for name, param in model.named_parameters():
            if log_weights and param.requires_grad:
                self.log_histogram(f'weights/{name}', param.data, step)

            if log_gradients and param.grad is not None:
                self.log_histogram(f'gradients/{name}', param.grad, step)

    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log hyperparameters and final metrics.

        Args:
            hparams: Hyperparameter dict
            metrics: Final metric values (optional)
        """
        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_hparams(
                hparams,
                metrics or {},
                run_name='.'
            )

        # WandB (already logged in init)
        if self.wandb_run is not None:
            wandb.config.update(hparams, allow_val_change=True)
            if metrics:
                wandb.summary.update(metrics)

    def log_text(self, tag: str, text: str, step: int):
        """
        Log text.

        Args:
            tag: Text tag
            text: Text content
            step: Global step
        """
        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_text(tag, text, step)

        # WandB
        if self.wandb_run is not None:
            wandb.log({tag: text}, step=step)

    def flush(self):
        """Flush pending logs to disk."""
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def close(self):
        """Close logger and cleanup resources."""
        if self.tb_writer is not None:
            self.tb_writer.close()
            print(f"[Logger] TensorBoard writer closed")

        if self.wandb_run is not None:
            wandb.finish()
            print(f"[Logger] WandB run finished")


def create_logger(
    log_dir: str,
    experiment_name: str,
    config: Dict[str, Any]
) -> Logger:
    """
    Factory function to create logger from config.

    Args:
        log_dir: Log directory
        experiment_name: Experiment name
        config: Config dict (should contain 'logging' section)

    Returns:
        Logger instance
    """
    logging_config = config.get('logging', {})

    return Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        config=config,
        use_tensorboard=logging_config.get('use_tensorboard', True),
        use_wandb=logging_config.get('use_wandb', False),
        wandb_project=logging_config.get('wandb_project'),
        wandb_entity=logging_config.get('wandb_entity'),
    )


if __name__ == '__main__':
    # Smoke test
    print("Testing logger...")

    import tempfile
    import shutil

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Basic logger creation
        print("\n1. Creating logger:")
        config = {
            'model': 'efficientnet_b0',
            'lr': 0.001,
            'batch_size': 32,
        }
        logger = Logger(
            log_dir=os.path.join(temp_dir, 'test_run'),
            experiment_name='test_experiment',
            config=config,
            use_tensorboard=True,
            use_wandb=False
        )
        print(f"   [OK] Logger created at {logger.log_dir}")

        # Test 2: Log scalars
        print("\n2. Logging scalars:")
        logger.log_scalar('loss/train', 0.5, step=0)
        logger.log_scalar('loss/train', 0.4, step=1)
        logger.log_scalar('loss/train', 0.3, step=2)
        logger.log_scalars({
            'accuracy/train': 85.0,
            'accuracy/val': 80.0
        }, step=2)
        print(f"   [OK] Scalars logged")

        # Test 3: Log images
        print("\n3. Logging images:")
        # Single image
        image = torch.rand(3, 224, 224)
        logger.log_image('test_image', image, step=0)
        # Multiple images
        images = torch.rand(4, 3, 64, 64)
        logger.log_images('test_images', images, step=0)
        print(f"   [OK] Images logged")

        # Test 4: Log histograms
        print("\n4. Logging histograms:")
        weights = torch.randn(1000)
        logger.log_histogram('weights/layer1', weights, step=0)
        print(f"   [OK] Histogram logged")

        # Test 5: Log model parameters
        print("\n5. Logging model parameters:")
        model = torch.nn.Linear(10, 5)
        # Create dummy gradients
        output = model(torch.randn(2, 10))
        loss = output.sum()
        loss.backward()
        logger.log_model_parameters(model, step=0)
        print(f"   [OK] Model parameters logged")

        # Test 6: Log text
        print("\n6. Logging text:")
        logger.log_text('status', 'Training started', step=0)
        print(f"   [OK] Text logged")

        # Test 7: Flush and close
        print("\n7. Flushing and closing logger:")
        logger.flush()
        logger.close()
        print(f"   [OK] Logger closed")

        # Test 8: Verify metadata file
        print("\n8. Verifying metadata file:")
        metadata_path = os.path.join(temp_dir, 'test_run', 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   [OK] Metadata saved")
            print(f"   Experiment: {metadata['experiment_name']}")
            print(f"   Config: {metadata['config']}")
        else:
            print(f"   [FAIL] Metadata file not found")

        # Test 9: Factory function
        print("\n9. Testing factory function:")
        config_with_logging = {
            'logging': {
                'use_tensorboard': True,
                'use_wandb': False,
            },
            'model': 'vit',
        }
        logger2 = create_logger(
            log_dir=os.path.join(temp_dir, 'test_run2'),
            experiment_name='test_experiment2',
            config=config_with_logging
        )
        logger2.log_scalar('test', 1.0, step=0)
        logger2.close()
        print(f"   [OK] Factory function works")

        print("\n[SUCCESS] All logger tests passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n[Cleanup] Temporary directory removed")
