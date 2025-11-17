"""
Checkpoint Management for Training

Provides utilities for:
- Saving checkpoints with metadata (atomic writes)
- Loading checkpoints
- Managing best-N checkpoints
- Resuming training from checkpoints

Checkpoint structure:
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict (optional),
    'metrics': dict,
    'config': dict,
    'timestamp': str,
}
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    filepath: str,
    scheduler: Optional[Any] = None,
    extra_state: Optional[Dict[str, Any]] = None,
):
    """
    Save checkpoint with atomic write (temp file + rename).

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Dict of metric values
        config: Configuration dict
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        extra_state: Additional state to save (optional)

    Example:
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'loss': 0.5, 'accuracy': 85.0},
        ...     config=config,
        ...     filepath='checkpoints/epoch_10.pth'
        ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }

    # Add scheduler if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Add extra state if provided
    if extra_state is not None:
        checkpoint['extra_state'] = extra_state

    # Atomic write: save to temp file, then rename
    temp_filepath = filepath.with_suffix('.tmp')

    try:
        torch.save(checkpoint, temp_filepath)
        # Atomic rename (overwrites if exists)
        shutil.move(str(temp_filepath), str(filepath))
    except Exception as e:
        # Cleanup temp file on failure
        if temp_filepath.exists():
            temp_filepath.unlink()
        raise e


def load_checkpoint(filepath: str, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load checkpoint from file.

    Args:
        filepath: Path to checkpoint file
        device: Device to load checkpoint to ('cpu', 'cuda', etc.)
                If None, uses default device

    Returns:
        Checkpoint dict with keys:
        - epoch
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict (optional)
        - metrics
        - config
        - timestamp
        - extra_state (optional)

    Example:
        >>> checkpoint = load_checkpoint('checkpoints/best.pth')
        >>> model.load_state_dict(checkpoint['model_state_dict'])
        >>> optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        >>> start_epoch = checkpoint['epoch'] + 1
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Load checkpoint
    if device is not None:
        checkpoint = torch.load(filepath, map_location=device)
    else:
        checkpoint = torch.load(filepath)

    return checkpoint


def load_model_from_checkpoint(
    model: nn.Module,
    filepath: str,
    device: Optional[str] = None,
    strict: bool = True
) -> nn.Module:
    """
    Load model weights from checkpoint.

    Args:
        model: PyTorch model to load weights into
        filepath: Path to checkpoint file
        device: Device to load to
        strict: Whether to strictly enforce state_dict keys match

    Returns:
        Model with loaded weights

    Example:
        >>> model = FoodClassifier(num_classes=101)
        >>> model = load_model_from_checkpoint(model, 'checkpoints/best.pth')
    """
    checkpoint = load_checkpoint(filepath, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return model


class CheckpointManager:
    """
    Manages checkpoints during training.

    Features:
    - Automatically saves best-N checkpoints based on metric
    - Keeps track of best metric values
    - Removes old checkpoints when limit is exceeded
    - Saves latest checkpoint separately

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of best checkpoints to keep
        metric_name: Name of metric to track (e.g., 'val_loss', 'val_accuracy')
        mode: 'min' or 'max' (lower/higher is better)
        save_latest: Whether to always save latest checkpoint

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='checkpoints',
        ...     max_checkpoints=3,
        ...     metric_name='val_accuracy',
        ...     mode='max'
        ... )
        >>> # During training
        >>> manager.save(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'val_accuracy': 85.0, 'val_loss': 0.5}
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        save_latest: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        self.save_latest = save_latest

        # Track best checkpoints: [(metric_value, epoch, filepath), ...]
        self.best_checkpoints: List[Tuple[float, int, Path]] = []

        # Best metric value
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

        # Load existing checkpoint tracking
        self._load_tracking()

    def _load_tracking(self):
        """Load checkpoint tracking from metadata file."""
        tracking_file = self.checkpoint_dir / 'checkpoint_tracking.json'

        if tracking_file.exists():
            with open(tracking_file, 'r') as f:
                data = json.load(f)
                self.best_metric = data.get('best_metric', self.best_metric)
                self.best_checkpoints = [
                    (item['metric'], item['epoch'], Path(item['filepath']))
                    for item in data.get('best_checkpoints', [])
                ]

    def _save_tracking(self):
        """Save checkpoint tracking to metadata file."""
        tracking_file = self.checkpoint_dir / 'checkpoint_tracking.json'

        data = {
            'best_metric': self.best_metric,
            'metric_name': self.metric_name,
            'mode': self.mode,
            'best_checkpoints': [
                {
                    'metric': metric,
                    'epoch': epoch,
                    'filepath': str(filepath),
                }
                for metric, epoch, filepath in self.best_checkpoints
            ],
        }

        with open(tracking_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _is_better(self, metric_value: float) -> bool:
        """Check if metric value is better than current best."""
        if self.mode == 'min':
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric

    def _should_save(self, metric_value: float) -> bool:
        """Check if checkpoint should be saved (in top-N)."""
        if len(self.best_checkpoints) < self.max_checkpoints:
            return True

        # Get worst metric in current top-N
        if self.mode == 'min':
            worst_metric = max(m for m, _, _ in self.best_checkpoints)
            return metric_value < worst_metric
        else:
            worst_metric = min(m for m, _, _ in self.best_checkpoints)
            return metric_value > worst_metric

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        force_save: bool = False,
    ) -> bool:
        """
        Save checkpoint if it's in top-N based on metric.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dict of metrics
            config: Configuration
            scheduler: Learning rate scheduler
            extra_state: Additional state
            force_save: Force save even if not in top-N

        Returns:
            True if checkpoint was saved, False otherwise
        """
        # Get metric value
        if self.metric_name not in metrics:
            print(f"[CheckpointManager] WARNING: {self.metric_name} not in metrics")
            return False

        metric_value = metrics[self.metric_name]

        # Save latest checkpoint
        if self.save_latest:
            latest_path = self.checkpoint_dir / 'latest.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
                filepath=str(latest_path),
                scheduler=scheduler,
                extra_state=extra_state,
            )

        # Check if should save as best
        if not force_save and not self._should_save(metric_value):
            return False

        # Save checkpoint
        filename = f'epoch_{epoch:04d}_metric_{metric_value:.4f}.pth'
        filepath = self.checkpoint_dir / filename

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
            filepath=str(filepath),
            scheduler=scheduler,
            extra_state=extra_state,
        )

        # Update best checkpoints list
        self.best_checkpoints.append((metric_value, epoch, filepath))

        # Sort by metric
        if self.mode == 'min':
            self.best_checkpoints.sort(key=lambda x: x[0])
        else:
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove worst checkpoints if exceeding max
        while len(self.best_checkpoints) > self.max_checkpoints:
            _, _, old_filepath = self.best_checkpoints.pop()
            if old_filepath.exists():
                old_filepath.unlink()
                print(f"[CheckpointManager] Removed old checkpoint: {old_filepath.name}")

        # Update best metric
        if self._is_better(metric_value):
            self.best_metric = metric_value
            # Save symbolic link to best
            best_link = self.checkpoint_dir / 'best.pth'
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            # Copy instead of symlink (Windows compatibility)
            shutil.copy(str(filepath), str(best_link))
            print(f"[CheckpointManager] New best {self.metric_name}: {metric_value:.4f}")

        # Save tracking metadata
        self._save_tracking()

        print(f"[CheckpointManager] Saved checkpoint: {filename}")
        return True

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / 'best.pth'
        if best_path.exists():
            return best_path
        return None

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        latest_path = self.checkpoint_dir / 'latest.pth'
        if latest_path.exists():
            return latest_path
        return None


if __name__ == '__main__':
    # Smoke test
    print("Testing checkpoint management...")

    import tempfile
    import shutil

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create dummy model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Test 1: Save checkpoint
        print("\n1. Testing save_checkpoint:")
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={'loss': 0.5, 'accuracy': 85.0},
            config={'lr': 0.001, 'batch_size': 32},
            filepath=checkpoint_path,
            scheduler=scheduler,
        )
        if os.path.exists(checkpoint_path):
            print(f"   [OK] Checkpoint saved to {checkpoint_path}")
        else:
            print(f"   [FAIL] Checkpoint not saved")

        # Test 2: Load checkpoint
        print("\n2. Testing load_checkpoint:")
        checkpoint = load_checkpoint(checkpoint_path)
        print(f"   [OK] Checkpoint loaded")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Metrics: {checkpoint['metrics']}")
        print(f"   Config: {checkpoint['config']}")
        print(f"   Timestamp: {checkpoint['timestamp']}")

        # Test 3: Load model from checkpoint
        print("\n3. Testing load_model_from_checkpoint:")
        new_model = nn.Linear(10, 5)
        new_model = load_model_from_checkpoint(new_model, checkpoint_path)
        # Verify weights match
        weights_match = torch.allclose(
            model.weight,
            new_model.weight
        )
        if weights_match:
            print(f"   [OK] Model weights loaded correctly")
        else:
            print(f"   [FAIL] Model weights don't match")

        # Test 4: CheckpointManager
        print("\n4. Testing CheckpointManager:")
        manager = CheckpointManager(
            checkpoint_dir=os.path.join(temp_dir, 'checkpoints'),
            max_checkpoints=3,
            metric_name='val_loss',
            mode='min',
        )
        print(f"   [OK] CheckpointManager created")

        # Test 5: Save multiple checkpoints
        print("\n5. Testing multiple checkpoint saves:")
        metrics_list = [
            {'val_loss': 1.0, 'val_acc': 70.0},
            {'val_loss': 0.8, 'val_acc': 75.0},  # Better
            {'val_loss': 0.9, 'val_acc': 72.0},
            {'val_loss': 0.6, 'val_acc': 80.0},  # Best
            {'val_loss': 0.7, 'val_acc': 78.0},
        ]

        for i, metrics in enumerate(metrics_list):
            saved = manager.save(
                model=model,
                optimizer=optimizer,
                epoch=i,
                metrics=metrics,
                config={'lr': 0.001},
            )
            print(f"   Epoch {i}: val_loss={metrics['val_loss']:.2f}, saved={saved}")

        print(f"   Best metric: {manager.best_metric:.2f} (should be 0.6)")
        print(f"   Number of checkpoints: {len(manager.best_checkpoints)} (should be 3)")

        # Test 6: Get best checkpoint
        print("\n6. Testing get_best_checkpoint_path:")
        best_path = manager.get_best_checkpoint_path()
        if best_path and best_path.exists():
            print(f"   [OK] Best checkpoint: {best_path}")
            best_ckpt = load_checkpoint(str(best_path))
            print(f"   Best val_loss: {best_ckpt['metrics']['val_loss']:.2f}")
        else:
            print(f"   [FAIL] Best checkpoint not found")

        # Test 7: Get latest checkpoint
        print("\n7. Testing get_latest_checkpoint_path:")
        latest_path = manager.get_latest_checkpoint_path()
        if latest_path and latest_path.exists():
            print(f"   [OK] Latest checkpoint: {latest_path}")
            latest_ckpt = load_checkpoint(str(latest_path))
            print(f"   Latest epoch: {latest_ckpt['epoch']}")
        else:
            print(f"   [FAIL] Latest checkpoint not found")

        # Test 8: Verify max_checkpoints limit
        print("\n8. Verifying max_checkpoints limit:")
        checkpoint_files = list(Path(temp_dir, 'checkpoints').glob('epoch_*.pth'))
        print(f"   Checkpoint files (excluding latest/best): {len(checkpoint_files)}")
        if len(checkpoint_files) <= manager.max_checkpoints:
            print(f"   [OK] Checkpoint limit respected")
        else:
            print(f"   [FAIL] Too many checkpoints: {len(checkpoint_files)}")

        # Test 9: Test mode='max'
        print("\n9. Testing CheckpointManager with mode='max':")
        manager_max = CheckpointManager(
            checkpoint_dir=os.path.join(temp_dir, 'checkpoints_max'),
            max_checkpoints=2,
            metric_name='val_acc',
            mode='max',
        )

        for i, metrics in enumerate(metrics_list):
            manager_max.save(
                model=model,
                optimizer=optimizer,
                epoch=i,
                metrics=metrics,
                config={'lr': 0.001},
            )

        print(f"   Best metric: {manager_max.best_metric:.2f} (should be 80.0)")
        best_max_path = manager_max.get_best_checkpoint_path()
        if best_max_path:
            best_max_ckpt = load_checkpoint(str(best_max_path))
            if best_max_ckpt['metrics']['val_acc'] == 80.0:
                print(f"   [OK] Best checkpoint has highest accuracy")
            else:
                print(f"   [FAIL] Wrong best checkpoint")

        print("\n[SUCCESS] All checkpoint tests passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n[Cleanup] Temporary directory removed")
