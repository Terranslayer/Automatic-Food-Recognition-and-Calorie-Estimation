"""
Resource Estimation Script for Training

Estimates computational requirements for training:
- GPU memory usage (model params + activations + gradients + optimizer)
- Training time per epoch
- Total training time
- Storage requirements for checkpoints

Supports models:
- classifier (EfficientNet/ViT)
- segmentation (Mask R-CNN)
- regression (Calorie regressor)
- end_to_end (Full pipeline)

Usage:
    python scripts/estimate_resources.py --model classifier --config configs/efficientnet.yaml
    python scripts/estimate_resources.py --model all
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config
from models.classifier import FoodClassifier
from models.segmentation import FoodSegmentation
from models.calorie_regressor import CalorieRegressor
from models.end_to_end import EndToEndFoodRecognition


def get_model_memory_mb(model: nn.Module) -> float:
    """
    Calculate model parameter memory in MB.

    Args:
        model: PyTorch model

    Returns:
        Memory in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024 ** 2)

    return total_size_mb


def estimate_activation_memory_mb(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int,
    device: str = 'cpu'
) -> float:
    """
    Estimate activation memory for forward pass.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch)
        batch_size: Batch size
        device: Device to run on

    Returns:
        Estimated activation memory in MB
    """
    model = model.to(device)
    model.train()

    # Create dummy input
    if len(input_shape) == 1:
        # Feature input (for regression)
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
    else:
        # Image input (for other models)
        dummy_input = torch.randn(batch_size, *input_shape).to(device)

    # Clear cache
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)

    # Forward pass
    try:
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"   [WARNING] Forward pass failed: {e}")
        return 0.0

    # Measure memory
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        activation_memory = peak_memory - initial_memory
    else:
        # Rough estimate for CPU (activation memory ~ model size * 2)
        activation_memory = get_model_memory_mb(model) * 2

    return activation_memory


def estimate_gradient_memory_mb(model: nn.Module) -> float:
    """
    Estimate gradient memory (same as parameter memory).

    Args:
        model: PyTorch model

    Returns:
        Gradient memory in MB
    """
    return get_model_memory_mb(model)


def estimate_optimizer_memory_mb(
    model: nn.Module,
    optimizer_name: str = 'adam'
) -> float:
    """
    Estimate optimizer state memory.

    Args:
        model: PyTorch model
        optimizer_name: Optimizer name ('adam', 'sgd', 'adamw')

    Returns:
        Optimizer memory in MB
    """
    param_memory = get_model_memory_mb(model)

    # Optimizer state memory multipliers
    optimizer_multipliers = {
        'sgd': 1.0,  # Only momentum (if used)
        'adam': 2.0,  # exp_avg + exp_avg_sq
        'adamw': 2.0,  # exp_avg + exp_avg_sq
    }

    multiplier = optimizer_multipliers.get(optimizer_name.lower(), 2.0)
    return param_memory * multiplier


def estimate_total_gpu_memory_mb(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int,
    optimizer_name: str = 'adam',
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Estimate total GPU memory requirements.

    Args:
        model: PyTorch model
        input_shape: Input shape (without batch)
        batch_size: Batch size
        optimizer_name: Optimizer name
        device: Device

    Returns:
        Dict with memory breakdown
    """
    model_memory = get_model_memory_mb(model)
    activation_memory = estimate_activation_memory_mb(model, input_shape, batch_size, device)
    gradient_memory = estimate_gradient_memory_mb(model)
    optimizer_memory = estimate_optimizer_memory_mb(model, optimizer_name)

    total_memory = (
        model_memory +
        activation_memory +
        gradient_memory +
        optimizer_memory
    )

    return {
        'model_mb': model_memory,
        'activations_mb': activation_memory,
        'gradients_mb': gradient_memory,
        'optimizer_mb': optimizer_memory,
        'total_mb': total_memory,
        'total_gb': total_memory / 1024,
    }


def estimate_training_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int,
    num_samples: int,
    num_epochs: int,
    device: str = 'cpu',
    num_warmup: int = 3,
    num_measure: int = 10
) -> Dict[str, float]:
    """
    Estimate training time by measuring time per batch.

    Args:
        model: PyTorch model
        input_shape: Input shape
        batch_size: Batch size
        num_samples: Total number of training samples
        num_epochs: Number of epochs
        device: Device
        num_warmup: Number of warmup iterations
        num_measure: Number of iterations to measure

    Returns:
        Dict with time estimates
    """
    model = model.to(device)
    model.train()

    # Create dummy dataset
    dummy_inputs = torch.randn(batch_size, *input_shape).to(device)

    # Create appropriate dummy targets based on input shape
    if len(input_shape) == 1:
        # Regression model - continuous targets
        dummy_targets = torch.randn(batch_size, 5).to(device)
    else:
        # Classification model - class labels
        dummy_targets = torch.randint(0, 10, (batch_size,)).to(device)

    # Create dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Select appropriate loss function
    if len(input_shape) == 1:
        criterion = nn.MSELoss()  # Regression
    else:
        criterion = nn.CrossEntropyLoss()  # Classification

    # Warmup iterations
    for _ in range(num_warmup):
        optimizer.zero_grad()
        try:
            outputs = model(dummy_inputs)
            # Handle different output types
            if isinstance(outputs, dict):
                loss = sum(outputs.values())  # For detection models
            elif len(input_shape) == 1:
                loss = criterion(outputs, dummy_targets)  # Regression
            else:
                loss = criterion(outputs, dummy_targets)  # Classification
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"   [WARNING] Training iteration failed: {e}")
            return {
                'seconds_per_batch': 0.0,
                'seconds_per_epoch': 0.0,
                'total_hours': 0.0,
            }

    # Measure time per batch
    start_time = time.time()

    for _ in range(num_measure):
        optimizer.zero_grad()
        try:
            outputs = model(dummy_inputs)
            if isinstance(outputs, dict):
                loss = sum(outputs.values())
            elif len(input_shape) == 1:
                loss = criterion(outputs, dummy_targets)
            else:
                loss = criterion(outputs, dummy_targets)
            loss.backward()
            optimizer.step()
        except Exception:
            break

    end_time = time.time()

    # Calculate statistics
    seconds_per_batch = (end_time - start_time) / num_measure
    batches_per_epoch = num_samples // batch_size
    seconds_per_epoch = seconds_per_batch * batches_per_epoch
    total_hours = (seconds_per_epoch * num_epochs) / 3600

    return {
        'seconds_per_batch': seconds_per_batch,
        'seconds_per_epoch': seconds_per_epoch,
        'hours_per_epoch': seconds_per_epoch / 3600,
        'total_hours': total_hours,
        'total_days': total_hours / 24,
    }


def estimate_checkpoint_storage_mb(
    model: nn.Module,
    num_checkpoints: int = 5
) -> Dict[str, float]:
    """
    Estimate storage requirements for checkpoints.

    Args:
        model: PyTorch model
        num_checkpoints: Number of checkpoints to save

    Returns:
        Dict with storage estimates
    """
    # Checkpoint contains: model + optimizer + metadata
    # Approximate: 3x model size per checkpoint
    model_memory = get_model_memory_mb(model)
    checkpoint_size_mb = model_memory * 3

    total_storage_mb = checkpoint_size_mb * num_checkpoints

    return {
        'checkpoint_size_mb': checkpoint_size_mb,
        'total_storage_mb': total_storage_mb,
        'total_storage_gb': total_storage_mb / 1024,
    }


def estimate_model_resources(
    model_type: str,
    config_path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Estimate resources for a specific model.

    Args:
        model_type: Model type ('classifier', 'segmentation', 'regression', 'end_to_end')
        config_path: Path to config file
        device: Device to use

    Returns:
        Dict with all resource estimates
    """
    # Load config
    config = load_config(config_path)

    # Get training config
    batch_size = config.get('data', {}).get('batch_size', 32)
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    optimizer_name = config.get('optimizer', {}).get('name', 'adam')

    # Model-specific parameters
    if model_type == 'classifier':
        model_config = config.get('model', {})
        model = FoodClassifier(
            backbone=model_config.get('backbone', 'efficientnet_b0'),
            num_classes=model_config.get('num_classes', 101),
            pretrained=model_config.get('pretrained', True),
        )
        input_shape = (3, 224, 224)
        num_samples = 3000  # Approximate

    elif model_type == 'segmentation':
        model = FoodSegmentation(num_classes=91)
        input_shape = (3, 512, 512)
        num_samples = 3000
        batch_size = min(batch_size, 4)  # Mask R-CNN uses smaller batches

    elif model_type == 'regression':
        model_config = config.get('model', {})
        model = CalorieRegressor(
            input_dim=model_config.get('input_dim', 1280),
            output_dim=model_config.get('output_dim', 5),
        )
        input_shape = (1280,)  # Feature input, not image
        num_samples = 3000

    elif model_type == 'end_to_end':
        model = EndToEndFoodRecognition(num_classes=91)
        input_shape = (3, 512, 512)
        num_samples = 3000
        batch_size = min(batch_size, 4)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Memory estimates
    memory_estimates = estimate_total_gpu_memory_mb(
        model, input_shape, batch_size, optimizer_name, device
    )

    # Time estimates
    time_estimates = estimate_training_time(
        model, input_shape, batch_size, num_samples, num_epochs, device
    )

    # Storage estimates
    storage_estimates = estimate_checkpoint_storage_mb(model)

    return {
        'model_type': model_type,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'num_samples': num_samples,
        'num_params': num_params,
        'num_trainable': num_trainable,
        'memory': memory_estimates,
        'time': time_estimates,
        'storage': storage_estimates,
    }


def print_resource_report(estimates: Dict[str, Any]):
    """Print formatted resource report."""
    print("\n" + "=" * 70)
    print(f"RESOURCE ESTIMATION: {estimates['model_type'].upper()}")
    print("=" * 70)

    print(f"\n[Model Parameters]")
    print(f"  Total parameters:     {estimates['num_params']:,}")
    print(f"  Trainable parameters: {estimates['num_trainable']:,}")

    print(f"\n[Training Configuration]")
    print(f"  Batch size:    {estimates['batch_size']}")
    print(f"  Num epochs:    {estimates['num_epochs']}")
    print(f"  Num samples:   {estimates['num_samples']}")

    mem = estimates['memory']
    print(f"\n[GPU Memory Requirements]")
    print(f"  Model parameters:  {mem['model_mb']:>8.1f} MB")
    print(f"  Activations:       {mem['activations_mb']:>8.1f} MB")
    print(f"  Gradients:         {mem['gradients_mb']:>8.1f} MB")
    print(f"  Optimizer state:   {mem['optimizer_mb']:>8.1f} MB")
    print(f"  " + "-" * 40)
    print(f"  TOTAL:             {mem['total_mb']:>8.1f} MB ({mem['total_gb']:.2f} GB)")

    time_est = estimates['time']
    if time_est['seconds_per_batch'] > 0:
        print(f"\n[Training Time Estimates]")
        print(f"  Time per batch:    {time_est['seconds_per_batch']:>8.2f} seconds")
        print(f"  Time per epoch:    {time_est['hours_per_epoch']:>8.2f} hours")
        print(f"  Total time:        {time_est['total_hours']:>8.2f} hours ({time_est['total_days']:.1f} days)")
    else:
        print(f"\n[Training Time Estimates]")
        print(f"  [WARNING] Time estimation failed (model may not support standard training)")

    stor = estimates['storage']
    print(f"\n[Storage Requirements]")
    print(f"  Checkpoint size:   {stor['checkpoint_size_mb']:>8.1f} MB")
    print(f"  Total (5 ckpts):   {stor['total_storage_mb']:>8.1f} MB ({stor['total_storage_gb']:.2f} GB)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Estimate training resource requirements')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['classifier', 'segmentation', 'regression', 'end_to_end', 'all'],
        help='Model type to estimate'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for estimation'
    )

    args = parser.parse_args()

    print(f"\n[Resource Estimator]")
    print(f"  Device: {args.device}")
    print(f"  Config: {args.config}")

    if args.device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # Estimate resources
    if args.model == 'all':
        models = ['classifier', 'segmentation', 'regression', 'end_to_end']
    else:
        models = [args.model]

    for model_type in models:
        try:
            estimates = estimate_model_resources(model_type, args.config, args.device)
            print_resource_report(estimates)
        except Exception as e:
            print(f"\n[ERROR] Failed to estimate {model_type}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    # If run without arguments, run smoke test
    if len(sys.argv) == 1:
        print("Running smoke test...")

        # Test with CPU and dummy config
        print("\n[Test 1] Testing classifier estimation:")
        try:
            # Create minimal config
            config = {
                'model': {
                    'backbone': 'efficientnet_b0',
                    'num_classes': 101,
                    'pretrained': False,
                },
                'data': {'batch_size': 16},
                'training': {'num_epochs': 10},
                'optimizer': {'name': 'adam'},
            }

            # Save temp config
            import tempfile
            import yaml
            temp_dir = tempfile.mkdtemp()
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            estimates = estimate_model_resources('classifier', config_path, device='cpu')
            print_resource_report(estimates)

            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            print("\n[SUCCESS] Smoke test passed!")
            print("\nUsage examples:")
            print("  python scripts/estimate_resources.py --model classifier --config configs/efficientnet.yaml")
            print("  python scripts/estimate_resources.py --model segmentation --config configs/mask_rcnn.yaml")
            print("  python scripts/estimate_resources.py --model all --config configs/base.yaml")

        except Exception as e:
            print(f"\n[FAIL] Smoke test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        main()
