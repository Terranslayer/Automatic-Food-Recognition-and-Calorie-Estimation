"""
Training script for food instance segmentation model.

Supports:
- Mask R-CNN architecture
- Debug mode for quick testing
- Resume from checkpoint
- TensorBoard logging
- Multi-loss tracking (classification, bbox, mask, objectness)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.config_loader import load_config
from utils.dataset import get_datasets
from utils.logger import Logger
from utils.metrics import MetricsTracker
from utils.checkpoint import CheckpointManager
from models.segmentation import FoodSegmentation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train food segmentation model')
    parser.add_argument('--config', type=str, default='configs/mask_rcnn.yaml',
                        help='Path to config file')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (10 samples, 1 epoch)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detect if not specified')
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        torch.use_deterministic_algorithms(True, warn_only=True)


def collate_fn(batch):
    """Custom collate function for detection/segmentation."""
    images = []
    targets = []

    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])

    return images, targets


def create_data_loaders(config, debug=False):
    """Create train and validation data loaders."""
    print("\n" + "="*80)
    print("Loading Dataset")
    print("="*80)

    try:
        train_dataset, val_dataset, _ = get_datasets(
            root=config['data']['dataset_root'],
            task='segmentation'
        )

        print(f"Dataset loaded successfully:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        # Debug mode: use only 10 samples
        if debug:
            train_dataset = Subset(train_dataset, list(range(min(10, len(train_dataset)))))
            val_dataset = Subset(val_dataset, list(range(min(10, len(val_dataset)))))
            print(f"\nDEBUG MODE: Reduced to {len(train_dataset)} train, {len(val_dataset)} val samples")

        # Create data loaders
        batch_size = config['training']['batch_size']
        num_workers = config['data'].get('num_workers', 4)

        # Debug mode overrides
        if debug:
            num_workers = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=config['data'].get('pin_memory', True)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=config['data'].get('pin_memory', True)
        )

        print(f"\nDataLoaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")

        return train_loader, val_loader

    except Exception as e:
        print(f"\nERROR: Failed to load dataset: {e}")
        print(f"Please ensure dataset exists at: {config['data']['dataset_root']}")
        sys.exit(1)


def train_epoch(model, train_loader, optimizer, device, metrics, logger, epoch, config):
    """Train for one epoch."""
    model.train()
    metrics.reset()

    log_interval = config['training'].get('log_interval', 10)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass - model returns loss dict in training mode
        optimizer.zero_grad()
        loss_dict = model(images, targets)

        # Sum all losses
        total_loss = sum(loss for loss in loss_dict.values())

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Update metrics
        with torch.no_grad():
            metrics.update('total_loss', total_loss.item())
            for k, v in loss_dict.items():
                metrics.update(k, v.item())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics.get_average('total_loss'):.4f}",
            'cls': f"{metrics.get_average('loss_classifier'):.3f}" if 'loss_classifier' in loss_dict else "N/A",
            'box': f"{metrics.get_average('loss_box_reg'):.3f}" if 'loss_box_reg' in loss_dict else "N/A",
            'mask': f"{metrics.get_average('loss_mask'):.3f}" if 'loss_mask' in loss_dict else "N/A"
        })

        # Log to tensorboard
        if (batch_idx + 1) % log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            logger.log_scalar('train/total_loss', total_loss.item(), global_step)
            for k, v in loss_dict.items():
                logger.log_scalar(f'train/{k}', v.item(), global_step)

    return metrics.get_averages()


def validate(model, val_loader, device, metrics, logger, epoch):
    """Validate the model."""
    model.train()  # Keep in train mode to get losses
    metrics.reset()

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for images, targets in pbar:
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            # Update metrics
            metrics.update('total_loss', total_loss.item())
            for k, v in loss_dict.items():
                metrics.update(k, v.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics.get_average('total_loss'):.4f}",
                'cls': f"{metrics.get_average('loss_classifier'):.3f}" if 'loss_classifier' in loss_dict else "N/A",
                'box': f"{metrics.get_average('loss_box_reg'):.3f}" if 'loss_box_reg' in loss_dict else "N/A",
                'mask': f"{metrics.get_average('loss_mask'):.3f}" if 'loss_mask' in loss_dict else "N/A"
            })

    # Log validation metrics
    val_metrics = metrics.get_averages()
    logger.log_scalar('val/total_loss', val_metrics['total_loss'], epoch)
    for k, v in val_metrics.items():
        if k != 'total_loss':
            logger.log_scalar(f'val/{k}', v, epoch)

    return val_metrics


def main():
    """Main training function."""
    args = parse_args()

    # Print banner
    print("\n" + "="*80)
    if args.debug:
        print(" "*25 + "DEBUG MODE ENABLED")
        print(" "*15 + "Using 10 samples, 1 epoch, batch_size=2")
    else:
        print(" "*20 + "FOOD SEGMENTATION TRAINING")
    print("="*80)

    # Load config
    try:
        config = load_config(args.config)
        print(f"\nConfig loaded from: {args.config}")
    except Exception as e:
        print(f"\nERROR: Failed to load config: {e}")
        sys.exit(1)

    # Debug mode overrides
    if args.debug:
        config['training']['batch_size'] = 2
        config['training']['num_epochs'] = 1
        config['training']['log_interval'] = 1
        config['data']['num_workers'] = 0

    # Print config summary
    print("\nConfiguration Summary:")
    print(f"  Model: Mask R-CNN")
    print(f"  Backbone: {config['model'].get('backbone', 'resnet50')}")
    print(f"  Num classes: {config['model']['num_classes']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Num epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['optimizer']['lr']}")
    print(f"  Optimizer: {config['optimizer']['name']}")
    print(f"  Random seed: {config['experiment']['random_seed']}")
    print(f"  Deterministic: {config['experiment']['deterministic']}")

    # Set random seed
    set_random_seed(
        config['experiment']['random_seed'],
        config['experiment']['deterministic']
    )

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Limit GPU memory usage to 75%
    if device.type == 'cuda':
        gpu_memory_fraction = config.get('training', {}).get('gpu_memory_fraction', 0.75)
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device_index)
        print(f"GPU memory limited to: {gpu_memory_fraction * 100:.0f}%")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, args.debug)

    # Create model
    print("\n" + "="*80)
    print("Creating Model")
    print("="*80)

    model = FoodSegmentation(
        num_classes=config['model']['num_classes'],
        backbone=config['model'].get('backbone', 'resnet50'),
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created: Mask R-CNN with {config['model'].get('backbone', 'resnet50')}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Setup optimizer (SGD recommended for detection)
    optimizer_name = config['optimizer']['name'].lower()
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['optimizer'].get('weight_decay', 0.0005)
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer'].get('weight_decay', 0)
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer'].get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"\nOptimizer: {optimizer_name}")

    # Setup scheduler
    scheduler = None
    if 'scheduler' in config and config['scheduler']['name']:
        scheduler_name = config['scheduler']['name'].lower()
        if scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['scheduler'].get('step_size', 5),
                gamma=config['scheduler'].get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs']
            )
        print(f"Scheduler: {scheduler_name}")

    # Setup logger
    exp_name = "maskrcnn_segmentation"
    if args.debug:
        exp_name += "_debug"

    log_dir = Path(config['experiment'].get('log_dir', './experiments')) / exp_name
    logger = Logger(log_dir, experiment_name=exp_name, use_tensorboard=config['logging'].get('use_tensorboard', True))
    print(f"\nLogging to: {log_dir}")

    # Setup checkpoint manager
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=5
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"\n" + "="*80)
        print(f"Resuming from checkpoint: {args.resume}")
        print("="*80)
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint: {e}")
            print("Starting training from scratch")

    # Setup metrics tracker
    train_metrics = MetricsTracker()
    val_metrics_tracker = MetricsTracker()

    # Early stopping
    patience = config['training'].get('early_stopping_patience', 10)
    patience_counter = 0

    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    print(f"Training from epoch {start_epoch} to {config['training']['num_epochs']}\n")

    start_time = time.time()

    try:
        for epoch in range(start_epoch, config['training']['num_epochs']):
            epoch_start_time = time.time()

            # Train
            train_results = train_epoch(
                model, train_loader, optimizer,
                device, train_metrics, logger, epoch, config
            )

            # Validate
            val_results = validate(
                model, val_loader, device,
                val_metrics_tracker, logger, epoch
            )

            # Update scheduler
            if scheduler:
                scheduler.step()
                logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"  Train - Total Loss: {train_results['total_loss']:.4f}")
            if 'loss_classifier' in train_results:
                print(f"          Classifier: {train_results['loss_classifier']:.4f}, "
                      f"BBox: {train_results.get('loss_box_reg', 0):.4f}, "
                      f"Mask: {train_results.get('loss_mask', 0):.4f}")
            print(f"  Val   - Total Loss: {val_results['total_loss']:.4f}")
            if 'loss_classifier' in val_results:
                print(f"          Classifier: {val_results['loss_classifier']:.4f}, "
                      f"BBox: {val_results.get('loss_box_reg', 0):.4f}, "
                      f"Mask: {val_results.get('loss_mask', 0):.4f}")

            # Save checkpoint
            is_best = val_results['total_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_results['total_loss']
                patience_counter = 0
                print(f"  >>> New best model! Val loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1

            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    'train_loss': train_results['total_loss'],
                    'val_loss': val_results['total_loss'],
                    'best_val_loss': best_val_loss
                },
                config=config,
                force_save=is_best
            )

            # Early stopping check
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break

            print()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean shutdown
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("Training Completed")
        print("="*80)
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print(f"Logs saved to: {log_dir}")

        logger.close()


if __name__ == '__main__':
    main()
