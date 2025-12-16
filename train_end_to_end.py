"""
Training script for end-to-end food recognition model.

Supports:
- Multi-task learning: segmentation + classification + regression
- Weighted loss combination
- Debug mode for quick testing
- Resume from checkpoint
- TensorBoard logging
- Component-wise metrics tracking
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

from utils.config_loader import load_config
from utils.dataset import get_datasets
from utils.logger import Logger
from utils.metrics import MetricsTracker
from utils.checkpoint import CheckpointManager
from models.end_to_end import EndToEndFoodRecognition


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train end-to-end food recognition model')
    parser.add_argument('--config', type=str, default='configs/end_to_end.yaml',
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
    """Custom collate function for end-to-end model."""
    images = []
    targets = []
    labels = []
    nutrition = []

    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
        labels.append(item['label'])
        nutrition.append(item['nutrition'])

    labels = torch.stack(labels)
    nutrition = torch.stack(nutrition)

    return {
        'images': images,
        'targets': targets,
        'labels': labels,
        'nutrition': nutrition
    }


def compute_normalization_stats(train_loader, device):
    """Compute mean and std for nutrition target normalization."""
    print("\nComputing normalization statistics...")

    all_targets = []

    for batch in tqdm(train_loader, desc="Computing stats"):
        nutrition = batch['nutrition']
        all_targets.append(nutrition.numpy())

    all_targets = np.concatenate(all_targets, axis=0)

    mean = np.mean(all_targets, axis=0)
    std = np.std(all_targets, axis=0)

    print(f"\nNormalization stats computed:")
    labels = ['calories', 'protein', 'carb', 'fat', 'mass']
    for i, label in enumerate(labels):
        print(f"  {label}: mean={mean[i]:.2f}, std={std[i]:.2f}")

    return torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device)


def create_data_loaders(config, debug=False):
    """Create train and validation data loaders."""
    print("\n" + "="*80)
    print("Loading Dataset")
    print("="*80)

    try:
        train_dataset, val_dataset, _ = get_datasets(
            root=config['data']['dataset_root'],
            task='end_to_end'
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


def train_epoch(model, train_loader, optimizer, device, metrics, logger, epoch, config, mean, std):
    """Train for one epoch."""
    model.train()
    metrics.reset()

    log_interval = config['training'].get('log_interval', 10)

    # Loss weights
    seg_weight = config['training'].get('segmentation_weight', 1.0)
    cls_weight = config['training'].get('classification_weight', 1.0)
    reg_weight = config['training'].get('regression_weight', 1.0)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = [img.to(device) for img in batch['images']]
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]
        labels = batch['labels'].to(device)
        nutrition = batch['nutrition'].to(device)

        # Normalize nutrition targets
        nutrition_norm = (nutrition - mean) / std

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, targets, labels, nutrition_norm)

        # Compute weighted total loss
        total_loss = (
            seg_weight * outputs['segmentation_loss'] +
            cls_weight * outputs['classification_loss'] +
            reg_weight * outputs['regression_loss']
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Update metrics
        with torch.no_grad():
            metrics.update('total_loss', total_loss.item())
            metrics.update('seg_loss', outputs['segmentation_loss'].item())
            metrics.update('cls_loss', outputs['classification_loss'].item())
            metrics.update('reg_loss', outputs['regression_loss'].item())

            # Classification accuracy
            if 'class_logits' in outputs:
                _, predicted = outputs['class_logits'].max(1)
                correct = predicted.eq(labels).sum().item()
                acc = correct / labels.size(0)
                metrics.update('cls_acc', acc)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics.get_average('total_loss'):.4f}",
            'seg': f"{metrics.get_average('seg_loss'):.3f}",
            'cls': f"{metrics.get_average('cls_loss'):.3f}",
            'reg': f"{metrics.get_average('reg_loss'):.3f}"
        })

        # Log to tensorboard
        if (batch_idx + 1) % log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            logger.log_scalar('train/total_loss', total_loss.item(), global_step)
            logger.log_scalar('train/seg_loss', outputs['segmentation_loss'].item(), global_step)
            logger.log_scalar('train/cls_loss', outputs['classification_loss'].item(), global_step)
            logger.log_scalar('train/reg_loss', outputs['regression_loss'].item(), global_step)
            if 'cls_acc' in metrics.metrics:
                logger.log_scalar('train/cls_acc', metrics.get_average('cls_acc'), global_step)

    return metrics.get_averages()


def validate(model, val_loader, device, metrics, logger, epoch, mean, std):
    """Validate the model."""
    model.eval()
    metrics.reset()

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            images = [img.to(device) for img in batch['images']]
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]
            labels = batch['labels'].to(device)
            nutrition = batch['nutrition'].to(device)

            # Normalize nutrition targets
            nutrition_norm = (nutrition - mean) / std

            # Forward pass
            outputs = model(images, targets, labels, nutrition_norm)

            # Compute total loss (unweighted for validation)
            total_loss = (
                outputs['segmentation_loss'] +
                outputs['classification_loss'] +
                outputs['regression_loss']
            )

            # Update metrics
            metrics.update('total_loss', total_loss.item())
            metrics.update('seg_loss', outputs['segmentation_loss'].item())
            metrics.update('cls_loss', outputs['classification_loss'].item())
            metrics.update('reg_loss', outputs['regression_loss'].item())

            # Classification accuracy
            if 'class_logits' in outputs:
                _, predicted = outputs['class_logits'].max(1)
                correct = predicted.eq(labels).sum().item()
                acc = correct / labels.size(0)
                metrics.update('cls_acc', acc)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics.get_average('total_loss'):.4f}",
                'seg': f"{metrics.get_average('seg_loss'):.3f}",
                'cls': f"{metrics.get_average('cls_loss'):.3f}",
                'reg': f"{metrics.get_average('reg_loss'):.3f}"
            })

    # Log validation metrics
    val_metrics = metrics.get_averages()
    logger.log_scalar('val/total_loss', val_metrics['total_loss'], epoch)
    logger.log_scalar('val/seg_loss', val_metrics['seg_loss'], epoch)
    logger.log_scalar('val/cls_loss', val_metrics['cls_loss'], epoch)
    logger.log_scalar('val/reg_loss', val_metrics['reg_loss'], epoch)
    if 'cls_acc' in val_metrics:
        logger.log_scalar('val/cls_acc', val_metrics['cls_acc'], epoch)

    return val_metrics


def main():
    """Main training function."""
    args = parse_args()

    # Print banner
    print("\n" + "="*80)
    if args.debug:
        print(" "*25 + "DEBUG MODE ENABLED")
        print(" "*15 + "Using 10 samples, 1 epoch, batch_size=1")
    else:
        print(" "*20 + "END-TO-END FOOD RECOGNITION TRAINING")
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
        config['training']['batch_size'] = 1
        config['training']['num_epochs'] = 1
        config['training']['log_interval'] = 1
        config['data']['num_workers'] = 0

    # Print config summary
    print("\nConfiguration Summary:")
    print(f"  Model: End-to-End (Segmentation + Classification + Regression)")
    print(f"  Num classes: {config['model']['num_classes']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Num epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['optimizer']['lr']}")
    print(f"  Optimizer: {config['optimizer']['name']}")
    print(f"  Loss weights:")
    print(f"    Segmentation: {config['training'].get('segmentation_weight', 1.0)}")
    print(f"    Classification: {config['training'].get('classification_weight', 1.0)}")
    print(f"    Regression: {config['training'].get('regression_weight', 1.0)}")
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

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, args.debug)

    # Compute normalization statistics
    mean, std = compute_normalization_stats(train_loader, device)

    # Create model
    print("\n" + "="*80)
    print("Creating Model")
    print("="*80)

    model = EndToEndFoodRecognition(
        num_classes=config['model']['num_classes'],
        num_regression_outputs=5,
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created: End-to-End Food Recognition")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Setup optimizer (SGD recommended for detection-based models)
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
    exp_name = "end_to_end"
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
            mean = checkpoint.get('normalization_mean', mean)
            std = checkpoint.get('normalization_std', std)
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
                device, train_metrics, logger, epoch, config, mean, std
            )

            # Validate
            val_results = validate(
                model, val_loader, device,
                val_metrics_tracker, logger, epoch, mean, std
            )

            # Update scheduler
            if scheduler:
                scheduler.step()
                logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"  Train - Total: {train_results['total_loss']:.4f}, "
                  f"Seg: {train_results['seg_loss']:.3f}, "
                  f"Cls: {train_results['cls_loss']:.3f}, "
                  f"Reg: {train_results['reg_loss']:.3f}")
            if 'cls_acc' in train_results:
                print(f"          Cls Acc: {train_results['cls_acc']:.4f}")

            print(f"  Val   - Total: {val_results['total_loss']:.4f}, "
                  f"Seg: {val_results['seg_loss']:.3f}, "
                  f"Cls: {val_results['cls_loss']:.3f}, "
                  f"Reg: {val_results['reg_loss']:.3f}")
            if 'cls_acc' in val_results:
                print(f"          Cls Acc: {val_results['cls_acc']:.4f}")

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
                    'val_seg_loss': val_results['seg_loss'],
                    'val_cls_loss': val_results['cls_loss'],
                    'val_reg_loss': val_results['reg_loss'],
                    'best_val_loss': best_val_loss,
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
