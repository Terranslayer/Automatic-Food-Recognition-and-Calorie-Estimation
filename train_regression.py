"""
Training script for calorie regression model.

Supports:
- Multi-output regression (calories, protein, carb, fat, mass)
- ResNet/EfficientNet backbones
- Debug mode for quick testing
- Resume from checkpoint
- TensorBoard logging
- Target normalization
- Per-output metrics tracking
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
from models.calorie_regressor import CalorieRegressor
from models.classifier import FoodClassifier


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train calorie regression model')
    parser.add_argument('--config', type=str, default='configs/regression.yaml',
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


def get_nutrition_targets(batch, device):
    """Extract nutrition targets from batch and stack them into a tensor."""
    targets = torch.stack([
        batch['calories'],
        batch['protein_g'],
        batch['carb_g'],
        batch['fat_g'],
        batch['mass_g']
    ], dim=1).to(device)
    return targets


def compute_normalization_stats(train_loader, device):
    """Compute mean and std for target normalization."""
    print("\nComputing normalization statistics...")

    all_targets = []

    for batch in tqdm(train_loader, desc="Computing stats"):
        targets = get_nutrition_targets(batch, device)
        all_targets.append(targets.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)

    mean = np.mean(all_targets, axis=0)
    std = np.std(all_targets, axis=0)

    # Avoid division by zero - replace 0 std with 1
    std = np.where(std < 1e-6, 1.0, std)

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
            task='regression'
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
            pin_memory=config['data'].get('pin_memory', True),
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
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


def compute_metrics(predictions, targets, mean, std):
    """Compute MAE, RMSE, MAPE for each output."""
    # Denormalize
    predictions = predictions * std + mean
    targets = targets * std + mean

    # Compute metrics per output
    mae = torch.abs(predictions - targets).mean(dim=0)
    rmse = torch.sqrt(((predictions - targets) ** 2).mean(dim=0))

    # MAPE: avoid division by zero
    epsilon = 1e-8
    mape = (torch.abs((targets - predictions) / (targets + epsilon)) * 100).mean(dim=0)

    return {
        'mae': mae.cpu().numpy(),
        'rmse': rmse.cpu().numpy(),
        'mape': mape.cpu().numpy()
    }


def train_epoch(model, feature_extractor, train_loader, criterion, optimizer, device, metrics, logger, epoch, config, mean, std):
    """Train for one epoch."""
    model.train()
    metrics.reset()

    log_interval = config['training'].get('log_interval', 10)
    labels = ['calories', 'protein', 'carb', 'fat', 'mass']

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = get_nutrition_targets(batch, device)

        # Normalize targets
        targets_norm = (targets - mean) / std

        # Extract features
        with torch.no_grad():
            features = feature_extractor.extract_features(images)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets_norm)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics (on denormalized values)
        with torch.no_grad():
            batch_metrics = compute_metrics(outputs, targets_norm, mean, std)

            metrics.update('loss', loss.item())
            for i, label in enumerate(labels):
                metrics.update(f'mae_{label}', batch_metrics['mae'][i])
                metrics.update(f'rmse_{label}', batch_metrics['rmse'][i])
                metrics.update(f'mape_{label}', batch_metrics['mape'][i])

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics.get_average('loss'):.4f}",
            'cal_mae': f"{metrics.get_average('mae_calories'):.2f}",
            'prot_mae': f"{metrics.get_average('mae_protein'):.2f}"
        })

        # Log to tensorboard
        if (batch_idx + 1) % log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            logger.log_scalar('train/loss', loss.item(), global_step)
            for i, label in enumerate(labels):
                logger.log_scalar(f'train/mae_{label}', batch_metrics['mae'][i], global_step)

    return metrics.get_averages()


def validate(model, feature_extractor, val_loader, criterion, device, metrics, logger, epoch, mean, std):
    """Validate the model."""
    model.eval()
    metrics.reset()

    labels = ['calories', 'protein', 'carb', 'fat', 'mass']

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            targets = get_nutrition_targets(batch, device)

            # Normalize targets
            targets_norm = (targets - mean) / std

            # Extract features
            features = feature_extractor.extract_features(images)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets_norm)

            # Compute metrics
            batch_metrics = compute_metrics(outputs, targets_norm, mean, std)

            metrics.update('loss', loss.item())
            for i, label in enumerate(labels):
                metrics.update(f'mae_{label}', batch_metrics['mae'][i])
                metrics.update(f'rmse_{label}', batch_metrics['rmse'][i])
                metrics.update(f'mape_{label}', batch_metrics['mape'][i])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics.get_average('loss'):.4f}",
                'cal_mae': f"{metrics.get_average('mae_calories'):.2f}",
                'prot_mae': f"{metrics.get_average('mae_protein'):.2f}"
            })

    # Log validation metrics
    val_metrics = metrics.get_averages()
    logger.log_scalar('val/loss', val_metrics['loss'], epoch)
    for label in labels:
        logger.log_scalar(f'val/mae_{label}', val_metrics[f'mae_{label}'], epoch)
        logger.log_scalar(f'val/rmse_{label}', val_metrics[f'rmse_{label}'], epoch)
        logger.log_scalar(f'val/mape_{label}', val_metrics[f'mape_{label}'], epoch)

    return val_metrics


def main():
    """Main training function."""
    args = parse_args()

    # Print banner
    print("\n" + "="*80)
    if args.debug:
        print(" "*25 + "DEBUG MODE ENABLED")
        print(" "*15 + "Using 10 samples, 1 epoch, batch_size=4")
    else:
        print(" "*20 + "CALORIE REGRESSION TRAINING")
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
        config['training']['batch_size'] = 4
        config['training']['num_epochs'] = 1
        config['training']['log_interval'] = 1
        config['data']['num_workers'] = 0

    # Print config summary
    print("\nConfiguration Summary:")
    print(f"  Model: CalorieRegressor (with {config['model'].get('feature_extractor', {}).get('backbone', 'efficientnet_b0')} backbone)")
    print(f"  Num outputs: 5 (calories, protein, carb, fat, mass)")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Num epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['optimizer']['lr']}")
    print(f"  Optimizer: {config['optimizer']['name']}")
    print(f"  Loss: {config['training'].get('loss', 'smooth_l1')}")
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

    # Get model config
    fe_config = config['model'].get('feature_extractor', {})
    regressor_config = config['model'].get('regressor', {})

    backbone = fe_config.get('backbone', 'efficientnet_b0')
    pretrained = fe_config.get('pretrained', True)
    freeze_backbone = fe_config.get('freeze_backbone', False)

    input_dim = regressor_config.get('input_dim', 1280)
    hidden_dims = tuple(regressor_config.get('hidden_dims', [512, 256, 128]))
    output_dim = regressor_config.get('output_dim', 5)
    dropout = regressor_config.get('dropout', 0.3)

    # Create feature extractor (classifier without final layer)
    feature_extractor = FoodClassifier(
        num_classes=132,  # Not used for feature extraction
        backbone=backbone,
        pretrained=pretrained
    )
    if freeze_backbone:
        feature_extractor.freeze_backbone()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()  # Keep in eval mode for feature extraction

    # Create regressor
    model = CalorieRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout
    )
    model = model.to(device)

    # Count parameters
    fe_params = sum(p.numel() for p in feature_extractor.parameters())
    reg_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = fe_params + reg_params
    print(f"\nModel created:")
    print(f"  Feature Extractor: {backbone} ({fe_params:,} params)")
    print(f"  Regressor: CalorieRegressor ({reg_params:,} params)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable regressor params: {trainable_params:,}")

    # Setup loss function
    loss_name = config['training'].get('loss', 'smooth_l1')
    if loss_name == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    elif loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    print(f"\nLoss function: {loss_name}")

    # Setup optimizer
    optimizer_name = config['optimizer']['name'].lower()
    if optimizer_name == 'adam':
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
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['optimizer'].get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Optimizer: {optimizer_name}")

    # Setup scheduler
    scheduler = None
    if 'scheduler' in config and config['scheduler']['name']:
        scheduler_name = config['scheduler']['name'].lower()
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs']
            )
        elif scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['scheduler'].get('step_size', 10),
                gamma=config['scheduler'].get('gamma', 0.1)
            )
        print(f"Scheduler: {scheduler_name}")

    # Setup logger
    exp_name = f"CalorieRegressor_regression"
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
                model, feature_extractor, train_loader, criterion, optimizer,
                device, train_metrics, logger, epoch, config, mean, std
            )

            # Validate
            val_results = validate(
                model, feature_extractor, val_loader, criterion, device,
                val_metrics_tracker, logger, epoch, mean, std
            )

            # Update scheduler
            if scheduler:
                scheduler.step()
                logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"  Train - Loss: {train_results['loss']:.4f}")
            print(f"          MAE: Calories={train_results['mae_calories']:.2f}, "
                  f"Protein={train_results['mae_protein']:.2f}, "
                  f"Carb={train_results['mae_carb']:.2f}, "
                  f"Fat={train_results['mae_fat']:.2f}, "
                  f"Mass={train_results['mae_mass']:.2f}")
            print(f"  Val   - Loss: {val_results['loss']:.4f}")
            print(f"          MAE: Calories={val_results['mae_calories']:.2f}, "
                  f"Protein={val_results['mae_protein']:.2f}, "
                  f"Carb={val_results['mae_carb']:.2f}, "
                  f"Fat={val_results['mae_fat']:.2f}, "
                  f"Mass={val_results['mae_mass']:.2f}")

            # Save checkpoint
            is_best = val_results['mae_calories'] < best_val_loss  # Use calorie MAE as primary metric
            if is_best:
                best_val_loss = val_results['mae_calories']
                patience_counter = 0
                print(f"  >>> New best model! Calorie MAE: {best_val_loss:.2f}")
            else:
                patience_counter += 1

            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    'train_loss': train_results['loss'],
                    'val_loss': val_results['loss'],
                    'val_mae_calories': val_results['mae_calories'],
                    'val_mae_protein': val_results['mae_protein'],
                    'val_mae_carb': val_results['mae_carb'],
                    'val_mae_fat': val_results['mae_fat'],
                    'val_mae_mass': val_results['mae_mass'],
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
        print(f"Best calorie MAE: {best_val_loss:.2f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print(f"Logs saved to: {log_dir}")

        logger.close()


if __name__ == '__main__':
    main()
