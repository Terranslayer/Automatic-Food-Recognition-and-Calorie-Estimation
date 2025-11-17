"""
Food Classifier Models

Implements classification models for Nutrition5k:
- EfficientNet-B0/B4 (CNN-based)
- ViT-B/16 (Transformer-based)

All models support:
- Config-driven initialization
- Pretrained weights (ImageNet)
- Flexible number of classes
- Feature extraction mode
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import timm


class FoodClassifier(nn.Module):
    """
    Food classification model with multiple backbone options.

    Supports:
    - efficientnet_b0
    - efficientnet_b4
    - vit_base_patch16_224

    Args:
        backbone: Model architecture name
        num_classes: Number of food categories
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        dropout: Dropout rate for classifier head

    Example:
        >>> model = FoodClassifier(
        ...     backbone='efficientnet_b0',
        ...     num_classes=101,
        ...     pretrained=True
        ... )
        >>> x = torch.randn(8, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([8, 101])
    """

    SUPPORTED_BACKBONES = {
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b4': 'efficientnet_b4',
        'vit_b16': 'vit_base_patch16_224',
        'vit_base_patch16_224': 'vit_base_patch16_224',
    }

    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        num_classes: int = 101,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        # Validate backbone
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._freeze_backbone_init = freeze_backbone

        # Create backbone using timm
        timm_backbone = self.SUPPORTED_BACKBONES[backbone]
        self.backbone = timm.create_model(
            timm_backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='avg'  # Global average pooling
        )

        # Get feature dimension
        if 'efficientnet' in backbone:
            # EfficientNet feature dimensions
            feature_dim_map = {
                'efficientnet_b0': 1280,
                'efficientnet_b4': 1792,
            }
            feature_dim = feature_dim_map.get(backbone, 1280)
        elif 'vit' in backbone:
            # ViT feature dimension
            feature_dim = 768  # ViT-Base
        else:
            # Fallback: try to infer
            feature_dim = self.backbone.num_features

        self.feature_dim = feature_dim  # Store for later use

        # Freeze backbone if requested
        if self._freeze_backbone_init:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )

        # Initialize classifier head
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier head with Xavier uniform."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]
            return_features: If True, return both logits and features

        Returns:
            If return_features=False: logits [B, num_classes]
            If return_features=True: dict with 'logits' and 'features'
        """
        # Extract features
        features = self.backbone(x)  # [B, feature_dim]

        # Classify
        logits = self.classifier(features)  # [B, num_classes]

        if return_features:
            return {
                'logits': logits,
                'features': features
            }
        return logits

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone."""
        return self.feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone without classification.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: Backbone features [B, feature_dim]
        """
        return self.backbone(x)

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_params = backbone_params + classifier_params

        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            'backbone': backbone_params,
            'classifier': classifier_params,
            'total': total_params,
            'trainable': trainable_params,
        }

    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def create_classifier(config: Dict) -> FoodClassifier:
    """
    Create classifier from config dictionary.

    Args:
        config: Configuration dict with keys:
            - backbone: str
            - num_classes: int
            - pretrained: bool
            - freeze_backbone: bool (optional)
            - dropout: float (optional)

    Returns:
        FoodClassifier instance

    Example:
        >>> config = {
        ...     'backbone': 'efficientnet_b0',
        ...     'num_classes': 101,
        ...     'pretrained': True,
        ...     'dropout': 0.3
        ... }
        >>> model = create_classifier(config)
    """
    return FoodClassifier(
        backbone=config.get('backbone', 'efficientnet_b0'),
        num_classes=config.get('num_classes', 101),
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', False),
        dropout=config.get('dropout', 0.3),
    )


# Convenience functions for specific models
def efficientnet_b0(num_classes: int = 101, pretrained: bool = True, **kwargs):
    """Create EfficientNet-B0 classifier."""
    return FoodClassifier(
        backbone='efficientnet_b0',
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


def efficientnet_b4(num_classes: int = 101, pretrained: bool = True, **kwargs):
    """Create EfficientNet-B4 classifier."""
    return FoodClassifier(
        backbone='efficientnet_b4',
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


def vit_b16(num_classes: int = 101, pretrained: bool = True, **kwargs):
    """Create ViT-Base/16 classifier."""
    return FoodClassifier(
        backbone='vit_b16',
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


if __name__ == '__main__':
    # Smoke test
    print("Testing FoodClassifier...")

    # Test EfficientNet-B0
    print("\n1. EfficientNet-B0:")
    model_b0 = efficientnet_b0(num_classes=101, pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = model_b0(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Parameters: {model_b0.get_num_params()}")
    print(f"   Model size: {model_b0.get_model_size_mb():.2f} MB")

    # Test EfficientNet-B4
    print("\n2. EfficientNet-B4:")
    model_b4 = efficientnet_b4(num_classes=101, pretrained=False)
    logits = model_b4(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Parameters: {model_b4.get_num_params()}")
    print(f"   Model size: {model_b4.get_model_size_mb():.2f} MB")

    # Test ViT-B/16
    print("\n3. ViT-Base/16:")
    model_vit = vit_b16(num_classes=101, pretrained=False)
    logits = model_vit(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Parameters: {model_vit.get_num_params()}")
    print(f"   Model size: {model_vit.get_model_size_mb():.2f} MB")

    # Test feature extraction
    print("\n4. Feature extraction mode:")
    output = model_b0(x, return_features=True)
    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Features shape: {output['features'].shape}")

    # Test freeze/unfreeze
    print("\n5. Freeze/unfreeze backbone:")
    model_b0.freeze_backbone()
    print(f"   After freeze: {model_b0.get_num_params()['trainable']} trainable params")
    model_b0.unfreeze_backbone()
    print(f"   After unfreeze: {model_b0.get_num_params()['trainable']} trainable params")

    print("\n[SUCCESS] All smoke tests passed!")
