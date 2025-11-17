"""
Food Instance Segmentation Model

Implements Mask R-CNN for food instance segmentation in Nutrition5k.

Features:
- ResNet-50/101-FPN backbone
- Pre-trained on COCO
- Fine-tunable for food categories
- Instance masks + bounding boxes + class labels
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class FoodSegmentation(nn.Module):
    """
    Mask R-CNN model for food instance segmentation.

    Args:
        num_classes: Number of food categories (+ background)
        backbone: Backbone architecture ('resnet50' or 'resnet101')
        pretrained: Whether to use COCO pretrained weights
        trainable_backbone_layers: Number of trainable backbone layers (0-5)
        min_size: Minimum input image size
        max_size: Maximum input image size
        **kwargs: Additional arguments passed to Mask R-CNN

    Example:
        >>> model = FoodSegmentation(num_classes=101, pretrained=True)
        >>> images = [torch.randn(3, 800, 800)]
        >>> outputs = model(images)
        >>> outputs[0].keys()
        dict_keys(['boxes', 'labels', 'scores', 'masks'])
    """

    def __init__(
        self,
        num_classes: int = 91,  # 90 food classes + background
        backbone: str = 'resnet50',
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        min_size: int = 800,
        max_size: int = 1333,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Create base Mask R-CNN model
        if backbone == 'resnet50':
            self.model = maskrcnn_resnet50_fpn(
                pretrained=pretrained,
                trainable_backbone_layers=trainable_backbone_layers,
                min_size=min_size,
                max_size=max_size,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet50'.")

        # Replace box predictor
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box, num_classes
        )

        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[Dict[str, torch.Tensor], ...] | Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: List of tensors, each of shape [C, H, W]
            targets: Optional list of dicts with keys:
                - boxes: [N, 4] tensor of bounding boxes
                - labels: [N] tensor of class labels
                - masks: [N, H, W] tensor of segmentation masks

        Returns:
            Training mode (targets provided):
                Dict with loss values: 'loss_classifier', 'loss_box_reg',
                'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'

            Inference mode (targets=None):
                List of dicts, one per image:
                - boxes: [N, 4] predicted bounding boxes
                - labels: [N] predicted class labels
                - scores: [N] prediction confidence scores
                - masks: [N, 1, H, W] predicted instance masks
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")

        return self.model(images, targets)

    def predict(
        self,
        images: List[torch.Tensor],
        score_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference with score filtering.

        Args:
            images: List of image tensors
            score_threshold: Minimum confidence score for predictions

        Returns:
            Filtered predictions per image
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model(images)

        # Filter by score threshold
        filtered_outputs = []
        for output in outputs:
            keep = output['scores'] > score_threshold
            filtered_output = {
                'boxes': output['boxes'][keep],
                'labels': output['labels'][keep],
                'scores': output['scores'][keep],
                'masks': output['masks'][keep],
            }
            filtered_outputs.append(filtered_output)

        return filtered_outputs

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            'total': total_params,
            'trainable': trainable_params,
        }

    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def create_segmentation_model(config: Dict) -> FoodSegmentation:
    """
    Create segmentation model from config dictionary.

    Args:
        config: Configuration dict with keys:
            - num_classes: int
            - backbone: str (default: 'resnet50')
            - pretrained: bool (default: True)
            - trainable_backbone_layers: int (default: 3)
            - min_size: int (default: 800)
            - max_size: int (default: 1333)

    Returns:
        FoodSegmentation instance

    Example:
        >>> config = {
        ...     'num_classes': 91,
        ...     'backbone': 'resnet50',
        ...     'pretrained': True
        ... }
        >>> model = create_segmentation_model(config)
    """
    return FoodSegmentation(
        num_classes=config.get('num_classes', 91),
        backbone=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True),
        trainable_backbone_layers=config.get('trainable_backbone_layers', 3),
        min_size=config.get('min_size', 800),
        max_size=config.get('max_size', 1333),
    )


# Convenience function
def mask_rcnn_resnet50(num_classes: int = 91, pretrained: bool = True, **kwargs):
    """Create Mask R-CNN with ResNet-50 backbone."""
    return FoodSegmentation(
        num_classes=num_classes,
        backbone='resnet50',
        pretrained=pretrained,
        **kwargs
    )


if __name__ == '__main__':
    # Smoke test
    print("Testing FoodSegmentation...")

    # Test model creation
    print("\n1. Creating Mask R-CNN model:")
    model = mask_rcnn_resnet50(num_classes=91, pretrained=False)
    print(f"   [OK] Model created")
    print(f"   Parameters: {model.get_num_params()}")
    print(f"   Model size: {model.get_model_size_mb():.2f} MB")

    # Test inference mode
    print("\n2. Testing inference mode:")
    model.eval()
    images = [torch.randn(3, 800, 800), torch.randn(3, 600, 800)]
    with torch.no_grad():
        outputs = model(images)
    print(f"   [OK] Inference successful")
    print(f"   Number of predictions per image:")
    for i, output in enumerate(outputs):
        print(f"     Image {i}: {len(output['boxes'])} detections")
        print(f"       Keys: {list(output.keys())}")

    # Test predict with filtering
    print("\n3. Testing predict() with score threshold:")
    filtered = model.predict(images, score_threshold=0.7)
    print(f"   [OK] Filtered predictions")
    for i, output in enumerate(filtered):
        print(f"     Image {i}: {len(output['boxes'])} detections (score > 0.7)")

    # Test training mode (dummy targets)
    print("\n4. Testing training mode:")
    model.train()
    targets = [
        {
            'boxes': torch.tensor([[10, 10, 100, 100], [200, 200, 300, 300]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64),
            'masks': torch.zeros((2, 800, 800), dtype=torch.uint8),
        },
        {
            'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
            'labels': torch.tensor([3], dtype=torch.int64),
            'masks': torch.zeros((1, 600, 800), dtype=torch.uint8),
        }
    ]
    loss_dict = model(images, targets)
    print(f"   [OK] Training mode successful")
    print(f"   Losses: {list(loss_dict.keys())}")
    print(f"   Total loss: {sum(loss_dict.values()):.4f}")

    print("\n[SUCCESS] All smoke tests passed!")
