"""
End-to-End Food Recognition and Calorie Estimation Pipeline

Integrates instance segmentation, classification, and calorie regression
into a unified pipeline for comprehensive food analysis.

Features:
- Mask R-CNN for food instance detection
- CNN-based classification for food categories
- MLP-based calorie regression
- Multi-instance handling and aggregation
- Geometric feature extraction from masks
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

try:
    from models.segmentation import FoodSegmentation
    from models.classifier import FoodClassifier
    from models.calorie_regressor import CalorieRegressor, CalorieRegressorWithGeometric
except ModuleNotFoundError:
    # Handle running as script from models/ directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.segmentation import FoodSegmentation
    from models.classifier import FoodClassifier
    from models.calorie_regressor import CalorieRegressor, CalorieRegressorWithGeometric


class EndToEndFoodRecognition(nn.Module):
    """
    Complete end-to-end pipeline for food recognition and calorie estimation.

    Pipeline:
    1. Instance segmentation (Mask R-CNN) - detect food items
    2. Feature extraction (EfficientNet/ViT) - extract visual features per instance
    3. Classification - predict food category per instance
    4. Calorie regression - predict nutritional values per instance
    5. Aggregation - combine predictions across multiple instances

    Args:
        num_classes: Number of food categories
        segmentation_config: Config dict for segmentation model
        classifier_config: Config dict for classifier model
        regressor_config: Config dict for regressor model
        use_geometric_features: Whether to use geometric features in regression
        aggregate_method: How to combine multi-instance predictions ('sum', 'mean')

    Example:
        >>> config = {
        ...     'num_classes': 101,
        ...     'classifier_config': {'backbone': 'efficientnet_b0'},
        ...     'regressor_config': {'output_dim': 5}
        ... }
        >>> model = EndToEndFoodRecognition(**config)
        >>> image = torch.randn(3, 800, 800)
        >>> results = model([image])
        >>> results[0].keys()
        dict_keys(['instances', 'total_calories', 'total_nutrition'])
    """

    def __init__(
        self,
        num_classes: int = 91,
        segmentation_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        regressor_config: Optional[Dict] = None,
        use_geometric_features: bool = True,
        aggregate_method: str = 'sum',
        min_detection_score: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_geometric_features = use_geometric_features
        self.aggregate_method = aggregate_method
        self.min_detection_score = min_detection_score

        # Initialize segmentation model
        seg_config = segmentation_config or {}
        self.segmentation = FoodSegmentation(
            num_classes=num_classes,
            **seg_config
        )

        # Initialize classifier
        cls_config = classifier_config or {}
        self.classifier = FoodClassifier(
            num_classes=num_classes,
            **cls_config
        )

        # Get feature dimension from classifier
        feature_dim = self.classifier.get_feature_dim()

        # Initialize regressor
        reg_config = regressor_config or {}
        # Filter to valid regressor parameters
        valid_reg_params = {'hidden_dims', 'output_dim', 'dropout', 'fusion_method'}
        filtered_reg_config = {k: v for k, v in reg_config.items() if k in valid_reg_params}

        if use_geometric_features:
            self.regressor = CalorieRegressorWithGeometric(
                visual_dim=feature_dim,
                geometric_dim=5,  # area, width, height, aspect_ratio, volume_proxy
                **filtered_reg_config
            )
        else:
            self.regressor = CalorieRegressor(
                input_dim=feature_dim,
                **filtered_reg_config
            )

        # Define image normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(
        self,
        images: List[torch.Tensor],
        return_features: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """
        End-to-end forward pass.

        Args:
            images: List of image tensors [C, H, W] (normalized)
            return_features: Whether to return intermediate features

        Returns:
            List of dicts (one per image) containing:
                - instances: List of per-instance predictions
                    - box: [4] bounding box
                    - mask: [H, W] segmentation mask
                    - score: detection confidence
                    - category: predicted food category
                    - category_prob: classification probability
                    - calories: predicted calories
                    - protein_g: predicted protein (if output_dim > 1)
                    - carb_g: predicted carbohydrates (if output_dim > 2)
                    - fat_g: predicted fat (if output_dim > 3)
                    - mass_g: predicted mass (if output_dim > 4)
                - total_calories: Sum/mean of all instance calories
                - total_nutrition: Dict with total/mean nutrition values
        """
        # Step 1: Instance segmentation
        self.segmentation.eval()
        with torch.no_grad():
            detections = self.segmentation(images)

        # Step 2-4: Process each image
        results = []
        for img_idx, (image, detection) in enumerate(zip(images, detections)):
            # Filter detections by score
            keep = detection['scores'] > self.min_detection_score
            boxes = detection['boxes'][keep]
            masks = detection['masks'][keep]
            scores = detection['scores'][keep]

            # Handle empty detections
            if len(boxes) == 0:
                results.append({
                    'instances': [],
                    'total_calories': torch.tensor(0.0),
                    'total_nutrition': {
                        'protein_g': torch.tensor(0.0),
                        'carb_g': torch.tensor(0.0),
                        'fat_g': torch.tensor(0.0),
                        'mass_g': torch.tensor(0.0),
                    }
                })
                continue

            # Process each detected instance
            instance_predictions = []
            for box, mask, score in zip(boxes, masks, scores):
                # Extract instance region
                instance_features = self._extract_instance_features(
                    image, box, mask
                )

                # FIX: Normalize instance features before classification/regression
                # This matches training_forward which normalizes at line ~283
                instance_features_norm = self.normalize(instance_features)

                # Classification
                with torch.no_grad():
                    class_logits = self.classifier(instance_features_norm.unsqueeze(0))
                    class_probs = torch.softmax(class_logits, dim=1)
                    category_idx = class_probs.argmax(dim=1).item()
                    category_prob = class_probs[0, category_idx].item()

                # Regression (use normalized features to match training)
                if self.use_geometric_features:
                    geometric_feats = self._extract_geometric_features(box, mask)
                    visual_feats = self.classifier.extract_features(
                        instance_features_norm.unsqueeze(0)
                    )
                    nutrition = self.regressor(visual_feats, geometric_feats)
                else:
                    visual_feats = self.classifier.extract_features(
                        instance_features_norm.unsqueeze(0)
                    )
                    nutrition = self.regressor(visual_feats)

                # Build instance prediction dict
                instance_pred = {
                    'box': box.cpu(),
                    'mask': mask.squeeze().cpu(),
                    'score': score.cpu().item(),
                    'category': category_idx,
                    'category_prob': category_prob,
                    'calories': nutrition[0, 0].cpu().item(),
                }

                # Add other nutrition values if available
                if nutrition.shape[1] > 1:
                    instance_pred['protein_g'] = nutrition[0, 1].cpu().item()
                if nutrition.shape[1] > 2:
                    instance_pred['carb_g'] = nutrition[0, 2].cpu().item()
                if nutrition.shape[1] > 3:
                    instance_pred['fat_g'] = nutrition[0, 3].cpu().item()
                if nutrition.shape[1] > 4:
                    instance_pred['mass_g'] = nutrition[0, 4].cpu().item()

                instance_predictions.append(instance_pred)

            # Aggregate predictions across instances
            total_nutrition = self._aggregate_predictions(instance_predictions)

            results.append({
                'instances': instance_predictions,
                'total_calories': total_nutrition['calories'],
                'total_nutrition': {
                    k: v for k, v in total_nutrition.items() if k != 'calories'
                }
            })

        return results

    def training_forward(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        nutrition: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with ground truth.

        FIXED: Uses detected instances instead of GT boxes for classification/regression.
        This makes training consistent with inference:
        1. Mask R-CNN detects instances
        2. Each instance predicts per-instance nutrition
        3. Sum all instance predictions
        4. Compare with total dish nutrition (the label)

        Args:
            images: List of image tensors [C, H, W]
            targets: List of target dicts for Mask R-CNN
            labels: Classification labels [batch_size]
            nutrition: Nutrition targets [batch_size, 5] - TOTAL dish nutrition

        Returns:
            Dict with losses:
                - segmentation_loss: Mask R-CNN loss
                - classification_loss: Classification CE loss
                - regression_loss: Regression MSE loss (sum of instances vs total)
                - class_logits: Raw classification outputs (from top-1 instance per image)
        """
        batch_size = len(images)
        device = images[0].device

        # Step 1: Segmentation loss (Mask R-CNN in training mode)
        self.segmentation.train()
        seg_loss_dict = self.segmentation(images, targets)
        segmentation_loss = sum(loss for loss in seg_loss_dict.values())

        # Step 2: Run Mask R-CNN in eval mode to get detected instances
        # This is the key fix - use detected instances, not GT boxes
        self.segmentation.eval()
        with torch.no_grad():
            detections = self.segmentation(images)

        # Step 3: Process each image - sum instance predictions, compare with total nutrition
        all_class_logits = []
        all_nutrition_predictions = []

        for img_idx, (image, detection) in enumerate(zip(images, detections)):
            # Filter detections by score (use lower threshold during training)
            keep = detection['scores'] > 0.1  # Lower threshold to get more instances
            boxes = detection['boxes'][keep]
            masks = detection['masks'][keep]
            scores = detection['scores'][keep]

            if len(boxes) == 0:
                # Fallback to GT box if no detection (ensures gradient flow)
                boxes = targets[img_idx]['boxes'][:1]
                masks = targets[img_idx]['masks'][:1]
                scores = torch.tensor([1.0], device=device)

            # Process all detected instances
            instance_nutrition_preds = []
            instance_class_logits = []

            for box, mask, score in zip(boxes, masks, scores):
                # Extract instance region
                instance_features = self._extract_instance_features(image, box, mask)
                instance_features_norm = self.normalize(instance_features)

                # Classification (get logits for loss)
                class_logits = self.classifier(instance_features_norm.unsqueeze(0))
                instance_class_logits.append(class_logits)

                # Regression (predict per-instance nutrition)
                visual_features = self.classifier.extract_features(instance_features_norm.unsqueeze(0))

                if self.use_geometric_features:
                    geometric_feats = self._extract_geometric_features(box, mask)
                    nutrition_pred = self.regressor(visual_features, geometric_feats)
                else:
                    nutrition_pred = self.regressor(visual_features)

                instance_nutrition_preds.append(nutrition_pred)

            # Sum all instance predictions for this image (matches inference aggregation)
            if len(instance_nutrition_preds) > 0:
                stacked_preds = torch.cat(instance_nutrition_preds, dim=0)  # [num_instances, 5]
                total_pred = stacked_preds.sum(dim=0, keepdim=True)  # [1, 5]
            else:
                total_pred = torch.zeros(1, 5, device=device)

            all_nutrition_predictions.append(total_pred)

            # Use top-1 instance class logits for classification loss
            if len(instance_class_logits) > 0:
                all_class_logits.append(instance_class_logits[0])  # Top confidence instance
            else:
                # Fallback: zero logits
                all_class_logits.append(torch.zeros(1, self.num_classes, device=device))

        # Stack predictions
        class_logits = torch.cat(all_class_logits, dim=0)  # [batch_size, num_classes]
        nutrition_predictions = torch.cat(all_nutrition_predictions, dim=0)  # [batch_size, 5]

        # Compute losses
        classification_loss = nn.CrossEntropyLoss()(class_logits, labels)
        regression_loss = nn.MSELoss()(nutrition_predictions, nutrition)

        return {
            'segmentation_loss': segmentation_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'class_logits': class_logits,
        }

    def _extract_instance_features(
        self,
        image: torch.Tensor,
        box: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract visual features from a detected food instance.

        Args:
            image: Input image [C, H, W]
            box: Bounding box [4] (x1, y1, x2, y2)
            mask: Segmentation mask [1, H, W]

        Returns:
            Cropped and resized instance image [C, 224, 224]
        """
        # Convert box coordinates to integers
        x1, y1, x2, y2 = box.int()

        # Crop instance region
        instance = image[:, y1:y2, x1:x2]

        # Resize to classifier input size (224x224)
        instance = T.Resize((224, 224))(instance)

        return instance

    def _extract_geometric_features(
        self,
        box: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract geometric features from bounding box and mask.

        Args:
            box: Bounding box [4] (x1, y1, x2, y2)
            mask: Segmentation mask [1, H, W]

        Returns:
            Geometric features [1, 5]: [area, width, height, aspect_ratio, volume_proxy]
        """
        # Bounding box dimensions
        x1, y1, x2, y2 = box
        width = (x2 - x1).float()
        height = (y2 - y1).float()
        box_area = width * height

        # Mask area (pixel count)
        mask_area = mask.sum().float()

        # Aspect ratio
        aspect_ratio = width / (height + 1e-6)

        # Volume proxy (assuming roughly cylindrical shape)
        # Use mask area as base, height as proxy for depth
        volume_proxy = mask_area * torch.sqrt(mask_area) / 1000.0

        # Normalize features to reasonable ranges
        geometric_features = torch.tensor([
            mask_area / 10000.0,  # Normalized area
            width / 100.0,         # Normalized width
            height / 100.0,        # Normalized height
            aspect_ratio,          # Aspect ratio
            volume_proxy,          # Volume proxy
        ], dtype=torch.float32).unsqueeze(0)

        # Move to same device as mask
        geometric_features = geometric_features.to(mask.device)

        return geometric_features

    def _aggregate_predictions(
        self,
        instances: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate nutritional predictions across multiple instances.

        Args:
            instances: List of per-instance predictions

        Returns:
            Aggregated nutrition dict
        """
        if len(instances) == 0:
            return {
                'calories': torch.tensor(0.0),
                'protein_g': torch.tensor(0.0),
                'carb_g': torch.tensor(0.0),
                'fat_g': torch.tensor(0.0),
                'mass_g': torch.tensor(0.0),
            }

        # Collect values
        calories = [inst['calories'] for inst in instances]

        aggregated = {
            'calories': torch.tensor(calories).sum() if self.aggregate_method == 'sum'
                       else torch.tensor(calories).mean()
        }

        # Add other nutrition values if present
        if 'protein_g' in instances[0]:
            protein = [inst['protein_g'] for inst in instances]
            aggregated['protein_g'] = (torch.tensor(protein).sum() if self.aggregate_method == 'sum'
                                      else torch.tensor(protein).mean())

        if 'carb_g' in instances[0]:
            carb = [inst['carb_g'] for inst in instances]
            aggregated['carb_g'] = (torch.tensor(carb).sum() if self.aggregate_method == 'sum'
                                   else torch.tensor(carb).mean())

        if 'fat_g' in instances[0]:
            fat = [inst['fat_g'] for inst in instances]
            aggregated['fat_g'] = (torch.tensor(fat).sum() if self.aggregate_method == 'sum'
                                  else torch.tensor(fat).mean())

        if 'mass_g' in instances[0]:
            mass = [inst['mass_g'] for inst in instances]
            aggregated['mass_g'] = (torch.tensor(mass).sum() if self.aggregate_method == 'sum'
                                   else torch.tensor(mass).mean())

        return aggregated

    def predict_from_pil(
        self,
        image: Image.Image,
        visualize: bool = False
    ) -> Dict:
        """
        Predict from PIL Image (for inference).

        Args:
            image: PIL Image
            visualize: Whether to return visualization-friendly outputs

        Returns:
            Prediction dict
        """
        # Convert to tensor
        image_tensor = T.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)

        # Run prediction
        self.eval()
        with torch.no_grad():
            results = self.forward([image_tensor])

        return results[0]

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'segmentation': self.segmentation.get_num_params()['total'],
            'classifier': self.classifier.get_num_params()['total'],
            'regressor': self.regressor.get_num_params()['total'],
            'total': sum(p.numel() for p in self.parameters()),
        }


def create_end_to_end_model(config: Dict) -> EndToEndFoodRecognition:
    """
    Create end-to-end model from config dictionary.

    Args:
        config: Configuration dict with keys:
            - num_classes: int
            - segmentation_config: dict (optional)
            - classifier_config: dict (optional)
            - regressor_config: dict (optional)
            - use_geometric_features: bool (default: True)
            - aggregate_method: str (default: 'sum')

    Returns:
        EndToEndFoodRecognition instance

    Example:
        >>> config = {
        ...     'num_classes': 101,
        ...     'classifier_config': {
        ...         'backbone': 'efficientnet_b0',
        ...         'pretrained': True
        ...     },
        ...     'regressor_config': {
        ...         'hidden_dims': (512, 256, 128),
        ...         'output_dim': 5
        ...     }
        ... }
        >>> model = create_end_to_end_model(config)
    """
    return EndToEndFoodRecognition(
        num_classes=config.get('num_classes', 91),
        segmentation_config=config.get('segmentation_config'),
        classifier_config=config.get('classifier_config'),
        regressor_config=config.get('regressor_config'),
        use_geometric_features=config.get('use_geometric_features', True),
        aggregate_method=config.get('aggregate_method', 'sum'),
        min_detection_score=config.get('min_detection_score', 0.5),
    )


if __name__ == '__main__':
    # Smoke test
    print("Testing EndToEndFoodRecognition...")

    # Test 1: Basic model creation
    print("\n1. Creating end-to-end model:")
    model = EndToEndFoodRecognition(
        num_classes=91,
        classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False},
        regressor_config={'output_dim': 5}
    )
    print(f"   [OK] Model created")
    params = model.get_num_params()
    print(f"   Component parameters:")
    print(f"     Segmentation: {params['segmentation']:,}")
    print(f"     Classifier: {params['classifier']:,}")
    print(f"     Regressor: {params['regressor']:,}")
    print(f"     Total: {params['total']:,}")

    # Test 2: Forward pass with dummy images
    print("\n2. Testing forward pass:")
    model.eval()
    images = [
        torch.randn(3, 800, 800),
        torch.randn(3, 600, 800)
    ]

    with torch.no_grad():
        results = model(images)

    print(f"   [OK] Processed {len(results)} images")
    for i, result in enumerate(results):
        num_instances = len(result['instances'])
        total_cal = result['total_calories'].item()
        print(f"   Image {i}: {num_instances} instances, {total_cal:.1f} kcal total")

        if num_instances > 0:
            inst = result['instances'][0]
            print(f"     First instance: category={inst['category']}, "
                  f"calories={inst['calories']:.1f}, score={inst['score']:.3f}")

    # Test 3: Different configurations
    print("\n3. Testing different configurations:")

    configs = [
        {
            'name': 'Without geometric features',
            'use_geometric_features': False,
            'classifier_config': {'backbone': 'efficientnet_b0', 'pretrained': False}
        },
        {
            'name': 'Mean aggregation',
            'aggregate_method': 'mean',
            'classifier_config': {'backbone': 'efficientnet_b0', 'pretrained': False}
        },
        {
            'name': 'High detection threshold',
            'min_detection_score': 0.8,
            'classifier_config': {'backbone': 'efficientnet_b0', 'pretrained': False}
        }
    ]

    for cfg in configs:
        name = cfg.pop('name')
        test_model = EndToEndFoodRecognition(num_classes=91, **cfg)
        print(f"   {name}: {test_model.get_num_params()['total']:,} params")

    # Test 4: PIL Image interface
    print("\n4. Testing PIL Image interface:")
    from PIL import Image
    pil_image = Image.new('RGB', (800, 800), color='red')

    model.eval()
    with torch.no_grad():
        result = model.predict_from_pil(pil_image)

    print(f"   [OK] PIL prediction: {len(result['instances'])} instances detected")

    # Test 5: Config-based creation
    print("\n5. Testing config-based model creation:")
    config = {
        'num_classes': 101,
        'classifier_config': {
            'backbone': 'efficientnet_b0',
            'pretrained': False,
            'dropout': 0.3
        },
        'regressor_config': {
            'hidden_dims': (512, 256, 128),
            'output_dim': 5,
            'dropout': 0.3
        },
        'use_geometric_features': True,
        'aggregate_method': 'sum'
    }

    config_model = create_end_to_end_model(config)
    print(f"   [OK] Config-based model created")
    print(f"   Total parameters: {config_model.get_num_params()['total']:,}")

    print("\n[SUCCESS] All smoke tests passed!")
