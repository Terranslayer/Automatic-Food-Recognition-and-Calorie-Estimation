"""
Comprehensive tests for all model architectures.

Tests:
- FoodClassifier (EfficientNet-B0/B4, ViT-B/16)
- FoodSegmentation (Mask R-CNN)
- CalorieRegressor (MLP, with/without geometric features)
- EndToEndFoodRecognition (full pipeline)
"""

import pytest
import torch
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classifier import FoodClassifier, create_classifier
from models.segmentation import FoodSegmentation, create_segmentation_model
from models.calorie_regressor import (
    CalorieRegressor,
    CalorieRegressorWithGeometric,
    create_calorie_regressor
)
from models.end_to_end import EndToEndFoodRecognition, create_end_to_end_model


# ============================================================================
# FoodClassifier Tests
# ============================================================================

class TestFoodClassifier:
    """Tests for FoodClassifier model."""

    @pytest.mark.parametrize("backbone", ['efficientnet_b0', 'efficientnet_b4', 'vit_b16'])
    def test_classifier_creation(self, backbone):
        """Test that classifier can be created with different backbones."""
        model = FoodClassifier(
            backbone=backbone,
            num_classes=101,
            pretrained=False  # Don't download weights for testing
        )
        assert isinstance(model, FoodClassifier)
        assert model.num_classes == 101
        assert model.backbone_name == backbone

    @pytest.mark.parametrize("backbone,expected_dim", [
        ('efficientnet_b0', 1280),
        ('efficientnet_b4', 1792),
        ('vit_b16', 768),
    ])
    def test_feature_dimensions(self, backbone, expected_dim):
        """Test that feature dimensions match expected values."""
        model = FoodClassifier(backbone=backbone, pretrained=False)
        assert model.get_feature_dim() == expected_dim

    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = FoodClassifier(backbone='efficientnet_b0', num_classes=101, pretrained=False)
        model.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 101)
        assert not torch.isnan(output).any()

    def test_forward_with_features(self):
        """Test forward pass with feature extraction."""
        model = FoodClassifier(backbone='efficientnet_b0', pretrained=False)
        model.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = model(x, return_features=True)

        assert isinstance(output, dict)
        assert 'logits' in output
        assert 'features' in output
        assert output['logits'].shape == (4, 101)
        assert output['features'].shape == (4, 1280)

    def test_extract_features(self):
        """Test feature extraction method."""
        model = FoodClassifier(backbone='efficientnet_b0', pretrained=False)
        model.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            features = model.extract_features(x)

        assert features.shape == (4, 1280)
        assert not torch.isnan(features).any()

    def test_freeze_unfreeze_backbone(self):
        """Test freezing and unfreezing backbone parameters."""
        model = FoodClassifier(backbone='efficientnet_b0', pretrained=False)

        # Initially all parameters should be trainable
        assert all(p.requires_grad for p in model.backbone.parameters())

        # Freeze backbone
        model.freeze_backbone()
        assert all(not p.requires_grad for p in model.backbone.parameters())

        # Unfreeze backbone
        model.unfreeze_backbone()
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_parameter_counts(self):
        """Test parameter count method."""
        model = FoodClassifier(backbone='efficientnet_b0', pretrained=False)
        params = model.get_num_params()

        assert 'total' in params
        assert 'backbone' in params
        assert 'classifier' in params
        assert params['total'] == params['backbone'] + params['classifier']
        assert params['total'] > 0

    def test_config_creation(self):
        """Test creating classifier from config dict."""
        config = {
            'backbone': 'efficientnet_b0',
            'num_classes': 50,
            'pretrained': False,
            'dropout': 0.5
        }
        model = create_classifier(config)

        assert isinstance(model, FoodClassifier)
        assert model.num_classes == 50


# ============================================================================
# FoodSegmentation Tests
# ============================================================================

class TestFoodSegmentation:
    """Tests for FoodSegmentation (Mask R-CNN) model."""

    def test_segmentation_creation(self):
        """Test that segmentation model can be created."""
        model = FoodSegmentation(num_classes=91, pretrained=False)
        assert isinstance(model, FoodSegmentation)
        assert model.num_classes == 91

    def test_forward_inference(self):
        """Test forward pass in inference mode."""
        model = FoodSegmentation(num_classes=91, pretrained=False)
        model.eval()

        images = [
            torch.randn(3, 800, 800),
            torch.randn(3, 600, 800)
        ]

        with torch.no_grad():
            outputs = model(images)

        assert len(outputs) == 2
        for output in outputs:
            assert 'boxes' in output
            assert 'labels' in output
            assert 'scores' in output
            assert 'masks' in output

    def test_forward_training(self):
        """Test forward pass in training mode."""
        model = FoodSegmentation(num_classes=91, pretrained=False)
        model.train()

        images = [torch.randn(3, 800, 800)]
        targets = [{
            'boxes': torch.tensor([[10, 10, 100, 100]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'masks': torch.zeros((1, 800, 800), dtype=torch.uint8),
        }]

        loss_dict = model(images, targets)

        assert isinstance(loss_dict, dict)
        assert 'loss_classifier' in loss_dict
        assert 'loss_box_reg' in loss_dict
        assert 'loss_mask' in loss_dict
        assert 'loss_objectness' in loss_dict
        assert 'loss_rpn_box_reg' in loss_dict

    def test_predict_with_threshold(self):
        """Test predict method with score threshold."""
        model = FoodSegmentation(num_classes=91, pretrained=False)
        images = [torch.randn(3, 800, 800)]

        outputs = model.predict(images, score_threshold=0.7)

        assert len(outputs) == 1
        # All scores should be above threshold
        if len(outputs[0]['scores']) > 0:
            assert all(s > 0.7 for s in outputs[0]['scores'])

    def test_parameter_counts(self):
        """Test parameter count method."""
        model = FoodSegmentation(num_classes=91, pretrained=False)
        params = model.get_num_params()

        assert 'total' in params
        assert 'trainable' in params
        assert params['total'] > 40_000_000  # Should be ~44M params

    def test_config_creation(self):
        """Test creating segmentation model from config."""
        config = {
            'num_classes': 50,
            'backbone': 'resnet50',
            'pretrained': False,
            'trainable_backbone_layers': 3
        }
        model = create_segmentation_model(config)

        assert isinstance(model, FoodSegmentation)
        assert model.num_classes == 50


# ============================================================================
# CalorieRegressor Tests
# ============================================================================

class TestCalorieRegressor:
    """Tests for CalorieRegressor (MLP) model."""

    def test_regressor_creation(self):
        """Test that regressor can be created."""
        model = CalorieRegressor(
            input_dim=1280,
            hidden_dims=(512, 256, 128),
            output_dim=5
        )
        assert isinstance(model, CalorieRegressor)
        assert model.input_dim == 1280
        assert model.output_dim == 5

    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = CalorieRegressor(input_dim=1280, output_dim=5)
        model.eval()

        features = torch.randn(8, 1280)

        with torch.no_grad():
            predictions = model(features)

        assert predictions.shape == (8, 5)
        assert not torch.isnan(predictions).any()

    @pytest.mark.parametrize("output_dim", [1, 4, 5])
    def test_different_output_dims(self, output_dim):
        """Test regressor with different output dimensions."""
        model = CalorieRegressor(input_dim=1280, output_dim=output_dim)
        features = torch.randn(4, 1280)

        with torch.no_grad():
            predictions = model(features)

        assert predictions.shape == (4, output_dim)

    def test_predict_with_names(self):
        """Test named prediction output."""
        model = CalorieRegressor(input_dim=1280, output_dim=5)
        model.eval()

        features = torch.randn(8, 1280)

        with torch.no_grad():
            predictions = model.predict_with_names(features)

        assert isinstance(predictions, dict)
        assert 'calories' in predictions
        assert 'protein' in predictions
        assert 'carb' in predictions
        assert 'fat' in predictions
        assert 'mass' in predictions

        for key, value in predictions.items():
            assert value.shape == (8,)

    def test_parameter_counts(self):
        """Test parameter count method."""
        model = CalorieRegressor(input_dim=1280, output_dim=5)
        params = model.get_num_params()

        assert 'total' in params
        assert 'trainable' in params
        assert params['total'] > 0

    def test_config_creation(self):
        """Test creating regressor from config."""
        config = {
            'input_dim': 1280,
            'hidden_dims': (256, 128),
            'output_dim': 5,
            'dropout': 0.3
        }
        model = create_calorie_regressor(config)

        assert isinstance(model, CalorieRegressor)
        assert model.input_dim == 1280
        assert model.output_dim == 5


class TestCalorieRegressorWithGeometric:
    """Tests for CalorieRegressorWithGeometric model."""

    def test_geometric_regressor_creation(self):
        """Test that geometric regressor can be created."""
        model = CalorieRegressorWithGeometric(
            visual_dim=1280,
            geometric_dim=5,
            output_dim=5
        )
        assert isinstance(model, CalorieRegressorWithGeometric)

    def test_forward_with_geometric_features(self):
        """Test forward pass with both visual and geometric features."""
        model = CalorieRegressorWithGeometric(
            visual_dim=1280,
            geometric_dim=5,
            output_dim=5
        )
        model.eval()

        visual_feats = torch.randn(8, 1280)
        geometric_feats = torch.randn(8, 5)

        with torch.no_grad():
            predictions = model(visual_feats, geometric_feats)

        assert predictions.shape == (8, 5)
        assert not torch.isnan(predictions).any()

    @pytest.mark.parametrize("fusion_method", ['concat', 'add'])
    def test_fusion_methods(self, fusion_method):
        """Test different fusion methods."""
        model = CalorieRegressorWithGeometric(
            visual_dim=1280,
            geometric_dim=5,
            fusion_method=fusion_method
        )

        visual_feats = torch.randn(4, 1280)
        geometric_feats = torch.randn(4, 5)

        with torch.no_grad():
            predictions = model(visual_feats, geometric_feats)

        assert predictions.shape == (4, 5)


# ============================================================================
# EndToEndFoodRecognition Tests
# ============================================================================

class TestEndToEndFoodRecognition:
    """Tests for end-to-end pipeline model."""

    def test_end_to_end_creation(self):
        """Test that end-to-end model can be created."""
        model = EndToEndFoodRecognition(
            num_classes=91,
            classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False},
            regressor_config={'output_dim': 5}
        )
        assert isinstance(model, EndToEndFoodRecognition)
        assert model.num_classes == 91

    def test_forward_pass(self):
        """Test forward pass with dummy images."""
        model = EndToEndFoodRecognition(
            num_classes=91,
            classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False},
            regressor_config={'output_dim': 5}
        )
        model.eval()

        images = [
            torch.randn(3, 800, 800),
            torch.randn(3, 600, 800)
        ]

        with torch.no_grad():
            results = model(images)

        assert len(results) == 2
        for result in results:
            assert 'instances' in result
            assert 'total_calories' in result
            assert 'total_nutrition' in result

    def test_predict_from_pil(self):
        """Test prediction from PIL Image."""
        model = EndToEndFoodRecognition(
            num_classes=91,
            classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False}
        )
        model.eval()

        pil_image = Image.new('RGB', (800, 800), color='red')

        with torch.no_grad():
            result = model.predict_from_pil(pil_image)

        assert 'instances' in result
        assert 'total_calories' in result
        assert 'total_nutrition' in result

    @pytest.mark.parametrize("use_geometric", [True, False])
    def test_geometric_features_toggle(self, use_geometric):
        """Test with and without geometric features."""
        model = EndToEndFoodRecognition(
            num_classes=91,
            classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False},
            use_geometric_features=use_geometric
        )

        assert model.use_geometric_features == use_geometric

    @pytest.mark.parametrize("aggregate_method", ['sum', 'mean'])
    def test_aggregation_methods(self, aggregate_method):
        """Test different aggregation methods."""
        model = EndToEndFoodRecognition(
            num_classes=91,
            classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False},
            aggregate_method=aggregate_method
        )

        assert model.aggregate_method == aggregate_method

    def test_parameter_counts(self):
        """Test parameter count method."""
        model = EndToEndFoodRecognition(
            num_classes=91,
            classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False}
        )
        params = model.get_num_params()

        assert 'segmentation' in params
        assert 'classifier' in params
        assert 'regressor' in params
        assert 'total' in params
        assert params['total'] > 40_000_000  # Should be ~49M params

    def test_config_creation(self):
        """Test creating end-to-end model from config."""
        config = {
            'num_classes': 101,
            'classifier_config': {
                'backbone': 'efficientnet_b0',
                'pretrained': False
            },
            'regressor_config': {
                'output_dim': 5
            },
            'use_geometric_features': True
        }
        model = create_end_to_end_model(config)

        assert isinstance(model, EndToEndFoodRecognition)
        assert model.num_classes == 101


# ============================================================================
# Integration Tests
# ============================================================================

class TestModelIntegration:
    """Integration tests across multiple models."""

    def test_classifier_to_regressor_pipeline(self):
        """Test passing classifier features to regressor."""
        classifier = FoodClassifier(backbone='efficientnet_b0', pretrained=False)
        regressor = CalorieRegressor(input_dim=1280, output_dim=5)

        classifier.eval()
        regressor.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            features = classifier.extract_features(x)
            predictions = regressor(features)

        assert features.shape == (4, 1280)
        assert predictions.shape == (4, 5)

    def test_all_models_loadable(self):
        """Test that all models can be instantiated."""
        models = [
            FoodClassifier(backbone='efficientnet_b0', pretrained=False),
            FoodSegmentation(num_classes=91, pretrained=False),
            CalorieRegressor(input_dim=1280, output_dim=5),
            CalorieRegressorWithGeometric(visual_dim=1280, geometric_dim=5),
            EndToEndFoodRecognition(
                num_classes=91,
                classifier_config={'backbone': 'efficientnet_b0', 'pretrained': False}
            )
        ]

        for model in models:
            assert model is not None
            assert isinstance(model, torch.nn.Module)


# ============================================================================
# Utility Tests
# ============================================================================

class TestModelUtilities:
    """Tests for model utility functions."""

    def test_model_to_device(self):
        """Test moving models to different devices."""
        model = FoodClassifier(backbone='efficientnet_b0', pretrained=False)

        # Test CPU
        model.cpu()
        assert next(model.parameters()).device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            model.cuda()
            assert next(model.parameters()).device.type == 'cuda'

    def test_model_eval_mode(self):
        """Test switching between train and eval modes."""
        model = FoodClassifier(backbone='efficientnet_b0', pretrained=False)

        # Train mode
        model.train()
        assert model.training

        # Eval mode
        model.eval()
        assert not model.training

    def test_model_state_dict(self):
        """Test saving and loading state dict."""
        model = FoodClassifier(backbone='efficientnet_b0', pretrained=False)

        # Get state dict
        state_dict = model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Create new model and load
        new_model = FoodClassifier(backbone='efficientnet_b0', pretrained=False)
        new_model.load_state_dict(state_dict)

        # Verify parameters match
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(),
            new_model.state_dict().items()
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
