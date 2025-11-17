"""
Nutrition5k Models

This module contains all model architectures for the Nutrition5k project:
- Classifier: EfficientNet-B0/B4, ViT-B/16
- Segmentation: Mask R-CNN
- Calorie Regressor: MLP
- End-to-End: Integrated pipeline
"""

from .classifier import FoodClassifier
from .segmentation import FoodSegmentation
from .calorie_regressor import CalorieRegressor, CalorieRegressorWithGeometric
from .end_to_end import EndToEndFoodRecognition

__all__ = [
    'FoodClassifier',
    'FoodSegmentation',
    'CalorieRegressor',
    'CalorieRegressorWithGeometric',
    'EndToEndFoodRecognition',
]
