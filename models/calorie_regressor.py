"""
Calorie Regression Model

Implements MLP-based calorie prediction from visual and geometric features.

Features:
- Multi-layer perceptron architecture
- Visual features from CNN backbone
- Optional geometric features (area, volume proxies)
- Regression output for calories and macronutrients
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class CalorieRegressor(nn.Module):
    """
    MLP-based calorie regression model.

    Predicts calories (and optionally macronutrients) from visual features
    extracted by a CNN backbone.

    Args:
        input_dim: Dimension of input visual features
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of regression targets (1 for calories only,
                   4 for calories + protein/carb/fat, 5 for + mass)
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'gelu', 'leaky_relu')

    Example:
        >>> model = CalorieRegressor(input_dim=1280, output_dim=5)
        >>> features = torch.randn(8, 1280)
        >>> predictions = model(features)
        >>> predictions.shape
        torch.Size([8, 5])  # [calories, protein, carb, fat, mass]
    """

    def __init__(
        self,
        input_dim: int = 1280,  # EfficientNet-B0 feature dim
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        output_dim: int = 5,  # calories, protein, carb, fat, mass
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (optional)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self.activation)

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            prev_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        # Output layer (no activation - direct regression)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Input visual features [B, input_dim]

        Returns:
            predictions: Regression outputs [B, output_dim]
                - Index 0: calories (kcal)
                - Index 1: protein (g) if output_dim > 1
                - Index 2: carbohydrates (g) if output_dim > 2
                - Index 3: fat (g) if output_dim > 3
                - Index 4: mass (g) if output_dim > 4
        """
        x = self.feature_layers(features)
        predictions = self.output_layer(x)
        return predictions

    def predict_with_names(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with named outputs.

        Args:
            features: Input visual features [B, input_dim]

        Returns:
            Dict with keys: 'calories', 'protein', 'carb', 'fat', 'mass'
            (depending on output_dim)
        """
        predictions = self.forward(features)

        output_names = ['calories', 'protein', 'carb', 'fat', 'mass']
        result = {}

        for i in range(min(self.output_dim, len(output_names))):
            result[output_names[i]] = predictions[:, i]

        return result

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


class CalorieRegressorWithGeometric(nn.Module):
    """
    Enhanced calorie regressor that combines visual and geometric features.

    Geometric features include:
    - Segmentation area
    - Bounding box dimensions
    - Aspect ratio
    - Volume proxies (if depth available)

    Args:
        visual_dim: Dimension of visual features from CNN
        geometric_dim: Dimension of geometric features
        hidden_dims: Hidden layer dimensions
        output_dim: Number of regression targets
        dropout: Dropout rate
        fusion_method: How to combine features ('concat', 'add')

    Example:
        >>> model = CalorieRegressorWithGeometric(
        ...     visual_dim=1280,
        ...     geometric_dim=5,
        ...     output_dim=5
        ... )
        >>> visual_feats = torch.randn(8, 1280)
        >>> geometric_feats = torch.randn(8, 5)
        >>> predictions = model(visual_feats, geometric_feats)
        >>> predictions.shape
        torch.Size([8, 5])
    """

    def __init__(
        self,
        visual_dim: int = 1280,
        geometric_dim: int = 5,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        output_dim: int = 5,
        dropout: float = 0.3,
        fusion_method: str = 'concat',
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.geometric_dim = geometric_dim
        self.fusion_method = fusion_method

        # Geometric feature projection
        self.geometric_proj = nn.Linear(geometric_dim, 128)

        # Determine combined dimension
        if fusion_method == 'concat':
            combined_dim = visual_dim + 128
        elif fusion_method == 'add':
            # Project visual features to match geometric projection
            self.visual_proj = nn.Linear(visual_dim, 128)
            combined_dim = 128
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        # Regression head
        self.regressor = CalorieRegressor(
            input_dim=combined_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        geometric_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with both visual and geometric features.

        Args:
            visual_features: Visual features [B, visual_dim]
            geometric_features: Geometric features [B, geometric_dim]

        Returns:
            predictions: Regression outputs [B, output_dim]
        """
        # Project geometric features
        geo_proj = self.geometric_proj(geometric_features)

        # Fuse features
        if self.fusion_method == 'concat':
            combined = torch.cat([visual_features, geo_proj], dim=1)
        elif self.fusion_method == 'add':
            vis_proj = self.visual_proj(visual_features)
            combined = vis_proj + geo_proj
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Regression
        return self.regressor(combined)

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        return self.regressor.get_num_params()


def create_calorie_regressor(config: Dict) -> CalorieRegressor:
    """
    Create calorie regressor from config dictionary.

    Args:
        config: Configuration dict

    Returns:
        CalorieRegressor instance
    """
    return CalorieRegressor(
        input_dim=config.get('input_dim', 1280),
        hidden_dims=config.get('hidden_dims', (512, 256, 128)),
        output_dim=config.get('output_dim', 5),
        dropout=config.get('dropout', 0.3),
        batch_norm=config.get('batch_norm', True),
        activation=config.get('activation', 'relu'),
    )


if __name__ == '__main__':
    # Smoke test
    print("Testing CalorieRegressor...")

    # Test basic regressor
    print("\n1. Basic CalorieRegressor:")
    model = CalorieRegressor(input_dim=1280, output_dim=5)
    print(f"   [OK] Model created")
    print(f"   Parameters: {model.get_num_params()}")
    print(f"   Model size: {model.get_model_size_mb():.2f} MB")

    features = torch.randn(8, 1280)
    predictions = model(features)
    print(f"   Input shape: {features.shape}")
    print(f"   Output shape: {predictions.shape}")

    # Test named predictions
    print("\n2. Named predictions:")
    named_preds = model.predict_with_names(features)
    print(f"   [OK] Named outputs: {list(named_preds.keys())}")
    for name, values in named_preds.items():
        print(f"     {name}: mean={values.mean().item():.2f}, std={values.std().item():.2f}")

    # Test different configurations
    print("\n3. Different architectures:")
    configs = [
        {'hidden_dims': (256,), 'output_dim': 1, 'activation': 'relu'},
        {'hidden_dims': (512, 256), 'output_dim': 4, 'activation': 'gelu'},
        {'hidden_dims': (1024, 512, 256, 128), 'output_dim': 5, 'activation': 'leaky_relu'},
    ]
    for i, config in enumerate(configs):
        model = create_calorie_regressor(config)
        params = model.get_num_params()
        print(f"   Config {i+1}: {params['total']} params, {config['hidden_dims']} layers")

    # Test geometric feature fusion
    print("\n4. CalorieRegressorWithGeometric:")
    model_geo = CalorieRegressorWithGeometric(
        visual_dim=1280,
        geometric_dim=5,
        output_dim=5
    )
    print(f"   [OK] Model created")
    print(f"   Parameters: {model_geo.get_num_params()}")

    visual_feats = torch.randn(8, 1280)
    geometric_feats = torch.randn(8, 5)  # [area, width, height, aspect_ratio, volume_proxy]
    predictions = model_geo(visual_feats, geometric_feats)
    print(f"   Visual input: {visual_feats.shape}")
    print(f"   Geometric input: {geometric_feats.shape}")
    print(f"   Output: {predictions.shape}")

    # Test different fusion methods
    print("\n5. Different fusion methods:")
    for fusion in ['concat', 'add']:
        model = CalorieRegressorWithGeometric(
            visual_dim=1280,
            geometric_dim=5,
            fusion_method=fusion
        )
        preds = model(visual_feats, geometric_feats)
        print(f"   {fusion}: output shape {preds.shape}")

    print("\n[SUCCESS] All smoke tests passed!")
