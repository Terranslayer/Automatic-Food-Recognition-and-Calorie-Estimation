"""
Configuration Loader with YAML Inheritance Support

Supports loading YAML configs with inheritance from base.yaml:
- Load base configuration
- Load experiment-specific configuration
- Merge configurations (specific overrides base)
- Support command-line overrides

Example usage:
    >>> config = load_config('configs/efficientnet.yaml')
    >>> config = load_config('configs/debug.yaml', overrides={'training.batch_size': 8})
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import copy


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load a YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary containing YAML content

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            content = yaml.safe_load(f)
            return content if content is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {filepath}: {e}")


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary

    Example:
        >>> base = {'a': 1, 'b': {'c': 2}}
        >>> override = {'b': {'d': 3}, 'e': 4}
        >>> deep_merge(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override value
            result[key] = copy.deepcopy(value)

    return result


def apply_overrides(config: Dict, overrides: Dict[str, Any]) -> Dict:
    """
    Apply command-line overrides to config using dot notation.

    Args:
        config: Base configuration
        overrides: Dict of overrides with dot-separated keys

    Returns:
        Config with overrides applied

    Example:
        >>> config = {'training': {'batch_size': 32}}
        >>> apply_overrides(config, {'training.batch_size': 64})
        {'training': {'batch_size': 64}}
    """
    result = copy.deepcopy(config)

    for key, value in overrides.items():
        # Split key by dots to handle nested paths
        keys = key.split('.')
        current = result

        # Navigate to the nested location
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the final value
        current[keys[-1]] = value

    return result


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    base_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration with inheritance support.

    Supports 'inherit' key in config to specify parent config.

    Args:
        config_path: Path to experiment config file
        overrides: Optional dict of command-line overrides (dot notation)
        base_config_path: Optional explicit path to base config (auto-detected if not provided)

    Returns:
        Merged configuration dictionary

    Example:
        >>> # configs/efficientnet.yaml contains: inherit: base.yaml
        >>> config = load_config('configs/efficientnet.yaml')

        >>> # With overrides
        >>> config = load_config('configs/vit.yaml', overrides={'training.num_epochs': 100})
    """
    # Load the experiment config
    config = load_yaml(config_path)

    # Check if there's an inherit directive
    if 'inherit' in config:
        inherit_file = config.pop('inherit')

        # Resolve base config path
        if base_config_path is None:
            config_dir = Path(config_path).parent
            base_config_path = config_dir / inherit_file

        # Load base config recursively (supports multi-level inheritance)
        base_config = load_config(str(base_config_path))

        # Merge: base config + experiment config (experiment overrides base)
        config = deep_merge(base_config, config)

    # Apply command-line overrides
    if overrides:
        config = apply_overrides(config, overrides)

    return config


def validate_config(config: Dict, required_keys: List[str]) -> None:
    """
    Validate that config contains required keys.

    Args:
        config: Configuration dict
        required_keys: List of required keys (supports dot notation)

    Raises:
        ValueError: If required key is missing

    Example:
        >>> validate_config(config, ['model.name', 'training.batch_size'])
    """
    for key in required_keys:
        keys = key.split('.')
        current = config

        for k in keys:
            if k not in current:
                raise ValueError(f"Required config key missing: {key}")
            current = current[k]


def save_config(config: Dict, filepath: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dict
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def print_config(config: Dict, indent: int = 0) -> None:
    """
    Pretty print configuration.

    Args:
        config: Configuration dict
        indent: Indentation level (for recursion)
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print('  ' * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print('  ' * indent + f"{key}: {value}")


if __name__ == '__main__':
    # Smoke test
    print("Testing config loader...")

    # Test 1: Load base config
    print("\n1. Loading base config:")
    try:
        base_config = load_config('configs/base.yaml')
        print(f"   [OK] Loaded base.yaml with {len(base_config)} top-level keys")
        print(f"   Keys: {list(base_config.keys())}")
    except Exception as e:
        print(f"   [FAIL] {e}")

    # Test 2: Load config with inheritance
    print("\n2. Loading config with inheritance (efficientnet.yaml):")
    try:
        eff_config = load_config('configs/efficientnet.yaml')
        print(f"   [OK] Loaded efficientnet.yaml")
        print(f"   Experiment name: {eff_config.get('experiment', {}).get('name', 'N/A')}")
        print(f"   Batch size: {eff_config.get('data', {}).get('batch_size', 'N/A')}")
        print(f"   Learning rate: {eff_config.get('optimizer', {}).get('lr', 'N/A')}")
    except Exception as e:
        print(f"   [FAIL] {e}")

    # Test 3: Load debug config
    print("\n3. Loading debug config:")
    try:
        debug_config = load_config('configs/debug.yaml')
        print(f"   [OK] Loaded debug.yaml")
        print(f"   Batch size: {debug_config.get('training', {}).get('batch_size', 'N/A')}")
        print(f"   Num epochs: {debug_config.get('training', {}).get('num_epochs', 'N/A')}")
    except Exception as e:
        print(f"   [FAIL] {e}")

    # Test 4: Apply overrides
    print("\n4. Testing overrides:")
    try:
        overridden = load_config(
            'configs/base.yaml',
            overrides={'training.batch_size': 128, 'optimizer.lr': 0.001}
        )
        print(f"   [OK] Applied overrides")
        print(f"   Batch size: {overridden['training']['batch_size']} (should be 128)")
        print(f"   Learning rate: {overridden['optimizer']['lr']} (should be 0.001)")
    except Exception as e:
        print(f"   [FAIL] {e}")

    # Test 5: Validation
    print("\n5. Testing validation:")
    try:
        validate_config(base_config, ['experiment.name', 'training.batch_size', 'optimizer.lr'])
        print(f"   [OK] Validation passed")
    except Exception as e:
        print(f"   [FAIL] {e}")

    # Test 6: Deep merge
    print("\n6. Testing deep merge:")
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    override = {'b': {'d': 4, 'e': 5}, 'f': 6}
    merged = deep_merge(base, override)
    print(f"   Base: {base}")
    print(f"   Override: {override}")
    print(f"   Merged: {merged}")
    expected = {'a': 1, 'b': {'c': 2, 'd': 4, 'e': 5}, 'f': 6}
    if merged == expected:
        print(f"   [OK] Deep merge correct")
    else:
        print(f"   [FAIL] Expected {expected}, got {merged}")

    print("\n[SUCCESS] Config loader smoke tests complete!")
