# Automatic Food Recognition and Calorie Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)

A comprehensive deep learning pipeline for automatic food recognition and nutritional estimation using the Nutrition5k dataset. This project combines instance segmentation, classification, and regression models for accurate dietary assessment.

**Authors:** Abdelrahman Aboelata, Jaewook Kwon, Yufeng Zhan
**Institution:** University of Maryland
**Course:** CMSC 703 - Machine Learning

---

## Overview

This system provides end-to-end food analysis through:

- **Food Classification**: Multi-class recognition using EfficientNet/Vision Transformers (132 food categories)
- **Instance Segmentation**: Mask R-CNN for identifying individual food items in images
- **Nutritional Estimation**: Regression models for calories, protein, carbohydrates, fat, and mass
- **Integrated Pipeline**: Combined multi-task model for complete dietary analysis

### Key Features

✅ **4 Model Architectures**: Classifier, Segmentation, Regression, End-to-End
✅ **Multiple Backbones**: EfficientNet-B0/B4, ViT-B/16, Mask R-CNN
✅ **Comprehensive Evaluation**: 68 unit tests, extensive metrics
✅ **Flexible Configuration**: YAML-based configs with inheritance
✅ **Production-Ready**: TensorBoard logging, checkpointing, reproducibility

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Terranslayer/Automatic-Food-Recognition-and-Calorie-Estimation.git
cd Automatic-Food-Recognition-and-Calorie-Estimation

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the [Nutrition5k dataset](https://www.kaggle.com/datasets/zygmuntyt/nutrition5k-dataset-side-angle-images) and place it in `data/nutrition5k/`.

Expected structure:
```
data/nutrition5k/
├── rgb_train_ids.txt
├── rgb_test_ids.txt
├── dish_*/
│   ├── rgb.png
│   └── ...
└── metadata.csv
```

### 3. Train Models

```bash
# Quick validation (< 2 min each)
python train_classifier.py --debug
python train_segmentation.py --debug
python train_regression.py --debug
python train_end_to_end.py --debug

# Full training
python train_classifier.py --config configs/efficientnet.yaml
python train_segmentation.py --config configs/mask_rcnn.yaml
python train_regression.py --config configs/regression.yaml
python train_end_to_end.py --config configs/end_to_end.yaml
```

### 4. Evaluate Models

```bash
python evaluate.py --model classifier --checkpoint experiments/.../best_model.pth
python evaluate.py --model segmentation --checkpoint experiments/.../best_model.pth
python evaluate.py --model regression --checkpoint experiments/.../best_model.pth
python evaluate.py --model end_to_end --checkpoint experiments/.../best_model.pth
```

---

## Project Structure

```
Automatic-Food-Recognition-and-Calorie-Estimation/
├── data/nutrition5k/             # Dataset (download separately)
├── models/                       # Model implementations
│   ├── classifier.py             # Food classification (EfficientNet/ViT)
│   ├── segmentation.py           # Instance segmentation (Mask R-CNN)
│   ├── calorie_regressor.py     # Nutrition regression (MLP)
│   └── end_to_end.py             # Integrated pipeline
├── configs/                      # Training configurations
│   ├── base.yaml                 # Base configuration
│   ├── debug.yaml                # Quick validation config
│   ├── efficientnet.yaml         # EfficientNet training
│   ├── vit.yaml                  # Vision Transformer training
│   ├── mask_rcnn.yaml            # Mask R-CNN training
│   ├── regression.yaml           # Regression training
│   └── end_to_end.yaml           # End-to-end training
├── utils/                        # Core utilities
│   ├── dataset.py                # Dataset class (4 modes)
│   ├── metrics.py                # Evaluation metrics
│   ├── logger.py                 # TensorBoard logging
│   ├── checkpoint.py             # Checkpoint management
│   ├── config_loader.py          # YAML config loader
│   └── visualize.py              # Visualization tools
├── scripts/                      # Utility scripts
│   ├── dataset_stats.py          # Dataset statistics
│   └── estimate_resources.py    # Resource estimation
├── tests/                        # Unit tests (68 tests)
│   ├── test_dataset.py           # Data pipeline tests
│   ├── test_models.py            # Model architecture tests
│   └── test_training.py          # Training infrastructure tests
├── train_classifier.py           # Classification training script
├── train_segmentation.py         # Segmentation training script
├── train_regression.py           # Regression training script
├── train_end_to_end.py           # End-to-end training script
├── evaluate.py                   # Unified evaluation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Model Architectures

### 1. Food Classifier
- **Backbones**: EfficientNet-B0/B4, Vision Transformer (ViT-B/16)
- **Task**: 132-class food classification
- **Output**: Food category probabilities
- **Metrics**: Top-1/5 accuracy, F1 score

### 2. Instance Segmentation
- **Architecture**: Mask R-CNN with ResNet-50-FPN
- **Task**: Detect and segment individual food items
- **Output**: Bounding boxes, masks, labels, scores
- **Metrics**: mAP@0.5, mAP@0.5:0.95, IoU

### 3. Calorie Regressor
- **Architecture**: 3-layer MLP [512, 256, 128]
- **Input**: Visual features from classifier
- **Output**: 5 values (calories, protein, carb, fat, mass)
- **Metrics**: MAE, RMSE, MAPE, R²

### 4. End-to-End Pipeline
- **Components**: Segmentation → Classification → Regression
- **Task**: Complete nutritional analysis from raw images
- **Multi-Instance**: Aggregates predictions across detected food items
- **Metrics**: Combined metrics from all tasks

---

## Configuration

All models are configured via YAML files with inheritance:

```yaml
# Example: configs/efficientnet.yaml
inherit: base.yaml  # Inherits common settings

model:
  backbone: efficientnet_b0
  num_classes: 132
  pretrained: true

training:
  batch_size: 32
  num_epochs: 50
  optimizer:
    lr: 1e-3
```

### Debug Mode

All scripts support `--debug` flag for quick validation (<2 min):
```bash
python train_classifier.py --debug  # Uses 10 samples, 1 epoch
```

---

## Running Tests

```bash
# Run all tests (68 tests total)
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v
pytest tests/test_training.py -v

# Run with coverage
pytest tests/ --cov=models --cov=utils --cov-report=html
```

---

## Dataset

**Nutrition5k Dataset Statistics:**
- Total Samples: 5,282 dish instances
- Train/Val/Test: 3,930 / 676 / 676 (74.4% / 12.8% / 12.8%)
- Food Categories: 132 unique dish types
- Nutritional Labels: Calories, mass, protein, carbs, fat (100% complete)
- Total Size: ~6.2 GB

**Dataset Modes:**
- `classification`: Food category prediction (132 classes)
- `regression`: Nutritional estimation (5 outputs)
- `segmentation`: Instance segmentation with masks
- `end_to_end`: Combined multi-task pipeline

---

## Requirements

### Hardware
- **GPU**: 8-24 GB VRAM (depending on model)
  - EfficientNet-B0: ~8 GB
  - Mask R-CNN: ~20 GB
  - End-to-End: ~24 GB
- **RAM**: 16-32 GB
- **Storage**: ~50 GB (dataset + checkpoints + logs)

### Software
- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- CUDA >= 11.8 (for GPU)
- See `requirements.txt` for full list

### Installation

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch, torchvision
- timm (for EfficientNet/ViT)
- tensorboard (logging)
- pytest (testing)
- PyYAML (configs)
- scikit-learn (metrics)

---

## Citation

If you use this code, please cite:

```bibtex
@misc{automatic-food-recognition-2025,
  title={Automatic Food Recognition and Calorie Estimation},
  author={Aboelata, Abdelrahman and Kwon, Jaewook and Zhan, Yufeng},
  year={2025},
  institution={University of Maryland},
  url={https://github.com/Terranslayer/Automatic-Food-Recognition-and-Calorie-Estimation}
}
```

For the Nutrition5k dataset:
```bibtex
@article{thames2021nutrition5k,
  title={Nutrition5k: Towards automatic nutritional understanding of generic food},
  author={Thames, Quin and Karpur, Arjun and Norris, Wade and Xia, Fangting and Panait, Liviu and Weyand, Tobias and Sim, Jack},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

---

## License

This project is for academic research purposes only.

---

## Contact

- **Abdelrahman Aboelata**: aaboelat@umd.edu
- **Jaewook Kwon**: jkwon@umd.edu
- **Yufeng Zhan**: yxz2803@umd.edu

**Institution:** University of Maryland, College Park
**Course:** CMSC 703 - Machine Learning

---

## Acknowledgments

- Nutrition5k dataset creators (Google Research)
- PyTorch and torchvision teams
- timm library for pre-trained models
- University of Maryland CMSC 703 course

---

## Project Status

- ✅ Data Pipeline Complete (Phase 2)
- ✅ Model Implementation Complete (Phase 3)
- ✅ Training Infrastructure Complete (Phase 4)
- 🔄 Evaluation & Results (Phase 5 - In Progress)

**Code Quality:**
- 68/68 Unit Tests Passing
- 8,500+ Lines of Code
- Comprehensive Documentation
- Production-Ready Infrastructure
