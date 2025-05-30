# Training Step 1: MIL-based Histopathology Image Classification

This directory contains the implementation of a Multiple Instance Learning (MIL) model for histopathology image classification. The model uses an attention-based approach to aggregate patch-level features for patient-level diagnosis.

## 🔬 Overview

The MIL classifier addresses the challenge of histopathology image analysis where:
- **Images are too large** to process directly (gigapixel whole slide images)
- **Labels are available at patient level** but not for individual patches
- **Spatial relationships** between patches contain important diagnostic information

### Key Features

- ✅ **Attention-based MIL**: Uses attention mechanisms to focus on relevant patches
- ✅ **Pre-trained Backbone**: ResNet18 with TIA Toolbox weights for histopathology
- ✅ **Color Normalization**: Optional Macenko color normalization for consistency
- ✅ **Balanced Sampling**: Handles class imbalance with balanced batch sampling
- ✅ **Comprehensive Metrics**: Accuracy, F1-score, and AUC tracking
- ✅ **Early Stopping**: Prevents overfitting with configurable patience
- ✅ **Reproducible**: Fixed random seeds for consistent results

## 📂 Project Structure

```
training_step_1/
├── README.md                    # This file
├── run_training.sh             # Training script runner
├── mil_classification/         # Main implementation directory
│   ├── train.py               # Main training script
│   ├── quick_training_test.py # Quick verification script
│   ├── requirements.txt       # Python dependencies
│   ├── src/                   # Source code
│   │   ├── models/           # Model architectures
│   │   │   └── mil_classifier.py
│   │   ├── data/             # Data loading utilities
│   │   │   └── datasets.py
│   │   ├── training/         # Training utilities
│   │   │   └── train.py
│   │   └── utils/            # Utility functions
│   │       └── color_normalization.py
│   ├── tests/                # Test suite
│   │   ├── test_model.py
│   │   ├── test_data.py
│   │   ├── test_training.py
│   │   └── test_integration.py
│   └── output/               # Training outputs
```

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**: Python 3.8+
2. **Dependencies**: Install from requirements.txt
3. **Data Structure**: Organized patient folders with patch images
4. **Labels**: Cancer and normal patient ID files

### Installation

```bash
# Navigate to the training directory
cd training_step_1

# Install dependencies
pip install -r mil_classification/requirements.txt

# Verify installation
python mil_classification/quick_training_test.py
```

### Data Preparation

Your data should be organized as follows:

```
data/
├── tma_core_normalized_with_normal/
│   ├── patient_001/
│   │   ├── patch_001.png
│   │   ├── patch_002.png
│   │   └── ...
│   ├── patient_002/
│   └── ...
├── cancer.txt              # Cancer patient IDs
└── normal.txt              # Normal patient IDs
```

**Label Files Format:**
```csv
patient_id
patient_001
patient_002
...
```

### Basic Training

Run training with default parameters:

```bash
# Using the bash script (recommended)
./run_training.sh

# Or directly with Python
cd mil_classification
python train.py --data_dir /path/to/your/data
```

### Advanced Training

Customize training parameters:

```bash
./run_training.sh \
    --data_dir /path/to/your/data \
    --epochs 150 \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --max_patches 200 \
    --use_color_normalization
```

## ⚙️ Configuration

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--root_dir` | str | PBS_JOBFS | Root directory with patient folders |
| `--cancer_labels_path` | str | `../../dataset/cancer.txt` | Path to cancer patient labels |
| `--normal_labels_path` | str | `../../dataset/normal.txt` | Path to normal patient labels |
| `--batch_size` | int | 8 | Training batch size |
| `--max_patches` | int | 100 | Maximum patches per patient |
| `--test_size` | float | 0.3 | Test set proportion (70-30 split) |
| `--learning_rate` | float | 0.001 | Optimizer learning rate |
| `--num_epochs` | int | 100 | Number of training epochs |
| `--early_stopping_patience` | int | 10 | Early stopping patience |
| `--use_color_normalization` | flag | False | Apply Macenko color normalization |
| `--model_save_path` | str | `./output/best_model.pth` | Model save location |

### Model Architecture

The MIL model consists of:

1. **Feature Extractor**: ResNet18 pre-trained on histopathology data
2. **Patch Projector**: Projects patch features to lower dimension
3. **Attention Module**: Computes attention weights for patch aggregation
4. **Patient Projector**: Creates patient-level representations
5. **Classifier**: Final classification head

```python
# Model instantiation
model = MILHistopathModel(
    num_classes=2,        # Binary classification
    feat_dim=512,         # Feature dimension
    proj_dim=128,         # Projection dimension
    model_name="hf-hub:1aurent/resnet18.tiatoolbox-kather100k"
)
```

## 📊 Training Process

### Training Pipeline

1. **Data Loading**: 
   - Load patient patch images
   - Apply color normalization (optional)
   - Random sampling of patches per patient

2. **Model Training**:
   - Forward pass through MIL architecture
   - Attention-weighted patch aggregation
   - Binary cross-entropy loss computation
   - Adam optimization with learning rate scheduling

3. **Validation**:
   - Model evaluation on held-out test set
   - Metric computation (accuracy, F1, AUC)
   - Early stopping based on monitored metric

4. **Output Generation**:
   - Best model checkpoint saving
   - Training history visualization
   - Comprehensive logging

### Loss Function

The model uses Binary Cross-Entropy with Logits Loss:

```python
criterion = nn.BCEWithLogitsLoss()
```

This is suitable for binary classification tasks and provides stable gradients.

### Metrics

Training tracks the following metrics:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1-score for class balance
- **AUC**: Area Under the ROC Curve
- **Loss**: Training and validation loss

## 🔧 Troubleshooting

### Common Issues

**Memory Issues:**
```bash
# Reduce batch size and max patches
python train.py --batch_size 4 --max_patches 50
```

**Slow Training:**
```bash
# Use fewer patches per patient
python train.py --max_patches 75
```

**Data Loading Errors:**
- Verify data directory structure
- Check label file formats
- Ensure image files are readable

**CUDA Errors:**
- Check GPU memory availability
- Reduce batch size if needed
- Verify CUDA installation

### Performance Optimization

**For Faster Training:**
- Use SSD storage for data
- Increase `num_workers` in DataLoader
- Use mixed precision training
- Cache preprocessed patches

**For Better Accuracy:**
- Increase `max_patches` per patient
- Use color normalization
- Tune learning rate and batch size
- Experiment with different backbones

## 📈 Expected Results

### Performance Metrics

On typical histopathology datasets, expect:
- **Accuracy**: 0.75-0.90
- **F1-Score**: 0.70-0.88
- **AUC**: 0.80-0.95

### Training Time

Approximate training times:
- **Quick Test** (2 epochs, 50 patches): 5-10 minutes
- **Full Training** (100 epochs, 100 patches): 2-6 hours
- **Large Scale** (150 epochs, 200 patches): 6-12 hours

Times vary based on dataset size, hardware, and configuration.

## 🧪 Testing

### Quick Verification

```bash
# Test model architecture
cd mil_classification
python quick_training_test.py

# Run unit tests
python -m tests.run_all_tests --type model

# Integration test
python -m tests.run_all_tests --type integration
```

### Full Test Suite

```bash
# Run all tests
python -m tests.run_all_tests --verbose
```

## 📝 Outputs

### Generated Files

After training, the following files are created:

```
output/
├── best_model.pth           # Best model checkpoint
├── training_history.png     # Training curves visualization
└── mil_training.log         # Detailed training logs
```

### Model Usage

Load and use the trained model:

```python
import torch
from src.models.mil_classifier import MILHistopathModel

# Load model
model = MILHistopathModel(num_classes=2)
checkpoint = torch.load('output/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    patches = torch.randn(1, 100, 3, 224, 224)  # Batch of patches
    _, _, logits, attention = model(patches)
    prediction = torch.sigmoid(logits) > 0.5
```

## 🔗 Integration

This training step integrates with:
- **Data Preparation**: Uses dataset from `dataset/training_dataset_step_1/`
- **Step 2**: Trained model can be used for self-supervised learning
- **Visualization**: Model outputs feed into XAI visualization pipeline

## 🤝 Contributing

To contribute to this module:

1. Follow the existing code structure
2. Add appropriate tests for new features
3. Update documentation as needed
4. Ensure backward compatibility

## 📚 References

- [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712)
- [TIA Toolbox for Computational Pathology](https://github.com/TissueImageAnalytics/tiatoolbox)
- [Macenko Color Normalization](https://ieeexplore.ieee.org/document/5193250)

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test suite output
3. Examine training logs in `output/mil_training.log`
4. Verify data format and structure

---

**Next Steps**: After successful training, proceed to `training_step_2/` for self-supervised learning or `visualization_step_1/` for explainability analysis. 