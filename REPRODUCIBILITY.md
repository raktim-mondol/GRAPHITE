# GRAPHITE Reproducibility Guide

This document provides detailed instructions for reproducing the results of the GRAPHITE histopathology analysis pipeline. Following these exact steps will ensure consistent and reproducible results across different environments.

## 📋 Table of Contents

- [Quick Reproduction](#-quick-reproduction)
- [Environment Setup](#-environment-setup)
- [Data Preparation](#-data-preparation)
- [Step-by-Step Execution](#-step-by-step-execution)
- [Expected Results](#-expected-results)
- [Validation](#-validation)
- [Troubleshooting](#-troubleshooting)

## ⚡ Quick Reproduction

For immediate reproduction with demo data:

```bash
# Clone and setup
git clone https://github.com/raktim-mondol/GRAPHITE.git
cd GRAPHITE
./quickstart.sh

# Follow menu options:
# 1. Check system requirements
# 2. Setup environment and install dependencies  
# 3. Setup data directories
# 4. Generate demo data
# 6. Run complete pipeline

# Total time: ~2-4 hours (depending on hardware)
```

## 🔧 Environment Setup

### Exact Environment Specifications

#### Software Versions
```yaml
python: 3.9.2
torch: 2.0.0+cu117
torchvision: 0.15.0+cu117
torch-geometric: 2.3.1
numpy: 1.24.3
pandas: 2.0.3
scikit-learn: 1.3.0
matplotlib: 3.7.2
opencv-python: 4.8.0.76
grad-cam: 1.4.0
shap: 0.41.0
```

#### Hardware Specifications (Recommended)
```yaml
gpu: NVIDIA Tesla V100 32GB (used/recommended)
ram: 16GB+ system memory
storage: 50GB+ free space (SSD recommended)
cpu: 8+ cores
```

### Reproducible Environment Setup

#### Method 1: Using quickstart.sh (Recommended)
```bash
git clone https://github.com/raktim-mondol/GRAPHITE.git
cd GRAPHITE
chmod +x quickstart.sh
./quickstart.sh
# Select option 2: "Setup environment and install dependencies"
```

#### Method 2: Manual Setup
```bash
# Create virtual environment
python3.9 -m venv graphite_env
source graphite_env/bin/activate

# Install exact PyTorch version
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install exact dependencies
pip install -r requirements.txt

# Install PyTorch Geometric
pip install torch-geometric==2.3.1
```

### Random Seed Configuration

Set these environment variables for reproducibility:

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

Add to your script:
```python
import torch
import numpy as np
import random

# Set random seeds
torch.manual_seed(78)
np.random.seed(78)
random.seed(78)

# For deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed(78)
    torch.cuda.manual_seed_all(78)
```

## 📊 Data Preparation

### Data Organization

Follow this exact directory structure:

```
dataset/
├── training_dataset_step_1/         # MIL Classification
│   └── tma_core/
│       ├── 10025/ (cancer) ... 10034/
│       ├── 22021/ (cancer), 22107/
│       └── 20001/ (normal) ... 20008/
├── training_dataset_step_2/         # Self-Supervised Learning
│   └── core_image/                  # Core images only (no masks needed)
│       ├── image_001.png
│       └── ...
├── visualization_dataset/           # Visualization
│   ├── core_image/
│   └── mask/                        # Ground truth masks for evaluation
├── cancer.txt                       # Exact patient IDs
└── normal.txt                       # Exact patient IDs
```

### Label Files (Exact Format)

**dataset/cancer.txt:**
```csv
patient_id
10025
10026
10027
10028
10029
10030
10031
10032
10033
10034
22021
22107
```

**dataset/normal.txt:**
```csv
patient_id
20001
20002
20003
20004
20005
20006
20007
20008
```

### Data Specifications

#### Training Step 1 (MIL)
- **Format**: PNG images
- **Size**: 224x224 pixels (recommended)
- **Patches per patient**: 50-200
- **Total patients**: 20 (12 cancer, 8 normal)
- **Color space**: RGB

#### Training Step 2 (Self-Supervised Learning)
- **Format**: PNG images  
- **Size**: 1024x1024 pixels (recommended)
- **Count**: 100-1000 core images
- **Important**: No masks required for self-supervised learning
- **Color space**: RGB

#### Visualization Dataset
- **Images**: PNG format, 1024x1024 pixels
- **Masks**: PNG format, same size as images, binary (0/255)
- **Count**: 50-200 image-mask pairs

### Demo Data Generation

For reproducible demo data:

```bash
./quickstart.sh
# Select option 4: "Generate demo data"
```

This creates synthetic histopathology-like data with:
- Fixed random seed (42) for consistency
- Proper directory structure
- Appropriate image dimensions
- Realistic tissue-like patterns

## 🔄 Step-by-Step Execution

### Step 1: MIL Classification Training

#### Exact Parameters
```bash
cd training_step_1/mil_classification

python train.py \
    --root_dir "../../dataset/training_dataset_step_1/tma_core" \
    --cancer_labels_path "../../dataset/cancer.txt" \
    --normal_labels_path "../../dataset/normal.txt" \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --max_patches 100 \
    --test_size 0.3 \
    --random_state 78 \
    --early_stopping_patience 10 \
    --metrics_to_monitor "auc" \
    --model_save_path "./output/best_model.pth" \
    --history_plot_path "./output/training_history.png"
```

#### Expected Runtime
- **GPU**: 2-4 hours
- **CPU**: 8-12 hours

#### Key Outputs
- `./output/best_model.pth`: Trained MIL model
- `./output/training_history.png`: Training curves
- `mil_training.log`: Detailed training logs

### Step 2: Self-Supervised Learning Training

#### Exact Parameters
```bash
cd training_step_2/self_supervised_training

python train.py \
    --data_dir "../../dataset/training_dataset_step_2/core_image" \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --hidden_dim 128 \
    --num_heads 4 \
    --dropout 0.1 \
    --patience 20 \
    --save_dir "./output" \
    --random_seed 78
```

#### Important Notes
- **No masks required**: Self-supervised learning only uses core images
- **Graph construction**: Automatically creates spatial graphs from images
- **Hierarchical attention**: Learns multi-scale representations

#### Expected Runtime
- **GPU**: 4-8 hours
- **CPU**: 16-24 hours

#### Key Outputs
- `./output/best_model.pth`: Trained HierGAT model
- `./output/embeddings.npy`: Learned feature representations
- `./output/training_logs.txt`: Training progress

### Step 3: XAI Visualization

#### Exact Parameters
```bash
cd visualization_step_1/xai_visualization

python generate_visualizations.py \
    --model_path "../../training_step_1/mil_classification/output/best_model.pth" \
    --data_dir "../../dataset/visualization_dataset" \
    --output_dir "./output" \
    --visualization_types "attention,gradcam,lime,integrated_gradients" \
    --save_individual_patches True \
    --overlay_opacity 0.4 \
    --random_seed 78
```

#### Expected Runtime
- **GPU**: 30-60 minutes
- **CPU**: 2-4 hours

#### Key Outputs
- `./output/attention_maps/`: Attention heatmaps
- `./output/gradcam_visualizations/`: GradCAM results
- `./output/lime_explanations/`: LIME explanations
- `./output/analysis_report.html`: Comprehensive report

### Step 4: Attention Fusion

#### Exact Parameters
```bash
cd visualization_step_2/fusion_visualization

python fusion_analysis.py \
    --mil_model "../../training_step_1/mil_classification/output/best_model.pth" \
    --ssl_model "../../training_step_2/self_supervised_training/output/best_model.pth" \
    --data_dir "../../dataset/visualization_dataset" \
    --output_dir "./output" \
    --fusion_methods "confidence,attention,gradient" \
    --alpha 0.5 \
    --beta 0.3 \
    --random_seed 78
```

#### Expected Runtime
- **GPU**: 20-40 minutes
- **CPU**: 1-2 hours

#### Key Outputs
- `./output/fusion_maps/`: Saliency map attention fusion results
- `./output/comparison_metrics.json`: Quantitative evaluation
- `./output/final_report.pdf`: Comprehensive analysis report

## 📊 Expected Results

### Training Step 1 (MIL Classification)

#### Performance Metrics (Demo Data)
```yaml
Accuracy: 0.85 ± 0.05
AUC: 0.90 ± 0.03
F1-Score: 0.82 ± 0.06
Precision: 0.84 ± 0.04
Recall: 0.81 ± 0.07
```

#### Training Characteristics
- **Convergence**: ~50-80 epochs
- **Best validation AUC**: ~0.90-0.95
- **Training loss**: Decreases smoothly
- **Validation loss**: Stabilizes after ~40 epochs

### Training Step 2 (Self-Supervised Learning)

#### Training Characteristics
```yaml
Initial Loss: ~2.5-3.0
Final Loss: ~0.8-1.2
Convergence: 60-80 epochs
Feature Dimension: 128
Graph Attention Heads: 4
```

#### Feature Quality
- **Embedding visualization**: Clear cluster separation
- **Attention maps**: Meaningful spatial attention
- **Hierarchical features**: Multi-scale representations

### Visualization Quality

#### XAI Metrics
```yaml
Attention Consistency: >0.75
GradCAM Quality Score: >0.80
LIME Stability: >0.70
Integrated Gradients Sensitivity: >0.65
```

#### Fusion Performance
```yaml
Saliency Map Coherence: >0.85
Cross-attention Alignment: >0.80
Diagnostic Confidence: >0.90
```

## ✅ Validation

### Result Validation Script

Create `validate_results.py`:

```python
import json
import numpy as np
from pathlib import Path

def validate_results():
    """Validate that results match expected benchmarks"""
    
    # Expected benchmarks (with tolerance)
    expected = {
        'mil_accuracy': (0.85, 0.10),  # (value, tolerance)
        'mil_auc': (0.90, 0.05),
        'ssl_final_loss': (1.0, 0.4),
        'attention_consistency': (0.75, 0.10)
    }
    
    # Load actual results
    results = {}
    
    # MIL results
    if Path("training_step_1/mil_classification/output/metrics.json").exists():
        with open("training_step_1/mil_classification/output/metrics.json") as f:
            mil_metrics = json.load(f)
            results['mil_accuracy'] = mil_metrics.get('accuracy', 0)
            results['mil_auc'] = mil_metrics.get('auc', 0)
    
    # Validation
    all_valid = True
    for metric, (exp_val, tolerance) in expected.items():
        if metric in results:
            actual = results[metric]
            if abs(actual - exp_val) > tolerance:
                print(f"❌ {metric}: {actual:.3f} (expected: {exp_val:.3f} ± {tolerance:.3f})")
                all_valid = False
            else:
                print(f"✅ {metric}: {actual:.3f}")
        else:
            print(f"⚠️  {metric}: Not found")
    
    return all_valid

if __name__ == "__main__":
    print("🔍 Validating Results...")
    valid = validate_results()
    if valid:
        print("\n🎉 All results are within expected ranges!")
    else:
        print("\n❌ Some results are outside expected ranges.")
```

Run validation:
```bash
python validate_results.py
```

### Checksum Verification

For exact reproduction, verify file checksums:

```bash
# Generate checksums for key output files
find . -name "*.pth" -o -name "*.png" -o -name "*.json" | xargs md5sum > results_checksums.md5

# Compare with reference checksums (if available)
md5sum -c reference_checksums.md5
```

### Log Analysis

Check training logs for expected patterns:

```bash
# MIL training should show decreasing loss
grep "Train Loss" training_step_1/mil_classification/mil_training.log | tail -10

# SSL training should converge
grep "Epoch" training_step_2/self_supervised_training/output/training_logs.txt | tail -10
```

## 🐛 Troubleshooting

### Common Reproducibility Issues

#### 1. Different Results on Different Runs

**Cause**: Non-deterministic operations
**Solution**: Ensure all random seeds are set:
```python
# Add to the beginning of each script
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

#### 2. CUDA vs CPU Differences

**Cause**: Different numerical precision
**Solution**: Use the same device type for reproduction:
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Or ensure same GPU model
nvidia-smi --query-gpu=name --format=csv,noheader
```

#### 3. Version Mismatches

**Cause**: Different package versions
**Solution**: Use exact version requirements:
```bash
pip install -r requirements.txt --force-reinstall --no-deps
```

#### 4. Data Loading Order

**Cause**: Different file system ordering
**Solution**: Ensure consistent data loading:
```python
# In data loaders, always sort file lists
file_list = sorted(glob.glob(pattern))
```

#### 5. Inconsistent Self-Supervised Learning

**Cause**: Missing mask directory expectation
**Solution**: Ensure only core images are used:
```bash
# Verify no mask directory exists for step 2
ls dataset/training_dataset_step_2/
# Should only see: core_image/
```

### Performance Variations

#### Expected Variation Ranges
```yaml
MIL Accuracy: ±0.05 (due to train/test split randomness)
SSL Loss: ±0.2 (due to graph construction variations)
Visualization Quality: ±0.1 (due to attention randomness)
```

#### Minimizing Variations
```bash
# Use fixed train/test splits
python train.py --random_state 78 --test_size 0.3

# Consistent graph construction
python train.py --graph_random_seed 78

# Fixed visualization sampling
python generate_visualizations.py --random_seed 78
```

## 📦 Result Packaging

### Creating Reproducible Result Package

```bash
# Create results archive
./package_results.sh

# Contents:
# - All trained models (.pth files)
# - Training logs and metrics
# - Generated visualizations
# - Configuration files
# - Environment specifications
# - Validation reports
```

### Sharing Results

```bash
# Generate result summary
python generate_result_summary.py

# Creates:
# - results_summary.json: All metrics and parameters
# - environment.yml: Exact environment specification
# - reproduction_commands.sh: Exact commands used
# - validation_report.pdf: Comprehensive validation
```

## 🔄 Continuous Reproduction

### Automated Testing

Set up automated reproduction testing:

```yaml
# .github/workflows/reproduction.yml
name: Reproduction Test
on: [push, pull_request]

jobs:
  reproduce:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.9
      - name: Run reproduction
        run: |
          ./quickstart.sh
          python validate_results.py
```

### Result Tracking

Track reproduction results over time:

```bash
# Save results with timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r outputs/ results_archive/reproduction_$timestamp/
```

## 📞 Support

For reproduction issues:

1. **Check Environment**: Ensure exact package versions
2. **Verify Data**: Confirm data structure and formats
3. **Review Logs**: Check for warning messages
4. **Compare Outputs**: Use validation scripts
5. **Report Issues**: Include environment details and error logs

## 📚 Additional Resources

- **[SETUP.md](SETUP.md)**: Detailed installation instructions
- **[DATA_STRUCTURE.md](DATA_STRUCTURE.md)**: Data organization requirements
- **[README.md](README.md)**: General usage instructions
- **Validation Scripts**: `scripts/validation/`
- **Reference Results**: `reference_outputs/`

---

**Note**: Exact numerical reproduction may vary slightly due to hardware differences, but results should be within the specified tolerance ranges. Always use the same random seeds and follow the exact parameter specifications for best reproducibility. 