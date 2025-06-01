# Quick Usage Guide - Training Step 1

This document provides quick start instructions for running the MIL classification training.

## 🚀 Quick Start Commands

### 1. Basic Training (Recommended)

**Linux:**
```bash
cd training_step_1
./run_training.sh
```

**Windows:**
```cmd
cd training_step_1
run_training.bat
```

**Cross-platform (Python):**
```bash
cd training_step_1
python run_training.py
```

### 2. Quick Test (2 epochs, fast)

**Linux:**
```bash
./run_training.sh --quick_test
```

**Windows:**
```cmd
run_training.bat --quick_test
```

**Python:**
```bash
python run_training.py --quick_test
```

### 3. Custom Training

**Example with custom parameters:**
```bash
./run_training.sh \
    --data_dir /path/to/your/data \
    --epochs 150 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --color_norm \
    --balanced_sampler
```

**Python equivalent:**
```bash
python run_training.py \
    --data_dir /path/to/your/data \
    --epochs 150 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --color_norm \
    --balanced_sampler
```

## 📁 Data Setup

Ensure your data is organized as:

```
data_directory/
├── patient_001/
│   ├── patch_001.png
│   ├── patch_002.png
│   └── ...
├── patient_002/
└── ...
```

And label files:
- `cancer.txt` - List of cancer patient IDs
- `normal.txt` - List of normal patient IDs

## 🔧 Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--data_dir` | Path to patient data | `--data_dir /path/to/data` |
| `--epochs` | Number of training epochs | `--epochs 150` |
| `--batch_size` | Training batch size | `--batch_size 16` |
| `--max_patches` | Max patches per patient | `--max_patches 200` |
| `--learning_rate` | Learning rate | `--learning_rate 0.0005` |
| `--color_norm` | Enable color normalization | `--color_norm` |
| `--balanced_sampler` | Use balanced sampling | `--balanced_sampler` |
| `--quick_test` | Fast 2-epoch test | `--quick_test` |
| `--dry_run` | Show command only | `--dry_run` |

## 📊 Expected Outputs

After training, check:

```
mil_classification/output/
├── best_model.pth           # Trained model
├── training_history.png     # Training curves
└── mil_training.log         # Training logs
```

## 🆘 Troubleshooting

**GPU Memory Issues:**
```bash
./run_training.sh --batch_size 4 --max_patches 50
```

**Quick Verification:**
```bash
./run_training.sh --quick_test --dry_run
```

**Installation Issues:**
```bash
pip install -r mil_classification/requirements.txt
```

## ⏱️ Training Times

- **Quick Test**: 5-10 minutes
- **Standard Training**: 2-6 hours  
- **Large Scale**: 6-12 hours

Times depend on dataset size and hardware.

---

For detailed documentation, see [README.md](README.md) 