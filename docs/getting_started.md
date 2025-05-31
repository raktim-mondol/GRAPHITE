# Getting Started with GRAPHITE

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

### Quick Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd full_final
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

### Docker Installation

Alternatively, you can use Docker:

```bash
docker-compose up --build
```

## Basic Usage

### Running the Complete Pipeline

Use the quickstart script:
```bash
bash quickstart.sh
```

### Step-by-Step Execution

1. **MIL Classification Training**:
   ```bash
   cd training_step_1
   python run_training.py
   ```

2. **Self-Supervised Learning**:
   ```bash
   cd training_step_2/self_supervised_training
   python train.py
   ```

3. **XAI Visualization**:
   ```bash
   cd visualization_step_1/xai_visualization
   python main.py
   ```

4. **Multi-Modal Fusion**:
   ```bash
   cd visualization_step_2/fusion_visualization
   python main_final_fusion.py
   ```

## Configuration

Each module includes configuration files that can be customized:

- `training_step_2/self_supervised_training/config/config.yaml`
- `visualization_step_1/xai_visualization/config/default.yaml`

## Data Requirements

See [DATA_STRUCTURE.md](../DATA_STRUCTURE.md) for detailed information about the expected data format and structure.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in configuration files
2. **Missing Dependencies**: Ensure all requirements are installed
3. **Data Format Issues**: Check data structure against DATA_STRUCTURE.md

### Getting Help

- Check the [SETUP.md](../SETUP.md) file for detailed setup instructions
- Review the [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for reproduction guidelines
- Open an issue in the repository for additional support
