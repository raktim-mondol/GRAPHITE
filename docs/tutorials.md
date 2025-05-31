# Tutorials

## Overview

This section provides detailed tutorials for using GRAPHITE effectively.

## Tutorial 1: Basic MIL Classification

Learn how to train a Multiple Instance Learning model for histopathology classification.

### Prerequisites
- Prepared histopathology dataset
- GRAPHITE environment setup

### Steps
1. Data preparation
2. Model configuration
3. Training execution
4. Results evaluation

```python
# Example: Basic MIL training
from training_step_1.mil_classification import train

# Configure training parameters
config = {
    'data_path': 'path/to/your/data',
    'batch_size': 32,
    'epochs': 100
}

# Run training
model = train.main(config)
```

## Tutorial 2: Self-Supervised Graph Learning

Understand how to leverage graph neural networks for hierarchical feature learning.

### Key Concepts
- Graph construction from histopathology patches
- Hierarchical attention mechanisms
- Self-supervised objectives

### Implementation
```python
# Example: Graph-based learning
from training_step_2.self_supervised_training import train

# Load configuration
config_path = 'config/config.yaml'
trainer = train.GraphTrainer(config_path)

# Start training
trainer.train()
```

## Tutorial 3: XAI Visualization

Generate explainable AI visualizations to understand model decisions.

### Visualization Types
- Attention heatmaps
- Feature importance maps
- Interactive dashboards

### Usage Example
```python
# Example: XAI visualization
from visualization_step_1.xai_visualization import main

# Configure visualization
viz_config = {
    'model_path': 'path/to/trained/model.pth',
    'data_path': 'path/to/test/data',
    'output_dir': 'visualizations/'
}

# Generate visualizations
main.generate_visualizations(viz_config)
```

## Tutorial 4: Multi-Modal Fusion

Combine different attention mechanisms for enhanced interpretability.

### Fusion Strategies
- Early fusion
- Late fusion
- Attention-based fusion

### Implementation Guide
```python
# Example: Multi-modal fusion
from visualization_step_2.fusion_visualization import main_final_fusion

# Setup fusion pipeline
fusion_pipeline = main_final_fusion.FusionPipeline()
fusion_pipeline.load_models(['model1.pth', 'model2.pth'])

# Generate fused visualizations
results = fusion_pipeline.generate_fusion_maps()
```

## Best Practices

### Data Preparation
- Ensure consistent patch sizes
- Normalize staining variations
- Validate data quality

### Model Training
- Monitor training metrics
- Use appropriate learning rates
- Implement early stopping

### Visualization
- Choose appropriate color maps
- Validate interpretations
- Document findings

## Common Pitfalls

1. **Insufficient data preprocessing**
2. **Inadequate hyperparameter tuning**
3. **Misinterpretation of attention maps**
4. **Ignoring class imbalance**

## Advanced Topics

### Custom Model Architecture
Learn how to modify the existing architectures for specific use cases.

### Integration with Clinical Workflows
Best practices for deploying GRAPHITE in clinical settings.

### Performance Optimization
Tips for improving training speed and memory efficiency.
