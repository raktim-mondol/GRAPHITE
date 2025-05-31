# Examples

## Complete Usage Examples

This section provides practical examples of using GRAPHITE for different scenarios.

## Example 1: End-to-End Pipeline

Run the complete GRAPHITE pipeline on a sample dataset:

```bash
#!/bin/bash
# Complete pipeline execution

echo "Starting GRAPHITE pipeline..."

# Step 1: MIL Classification
echo "Step 1: MIL Classification Training"
cd training_step_1
python run_training.py --config config/default.yaml
cd ..

# Step 2: Self-supervised learning
echo "Step 2: Self-supervised Learning"
cd training_step_2/self_supervised_training
python train.py --config config/config.yaml
cd ../..

# Step 3: XAI Visualization
echo "Step 3: XAI Visualization"
cd visualization_step_1/xai_visualization
python main.py --model_path ../../training_step_1/output/best_model.pth
cd ../..

# Step 4: Fusion Visualization
echo "Step 4: Multi-modal Fusion"
cd visualization_step_2/fusion_visualization
python main_final_fusion.py
cd ../..

echo "Pipeline completed successfully!"
```

## Example 2: Custom Data Loading

```python
# Example: Custom data loader for histopathology images
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class HistopathologyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # Load your histopathology data
        # This is a placeholder - implement based on your data structure
        samples = []
        # Add your data loading logic here
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Usage
dataset = HistopathologyDataset('data/train')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Example 3: Model Inference

```python
# Example: Using trained GRAPHITE model for inference
import torch
from training_step_1.mil_classification.src.models import MILModel

def load_model(model_path):
    """Load a trained GRAPHITE model"""
    model = MILModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image_batch):
    """Make predictions on a batch of images"""
    with torch.no_grad():
        outputs = model(image_batch)
        predictions = torch.softmax(outputs, dim=1)
    return predictions

# Usage
model = load_model('models/best_model.pth')
predictions = predict(model, image_batch)
```

## Example 4: Configuration Management

```python
# Example: Working with YAML configurations
import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def update_config(config, updates):
    """Update configuration with new values"""
    for key, value in updates.items():
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value
    return config

# Usage
config = load_config('config/config.yaml')
config = update_config(config, {
    'training.batch_size': 64,
    'model.attention_dim': 256
})
```

## Example 5: Visualization Generation

```python
# Example: Generating attention visualizations
import matplotlib.pyplot as plt
import numpy as np
from visualization_step_1.xai_visualization import main

def generate_attention_heatmap(model, image, save_path=None):
    """Generate attention heatmap for a single image"""
    
    # Get attention weights from model
    attention_weights = model.get_attention_weights(image)
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Attention heatmap
    im = ax2.imshow(attention_weights, cmap='hot', alpha=0.8)
    ax2.imshow(image, alpha=0.5)
    ax2.set_title('Attention Heatmap')
    ax2.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Usage
generate_attention_heatmap(model, test_image, 'attention_heatmap.png')
```

## Example 6: Batch Processing

```python
# Example: Processing multiple samples in batch
from pathlib import Path
import pandas as pd

def process_batch(data_dir, model, output_dir):
    """Process a batch of histopathology images"""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for image_path in data_dir.glob('*.png'):
        # Load and preprocess image
        image = load_and_preprocess(image_path)
        
        # Make prediction
        prediction = model.predict(image)
        
        # Generate visualization
        viz_path = output_dir / f"{image_path.stem}_attention.png"
        generate_attention_heatmap(model, image, viz_path)
        
        # Store results
        results.append({
            'image_name': image_path.name,
            'prediction': prediction.item(),
            'confidence': prediction.max().item(),
            'visualization_path': str(viz_path)
        })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'results_summary.csv', index=False)
    
    return results_df

# Usage
results = process_batch('data/test', model, 'output/batch_results')
```

## Docker Examples

### Building and Running with Docker

```bash
# Build the Docker image
docker build -t graphite:latest .

# Run with GPU support
docker run --gpus all -v $(pwd)/data:/app/data graphite:latest

# Run with docker-compose
docker-compose up
```

### Docker Compose Configuration

```yaml
# docker-compose.yml example
version: '3.8'
services:
  graphite:
    build: .
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Testing Examples

```python
# Example: Unit testing for GRAPHITE components
import unittest
import torch
from training_step_1.mil_classification.src.models import MILModel

class TestMILModel(unittest.TestCase):
    def setUp(self):
        self.model = MILModel(input_dim=512, hidden_dim=256, num_classes=2)
        self.test_input = torch.randn(1, 100, 512)  # batch_size=1, num_patches=100, feature_dim=512
    
    def test_forward_pass(self):
        """Test forward pass of MIL model"""
        output = self.model(self.test_input)
        self.assertEqual(output.shape, (1, 2))  # batch_size=1, num_classes=2
    
    def test_attention_weights(self):
        """Test attention weight generation"""
        attention_weights = self.model.get_attention_weights(self.test_input)
        self.assertEqual(attention_weights.shape, (1, 100))  # batch_size=1, num_patches=100

if __name__ == '__main__':
    unittest.main()
```
