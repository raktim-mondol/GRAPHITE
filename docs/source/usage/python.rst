Python API
==========

GRAPHITE provides a comprehensive Python API for integrating its functionality into your own applications and research workflows.

Basic Usage
-----------

**Import GRAPHITE Components**

.. code-block:: python

   # Import core modules
   from training_step_1.mil_classification import train as mil_train
   from training_step_2.self_supervised_training import train as ssl_train
   from visualization_step_1.xai_visualization import main as xai_viz
   from visualization_step_2.fusion_visualization import main_final_fusion

**Quick Training Example**

.. code-block:: python

   import torch
   from training_step_1.mil_classification.train import main as train_mil
   
   # Set up training configuration
   config = {
       'epochs': 100,
       'batch_size': 32,
       'learning_rate': 0.001,
       'data_path': '/path/to/data',
       'output_path': '/path/to/output'
   }
   
   # Run MIL training
   model = train_mil(config)

MIL Classification API
----------------------

**Training a MIL Model**

.. code-block:: python

   from training_step_1.mil_classification.src.models import AttentionMIL
   from training_step_1.mil_classification.src.datasets import HistoDataset
   from torch.utils.data import DataLoader
   
   # Create dataset and dataloader
   dataset = HistoDataset(data_path='/path/to/data')
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   
   # Initialize model
   model = AttentionMIL(input_dim=2048, hidden_dim=256, n_classes=2)
   
   # Train model
   trained_model = train_model(model, dataloader, epochs=100)

**Making Predictions**

.. code-block:: python

   import torch
   
   # Load trained model
   model = torch.load('/path/to/trained_model.pth')
   model.eval()
   
   # Make predictions
   with torch.no_grad():
       predictions = model(input_tensor)
       probabilities = torch.softmax(predictions, dim=1)

Self-Supervised Learning API
----------------------------

**Training Self-Supervised Models**

.. code-block:: python

   from training_step_2.self_supervised_training.train import SSLTrainer
   from training_step_2.self_supervised_training.config import load_config
   
   # Load configuration
   config = load_config('config/ssl_config.yaml')
   
   # Initialize trainer
   trainer = SSLTrainer(config)
   
   # Start training
   trainer.train()

**Feature Extraction**

.. code-block:: python

   from training_step_2.self_supervised_training.models import SSLModel
   
   # Load pre-trained SSL model
   ssl_model = SSLModel.load_pretrained('/path/to/ssl_model.pth')
   
   # Extract features
   features = ssl_model.extract_features(image_tensor)

Visualization API
-----------------

**XAI Visualization**

.. code-block:: python

   from visualization_step_1.xai_visualization.src.visualizer import XAIVisualizer
   
   # Initialize visualizer
   visualizer = XAIVisualizer(model_path='/path/to/model')
   
   # Generate attention maps
   attention_maps = visualizer.generate_attention_maps(image_path='/path/to/image')
   
   # Create interactive visualization
   visualizer.create_interactive_viz(output_path='/path/to/output')

**Fusion Visualization**

.. code-block:: python

   from visualization_step_2.fusion_visualization.main_final_fusion import FusionVisualizer
   
   # Initialize fusion visualizer
   fusion_viz = FusionVisualizer()
   
   # Generate multi-level fusion visualization
   fusion_viz.generate_fusion_maps(
       model_outputs=model_outputs,
       attention_maps=attention_maps,
       output_path='/path/to/output'
   )

Configuration Management
------------------------

**Loading Configuration Files**

.. code-block:: python

   import yaml
   
   def load_config(config_path):
       with open(config_path, 'r') as file:
           config = yaml.safe_load(file)
       return config
   
   # Usage
   config = load_config('config/training_config.yaml')

**Creating Custom Configurations**

.. code-block:: python

   # Create custom training configuration
   custom_config = {
       'model': {
           'type': 'AttentionMIL',
           'input_dim': 2048,
           'hidden_dim': 256,
           'n_classes': 2
       },
       'training': {
           'epochs': 100,
           'batch_size': 32,
           'learning_rate': 0.001,
           'optimizer': 'Adam'
       },
       'data': {
           'train_path': '/path/to/train',
           'val_path': '/path/to/val',
           'test_path': '/path/to/test'
       }
   }

Utility Functions
-----------------

**Data Preprocessing**

.. code-block:: python

   from training_step_1.mil_classification.src.utils import preprocess_data
   
   # Preprocess histopathology images
   processed_data = preprocess_data(
       data_path='/path/to/raw_data',
       output_path='/path/to/processed',
       patch_size=224,
       overlap=0.1
   )

**Model Evaluation**

.. code-block:: python

   from training_step_1.mil_classification.src.evaluation import evaluate_model
   
   # Evaluate trained model
   results = evaluate_model(
       model=trained_model,
       test_loader=test_dataloader,
       metrics=['accuracy', 'auc', 'f1']
   )
   
   print(f"Test Accuracy: {results['accuracy']:.4f}")
   print(f"Test AUC: {results['auc']:.4f}")

Error Handling
--------------

**Common Error Patterns**

.. code-block:: python

   try:
       # GRAPHITE operations
       model = train_mil_model(config)
   except FileNotFoundError as e:
       print(f"Data file not found: {e}")
   except torch.cuda.OutOfMemoryError as e:
       print(f"GPU memory error: {e}")
       print("Try reducing batch size or using CPU")
   except Exception as e:
       print(f"Unexpected error: {e}")

Integration Examples
--------------------

**Complete Workflow**

.. code-block:: python

   # Complete GRAPHITE workflow example
   def run_graphite_pipeline(data_path, output_path):
       # Step 1: Train MIL classifier
       mil_config = load_config('configs/mil_config.yaml')
       mil_model = train_mil_classifier(mil_config)
       
       # Step 2: Train SSL model
       ssl_config = load_config('configs/ssl_config.yaml')
       ssl_model = train_ssl_model(ssl_config)
       
       # Step 3: Generate visualizations
       visualizer = XAIVisualizer(mil_model, ssl_model)
       attention_maps = visualizer.generate_maps(data_path)
       
       # Step 4: Create fusion visualizations
       fusion_viz = FusionVisualizer()
       fusion_results = fusion_viz.generate_fusion(attention_maps)
       
       # Save results
       save_results(fusion_results, output_path)
       
       return fusion_results
   
   # Run the complete pipeline
   results = run_graphite_pipeline('/path/to/data', '/path/to/output')
