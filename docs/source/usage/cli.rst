Command Line Interface
======================

GRAPHITE provides a comprehensive command-line interface for all major operations. This section covers the available CLI commands and their usage.

Training Commands
-----------------

**Step 1: MIL Classification Training**

.. code-block:: bash

   # Navigate to training directory
   cd training_step_1/mil_classification
   
   # Run training with default parameters
   python train.py
   
   # Run training with custom parameters
   python train.py --epochs 100 --batch_size 32 --lr 0.001

**Step 2: Self-Supervised Training**

.. code-block:: bash

   # Navigate to self-supervised training directory
   cd training_step_2/self_supervised_training
   
   # Run self-supervised training
   python train.py --config config/default.yaml
   
   # Monitor training progress
   python monitor_training.py

Visualization Commands
----------------------

**XAI Visualization**

.. code-block:: bash

   # Navigate to XAI visualization directory
   cd visualization_step_1/xai_visualization
   
   # Generate attention visualizations
   python main.py --model_path /path/to/model --data_path /path/to/data
   
   # Generate interactive visualizations
   python main.py --interactive --port 8080

**Fusion Visualization**

.. code-block:: bash

   # Navigate to fusion visualization directory
   cd visualization_step_2/fusion_visualization
   
   # Run final fusion visualization
   python main_final_fusion.py
   
   # Run multi-level fusion visualization
   python main_multi_level_fusion.py

Quick Setup Commands
--------------------

**Automated Setup**

.. code-block:: bash

   # Run complete setup
   ./quickstart.sh
   
   # Setup with Docker
   docker-compose up --build

**Environment Setup**

.. code-block:: bash

   # Install dependencies
   pip install -r requirements.txt
   
   # Verify installation
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

Common Command Options
----------------------

Most GRAPHITE commands support the following common options:

- ``--help``: Show help message and exit
- ``--verbose``: Enable verbose output
- ``--config``: Specify configuration file path
- ``--output``: Specify output directory
- ``--gpu``: Specify GPU device ID (default: 0)
- ``--seed``: Set random seed for reproducibility

Examples
--------

**Complete Pipeline Execution**

.. code-block:: bash

   # 1. Train MIL classifier
   cd training_step_1/mil_classification && python train.py
   
   # 2. Train self-supervised model
   cd ../../training_step_2/self_supervised_training && python train.py
   
   # 3. Generate visualizations
   cd ../../visualization_step_1/xai_visualization && python main.py
   
   # 4. Create fusion visualizations
   cd ../../visualization_step_2/fusion_visualization && python main_final_fusion.py

**Custom Configuration**

.. code-block:: bash

   # Run with custom configuration
   python train.py --config my_config.yaml --output ./my_results --gpu 1
