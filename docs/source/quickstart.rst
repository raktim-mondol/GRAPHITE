Quick Start Guide
=================

This guide will help you get GRAPHITE up and running quickly with minimal setup.

🚀 Getting Started in 5 Minutes
--------------------------------

**Step 1: Clone and Setup**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/raktim-mondol/GRAPHITE.git
   cd GRAPHITE
   
   # Run automated setup
   ./quickstart.sh

**Step 2: Prepare Your Data**

GRAPHITE expects data in a specific structure. For this quickstart, we'll use the demo dataset:

.. code-block:: bash

   # Download demo dataset (small sample for testing)
   ./quickstart.sh  # Select option 3: "Download demo dataset"

Your data structure should look like:

.. code-block:: text

   dataset/
   ├── training_dataset_step_1/tma_core/
   │   ├── 10025/ (cancer samples)
   │   ├── 20001/ (normal samples)
   │   └── ...
   ├── training_dataset_step_2/core_image/
   │   ├── image1.tiff
   │   └── ...
   └── visualization_dataset/
       ├── core_image/
       └── mask/

**Step 3: Run the Complete Pipeline**

.. code-block:: bash

   # Run the full GRAPHITE pipeline
   ./quickstart.sh  # Select option 1: "Run complete pipeline"

🔬 Pipeline Overview
--------------------

GRAPHITE consists of 4 main steps:

.. tabs::

   .. tab:: Step 1: MIL Classification

      **Multiple Instance Learning for patch-level classification**
      
      .. code-block:: bash

         # Run individually
         python training_step_1/run_training.py --epochs 50
      
      **What it does:**
      
      - Trains attention-based MIL model
      - Processes histopathology patches
      - Generates patient-level predictions
      - Creates attention weights for interpretability

   .. tab:: Step 2: Self-Supervised Learning

      **Hierarchical Graph Attention Networks**
      
      .. code-block:: bash

         # Run individually  
         python training_step_2/self_supervised_training/train.py --epochs 100
      
      **What it does:**
      
      - Learns spatial relationships between tissue regions
      - Creates hierarchical graph representations
      - Generates rich feature embeddings
      - No manual annotations required

   .. tab:: Step 3: XAI Visualization

      **Explainable AI visualization and analysis**
      
      .. code-block:: bash

         # Run individually
         python visualization_step_1/xai_visualization/main.py
      
      **What it does:**
      
      - Generates attention heatmaps
      - Creates GradCAM visualizations
      - Produces LIME explanations
      - Provides quantitative explainability metrics

   .. tab:: Step 4: Saliency Fusion

      **Multi-modal attention fusion for enhanced interpretability**
      
      .. code-block:: bash

         # Run individually
         python visualization_step_2/fusion_visualization/main_final_fusion.py
      
      **What it does:**
      
      - Fuses attention from multiple models
      - Creates comprehensive saliency maps
      - Generates diagnostic insights
      - Produces final interpretability reports

📊 Expected Results
-------------------

After running the pipeline, you should see:

**Training Results:**

.. code-block:: text

   outputs/
   ├── training_step_1/
   │   ├── best_model.pth          # Trained MIL model
   │   ├── training_history.png    # Training curves
   │   └── metrics.json           # Performance metrics
   ├── training_step_2/
   │   ├── ssl_model.pth          # Self-supervised model
   │   └── embeddings.npy         # Feature embeddings
   └── ...

**Performance Benchmarks:**

- **MIL Classification**: 85-92% accuracy, 0.90-0.95 AUC
- **Training Time**: 2-8 hours depending on dataset size
- **Visualization Quality**: High-resolution attention maps and saliency visualizations

⚙️ Configuration Options
------------------------

**Quick Configuration Examples:**

.. code-block:: bash

   # Adjust batch size for your GPU memory
   python training_step_1/run_training.py --batch_size 8 --epochs 25
   
   # Enable mixed precision for faster training
   python training_step_1/run_training.py --mixed_precision --batch_size 16
   
   # Use specific GPU
   CUDA_VISIBLE_DEVICES=0 python training_step_1/run_training.py

**Configuration Files:**

Edit configuration files for more control:

.. code-block:: yaml

   # training_step_2/config/config.yaml
   model:
     hidden_dim: 256
     num_heads: 8
     num_layers: 3
   
   training:
     batch_size: 16
     learning_rate: 0.001
     epochs: 100

🐛 Quick Troubleshooting
------------------------

**Common Issues and Quick Fixes:**

.. admonition:: GPU Memory Error
   :class: error

   **Error**: "CUDA out of memory"
   
   **Solution**: Reduce batch size
   
   .. code-block:: bash
   
      python train.py --batch_size 4

.. admonition:: Data Not Found
   :class: error

   **Error**: "Dataset directory not found"
   
   **Solution**: Check data structure
   
   .. code-block:: bash
   
      # Validate data structure
      python -c "from src.utils.data_validation import validate_dataset; validate_dataset('dataset/')"

.. admonition:: CUDA Not Available
   :class: error

   **Error**: "CUDA not available"
   
   **Solution**: Install CUDA-enabled PyTorch
   
   .. code-block:: bash
   
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

📋 Next Steps
-------------

**After successful quickstart:**

1. **Explore Your Results**: Check the ``outputs/`` directory for visualizations and metrics
2. **Customize Parameters**: Modify configuration files for your specific needs  
3. **Use Your Data**: Replace demo data with your histopathology datasets
4. **Advanced Features**: Explore :doc:`tutorials/advanced_features` for more options

**Recommended Learning Path:**

1. :doc:`user_guide/pipeline_overview` - Understand the complete workflow
2. :doc:`tutorials/basic_usage` - Learn basic usage patterns
3. :doc:`tutorials/custom_datasets` - Adapt GRAPHITE to your data
4. :doc:`api/training_step_1` - Dive into API documentation

💡 Tips for Success
-------------------

**Best Practices:**

- **Start Small**: Use demo dataset to test installation
- **Monitor Resources**: Watch GPU/CPU usage during training
- **Save Checkpoints**: Enable model checkpointing for long training runs
- **Validate Results**: Always check output quality and metrics

**Performance Tips:**

- Use SSD storage for faster data loading
- Optimize batch size for your hardware
- Enable mixed precision training when possible
- Use multiple GPUs if available

📞 Getting Help
---------------

If you encounter issues:

1. **Check Logs**: Review output logs for error details
2. **Validate Setup**: Run ``python -m pytest tests/`` to check installation
3. **Read Documentation**: See :doc:`development/troubleshooting` for detailed help
4. **Create Issue**: Report bugs on GitHub with system info and error logs

🎯 What's Next?
---------------

- **Learn More**: Explore detailed :doc:`user_guide/pipeline_overview`
- **Customize**: Check :doc:`tutorials/model_customization`
- **Contribute**: See :doc:`development/contributing` to help improve GRAPHITE
- **Stay Updated**: Star the repository for updates and new features
