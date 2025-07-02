# GRAPHITE: Deep Learning Framework for Histopathology Images

The repository implements **GRAPHITE**—a deep learning framework for histopathology images. It combines Multiple Instance Learning (MIL), a hierarchical Graph Attention Network (HierGAT), and several explainable-AI (XAI) techniques into one pipeline.

The main README describes the project goal and a 4-step pipeline:

1.  **MIL Classification** – Train attention-based models on patch data.
2.  **Self-Supervised Learning** – Learn hierarchical representations with Graph Neural Networks.
3.  **XAI Visualization** – Produce attention maps and feature visualizations.
4.  **Saliency Map Fusion** – Combine attention mechanisms for better interpretability.

A figure (“GRAPHITE Pipeline Architecture”) illustrates how these pieces connect.

## Directory Layout

```bash
training_step_1/                  # MIL training code
training_step_2/                  # self-supervised HierGAT training
visualization_step_1/             # XAI visualizations
visualization_step_2/             # saliency map fusion
docs/                             # Sphinx documentation
quickstart.sh                     # interactive setup/runner
DATA_STRUCTURE.md                 # how to arrange datasets
REPRODUCIBILITY.md                # exact steps to reproduce results
```

## Data Organization

`DATA_STRUCTURE.md` explains the required folders:

```bash
dataset/
├── training_dataset_step_1/tma_core/   # patch folders per patient
├── training_dataset_step_2/core_image/ # full core images for SSL
├── visualization_dataset/{core_image,mask}/
├── cancer.txt
└── normal.txt
```

## Training Step 1 – MIL

This directory implements an attention-based MIL classifier. The README highlights features like color normalization, balanced sampling, and reproducible training.

Core code lives under `mil_classification/src`:

* `data/datasets.py` – dataset & loader utilities (custom collate, balanced sampler).
* `models/mil_classifier.py` – the ResNet18-based MIL model with attention.
* `training/train.py` – main training loop.

Scripts `run_training.py` (Python) and `run_training.sh` (bash) offer cross-platform entry points.

## Training Step 2 – HierGAT Self-Supervised Learning

`training_step_2/self_supervised_training` implements the hierarchical GAT model. The README details its multi-scale architecture and InfoMax/Scale-wise losses:

> GRAPHITE implements a novel self-supervised learning approach that:
>
> * Hierarchical Learning across magnifications
> * Graph-based architecture
> * Dual loss (InfoMax + Scale-wise)

Important modules include:

* `data/` – patch extraction, slide processing, and dataset classes.
* `models/` – HierGAT layers, attention mechanisms, and graph builders.
* `training/` – loss functions (HierarchicalInfoMaxLoss) and the `HierGATSSLTrainer`.
* `train.py` – command-line training entry point.

## Visualization Step 1 – XAI Visualizations

`visualization_step_1/xai_visualization` provides several explainability methods (Grad-CAM, SHAP, LIME, MIL attention). Its README shows how to invoke the tool with various `--method` options and describes the project layout.

`main.py` parses CLI arguments, loads a trained MIL model, and invokes specific visualizers.

## Visualization Step 2 – Fusion

`visualization_step_2/fusion_visualization` fuses outputs from HierGAT, MIL, and CAM methods. The README describes the available CAM algorithms and fusion options—confidence, optimal, weighted, etc.—and outlines the metrics that can be computed.

`main_final_fusion.py` implements the detailed fusion pipeline and writes summaries of performance metrics.

## Supporting Scripts and Documentation

* `quickstart.sh` – an interactive script that checks prerequisites, creates a virtual environment, installs dependencies, and sets up data directories.
* `Dockerfile` – builds a container with all dependencies, GPU support, and a helpful entrypoint.
* `docs/` – Sphinx documentation; `make html` generates HTML documentation, and `make livehtml` starts an auto-reloading server.

## What to Explore Next

* **Data Preparation** – Review `DATA_STRUCTURE.md` carefully to arrange your dataset.
* **Reproducibility** – `REPRODUCIBILITY.md` lists exact environment versions and commands to replicate results.
* **Run the Pipeline** – Use `quickstart.sh` to set up the environment, then execute the individual training steps:
    ```bash
    python training_step_1/run_training.py --epochs 50
    python training_step_2/self_supervised_training/train.py --epochs 100
    python visualization_step_1/xai_visualization/main.py --method gradcam
    python visualization_step_2/fusion_visualization/main_final_fusion.py
    ```
* **Documentation** – Build the docs under `docs/` for API references and additional guides.
* **Tests** – Each main module has a `tests` folder with sample tests to check installation.

## Overall Summary

Overall, GRAPHITE is structured as a full end-to-end pipeline: MIL classification, self-supervised HierGAT learning, XAI visualization, and multi-attention fusion. The repository includes ready-to-use scripts, configuration files, and extensive documentation to help new users reproduce experiments and adapt the pipeline to their own histopathology datasets.
