# Graph Prototype Network for Few-Shot Node Classification

A PyTorch implementation of Graph Prototype Network (GPN) for few-shot node classification on graph-structured data with out-of-distribution (OOD) detection capabilities.

## Overview

This project implements a meta-learning approach for few-shot node classification that:
- Uses Graph Convolutional Networks (GCNs) for node embedding
- Employs prototype-based learning for classification
- Incorporates a scoring mechanism for node importance weighting
- Handles out-of-distribution samples during training

## Features

- **Few-shot Learning**: Supports N-way K-shot learning scenarios
- **Graph Neural Networks**: Utilizes GCN layers for node representation learning
- **Prototype Learning**: Creates class prototypes from support samples
- **OOD Detection**: Trains with out-of-distribution samples for robustness
- **Multiple Datasets**: Supports Amazon Clothing, Amazon Electronics, DBLP, and CoraFull datasets

## Architecture

The model consists of two main components:
1. **GPN_Encoder**: Graph convolutional encoder for node embeddings
2. **GPN_Valuator**: Scoring network for node importance estimation

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- NumPy
- SciPy
- scikit-learn
- NetworkX

## Installation

```bash
pip install torch torch-geometric numpy scipy scikit-learn networkx
```

## Usage

### Training

```bash
python train.py --dataset Amazon_clothing --way 5 --shot 3 --qry 15 --episodes 2000
```

### Parameters

- `--dataset`: Dataset name (Amazon_clothing, Amazon_electronics, dblp, corafull)
- `--way`: Number of classes per episode (N-way)
- `--shot`: Number of support samples per class (K-shot)
- `--qry`: Number of query samples per class
- `--episodes`: Number of training episodes
- `--lr`: Learning rate (default: 0.0005)
- `--hidden`: Hidden layer size (default: 16)
- `--alpha`: Loss weighting parameter (default: 0.7)
- `--outlier_num`: Number of OOD samples per episode (default: 10)

## Dataset Structure

Expected data format:
- Network edges: `few_shot_data/{dataset}_network`
- Training data: `few_shot_data/{dataset}_train.mat`
- Test data: `few_shot_data/{dataset}_test.mat`

## Model Components

### Graph Convolution Layer
- Implements basic GCN operations
- Supports sparse adjacency matrices
- Includes learnable weight parameters

### Encoder Network
- Two-layer GCN architecture
- ReLU activation and dropout
- Produces node embeddings

### Valuator Network
- Estimates node importance scores
- Uses graph structure and features
- Outputs scalar scores for weighting