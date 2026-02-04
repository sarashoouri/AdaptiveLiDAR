# Adaptive LiDAR Scanning: Harnessing Temporal Cues for Efficient 3D Object Detection via Multi-Modal Fusion

## Overview

This is the PyTorch code for our AAAI 2026 paper "Adaptive LiDAR Scanning: Harnessing Temporal Cues for Efficient 3D Object Detection via Multi-Modal Fusion
". 

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](arXiv:2508.01562)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![Overview Figure](overview.png)

## Introduction

Multi-sensor fusion using LiDAR and RGB cameras significantly enhances 3D object detection task. However, conventional LiDAR sensors perform dense, stateless scans, ignoring the strong temporal continuity in real-world scenes. This leads to substantial sensing redundancy and excessive power consumption, limiting their practicality on resource-constrained platforms. To address this inefficiency, we propose a predictive, history-aware adaptive scanning framework that anticipates informative regions of interest (ROI) based on past observations. Our approach introduces a lightweight predictor network that distills historical spatial and temporal contexts into refined query embeddings. These embeddings guide a differentiable Mask Generator network, which leverages Gumbel-Softmax sampling to produce binary masks identifying critical ROIs for the upcoming frame. Our method significantly reduces unnecessary data acquisition by concentrating dense LiDAR scanning only within these ROIs and sparsely sampling elsewhere. Experiments on nuScenes and Lyft benchmarks demonstrate that our adaptive scanning strategy reduces LiDAR energy consumption by over 65% while maintaining competitive or even superior 3D object detection performance compared to traditional LiDAR-camera fusion methods with dense LiDAR scanning.

## Installation

```bash
# Clone the repository
git clone https://github.com/sarashoouri/AdaptiveLiDAR.git
cd AdaptiveLiDAR

# Create conda environment
conda env create -f environment.yml
conda activate fusion

# Install the package
pip install -e .
```

## Data

Download Lyft from: https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles

Download Nuscenes from: https://www.nuscenes.org/nuscenes
