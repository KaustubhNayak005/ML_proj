# Camouflage-Robust Graph Neural Networks for Fraud Ring Identification

## Problem Statement
Traditional fraud detection often treats transactions in isolation. However, sophisticated fraud rings operate in interconnected networks. While Graph Neural Networks (GNNs) can leverage these connections, fraudsters actively "camouflage" their behavior by linking themselves to normal, legitimate accounts. This project aims to build a graph-based fraud detector that explicitly resists this camouflaging behavior.

## Novelty
Our approach benchmarks a camouflage-resistant neighbor-selection mechanism against from-scratch GraphSAGE and GAT baselines. By carefully ablating this module on the IEEE-CIS transaction graph, we isolate how much of the performance gain is attributable strictly to the camouflage-resistance (neighbor filtering/adaptive similarity) versus attention alone, offering a direct structural comparison against recent GAT+RL baselines in the field.

## Setup Instructions
1. Create a Python environment.
2. Install the PyTorch build that matches your CUDA driver.
3. Run `pip install -r requirements.txt`.
4. Download the IEEE-CIS dataset from Kaggle to `data/raw/`. You need `train_transaction.csv` and `train_identity.csv`.
5. Run `python src/data/build_graph.py` to construct the graph.
