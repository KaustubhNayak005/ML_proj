# Project Status Report

## Completed Phases
* **Phase 0 (Environment & Repo Setup):** 
  * Repository structure initialized, GitHub repo created, `.gitignore` setup.
  * Environment setup and `requirements.txt` generated.
  * Random seed utility implemented (`src/utils/seed.py`).
  * `README.md` stubbed with project summary and novelty statement.
* **Phase 1 (Problem framing & dataset decision):**
  * Dataset chosen: IEEE-CIS (documented in `README.md`).
  * Problem and novelty statements defined.
  * **[Completed]** EDA notebook (`notebooks/eda.ipynb`) created analyzing class imbalance and feature cardinality.
* **Phase 2 (Graph construction pipeline):**
  * Graph construction logic implemented in `src/data/build_graph.py`.
* **Phase 3 (GraphSAGE from scratch):**
  * SAGEConv layer implemented (`src/models/layers/sage.py`).
  * Unit tests written (`tests/test_sage.py`).
* **Phase 4 (GAT from scratch):**
  * GATConv layer implemented (`src/models/layers/gat.py`).
  * Unit tests written (`tests/test_gat.py`).
* **Phase 5 (Baseline training & evaluation harness):**
  * **[Completed]** `train.py` and `evaluate.py` created.
  * **[Completed]** Logging and `configs/` structure established for ablations.
  * **[Completed]** Class imbalance handling via `pos_weight` in BCE loss.
* **Phase 6 (Literature deep-dive: camouflage-resistant GNNs):**
  * **[Completed]** `report/related_work.md` written, including mechanism summaries and our proposed novelty.
* **Phase 7 (Design & implement the novel module):** 
  * Camouflage GAT module implemented (`src/models/layers/camouflage_gat.py`).
  * Unit tests written and passing (`tests/test_camouflage.py`).

## Missing / Incomplete Tasks
* **Phase 8 (Experiments & ablations):**
  * Actual model training on the full graph needs to be run.
  * Run the ablation configurations (`sage_baseline.yaml`, `gat_baseline.yaml`, `camo_gat.yaml`) across 3 random seeds.
  * The final report synthesizing results, figures, and limitations needs to be written.

## What is Next
1. **Run Experiments (Phase 8):** Download the full dataset locally (or attach to a Kaggle session) and run the `train.py` script for each config. 
2. **Compile Results:** Record the mean ± standard deviation for the PR-AUC and ROC-AUC scores across the 3 seeds for each configuration.
3. **Write the Final Report:** Combine the novelty statement, related work, and experiment results into a final deliverable report.
