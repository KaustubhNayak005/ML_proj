# Project Status Report

## Completed Phases
* **Phase 0 (Environment & Repo Setup):** 
  * Repository structure initialized.
  * Environment setup and `requirements.txt` generated.
  * Random seed utility implemented (`src/utils/seed.py`).
  * `README.md` stubbed with project summary and novelty statement.
* **Phase 1 (Problem framing & dataset decision):**
  * Dataset chosen: IEEE-CIS (documented in `README.md`).
  * Problem and novelty statements defined.
* **Phase 2 (Graph construction pipeline):**
  * Graph construction logic implemented in `src/data/build_graph.py`.
* **Phase 3 (GraphSAGE from scratch):**
  * SAGEConv layer implemented (`src/models/layers/sage.py`).
  * Unit tests written (`tests/test_sage.py`).
* **Phase 4 (GAT from scratch):**
  * GATConv layer implemented (`src/models/layers/gat.py`).
  * Unit tests written (`tests/test_gat.py`).
* **Phase 7 (Design & implement the novel module):** *(Partially Completed, executed out of order)*
  * Camouflage GAT module implemented (`src/models/layers/camouflage_gat.py`).
  * Unit tests written (`tests/test_camouflage.py`).

## Missing / Incomplete Tasks
* **Phase 1 (Problem framing):**
  * Missing the EDA notebook (`notebooks/` is empty).
* **Phase 5 (Baseline training & evaluation harness):**
  * `metrics.py` and `test_train_loop.py` are present, but the main `train.py` and `evaluate.py` harnesses are missing.
  * Need to run baselines to convergence and save a results table.
* **Phase 6 (Literature deep-dive: camouflage-resistant GNNs):**
  * Missing the 3-4 paper mechanism summaries in the `report/` directory.
  * Missing the paragraph detailing the specific proposed twist for Phase 7.
* **Phase 8 (Experiments & ablations):**
  * Main comparison table and ablations are yet to be run (the `experiments/` directory is empty).
  * Report writing is pending.

## What is Next
1. **Complete Phase 1 (EDA):** Create the required EDA notebook analyzing class balance, missing values, and shared entity cardinality for the IEEE-CIS dataset.
2. **Implement Phase 5 (Training Harness):** Build `train.py` and `evaluate.py` to establish the baseline performance for GraphSAGE and GAT. This is a prerequisite before properly testing the camouflage module.
3. **Complete Phase 6 (Literature Review):** Perform the literature review and write the required mechanism summaries.
4. **Integrate Phase 7 & 8:** Integrate the existing `camouflage_gat.py` into the training harness and run the full ablation studies.
