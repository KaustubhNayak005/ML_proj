# RESULTS NOT YET GENERATED
Awaiting a real Phase 8 run. Do not treat any numbers as final until this line is replaced.

## Known Limitations & Corrections for Final Draft
* **Relation-Awareness:** The current graph construction (`build_graph.py`) extracts multiple relation types (card1, card2, etc.), but the model (`CamouflageGATConv`) does not yet differentiate between them. All edge types are flattened into a single homogeneous graph before being passed to the model. This is a known architectural limitation.
* **Model Architecture (Correction):** The custom layer is a native PyTorch implementation, not a custom PyTorch Geometric (PyG) layer. 
* **Honest Framing (Correction):** Do not use language like "definitively prove" or "massive margin". The analysis must reflect the objective reality of the numbers, acknowledging if baselines (like GraphSAGE) perform better.
