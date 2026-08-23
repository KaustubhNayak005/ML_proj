# Camouflage-Robust Graph Neural Networks for Fraud Ring Identification — Project Roadmap

## 0. How to use this document

**You (the student):** Read it once top to bottom so you know the shape of the whole
project. After that, use it as a checklist — each phase has a "Definition of Done"
you can tick off. The "Concepts to understand" boxes are your personal study list;
you should be able to explain those in your own words before moving on, regardless
of what the agent writes for you.

**Antigravity (the coding agent), if you're reading this file:** Work through the
phases in order. Do not start Phase *N+1* until Phase *N*'s Definition of Done is
satisfied. Each phase's task list is your task list for that session. Obey every
"Agent guardrails" note literally — they exist because of constraints the professor
set, not stylistic preferences. If a request from the user conflicts with a
guardrail (e.g. "just import GATConv to save time"), flag the conflict instead of
silently complying or silently refusing.

> **Non-negotiable rule for Phase 7:** do not port or closely translate code from
> the CARE-GNN or PC-GNN reference repositories. Read them, understand the
> mechanism, then design your own variant. This is a graded, original-work
> deliverable — copying a reference implementation defeats the point and risks
> an academic integrity problem, not just a weak grade.

---

## 1. Project summary

Build a fraud detector over a transaction graph (accounts/transactions as nodes,
relationships as edges) using Graph Neural Networks. Part 1 hand-codes GraphSAGE
and GAT from scratch as a foundation. Part 2 adds one genuinely new piece: a
mechanism that resists "camouflage" — fraud rings that deliberately connect
themselves to normal accounts to look legitimate. This is the actively-studied gap
in this field right now (see Appendix D).

**Novelty, precisely stated (revised):** camouflage-resistant GNNs were
originally built and tested on review-fraud graphs (Yelp/Amazon), but
transaction-graph benchmarks for this exact problem already exist too
(T-Finance, T-Social, S-FFSD — Appendix D), so "first to apply this to
transaction data" is not an accurate or defensible claim — don't pitch it that
way to your professor or in the report. What's still genuinely open on
**IEEE-CIS specifically**: no prior work runs a CARE-GNN/PC-GNN-style explicit
neighbor-filtering mechanism against from-scratch GraphSAGE/GAT baselines with a
proper multi-seed ablation on this dataset, isolating how much of any gain
comes from camouflage-resistance specifically versus attention alone. The
closest existing work (Appendix D — RL-GNN, 2025) combines GAT with an RL
controller directly on IEEE-CIS and reports 0.872 AUROC — a useful external
number to benchmark against in Phase 8, and a paper you need to explicitly
differentiate from in your report (it doesn't target camouflage/heterophily
resistance specifically, and doesn't ablate against baselines you built
yourself).

## 2. Top-level definition of done

- [ ] From-scratch GraphSAGE and GAT baselines, trained and evaluated on a real
      fraud dataset, with imbalance-aware metrics (not accuracy)
- [ ] One clearly-scoped novel extension, implemented, ablated, and compared
      fairly against the baselines on identical splits/seeds
- [ ] A codebase a stranger could clone and reproduce your headline number from
- [ ] A written report: motivation, related work, method, results, honest
      limitations
- [ ] All of the above fits inside your actual semester timeline

## 3. Tech stack

- Python 3.10+, PyTorch (CUDA build matching your GPU driver)
- Scatter/reduce ops: prefer **native PyTorch** (`torch.Tensor.scatter_reduce_`,
  `index_add_`) over the separate `torch_scatter` package. `torch_scatter`'s own
  maintainers note most of its functionality now lives in PyTorch directly, and
  the package is a common source of install pain (exact CUDA/torch/OS wheel
  matching). At this graph scale (≤~600K nodes) writing scatter-mean and
  scatter-softmax yourself with native ops is both more reliable and more in
  the spirit of Appendix C's "from scratch" scope — fall back to `torch_scatter`
  only if you hit a specific performance wall
- PyTorch Geometric **only** for its `Data`/`Dataset` container and any dataset
  download helpers (e.g. `EllipticBitcoinDataset`) — not for its `nn` layers
- pandas / numpy / scikit-learn for tabular EDA and metrics
- matplotlib for figures

## 4. Repository structure

```
fraud-gnn/
  data/
    raw/                  # untouched downloads
    processed/             # serialized graph objects
  src/
    data/                  # graph construction scripts
    models/
      layers/              # your from-scratch SAGEConv, GATConv
      baselines.py
      camo_module.py       # your novel Part 2 piece
    train.py
    evaluate.py
    utils/
  notebooks/                # EDA only — never production logic
  experiments/               # one config + result log per run
  tests/                     # unit tests for your hand-written layers
  report/
    figures/
  README.md
  requirements.txt
  ROADMAP.md                 # this file
```

---

## Phase 0 — Environment & repo setup
**~2–3 days**

Goal: a reproducible environment and a scaffold, before any modeling.

Tasks
- [ ] Create a conda/venv environment; install the PyTorch build matching your
      CUDA version
- [ ] `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_properties(0).total_memory)"` — confirm the GPU is visible
- [ ] Initialize git; create the folder structure in §4
- [ ] Write a `set_seed(seed)` utility used everywhere (Python, numpy, torch, cuda)
- [ ] Stub `README.md` with the one-paragraph project summary

Concepts to understand: why CUDA/driver mismatches happen; why seeding every RNG
source (not just `torch.manual_seed`) matters for reproducible results.

Agent guardrails: pin exact package versions in `requirements.txt` as you install
them — don't let this drift silently later.

Definition of Done: GPU check prints `True` and a plausible memory figure; repo
pushed with the scaffold folders (empty is fine) committed.

---

## Phase 1 — Problem framing & dataset decision
**~3–4 days**

Goal: lock the dataset and write down precisely what you're building, before
writing model code.

**Decision point — IEEE-CIS vs. Elliptic** (full comparison in Appendix A).
Default recommendation: **IEEE-CIS**, because constructing the multi-relation
graph yourself is real learning *and* sets up the camouflage angle in Phase 7
naturally — a fraud ring sharing a device or card is a direct camouflage signal.
Elliptic is already a graph (less construction work, less to learn there) and is a
reasonable fallback if graph construction eats too much of your timeline.

Tasks
- [ ] Download the chosen dataset (Kaggle for both; Elliptic also available via
      `torch_geometric.datasets.EllipticBitcoinDataset`)
- [ ] EDA notebook: class balance, missing values, feature types; for IEEE-CIS
      specifically, check the cardinality of candidate "shared entity" columns
      (`card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`,
      `DeviceInfo`)
- [ ] Write a one-paragraph problem statement and a one-paragraph novelty
      statement — this becomes tomorrow's pitch to your professor and later the
      intro of your report

Concepts to understand: why accuracy is a bad headline metric under class
imbalance — concretely, a model that always predicts "not fraud" already scores
~96.5% on IEEE-CIS (20,663 fraud / 590,540 total) and ~90.2% on Elliptic *if
evaluated on labeled nodes only* (4,545 illicit / 46,564 labeled — the higher
97%+ figure sometimes quoted only holds if you count the unlabeled 77% as
implicit negatives, which isn't standard practice and won't match how you'll
actually evaluate in Phase 5); why a **random** train/test split can leak
information in transaction data (better: time-based split, especially for
Elliptic's 49 timesteps).

Agent guardrails: this phase is analysis only — no training yet.

Definition of Done: EDA notebook committed; problem + novelty statements written;
dataset choice locked in `README.md`.

---

## Phase 2 — Graph construction pipeline
**~1 week**

Goal: a deterministic script that turns raw data into a graph object your models
can consume, run once, reused by every later phase.

**If IEEE-CIS:**
- [ ] Decide node scope: transactions as the only node type (simplest — shared
      entities become *relation types*, i.e. edges), vs. a fully heterogeneous
      graph with separate card/device/email nodes. Pick one and write down why —
      this is a real design decision, not busywork.
- [ ] Build one edge relation per shared-entity type (same card1+card2 → edge;
      same DeviceInfo → edge; same email domain → edge). This mirrors CARE-GNN's
      multi-relation design (it used relations like same-user / same-time /
      same-star on review data) — you're doing the same idea on transaction data.
- [ ] **Cap degree on hub entities.** A shared value like `gmail.com` will connect
      a huge fraction of all transactions if you don't cap it — this is a known,
      easy-to-miss gotcha that silently turns your graph into a near-clique.
- [ ] Serialize: per-relation `edge_index` tensors, node feature matrix, label
      vector, train/val/test masks → disk, so nothing downstream re-parses raw CSVs.

**If Elliptic:**
- [ ] Load the provided node features / edges / labels directly
- [ ] Build a **time-based** split (paper convention: train on early timesteps,
      test on later ones — e.g. steps 1–34 train, 35–49 test) — not a random split
- [ ] Decide how to handle the ~77% of nodes with unknown labels: drop them, or
      keep them for a semi-supervised setup — pick one and justify it

Concepts to understand: `edge_index` vs. dense adjacency representation; why hub
nodes distort message passing; transductive vs. inductive setting.

Agent guardrails: every threshold or cap you pick (degree cap, which columns count
as "shared entity") goes into a config value with a comment explaining the choice
— not a magic number buried in code. This script must be re-runnable end-to-end
from one command.

Definition of Done: `python src/data/build_graph.py` goes from raw file to saved
graph object in one run, printing node count, edge count per relation, and class
balance at the end.

---

## Phase 3 — GraphSAGE from scratch
**~1 week**

Goal: hand-implement mean-aggregation message passing and a 2-layer SAGE model.

Concepts to understand before coding:
- Neighbor sampling, and why full-batch training doesn't scale to large graphs
  in general — though at your actual hardware (24GB VRAM, 128GB RAM) full-batch
  is likely feasible for both datasets at this scale (≤~600K nodes); PC-GNN's
  own paper ran comparable-scale experiments on 128GB RAM with no GPU-memory
  issues reported. Implement sampling anyway for the learning value and so your
  pipeline generalizes, but don't let it block progress if full-batch trains
  cleanly first
- The update rule: new embedding for node *v* = `σ(W · CONCAT(h_v, AGG({h_u for u in neighbors(v)})))`
- Why the "mean aggregator" is *not* the same as the GCN aggregator (different
  self-loop and normalization handling)

Tasks
- [ ] Implement a `SAGEConv` layer by hand using primitive tensor ops (see
      Appendix C for what "by hand" allows)
- [ ] Stack two layers, add a binary classification head
- [ ] Unit test on a tiny synthetic 5-node graph: check output shape, and that
      `loss.backward()` runs cleanly with nonzero gradients

Agent guardrails: **do not** import `torch_geometric.nn.SAGEConv`,
`dgl.nn.SAGEConv`, or any prebuilt message-passing layer. Comment each line of
the layer with which part of the formula above it implements.

Definition of Done: unit tests pass; trains without NaN loss on a small
subsample within a few minutes.

---

## Phase 4 — GAT from scratch
**~1 week**

Goal: hand-implement attention-based aggregation.

Concepts to understand: the attention-coefficient formula
`e_ij = LeakyReLU(a^T [W·h_i || W·h_j])`, softmax-normalized over each node's
neighborhood; multi-head attention as several independent attention computations
concatenated together.

Tasks
- [ ] Implement `GATConv` by hand — single head first, then extend to multi-head
- [ ] Unit test: attention weights sum to 1 across each node's neighborhood
- [ ] Sanity-visualize attention weights on a handful of nodes — are they
      spread out and meaningful, or collapsing to near-uniform?

Agent guardrails: same rule as Phase 3 — no `GATConv` import, ever.

Definition of Done: attention-sums-to-1 test passes; training is stable (loss
decreases, no NaNs).

---

## Phase 5 — Baseline training & evaluation harness
**~4–5 days**

Goal: a rigorous, reusable train/eval loop *before* touching the novel idea, so
Phase 7 has a trustworthy number to beat.

Tasks
- [ ] Handle class imbalance: class-weighted BCE loss at minimum; consider focal
      loss if weighting alone underperforms
- [ ] Metrics: **PR-AUC as the primary metric**, plus ROC-AUC, F1 at a chosen
      threshold, and recall at a fixed precision — accuracy is reported only as a
      footnote, never as the headline number
- [ ] Config-driven `train.py --config configs/sage_baseline.yaml`, logging
      metrics every epoch
- [ ] Run both baselines to convergence; save a results table

Concepts to understand: why accuracy misleads under this level of class
imbalance (see Phase 1 for the exact per-dataset figures); early-stopping on
PR-AUC rather than raw loss.

Agent guardrails: every run's config and metrics get logged under
`experiments/<run-name>/` — nothing lives only in terminal output. Record the
seed used for each run.

Definition of Done: a checked-in results table (model, PR-AUC, ROC-AUC, F1,
recall) for both baselines.

---

## Phase 6 — Literature deep-dive: camouflage-resistant GNNs
**~4–5 days, can run in parallel with Phase 4–5**

Goal: understand 3–4 papers' actual mechanisms well enough to explain them
without notes. This is what makes Phase 7 a real contribution instead of a
reskin of someone else's idea.

Required reading (write a ≤1-page mechanism summary per paper, in your own
words — this becomes part of your report's related-work section):
- **CARE-GNN** (Dou et al., CIKM 2020) — filters which neighbors get aggregated
  per relation using a label-aware similarity measure, with the similarity
  threshold adapted during training via a reinforcement-learning module
- **PC-GNN** (Liu et al., WWW 2021) — a node-level resampler ("pick and choose")
  combined with a label-aware neighbor selector, aimed at the class-imbalance
  side specifically
- **RL-GNN** (Scientific Reports, 2025) — required, not optional. This is the
  closest existing work to your Phase 7 (GAT + RL controller, evaluated on
  IEEE-CIS directly, 0.872 AUROC). You need to be able to state precisely, in
  one paragraph, how your approach differs from this specific paper
- One more paper from Appendix D's "recent camouflage-specific work" list
  (PROD or SCFCRC are good picks — both explicitly target the combined
  feature-camouflage + relation-camouflage problem, close to your Phase 7
  framing)

Tasks
- [ ] Write the four mechanism summaries
- [ ] Write one clear paragraph stating exactly what you will do **differently**
      — a simplified or modified selection rule, a different similarity measure,
      combining ideas from two papers, the specific from-scratch/ablation angle
      — anything specific and defensible. "Applying this to transaction data"
      alone is *not* a valid answer anymore (see the revised novelty note in
      §1) — be precise about what's actually new

Agent guardrails: this phase produces prose notes, not code. If asked to
"implement CARE-GNN," push back and confirm scope with the user first — Phase 7
must be an original variant, not a port of a reference repo.

Definition of Done: four mechanism summaries committed; one paragraph on your
specific proposed twist, checked against "is this actually different, and can I
defend that in five minutes to my professor."

---

## Phase 7 — Design & implement the novel module
**~2–2.5 weeks**

Goal: your actual contribution.

Recommended default direction: a camouflage-resistant neighbor-selection
mechanism layered on top of your Phase 4 GAT, applied to IEEE-CIS. CARE-GNN and
PC-GNN's original papers validate on review-fraud graphs (Yelp/Amazon); the
defensible novelty claim here isn't "first on transaction data" (it isn't —
T-Finance/T-Social/S-FFSD already cover that ground, see Appendix D) but the
specific combination you're running: an explicit, ablated neighbor-filtering
mechanism, benchmarked against from-scratch GraphSAGE/GAT baselines you built
yourself, on IEEE-CIS specifically, with a direct comparison point against the
2025 GAT+RL result (Appendix D — RL-GNN). Carry this exact framing into your
report's contribution statement — it's more precise than the original pitch and
it holds up against a literature-aware reader.

Pick **one** of these starting mechanisms and adapt it — don't try to build all
three:
- [ ] **Similarity-gated attention** — before computing GAT attention, compute a
      feature-similarity score per node pair and down-weight or mask edges below
      a threshold (learned or heuristic)
- [ ] **Per-relation adaptive filtering** — for each relation type from Phase 2,
      learn a separate filtering rule (same spirit as CARE-GNN's per-relation
      similarity measure, but you define your own scoring function and update
      rule — don't copy theirs)
- [ ] **Label-aware contrastive term** — an auxiliary loss that pulls same-label
      neighbor embeddings together and pushes different-label pairs apart, making
      it structurally harder for a fraud node to hide inside a normal-looking
      neighborhood

Tasks
- [ ] Implement the chosen mechanism as a module wrapping/extending your Phase 4
      GAT
- [ ] Get it training end-to-end on a small subsample first for fast iteration,
      then on the full graph
- [ ] Compare against the Phase 5 baseline numbers on the **same split and seed**

Agent guardrails: keep the mechanism swappable behind a config flag so Phase 8's
ablations are config changes, not code forks.

Definition of Done: trains stably; is at least directionally comparable to the
baseline on PR-AUC. If it's worse, that is still a valid, reportable result as
long as you can explain why — flag this to the user rather than quietly tuning
until the number looks better.

---

## Phase 8 — Experiments & ablations
**~1–1.5 weeks**

Goal: turn one result into a defensible set of experiments.

Tasks
- [ ] Main comparison table: GraphSAGE, GAT, GAT + your module
- [ ] Ablation: your module with each key component removed, one at a time
- [ ] Run 3 random seeds per config; report mean ± std, never a single run
- [ ] **If using IEEE-CIS:** note RL-GNN's published 0.872 AUROC / 0.683 AP
      (Appendix D) alongside your table as an external reference point — not a
      strict apples-to-apples comparison (different splits/preprocessing almost
      certainly), but useful context, and expect your professor or a reviewer
      to ask how you compare to it
- [ ] **If using Elliptic:** explicitly check performance across the later,
      harder timesteps. Recent work has raised the concern that temporal
      distribution shift alone explains a meaningful chunk of apparent GNN gains
      on Elliptic — worth checking honestly rather than assuming the graph
      structure is doing all the work.

Agent guardrails: never hand-pick the best-looking seed as the headline number —
report the aggregate across seeds.

Definition of Done: results table plus 1–2 figures (a PR curve, or an ablation
bar chart) saved to `report/figures/`.

---

## Phase 9 — Error analysis (stretch goal)
**~3–5 days**

Goal: a qualitative story for your report/defense — *why* it works, not just
*that* it works.

Tasks
- [ ] Find cases the baseline got wrong that your module fixed, and vice versa
- [ ] Inspect attention weights on a handful of known-fraud nodes, before vs.
      after your module
- [ ] Optional: a simple explanation output — e.g. the top-k neighbors or
      relations that most influenced a flagged node's score

Definition of Done: 3–5 concrete examples, each with a short written explanation.

---

## Phase 10 — Report, reproducibility & final packaging
**~1.5–2 weeks, overlapping with Phase 8–9**

Tasks
- [ ] Write the report: motivation, related work (from Phase 6), method,
      experiments (Phase 8), results, limitations, honest discussion of what
      didn't work
- [ ] Clean the repo: final `README.md` with exact run commands, pinned
      `requirements.txt`, dead notebooks/code removed
- [ ] **Confirm a fresh clone + fresh environment reproduces your headline
      number.** This is the single most common thing that quietly breaks.
- [ ] Prepare a short talking-point summary for professor discussion — e.g.
      *"Normal fraud-detection AI gets fooled when criminals deliberately make
      themselves look normal — my project builds one that's harder to fool."*

Definition of Done: fresh-clone reproducibility check passes; report draft
complete; repo tagged (e.g. `v1.0-submission`).

---

## Appendix A — Dataset decision matrix

| Factor | IEEE-CIS | Elliptic |
|---|---|---|
| Graph readiness | Tabular — you build the graph (more work, more learning) | Already a graph (nodes/edges provided) |
| Size | ~590K transactions | 203,769 nodes, 234,355 edges |
| Features | Mixed transaction + identity fields, engineered "V" features | 166 numeric features (94 local + 72 neighbor-aggregate, already computed) — sources disagree by one (165 vs. 166); confirm with `print(data.x.shape)` once loaded rather than trusting either number |
| Labels | Binary `isFraud`, ~3–4% positive | 3-way (licit/illicit/unknown); ~2% illicit, 21% licit, 77% unknown |
| Split strategy | Time-based recommended | Time-based required (49 sequential steps) |
| Fit for the camouflage angle | Strong — shared card/device/email is a direct camouflage signal | Weaker — features are anonymized aggregates, less of an obvious "disguise" story |
| Main risk | Graph construction (hub nodes, relation design) eats your timeline | Structure-vs-temporal-shift confound (Appendix D) makes some GNN gains hard to attribute cleanly |

## Appendix B — Timeline (12-week default)

Adjust to your actual deadline — compress by dropping Phase 9, or stretch Phase 7
if your novel mechanism needs more iteration.

| Week | Phase(s) | Milestone |
|---|---|---|
| 1 | 0, 1 | Env ready, dataset locked, problem statement written |
| 2 | 2 | Graph construction script done |
| 3 | 3 | GraphSAGE from scratch, tested |
| 4 | 4, 6 (start) | GAT from scratch, tested; reading started |
| 5 | 5, 6 (finish) | Baseline results table; mechanism summaries done |
| 6–7 | 7 | Novel module implemented, training end-to-end |
| 8 | 7 (finish), 8 (start) | Novel module beats/matches baseline directionally |
| 9 | 8 | Ablations + multi-seed results done |
| 10 | 9 | Error analysis examples collected |
| 11–12 | 10 | Report written, repro check passed, submission packaged |

## Appendix C — "From scratch" scope clarification

**Allowed:** `torch.nn.Linear`, autograd, optimizers, `torch.sparse`, raw tensor
indexing, native PyTorch scatter/reduce ops (`scatter_reduce_`, `index_add_`) or
`torch_scatter`'s reduction functions as a fallback (see §3 — both are
primitive index-reduce operations, not models), and PyTorch Geometric's
`Data`/`Dataset` classes purely for loading/storing graphs.

**Not allowed:** `torch_geometric.nn.SAGEConv`, `GATConv`, or any other
prebuilt message-passing layer from PyG, DGL, or similar libraries — for either
the baselines or the novel module.

This line (data-loading utilities are fine, model layers are not) is a
reasonable reading of "no pretrained models," but it's still worth a one-line
confirmation from your professor early on, since "no pretrained models" is
slightly ambiguous about utility functions like this.

## Appendix D — Reading list (for citation, not for copying)

**Foundational methods (original camouflage-resistant GNNs):**
- CARE-GNN — Dou et al., *Enhancing Graph Neural Network-based Fraud Detectors
  against Camouflaged Fraudsters*, CIKM 2020. Code: github.com/YingtongDou/CARE-GNN
- PC-GNN — Liu et al., *Pick and Choose: A GNN-based Imbalanced Learning
  Approach for Fraud Detection*, WWW 2021
- H2-FDetector — Shi et al., *H2-FDetector: A GNN-based Fraud Detector with
  Homophilic and Heterophilic Connections*, WWW 2022 — separate aggregation
  strategies for homophilic vs. heterophilic connections
- GAGA — Wang et al., *Label Information Enhanced Fraud Detection against Low
  Homophily in Graphs*, WWW 2023 — group aggregation for distinguishable
  multi-hop neighborhood information

**Transaction-graph benchmarks and closest related work — read these before
finalizing your novelty paragraph, they directly constrain what you can claim:**
- T-Finance / T-Social — Tang et al., *Rethinking Graph Neural Networks for
  Anomaly Detection*, ICML 2022 — transaction/account-graph fraud benchmarks;
  establishes that "transaction data" alone is not the open gap
- S-FFSD — Xiang et al., *Semi-supervised Credit Card Fraud Detection via
  Attribute-driven Graph Representation*, AAAI 2023 — simulated credit-card
  transaction graph, same purpose as T-Finance
- **RL-GNN** — *Reinforcement learning with graph neural network (RL-GNN)
  fusion for real-time financial fraud detection*, Scientific Reports, Dec
  2025 — GAT + RL controller evaluated directly on IEEE-CIS, 0.872 AUROC /
  0.683 AP. The closest existing work to Phase 7 — required reading (Phase 6),
  and the paper you need to explicitly differentiate from in your report
- GADBench — benchmark paper standardizing evaluation of CARE-GNN, PC-GNN, and
  related methods; useful for baseline-comparison methodology

**Recent camouflage-specific work (2025) — for currency in your related-work
section:**
- PROD — *Projected and Orthogonal Disentanglement*, Knowledge-Based Systems,
  2025 — tackles scarce labeled data and camouflage jointly via risk-aware
  encoding and disentanglement
- SCFCRC — *Simultaneously Counteract Feature Camouflage and Relation
  Camouflage for Fraud Detection*, arXiv 2025 — directly targets both
  camouflage types together, same framing as your Phase 7 problem statement
- HA-GNN (2025 update) — argues CARE-GNN-style neighbor selectors handle
  relation camouflage well but degrade when feature camouflage is layered on
  top too — a legitimate, citable limitation of the baseline you're extending

**Datasets:**
- IEEE-CIS Fraud Detection dataset — Kaggle
- Elliptic Bitcoin dataset — Kaggle, or `torch_geometric.datasets.EllipticBitcoinDataset`

## Appendix E — Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Graph construction (Phase 2) takes longer than a week | Medium | Time-box it; fall back to Elliptic if you're not done by end of week 2 |
| Novel module (Phase 7) doesn't beat baseline | Medium | Still a valid, explainable result — budget time to analyze *why*, don't just keep tuning |
| Hub-node explosion makes the graph unusable | Medium–High (IEEE-CIS) | Degree caps from Phase 2, checked immediately after construction, not discovered mid-training |
| Running out of time for the report | High if left until the end | Start the related-work section in Phase 6, not Phase 10 |
| "From scratch" scope dispute with professor | Low, but costly if it happens | Confirm Appendix C's line with them in week 1 |
| Novelty claim challenged as "already done" (T-Finance/S-FFSD/RL-GNN exist) | Was High, now mitigated | Precise novelty statement in §1 and Phase 7 (this revision); RL-GNN added as required reading in Phase 6 so the differentiation paragraph is specific, not naive |
