# Camouflage-Robust Graph Neural Networks for Fraud Ring Identification — Free-Compute Roadmap

## 0. How to use this document

**This is the sister document to `ROADMAP.md`, not a replacement for it.** Same
project, same novelty claim, same phases, same academic-integrity rules. The
only thing that changes is where the code actually runs: this version assumes
**no dedicated GPU at all** — no lab machine, no paid cloud — only Kaggle's and
Colab's free tiers plus whatever CPU you have locally. Use this version if lab
access falls through, is unreliably remote, or never materializes.

**You (the student):** Read §4 (Compute strategy) before anything else — it's
the one genuinely new idea in this version, and every phase below assumes you
already understand it.

**Antigravity, if you're reading this file:** Your operating boundary changed.
You have no way to execute anything on Kaggle or Colab directly — there is no
SSH or remote-dev surface into either platform for you to reach (and trying to
tunnel one open, e.g. via `colab-ssh`-style tools, is against Colab's terms and
risks the user losing free-tier access entirely — never suggest it). Your job
in the compute-heavy phases (0, 2, 5, 7, 8) is to prepare scripts that are
correct and ready to run unattended the first time, not to run them yourself.
Everything else — work through phases in order, don't start Phase *N+1* before
Phase *N*'s Definition of Done, obey every "Agent guardrails" note literally —
is unchanged from `ROADMAP.md`.

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

**Novelty, precisely stated:** camouflage-resistant GNNs were originally built
and tested on review-fraud graphs (Yelp/Amazon), but transaction-graph
benchmarks for this exact problem already exist too (T-Finance, T-Social,
S-FFSD — Appendix D), so "first to apply this to transaction data" is not an
accurate or defensible claim — don't pitch it that way to your professor or in
the report. What's still genuinely open on **IEEE-CIS specifically**: no prior
work runs a CARE-GNN/PC-GNN-style explicit neighbor-filtering mechanism against
from-scratch GraphSAGE/GAT baselines with a proper multi-seed ablation on this
dataset, isolating how much of any gain comes from camouflage-resistance
specifically versus attention alone. The closest existing work (Appendix D —
RL-GNN, 2025) combines GAT with an RL controller directly on IEEE-CIS and
reports 0.872 AUROC — a useful external number to benchmark against in Phase 8,
and a paper you need to explicitly differentiate from in your report (it
doesn't target camouflage/heterophily resistance specifically, and doesn't
ablate against baselines you built yourself).

## 2. Top-level definition of done

- [ ] From-scratch GraphSAGE and GAT baselines, trained and evaluated on a real
      fraud dataset, with imbalance-aware metrics (not accuracy)
- [ ] One clearly-scoped novel extension, implemented, ablated, and compared
      fairly against the baselines on identical splits/seeds
- [ ] A codebase a stranger could clone and reproduce your headline number from
      — across **both** a local CPU environment and a fresh Kaggle notebook
- [ ] A written report: motivation, related work, method, results, honest
      limitations (including compute constraints, if they shaped your ablation
      scope — see Phase 8)
- [ ] All of the above fits inside your actual semester timeline **and** your
      free-tier GPU-hour budget

## 3. Tech stack

- Python 3.10+. **Local: CPU-only PyTorch build** (no CUDA matching needed —
  you're not training real jobs here). Kaggle and Colab notebooks ship with a
  CUDA-enabled PyTorch already installed, so there's no CUDA setup step on the
  remote side either.
- Scatter/reduce ops: prefer **native PyTorch** (`torch.Tensor.scatter_reduce_`,
  `index_add_`) over the separate `torch_scatter` package. `torch_scatter`'s own
  maintainers note most of its functionality now lives in PyTorch directly, and
  the package is a common source of install pain (exact CUDA/torch/OS wheel
  matching) — pain you especially don't want to hit inside a Kaggle/Colab
  session where you can't easily fix a broken environment and try again later.
  At this graph scale (≤~600K nodes) writing scatter-mean and scatter-softmax
  yourself with native ops is both more reliable and more in the spirit of
  Appendix C's "from scratch" scope.
- PyTorch Geometric **only** for its `Data`/`Dataset` container and any dataset
  download helpers (e.g. `EllipticBitcoinDataset`) — not for its `nn` layers
- pandas / numpy / scikit-learn for tabular EDA and metrics
- matplotlib for figures

## 4. Compute strategy

This is the section that makes this a different roadmap, not just a shorter one.

**The core loop:** develop and debug locally, on CPU, against a small
subsample. Only hand off to Kaggle or Colab once a script is finished and
you're confident it'll run correctly unattended. GPU time is a rationed
resource here, not a given — treat every remote run like it costs something,
because it does.

- **Kaggle is primary.** ~30 GPU-hours/week, published and consistent. GPU is
  a P100 (16GB) or 2×T4 (16GB each, not unified — don't write multi-GPU code
  for this, it's not worth the complexity at this graph scale). 12-hour session
  cap. No credit card needed.
- **Colab is overflow, not primary.** Free-tier quota is dynamic and
  unpublished — Google's own FAQ says GPU access is "heavily restricted" for
  non-paying users, and you can be denied a GPU entirely on a busy day. Use it
  when Kaggle's weekly quota runs dry, not as your main platform, since you
  can't plan around a number you can't see in advance.
- **Never try to SSH or tunnel into either platform** to get Antigravity live
  access. Colab explicitly prohibits this, and the community tools built
  around it warn it can get your account restricted — not a risk worth taking
  on your only free compute for the semester.
- **Moving data in:** package your Phase 2 graph output as a private Kaggle
  Dataset (not just a notebook output file) so any future notebook — this
  session, next week's, a Colab session via a Drive copy — can attach it
  without re-running construction or re-uploading raw CSVs. Both IEEE-CIS and
  Elliptic are already available as Kaggle-native competition/dataset data, so
  "Add Data" inside a Kaggle notebook gets you the raw files with no manual
  download step at all — you mainly need to `pip install kaggle` and download
  locally for the small subsample you use in local development.
- **Moving results out:** checkpoint frequently (every N epochs, not just at
  the end) and save to your Kaggle Dataset or Drive incrementally — sessions
  can disconnect before a run finishes, and losing an unsaved 3-hour run to a
  timeout is the single most avoidable waste of quota in this plan.
- **Budget by measuring, not guessing.** The first real task in Phase 5 is
  timing one full epoch on the real data, on the real platform. Multiply that
  by planned epochs and configs to get your actual GPU-hour budget — nobody,
  including this document, can tell you that number in advance without
  knowing your exact batch size and model size.

## 5. Repository structure

```
fraud-gnn/
  data/
    raw/                  # small local subsample only — full data lives on Kaggle
    processed/             # serialized graph objects (also packaged as a Kaggle Dataset)
  src/
    data/                  # graph construction scripts
    models/
      layers/              # your from-scratch SAGEConv, GATConv
      baselines.py
      camo_module.py       # your novel Part 2 piece
    train.py                # device-agnostic: same file runs locally (CPU, subsample)
                             # or on Kaggle/Colab (GPU, full data) via a config flag
    evaluate.py
    utils/
  notebooks/                # EDA only — never production logic
  experiments/               # one config + result log per run, downloaded back locally
  tests/                     # unit tests for your hand-written layers — run locally, free
  report/
    figures/
  README.md
  requirements.txt            # keep a requirements-cpu.txt if versions diverge from Kaggle's image
  ROADMAP.md                  # the main (lab-GPU) roadmap
  ROADMAP_FREE_COMPUTE.md      # this file
```

Keep `train.py` and friends **device-agnostic** (`.to(device)` throughout, device
chosen by a config flag or auto-detected) so the exact same file you tested
locally on CPU is the file you paste or upload to Kaggle — no separate
"Kaggle version" to maintain and let drift out of sync.

---

## Phase 0 — Environment & repo setup
**~3–4 days** (a bit longer than the lab-GPU version — you're setting up two
environments, not one)

Goal: a reproducible local environment, verified Kaggle access, and a scaffold,
before any modeling.

Tasks
- [ ] Create a conda/venv environment locally with the **CPU** PyTorch build
- [ ] Initialize git; create the folder structure in §5
- [ ] Write a `set_seed(seed)` utility used everywhere (Python, numpy, torch, cuda)
- [ ] Stub `README.md` with the one-paragraph project summary, noting this repo
      targets free-tier compute (link both roadmap files)
- [ ] Create a Kaggle account if you don't have one; accept the IEEE-CIS
      competition rules (required even though it's closed — skipping this
      breaks both manual and API downloads with a 403)
- [ ] **Smoke test:** open a new Kaggle notebook, enable a GPU, run
      `!nvidia-smi`, confirm it returns a real GPU — this is the remote
      equivalent of the old "confirm the GPU is visible" check
- [ ] Create/confirm a Google account for Colab as the overflow platform (no
      setup needed beyond having the account — you'll configure it if and when
      you actually need the overflow capacity)

Concepts to understand: why CUDA/driver mismatches happen (even though you're
not managing this locally anymore, you'll see it in Kaggle/Colab error messages
occasionally); why seeding every RNG source (not just `torch.manual_seed`)
matters for reproducible results.

Agent guardrails: pin exact package versions in `requirements.txt` as you
install them locally. Don't assume Kaggle's pre-installed package versions
match — if a version-sensitive bug shows up only on Kaggle, check
`!pip list` there before assuming your local code is wrong.

Definition of Done: local CPU environment works (`import torch` succeeds); a
Kaggle notebook successfully shows a GPU via `nvidia-smi`; repo pushed with the
scaffold folders committed.

---

## Phase 1 — Problem framing & dataset decision
**~3–4 days**

Goal: lock the dataset and write down precisely what you're building, before
writing model code. Unchanged from `ROADMAP.md` — nothing about this phase is
compute-dependent.

**Decision point — IEEE-CIS vs. Elliptic** (full comparison in Appendix A).
Default recommendation: **IEEE-CIS**, because constructing the multi-relation
graph yourself is real learning *and* sets up the camouflage angle in Phase 7
naturally — a fraud ring sharing a device or card is a direct camouflage signal.
Elliptic is already a graph (less construction work, less to learn there) and is a
reasonable fallback if graph construction eats too much of your timeline.

Tasks
- [ ] Download a **small local subsample** via the Kaggle API (full files stay
      on Kaggle — see §4). Enough rows for EDA and later unit-testing, not the
      full 590K
- [ ] EDA notebook: class balance, missing values, feature types; for IEEE-CIS
      specifically, check the cardinality of candidate "shared entity" columns
      (`card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`,
      `DeviceInfo`). If your local machine struggles with even the subsample,
      do this step in a free Kaggle **CPU** session instead — CPU sessions
      don't touch your GPU-hour quota at all
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
**~1–1.5 weeks** (a little longer than the lab-GPU version — add a day for
packaging the output for remote use)

Goal: a deterministic script that turns raw data into a graph object your models
can consume, run once, reused by every later phase — and packaged so Kaggle/Colab
sessions can load it without re-running construction.

**If IEEE-CIS:**
- [ ] Decide node scope: transactions as the only node type (simplest — shared
      entities become *relation types*, i.e. edges), vs. a fully heterogeneous
      graph with separate card/device/email nodes. Pick one and write down why
- [ ] Build one edge relation per shared-entity type (same card1+card2 → edge;
      same DeviceInfo → edge; same email domain → edge)
- [ ] **Cap degree on hub entities.** A shared value like `gmail.com` will connect
      a huge fraction of all transactions if you don't cap it
- [ ] Serialize: per-relation `edge_index` tensors, node feature matrix, label
      vector, train/val/test masks → disk

**If Elliptic:**
- [ ] Load the provided node features / edges / labels directly
- [ ] Build a **time-based** split (steps 1–34 train, 35–49 test) — not a random split
- [ ] Decide how to handle the ~77% of nodes with unknown labels: drop them, or
      keep them for a semi-supervised setup

**Both paths — new for this version:**
- [ ] Run the full-scale construction on a Kaggle **CPU** session (this is
      data wrangling, not model training — no GPU needed, and it keeps this
      step off your GPU-hour budget entirely)
- [ ] Package the serialized graph as a **private Kaggle Dataset**. This is
      what every later GPU session attaches instead of re-parsing raw CSVs

Concepts to understand: `edge_index` vs. dense adjacency representation; why hub
nodes distort message passing; transductive vs. inductive setting.

Agent guardrails: every threshold or cap you pick goes into a config value with
a comment explaining the choice. This script must be re-runnable end-to-end
from one command, and it must run correctly as a plain script (not just
interactively in a notebook), since that's what you'll be uploading.

Definition of Done: construction script runs cleanly on Kaggle CPU end-to-end;
resulting graph is packaged as a private Kaggle Dataset and successfully
attached to a fresh test notebook.

---

## Phase 3 — GraphSAGE from scratch
**~1 week**

Goal: hand-implement mean-aggregation message passing and a 2-layer SAGE model.
Entirely local, CPU, free — this phase doesn't touch your GPU budget at all.

Concepts to understand before coding:
- Neighbor sampling / mini-batching, and why it matters here specifically:
  unlike the lab-GPU version of this plan, you're not just implementing this
  for the learning value — mini-batching is what keeps a training run inside
  Kaggle's 16GB and inside a time budget you can actually afford out of a
  30-hour weekly quota. A full-batch run that happens to fit in memory but
  takes 4 hours is a much worse idea here than it would be on unlimited local
  hardware
- The update rule: new embedding for node *v* = `σ(W · CONCAT(h_v, AGG({h_u for u in neighbors(v)})))`
- Why the "mean aggregator" is *not* the same as the GCN aggregator (different
  self-loop and normalization handling)

Tasks
- [ ] Implement a `SAGEConv` layer by hand using primitive tensor ops (see
      Appendix C for what "by hand" allows)
- [ ] Stack two layers, add a binary classification head
- [ ] Implement neighbor sampling / mini-batching — not optional in this version
- [ ] Unit test on a tiny synthetic 5-node graph: check output shape, and that
      `loss.backward()` runs cleanly with nonzero gradients — runs in seconds
      on CPU, no reason to ever run this test on Kaggle

Agent guardrails: **do not** import `torch_geometric.nn.SAGEConv`,
`dgl.nn.SAGEConv`, or any prebuilt message-passing layer. Comment each line of
the layer with which part of the formula above it implements. Write the whole
layer and training loop with `.to(device)` throughout so it needs zero changes
to run on Kaggle later.

Definition of Done: unit tests pass locally; trains without NaN loss on a small
local subsample within a few minutes on CPU.

---

## Phase 4 — GAT from scratch
**~1 week**

Goal: hand-implement attention-based aggregation. Same as Phase 3 — entirely
local, CPU, free.

Concepts to understand: the attention-coefficient formula
`e_ij = LeakyReLU(a^T [W·h_i || W·h_j])`, softmax-normalized over each node's
neighborhood; multi-head attention as several independent attention computations
concatenated together.

Tasks
- [ ] Implement `GATConv` by hand — single head first, then extend to multi-head
- [ ] Unit test: attention weights sum to 1 across each node's neighborhood
- [ ] Sanity-visualize attention weights on a handful of nodes

Agent guardrails: same rule as Phase 3 — no `GATConv` import, ever.

Definition of Done: attention-sums-to-1 test passes locally; training is
stable on a local subsample (loss decreases, no NaNs).

---

## Phase 5 — Baseline training & evaluation harness
**~1–1.5 weeks** (longer than the lab-GPU version — this is where remote
execution actually starts, and first-time setup friction is real)

Goal: a rigorous, reusable train/eval loop *before* touching the novel idea,
proven correct locally, then handed off to Kaggle for the real runs.

Tasks
- [ ] Handle class imbalance: class-weighted BCE loss at minimum; consider focal
      loss if weighting alone underperforms
- [ ] Metrics: **PR-AUC as the primary metric**, plus ROC-AUC, F1 at a chosen
      threshold, and recall at a fixed precision — accuracy is reported only as a
      footnote, never as the headline number
- [ ] Config-driven `train.py --config configs/sage_baseline.yaml`, logging
      metrics every epoch, checkpointing every N epochs (not just at the end)
- [ ] **Prove it locally first:** run the full harness on your local subsample
      on CPU until you're confident it's correct — loss goes down, metrics
      compute without errors, checkpointing/resuming works
- [ ] **First remote run, timed:** on Kaggle, attach your Phase 2 Dataset, run
      one epoch on the real data, and record the wall-clock time. This is the
      number that turns §4's budgeting advice into an actual plan for this
      phase and Phase 8
- [ ] Run both baselines to convergence on Kaggle; download results and
      checkpoints back into `experiments/` locally

Concepts to understand: why accuracy misleads under this level of class
imbalance (see Phase 1); early-stopping on PR-AUC rather than raw loss.

Agent guardrails: your role in the Kaggle steps is to prepare a script correct
enough to run unattended — you cannot execute it yourself. Every run's config
and metrics get logged under `experiments/<run-name>/` once the user downloads
them back. Record the seed used for each run.

Definition of Done: a checked-in results table (model, PR-AUC, ROC-AUC, F1,
recall) for both baselines, run on Kaggle, with checkpoints saved to your
Kaggle Dataset or downloaded locally — and a measured per-epoch time you can
use to budget Phase 8.

---

## Phase 6 — Literature deep-dive: camouflage-resistant GNNs
**~4–5 days, can run in parallel with Phase 4–5**

Goal: understand 3–4 papers' actual mechanisms well enough to explain them
without notes. Unchanged from `ROADMAP.md` — pure reading and writing, zero
compute dependency.

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
  (PROD or SCFCRC are good picks)

Tasks
- [ ] Write the four mechanism summaries
- [ ] Write one clear paragraph stating exactly what you will do **differently**
      — "applying this to transaction data" alone is *not* a valid answer (see
      §1) — be precise about what's actually new

Agent guardrails: this phase produces prose notes, not code. If asked to
"implement CARE-GNN," push back and confirm scope with the user first.

Definition of Done: four mechanism summaries committed; one paragraph on your
specific proposed twist, checked against "is this actually different, and can I
defend that in five minutes to my professor."

---

## Phase 7 — Design & implement the novel module
**~2.5–3 weeks** (extended from the lab-GPU version's 2–2.5 — this is your
most iteration-heavy phase, and iteration against a rationed GPU is slower
going)

Goal: your actual contribution.

Recommended default direction: a camouflage-resistant neighbor-selection
mechanism layered on top of your Phase 4 GAT, applied to IEEE-CIS. The
defensible novelty claim isn't "first on transaction data" (see §1) but the
specific combination you're running: an explicit, ablated neighbor-filtering
mechanism, benchmarked against from-scratch GraphSAGE/GAT baselines you built
yourself, on IEEE-CIS specifically, with a direct comparison point against the
2025 GAT+RL result (Appendix D — RL-GNN).

Pick **one** of these starting mechanisms and adapt it:
- [ ] **Similarity-gated attention** — before computing GAT attention, compute a
      feature-similarity score per node pair and down-weight or mask edges below
      a threshold (learned or heuristic)
- [ ] **Per-relation adaptive filtering** — for each relation type from Phase 2,
      learn a separate filtering rule (your own scoring function — don't copy
      CARE-GNN's)
- [ ] **Label-aware contrastive term** — an auxiliary loss that pulls same-label
      neighbor embeddings together and pushes different-label pairs apart

Tasks
- [ ] Implement the chosen mechanism as a module wrapping/extending your Phase 4
      GAT
- [ ] Get it training end-to-end **locally, on the subsample** first — do as
      much correctness debugging here as physically possible before spending
      any Kaggle time
- [ ] Only once it's stable locally, move to Kaggle for the full graph.
      **Batch your iteration:** if you have several small variants to test,
      queue them within a single Kaggle session rather than starting a fresh
      session per idea — you're paying per hour, not per run
- [ ] Compare against the Phase 5 baseline numbers on the **same split and seed**

Agent guardrails: keep the mechanism swappable behind a config flag so Phase 8's
ablations are config changes, not code forks. Never suggest debugging a new
idea by iterating directly on Kaggle — that's local-subsample work by
definition; Kaggle is for confirmed-stable code only.

Definition of Done: trains stably on Kaggle; is at least directionally
comparable to the baseline on PR-AUC. If it's worse, that's still a valid,
reportable result as long as you can explain why.

---

## Phase 8 — Experiments & ablations
**~2 weeks** (extended from 1–1.5 — this phase is where the GPU-hour budget is
tightest, and it needs room to be paced across quota resets rather than
crammed)

Goal: turn one result into a defensible set of experiments, scoped to what
your free-tier budget actually supports.

Tasks
- [ ] **Build the budget table first, before running anything:** using the
      per-epoch time from Phase 5, multiply out (configs × seeds × epochs) for
      your intended full matrix — GraphSAGE, GAT, GAT + your module, each
      ablation variant, 3 seeds each — and compare the total against ~30
      Kaggle hours/week (plus occasional Colab overflow) across this phase's
      ~2-week window
- [ ] If it doesn't fit — and for a full 3-seed ablation matrix at this scale,
      it plausibly won't — cut scope **before** you start: drop to 2 seeds,
      or trim ablation variants to the ones that answer your most important
      question. Write the cut down as a stated limitation for the report now,
      not as a surprise later
- [ ] Main comparison table: GraphSAGE, GAT, GAT + your module
- [ ] Ablation: your module with each key component removed, one at a time
      (whatever scope survived the budget cut above)
- [ ] Report mean ± std across whatever seed count you settled on — never a
      single run passed off as the result
- [ ] **If using IEEE-CIS:** note RL-GNN's published 0.872 AUROC / 0.683 AP
      (Appendix D) alongside your table as an external reference point
- [ ] **If using Elliptic:** explicitly check performance across the later,
      harder timesteps — temporal shift alone explains a meaningful chunk of
      apparent GNN gains on this dataset in recent work, worth checking honestly

Agent guardrails: never hand-pick the best-looking seed as the headline number.
If the budget table says the planned matrix doesn't fit, say so plainly and
propose the cut rather than quietly running a smaller matrix without flagging
it — the report needs to state this as a deliberate, explained limitation.

Definition of Done: budget table committed alongside the results; results
table plus 1–2 figures saved to `report/figures/`; any scope cuts from the
original ablation plan are explicitly documented with a reason.

---

## Phase 9 — Error analysis (stretch goal)
**~3–5 days**

Goal: a qualitative story for your report/defense. Unchanged from
`ROADMAP.md` — this works entirely from checkpoints and predictions you
already downloaded from Kaggle in Phases 5/7/8, so it needs no new GPU time.

Tasks
- [ ] Find cases the baseline got wrong that your module fixed, and vice versa
- [ ] Inspect attention weights on a handful of known-fraud nodes, before vs.
      after your module
- [ ] Optional: a simple explanation output — e.g. the top-k neighbors or
      relations that most influenced a flagged node's score

Definition of Done: 3–5 concrete examples, each with a short written explanation.

---

## Phase 10 — Report, reproducibility & final packaging
**~2 weeks, overlapping with Phase 8–9** (slightly longer than the lab-GPU
version — reproducibility now spans two environments, which is a real extra
check, not just a formality)

Tasks
- [ ] Write the report: motivation, related work (from Phase 6), method,
      experiments (Phase 8), results, limitations — **including your compute
      constraints and how they shaped Phase 8's scope.** This is a legitimate,
      honest thing to state plainly, not something to hide
- [ ] Clean the repo: final `README.md` with exact run commands for **both**
      environments — local CPU (env setup, tests, subsample run) and Kaggle
      (which Dataset to attach, which script to run)
- [ ] **Two-part reproducibility check, not one:** (a) a fresh clone in a fresh
      local CPU environment runs the tests and a subsample training pass
      cleanly; (b) the exact same code, freshly uploaded to a new Kaggle
      notebook session, reproduces your headline number. Checking only (a) and
      assuming (b) follows is the most likely thing to quietly break in this
      version of the plan
- [ ] Prepare a short talking-point summary for professor discussion

Definition of Done: both halves of the reproducibility check pass; report
draft complete, including the compute-constraints discussion; repo tagged
(e.g. `v1.0-submission`).

---

## Appendix A — Dataset decision matrix

| Factor | IEEE-CIS | Elliptic |
|---|---|---|
| Graph readiness | Tabular — you build the graph (more work, more learning) | Already a graph (nodes/edges provided) |
| Size | ~590K transactions | 203,769 nodes, 234,355 edges |
| Features | Mixed transaction + identity fields, engineered "V" features | 166 numeric features (94 local + 72 neighbor-aggregate) — sources disagree by one (165 vs. 166); confirm with `print(data.x.shape)` once loaded |
| Labels | Binary `isFraud`, ~3–4% positive | 3-way (licit/illicit/unknown); ~2% illicit, 21% licit, 77% unknown |
| Split strategy | Time-based recommended | Time-based required (49 sequential steps) |
| Fit for the camouflage angle | Strong — shared card/device/email is a direct camouflage signal | Weaker — features are anonymized aggregates |
| Kaggle-native availability | Yes — competition data, attachable via "Add Data" | Yes — also available as a Kaggle Dataset |
| Main risk | Graph construction (hub nodes, relation design) eats your timeline | Structure-vs-temporal-shift confound (Appendix D) |

## Appendix B — Timeline (13–14-week default)

A week longer than `ROADMAP.md`'s baseline — the extra time is entirely in
Phases 0, 2, 5, 7, 8, and 10, where remote execution and GPU-hour pacing add
real days that a local-GPU plan doesn't need. Adjust to your actual deadline.

| Week | Phase(s) | Milestone | GPU-hour load |
|---|---|---|---|
| 1 | 0, 1 | Local + Kaggle env verified, dataset locked | None |
| 2 | 2 | Graph built on Kaggle CPU, packaged as a Dataset | None (CPU session) |
| 3 | 3 | GraphSAGE from scratch, tested locally | None |
| 4 | 4, 6 (start) | GAT from scratch, tested locally; reading started | None |
| 5 | 5 | Baseline harness proven locally, first timed Kaggle run | Light — calibration run |
| 6 | 5 (finish), 6 (finish) | Baseline results table done; mechanism summaries done | Moderate — both baselines to convergence |
| 7–9 | 7 | Novel module: local dev, then Kaggle iteration | Heavy — pace across 3 weeks, don't cram |
| 10–11 | 8 | Budget table, ablations, multi-seed results | Heaviest — this is the week to watch your quota closest |
| 12 | 9 | Error analysis from already-downloaded results | None |
| 13–14 | 10 | Report, two-part repro check, submission packaged | Light — repro check only |

## Appendix C — "From scratch" scope clarification

**Allowed:** `torch.nn.Linear`, autograd, optimizers, `torch.sparse`, raw tensor
indexing, native PyTorch scatter/reduce ops (`scatter_reduce_`, `index_add_`) or
`torch_scatter`'s reduction functions as a fallback, and PyTorch Geometric's
`Data`/`Dataset` classes purely for loading/storing graphs.

**Not allowed:** `torch_geometric.nn.SAGEConv`, `GATConv`, or any other
prebuilt message-passing layer from PyG, DGL, or similar libraries.

Worth a one-line confirmation from your professor early on, since "no
pretrained models" is slightly ambiguous about utility functions like this.

## Appendix D — Reading list (for citation, not for copying)

**Foundational methods (original camouflage-resistant GNNs):**
- CARE-GNN — Dou et al., *Enhancing Graph Neural Network-based Fraud Detectors
  against Camouflaged Fraudsters*, CIKM 2020. Code: github.com/YingtongDou/CARE-GNN
- PC-GNN — Liu et al., *Pick and Choose: A GNN-based Imbalanced Learning
  Approach for Fraud Detection*, WWW 2021
- H2-FDetector — Shi et al., *H2-FDetector: A GNN-based Fraud Detector with
  Homophilic and Heterophilic Connections*, WWW 2022
- GAGA — Wang et al., *Label Information Enhanced Fraud Detection against Low
  Homophily in Graphs*, WWW 2023

**Transaction-graph benchmarks and closest related work — read these before
finalizing your novelty paragraph, they directly constrain what you can claim:**
- T-Finance / T-Social — Tang et al., *Rethinking Graph Neural Networks for
  Anomaly Detection*, ICML 2022 — establishes that "transaction data" alone is
  not the open gap
- S-FFSD — Xiang et al., *Semi-supervised Credit Card Fraud Detection via
  Attribute-driven Graph Representation*, AAAI 2023
- **RL-GNN** — *Reinforcement learning with graph neural network (RL-GNN)
  fusion for real-time financial fraud detection*, Scientific Reports, Dec
  2025 — GAT + RL controller evaluated directly on IEEE-CIS, 0.872 AUROC /
  0.683 AP. Required reading (Phase 6), and the paper you need to explicitly
  differentiate from in your report
- GADBench — benchmark paper standardizing evaluation of CARE-GNN, PC-GNN, and
  related methods

**Recent camouflage-specific work (2025) — for currency in your related-work
section:**
- PROD — *Projected and Orthogonal Disentanglement*, Knowledge-Based Systems,
  2025
- SCFCRC — *Simultaneously Counteract Feature Camouflage and Relation
  Camouflage for Fraud Detection*, arXiv 2025
- HA-GNN (2025 update) — argues CARE-GNN-style neighbor selectors degrade when
  feature camouflage is layered on top of relation camouflage

**Datasets:**
- IEEE-CIS Fraud Detection dataset — Kaggle
- Elliptic Bitcoin dataset — Kaggle, or `torch_geometric.datasets.EllipticBitcoinDataset`

## Appendix E — Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Graph construction (Phase 2) takes longer than a week | Medium | Time-box it; fall back to Elliptic if you're not done by end of week 2 |
| Novel module (Phase 7) doesn't beat baseline | Medium | Still a valid, explainable result — budget time to analyze *why* |
| Hub-node explosion makes the graph unusable | Medium–High (IEEE-CIS) | Degree caps from Phase 2, checked immediately after construction |
| Running out of time for the report | High if left until the end | Start the related-work section in Phase 6, not Phase 10 |
| "From scratch" scope dispute with professor | Low, but costly if it happens | Confirm Appendix C's line with them in week 1 |
| Novelty claim challenged as "already done" | Was High, now mitigated | Precise novelty statement in §1 and Phase 7 |
| **GPU quota burned on debugging instead of finished runs** | **High if not disciplined** | **Never run untested code on Kaggle — local-subsample-first is not optional in this version** |
| **Session disconnects mid-run, losing an unsaved training pass** | **Medium** | **Checkpoint every N epochs, not just at the end; save to Kaggle Dataset or Drive incrementally** |
| **Phase 8's full ablation matrix doesn't fit the weekly quota** | **High** | **Budget table built before running anything; pre-planned scope cut, documented as a stated limitation, not discovered mid-phase** |
| **Tempted to SSH/tunnel into Colab for a smoother Antigravity workflow** | **Medium** | **Don't — against ToS, risks losing free-tier access entirely; use the local-dev-then-handoff loop in §4 instead** |
