# SQ4 — Technical evaluation of GNN explainability methods

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Technical demonstration for sub-question 4: which GNN explainability methods
produce usable results on a realistic financial graph, and which properties
drive requirement satisfaction.

Three methods are run against one fixed classifier, over one fixed sample of
nodes, under one fixed set of seeds:

| Method | Family | Output | Deterministic |
| --- | --- | --- | --- |
| `gnnexplainer` | Mask optimisation (Ying et al., 2019) | Subgraph + feature scores | No |
| `shap_sampled` | Shapley value sampling via Captum | Feature scores | No |
| `shap_exact` | Exact Shapley over feature groups | Group scores | Yes |

The third is not a fourth method so much as a controlled condition. SQ3 scores
SHAP's reproducibility conditionally — an exact or seed-fixed set-up reaches
the deployment band, a sampled one does not — and running the sampled and the
exact variant side by side is what turns that conditional into a measurement.

## Protocol provenance

Nothing here is tuned. The classifier is held at the values published in
Weber et al. (2019), *Anti-Money Laundering in Bitcoin*, arXiv:1908.02591,
Section 4, so that model performance is not a variable in SQ4:

| Setting | Value | Source |
| --- | --- | --- |
| Split | time steps 1–34 train, 35–49 test, inductive | paper, §4 |
| Architecture | 2-layer GCN | paper, §3.2 |
| Hidden width | 100 | paper, §4 (after their tuning) |
| Epochs / optimiser / lr | 1000 / Adam / 0.001 | paper, §4 |
| Class weights | 0.3 licit, 0.7 illicit | paper, §4 |

Reference results for the illicit class: GCN 0.812 / 0.512 / 0.628 and
Skip-GCN 0.812 / 0.623 / 0.705 (precision / recall / F1). `sq4 train` prints
these next to your own numbers and refuses to continue if the illicit F1 falls
below `training.min_illicit_f1`, since explanations of a broken classifier
carry no information.

Two things the paper leaves open, which you therefore have to fix and report
yourself: initialisation, dropout, early stopping and the random seed are not
stated, and Table 2 reports "GCN" at the Skip-GCN figures of Table 1 without
explanation. Both are recorded in `artifacts/manifest_train.json`.

## Known discrepancy: feature count

Weber et al. describe 166 node features (94 local including the time step, 72
aggregated). PyG's `EllipticBitcoinDataset` reports 165; the time step appears
to be held out for mask construction. The loader logs a warning when the count
differs from `data.expected_num_features`. Record the difference in the method
chapter rather than silencing the warning.

The split is rebuilt from the time step column of the raw CSV rather than
taken from PyG's own masks, so that it provably matches the published protocol
and so that per-time-step reporting is possible. The paper notes a dark market
shutdown at step 43 after which every method degrades; `sq4 train` writes
`classifier_per_timestep.csv` so this shows up as a finding rather than
looking like a pipeline fault.

## Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra dev
```

The dataset downloads itself on first run into `data/elliptic/`. Only the
core `torch-geometric` package is needed; `torch-sparse` and `torch-scatter`
are not required for these operations.

## Usage

```bash
uv run sq4 train      # fit the backbone, check it against the published numbers
uv run sq4 explain    # run every method over the sampled nodes
uv run sq4 report     # summarise runtime, fidelity and consistency
```

Or through the Makefile: `make train explain report`.

Any setting can be overridden without editing the file:

```bash
SQ4_EXPLAIN__NUM_NODES=5 SQ4_TRAINING__EPOCHS=50 uv run sq4 explain
```

Start with a small `num_nodes` and a short seed list. GNNExplainer runs a
fresh optimisation per explanation and is the slowest arm by a wide margin;
Amato et al. report on the order of 100 seconds per graph.

## What the run measures

- **Fidelity+ / Fidelity−** at a *fixed* `top_k`. Yuan et al. (2023) note that
  Fidelity+ is only comparable across methods at matched sparsity, so sparsity
  is held constant by construction rather than reported after the fact.
- **Runtime** per explanation, wall clock. This is the performance-impact
  metadata SQ2 moved out of the scored rubric.
- **Consistency**: pairwise top-k Jaccard and Spearman rank correlation across
  repeated seeds on the same node. This is the direct measurement of R4, the
  knock-out criterion, and it is the part of SQ4 that no reviewed paper does
  on a supervisory-style dataset.

Elliptic carries no ground-truth explanations, so these measure faithfulness
to the model, not correctness of the explanation. State that as a limitation.

## Design notes

**Why one contract.** Every method implements `NodeExplainer.explain()` and
returns an attribution vector over feature columns. That is the only output
the rubric's five requirements can be applied to uniformly, and matched-sparsity
comparison requires a common unit. Edge-level SHAP variants (EdgeSHAPer,
GNNShap) are excluded deliberately: an edge score attributes a decision to a
third party's data, which is the problem the rubric already penalises in the
structure-only methods.

**Adding a method.** Write a module in `src/sq4_explain/explainers/`, decorate
the class with `@register_explainer`, import it in the package `__init__`, and
add its name to `explain.methods` in `config.toml`. The experiment loop, the
metrics and the reporting need no changes. PGExplainer, DG-SHAP, GraphSVX or
a purpose-built explainer all fit this shape.

**Masking semantics for exact SHAP.** A feature is made absent by replacing
its column with a baseline across *every node in the k-hop subgraph*, not only
the target. A GNN's prediction depends on its neighbours' features, so masking
only the target node would leave most of the influence unmeasured. The value
function is the illicit-class logit; probabilities compress differences near
the decision boundary. Both choices belong in the method chapter, because both
change the numbers.

**Group granularity.** Exact Shapley values are #P-hard over all features
(Van den Broeck et al., 2022), so the columns are partitioned into
`num_groups` players and all 2^`num_groups` coalitions are evaluated — 4096
forward passes at the default of 12. This mirrors the reduce-then-solve
argument of DG-SHAP (Chen et al., 2026) and buys determinism at the cost of
resolution. For cross-method ranking comparisons, aggregate the other methods
to the same partition with `metrics.to_groups`; comparing a per-feature
ranking against a group-level one otherwise flatters the coarser method.

## Audit trail

Every command writes a manifest next to its results containing the full
resolved settings, library versions, device and timestamp. The project's own
argument is that undocumented set-up is what sinks the reproducibility
requirement, so the manifest is not optional tooling — it is the artifact that
makes a claimed number checkable.

```
artifacts/
├── backbone.pt
├── classifier_per_timestep.csv
├── manifest_train.json
├── manifest_explain.json
├── explanations.csv        # one row per method × node × seed
├── consistency.csv         # one row per method × node
├── attributions.pt         # raw vectors, for re-analysis without re-running
└── summary.csv
```

## Tests

```bash
uv run pytest
```

`tests/test_shap_exact.py` checks the exact Shapley implementation against the
efficiency, linearity, dummy-player and symmetry axioms. These run without the
dataset or a trained model, so "is the exact arm actually exact" has an answer
that does not depend on a successful experiment run.

## Layout

```
src/sq4_explain/
├── config.py          # pydantic settings, TOML + env overrides
├── runtime.py         # logging, seeding, device, run manifests
├── data.py            # Elliptic loading, temporal split, baselines
├── models.py          # GCN / Skip-GCN, registry
├── train.py           # training loop, metrics, acceptance gate
├── metrics.py         # fidelity, consistency, group aggregation
├── experiment.py      # node sampling and the method × seed loop
├── cli.py             # train / explain / report
└── explainers/
    ├── base.py        # NodeExplainer contract + registry
    ├── gnn_explainer.py
    ├── shap_sampled.py
    └── shap_exact.py
```

## References

- Weber et al. (2019). *Anti-Money Laundering in Bitcoin*. arXiv:1908.02591
- Ying et al. (2019). *GNNExplainer*
- Van den Broeck et al. (2022). *On the Tractability of SHAP Explanations*
- Yuan et al. (2023). *Explainability in Graph Neural Networks: A Taxonomic Survey*
- Chen et al. (2026). *DG-SHAP*
- Unagar and Borisaniya (2025). GNNExplainer on Elliptic — comparison point
