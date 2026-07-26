"""Metrics for comparing explanations across methods.

Fidelity is implemented here rather than taken from ``torch_geometric.explain.
metric`` for one reason: that module operates on PyG ``Explanation`` objects,
and the exact Shapley method does not produce one. A single implementation
applied to the common attribution vector keeps the numbers comparable, which
is the whole point of the shared contract in ``explainers.base``.

Yuan et al. (2023) note that Fidelity+ is only comparable between methods at
the same sparsity level. Every function here therefore takes an explicit
``top_k`` and the reported sparsity is fixed by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
from scipy.stats import spearmanr
from torch import Tensor, nn
from torch_geometric.data import Data

from sq4_explain.data import ILLICIT_CLASS
from sq4_explain.explainers.base import ExplanationResult


@dataclass(frozen=True)
class FidelityScores:
    """Prediction change when the explanation is removed or kept in isolation.

    Attributes:
        fidelity_plus: Change in the illicit-class probability after masking
            out the top-k features. Large is good: the explanation named the
            features the model actually relies on.
        fidelity_minus: Change after keeping only the top-k features. Small is
            good: the explanation is sufficient on its own.
        sparsity: Fraction of features left out of the explanation.
    """

    fidelity_plus: float
    fidelity_minus: float
    sparsity: float


def to_groups(attribution: Tensor, groups: list[list[int]]) -> Tensor:
    """Aggregate a per-feature attribution into per-group totals.

    Needed because the exact method resolves to groups. Comparing a sampled
    per-feature ranking against a group-level one is only meaningful once both
    are expressed on the same partition.
    """
    return torch.stack(
        [attribution[torch.as_tensor(columns)].sum() for columns in groups]
    )


@torch.no_grad()
def fidelity(
    model: nn.Module,
    data: Data,
    node_index: int,
    attribution: Tensor,
    baseline: Tensor,
    top_k: int,
) -> FidelityScores:
    """Measure how much the model's output depends on the named features.

    Args:
        model: The trained backbone, in eval mode.
        data: The full graph.
        node_index: The explained node.
        attribution: Per-feature scores from an explanation.
        baseline: The absence value per feature.
        top_k: Number of features counted as "the explanation".

    Returns:
        Fidelity+, Fidelity- and the sparsity they were measured at.
    """
    model.eval()
    device = data.x.device
    selected = torch.topk(attribution.abs(), k=top_k).indices.to(device)
    baseline = baseline.to(device)

    def probability(features: Tensor) -> float:
        logits = model(features, data.edge_index)
        return float(logits[node_index].softmax(dim=-1)[ILLICIT_CLASS])

    original = probability(data.x)

    without = data.x.clone()
    without[:, selected] = baseline[selected]

    only = data.x.clone()
    complement = torch.ones(data.num_features, dtype=torch.bool, device=device)
    complement[selected] = False
    only[:, complement] = baseline[complement]

    return FidelityScores(
        fidelity_plus=abs(original - probability(without)),
        fidelity_minus=abs(original - probability(only)),
        sparsity=1.0 - top_k / int(data.num_features),
    )


def top_k_jaccard(first: Tensor, second: Tensor, top_k: int) -> float:
    """Overlap between the top-k sets of two attribution vectors.

    This is the direct measurement of the reproducibility requirement: two runs
    of the same method on the same node should name the same features. A value
    of 1.0 means identical sets, 0.0 means disjoint.
    """
    left = set(torch.topk(first.abs(), k=top_k).indices.tolist())
    right = set(torch.topk(second.abs(), k=top_k).indices.tolist())
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def rank_correlation(first: Tensor, second: Tensor) -> float:
    """Spearman correlation between two full attribution vectors.

    Complements the top-k overlap: a method can keep the same top-k while
    reshuffling everything below it, or move one feature in and out of the cut
    while otherwise agreeing closely.
    """
    result = spearmanr(first.numpy(), second.numpy())
    correlation = float(result.statistic)
    return 0.0 if correlation != correlation else correlation  # guard against NaN


@dataclass(frozen=True)
class ConsistencyScores:
    """Agreement between repeated runs of one method on one node."""

    node_index: int
    method: str
    num_runs: int
    mean_jaccard: float
    min_jaccard: float
    mean_spearman: float


def consistency(results: list[ExplanationResult], top_k: int) -> ConsistencyScores:
    """Compare every pair of repeated runs for one method on one node.

    Args:
        results: Repeated explanations of the same node by the same method.
        top_k: Cut-off for the overlap measure.

    Returns:
        Pairwise agreement summarised over all run pairs. A single run yields
        perfect scores by definition, which is the correct reading for a
        deterministic method.
    """
    if not results:
        msg = "consistency() needs at least one result"
        raise ValueError(msg)
    method = results[0].method
    node_index = results[0].node_index
    if len(results) == 1:
        return ConsistencyScores(node_index, method, 1, 1.0, 1.0, 1.0)

    jaccards: list[float] = []
    correlations: list[float] = []
    for left, right in combinations(results, 2):
        jaccards.append(
            top_k_jaccard(left.feature_attribution, right.feature_attribution, top_k)
        )
        correlations.append(
            rank_correlation(left.feature_attribution, right.feature_attribution)
        )
    return ConsistencyScores(
        node_index=node_index,
        method=method,
        num_runs=len(results),
        mean_jaccard=sum(jaccards) / len(jaccards),
        min_jaccard=min(jaccards),
        mean_spearman=sum(correlations) / len(correlations),
    )

