"""Exact Shapley values over a reduced player set.

Computing exact Shapley values over all node features is #P-hard (Van den
Broeck et al., 2022), which is why every workable SHAP set-up samples. Chen et
al. (2026) sidestep this in DG-SHAP by reducing the input dimension first and
then solving exactly, trading some granularity for an answer that is identical
on every run.

This module reproduces that argument at thesis scale. The feature columns are
partitioned into ``num_groups`` groups, each group becomes one player in the
coalitional game, and all ``2**num_groups`` coalitions are evaluated. The
result is deterministic by construction: no seed, no sampling error.

Two design decisions have to be reported in the method chapter, because both
change the numbers:

1. A feature is made "absent" by replacing its column with a baseline value
   across every node of the k-hop subgraph, not only the target node. A GNN's
   prediction for a node depends on its neighbours' features, so masking only
   the target would leave most of the influence unmeasured.
2. The value function is the model's logit for the illicit class. Probabilities
   are bounded and would compress differences near the decision boundary.
"""

from __future__ import annotations

import time
from itertools import combinations
from math import factorial

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph

from sq4_explain.config import Settings
from sq4_explain.data import ILLICIT_CLASS, feature_baseline
from sq4_explain.explainers.base import (
    ExplanationResult,
    NodeExplainer,
    register_explainer,
)

EFFICIENCY_TOLERANCE = 1e-3


def contiguous_groups(num_features: int, num_groups: int) -> list[list[int]]:
    """Partition feature indices into contiguous, near-equal groups.

    Contiguous grouping preserves the ordering of the raw Elliptic file, in
    which the local features precede the aggregated ones, so a group stays
    interpretable as a block of related columns.

    Args:
        num_features: Total number of feature columns.
        num_groups: Number of groups (players) to create.

    Returns:
        A list of ``num_groups`` lists of feature indices.
    """
    if num_groups > num_features:
        msg = f"Cannot split {num_features} features into {num_groups} groups"
        raise ValueError(msg)
    base, remainder = divmod(num_features, num_groups)
    groups: list[list[int]] = []
    cursor = 0
    for index in range(num_groups):
        size = base + (1 if index < remainder else 0)
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def shapley_weights(num_players: int) -> list[float]:
    """Return the Shapley weight for each coalition size ``s`` in ``0..n-1``.

    The weight is ``s! * (n - s - 1)! / n!``, the probability that a given
    player joins after exactly ``s`` others in a uniformly random ordering.
    """
    total = factorial(num_players)
    return [
        factorial(size) * factorial(num_players - size - 1) / total
        for size in range(num_players)
    ]


def exact_shapley_from_values(values: dict[int, float], num_players: int) -> list[float]:
    """Compute exact Shapley values from a full table of coalition values.

    Args:
        values: Mapping from coalition bitmask to the value of that coalition.
            Must contain all ``2**num_players`` entries.
        num_players: Number of players in the game.

    Returns:
        One Shapley value per player, in player order.
    """
    weights = shapley_weights(num_players)
    attributions = [0.0] * num_players
    others = list(range(num_players))
    for player in range(num_players):
        rest = [index for index in others if index != player]
        for size in range(num_players):
            weight = weights[size]
            for subset in combinations(rest, size):
                mask = 0
                for member in subset:
                    mask |= 1 << member
                attributions[player] += weight * (
                    values[mask | (1 << player)] - values[mask]
                )
    return attributions


@register_explainer
class ExactShapRunner(NodeExplainer):
    """Exact Shapley values over feature groups, evaluated by enumeration."""

    name = "shap_exact"
    is_deterministic = True

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        settings: Settings,
        device: torch.device,
    ) -> None:
        """Precompute the feature partition and the absence baseline."""
        super().__init__(model, data, settings, device)
        method = settings.explain.shap_exact
        self._groups = contiguous_groups(int(data.num_features), method.num_groups)
        self._baseline = feature_baseline(data, settings.explain.baseline).to(device)

    def explain(self, node_index: int, seed: int | None = None) -> ExplanationResult:
        """Compute exact group-level Shapley values for ``node_index``.

        The ``seed`` argument is accepted to satisfy the common contract and is
        ignored: this method has no stochastic component, which is the property
        under test.
        """
        started = time.perf_counter()
        subset, edge_index, mapping, _ = k_hop_subgraph(
            int(node_index),
            num_hops=self._settings.explain.num_hops,
            edge_index=self._data.edge_index,
            relabel_nodes=True,
            num_nodes=self._data.num_nodes,
        )
        sub_x = self._data.x[subset].to(self._device)
        edge_index = edge_index.to(self._device)
        target = int(mapping[0])

        num_players = len(self._groups)
        values = self._coalition_values(sub_x, edge_index, target, num_players)
        group_attribution = exact_shapley_from_values(values, num_players)
        runtime = time.perf_counter() - started

        full_mask = (1 << num_players) - 1
        efficiency_gap = abs(
            sum(group_attribution) - (values[full_mask] - values[0])
        )

        return ExplanationResult(
            method=self.name,
            node_index=int(node_index),
            seed=None,
            feature_attribution=self._expand_to_features(group_attribution),
            runtime_s=runtime,
            metadata={
                "num_groups": num_players,
                "num_coalitions": 1 << num_players,
                "group_attribution": group_attribution,
                "group_assignment": self._groups,
                "subgraph_nodes": int(subset.numel()),
                "efficiency_gap": efficiency_gap,
                "baseline": self._settings.explain.baseline,
            },
        )

    @torch.no_grad()
    def _coalition_values(
        self, sub_x: Tensor, edge_index: Tensor, target: int, num_players: int
    ) -> dict[int, float]:
        """Evaluate the value function for every coalition of feature groups.

        Coalitions are evaluated in batches of replicated subgraphs so the
        ``2**num_players`` forward passes cost a handful of batched calls
        rather than one call each.
        """
        chunk = self._settings.explain.shap_exact.batch_size
        masks = list(range(1 << num_players))
        values: dict[int, float] = {}
        self._model.eval()

        for start in range(0, len(masks), chunk):
            block = masks[start : start + chunk]
            graphs = [
                Data(x=self._masked_features(sub_x, mask), edge_index=edge_index)
                for mask in block
            ]
            batch = Batch.from_data_list(graphs).to(self._device)
            logits = self._model(batch.x, batch.edge_index)
            offsets = torch.arange(len(block), device=self._device) * sub_x.size(0)
            selected = logits[offsets + target, ILLICIT_CLASS]
            for position, mask in enumerate(block):
                values[mask] = float(selected[position])
        return values

    def _masked_features(self, sub_x: Tensor, coalition: int) -> Tensor:
        """Return a copy of ``sub_x`` with every absent group set to the baseline."""
        masked = sub_x.clone()
        for player, columns in enumerate(self._groups):
            if coalition & (1 << player):
                continue
            index = torch.as_tensor(columns, device=masked.device)
            masked[:, index] = self._baseline[index]
        return masked

    def _expand_to_features(self, group_attribution: list[float]) -> Tensor:
        """Spread each group's value evenly over its member features.

        The expansion exists only so this method's output shares a shape with
        the others. Exact SHAP resolves to group granularity, and comparisons
        of ranking should be made at group level; see ``metrics.to_groups``.
        """
        attribution = torch.zeros(int(self._data.num_features))
        for value, columns in zip(group_attribution, self._groups, strict=True):
            attribution[torch.as_tensor(columns)] = value / len(columns)
        return attribution

    def describe(self) -> dict[str, object]:
        """Return the method settings for the run manifest."""
        return {
            **super().describe(),
            **self._settings.explain.shap_exact.model_dump(),
            "baseline": self._settings.explain.baseline,
        }
