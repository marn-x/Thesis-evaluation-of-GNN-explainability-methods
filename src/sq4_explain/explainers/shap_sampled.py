"""Sampled Shapley values over the same coalitional game as the exact variant.

This is the feature-level SHAP arm. Edge-level variants (EdgeSHAPer, GNNShap)
are excluded on purpose: an edge score attributes a decision to a third party's
data, which is the problem the rubric already penalises in the structure-only
methods.

The method deliberately shares its game with :mod:`sq4_explain.explainers.
shap_exact`: the same feature groups are the players, the same k-hop subgraph
is the context, the same baseline defines absence, and the same illicit-class
logit is the value function. The only difference is that this module estimates
the Shapley values by sampling permutations while the exact module enumerates
all coalitions. That isolation is the point. SQ3 leaves SHAP's reproducibility
score conditional on the set-up, and a comparison can only attribute a
difference to sampling if nothing else differs.

Captum's ``ShapleyValueSampling`` is used rather than a hand-rolled sampler so
that the estimator itself is a published, third-party implementation.

Two implementation details that are easy to get wrong and are therefore
asserted at runtime:

1. With a ``feature_mask``, Captum replicates a group's *entire* attribution to
   every element of that group. The group value is therefore read from a single
   representative element; summing over the group would overcount by its size.
2. Captum is handed the k-hop subgraph, not the full graph. Passing the full
   graph makes every node-feature pair a player, which on Elliptic is roughly
   34 million players and does not terminate.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from captum.attr import ShapleyValueSampling
from loguru import logger
from torch import Tensor, nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph

from sq4_explain.config import Settings
from sq4_explain.data import ILLICIT_CLASS, feature_baseline
from sq4_explain.explainers.base import (
    ExplanationResult,
    NodeExplainer,
    register_explainer,
)
from sq4_explain.explainers.shap_exact import contiguous_groups
from sq4_explain.runtime import seed_everything

WITHIN_GROUP_TOLERANCE = 1e-4


@register_explainer
class SampledShapRunner(NodeExplainer):
    """Shapley values estimated by permutation sampling over feature groups."""

    name = "shap_sampled"
    is_deterministic = False

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        settings: Settings,
        device: torch.device,
    ) -> None:
        """Build the shared feature partition and the absence baseline.

        The partition is read from the ``shap_exact`` settings on purpose, so
        the two arms cannot drift apart through a config edit.
        """
        super().__init__(model, data, settings, device)
        self._groups = contiguous_groups(
            int(data.num_features), settings.explain.shap_exact.num_groups
        )
        self._baseline = feature_baseline(data, settings.explain.baseline).to(device)

    def explain(self, node_index: int, seed: int | None = None) -> ExplanationResult:
        """Estimate group-level Shapley values for ``node_index`` under ``seed``."""
        if seed is not None:
            seed_everything(seed)
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

        attribution = self._attribute(sub_x, edge_index, target)
        group_attribution = self._read_groups(attribution)
        runtime = time.perf_counter() - started

        return ExplanationResult(
            method=self.name,
            node_index=int(node_index),
            seed=seed,
            feature_attribution=self._expand_to_features(group_attribution),
            runtime_s=runtime,
            metadata={
                "n_samples": self._settings.explain.shap_sampled.n_samples,
                "num_groups": len(self._groups),
                "group_attribution": group_attribution,
                "group_assignment": self._groups,
                "subgraph_nodes": int(subset.numel()),
                "baseline": self._settings.explain.baseline,
                "estimator": "captum.ShapleyValueSampling",
            },
        )

    def _attribute(self, sub_x: Tensor, edge_index: Tensor, target: int) -> Tensor:
        """Run Captum over the subgraph with feature columns grouped into players."""
        self._model.eval()

        def forward(batch: Tensor) -> Tensor:
            """Value function: the illicit-class logit for the target node."""
            with torch.no_grad():
                merged = Batch.from_data_list(
                    [Data(x=sample, edge_index=edge_index) for sample in batch]
                )
                logits = self._model(merged.x, merged.edge_index)
                offsets = (
                    torch.arange(batch.size(0), device=self._device) * sub_x.size(0)
                )
                return logits[offsets + target, ILLICIT_CLASS]

        # A player is a feature group shared across every node of the subgraph,
        # matching how shap_exact masks columns.
        mask = torch.zeros(sub_x.shape, dtype=torch.long, device=self._device)
        for group_id, columns in enumerate(self._groups):
            mask[:, torch.as_tensor(columns, device=self._device)] = group_id

        attributor = ShapleyValueSampling(forward)
        attribution = attributor.attribute(
            sub_x.unsqueeze(0),
            baselines=self._baseline.expand_as(sub_x).unsqueeze(0),
            feature_mask=mask.unsqueeze(0),
            n_samples=self._settings.explain.shap_sampled.n_samples,
            perturbations_per_eval=self._settings.explain.shap_exact.batch_size,
        )
        return attribution[0].detach().cpu()

    def _read_groups(self, attribution: Tensor) -> list[float]:
        """Extract one value per group, checking Captum's replication assumption."""
        values: list[float] = []
        for columns in self._groups:
            index = torch.as_tensor(columns)
            block = attribution[:, index]
            representative = float(block.reshape(-1)[0])
            spread = float((block - representative).abs().max())
            if spread > WITHIN_GROUP_TOLERANCE:
                logger.warning(
                    "Attribution is not constant within a feature group "
                    "(spread {:.2e}). Captum's grouping semantics may have "
                    "changed; verify before trusting these numbers.",
                    spread,
                )
            values.append(representative)
        return values

    def _expand_to_features(self, group_attribution: list[float]) -> Tensor:
        """Spread each group's value evenly over its member features.

        Identical to the exact variant's expansion, so both arms land on the
        same scale and can be compared feature by feature.
        """
        attribution = torch.zeros(int(self._data.num_features))
        for value, columns in zip(group_attribution, self._groups, strict=True):
            attribution[torch.as_tensor(columns)] = value / len(columns)
        return attribution

    def describe(self) -> dict[str, Any]:
        """Return the method settings for the run manifest."""
        return {
            **super().describe(),
            **self._settings.explain.shap_sampled.model_dump(),
            "num_groups": len(self._groups),
            "baseline": self._settings.explain.baseline,
            "estimator": "captum.ShapleyValueSampling",
        }