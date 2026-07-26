"""GNNExplainer, via PyG's built-in implementation.

Ying et al. (2019). Included because it is the most widely used GNN explainer
in the surveyed literature and because Unagar and Borisaniya (2025) already
apply it to this dataset, which gives the SQ4 result a direct comparison point.

It runs a fresh mask optimisation for every explanation, so it is stochastic:
the reproducibility measurement is the point of including it here.
"""

from __future__ import annotations

import time

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig

from sq4_explain.config import Settings
from sq4_explain.explainers.base import (
    ExplanationResult,
    NodeExplainer,
    register_explainer,
)
from sq4_explain.runtime import seed_everything


@register_explainer
class GNNExplainerRunner(NodeExplainer):
    """Adapter turning PyG's GNNExplainer into a :class:`NodeExplainer`."""

    name = "gnnexplainer"
    is_deterministic = False

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        settings: Settings,
        device: torch.device,
    ) -> None:
        """Configure the PyG explainer wrapper once and reuse it per node."""
        super().__init__(model, data, settings, device)
        method = settings.explain.gnnexplainer
        self._explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=method.epochs, lr=method.learning_rate),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=ModelConfig(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

    def explain(self, node_index: int, seed: int | None = None) -> ExplanationResult:
        """Optimise soft masks for ``node_index`` and reduce them to per-feature scores."""
        if seed is not None:
            seed_everything(seed)
        started = time.perf_counter()
        explanation = self._explainer(
            self._data.x, self._data.edge_index, index=int(node_index)
        )
        runtime = time.perf_counter() - started

        node_mask: Tensor = explanation.get("node_mask")
        # node_mask is [num_nodes, num_features]; the rubric compares methods at
        # the feature level, so collapse the node axis by summing contributions.
        feature_attribution = node_mask.abs().sum(dim=0).detach().cpu()
        edge_mask = explanation.get("edge_mask")
        return ExplanationResult(
            method=self.name,
            node_index=int(node_index),
            seed=seed,
            feature_attribution=feature_attribution,
            edge_attribution=None if edge_mask is None else edge_mask.detach().cpu(),
            runtime_s=runtime,
            metadata={
                "epochs": self._settings.explain.gnnexplainer.epochs,
                "lr": self._settings.explain.gnnexplainer.learning_rate,
            },
        )

    def describe(self) -> dict[str, object]:
        """Return the method settings for the run manifest."""
        return {
            **super().describe(),
            **self._settings.explain.gnnexplainer.model_dump(),
        }
