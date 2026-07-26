"""The contract every explanation method must satisfy.

This is the abstraction that makes the study extensible. Adding PGExplainer,
DG-SHAP, GraphSVX or a purpose-built explainer later means writing one new
module and decorating it with :func:`register_explainer`; nothing in the
experiment loop, the metrics or the reporting changes.

The contract is narrow on purpose. Every method must produce a single
attribution vector over feature columns, because that is the only output the
five requirements of the rubric can be applied to uniformly, and because
comparing methods at matched sparsity requires a common unit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from sq4_explain.config import Settings


@dataclass(frozen=True)
class ExplanationResult:
    """One explanation of one node by one method under one seed.

    Attributes:
        method: Registered name of the producing method.
        node_index: Index of the explained node in the full graph.
        seed: Seed used, or ``None`` for deterministic methods.
        feature_attribution: Signed or unsigned score per feature column,
            shape ``[num_features]``.
        edge_attribution: Optional score per edge of the k-hop subgraph.
        runtime_s: Wall-clock seconds for this single explanation.
        metadata: Method-specific extras worth carrying into the audit trail.
    """

    method: str
    node_index: int
    seed: int | None
    feature_attribution: Tensor
    runtime_s: float
    edge_attribution: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_k_features(self, k: int) -> Tensor:
        """Return the indices of the ``k`` features with the largest magnitude."""
        return torch.topk(self.feature_attribution.abs(), k=k).indices


class NodeExplainer(ABC):
    """Base class for a post-hoc explainer of a single node prediction.

    Subclasses declare whether they are deterministic. That flag is not
    decoration: the reproducibility requirement is the knock-out criterion of
    the rubric, and the experiment loop uses it to decide whether repeated
    seeds are meaningful for a given method.
    """

    name: ClassVar[str]
    is_deterministic: ClassVar[bool]

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        settings: Settings,
        device: torch.device,
    ) -> None:
        """Bind the trained model and graph this explainer will operate on."""
        self._model = model
        self._data = data
        self._settings = settings
        self._device = device

    @abstractmethod
    def explain(self, node_index: int, seed: int | None = None) -> ExplanationResult:
        """Explain the model's prediction for ``node_index``.

        Args:
            node_index: Node to explain, indexed into the full graph.
            seed: Seed for stochastic methods; ignored when deterministic.

        Returns:
            An :class:`ExplanationResult` with a feature attribution vector.
        """

    def describe(self) -> dict[str, Any]:
        """Return the method's own settings, for the run manifest."""
        return {"name": self.name, "deterministic": self.is_deterministic}


ExplainerFactory = Callable[..., NodeExplainer]
_REGISTRY: dict[str, type[NodeExplainer]] = {}


def register_explainer(cls: type[NodeExplainer]) -> type[NodeExplainer]:
    """Class decorator registering an explainer under its ``name`` attribute."""
    if not getattr(cls, "name", None):
        msg = f"{cls.__name__} must define a class-level `name`"
        raise ValueError(msg)
    if cls.name in _REGISTRY:
        msg = f"Explainer {cls.name!r} is already registered"
        raise ValueError(msg)
    _REGISTRY[cls.name] = cls
    return cls


def available_explainers() -> list[str]:
    """Return the sorted names of every registered explainer."""
    return sorted(_REGISTRY)


def build_explainer(
    name: str,
    model: nn.Module,
    data: Data,
    settings: Settings,
    device: torch.device,
) -> NodeExplainer:
    """Instantiate a registered explainer by name."""
    if name not in _REGISTRY:
        known = ", ".join(available_explainers())
        msg = f"Unknown explainer {name!r}. Registered: {known}"
        raise KeyError(msg)
    return _REGISTRY[name](model, data, settings, device)
