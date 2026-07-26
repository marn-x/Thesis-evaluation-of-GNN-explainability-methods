"""Backbone classifiers, kept deliberately plain.

The thesis is about the explanation, not the classifier, so these follow the
architectures described by Weber et al. (2019) with no additions. New
architectures register themselves through :func:`register_model`, so nothing
in the training or explanation code has to change to accommodate one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

from sq4_explain.config import ModelSettings

ModelFactory = Callable[[int, int, ModelSettings], nn.Module]
_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a model factory under ``name`` (Open-Closed extension point)."""

    def decorator(factory: ModelFactory) -> ModelFactory:
        if name in _REGISTRY:
            msg = f"Model {name!r} is already registered"
            raise ValueError(msg)
        _REGISTRY[name] = factory
        return factory

    return decorator


def build_model(
    in_channels: int, out_channels: int, settings: ModelSettings
) -> nn.Module:
    """Instantiate the architecture named in the settings."""
    if settings.architecture not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        msg = f"Unknown architecture {settings.architecture!r}. Known: {known}"
        raise KeyError(msg)
    return _REGISTRY[settings.architecture](in_channels, out_channels, settings)


class GCN(nn.Module):
    """A plain multi-layer GCN, matching the paper's 2-layer configuration.

    The forward signature is ``(x, edge_index)`` and the return value is raw
    logits. Both are required by PyG's ``Explainer`` wrapper, so keeping them
    stable is what lets every explainer treat the model identically.
    """

    default_activation: ClassVar[Callable[[Tensor], Tensor]] = torch.relu

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Build ``num_layers`` graph convolutions with a shared hidden width."""
        super().__init__()
        if num_layers < 2:
            msg = "num_layers must be at least 2"
            raise ValueError(msg)
        widths = [in_channels, *[hidden_channels] * (num_layers - 1), out_channels]
        self.convs = nn.ModuleList(
            GCNConv(widths[index], widths[index + 1]) for index in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return class logits for every node."""
        for conv in self.convs[:-1]:
            x = self.dropout(self.default_activation(conv(x, edge_index)))
        return self.convs[-1](x, edge_index)


class SkipGCN(GCN):
    """GCN with a linear skip connection from the input features to the output.

    Weber et al. report noticeably better illicit-class recall for this variant,
    which matters here only because a higher recall yields more predicted-illicit
    test nodes available to explain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Add the skip projection on top of the base GCN."""
        super().__init__(in_channels, out_channels, hidden_channels, num_layers, dropout)
        self.skip = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return class logits, adding the projected raw features."""
        return super().forward(x, edge_index) + self.skip(x)


@register_model("gcn")
def _build_gcn(
    in_channels: int, out_channels: int, settings: ModelSettings
) -> nn.Module:
    return GCN(
        in_channels,
        out_channels,
        settings.hidden_channels,
        settings.num_layers,
        settings.dropout,
    )


@register_model("skipgcn")
def _build_skipgcn(
    in_channels: int, out_channels: int, settings: ModelSettings
) -> nn.Module:
    return SkipGCN(
        in_channels,
        out_channels,
        settings.hidden_channels,
        settings.num_layers,
        settings.dropout,
    )
