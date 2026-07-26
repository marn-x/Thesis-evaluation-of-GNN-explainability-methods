"""Elliptic Bitcoin dataset loading and the temporal split of Weber et al. (2019).

PyG's ``EllipticBitcoinDataset`` supplies its own train/test masks. This module
rebuilds them explicitly from the time step column of the raw CSV instead, for
two reasons: the split then provably matches the protocol described in the
paper (first 34 time steps train, last 15 test), and the per-time-step index is
needed to report results around the dark market shutdown at step 43.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.utils import to_undirected

from sq4_explain.config import DataSettings

RAW_FEATURE_FILE = "elliptic_txs_features.csv"
TIMESTEP_COLUMN = 1  # column 0 is the transaction id

# PyG's EllipticBitcoinDataset encodes labels as: 0 licit, 1 illicit, 2 unknown.
# Unknown is NOT negative, so "labelled" must exclude class 2 explicitly rather
# than testing y >= 0, which would sweep in every node.
LICIT_CLASS = 0
ILLICIT_CLASS = 1
UNKNOWN_CLASS = 2


@dataclass(frozen=True)
class EllipticBundle:
    """A loaded graph plus everything downstream code needs about the split.

    Attributes:
        data: The single PyG graph, with ``train_mask`` and ``test_mask`` set.
        timestep: Per-node time step, 1-based, aligned with ``data.x``.
        num_features: Feature count as actually loaded (see the note below).
        feature_names: Positional names; the raw CSV carries no header.
    """

    data: Data
    timestep: Tensor
    num_features: int
    feature_names: list[str]

    @property
    def num_nodes(self) -> int:
        """Number of transaction nodes in the graph."""
        return int(self.data.num_nodes)

    def labelled_mask(self) -> Tensor:
        """Nodes carrying a licit/illicit label (the rest are unknown)."""
        return self.data.y != UNKNOWN_CLASS


class EllipticDataModule:
    """Loads the Elliptic graph and derives the published temporal split."""

    def __init__(self, settings: DataSettings, data_root: Path) -> None:
        """Store configuration; loading is deferred to :meth:`load`."""
        self._settings = settings
        self._root = data_root

    def load(self) -> EllipticBundle:
        """Download (if needed), load and split the dataset.

        Returns:
            An :class:`EllipticBundle` with masks following Weber et al. (2019).
        """
        dataset = EllipticBitcoinDataset(root=str(self._root / "elliptic"))
        data: Data = dataset[0]
        # Weber et al.'s symmetric normalisation presupposes an undirected
        # adjacency; PyG ships the directed payment flows.
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        num_features = int(data.num_features)

        if num_features != self._settings.expected_num_features:
            logger.warning(
                "Feature count is {} but config expects {}. Weber et al. describe "
                "166 features; PyG holds the time step out. Record this difference "
                "in the thesis rather than silencing it.",
                num_features,
                self._settings.expected_num_features,
            )

        timestep = self._load_timesteps(dataset.raw_dir, data.num_nodes)
        data.train_mask, data.test_mask = self._temporal_masks(timestep, data.y)
        data.timestep = timestep

        logger.info(
            "Elliptic loaded: {} nodes, {} edges, {} features; "
            "train {} / test {} labelled nodes (split at time step {}).",
            data.num_nodes,
            data.num_edges,
            num_features,
            int(data.train_mask.sum()),
            int(data.test_mask.sum()),
            self._settings.train_timesteps_end,
        )
        return EllipticBundle(
            data=data,
            timestep=timestep,
            num_features=num_features,
            feature_names=[f"f{index:03d}" for index in range(num_features)],
        )

    def _load_timesteps(self, raw_dir: str, num_nodes: int) -> Tensor:
        """Read the per-node time step from the raw feature CSV.

        Falls back to a single pseudo-step if the raw file is unavailable, which
        keeps the pipeline runnable but disables the per-time-step report.
        """
        raw_path = Path(raw_dir) / RAW_FEATURE_FILE
        if not raw_path.is_file():
            logger.error(
                "Raw file {} not found; cannot rebuild the published split. "
                "Falling back to PyG's own masks.",
                raw_path,
            )
            return torch.ones(num_nodes, dtype=torch.long)

        column = pd.read_csv(
            raw_path, header=None, usecols=[TIMESTEP_COLUMN]
        ).squeeze("columns")
        timestep = torch.as_tensor(column.to_numpy().copy(), dtype=torch.long)
        if timestep.numel() != num_nodes:
            msg = (
                f"Raw CSV has {timestep.numel()} rows but the graph has "
                f"{num_nodes} nodes; node ordering cannot be assumed."
            )
            raise RuntimeError(msg)
        return timestep

    def _temporal_masks(self, timestep: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Build train/test masks over labelled nodes only, split by time step."""
        labelled = labels != UNKNOWN_CLASS
        boundary = self._settings.train_timesteps_end
        train_mask = labelled & (timestep <= boundary)
        test_mask = labelled & (timestep > boundary)
        return train_mask, test_mask


def feature_baseline(data: Data, strategy: str) -> Tensor:
    """Return the reference vector used to represent an 'absent' feature.

    Shapley values are only defined relative to a baseline, and the choice
    changes the numbers. Making it explicit and configurable is part of the
    set-up discipline the reproducibility requirement asks for.

    Args:
        data: The graph, used for the training-set mean.
        strategy: Either ``"train_mean"`` or ``"zeros"``.

    Returns:
        A tensor of shape ``[num_features]``.
    """
    if strategy == "zeros":
        return torch.zeros(data.num_features, dtype=data.x.dtype)
    if strategy == "train_mean":
        return data.x[data.train_mask].mean(dim=0)
    msg = f"Unknown baseline strategy: {strategy!r}"
    raise ValueError(msg)
