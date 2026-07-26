"""Training and evaluation of the backbone classifier.

A deliberately small, dependency-free loop: full-batch, weighted cross entropy,
Adam. No trainer abstraction, because there is exactly one model and one
training regime, and the settings are inherited from the literature rather
than searched over.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor, nn
from torch_geometric.data import Data

from sq4_explain.config import Settings
from sq4_explain.data import ILLICIT_CLASS, EllipticBundle
from sq4_explain.models import build_model
from sq4_explain.runtime import seed_everything

CHECKPOINT_NAME = "backbone.pt"


@dataclass(frozen=True)
class ClassificationReport:
    """Illicit-class metrics plus the micro average, as reported in the paper."""

    precision: float
    recall: float
    f1: float
    micro_f1: float
    support: int

    def as_dict(self) -> dict[str, float]:
        """Return the report as a flat dictionary for logging or CSV export."""
        return {key: float(value) for key, value in asdict(self).items()}


def binary_report(
    predictions: Tensor, targets: Tensor, positive: int = ILLICIT_CLASS
) -> ClassificationReport:
    """Compute precision, recall, F1 for ``positive`` and the micro-averaged F1."""
    predicted_positive = predictions == positive
    actual_positive = targets == positive
    true_positive = float((predicted_positive & actual_positive).sum())
    precision_den = float(predicted_positive.sum())
    recall_den = float(actual_positive.sum())

    precision = true_positive / precision_den if precision_den else 0.0
    recall = true_positive / recall_den if recall_den else 0.0
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator else 0.0
    micro_f1 = float((predictions == targets).float().mean())
    return ClassificationReport(precision, recall, f1, micro_f1, int(recall_den))


class BackboneTrainer:
    """Trains the GNN that the explainers will later be run against."""

    def __init__(self, settings: Settings, device: torch.device) -> None:
        """Store settings and the target device."""
        self._settings = settings
        self._device = device

    def fit(self, bundle: EllipticBundle) -> tuple[nn.Module, ClassificationReport]:
        """Train on the early time steps and evaluate on the later ones.

        Returns:
            The trained model in eval mode, and its test-set report.

        Raises:
            RuntimeError: If the classifier fails the configured acceptance gate.
                Explanations of a non-functioning classifier carry no information,
                so this is a hard stop rather than a warning.
        """
        config = self._settings.training
        seed_everything(config.seed)

        data = bundle.data.to(self._device)
        model = build_model(bundle.num_features, 2, self._settings.model).to(self._device)
        optimiser = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(config.class_weights, device=self._device)
        )

        model.train()
        for epoch in range(1, config.epochs + 1):
            optimiser.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimiser.step()
            if epoch % config.log_every == 0 or epoch == 1:
                logger.info("epoch {:>4} | loss {:.4f}", epoch, loss.item())

        model.eval()
        report = self.evaluate(model, data, data.test_mask)
        logger.info(
            "Test illicit precision {:.3f} / recall {:.3f} / F1 {:.3f}; micro F1 {:.3f}",
            report.precision,
            report.recall,
            report.f1,
            report.micro_f1,
        )
        logger.info(
            "Weber et al. (2019) report 0.812 / 0.512 / 0.628 for GCN "
            "and 0.812 / 0.623 / 0.705 for Skip-GCN."
        )
        if report.f1 < config.min_illicit_f1:
            msg = (
                f"Illicit F1 {report.f1:.3f} is below the acceptance gate "
                f"{config.min_illicit_f1}. Fix the classifier before explaining it."
            )
            raise RuntimeError(msg)
        return model, report

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, data: Data, mask: Tensor
    ) -> ClassificationReport:
        """Evaluate the model on the nodes selected by ``mask``."""
        model.eval()
        predictions = model(data.x, data.edge_index).argmax(dim=-1)
        return binary_report(predictions[mask], data.y[mask])

    @torch.no_grad()
    def evaluate_per_timestep(
        self, model: nn.Module, bundle: EllipticBundle
    ) -> dict[int, ClassificationReport]:
        """Report per time step, so the dark market shutdown is visible.

        Weber et al. observe that every method degrades sharply after the
        shutdown at step 43; a single aggregate number over the test window
        hides that and makes the drop look like a pipeline fault.
        """
        data = bundle.data.to(self._device)
        model.eval()
        predictions = model(data.x, data.edge_index).argmax(dim=-1)
        timestep = bundle.timestep.to(self._device)
        results: dict[int, ClassificationReport] = {}
        for step in range(
            self._settings.data.train_timesteps_end + 1,
            self._settings.data.num_timesteps + 1,
        ):
            mask = data.test_mask & (timestep == step)
            if not bool(mask.any()):
                continue
            results[step] = binary_report(predictions[mask], data.y[mask])
        return results


def save_checkpoint(model: nn.Module, artifact_root: Path) -> Path:
    """Persist model weights so explanation runs need not retrain."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    path = artifact_root / CHECKPOINT_NAME
    torch.save(model.state_dict(), path)
    logger.info("Saved backbone weights to {}", path)
    return path


def load_checkpoint(
    bundle: EllipticBundle, settings: Settings, device: torch.device
) -> nn.Module:
    """Rebuild the architecture and restore weights from disk."""
    path = settings.paths.artifact_root / CHECKPOINT_NAME
    if not path.is_file():
        msg = f"No checkpoint at {path}. Run `uv run sq4 train` first."
        raise FileNotFoundError(msg)
    model = build_model(bundle.num_features, 2, settings.model).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
