"""Orchestration of the SQ4 experiment.

The loop is deliberately dumb: for each sampled node, for each method, for
each seed, produce one explanation and one row. All the judgement lives in the
explainers and the metrics; this module only makes sure the same nodes and the
same seeds are used for every method, so that differences between methods are
not differences in what they were asked to explain.
"""

from __future__ import annotations

from dataclasses import replace

from sq4_explain.explainers.shap_exact import contiguous_groups
from sq4_explain.metrics import to_groups

from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm

from sq4_explain.config import Settings
from sq4_explain.data import ILLICIT_CLASS, EllipticBundle, feature_baseline
from sq4_explain.explainers import build_explainer
from sq4_explain.explainers.base import ExplanationResult
from sq4_explain.metrics import consistency, fidelity

EXPLANATIONS_FILE = "explanations.csv"
CONSISTENCY_FILE = "consistency.csv"
ATTRIBUTIONS_FILE = "attributions.pt"


@torch.no_grad()
def sample_target_nodes(
    model: nn.Module, bundle: EllipticBundle, settings: Settings
) -> Tensor:
    """Pick the test nodes to explain: those the model flags as illicit.

    Explaining nodes the model did not flag would answer a different question.
    The supervisory scenario is "why was this taxpayer selected", so the sample
    is drawn from predicted-illicit test nodes. The draw is seeded separately
    from the training seed so the node set stays fixed while other things vary.
    """
    data = bundle.data
    model.eval()
    predictions = model(data.x, data.edge_index).argmax(dim=-1)
    candidates = torch.nonzero(
        data.test_mask & (predictions == ILLICIT_CLASS), as_tuple=False
    ).flatten()
    if candidates.numel() == 0:
        msg = "The model flagged no test node as illicit; nothing to explain."
        raise RuntimeError(msg)

    wanted = min(settings.explain.num_nodes, int(candidates.numel()))
    if wanted < settings.explain.num_nodes:
        logger.warning(
            "Only {} predicted-illicit test nodes available; requested {}.",
            wanted,
            settings.explain.num_nodes,
        )
    generator = torch.Generator().manual_seed(settings.explain.node_sample_seed)
    order = torch.randperm(candidates.numel(), generator=generator)[:wanted]
    return candidates[order]


class ExplanationExperiment:
    """Runs every configured method over the same nodes and seeds."""

    def __init__(
        self,
        model: nn.Module,
        bundle: EllipticBundle,
        settings: Settings,
        device: torch.device,
    ) -> None:
        """Build one explainer instance per configured method."""
        self._model = model
        self._bundle = bundle
        self._settings = settings
        self._device = device
        self._baseline = feature_baseline(bundle.data, settings.explain.baseline)
        self._groups = contiguous_groups(
            int(bundle.num_features), settings.explain.shap_exact.num_groups
        )
        self._explainers = {
            name: build_explainer(name, model, bundle.data, settings, device)
            for name in settings.explain.methods
        }
        logger.info("Methods under test: {}", ", ".join(self._explainers))

    def run(self, nodes: Tensor) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Explain every node with every method under every seed.

        Returns:
            A tidy per-explanation frame, a per-node/per-method consistency
            frame, and the raw attribution tensors for later re-analysis.
        """
        rows: list[dict] = []
        consistency_rows: list[dict] = []
        attributions: dict[str, Tensor] = {}

        for name, explainer in self._explainers.items():
            seeds = self._seeds_for(explainer.is_deterministic)
            total = len(nodes) * len(seeds)
            progress = tqdm(total=total, desc=name, unit="expl")
            for node in nodes.tolist():
                results = [self._explain_one(explainer, node, seed) for seed in seeds]
                for result in results:
                    rows.append(self._to_row(result))
                    key = f"{name}|{result.node_index}|{result.seed}"
                    attributions[key] = result.feature_attribution
                    progress.update(1)
                feature_scores = consistency(results, self._settings.explain.top_k)
                grouped = [
                    replace(
                        result,
                        feature_attribution=to_groups(
                            result.feature_attribution, self._groups
                        ),
                    )
                    for result in results
                ]
                group_scores = consistency(
                    grouped, self._settings.explain.top_k_groups
                )
                consistency_rows.append(
                    {
                        **feature_scores.__dict__,
                        "group_mean_jaccard": group_scores.mean_jaccard,
                        "group_min_jaccard": group_scores.min_jaccard,
                        "group_mean_spearman": group_scores.mean_spearman,
                    }
                )
            progress.close()

        return pd.DataFrame(rows), pd.DataFrame(consistency_rows), attributions

    def _seeds_for(self, deterministic: bool) -> list[int | None]:
        """Deterministic methods run once; stochastic ones run over every seed."""
        if deterministic:
            return [None]
        return list(self._settings.explain.seeds)

    def _explain_one(
        self, explainer: object, node: int, seed: int | None
    ) -> ExplanationResult:
        """Produce one explanation and attach its fidelity scores."""
        result: ExplanationResult = explainer.explain(node, seed)  # type: ignore[attr-defined]
        scores = fidelity(
            self._model,
            self._bundle.data,
            node,
            result.feature_attribution,
            self._baseline,
            self._settings.explain.top_k,
        )
        result.metadata["fidelity_plus"] = scores.fidelity_plus
        result.metadata["fidelity_minus"] = scores.fidelity_minus
        result.metadata["sparsity"] = scores.sparsity
        return result

    def _to_row(self, result: ExplanationResult) -> dict:
        """Flatten one result into a CSV row, keeping large objects out."""
        top_k = self._settings.explain.top_k
        return {
            "method": result.method,
            "node_index": result.node_index,
            "seed": result.seed,
            "runtime_s": result.runtime_s,
            "fidelity_plus": result.metadata.get("fidelity_plus"),
            "fidelity_minus": result.metadata.get("fidelity_minus"),
            "sparsity": result.metadata.get("sparsity"),
            "top_features": ",".join(
                str(index) for index in result.top_k_features(top_k).tolist()
            ),
            "efficiency_gap": result.metadata.get("efficiency_gap"),
            "top_groups": ",".join(
                str(index)
                for index in torch.topk(
                    to_groups(result.feature_attribution, self._groups).abs(),
                    k=self._settings.explain.top_k_groups,
                ).indices.tolist()
            ),
        }


def save_results(
    explanations: pd.DataFrame,
    consistency_frame: pd.DataFrame,
    attributions: dict[str, Tensor],
    artifact_root: Path,
) -> list[Path]:
    """Write the three experiment artifacts and return their paths."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths = [
        artifact_root / EXPLANATIONS_FILE,
        artifact_root / CONSISTENCY_FILE,
        artifact_root / ATTRIBUTIONS_FILE,
    ]
    explanations.to_csv(paths[0], index=False)
    consistency_frame.to_csv(paths[1], index=False)
    torch.save(attributions, paths[2])
    logger.info("Wrote {} explanations to {}", len(explanations), paths[0])
    return paths
