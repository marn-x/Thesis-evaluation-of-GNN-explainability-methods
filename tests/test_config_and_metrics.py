"""Tests for settings loading and the metric functions."""

from pathlib import Path

import pytest
import torch

from sq4_explain.config import Settings, load_settings
from sq4_explain.explainers.base import ExplanationResult
from sq4_explain.metrics import (
    consistency,
    rank_correlation,
    to_groups,
    top_k_jaccard,
)


def _result(method: str, node: int, seed: int | None, values: list[float]) -> ExplanationResult:
    return ExplanationResult(
        method=method,
        node_index=node,
        seed=seed,
        feature_attribution=torch.tensor(values),
        runtime_s=0.0,
    )


def test_settings_load_from_project_config() -> None:
    """The shipped config.toml must validate against the settings models."""
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.data.train_timesteps_end == 34
    assert settings.model.hidden_channels == 100
    assert settings.training.class_weights == (0.3, 0.7)


def test_settings_paths_are_absolute() -> None:
    """Paths are resolved against the project root, never the working directory."""
    settings = load_settings()
    assert Path(settings.paths.artifact_root).is_absolute()


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """SQ4_SECTION__KEY overrides the TOML value."""
    monkeypatch.setenv("SQ4_TRAINING__EPOCHS", "7")
    settings = load_settings()
    assert settings.training.epochs == 7


def test_invalid_num_groups_is_rejected() -> None:
    """An intractable coalition count must fail loudly at load time."""
    with pytest.raises(ValueError, match="not tractable"):
        Settings.model_validate({"explain": {"shap_exact": {"num_groups": 30}}})


def test_jaccard_bounds() -> None:
    """Identical rankings give 1.0, disjoint top-k sets give 0.0."""
    first = torch.tensor([5.0, 4.0, 3.0, 0.1, 0.2])
    assert top_k_jaccard(first, first, top_k=2) == pytest.approx(1.0)
    second = torch.tensor([0.1, 0.2, 0.3, 9.0, 8.0])
    assert top_k_jaccard(first, second, top_k=2) == pytest.approx(0.0)


def test_jaccard_uses_magnitude_not_sign() -> None:
    """A large negative attribution is as important as a large positive one."""
    positive = torch.tensor([5.0, 1.0, 0.0])
    negative = torch.tensor([-5.0, 1.0, 0.0])
    assert top_k_jaccard(positive, negative, top_k=1) == pytest.approx(1.0)


def test_rank_correlation_of_reversed_order() -> None:
    """A perfectly reversed ranking gives -1."""
    ascending = torch.arange(10, dtype=torch.float32)
    assert rank_correlation(ascending, ascending.flip(0)) == pytest.approx(-1.0)


def test_to_groups_sums_within_partition() -> None:
    """Group totals must preserve the sum of the underlying attributions."""
    attribution = torch.tensor([1.0, 2.0, 3.0, 4.0])
    grouped = to_groups(attribution, [[0, 1], [2, 3]])
    assert grouped.tolist() == [3.0, 7.0]
    assert grouped.sum() == pytest.approx(attribution.sum())


def test_single_run_is_perfectly_consistent() -> None:
    """A deterministic method produces one run, which scores 1.0 by definition."""
    scores = consistency([_result("shap_exact", 3, None, [1.0, 2.0, 3.0])], top_k=2)
    assert scores.num_runs == 1
    assert scores.mean_jaccard == pytest.approx(1.0)


def test_consistency_detects_disagreement() -> None:
    """Two runs naming different features must not score as consistent."""
    results = [
        _result("gnnexplainer", 3, 0, [9.0, 8.0, 0.1, 0.0]),
        _result("gnnexplainer", 3, 1, [0.1, 0.0, 9.0, 8.0]),
    ]
    scores = consistency(results, top_k=2)
    assert scores.num_runs == 2
    assert scores.mean_jaccard == pytest.approx(0.0)
    assert scores.min_jaccard == pytest.approx(0.0)


def test_consistency_requires_results() -> None:
    """An empty result list is a programming error, not an empty score."""
    with pytest.raises(ValueError, match="at least one result"):
        consistency([], top_k=2)
