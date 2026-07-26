"""The exact Shapley implementation is the only novel maths in this project.

These tests check it against the axioms that define the Shapley value. If a
reviewer asks whether the "exact" arm is actually exact, this file is the
answer, and it runs without a trained model or the dataset.
"""

from math import factorial

import pytest
import torch

from sq4_explain.explainers.shap_exact import (
    contiguous_groups,
    exact_shapley_from_values,
    shapley_weights,
)


@pytest.mark.parametrize("num_players", [2, 3, 5, 8])
def test_efficiency_axiom(num_players: int) -> None:
    """The attributions must sum to the value of the grand coalition."""
    generator = torch.Generator().manual_seed(0)
    values = {
        mask: float(torch.randn(1, generator=generator))
        for mask in range(1 << num_players)
    }
    attributions = exact_shapley_from_values(values, num_players)
    grand = values[(1 << num_players) - 1] - values[0]
    assert sum(attributions) == pytest.approx(grand, abs=1e-9)


def test_linear_game_recovers_coefficients() -> None:
    """In an additive game each player's value is exactly its own coefficient."""
    coefficients = [1.0, -2.0, 3.5, 0.0, 7.25, -0.5]
    num_players = len(coefficients)
    values = {
        mask: sum(
            coefficients[index]
            for index in range(num_players)
            if mask & (1 << index)
        )
        for mask in range(1 << num_players)
    }
    attributions = exact_shapley_from_values(values, num_players)
    assert attributions == pytest.approx(coefficients, abs=1e-9)


def test_dummy_player_and_symmetry() -> None:
    """A player who never changes the value gets zero; equal players get equal shares."""
    num_players = 4
    values = {
        mask: 10.0 * bin(mask & 0b0111).count("1") for mask in range(1 << num_players)
    }
    attributions = exact_shapley_from_values(values, num_players)
    assert attributions[3] == pytest.approx(0.0, abs=1e-9)
    assert attributions[0] == pytest.approx(attributions[1], abs=1e-9)
    assert attributions[1] == pytest.approx(attributions[2], abs=1e-9)


@pytest.mark.parametrize("num_players", [3, 6, 12])
def test_weights_normalise(num_players: int) -> None:
    """Weights times the number of coalitions of each size must sum to one."""
    weights = shapley_weights(num_players)
    total = sum(
        factorial(num_players - 1)
        / (factorial(size) * factorial(num_players - 1 - size))
        * weights[size]
        for size in range(num_players)
    )
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(("num_features", "num_groups"), [(165, 12), (166, 12), (100, 7)])
def test_grouping_is_a_partition(num_features: int, num_groups: int) -> None:
    """Every feature belongs to exactly one group and no group is empty."""
    groups = contiguous_groups(num_features, num_groups)
    flattened = [column for group in groups for column in group]
    assert len(groups) == num_groups
    assert sorted(flattened) == list(range(num_features))
    assert all(group for group in groups)


def test_grouping_rejects_impossible_split() -> None:
    """More groups than features is a configuration error, not a silent fallback."""
    with pytest.raises(ValueError, match="Cannot split"):
        contiguous_groups(5, 9)
