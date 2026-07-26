"""Typed, validated settings for the SQ4 experiment.

Settings live in ``config.toml`` and are parsed into pydantic models. Nothing
in the code base reads a literal path, hyperparameter or seed; everything
arrives through :class:`Settings`. This matters beyond tidiness: the thesis
argues that undocumented set-up is what sinks the reproducibility requirement,
so the run configuration is dumped alongside every artifact.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ENV_PREFIX = "SQ4_"
NESTED_DELIMITER = "__"


class PathSettings(BaseModel):
    """Filesystem locations. All relative paths resolve against the project root."""

    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    log_file: Path = Path("artifacts/sq4.log")

    def resolve(self, project_root: Path) -> PathSettings:
        """Return a copy with every path made absolute against ``project_root``."""
        return PathSettings(
            data_root=project_root / self.data_root,
            artifact_root=project_root / self.artifact_root,
            log_file=project_root / self.log_file,
        )


class LoggingSettings(BaseModel):
    """Loguru sink configuration."""

    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "30 days"


class DataSettings(BaseModel):
    """Elliptic dataset and the temporal split of Weber et al. (2019)."""

    num_timesteps: int = 49
    train_timesteps_end: int = 34
    expected_num_features: int = 165
    dark_market_shutdown_timestep: int = 43

    @field_validator("train_timesteps_end")
    @classmethod
    def _split_inside_range(cls, value: int) -> int:
        if value < 1:
            msg = "train_timesteps_end must be at least 1"
            raise ValueError(msg)
        return value


class ModelSettings(BaseModel):
    """Backbone classifier. Held at published values; see config.toml."""

    architecture: Literal["gcn", "skipgcn"] = "gcn"
    hidden_channels: int = 100
    num_layers: int = 2
    dropout: float = 0.0


class TrainingSettings(BaseModel):
    """Optimisation settings and the acceptance gate for the classifier."""

    epochs: int = 1000
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    class_weights: tuple[float, float] = (0.3, 0.7)
    seed: int = 42
    log_every: int = 50
    min_illicit_f1: float = 0.45


class GNNExplainerSettings(BaseModel):
    """Method-specific settings for GNNExplainer."""

    epochs: int = 200
    learning_rate: float = 0.01


class SampledShapSettings(BaseModel):
    """Method-specific settings for Shapley value sampling via Captum."""

    n_samples: int = 200
    node_mask_type: Literal["common_attributes", "attributes"] = "common_attributes"


class ExactShapSettings(BaseModel):
    """Method-specific settings for exact Shapley values over feature groups."""

    num_groups: int = 12
    grouping: Literal["contiguous"] = "contiguous"
    batch_size: int = 256

    @field_validator("num_groups")
    @classmethod
    def _tractable(cls, value: int) -> int:
        if not 2 <= value <= 20:
            msg = f"num_groups must be in [2, 20]; 2^{value} coalitions is not tractable"
            raise ValueError(msg)
        return value


class ExplainSettings(BaseModel):
    """The explanation experiment itself."""

    methods: list[str] = Field(default_factory=list)
    num_nodes: int = 30
    node_sample_seed: int = 7
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    num_hops: int = 2
    top_k: int = 10
    top_k_groups: int = 3
    baseline: Literal["train_mean", "zeros"] = "train_mean"

    gnnexplainer: GNNExplainerSettings = GNNExplainerSettings()
    shap_sampled: SampledShapSettings = SampledShapSettings()
    shap_exact: ExactShapSettings = ExactShapSettings()
    @field_validator("methods", "seeds", mode="before")
    @classmethod
    def _wrap_scalar(cls, value: Any) -> Any:
        """Accept a single value where a list is expected.

        Environment overrides arrive as scalars unless they contain a comma, so
        SQ4_EXPLAIN__SEEDS=0 would otherwise fail while SQ4_EXPLAIN__SEEDS=0,1
        succeeds. Wrapping here keeps the asymmetry out of the CLI.
        """
        if isinstance(value, str | int):
            return [value]
        return value


class Settings(BaseModel):
    """Root settings object handed to every component."""

    paths: PathSettings = PathSettings()
    logging: LoggingSettings = LoggingSettings()
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    explain: ExplainSettings = ExplainSettings()


def _coerce(raw: str) -> Any:
    """Parse an environment-variable string into a Python scalar or list."""
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if "," in raw:
        return [_coerce(part) for part in raw.split(",")]
    return raw


def _apply_env_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``SQ4_SECTION__KEY`` environment variables onto the TOML payload."""
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split(NESTED_DELIMITER)
        cursor: dict[str, Any] = payload
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(raw)
    return payload


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` until a directory containing config.toml is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "config.toml").is_file():
            return candidate
    return Path.cwd()


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from TOML, apply environment overrides and validate.

    Args:
        config_path: Explicit path to a TOML file. Defaults to ``config.toml``
            at the project root.

    Returns:
        A validated :class:`Settings` instance with absolute paths.
    """
    root = find_project_root(config_path)
    path = config_path or root / "config.toml"
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    settings = Settings.model_validate(_apply_env_overrides(payload))
    return settings.model_copy(update={"paths": settings.paths.resolve(root)})
