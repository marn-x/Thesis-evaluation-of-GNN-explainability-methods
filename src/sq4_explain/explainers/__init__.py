"""Explanation methods.

Importing this package registers every bundled method. To add one, drop a
module here that decorates its class with ``@register_explainer`` and import
it below; no other file needs to change.
"""

from sq4_explain.explainers.base import (
    ExplanationResult,
    NodeExplainer,
    available_explainers,
    build_explainer,
    register_explainer,
)
from sq4_explain.explainers.gnn_explainer import GNNExplainerRunner
from sq4_explain.explainers.shap_exact import ExactShapRunner
from sq4_explain.explainers.shap_sampled import SampledShapRunner

__all__ = [
    "ExactShapRunner",
    "ExplanationResult",
    "GNNExplainerRunner",
    "NodeExplainer",
    "SampledShapRunner",
    "available_explainers",
    "build_explainer",
    "register_explainer",
]
