"""Bilevel learning of prior parameters with MAID.

MAID, the Method of Adaptive Inexact Descent, solves

.. math::

    \\min_\\theta f(\\theta) = g(x^\\star(\\theta)),
    \\qquad x^\\star(\\theta) = \\arg\\min_x h(x, \\theta),

choosing at each outer step how accurately the lower level is solved, and
certifying each reconstruction with an a posteriori bound. Implements
:footcite:t:`salehi2025adaptively`.
"""

from .maid import MAID, MAIDConfig, accelerated_maid_config
from .oracle import (
    HypergradientOracle,
    HypergradientState,
    LowerLevelState,
    strong_convexity_distance_bound,
)
from .quadratic_ls import QuadraticBilevelLS
from .smooth import (
    SmoothHypergradientOracle,
    hypergradient_error_bound,
    inexact_gradient,
    inexact_gradient_from_oracle,
    smooth_hypergradient_error_bound,
)

__all__ = [
    "MAID",
    "MAIDConfig",
    "accelerated_maid_config",
    "HypergradientOracle",
    "HypergradientState",
    "LowerLevelState",
    "strong_convexity_distance_bound",
    "QuadraticBilevelLS",
    "SmoothHypergradientOracle",
    "hypergradient_error_bound",
    "inexact_gradient",
    "inexact_gradient_from_oracle",
    "smooth_hypergradient_error_bound",
]
