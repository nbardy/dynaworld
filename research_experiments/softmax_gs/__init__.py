"""Softmax-GS reference utilities for DynaWorld experiments."""

from .reference import (
    SoftmaxGSBoundedTape,
    SoftmaxGSDebugRow,
    SoftmaxGSTapeRow,
    softmax_gs_bounded_contribution_tape,
    softmax_gs_composite,
    softmax_gs_contribution_tape,
    vanilla_alpha_over,
)

__all__ = [
    "SoftmaxGSBoundedTape",
    "SoftmaxGSDebugRow",
    "SoftmaxGSTapeRow",
    "softmax_gs_bounded_contribution_tape",
    "softmax_gs_composite",
    "softmax_gs_contribution_tape",
    "vanilla_alpha_over",
]
