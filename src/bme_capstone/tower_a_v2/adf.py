"""ADF-based Dirichlet wrappers and distance helpers for Tower A v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import torch
from torch import nn

from pina import LabelTensor

from .geometry import PlanePatch, _AXES


def _abs_smooth(x: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0.0:
        return torch.abs(x)
    return torch.sqrt(x * x + eps * eps)


def distance_to_rectangular_patch(
    coords: torch.Tensor,
    patch: PlanePatch,
    *,
    smooth_eps: float = 1e-6,
    normalise: bool = True,
) -> torch.Tensor:
    """Return distance to a rectangular gauge patch (vanishes on the patch).

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape (batch, 3) with (x, y, z) coordinates.
    patch : PlanePatch
        Patch representing the Dirichlet gauge.
    smooth_eps : float, optional
        If > 0, applies a smooth approximation to |x| and ReLU to avoid kinks.
    normalise : bool, optional
        If True, divide the distance by max half-width to keep values O(1).
    """

    axis_idx = _AXES.index(patch.axis)
    other_axes = [ax for ax in _AXES if ax != patch.axis]
    other_indices = [
        _AXES.index(ax)
        for ax in other_axes
    ]

    half_sizes = []
    centres = []
    for ax in other_axes:
        lo, hi = patch.span[ax]
        centres.append((lo + hi) * 0.5)
        half_sizes.append((hi - lo) * 0.5)

    centres_tensor = coords.new_tensor(centres)
    half_tensor = coords.new_tensor(half_sizes)

    delta = coords[:, other_indices] - centres_tensor
    abs_delta = _abs_smooth(delta, smooth_eps)
    outside = abs_delta - half_tensor
    if smooth_eps > 0.0:
        outside = torch.nn.functional.softplus(outside / smooth_eps) * smooth_eps
    else:
        outside = torch.clamp(outside, min=0.0)
    rho = torch.sqrt((outside * outside).sum(dim=1) + smooth_eps * smooth_eps)

    perpendicular = _abs_smooth(coords[:, axis_idx] - float(patch.value), smooth_eps)

    distance = rho + perpendicular

    if normalise:
        scale = torch.max(half_tensor.max(), coords.new_tensor(1.0))
        distance = distance / scale

    return distance


class DirichletGaugeWrapper(nn.Module):
    """Wrap a base model so the output is constrained via an ADF distance."""

    def __init__(
        self,
        base_model: nn.Module,
        patch: PlanePatch,
        *,
        distance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        smooth_eps: float = 1e-6,
        output_labels: Optional[Union[Sequence[str], dict]] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.patch = patch
        self.smooth_eps = smooth_eps
        self._distance_fn = distance_fn
        self._output_labels: Union[Sequence[str], dict] = output_labels or ["phi"]

    def _compute_distance(self, coords: torch.Tensor) -> torch.Tensor:
        if self._distance_fn is not None:
            return self._distance_fn(coords)
        return distance_to_rectangular_patch(coords, self.patch, smooth_eps=self.smooth_eps)

    def forward(self, inputs):  # type: ignore[override]
        if isinstance(inputs, LabelTensor):
            coords = inputs.tensor
        else:
            coords = inputs
        distance = self._compute_distance(coords).unsqueeze(-1)

        raw = self.base_model(inputs)
        if isinstance(raw, LabelTensor):
            base_tensor = raw.tensor
            labels = getattr(raw, "_labels", None)
        else:
            base_tensor = raw
            labels = None

        constrained = distance * base_tensor

        if labels is None:
            labels = self._output_labels

        return LabelTensor(constrained, labels)
