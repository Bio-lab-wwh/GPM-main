from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn

_RUNTIME_PERT_GROUP_SIZE: Optional[int] = None
_MIN_EPS = 1e-6


def set_pert_group_size(size: int) -> int:
    global _RUNTIME_PERT_GROUP_SIZE
    _RUNTIME_PERT_GROUP_SIZE = max(1, int(size))

    return _RUNTIME_PERT_GROUP_SIZE


def perturbation_group_size() -> int:
    if _RUNTIME_PERT_GROUP_SIZE is not None:
        return _RUNTIME_PERT_GROUP_SIZE
    from GPM.train_config import load_training_settings

    return set_pert_group_size(int(load_training_settings()["pert_group_size"]))


def default_loss_weights() -> dict[str, float]:
    from GPM.train_config import load_loss_weights

    return load_loss_weights()


def merge_loss_weights(overrides: Optional[Mapping[str, float]] = None) -> dict[str, float]:
    if overrides is None:
        try:
            weights = default_loss_weights()
        except Exception:
            weights = {}
    else:
        weights = {str(key): float(value) for key, value in overrides.items()}
    resolved = {
        "expression_mse": float(weights.get("expression_mse", 1.0)),
        "ot": float(weights.get("ot", 0.05)),
    }
    if any(value < 0.0 for value in resolved.values()):
        raise ValueError("Loss weights must be non-negative.")
    return resolved


@dataclass
class GPMLossTargets:
    target_expression: torch.Tensor
    pert_group_slices: Optional[list[tuple[int, int]]] = None


@dataclass
class GPMLossOutput:
    loss_mse_expression: torch.Tensor
    loss_sinkhorn: torch.Tensor
    loss_total: torch.Tensor


def _fp32(value: torch.Tensor) -> torch.Tensor:
    return value if value.dtype in (torch.float32, torch.float64) else value.float()


def _group_slices(
    size: int, slices: Optional[Sequence[tuple[int, int]]]
) -> list[tuple[int, int]]:
    groups = [
        (int(start), int(end))
        for start, end in (slices or [(0, size)])
        if int(end) > int(start)
    ]
    if not groups:
        raise ValueError("At least one non-empty perturbation group is required.")
    return groups


def grouped_mean_expression_mse(
    pred_expression: torch.Tensor,
    target_expression: torch.Tensor,
    pert_group_slices: Optional[Sequence[tuple[int, int]]],
) -> torch.Tensor:
    if pred_expression.shape != target_expression.shape:
        raise ValueError("Predicted and target expression shapes differ.")
    losses = []
    for start, end in _group_slices(pred_expression.shape[0], pert_group_slices):
        pred_mean = _fp32(pred_expression[start:end]).mean(dim=0)
        true_mean = _fp32(target_expression[start:end]).mean(dim=0).detach()
        losses.append((pred_mean - true_mean).square().mean())
    return torch.stack(losses).mean()


def _squared_cost(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(float(left.shape[-1]))
    left = _fp32(left) * scale
    right = _fp32(right) * scale
    return (
        left.square().sum(dim=-1, keepdim=True)
        + right.square().sum(dim=-1, keepdim=True).transpose(1, 2)
        - 2.0 * torch.bmm(left, right.transpose(1, 2))
    ).clamp_min(0.0)


@torch.jit.script
def _uniform_log_plan(cost: torch.Tensor, log_mass: float, iterations: int) -> torch.Tensor:
    row = cost.new_zeros((cost.size(0), cost.size(1)))
    column = cost.new_zeros((cost.size(0), cost.size(2)))
    for _ in range(iterations):
        row = log_mass - torch.logsumexp(column.unsqueeze(1) - cost, dim=2)
        column = log_mass - torch.logsumexp(row.unsqueeze(2) - cost, dim=1)
    return torch.exp(row.unsqueeze(2) + column.unsqueeze(1) - cost)


def grouped_sinkhorn_loss(
    pred_expression: torch.Tensor,
    target_expression: torch.Tensor,
    pert_group_slices: Optional[Sequence[tuple[int, int]]],
    *,
    blur: float,
    sinkhorn_iters: int,
) -> torch.Tensor:
    if pred_expression.shape != target_expression.shape:
        raise ValueError("Predicted and target expression shapes differ.")
    groups = _group_slices(pred_expression.shape[0], pert_group_slices)
    if len({end - start for start, end in groups}) != 1:
        raise ValueError("All OT groups in one micro-batch must have equal size.")
    pred = torch.stack([pred_expression[start:end] for start, end in groups])
    target = torch.stack([target_expression[start:end] for start, end in groups]).detach()
    num_groups, group_size, _ = pred.shape
    left = torch.cat((pred, pred, target), dim=0)
    right = torch.cat((target, pred, target), dim=0)
    cost = _squared_cost(left, right)
    plan = _uniform_log_plan(
        cost / max(float(blur), _MIN_EPS),
        -math.log(float(group_size)),
        max(1, int(sinkhorn_iters)),
    )
    transport = (plan * cost).sum(dim=(1, 2)).reshape(3, num_groups)
    return (transport[0] - 0.5 * (transport[1] + transport[2])).clamp_min(0.0).mean()


def compute_gpm_loss(
    pred_expression: torch.Tensor,
    targets: GPMLossTargets,
    *,
    blur: float = 0.02,
    sinkhorn_iters: int = 50,
    weights: Optional[Mapping[str, float]] = None,
) -> GPMLossOutput:
    resolved = merge_loss_weights(weights)
    mse = grouped_mean_expression_mse(
        pred_expression, targets.target_expression, targets.pert_group_slices
    )
    sinkhorn = grouped_sinkhorn_loss(
        pred_expression,
        targets.target_expression,
        targets.pert_group_slices,
        blur=blur,
        sinkhorn_iters=sinkhorn_iters,
    )
    total = resolved["expression_mse"] * mse + resolved["ot"] * sinkhorn
    return GPMLossOutput(mse, sinkhorn, total)


class GPMLoss(nn.Module):
    def __init__(
        self,
        *,
        blur: float = 0.02,
        sinkhorn_iters: int = 50,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.blur = float(blur)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.weights = merge_loss_weights(weights)

    def forward(
        self, pred_expression: torch.Tensor, targets: GPMLossTargets
    ) -> GPMLossOutput:
        return compute_gpm_loss(
            pred_expression,
            targets,
            blur=self.blur,
            sinkhorn_iters=self.sinkhorn_iters,
            weights=self.weights,
        )
