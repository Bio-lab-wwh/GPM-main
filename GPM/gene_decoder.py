from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectPerturbationEffectDecoder(nn.Module):
    """Decode the total perturbation response into signed gene-wise expression changes."""

    def __init__(
        self,
        response_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        # Bias-free layers map zero response to zero expression change.
        self.response_projection = nn.Linear(int(response_dim), hidden, bias=False)
        self.hidden = nn.Linear(hidden, hidden // 2, bias=False)
        self.output = nn.Linear(hidden // 2, 1, bias=False)

    def forward(
        self,
        propagated_response: torch.Tensor,
    ) -> torch.Tensor:
        response_hidden = self.response_projection(propagated_response)
        return self.output(
            F.silu(self.hidden(F.silu(response_hidden)))
        ).squeeze(-1)
