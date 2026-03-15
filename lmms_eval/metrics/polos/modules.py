"""Feed-forward network, scalar mix, and activation helpers.

Vendored from polos/modules/ with import paths updated.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.nn import Parameter, ParameterList

# ── Activation helpers ──────────────────────────────────────────────


def build_activation(activation: str) -> nn.Module:
    """Return an activation function module by name."""
    if hasattr(nn, activation):
        return getattr(nn, activation)()
    if activation == "Swish":
        return Swish()
    raise ValueError(f"{activation} is not a valid activation function.")


class Swish(nn.Module):
    """Swish(x) = x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# ── FeedForward ─────────────────────────────────────────────────────


class FeedForward(nn.Module):
    """Simple feed-forward network for regression / classification."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: str | list[int] = "3072,1536,768",
        activations: str = "Sigmoid",
        final_activation: str | None = "Sigmoid",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if isinstance(hidden_sizes, str):
            hidden_sizes = [int(x) for x in hidden_sizes.split(",")]
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        activation_func = build_activation(activations)

        self.ff = nn.Sequential()
        self.ff.add_module("linear_1", nn.Linear(in_dim, hidden_sizes[0]))
        self.ff.add_module("activation_1", activation_func)
        self.ff.add_module("dropout_1", nn.Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            self.ff.add_module(
                f"linear_{i + 1}",
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
            )
            self.ff.add_module(f"activation_{i + 1}", activation_func)
            self.ff.add_module(f"dropout_{i + 1}", nn.Dropout(dropout))

        self.ff.add_module(
            f"linear_{len(hidden_sizes) + 1}",
            nn.Linear(hidden_sizes[-1], int(out_dim)),
        )
        if final_activation:
            self.ff.add_module(
                f"activation_{len(hidden_sizes) + 1}",
                build_activation(final_activation),
            )

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)


# ── ScalarMixWithDropout ────────────────────────────────────────────


class ScalarMixWithDropout(nn.Module):
    """Layer-wise attention mechanism for mixing encoder layers."""

    def __init__(
        self,
        mixture_size: int,
        do_layer_norm: bool = False,
        initial_scalar_parameters: list[float] | None = None,
        trainable: bool = True,
        dropout: float | None = None,
        dropout_value: float = -1e20,
    ) -> None:
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.dropout = dropout

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]),
                    requires_grad=trainable,
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

        if self.dropout:
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(dropout_value)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(
        self,
        tensors: List[torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted average of tensors."""
        if len(tensors) != self.mixture_size:
            raise ValueError(f"{len(tensors)} tensors passed, but module " f"initialized to mix {self.mixture_size} tensors.")

        weights = torch.cat([p for p in self.scalar_parameters])

        if self.training and self.dropout:
            weights = torch.where(
                self.dropout_mask.uniform_() > self.dropout,
                weights,
                self.dropout_fill,
            )

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = [w * t for w, t in zip(normed_weights, tensors)]
            return self.gamma * sum(pieces)

        mask_float = mask.float()
        broadcast_mask = mask_float.unsqueeze(-1)
        input_dim = tensors[0].size(-1)
        num_elements = torch.sum(mask_float) * input_dim

        def _do_layer_norm(
            tensor: torch.Tensor,
        ) -> torch.Tensor:
            masked = tensor * broadcast_mask
            mean = torch.sum(masked) / num_elements
            variance = torch.sum(((masked - mean) * broadcast_mask) ** 2) / num_elements
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        pieces = [w * _do_layer_norm(t) for w, t in zip(normed_weights, tensors)]
        return self.gamma * sum(pieces)
