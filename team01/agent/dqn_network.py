"""Small neural-network and replay-buffer components for Project 2 DQN."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


FeatureVector = tuple[float, ...]


def ordered_feature_vector(
    feature_values: Mapping[str, float],
    feature_names: Sequence[str],
) -> FeatureVector:
    """Return feature values in the explicit schema order."""
    if set(feature_values) != set(feature_names):
        raise ValueError("Feature values do not match the supplied feature names")
    return tuple(float(feature_values[name]) for name in feature_names)


class DQNNetwork(nn.Module):
    """Score one legal action from its state-action feature vector.

    Actions are evaluated separately: the action is encoded in the input
    features, and the network returns one scalar Q(s, a), not one output per
    action.
    """

    def __init__(self, input_size: int):
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be at least 1")

        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.layer1(features))
        hidden = self.relu(self.layer2(hidden))
        return self.output(hidden)


@dataclass(frozen=True)
class Transition:
    """One action-conditioned experience replay item."""

    features: FeatureVector
    reward: float
    next_action_features: tuple[FeatureVector, ...]
    done: bool


class ReplayBuffer:
    """Store recent transitions and sample uniform minibatches.

    Sampling reuses experience across updates and reduces the correlation
    between consecutive environment steps.
    """

    def __init__(self, capacity: int = 50_000, feature_size: int | None = None):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if feature_size is not None and feature_size < 1:
            raise ValueError("feature_size must be at least 1 when supplied")
        self.capacity = capacity
        self.feature_size = feature_size
        self._transitions: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        if self.feature_size is not None:
            if len(transition.features) != self.feature_size:
                raise ValueError(
                    "Transition feature size does not match replay schema: "
                    f"{len(transition.features)} != {self.feature_size}"
                )
            if any(
                len(features) != self.feature_size
                for features in transition.next_action_features
            ):
                raise ValueError(
                    "Next-action feature size does not match replay schema "
                    f"of {self.feature_size}"
                )
        self._transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > len(self._transitions):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer "
                f"containing {len(self._transitions)}"
            )
        return random.sample(list(self._transitions), batch_size)

    def __len__(self) -> int:
        return len(self._transitions)
