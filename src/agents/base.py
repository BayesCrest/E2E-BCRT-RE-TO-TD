"""Base classes and utilities for reinforcement-learning trading agents."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class AgentArtifacts:
    """Artifacts returned after agent training."""

    policy_path: Path
    training_metrics: Dict[str, Any] = field(default_factory=dict)


class TradingAgent(abc.ABC):
    """Abstract reinforcement-learning agent interface."""

    def __init__(self, config: Dict[str, Any], checkpoint_dir: Path) -> None:
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def train(self, market_data: pd.DataFrame, features: pd.DataFrame) -> AgentArtifacts:
        """Train the agent and return generated artifacts."""

    @abc.abstractmethod
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Produce an action given the current state."""

    def save_metadata(self, artifacts: AgentArtifacts) -> None:
        metadata_path = artifacts.policy_path.with_suffix(".json")
        metadata_path.write_text(str(artifacts.training_metrics))


__all__ = ["AgentArtifacts", "TradingAgent"]
