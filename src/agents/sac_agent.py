"""Soft Actor-Critic trading agent implementation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import AgentArtifacts, TradingAgent

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    from ray.rllib.algorithms.sac import SACConfig
except ImportError:  # pragma: no cover
    SACConfig = None


class SACTradingAgent(TradingAgent):
    """SAC agent with optional RLlib integration and heuristic fallback."""

    def __init__(self, config: Dict[str, Any], checkpoint_dir: Path) -> None:
        super().__init__(config=config, checkpoint_dir=checkpoint_dir)
        self.policy_path = checkpoint_dir / "sac_policy.json"

    def train(self, market_data: pd.DataFrame, features: pd.DataFrame) -> AgentArtifacts:
        if SACConfig is None:
            LOGGER.warning("RLlib not installed; using entropy-scaled heuristic for SAC.")
            volatility = features.std().fillna(0.0)
            policy = {
                "type": "volatility_scaled",
                "volatility": volatility.to_dict(),
            }
            self.policy_path.write_text(json.dumps(policy, indent=2))
            metrics = {"method": "volatility_scaled", "num_features": len(volatility)}
            artifacts = AgentArtifacts(policy_path=self.policy_path, training_metrics=metrics)
            self.save_metadata(artifacts)
            return artifacts

        LOGGER.info("Training SAC agent with RLlib...")
        algo_config = (
            SACConfig()
            .environment(env=self.config.get("env_name", "TradingEnv-v0"))
            .framework("torch")
            .training(train_batch_size=self.config.get("train_batch_size", 256))
        )
        algo = algo_config.build()
        for i in range(self.config.get("training_iterations", 5)):
            result = algo.train()
            LOGGER.info("Iteration %s: reward=%s", i, result["episode_reward_mean"])
        checkpoint = algo.save(checkpoint_dir=str(self.checkpoint_dir))
        metrics = {"checkpoint": checkpoint, "algorithm": "SAC"}
        self.policy_path.write_text(json.dumps(metrics, indent=2))
        artifacts = AgentArtifacts(policy_path=self.policy_path, training_metrics=metrics)
        self.save_metadata(artifacts)
        return artifacts

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.policy_path.exists():
            raise FileNotFoundError("Policy not trained; run `train` first.")

        policy = json.loads(self.policy_path.read_text())
        if policy.get("type") == "volatility_scaled":
            volatility = policy.get("volatility", {})
            signal = sum(abs(state.get(feature, 0.0)) * vol for feature, vol in volatility.items())
            action = float(np.clip(signal, 0, 1))
            return {"allocation": action}

        LOGGER.warning("Using conservative action; RLlib policy loading not implemented in offline mode.")
        return {"allocation": 0.0}


__all__ = ["SACTradingAgent"]
