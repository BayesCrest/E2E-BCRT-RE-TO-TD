"""Proximal Policy Optimization trading agent implementation."""

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
    from ray.rllib.algorithms.ppo import PPOConfig
except ImportError:  # pragma: no cover
    PPOConfig = None


class PPOTradingAgent(TradingAgent):
    """Baseline PPO agent using RLlib when available."""

    def __init__(self, config: Dict[str, Any], checkpoint_dir: Path) -> None:
        super().__init__(config=config, checkpoint_dir=checkpoint_dir)
        self.policy_path = checkpoint_dir / "ppo_policy.json"

    def train(self, market_data: pd.DataFrame, features: pd.DataFrame) -> AgentArtifacts:
        if PPOConfig is None:
            LOGGER.warning("RLlib not installed; using heuristic PPO fallback policy.")
            weights = features.corrwith(market_data["Close"].pct_change().fillna(0.0)).fillna(0.0)
            policy = {
                "type": "heuristic",
                "weights": weights.to_dict(),
            }
            self.policy_path.write_text(json.dumps(policy, indent=2))
            metrics = {"method": "correlation_heuristic", "num_steps": len(features)}
            artifacts = AgentArtifacts(policy_path=self.policy_path, training_metrics=metrics)
            self.save_metadata(artifacts)
            return artifacts

        LOGGER.info("Training PPO agent with RLlib...")
        algo_config = (
            PPOConfig()
            .environment(env=self.config.get("env_name", "TradingEnv-v0"))
            .framework("torch")
            .rollouts(num_rollout_workers=self.config.get("num_workers", 1))
            .training(train_batch_size=self.config.get("train_batch_size", 4000))
        )
        algo = algo_config.build()
        for i in range(self.config.get("training_iterations", 5)):
            result = algo.train()
            LOGGER.info("Iteration %s: reward=%s", i, result["episode_reward_mean"])
        checkpoint = algo.save(checkpoint_dir=str(self.checkpoint_dir))
        metrics = {"checkpoint": checkpoint, "algorithm": "PPO"}
        self.policy_path.write_text(json.dumps(metrics, indent=2))
        artifacts = AgentArtifacts(policy_path=self.policy_path, training_metrics=metrics)
        self.save_metadata(artifacts)
        return artifacts

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.policy_path.exists():
            raise FileNotFoundError("Policy not trained; run `train` first.")

        policy = json.loads(self.policy_path.read_text())
        if policy.get("type") == "heuristic":
            weights = policy.get("weights", {})
            signal = sum(state.get(feature, 0.0) * weight for feature, weight in weights.items())
            action = np.tanh(signal)
            return {"allocation": float(action)}

        LOGGER.warning("Using zero-allocation action; RLlib policy loading not implemented in offline mode.")
        return {"allocation": 0.0}


__all__ = ["PPOTradingAgent"]
