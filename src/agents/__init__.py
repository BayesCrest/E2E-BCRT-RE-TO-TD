"""Agents package providing RL-based trading agents and research helpers."""

from .base import AgentArtifacts, TradingAgent
from .meta_optimizer import Candidate, EvolutionStrategy, EvolutionaryTuner
from .ppo_agent import PPOTradingAgent
from .research_helpers import ResearchAssistant, ResearchReport
from .sac_agent import SACTradingAgent

__all__ = [
    "AgentArtifacts",
    "Candidate",
    "EvolutionStrategy",
    "EvolutionaryTuner",
    "PPOTradingAgent",
    "ResearchAssistant",
    "ResearchReport",
    "SACTradingAgent",
    "TradingAgent",
]
