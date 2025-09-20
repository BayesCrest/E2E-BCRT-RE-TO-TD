"""Evolutionary tuner for agent hyper-parameters."""

from __future__ import annotations

import dataclasses
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class Candidate:
    params: Dict[str, Any]
    score: float = 0.0


@dataclass
class EvolutionStrategy:
    population_size: int = 8
    mutation_rate: float = 0.2
    elite_fraction: float = 0.25
    generations: int = 5
    search_space: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    def sample_params(self) -> Dict[str, Any]:
        sampled = {}
        for key, value in self.search_space.items():
            if isinstance(value, tuple) and len(value) == 2:
                low, high = value
                if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    sampled[key] = random.uniform(low, high)
                else:
                    sampled[key] = random.choice(value)
            else:
                sampled[key] = value
        return sampled


class EvolutionaryTuner:
    """Simple evolutionary strategy to tune RL agent hyper-parameters."""

    def __init__(self, strategy: EvolutionStrategy) -> None:
        self.strategy = strategy

    def optimize(self, evaluate_fn) -> Candidate:
        """Run the optimization loop and return the best candidate."""

        population = [Candidate(params=self.strategy.sample_params()) for _ in range(self.strategy.population_size)]
        best = population[0]
        for generation in range(self.strategy.generations):
            LOGGER.info("Evolutionary generation %s", generation)
            for candidate in population:
                candidate.score = evaluate_fn(candidate.params)
                if candidate.score > best.score:
                    best = dataclasses.replace(candidate)

            population = self._reproduce(population)
        return best

    def _reproduce(self, population: List[Candidate]) -> List[Candidate]:
        population = sorted(population, key=lambda c: c.score, reverse=True)
        elite_count = max(1, int(self.strategy.elite_fraction * len(population)))
        elites = population[:elite_count]
        new_population = elites.copy()
        while len(new_population) < self.strategy.population_size:
            parent = random.choice(elites)
            child_params = parent.params.copy()
            if random.random() < self.strategy.mutation_rate:
                key = random.choice(list(child_params.keys()))
                space = self.strategy.search_space.get(key)
                if isinstance(space, tuple) and len(space) == 2 and all(
                    isinstance(val, (int, float)) for val in space
                ):
                    child_params[key] = random.uniform(space[0], space[1])
            new_population.append(Candidate(params=child_params))
        return new_population


__all__ = ["Candidate", "EvolutionStrategy", "EvolutionaryTuner"]
