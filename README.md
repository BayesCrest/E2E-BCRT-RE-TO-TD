# E2E-BCRT-RE-TO-TD

End-to-end research-to-trading toolkit featuring data ingestion, feature engineering, reinforcement-learning agents, LLM-driven research assistance, evaluation utilities, and dashboard monitoring.

## Project Structure

```
.
├── pyproject.toml
├── src/
│   ├── config.py
│   ├── data/
│   │   └── ingestion.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── macro.py
│   │   ├── sentiment.py
│   │   └── technical.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── meta_optimizer.py
│   │   ├── ppo_agent.py
│   │   ├── research_helpers.py
│   │   └── sac_agent.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── backtest.py
│   │   ├── metrics.py
│   │   └── reporting.py
│   └── dashboard/
│       └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── examples/
│   └── end_to_end.py
└── notebooks/
    └── quickstart.ipynb
```

## Getting Started

### Prerequisites

* Python 3.9+
* Recommended: virtual environment (``python -m venv .venv``)

### Installation

```bash
pip install -e .[notebooks,rllib,finrl]
```

The optional extras install RLlib/FinRL integrations and notebook tooling. Omit extras as needed.

### Configuration

Project-level defaults live in [`src/config.py`](src/config.py). Override them via environment variables, custom scripts, or by instantiating your own `ProjectConfig`.

Directories for raw data, processed data, agent checkpoints, and evaluation artifacts are created automatically when `ProjectConfig.ensure_directories()` is called.

## Workflow Overview

1. **Data ingestion** – Pull historical market prices and synthetic alternative datasets using [`src/data/ingestion.py`](src/data/ingestion.py). Offline environments fall back to deterministic synthetic data.
2. **Feature engineering** – Create technical, sentiment, and macro features with [`src/features`](src/features).
3. **Agents & tuning** – Train PPO/SAC agents or heuristics and tune them with the evolutionary meta-optimizer in [`src/agents`](src/agents).
4. **Research assistance** – Generate LLM-driven research reports coordinating agent signals via [`ResearchAssistant`](src/agents/research_helpers.py).
5. **Evaluation** – Run backtests, compute metrics, and export HTML reports using [`src/evaluation`](src/evaluation).
6. **Dashboard** – Monitor pipeline outputs through the Streamlit dashboard [`src/dashboard/app.py`](src/dashboard/app.py).

## Example End-to-End Script

[`examples/end_to_end.py`](examples/end_to_end.py) demonstrates a lightweight run that:

1. Loads default configuration and ensures directories exist.
2. Runs data ingestion and builds technical/sentiment features.
3. Trains a heuristic PPO agent (or RLlib PPO if installed).
4. Executes a simple backtest.
5. Generates a template research report and stores artifacts for the dashboard.

Execute the script with:

```bash
python examples/end_to_end.py
```

## Streamlit Dashboard

After running the example workflow, launch the dashboard:

```bash
streamlit run src/dashboard/app.py
```

The dashboard displays ingestion metadata, equity curve metrics, and the latest research report produced by the research assistant.

## Notebooks

`notebooks/quickstart.ipynb` is a compact notebook walking through the ingestion → features → agent → evaluation lifecycle. Launch Jupyter Lab after installing the `notebooks` extra:

```bash
jupyter lab
```

## Testing & Development

* Compile sources: `python -m compileall src`
* Run linting/formatting as desired (e.g., `ruff`, `black`).

## License

MIT
