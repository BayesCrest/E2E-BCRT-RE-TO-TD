"""Sample script demonstrating an end-to-end research-to-trading workflow."""

from __future__ import annotations

import json

import pandas as pd

from e2e_bcrt_re_to_td import DEFAULT_CONFIG
from e2e_bcrt_re_to_td.agents import PPOTradingAgent, ResearchAssistant
from e2e_bcrt_re_to_td.data.ingestion import run_data_ingestion
from e2e_bcrt_re_to_td.evaluation import generate_html_report, simple_backtest
from e2e_bcrt_re_to_td.features import SentimentExtractor, assemble_technical_features


def main() -> None:
    config = DEFAULT_CONFIG
    config.ensure_directories()

    ingestion_result = run_data_ingestion(
        symbols=config.data.symbols,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        interval=config.data.interval,
    )
    symbol = config.data.symbols[0]
    market_df = ingestion_result.market_data[symbol]
    technical_features = assemble_technical_features(market_df, config.features.technical_indicators)

    sentiment_extractor = SentimentExtractor(model=config.features.sentiment_model)
    research_assistant = ResearchAssistant(sentiment_extractor=sentiment_extractor)
    context = research_assistant.build_context(market_df.tail(5), technical_features.tail(5))

    agent = PPOTradingAgent(config=config.agent.__dict__, checkpoint_dir=config.agent.checkpoint_dir)
    agent.train(market_df, technical_features)
    signals = pd.Series(
        [agent.act(row.to_dict())["allocation"] for _, row in technical_features.iterrows()],
        index=technical_features.index,
    )

    backtest = simple_backtest(prices=market_df["Close"], signals=signals)
    equity_path = config.backtest.evaluation_dir / "equity.parquet"
    equity_path.parent.mkdir(parents=True, exist_ok=True)
    backtest.trades.reset_index(names="date").to_parquet(equity_path)
    report_path = config.backtest.evaluation_dir / "report.html"
    metrics = generate_html_report(backtest.equity_curve, backtest.trades, report_path)

    documents = [
        "Company beat earnings expectations with strong revenue growth.",
        "Management guided for modest slowdown next quarter but remains optimistic.",
    ]
    research_report = research_assistant.generate_report(documents, agent_signals=metrics, context=context)
    (config.backtest.evaluation_dir / "research_report.json").write_text(json.dumps(research_report.__dict__, indent=2))

    print("Workflow complete.")
    print("Metrics:", metrics)
    print("Research Report:", research_report)


if __name__ == "__main__":
    main()
