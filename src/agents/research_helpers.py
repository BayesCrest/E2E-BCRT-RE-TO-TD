"""LLM-powered research and reporting helpers integrated with RL agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from ..features.sentiment import SentimentExtractor, build_sentiment_features

try:  # pragma: no cover
    from langchain.chains import LLMChain
    from langchain.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover
    LLMChain = None
    ChatPromptTemplate = None

LOGGER = logging.getLogger(__name__)


@dataclass
class ResearchReport:
    title: str
    summary: str
    recommended_action: str
    supporting_points: List[str]


class ResearchAssistant:
    """Coordinate LLM insights with RL agent outputs for decision support."""

    def __init__(self, sentiment_extractor: SentimentExtractor) -> None:
        self.sentiment_extractor = sentiment_extractor
        if LLMChain is None:
            LOGGER.warning("LangChain not installed; research assistant will generate template reports.")
        else:
            prompt = ChatPromptTemplate.from_template(
                """
                You are part of a quantitative trading research pod.
                Given market context, agent signals, and sentiment summaries, draft an actionable report with:
                - Title
                - Executive summary
                - Recommended action (long/short/hold)
                - Three supporting points
                Context:
                {context}
                Sentiment:
                {sentiment}
                Agent Signals:
                {signals}
                """
            )
            self.chain = LLMChain(prompt=prompt)

    def build_context(self, market_data: Any, features: Any) -> str:
        return (
            "Market snapshot rows="
            f"{getattr(market_data, 'shape', ('?', '?'))}, features={list(getattr(features, 'columns', []))}"
        )

    def generate_report(self, documents: List[str], agent_signals: Dict[str, Any], context: str) -> ResearchReport:
        sentiment_results = self.sentiment_extractor.score_documents(documents)
        sentiment_df = build_sentiment_features(sentiment_results)
        aggregate_sentiment = sentiment_df["sentiment_score"].mean()
        sentiment_summary = ", ".join(
            f"score={result.sentiment_score:.2f} ({result.rationale[:60]}...)" for result in sentiment_results
        )
        sentiment_summary += f" | average={aggregate_sentiment:.2f}"

        if LLMChain is None:
            LOGGER.info("Generating template research report.")
            return ResearchReport(
                title="Quant Research Update",
                summary=f"Context: {context}. Sentiment: {sentiment_summary}.",
                recommended_action="hold",
                supporting_points=[
                    "Agent signals pending advanced model training.",
                    "Sentiment heuristic suggests cautious positioning.",
                    "Macro regime alignment requires human validation.",
                ],
            )

        response = self.chain.run(
            context=context,
            sentiment=sentiment_summary,
            signals=str(agent_signals),
        )
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        title = lines[0] if lines else "Research Update"
        summary = next((line for line in lines if line.lower().startswith("summary")), "")
        action = next(
            (line for line in lines if any(key in line.lower() for key in ("long", "short", "hold"))),
            "hold",
        )
        supporting = lines[-3:] if len(lines) >= 3 else lines
        return ResearchReport(
            title=title,
            summary=summary,
            recommended_action=action,
            supporting_points=supporting,
        )


__all__ = ["ResearchAssistant", "ResearchReport"]
