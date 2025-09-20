"""LLM-powered sentiment extraction utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

try:  # pragma: no cover
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
except ImportError:  # pragma: no cover
    ChatOpenAI = None
    HumanMessage = None

LOGGER = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Structured result returned by the sentiment extractor."""

    text: str
    sentiment_score: float
    rationale: str


class SentimentExtractor:
    """Wrapper around LLM APIs to obtain sentiment for news or transcripts."""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.client = None
        if ChatOpenAI is None:
            LOGGER.warning(
                "LangChain not available. Sentiment extraction will fall back to rule-based heuristics."
            )
        else:
            self.client = ChatOpenAI(model=self.model, temperature=self.temperature, openai_api_key=api_key)

    def score_documents(self, documents: Iterable[str]) -> List[SentimentResult]:
        """Generate sentiment scores for the provided documents."""

        results: List[SentimentResult] = []
        for doc in documents:
            if self.client is None:
                score = self._rule_based_score(doc)
                results.append(
                    SentimentResult(
                        text=doc,
                        sentiment_score=score,
                        rationale="Rule-based heuristic score due to missing LangChain dependency.",
                    )
                )
                continue

            response = self.client([HumanMessage(content=self._build_prompt(doc))])
            metadata = response.additional_kwargs.get("function_call") or {}
            score = self._parse_score(response.content) or float(metadata.get("sentiment", 0.0))
            rationale = response.content
            results.append(SentimentResult(text=doc, sentiment_score=score, rationale=rationale))
        return results

    @staticmethod
    def _build_prompt(document: str) -> str:
        return (
            "You are a financial research assistant. Score the sentiment of the following document on a scale "
            "from -1 (very bearish) to +1 (very bullish). Provide a short explanation.\n\n"
            f"Document:\n{document}"
        )

    @staticmethod
    def _parse_score(response_text: str) -> Optional[float]:
        try:
            for token in response_text.split():
                if token.replace(".", "", 1).replace("-", "", 1).isdigit():
                    return float(token)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to parse sentiment score: %s", exc)
        return None

    @staticmethod
    def _rule_based_score(document: str) -> float:
        positive_words = {"beat", "growth", "outperform", "bullish"}
        negative_words = {"miss", "slowdown", "layoff", "bearish"}
        tokens = {word.strip(".,!?").lower() for word in document.split()}
        score = 0.0
        score += sum(1 for token in tokens if token in positive_words)
        score -= sum(1 for token in tokens if token in negative_words)
        return max(min(score / 5.0, 1.0), -1.0)


def build_sentiment_features(sentiments: List[SentimentResult]) -> pd.DataFrame:
    """Aggregate sentiment results into a tabular feature matrix."""

    df = pd.DataFrame(
        {
            "text": [result.text for result in sentiments],
            "sentiment_score": [result.sentiment_score for result in sentiments],
            "rationale": [result.rationale for result in sentiments],
        }
    )
    df["sentiment_bucket"] = pd.cut(
        df["sentiment_score"], bins=[-1.0, -0.2, 0.2, 1.0], labels=["bearish", "neutral", "bullish"]
    )
    return df


__all__ = [
    "SentimentExtractor",
    "SentimentResult",
    "build_sentiment_features",
]
