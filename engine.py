"""Data analysis, explainable signals, and portfolio allocation for the demo app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the bundled, synthetic dataset used by the runnable demonstration."""
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
    fundamentals = pd.read_csv(DATA_DIR / "fundamentals.csv")
    discussions = pd.read_csv(DATA_DIR / "news_social.csv", parse_dates=["published_at"])
    return prices, fundamentals, discussions


def technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Add common trend and momentum indicators independently for each symbol."""
    frames: list[pd.DataFrame] = []
    for ticker, frame in prices.groupby("ticker", sort=False):
        frame = frame.sort_values("date").copy()
        close = frame["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
        relative_strength = gain / loss.replace(0, np.nan)
        frame["sma_20"] = close.rolling(20).mean()
        frame["sma_50"] = close.rolling(50).mean()
        rsi = 100 - (100 / (1 + relative_strength))
        rsi = rsi.mask((loss == 0) & (gain > 0), 100)
        rsi = rsi.mask((loss == 0) & (gain == 0), 50)
        frame["rsi_14"] = rsi
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        frame["macd"] = ema_12 - ema_26
        frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
        frame["daily_return"] = close.pct_change()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


POSITIVE_WORDS = {
    "beat", "growth", "upgrade", "strong", "record", "profit", "surge", "bullish",
    "innovation", "expansion", "positive", "outperform", "improve", "momentum", "gain",
}
NEGATIVE_WORDS = {
    "miss", "decline", "downgrade", "weak", "loss", "drop", "bearish", "risk",
    "lawsuit", "concern", "negative", "underperform", "cut", "fall", "pressure",
}


def _lexicon_sentiment(text: str) -> float:
    words = {word.strip(".,!?;:()[]{}\"'").lower() for word in str(text).split()}
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    total = positive + negative
    return 0.0 if total == 0 else float((positive - negative) / total)


def score_discussions(discussions: pd.DataFrame, use_finbert: bool = False) -> tuple[pd.DataFrame, str]:
    """Score news and social text; FinBERT can be enabled when its optional stack is installed."""
    result = discussions.copy()
    if use_finbert:
        try:
            from transformers import pipeline

            classifier = pipeline(
                "text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert",
                truncation=True, max_length=512,
            )
            scores = []
            for text in result["text"].astype(str):
                predictions = classifier(text)[0]
                label = str(predictions["label"]).lower()
                confidence = float(predictions["score"])
                scores.append(confidence if label == "positive" else -confidence if label == "negative" else 0.0)
            result["sentiment"] = scores
            return result, "FinBERT transformer"
        except ImportError as exc:
            raise RuntimeError("FinBERT requires the optional packages in requirements-ml.txt.") from exc
    result["sentiment"] = result["text"].map(_lexicon_sentiment)
    return result, "Lexicon fallback (FinBERT is optional)"


def _percentile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    rank = values.rank(pct=True, method="average").fillna(0.5)
    return rank if higher_is_better else 1 - rank


def build_recommendations(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    discussions: pd.DataFrame,
    use_finbert: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Combine transparent fundamentals, technicals, and text-sentiment factors."""
    indicators = technical_indicators(prices)
    latest = indicators.sort_values("date").groupby("ticker", as_index=False).tail(1)
    sentiment_rows, sentiment_method = score_discussions(discussions, use_finbert=use_finbert)
    sentiment = sentiment_rows.groupby("ticker")["sentiment"].mean().rename("raw_sentiment")
    rows = fundamentals.merge(latest, on="ticker", how="inner").join(sentiment, on="ticker")
    rows["raw_sentiment"] = rows["raw_sentiment"].fillna(0)

    rows["fundamental_score"] = (
        0.30 * _percentile_score(rows["revenue_growth_pct"])
        + 0.30 * _percentile_score(rows["return_on_equity_pct"])
        + 0.25 * _percentile_score(rows["pe_ratio"], higher_is_better=False)
        + 0.15 * _percentile_score(rows["debt_to_equity"], higher_is_better=False)
    )
    trend = (rows["close"] > rows["sma_50"]).astype(float)
    short_trend = (rows["sma_20"] > rows["sma_50"]).astype(float)
    rsi_momentum = ((rows["rsi_14"].fillna(50) - 50) / 50).clip(-1, 1)
    macd_direction = np.sign(rows["macd"] - rows["macd_signal"])
    rows["technical_score"] = (0.35 * trend + 0.30 * short_trend + 0.20 * ((rsi_momentum + 1) / 2) + 0.15 * ((macd_direction + 1) / 2)).clip(0, 1)
    rows["sentiment_score"] = ((rows["raw_sentiment"] + 1) / 2).clip(0, 1)
    base_score = 0.40 * rows["fundamental_score"] + 0.35 * rows["technical_score"] + 0.25 * rows["sentiment_score"]
    forecast_changes = {}
    for path in (PROJECT_ROOT / "outputs").glob("*_forecast.json"):
        try:
            forecast = json.loads(path.read_text(encoding="utf-8"))
            ticker = str(forecast["ticker"])
            last = float(forecast["last_observed_close"])
            predicted = float(forecast["next_session_forecast"])
            forecast_changes[ticker] = predicted / last - 1 if last else 0.0
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    rows["lstm_forecast_pct"] = rows["ticker"].map(forecast_changes)
    rows["lstm_score"] = ((rows["lstm_forecast_pct"].fillna(0) + 0.05) / 0.10).clip(0, 1)
    has_forecast = rows["lstm_forecast_pct"].notna()
    rows["score"] = base_score
    rows.loc[has_forecast, "score"] = 0.90 * base_score[has_forecast] + 0.10 * rows.loc[has_forecast, "lstm_score"]
    rows["recommendation"] = np.select([rows["score"] >= 0.60, rows["score"] <= 0.40], ["BUY", "SELL"], default="HOLD")
    # Use each symbol's actual most-recent 21-session return even when indexes differ.
    momentum = []
    for ticker in rows["ticker"]:
        series = indicators.loc[indicators["ticker"] == ticker].sort_values("date")["close"]
        momentum.append(float(series.iloc[-1] / series.iloc[-22] - 1) if len(series) >= 22 else 0.0)
    rows["momentum_1m_pct"] = momentum
    return rows.sort_values("score", ascending=False).reset_index(drop=True), sentiment_method


RISK_AVERSION = {"Conservative": 7.5, "Balanced": 3.0, "Growth": 0.9}


def optimize_portfolio(
    prices: pd.DataFrame,
    risk_appetite: str = "Balanced",
    tickers: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Long-only mean-variance allocation with a 40% position cap."""
    selected = list(tickers) if tickers else sorted(prices["ticker"].unique().tolist())
    if not selected:
        raise ValueError("Choose at least one stock to optimize.")
    close = prices[prices["ticker"].isin(selected)].pivot(index="date", columns="ticker", values="close").sort_index()
    returns = close.pct_change().dropna(how="all").fillna(0)
    if len(returns) < 2:
        raise ValueError("Not enough price history to estimate portfolio risk.")
    annual_returns = returns.mean() * 252
    covariance = returns.cov() * 252
    n = len(close.columns)
    risk_lambda = RISK_AVERSION.get(risk_appetite, RISK_AVERSION["Balanced"])

    def objective(weights: np.ndarray) -> float:
        expected = float(weights @ annual_returns.to_numpy())
        variance = float(weights @ covariance.to_numpy() @ weights)
        return -(expected - 0.5 * risk_lambda * variance)

    initial = np.repeat(1 / n, n)
    result = minimize(
        objective, initial, method="SLSQP", bounds=[(0.0, 0.40)] * n,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    weights = result.x if result.success else initial
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    allocation = pd.DataFrame({"ticker": close.columns, "weight": weights}).sort_values("weight", ascending=False).reset_index(drop=True)
    expected_return = float(weights @ annual_returns.to_numpy())
    volatility = float(np.sqrt(max(0, weights @ covariance.to_numpy() @ weights)))
    stats = {"expected_return": expected_return, "volatility": volatility, "sharpe": expected_return / volatility if volatility else 0.0}
    return allocation, stats


def monthly_attribution(prices: pd.DataFrame, positions: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Explain one-month paper-portfolio P&L by holding, at constant share counts."""
    end_date = pd.to_datetime(prices["date"].max())
    start_cutoff = end_date - pd.Timedelta(days=31)
    rows = []
    total_start = 0.0
    total_end = 0.0
    for ticker, shares in positions.items():
        if shares <= 0:
            continue
        history = prices.loc[(prices["ticker"] == ticker) & (prices["date"] >= start_cutoff)].sort_values("date")
        if history.empty:
            continue
        start_price = float(history.iloc[0]["close"])
        end_price = float(history.iloc[-1]["close"])
        pnl = (end_price - start_price) * shares
        total_start += start_price * shares
        total_end += end_price * shares
        rows.append({"Ticker": ticker, "Shares": shares, "Start price": start_price, "End price": end_price, "P&L": pnl, "Return": end_price / start_price - 1})
    attribution = pd.DataFrame(rows)
    total_return = total_end / total_start - 1 if total_start else 0.0
    if not attribution.empty:
        attribution["Contribution"] = attribution["P&L"] / total_start if total_start else 0.0
        attribution = attribution.sort_values("P&L", ascending=False).reset_index(drop=True)
    summary = {"start_value": total_start, "end_value": total_end, "return": total_return, "pnl": total_end - total_start}
    return attribution, summary


def model_monthly_attribution(
    prices: pd.DataFrame,
    allocation: pd.DataFrame,
    starting_value: float = 100_000.0,
) -> tuple[pd.DataFrame, dict[str, float], tuple[pd.Timestamp, pd.Timestamp]]:
    """Backtest a fixed allocation over the latest calendar month in the sample."""
    end_date = pd.to_datetime(prices["date"].max())
    cutoff = end_date - pd.Timedelta(days=31)
    rows = []
    for item in allocation.itertuples(index=False):
        history = prices.loc[(prices["ticker"] == item.ticker) & (prices["date"] >= cutoff)].sort_values("date")
        if history.empty:
            continue
        start_price = float(history.iloc[0]["close"])
        end_price = float(history.iloc[-1]["close"])
        stock_return = end_price / start_price - 1
        contribution = float(item.weight) * stock_return
        rows.append({
            "Ticker": item.ticker, "Weight": float(item.weight), "Start price": start_price,
            "End price": end_price, "Return": stock_return,
            "Contribution": contribution, "P&L": starting_value * contribution,
        })
    attribution = pd.DataFrame(rows)
    total_return = float(attribution["Contribution"].sum()) if not attribution.empty else 0.0
    if not attribution.empty:
        attribution = attribution.sort_values("P&L", ascending=False).reset_index(drop=True)
    summary = {"start_value": starting_value, "end_value": starting_value * (1 + total_return), "return": total_return, "pnl": starting_value * total_return}
    period_start = pd.to_datetime(prices.loc[prices["date"] >= cutoff, "date"].min())
    return attribution, summary, (period_start, end_date)

