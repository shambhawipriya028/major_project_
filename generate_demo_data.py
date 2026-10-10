"""Generate the deterministic synthetic market dataset bundled with this project."""

from __future__ import annotations

from pathlib import Path

import csv
import math
import random
from datetime import date, timedelta


HERE = Path(__file__).resolve().parent
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "TSLA", "KO"]
STARTING_PRICES = {"AAPL": 185, "MSFT": 370, "NVDA": 48, "AMZN": 150, "GOOGL": 140, "JPM": 170, "TSLA": 245, "KO": 59}
DRIFT = {"AAPL": 0.12, "MSFT": 0.14, "NVDA": 0.23, "AMZN": 0.13, "GOOGL": 0.12, "JPM": 0.09, "TSLA": 0.16, "KO": 0.07}
VOLATILITY = {"AAPL": 0.25, "MSFT": 0.23, "NVDA": 0.47, "AMZN": 0.31, "GOOGL": 0.29, "JPM": 0.22, "TSLA": 0.52, "KO": 0.17}
FUNDAMENTALS = [
    {"ticker": "AAPL", "company": "Apple Inc.", "sector": "Technology", "pe_ratio": 29.4, "debt_to_equity": 1.8, "revenue_growth_pct": 7.5, "return_on_equity_pct": 147.0, "dividend_yield_pct": 0.5},
    {"ticker": "MSFT", "company": "Microsoft Corp.", "sector": "Technology", "pe_ratio": 35.2, "debt_to_equity": 0.4, "revenue_growth_pct": 15.7, "return_on_equity_pct": 35.0, "dividend_yield_pct": 0.7},
    {"ticker": "NVDA", "company": "NVIDIA Corp.", "sector": "Semiconductors", "pe_ratio": 41.8, "debt_to_equity": 0.2, "revenue_growth_pct": 74.0, "return_on_equity_pct": 91.0, "dividend_yield_pct": 0.03},
    {"ticker": "AMZN", "company": "Amazon.com Inc.", "sector": "Consumer Discretionary", "pe_ratio": 43.0, "debt_to_equity": 0.6, "revenue_growth_pct": 11.0, "return_on_equity_pct": 22.0, "dividend_yield_pct": 0.0},
    {"ticker": "GOOGL", "company": "Alphabet Inc.", "sector": "Communication Services", "pe_ratio": 24.8, "debt_to_equity": 0.1, "revenue_growth_pct": 12.0, "return_on_equity_pct": 27.0, "dividend_yield_pct": 0.5},
    {"ticker": "JPM", "company": "JPMorgan Chase & Co.", "sector": "Financials", "pe_ratio": 12.5, "debt_to_equity": 1.1, "revenue_growth_pct": 8.0, "return_on_equity_pct": 16.0, "dividend_yield_pct": 2.2},
    {"ticker": "TSLA", "company": "Tesla Inc.", "sector": "Consumer Discretionary", "pe_ratio": 62.0, "debt_to_equity": 0.2, "revenue_growth_pct": 19.0, "return_on_equity_pct": 20.0, "dividend_yield_pct": 0.0},
    {"ticker": "KO", "company": "Coca-Cola Co.", "sector": "Consumer Staples", "pe_ratio": 24.2, "debt_to_equity": 1.6, "revenue_growth_pct": 4.0, "return_on_equity_pct": 40.0, "dividend_yield_pct": 3.0},
]


def main() -> None:
    rng = random.Random(4217)
    dates = []
    current = date(2023, 1, 2)
    while current <= date(2025, 12, 31):
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    market = [rng.gauss(0.00025, 0.009) for _ in dates]
    price_rows = []
    for ticker in TICKERS:
        beta = rng.uniform(0.65, 1.4)
        close = STARTING_PRICES[ticker]
        prior = STARTING_PRICES[ticker]
        for session, market_return in zip(dates, market):
            daily_return = DRIFT[ticker] / 252 + beta * market_return + rng.gauss(0, VOLATILITY[ticker] / math.sqrt(252))
            close *= math.exp(daily_return)
            open_price = prior * (1 + rng.gauss(0, 0.004))
            high = max(open_price, close) * (1 + abs(rng.gauss(0.003, 0.002)))
            low = min(open_price, close) * (1 - abs(rng.gauss(0.003, 0.002)))
            volume = int(rng.lognormvariate(16.0, 0.35))
            price_rows.append({"date": session.isoformat(), "ticker": ticker, "open": round(open_price, 2), "high": round(high, 2), "low": round(low, 2), "close": round(close, 2), "volume": volume})
            prior = close
    write_csv(HERE / "prices.csv", price_rows)
    write_csv(HERE / "fundamentals.csv", FUNDAMENTALS)

    snippets = [
        ("news", "Quarterly results show strong revenue growth and operating margin improvement."),
        ("news", "Analysts flag valuation risk as the share price runs ahead of earnings."),
        ("news", "Product expansion supports a positive long-term demand outlook."),
        ("news", "A weaker forecast adds pressure after the company reported a profit decline."),
        ("news", "Management announced a new investment plan focused on innovation."),
        ("news", "Regulatory concerns and a pending lawsuit weigh on the near-term outlook."),
        ("social", "Bullish discussion points to strong momentum and improving customer demand."),
        ("social", "Investors debate whether recent gains have made the stock too expensive."),
        ("social", "Community members remain concerned about execution risk and margin pressure."),
        ("social", "A product launch is drawing positive attention across investor discussions."),
        ("social", "Several posters expect a downgrade after the latest weak guidance."),
        ("social", "Long-term holders cite consistent growth, cash generation, and a strong balance sheet."),
    ]
    discussions = []
    for ticker_index, ticker in enumerate(TICKERS):
        for item_index, (source, text) in enumerate(snippets):
            published = dates[(ticker_index * 13 + item_index * 47 + 12) % len(dates)]
            discussions.append({"published_at": published.isoformat(), "ticker": ticker, "source": source, "text": text})
    write_csv(HERE / "news_social.csv", sorted(discussions, key=lambda row: row["published_at"]))
    print(f"Wrote {len(price_rows):,} synthetic price rows and {len(discussions)} synthetic discussion rows.")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
