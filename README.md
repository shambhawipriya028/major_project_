# Intelligent Stock Portfolio Advisor

An educational investment research platform that combines company fundamentals, technical indicators, market-text sentiment, portfolio optimization, a paper trading simulator, and monthly performance attribution. The interactive application runs locally with Streamlit.

The intended users are retail investors exploring portfolio ideas, financial advisors preparing research views, and investment clubs comparing strategies. This prototype runs as a single local session; it does not provide separate client accounts or shared club workspaces.

## Project overview

The dashboard brings together several views that are often spread across separate tools:

- **Stock research:** a transparent buy, hold, or sell score using fundamental quality, price trend and momentum, and news/social discussion sentiment.
- **Price analysis:** 20- and 50-session moving averages, RSI, MACD, and recent return calculations.
- **News and social sentiment:** an optional FinBERT transformer path, with a lightweight lexicon fallback so the demo works without downloading a model.
- **Price forecasting:** a per-stock LSTM training script with a chronological holdout and error reporting. Saved forecasts can contribute 10% to the signal score for the ticker that was trained.
- **Portfolio construction:** a long-only Modern Portfolio Theory optimizer using historical mean returns and covariance, with a 40% position cap and conservative, balanced, and growth profiles.
- **Paper trading:** simulated buy/sell orders against the last sample close, with cash and holdings tracked in the browser session.
- **Monthly attribution:** a backtested fixed model allocation that shows each stock's return contribution over the latest month in the bundled sample, including positive and negative contributors.

## Start the dashboard

From this folder in PowerShell, create the Python environment outside the project so it does not inflate the project folder:

```powershell
$envPath = Join-Path $env:LOCALAPPDATA 'venvs\intelligent-stock-portfolio-advisor'
py -m venv $envPath
& (Join-Path $envPath 'Scripts\Activate.ps1')
py -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in a browser. It needs no API key for the included demo. In VS Code, select the interpreter at `$envPath\Scripts\python.exe`. Keep virtual environments outside this folder to keep the project below 25 MB.

### Optional machine-learning dependencies

TensorFlow and Hugging Face Transformers are separated from the default environment because they add large downloads. Python 3.10 or 3.11 is recommended for the optional TensorFlow stack. To enable LSTM training and FinBERT scoring:

```powershell
py -m pip install -r requirements-ml.txt
py train_lstm.py --ticker MSFT --epochs 30
```

Train another ticker by changing `--ticker` to one of `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, `JPM`, `TSLA`, or `KO`. The saved model goes in `models/`; its forecast and holdout errors go in `outputs/`. In **Stock signals**, enable the FinBERT switch to run the `ProsusAI/finbert` transformer on the discussion text. The first use downloads pretrained weights and needs internet access.

You can regenerate the reproducible synthetic CSVs with:

```powershell
py data/generate_demo_data.py
```

## Run layout

```text
app.py                         Streamlit dashboard and paper trading UI
train_lstm.py                  Per-ticker LSTM training and chronological evaluation
src/stock_advisor/engine.py    Indicators, sentiment, signal scoring, MPT, attribution
data/prices.csv                Synthetic daily OHLCV history, 2023–2025
data/fundamentals.csv          Synthetic company fundamental snapshot
data/news_social.csv           Generated sample news and social discussion text
data/generate_demo_data.py     Deterministic dataset generator using Python stdlib
models/                        Saved LSTM models after optional training
outputs/                       Saved LSTM forecast metrics; the UI exports attribution CSVs
report/                        Two-page Word report and PDF companion
```

## Signal design

The default composite signal is:

```text
40% fundamental score + 35% technical score + 25% text sentiment score
```

Fundamentals combine cross-sectional percentile ranks for revenue growth, return on equity, valuation, and debt-to-equity. Technicals combine price position versus the 50-session average, 20/50-session trend, RSI momentum, and MACD direction. Discussion sentiment is mapped to a 0–1 scale. The labels are `BUY` at 0.60 or above, `SELL` at 0.40 or below, and `HOLD` between those thresholds.

After an LSTM forecast is trained for a symbol, its predicted next-session percentage change is mapped to a 0–1 forecast score and blended at 10% with the existing composite for that symbol. Other symbols retain the default three-factor score until they have their own trained model. The forecast is not treated as a price target.

The FinBERT path uses the pretrained `ProsusAI/finbert` model for positive, negative, or neutral financial-text classification. The default fallback counts a small set of positive and negative terms. The fallback is intentionally simple and is not a substitute for contextual NLP.

The optimizer annualizes historical daily sample means and covariance by 252, then maximizes expected return minus a risk-aversion-weighted variance penalty. Allocations are long-only, fully invested, and capped at 40% per stock. This is a teaching implementation of mean-variance optimization, not a robust production allocator.

## Data and limitations

**All included stock prices, company fundamentals, news headlines, and social posts are synthetic and generated for demonstration.** Ticker symbols identify example assets but do not imply that the values describe the real companies. The latest sample date is December 31, 2025. No live quotes, licensed news, brokerage connection, or user account storage is configured.

The sample data is not appropriate for investment decisions. The sample is short, generated, and excludes corporate actions, survivorship effects, transaction fees, taxes, dividends, and market impact. The monthly report is a fixed-weight historical model portfolio, not an actual investor statement. Allocation weights use only data available before the report period, then remain fixed through that period. The paper trading ledger is in-memory and resets when the browser session ends. LSTM and sentiment outputs can be unstable and do not establish predictive value.

For a production system, connect licensed and point-in-time market/fundamental/news feeds, retain source timestamps and provenance, add walk-forward and out-of-sample model validation, handle corporate actions, incorporate realistic execution costs, persist each user's paper ledger securely, and add portfolio suitability and audit controls before presenting recommendations to investors.

## Main dependencies

- Python 3.10 or newer (3.10/3.11 recommended for optional TensorFlow)
- Streamlit, Plotly, pandas, NumPy, and SciPy
- Optional: TensorFlow, PyTorch, and Transformers for the LSTM and FinBERT paths

