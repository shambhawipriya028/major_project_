"""Interactive Streamlit demo for the Intelligent Stock Portfolio Advisor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stock_advisor.engine import (  # noqa: E402
    build_recommendations,
    load_demo_data,
    model_monthly_attribution,
    optimize_portfolio,
    technical_indicators,
)


st.set_page_config(page_title="Northstar | Stock Portfolio Advisor", page_icon="◒", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .block-container { max-width: 1440px; padding-top: 1.7rem; padding-bottom: 3rem; }
      .eyebrow { color: #6f7d8d; font-size: .73rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; }
      .hero { background: linear-gradient(110deg,#101e2b,#172c3a); border: 1px solid #263b48; border-radius: 16px; padding: 27px 30px; color: #f3f5f4; margin-bottom: 19px; }
      .hero h1 { margin: .25rem 0 .45rem; font-size: 2rem; letter-spacing: -.04em; }
      .hero p { color: #b6c3c8; margin: 0; max-width: 720px; }
      .tag { color: #c2f0df; border: 1px solid #386353; border-radius: 999px; padding: 5px 10px; font-size: .73rem; white-space: nowrap; }
      .signal-card { border: 1px solid #e3e9e8; border-radius: 13px; padding: 15px 16px; background: #fff; min-height: 125px; }
      .signal-card .symbol { font-family: 'DM Mono', monospace; font-size: 1.2rem; color: #152a36; font-weight: 500; }
      .signal-card .name { color: #74808a; font-size: .8rem; margin: 2px 0 12px; }
      .buy { color: #087958; background: #e7f6ef; }
      .hold { color: #8c6416; background: #fff4d8; }
      .sell { color: #a23b36; background: #fcebea; }
      .pill { display: inline-block; border-radius: 99px; padding: 4px 9px; font-size: .68rem; letter-spacing: .07em; font-weight: 700; }
      .mono { font-family: 'DM Mono', monospace; }
      .note { border-left: 3px solid #d9a743; padding: 9px 13px; background: #fffaf0; color: #63563c; font-size: .86rem; }
      div[data-testid="stMetric"] { background: #f5f8f7; border: 1px solid #e3eae8; padding: 13px 16px; border-radius: 12px; }
      div[data-testid="stMetricLabel"] { color: #687681; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    return load_demo_data()


prices, fundamentals, discussions = get_data()
symbols = sorted(prices["ticker"].unique())
last_date = pd.to_datetime(prices["date"].max())
latest_prices = prices.sort_values("date").groupby("ticker").tail(1).set_index("ticker")["close"]

if "paper_cash" not in st.session_state:
    st.session_state.paper_cash = 100_000.0
    st.session_state.paper_positions = {}
    st.session_state.paper_trades = []

st.sidebar.markdown("<div class='eyebrow'>NORTHSTAR / INVESTING LAB</div>", unsafe_allow_html=True)
st.sidebar.title("Portfolio Advisor")
page = st.sidebar.radio("Workspace", ["Overview", "Stock signals", "Portfolio optimizer", "Paper trading", "Monthly report", "LSTM forecast"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**Profile**")
risk_profile = st.sidebar.select_slider("Risk appetite", options=["Conservative", "Balanced", "Growth"], value="Balanced")
st.sidebar.caption("Research dashboard · simulated market dataset")


def page_header(kicker: str, title: str, subtitle: str):
    st.markdown(f"<div class='eyebrow'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def currency(value: float) -> str:
    return f"${value:,.2f}"


def recs(use_finbert: bool = False):
    return build_recommendations(prices, fundamentals, discussions, use_finbert=use_finbert)


def signal_card(row):
    action = str(row.recommendation).lower()
    return f"""
      <div class="signal-card">
        <div class="symbol">{row.ticker}</div><div class="name">{row.company}</div>
        <span class="pill {action}">{row.recommendation}</span>
        <span style="float:right;color:#6d7880;font-size:.78rem">Signal {row.score:.0%}</span>
      </div>
    """


if page == "Overview":
    st.markdown(
        "<div class='hero'><span class='tag'>RESEARCH DEMO · SYNTHETIC DATA</span><h1>Invest with a clearer view.</h1><p>One workspace for company fundamentals, price trends, market conversation, portfolio construction, and paper trading.</p></div>",
        unsafe_allow_html=True,
    )
    recommendations, sentiment_method = recs()
    equal_weight_daily = prices.pivot(index="date", columns="ticker", values="close").pct_change().tail(21).mean(axis=1).dropna()
    equal_weight_return = (1 + equal_weight_daily).prod() - 1
    buy_count = int((recommendations["recommendation"] == "BUY").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks monitored", f"{len(symbols)}", "Across 6 sectors")
    m2.metric("Buy signals", f"{buy_count}", "Fundamental + technical + text")
    m3.metric("Equal-weight · 1 month", f"{equal_weight_return:+.2%}", "Historical sample return")
    m4.metric("Sample as of", last_date.strftime("%d %b %Y"), "Synthetic price history")
    st.markdown("### Signals in focus")
    cards = st.columns(4)
    for column, row in zip(cards, recommendations.head(4).itertuples(index=False)):
        column.markdown(signal_card(row), unsafe_allow_html=True)
    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("### Price and trend")
        ticker = st.selectbox("Company", symbols, index=symbols.index("MSFT"), key="overview_ticker")
        indicators = technical_indicators(prices)
        history = indicators[indicators["ticker"] == ticker].tail(180)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["date"], y=history["close"], name="Close", line={"color": "#16846b", "width": 2.5}))
        fig.add_trace(go.Scatter(x=history["date"], y=history["sma_20"], name="20 session", line={"color": "#e2a546", "width": 1.5}))
        fig.add_trace(go.Scatter(x=history["date"], y=history["sma_50"], name="50 session", line={"color": "#72849b", "width": 1.5}))
        fig.update_layout(height=340, margin={"l": 0, "r": 10, "t": 10, "b": 0}, legend={"orientation": "h", "y": 1.08}, yaxis_title="USD", xaxis_title=None, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Latest discussion")
        recent = discussions.sort_values("published_at", ascending=False).head(5)
        for item in recent.itertuples(index=False):
            st.markdown(f"**{item.ticker}** · {item.source.title()}  ")
            st.caption(item.text)
            st.markdown("---")
    st.markdown(f"<div class='note'>All quotes, fundamentals, headlines, and social posts are generated demo content. No live market feed is connected. Text scoring currently uses {sentiment_method}; recommendations are educational examples.</div>", unsafe_allow_html=True)

elif page == "Stock signals":
    page_header("RESEARCH / EQUITIES", "Stock signals", "See the factors behind each buy, hold, or sell label.")
    use_finbert = st.toggle("Score discussions with FinBERT transformer", value=False, help="Loads ProsusAI/finbert on first use. Requires the optional ML packages and internet access to download model weights.")
    try:
        recommendations, sentiment_method = recs(use_finbert)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    trained_count = int(recommendations["lstm_forecast_pct"].notna().sum())
    signal_weights = "36% fundamentals, 31.5% technicals, 22.5% sentiment, 10% LSTM forecast for trained tickers" if trained_count else "40% fundamentals, 35% technicals, 25% sentiment; train an LSTM to add a forecast factor"
    st.caption(f"Text scoring: **{sentiment_method}** · Signal weights: {signal_weights}.")
    table = recommendations[["ticker", "company", "sector", "recommendation", "score", "fundamental_score", "technical_score", "raw_sentiment", "lstm_forecast_pct", "momentum_1m_pct", "close", "rsi_14", "pe_ratio", "revenue_growth_pct"]].copy()
    table = table.rename(columns={"ticker": "Ticker", "company": "Company", "sector": "Sector", "recommendation": "Action", "score": "Composite", "fundamental_score": "Fundamentals", "technical_score": "Technical", "raw_sentiment": "Text sentiment", "lstm_forecast_pct": "LSTM forecast", "momentum_1m_pct": "1M return", "close": "Last close", "rsi_14": "RSI 14", "pe_ratio": "P/E", "revenue_growth_pct": "Revenue growth %"})
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        "Composite": st.column_config.ProgressColumn("Composite", min_value=0, max_value=1, format="%.0f%%"),
        "Fundamentals": st.column_config.ProgressColumn("Fundamentals", min_value=0, max_value=1, format="%.0f%%"),
        "Technical": st.column_config.ProgressColumn("Technical", min_value=0, max_value=1, format="%.0f%%"),
        "Text sentiment": st.column_config.NumberColumn("Text sentiment", format="%.2f"),
        "LSTM forecast": st.column_config.NumberColumn("LSTM forecast", format="%+.1f%%"),
        "1M return": st.column_config.NumberColumn("1M return", format="%+.1f%%"),
        "Last close": st.column_config.NumberColumn("Last close", format="$%.2f"),
    })
    st.markdown("### Company profile and discussion")
    chosen = st.selectbox("Select a stock", symbols, key="signal_ticker")
    company = recommendations[recommendations["ticker"] == chosen].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market price", currency(company.close))
    c2.metric("RSI (14)", f"{company.rsi_14:.1f}" if pd.notna(company.rsi_14) else "—")
    c3.metric("P/E ratio", f"{company.pe_ratio:.1f}×")
    c4.metric("ROE", f"{company.return_on_equity_pct:.1f}%")
    rows = discussions[discussions["ticker"] == chosen].sort_values("published_at", ascending=False)
    for item in rows.head(5).itertuples(index=False):
        st.markdown(f"**{item.source.title()}** · {pd.to_datetime(item.published_at).strftime('%d %b %Y')} · {item.ticker}")
        st.write(item.text)
    st.markdown("<div class='note'>The sample corpus is synthetic and intentionally small. Social posts are unverified text and should not be treated as evidence or a trading trigger.</div>", unsafe_allow_html=True)

elif page == "Portfolio optimizer":
    page_header("PORTFOLIO / CONSTRUCTION", "Build a diversified allocation", "Modern Portfolio Theory balances expected return against historical covariance.")
    selected = st.multiselect("Universe", symbols, default=symbols)
    if len(selected) < 3:
        st.info("Select at least three stocks so the optimizer can honor the 40% single-stock cap.")
        st.stop()
    selected_risk = st.radio("Risk profile", ["Conservative", "Balanced", "Growth"], index=["Conservative", "Balanced", "Growth"].index(risk_profile), horizontal=True)
    allocation, stats = optimize_portfolio(prices, selected_risk, selected)
    a, b, c = st.columns(3)
    a.metric("Estimated annual return", f"{stats['expected_return']:.1%}", "Annualized historical mean")
    b.metric("Estimated volatility", f"{stats['volatility']:.1%}", "Annualized sample covariance")
    c.metric("Historical Sharpe", f"{stats['sharpe']:.2f}", "Assumes zero risk-free rate")
    left, right = st.columns([1.2, 1])
    with left:
        fig = px.pie(allocation, names="ticker", values="weight", hole=0.64, color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(textposition="inside", textinfo="label+percent", hole=0.64)
        fig.update_layout(height=380, margin={"l": 0, "r": 0, "t": 10, "b": 10}, showlegend=False, annotations=[{"text": selected_risk, "x": .5, "y": .5, "font_size": 18, "showarrow": False}])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        shown = allocation.copy()
        shown["weight"] = shown["weight"].map(lambda value: f"{value:.1%}")
        st.markdown("#### Suggested allocation")
        st.dataframe(shown.rename(columns={"ticker": "Ticker", "weight": "Allocation"}), hide_index=True, use_container_width=True)
        st.markdown("#### How to read this")
        st.write("The optimizer uses historical average returns and the covariance matrix of daily returns. It is long-only, fully invested, and caps each position at 40%.")
        st.write(f"**{selected_risk}** changes the penalty applied to portfolio variance. Estimates are sensitive to the short synthetic sample and are not forward-looking guarantees.")
    st.markdown("<div class='note'>Historical means can be noisy and covariance can change sharply. Review the assumptions and concentration before using any allocation.</div>", unsafe_allow_html=True)

elif page == "Paper trading":
    page_header("SIMULATION / ORDERS", "Paper trading", "Place simulated orders using the last close in the bundled market sample.")
    holdings = st.session_state.paper_positions
    portfolio_value = sum(float(shares) * float(latest_prices[ticker]) for ticker, shares in holdings.items())
    net_liquidation = st.session_state.paper_cash + portfolio_value
    c1, c2, c3 = st.columns(3)
    c1.metric("Paper account value", currency(net_liquidation), f"{net_liquidation - 100_000:+,.2f} vs. start")
    c2.metric("Available cash", currency(st.session_state.paper_cash))
    c3.metric("Invested", currency(portfolio_value))
    st.markdown("### New simulated order")
    with st.form("paper_order"):
        col1, col2, col3 = st.columns([1, 1, 1])
        selected_ticker = col1.selectbox("Ticker", symbols)
        side = col2.selectbox("Side", ["Buy", "Sell"])
        quantity = col3.number_input("Shares", min_value=1, max_value=100_000, value=5, step=1)
        submitted = st.form_submit_button("Place paper order", type="primary")
    if submitted:
        quote = float(latest_prices[selected_ticker])
        shares = float(quantity)
        held = float(holdings.get(selected_ticker, 0))
        if side == "Buy" and shares * quote > st.session_state.paper_cash:
            st.error("Not enough paper cash for that order.")
        elif side == "Sell" and shares > held:
            st.error(f"You hold only {held:g} shares of {selected_ticker}.")
        else:
            if side == "Buy":
                st.session_state.paper_cash -= shares * quote
                holdings[selected_ticker] = held + shares
            else:
                st.session_state.paper_cash += shares * quote
                holdings[selected_ticker] = held - shares
                if holdings[selected_ticker] == 0:
                    del holdings[selected_ticker]
            st.session_state.paper_trades.insert(0, {"Date": last_date.strftime("%Y-%m-%d"), "Ticker": selected_ticker, "Side": side, "Shares": shares, "Fill price": quote, "Notional": shares * quote})
            st.rerun()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Holdings")
        if holdings:
            holding_rows = [{"Ticker": ticker, "Shares": shares, "Last close": float(latest_prices[ticker]), "Market value": float(shares) * float(latest_prices[ticker])} for ticker, shares in holdings.items()]
            st.dataframe(pd.DataFrame(holding_rows), hide_index=True, use_container_width=True, column_config={"Last close": st.column_config.NumberColumn(format="$%.2f"), "Market value": st.column_config.NumberColumn(format="$%.2f")})
        else:
            st.info("Your account starts in cash. Place a simulated buy order to create a position.")
    with right:
        st.markdown("#### Recent orders")
        if st.session_state.paper_trades:
            st.dataframe(pd.DataFrame(st.session_state.paper_trades), hide_index=True, use_container_width=True)
        else:
            st.caption("No simulated orders yet.")
    st.markdown("<div class='note'>Orders execute immediately at the sample's last close with no fees or slippage. The account resets when this browser session ends. This simulator is disconnected from brokerage accounts.</div>", unsafe_allow_html=True)

elif page == "Monthly report":
    page_header("REPORTING / ATTRIBUTION", "Monthly performance report", "Review a fixed, optimized model portfolio over the latest one-month sample period.")
    report_risk = st.selectbox("Model portfolio risk profile", ["Conservative", "Balanced", "Growth"], index=["Conservative", "Balanced", "Growth"].index(risk_profile))
    selected = st.multiselect("Stocks included", symbols, default=symbols, key="report_universe")
    if len(selected) < 3:
        st.info("Choose at least three stocks to keep the report consistent with the 40% allocation cap.")
        st.stop()
    period_cutoff = pd.to_datetime(prices["date"].max()) - pd.Timedelta(days=31)
    training_prices = prices.loc[prices["date"] < period_cutoff]
    allocation, _ = optimize_portfolio(training_prices, report_risk, selected)
    attribution, summary, period = model_monthly_attribution(prices, allocation)
    st.caption(f"Period: **{period[0].strftime('%d %b %Y')} – {period[1].strftime('%d %b %Y')}** · Starting value: **$100,000** · Weights estimated before period start and held fixed.")
    a, b, c = st.columns(3)
    a.metric("Model portfolio return", f"{summary['return']:+.2%}")
    b.metric("Portfolio P&L", f"{currency(summary['pnl'])}")
    c.metric("Ending value", currency(summary["end_value"]))
    st.markdown("### Return attribution")
    st.write("Contribution = starting portfolio weight × stock return during the report period. Positive rows added return; negative rows reduced it.")
    display = attribution.copy()
    display["Weight"] = display["Weight"].map(lambda value: f"{value:.1%}")
    display["Return"] = display["Return"].map(lambda value: f"{value:+.2%}")
    display["Contribution"] = display["Contribution"].map(lambda value: f"{value:+.2%}")
    display["P&L"] = display["P&L"].map(currency)
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.download_button("Download attribution CSV", attribution.to_csv(index=False).encode("utf-8"), file_name=f"monthly_attribution_{period[1].date()}.csv", mime="text/csv")
    st.markdown("<div class='note'>This is a backtest on synthetic prices with a fixed allocation for one month. It is not a realized client account statement and excludes fees, taxes, dividends, and trading costs.</div>", unsafe_allow_html=True)

elif page == "LSTM forecast":
    page_header("MODELS / PRICE FORECAST", "LSTM price forecast", "Train a separate recurrent neural network for each ticker and review a held-out error summary.")
    trained = sorted((ROOT / "outputs").glob("*_forecast.json"))
    if not trained:
        st.info("No trained LSTM model is saved yet. The bundled price data is ready for a local training run.")
        st.code("py -m pip install -r requirements-ml.txt\npy train_lstm.py --ticker MSFT --epochs 30", language="powershell")
        st.caption("Training uses the first 80% of the time-ordered price series and reports error on the final 20%. TensorFlow and model weights are optional downloads.")
    else:
        ticker = st.selectbox("Trained ticker", [path.stem.replace("_forecast", "") for path in trained])
        forecast = json.loads((ROOT / "outputs" / f"{ticker}_forecast.json").read_text(encoding="utf-8"))
        last_observed = forecast["last_observed_close"]
        predicted = forecast["next_session_forecast"]
        a, b, c = st.columns(3)
        a.metric("Last observed close", currency(last_observed))
        b.metric("Next-session model output", currency(predicted), f"{predicted / last_observed - 1:+.2%}")
        c.metric("Holdout MAE", currency(forecast["test_mae"]))
        st.json(forecast)
        st.markdown("<div class='note'>An LSTM output is a model estimate, not a target price. Synthetic training data does not demonstrate predictive value on live markets.</div>", unsafe_allow_html=True)

