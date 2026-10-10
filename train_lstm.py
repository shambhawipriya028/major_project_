"""Train a per-ticker LSTM close-price model from the bundled CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="MSFT", help="Ticker to train, for example MSFT")
    parser.add_argument("--lookback", type=int, default=60, help="Trading sessions per input sequence")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is optional. Install packages from requirements-ml.txt first.") from exc

    prices = pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])
    frame = prices.loc[prices["ticker"] == args.ticker.upper()].sort_values("date")
    if frame.empty:
        raise SystemExit(f"No price rows for ticker {args.ticker!r}.")
    raw = frame["close"].to_numpy(dtype=np.float32).reshape(-1, 1)
    split = int(len(raw) * 0.8)
    train_min, train_max = float(raw[:split].min()), float(raw[:split].max())
    scaled = (raw - train_min) / (train_max - train_min)

    x, y = [], []
    for end in range(args.lookback, len(scaled)):
        x.append(scaled[end - args.lookback:end, 0])
        y.append(scaled[end, 0])
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    train_end = split - args.lookback
    x_train, y_train = x[:train_end], y[:train_end]
    x_test, y_test = x[train_end:], y[train_end:]

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(args.lookback, 1)),
        tf.keras.layers.LSTM(48, return_sequences=True),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.LSTM(24),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    model.fit(x_train.reshape(-1, args.lookback, 1), y_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size, shuffle=False, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    predicted_scaled = model.predict(x_test.reshape(-1, args.lookback, 1), verbose=0).reshape(-1)
    predicted = predicted_scaled * (train_max - train_min) + train_min
    actual = y_test * (train_max - train_min) + train_min
    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    baseline = raw[split - 1:-1, 0]
    baseline_mae = float(np.mean(np.abs(baseline - actual)))
    baseline_rmse = float(np.sqrt(np.mean((baseline - actual) ** 2)))
    next_scaled = model.predict(scaled[-args.lookback:].reshape(1, args.lookback, 1), verbose=0)[0, 0]
    next_close = float(next_scaled * (train_max - train_min) + train_min)

    model_dir = ROOT / "models"
    output_dir = ROOT / "outputs"
    model_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    model.save(model_dir / f"{args.ticker.upper()}_lstm.keras")
    forecast = {
        "ticker": args.ticker.upper(), "model": "LSTM", "lookback_sessions": args.lookback,
        "training_rows": int(len(x_train)), "test_rows": int(len(x_test)), "test_mae": mae,
        "test_rmse": rmse, "naive_previous_close_mae": baseline_mae,
        "naive_previous_close_rmse": baseline_rmse, "last_observed_close": float(raw[-1, 0]),
        "next_session_forecast": next_close, "data_note": "Synthetic demo price history; forecast is educational only.",
    }
    (output_dir / f"{args.ticker.upper()}_forecast.json").write_text(json.dumps(forecast, indent=2), encoding="utf-8")
    print(json.dumps(forecast, indent=2))


if __name__ == "__main__":
    main()
