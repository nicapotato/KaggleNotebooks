#!/usr/bin/env python3
"""
Shared multi-ticker LSTM: multi-step residuals per ticker + auxiliary horizon mean.

Mirrors ``S&P Keras Timeseries Neural Net - Multi-Output.ipynb`` and the Functional API
pattern from ``keras/tutorials/keras-multi-output-2026-working.ipynb``.

- Input: (lookback, K) — K tickers, aligned dates (inner join when loading CSV).
- Output ``adj_close_path``: (K, H) — next H residuals vs last bar of window (normalized).
- Output ``horizon_mean_residual``: (K,) — mean of those H residuals per ticker.

Usage:
  python snp_panel_lstm_multi_output.py --debug     # synthetic data, fast smoke test
  python snp_panel_lstm_multi_output.py             # real CSV (kagglehub / env / local data/)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import keras
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column / path helpers (same as notebook)
# ---------------------------------------------------------------------------

_CANONICAL_COL = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
    "name": "Name",
    "symbol": "Symbol",
    "ticker": "Symbol",
    "company_name": "Name",
    "company name": "Name",
}


def _canonicalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_CANONICAL_COL.get(c.strip().lower(), c.strip()) for c in out.columns]
    return out


def _ticker_column(df: pd.DataFrame) -> str:
    for c in ("Name", "Symbol"):
        if c in df.columns:
            return c
    raise ValueError(
        "Need a ticker/name column (Name, Symbol, …). " f"Columns: {list(df.columns)}"
    )


def _is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()


def _local_data_roots():
    cwd = Path.cwd()
    return [cwd, cwd / "standard-poors", cwd.parent / "standard-poors"]


def resolve_sandp500_csv() -> Path:
    kaggle_roots = [
        Path("/kaggle/input/sandp500"),
        Path("/kaggle/input/camnugent-sandp500"),
        Path("/kaggle/input/sp500"),
    ]
    for root in kaggle_roots:
        if not root.is_dir():
            continue
        for name in ("all_stocks_5yr.csv", "all_stocks.csv"):
            p = root / name
            if p.is_file():
                return p
        found = list(root.rglob("all_stocks*.csv"))
        if found:
            return found[0]

    try:
        import kagglehub

        root = Path(kagglehub.dataset_download("camnugent/sandp500"))
        for name in ("all_stocks_5yr.csv", "all_stocks.csv"):
            p = root / name
            if p.is_file():
                return p
        found = list(root.rglob("all_stocks*.csv"))
        if found:
            return found[0]
    except Exception:
        pass

    for base in _local_data_roots():
        local = base / "data" / "sandp500"
        if local.is_dir():
            found = list(local.rglob("all_stocks*.csv")) + list(local.rglob("*.csv"))
            if found:
                return found[0]

    raise FileNotFoundError(
        "Could not find S&P 500 CSV. On Kaggle, add dataset camnugent/sandp500. "
        "Locally: pip install kagglehub and authenticate, or copy CSV under standard-poors/data/sandp500/. "
        "Or run with --debug for synthetic data."
    )


def load_ticker_ohlcv(csv_path: Path, ticker: str) -> pd.DataFrame:
    df = _canonicalize_ohlcv_columns(pd.read_csv(csv_path))
    name_col = _ticker_column(df)
    if "Date" not in df.columns:
        raise ValueError(f"Expected a date column; got {list(df.columns)}")

    needle = ticker.strip().upper()
    sub = df[df[name_col].astype(str).str.strip().str.upper() == needle].copy()
    if sub.empty:
        raise ValueError(
            f"No rows for ticker {ticker!r} in column {name_col!r}. "
            f"Sample values: {df[name_col].dropna().astype(str).unique()[:8].tolist()}"
        )

    sub["Date"] = pd.to_datetime(sub["Date"], utc=True, errors="coerce")
    sub = sub.sort_values("Date").reset_index(drop=True)

    if "Adj Close" not in sub.columns:
        if "Close" not in sub.columns:
            raise ValueError(f"Need Adj Close or Close; columns are {list(sub.columns)}")
        sub["Adj Close"] = sub["Close"]
        if not getattr(load_ticker_ohlcv, "_warned_no_adj", False):
            print(
                "Note: CSV has no 'Adj Close' column; using 'Close' as the price series "
                "(camnugent/sandp500 is unadjusted)."
            )
            load_ticker_ohlcv._warned_no_adj = True

    return sub[["Adj Close", "Date"]]


def normalize(data: np.ndarray, train_split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    data_std = np.where(data_std == 0, 1.0, data_std)
    return (data - data_mean) / data_std, data_mean, data_std


def align_panel_prices(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    tickers = list(ohlcv_by_ticker.keys())
    merged = None
    for t in tickers:
        df = ohlcv_by_ticker[t][["Date", "Adj Close"]].copy()
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.rename(columns={"Adj Close": t}).dropna(subset=["Date"])
        part = df.set_index("Date")[[t]]
        merged = part if merged is None else merged.join(part, how="inner")
    merged = merged.sort_index().reset_index()
    dates = merged["Date"]
    panel = merged[tickers]
    return panel, dates


def synthetic_aligned_panel(
    n_rows: int, tickers: list[str], seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Random-walk prices on a business-day calendar (no CSV)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B", tz="UTC")
    inc = rng.standard_normal((n_rows, len(tickers))).astype(np.float64) * 0.4
    levels = 100.0 + np.cumsum(inc, axis=0)
    panel = pd.DataFrame(levels, columns=tickers)
    return panel, pd.Series(dates)


def build_arrays_panel_multi_output(
    panel: pd.DataFrame, lookback: int, horizon: int, train_frac: float
):
    values = panel.values.astype(np.float64)
    n, n_tickers = values.shape
    split = int(n * train_frac)
    if split <= lookback + horizon + 10:
        raise ValueError("Not enough rows for lookback, horizon, and split.")

    norm, mean, std = normalize(values, split)

    def windows(start: int, end: int):
        xs, path, mean_res = [], [], []
        last = end - lookback - horizon
        for i in range(start, last + 1):
            win = norm[i : i + lookback]
            anchor = win[-1, :]
            fut = norm[i + lookback : i + lookback + horizon, :]
            resid = fut - anchor
            xs.append(win)
            path.append(resid.T)
            mean_res.append(np.mean(resid, axis=0))
        return (
            np.asarray(xs, dtype=np.float32),
            np.asarray(path, dtype=np.float32),
            np.asarray(mean_res, dtype=np.float32),
        )

    train_end = split
    x_tr, p_tr, m_tr = windows(0, train_end)
    x_va, p_va, m_va = windows(split, n)
    y_tr = {"adj_close_path": p_tr, "horizon_mean_residual": m_tr}
    y_va = {"adj_close_path": p_va, "horizon_mean_residual": m_va}
    return (x_tr, y_tr, x_va, y_va), {"mean": mean, "std": std}


def make_panel_model(
    lookback: int,
    n_tickers: int,
    horizon: int,
    lstm_units: int,
    lr: float,
    aux_mean_weight: float,
):
    inputs = keras.layers.Input(shape=(lookback, n_tickers), name="window")
    last = inputs[:, -1, :]
    u2 = max(lstm_units // 2, 8)
    dense_hidden = max(lstm_units // 2, 32)
    x = keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.6)(x)
    x = keras.layers.LSTM(u2)(x)
    x = keras.layers.Dropout(0.6)(x)
    x = keras.layers.Concatenate()([x, last])
    x = keras.layers.Dense(dense_hidden, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    flat_h = n_tickers * horizon
    path_flat = keras.layers.Dense(flat_h)(x)
    out_path = keras.layers.Reshape((n_tickers, horizon), name="adj_close_path")(path_flat)
    out_mean = keras.layers.Dense(n_tickers, name="horizon_mean_residual")(x)
    model = keras.Model(
        inputs=inputs,
        outputs={"adj_close_path": out_path, "horizon_mean_residual": out_mean},
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            "adj_close_path": "mse",
            "horizon_mean_residual": "mse",
        },
        loss_weights={
            "adj_close_path": 1.0,
            "horizon_mean_residual": aux_mean_weight,
        },
        metrics={
            "adj_close_path": ["mae"],
            "horizon_mean_residual": ["mae"],
        },
    )
    return model


def parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-ticker multi-step LSTM (panel).")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Synthetic panel, small hyperparameters, few epochs (no CSV).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plots (default in debug).",
    )
    args = parser.parse_args()
    debug = args.debug
    do_plot = not args.no_plot and not debug

    if debug:
        tickers = ["AAA", "BBB", "CCC"]
        lookback = 16
        horizon = 5
        train_frac = 0.75
        lstm_units = 32
        lr = 1e-3
        batch_size = 32
        epochs = 3
        aux_w = 0.35
        n_rows = 220
        panel_prices, panel_dates = synthetic_aligned_panel(n_rows, tickers, seed=42)
        print(f"[debug] Synthetic panel: {len(panel_prices)} rows × {len(tickers)} tickers")
    else:
        raw = os.environ.get("SNP_TICKERS", "AAPL,MSFT,GOOG,AMZN,JPM")
        tickers = parse_tickers(raw)
        lookback = int(os.environ.get("SNP_LOOKBACK", "60"))
        horizon = int(os.environ.get("SNP_HORIZON", "20"))
        train_frac = float(os.environ.get("SNP_TRAIN_FRAC", "0.8"))
        lstm_units = int(os.environ.get("SNP_LSTM_UNITS", "128"))
        lr = float(os.environ.get("SNP_LR", "1e-3"))
        batch_size = int(os.environ.get("SNP_BATCH", "64"))
        epochs = int(os.environ.get("SNP_EPOCHS", "40"))
        aux_w = float(os.environ.get("SNP_AUX_MEAN_WEIGHT", "0.35"))

        csv_path = resolve_sandp500_csv()
        print("Using CSV:", csv_path)
        ohlcv_by_ticker = {t: load_ticker_ohlcv(csv_path, t) for t in tickers}
        panel_prices, panel_dates = align_panel_prices(ohlcv_by_ticker)
        print(
            f"Aligned panel: {len(panel_prices)} rows × {len(tickers)} tickers "
            "(inner join on Date)"
        )

    (x_train, y_train, x_val, y_val), scaler = build_arrays_panel_multi_output(
        panel_prices, lookback, horizon, train_frac
    )
    k = len(tickers)
    print(
        f"x_train {x_train.shape}, adj_close_path {y_train['adj_close_path'].shape}, "
        f"horizon_mean_residual {y_train['horizon_mean_residual'].shape}"
    )

    model = make_panel_model(lookback, k, horizon, lstm_units, lr, aux_w)
    model.summary()

    patience = 5 if debug else 40
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )
    print(f"Finished after {len(history.history['loss'])} epochs (max {epochs})")

    # Quick numeric check: val prediction shapes
    pred_pack = model.predict(x_val[: min(8, len(x_val))], verbose=0)
    assert pred_pack["adj_close_path"].shape[-2:] == (k, horizon)
    assert pred_pack["horizon_mean_residual"].shape[-1] == k
    print("Predict shapes OK:", pred_pack["adj_close_path"].shape, pred_pack["horizon_mean_residual"].shape)

    if do_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_tl, ax_tr, ax_bl, ax_br = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        h = history.history
        ax_tl.plot(h["loss"], label="train total")
        ax_tr.plot(h["val_loss"], label="val total")
        pk = next((k for k in h if k.endswith("adj_close_path_loss") and not k.startswith("val_")), None)
        vk = next((k for k in h if k.endswith("adj_close_path_loss") and k.startswith("val_")), None)
        if pk:
            ax_bl.plot(h[pk], label="train path")
        if vk:
            ax_br.plot(h[vk], label="val path")
        ax_tl.set_title("Train loss (total)")
        ax_tr.set_title("Val loss (total)")
        ax_bl.set_title("Train path-head loss")
        ax_br.set_title("Val path-head loss")
        for ax in axes.flat:
            ax.legend()
            ax.set_xlabel("epoch")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Kaggle kernel:", _is_kaggle())
    main()
