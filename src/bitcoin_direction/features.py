from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG


RENAME_MAP = {
    "Open time": "open_time",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Close time": "close_time",
    "Quote asset volume": "quote_asset_volume",
    "Number of trades": "number_of_trades",
    "Taker buy base asset volume": "taker_buy_base_volume",
    "Taker buy quote asset volume": "taker_buy_quote_volume",
    "Ignore": "ignore",
}


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    from .config import DATA_PATH

    csv_path = DATA_PATH if path is None else path
    df = pd.read_csv(csv_path).rename(columns=RENAME_MAP)

    required = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["open_time"], utc=True, errors="raise")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        bad = df[numeric_cols].isna().sum()
        raise ValueError(f"Numeric conversion introduced missing values:\n{bad[bad > 0]}")

    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def create_feature_table(
    raw: pd.DataFrame,
    *,
    drop_unlabeled: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create leakage-safe features and 1-, 5-, and 7-day direction targets."""
    df = raw.copy()

    # Price changes and momentum.
    for window in (1, 2, 3, 5, 7, 14, 30):
        df[f"return_{window}d"] = df["close"].pct_change(window)
        df[f"log_return_{window}d"] = np.log(df["close"] / df["close"].shift(window))

    # Candle geometry.
    safe_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["open_close_return"] = df["close"] / df["open"] - 1
    df["high_low_range"] = df["high"] / df["low"] - 1
    df["body_to_range"] = (df["close"] - df["open"]) / safe_range
    df["upper_wick_to_range"] = (df["high"] - df[["open", "close"]].max(axis=1)) / safe_range
    df["lower_wick_to_range"] = (df[["open", "close"]].min(axis=1) - df["low"]) / safe_range
    df["close_location"] = (df["close"] - df["low"]) / safe_range

    # Trend features.
    for window in (5, 10, 20, 50, 100, 200):
        sma = df["close"].rolling(window).mean()
        df[f"close_to_sma_{window}"] = df["close"] / sma - 1
        if window <= 50:
            df[f"sma_{window}_slope_5d"] = sma.pct_change(5)

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["close_to_ema_12"] = df["close"] / ema_12 - 1
    df["close_to_ema_26"] = df["close"] / ema_26 - 1
    df["macd_normalized"] = macd / df["close"]
    df["macd_signal_normalized"] = macd_signal / df["close"]
    df["macd_hist_normalized"] = (macd - macd_signal) / df["close"]

    # Volatility and bands.
    for window in (5, 10, 20, 30):
        df[f"volatility_{window}d"] = df["log_return_1d"].rolling(window).std()

    rolling_mean_20 = df["close"].rolling(20).mean()
    rolling_std_20 = df["close"].rolling(20).std()
    upper_band = rolling_mean_20 + 2 * rolling_std_20
    lower_band = rolling_mean_20 - 2 * rolling_std_20
    band_width = (upper_band - lower_band).replace(0, np.nan)
    df["bollinger_width"] = band_width / rolling_mean_20
    df["bollinger_position"] = (df["close"] - lower_band) / band_width
    df["rsi_14"] = _rsi(df["close"], 14) / 100.0
    df["atr_14_normalized"] = _atr(df, 14) / df["close"]

    # Liquidity and order-flow proxies.
    df["log_volume"] = np.log1p(df["volume"])
    df["log_quote_volume"] = np.log1p(df["quote_asset_volume"])
    df["log_number_of_trades"] = np.log1p(df["number_of_trades"])
    for window in (1, 5, 7, 30):
        df[f"volume_change_{window}d"] = df["volume"].pct_change(window)
        df[f"trades_change_{window}d"] = df["number_of_trades"].pct_change(window)

    for window in (7, 30):
        volume_average = df["volume"].rolling(window).mean()
        trades_average = df["number_of_trades"].rolling(window).mean()
        df[f"volume_to_sma_{window}"] = df["volume"] / volume_average - 1
        df[f"trades_to_sma_{window}"] = df["number_of_trades"] / trades_average - 1

    df["taker_buy_base_ratio"] = df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)
    df["taker_buy_quote_ratio"] = df["taker_buy_quote_volume"] / df["quote_asset_volume"].replace(0, np.nan)
    df["average_trade_size_btc"] = df["volume"] / df["number_of_trades"].replace(0, np.nan)
    df["average_trade_size_usdt"] = df["quote_asset_volume"] / df["number_of_trades"].replace(0, np.nan)

    # Calendar seasonality, represented cyclically.
    day_of_week = df["date"].dt.dayofweek
    month = df["date"].dt.month
    df["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Future returns and binary direction targets.
    for horizon in CONFIG.horizons:
        future_return = df["close"].shift(-horizon) / df["close"] - 1
        df[f"future_return_{horizon}d"] = future_return
        df[f"target_{horizon}d"] = np.where(
            future_return.notna(),
            (future_return > 0).astype(int),
            np.nan,
        )

    excluded = {
        "date",
        "open_time",
        "close_time",
        "ignore",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    }
    excluded.update({f"future_return_{h}d" for h in CONFIG.horizons})
    excluded.update({f"target_{h}d" for h in CONFIG.horizons})

    feature_columns = [column for column in df.columns if column not in excluded]

    df = df.replace([np.inf, -np.inf], np.nan)
    required_columns = list(feature_columns)
    if drop_unlabeled:
        required_columns += [f"target_{h}d" for h in CONFIG.horizons]
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    if drop_unlabeled:
        for horizon in CONFIG.horizons:
            df[f"target_{horizon}d"] = df[f"target_{horizon}d"].astype(int)

    return df, feature_columns


def repeating_week_split_assignments(
    dates: pd.Series,
    *,
    anchor_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Assign rows to the user's repeating 10-week train/validation/test cycle.

    Calendar weeks run Monday through Sunday. The cycle repeats as follows:
    training = weeks 1, 2, 4, 7, 9, 10; validation = weeks 3, 6;
    test = weeks 5, 8. A row is assigned according to its feature/target date.
    """
    date_series = pd.Series(pd.to_datetime(dates, utc=True), copy=False).reset_index(drop=True)
    if len(date_series) < 100:
        raise ValueError("At least 100 rows are required for the repeating week split.")

    if anchor_date is None:
        anchor = date_series.min()
    else:
        anchor = pd.to_datetime(anchor_date, utc=True)

    # Anchor to the Monday of the first calendar week.
    week_anchor = anchor.normalize() - pd.Timedelta(days=int(anchor.dayofweek))
    elapsed_days = (date_series.dt.normalize() - week_anchor).dt.days
    absolute_week = (elapsed_days // 7 + 1).astype(int)
    cycle_week = ((absolute_week - 1) % CONFIG.split_cycle_length_weeks + 1).astype(int)

    split = np.select(
        [
            cycle_week.isin(CONFIG.train_cycle_weeks),
            cycle_week.isin(CONFIG.validation_cycle_weeks),
            cycle_week.isin(CONFIG.test_cycle_weeks),
        ],
        ["train", "validation", "test"],
        default="unassigned",
    )
    if np.any(split == "unassigned"):
        raise RuntimeError("The repeating week configuration left rows unassigned.")

    return pd.DataFrame(
        {
            "absolute_week": absolute_week.to_numpy(),
            "cycle_week": cycle_week.to_numpy(),
            "split": split,
        }
    )


def repeating_week_split_indices(
    dates: pd.Series,
    *,
    anchor_date: pd.Timestamp | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    assignments = repeating_week_split_assignments(dates, anchor_date=anchor_date)
    train_idx = np.flatnonzero(assignments["split"].to_numpy() == "train")
    validation_idx = np.flatnonzero(assignments["split"].to_numpy() == "validation")
    test_idx = np.flatnonzero(assignments["split"].to_numpy() == "test")

    if min(len(train_idx), len(validation_idx), len(test_idx)) == 0:
        raise RuntimeError("The repeating week split produced an empty partition.")

    return train_idx, validation_idx, test_idx, assignments
