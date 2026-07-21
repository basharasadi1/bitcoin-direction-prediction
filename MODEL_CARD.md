# Model Card — Bitcoin Direction Prediction

## Intended use

This is an academic and portfolio project demonstrating a complete time-series machine-learning workflow. It predicts whether the BTC/USDT close will be higher after 1, 5 or 7 days.

It is **not** intended for live trading, investment advice or automated financial decisions.

## Targets

- Next day: `close[t+1] > close[t]`
- Next 5 days: `close[t+5] > close[t]`
- Next week: `close[t+7] > close[t]`

## Models

- Majority-class baseline
- Logistic regression
- Random forest
- XGBoost
- PyTorch feedforward neural network
- PyTorch LSTM using a 30-day feature sequence

## Data and features

The source data contains Binance daily OHLCV, quote volume, number of trades and taker-buy volumes. The model uses 66 engineered features including returns, moving-average ratios, RSI, MACD, Bollinger bands, ATR, volatility, candle geometry, volume/trade changes, taker-buy ratios and cyclical calendar variables.

No future value is used as a feature.

## Evaluation design

- Repeating 10-week interleaved split over the full timeline
- Training weeks: 1, 2, 4, 7, 9, 10
- Validation weeks: 3, 6
- Test weeks: 5, 8
- Monday–Sunday calendar weeks; approximately 60% / 20% / 20%
- Model and threshold selection on validation weeks only
- Separate three-fold expanding-window evaluation for classical models
- Simplified date-based non-overlapping backtest with 0.1% transaction cost per side

This interleaved design is useful for testing whether patterns transfer across held-out weeks and market regimes, but it is not a strict future-only holdout because training weeks can occur later than test weeks.

## Observed limitations

Performance depends strongly on the evaluation design. The interleaved split produces stronger results than the strict expanding-window benchmark, especially for the 7-day LSTM. This is an important result rather than a defect to hide: crypto-market relationships are noisy, non-stationary and sensitive to regime changes.

Additional limitations:

- One exchange and one asset only
- No order-book, news, macroeconomic, sentiment or on-chain data
- Simplified execution and transaction-cost assumptions
- Interleaved weeks create temporal dependence across partitions and overlapping 5- and 7-day label windows
- Hyperparameters are deliberately moderate rather than extensively optimized
