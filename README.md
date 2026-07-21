# Bitcoin Direction Prediction with Machine Learning

A complete time-series machine-learning portfolio project that predicts whether the BTC/USDT closing price will be higher after **1 day, 5 days, or 7 days**.

The project includes data validation, feature engineering, classical models, neural networks, a repeating 10-week interleaved evaluation, strict walk-forward validation, simplified backtesting, saved production models, a CLI predictor, a Jupyter notebook, and a Streamlit dashboard.

> **Important:** This is an academic project, not financial advice. The final holdout results do not demonstrate a stable profitable edge.

## What is included

- 3,089 daily BTC/USDT observations from 2018-01-01 through 2026-06-16
- 66 leakage-safe engineered features
- Majority baseline, logistic regression, random forest and XGBoost
- PyTorch feedforward neural network and 30-day LSTM
- Repeating 10-week split across the full timeline: train weeks 1, 2, 4, 7, 9, 10; validation weeks 3, 6; test weeks 5, 8
- Three-fold expanding-window benchmark for the classical models
- Validation-based probability thresholds
- Non-overlapping backtests with transaction costs
- Production refit using all labeled rows after evaluation
- Streamlit dashboard and reusable prediction CLI

## Targets

| Horizon | Binary target |
|---|---|
| Next day | `close[t+1] > close[t]` |
| Next 5 days | `close[t+5] > close[t]` |
| Next week | `close[t+7] > close[t]` |

## Repeating-week test results

Models were selected **only from validation ROC-AUC**, then evaluated on the assigned test weeks. The split repeats over the whole timeline, so these are interleaved generalization results—not a future-only holdout.

| Horizon | Validation-selected model | Accuracy | Balanced accuracy | ROC-AUC | F1 | Strategy return | Buy-and-hold return |
|---|---|---:|---:|---:|---:|---:|---:|
| Next Day | `logistic_regression` | 0.521 | 0.528 | 0.529 | 0.476 | +36.6% | +54.8% |
| Next 5 Days | `mlp` | 0.544 | 0.529 | 0.529 | 0.627 | +98.6% | +153.2% |
| Next Week | `lstm` | 0.556 | 0.574 | 0.631 | 0.351 | +142.9% | +54.8% |

The 7-day model scores substantially better under this interleaved split. This result must be interpreted cautiously because training, validation and test weeks all occur throughout 2018–2026. The strict expanding-window benchmark below remains the better estimate of future-only forecasting performance.

## Walk-forward benchmark

The default expanding-window benchmark uses three folds to keep execution practical. The full fold-level results are stored in `artifacts/results/walk_forward_folds.csv`.

| Horizon | Best classical model | Mean ROC-AUC | Mean balanced accuracy |
|---|---|---:|---:|
| Next Day | `logistic_regression` | 0.538 ± 0.030 | 0.537 |
| Next 5 Days | `random_forest` | 0.518 ± 0.024 | 0.518 |
| Next Week | `random_forest` | 0.514 ± 0.074 | 0.510 |

## Latest production-model outputs

These are computed from the included data through **2026-06-16** after refitting the validation-selected architecture on all labeled rows.

| Horizon | Model | P(UP) | Threshold | Output |
|---|---|---:|---:|---|
| Next Day | `logistic_regression` | 50.2% | 53.0% | **DOWN** |
| Next 5 Days | `mlp` | 48.1% | 49.0% | **DOWN** |
| Next Week | `lstm` | 50.2% | 57.5% | **DOWN** |

These outputs are demonstrations of inference, not recommendations.

## Project structure

```text
bitcoin_direction_ml/
├── app.py                         # Streamlit dashboard
├── train.py                       # Complete training/evaluation pipeline
├── walk_forward.py                # Expanding-window evaluation
├── predict.py                     # CLI inference on a compatible CSV
├── requirements.txt
├── MODEL_CARD.md
├── SPLIT_DESIGN.md                 # Exact repeating-week allocation and interpretation
├── data/
│   ├── README.md
│   └── raw/btc_usdt_1d_2018_2026.csv
├── notebooks/
│   └── Bitcoin_Direction_Prediction.ipynb
├── src/bitcoin_direction/
│   ├── config.py
│   ├── features.py
│   ├── inference.py
│   ├── metrics.py
│   ├── torch_models.py
│   ├── training.py
│   └── walk_forward.py
├── tests/test_features.py
└── artifacts/
    ├── models/
    ├── results/
    ├── predictions/
    ├── backtests/
    └── figures/
```

## Run on Windows

```powershell
cd bitcoin_direction_ml
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python walk_forward.py
python predict.py
streamlit run app.py
```

Or run:

```powershell
run_windows.bat
```

## Run on Linux or macOS

```bash
cd bitcoin_direction_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python walk_forward.py
python predict.py
streamlit run app.py
```

## Predict from a newer CSV

The new file must use the same Binance daily-column schema.

```bash
python predict.py path/to/new_btc_daily.csv
```

The Streamlit dashboard also has a **Predict new CSV** tab.

## Methodology

### Feature engineering

The 66 inputs include:

- returns and log returns over 1–30 days;
- candle body, range, wick and closing-location features;
- SMA and EMA ratios and slopes;
- RSI, MACD, Bollinger position/width and normalized ATR;
- rolling volatility;
- volume and trade-count momentum;
- taker-buy ratios and average trade sizes;
- cyclical day-of-week and month variables.

Every feature uses information available at or before day `t`. Future prices are used only to construct labels.

### Evaluation

The primary split uses Monday–Sunday calendar weeks in a repeating 10-week cycle:

- training: weeks 1, 2, 4, 7, 9 and 10;
- validation: weeks 3 and 6;
- test: weeks 5 and 8.

This yields approximately 60% / 20% / 20% while giving all three sets coverage across the complete timeline. Probability thresholds are chosen on validation weeks only, and test weeks are not used for model or threshold selection. Because later training weeks can occur after earlier test weeks, this design is an interleaved experiment rather than a strict simulated future forecast. `walk_forward.py` provides the separate chronological benchmark.

### Backtest

For an `h`-day target, the backtest uses non-overlapping `h`-day periods. It holds BTC when the probability exceeds the validation threshold and otherwise holds cash. A 0.1% cost is charged for entry and another 0.1% for exit. This is intentionally simple and does not simulate slippage, spread, latency, taxes or exchange outages.

## Strong next extensions

1. Add BTC funding rates, open interest and order-book imbalance.
2. Add on-chain activity and macroeconomic variables.
3. Use regime detection and rolling retraining.
4. Calibrate probabilities and study threshold stability.
5. Compare direct multi-horizon models with separate horizon models.
6. Extend the daily model with the included 4-hour data.
