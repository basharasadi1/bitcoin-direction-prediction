# Dataset

The project uses the included daily Binance BTC/USDT file:

`data/raw/btc_usdt_1d_2018_2026.csv`

## Coverage

- Frequency: 1 day
- Time zone: UTC
- First row: 2018-01-01
- Last row: 2026-06-16
- Rows: 3,089
- Missing values in the source file: none

## Original columns

| Column | Meaning |
|---|---|
| Open time | Start of the daily candle |
| Open, High, Low, Close | BTC/USDT OHLC prices |
| Volume | BTC traded |
| Close time | End of the daily candle |
| Quote asset volume | USDT traded |
| Number of trades | Number of Binance trades |
| Taker buy base asset volume | BTC bought by takers |
| Taker buy quote asset volume | USDT value bought by takers |
| Ignore | Unused Binance field |

The original archive also contained 15-minute, 1-hour and 4-hour files. The MVP intentionally uses the daily file so every target and feature has a clear daily interpretation.
