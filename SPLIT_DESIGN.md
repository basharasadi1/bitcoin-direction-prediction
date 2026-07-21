# Repeating 10-Week Data Split

The primary experiment divides Monday-Sunday calendar weeks into a repeating 10-week cycle.

| Cycle week | Assignment |
|---:|---|
| 1 | Training |
| 2 | Training |
| 3 | Validation |
| 4 | Training |
| 5 | Test |
| 6 | Validation |
| 7 | Training |
| 8 | Test |
| 9 | Training |
| 10 | Training |

The cycle then starts again at week 1. On the included engineered dataset, this produces:

- Training: 1,735 rows (60.18%)
- Validation: 574 rows (19.91%)
- Test: 574 rows (19.91%)

A row is assigned according to the date on which its features are observed and its prediction is made. The 1-, 5-, and 7-day labels still use the corresponding future close.

## Interpretation

This split gives all three partitions coverage across the entire 2018-2026 market history. It is useful for testing whether patterns transfer to held-out weeks across different regimes.

It is not a strict future-only simulation: a model can train on a later calendar week and be evaluated on an earlier test week. The project therefore retains `walk_forward.py` as a separate chronological forecasting benchmark.
