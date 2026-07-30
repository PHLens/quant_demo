# Learn / Lab-1

`lab/` is an isolated validation-only experiment domain. It intentionally does
not import `web.state`, strategy registries, best profiles, ETF/index loaders,
or legacy cache writers.

## Frozen vertical slice

- Template: `CSI1000 Trend 101`
- Editable field: only `trend_window ∈ {20, 50, 100}`; `50` is the locked
  packaged baseline and a user Variant must choose `20` or `100`.
- Snapshot:
  `snapshot_csi1000_trend_101_v1_8193bfff89f4ae4b`,
  content SHA256
  `8193bfff89f4ae4b7afce8f5115f7b4ef661371b6093ad7e076b029eb748ec66`.
- Input/warmup range: `2023-07-03..2025-12-31`.
- Validation range: `2024-01-02..2025-12-31` (485 bars).
- Execution: index close(t) signal, 510980 qfq ETF open(t+1) execution,
  ETF close(t+1) mark.
- OOS: `not_configured`.

The default runtime store is `stock_trade_demo/data/lab_artifacts/` (gitignored)
and can be overridden with `QUANT_LAB_ARTIFACT_DIR`. It contains only Lab
experiments, variants, run attempts, and immutable results.

## Baseline maintenance

The Web app fails closed if the packaged baseline no longer matches the current
runner fingerprint or snapshot hash. A reviewed change can check or rebuild it
offline:

```bash
PYTHONPATH=stock_trade_demo python -m lab.build_baseline --check
PYTHONPATH=stock_trade_demo python -m lab.build_baseline --write
```

Rebuilding is not a deployment action and never writes Research caches,
registries, source configuration, or `/live`.
