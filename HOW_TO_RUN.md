# How to Run

This guide explains how to run and test the models in this repo. The two tasks
are kept separate: **latent demand recovery** (reconstruct the true demand from
censored sales) and **demand forecasting** (predict future sales on either the
censored sales or the recovered demand).

Two launcher scripts at the repo root wrap the underlying commands:

- `run_latent_demand_recovery.sh`
- `run_demand_forecasting.sh`

## Setup

The project is managed by [**uv**](https://docs.astral.sh/uv/). Python 3.8 is pinned in
`.python-version`, dependencies are declared in `pyproject.toml`, and `uv.lock` pins the full
resolved set. Create the environment once:

```bash
uv sync          # creates ./.venv from uv.lock and installs everything
```

**You do not need to activate the venv** — the launcher scripts automatically use `./.venv` if it
exists, in any shell (bash/fish/nu), so a bare `./run_demand_forecasting.sh` just works. To run an
arbitrary command in the environment, use `uv run`, e.g. `uv run python app.py ...`. To override
which interpreter the launchers use:

- have your own conda/venv active → the scripts defer to it,
- or set `USE_VENV=0` to force the system `python`,
- or set `PATH`/activate manually as usual.

Notes before the first run:

- **Data download**: every script calls `load_dataset("Dingdong-Inc/FreshRetailNet-50K")`
  from HuggingFace. The first run pulls the full dataset (sizable); afterwards it
  is cached locally by `datasets`.
- **GPU**: TFT hardcodes GPU use, so it needs CUDA. Latent demand recovery falls
  back to CPU automatically. SSA is pure NumPy/pandas (no GPU needed).

## Latent Demand Recovery

Reconstructs the latent demand from censored sales, or evaluates imputation
quality under artificial MNAR masking.

```bash
./run_latent_demand_recovery.sh [MODEL] [MISSING_RATE]
```

| Argument       | Default    | Meaning                                                                 |
| -------------- | ---------- | ----------------------------------------------------------------------- |
| `MODEL`        | `TimesNet` | One of: `TimesNet ImputeFormer SAITS iTransformer GPVAE CSDI DLinear`   |
| `MISSING_RATE` | `0`        | `0` = real recovery (writes parquet); `>0` = MNAR eval (prints WAPE/WPE) |

- With `MISSING_RATE=0` it performs the *real* reconstruction and writes
  `latent_demand_recovery/exp/demand/demand.parquet`. This file is what the
  forecasting scripts consume in `recovered` mode.
- With `MISSING_RATE>0` it masks observed values to *evaluate* imputation
  quality instead — no parquet is written.

Examples:

```bash
./run_latent_demand_recovery.sh                 # TimesNet, real recovery → writes demand.parquet
./run_latent_demand_recovery.sh DLinear         # fastest model, real recovery
./run_latent_demand_recovery.sh SAITS 0.3       # MNAR eval at 30% missing
```

## Demand Forecasting

Trains/evaluates a forecaster on censored sales (default) or on recovered
demand.

```bash
./run_demand_forecasting.sh [METHOD] [SOURCE]
```

| Argument | Default    | Meaning                                  |
| -------- | ---------- | ---------------------------------------- |
| `METHOD` | `SSA`      | One of: `SSA TFT DLinear`                |
| `SOURCE` | `censored` | `censored` or `recovered`                |

- `SSA` is statistics-based (no training) — fastest to smoke-test.
- `TFT` runs train then predict automatically.
- `DLinear` delegates to its own `train_predict[_on_recovered].sh` wrapper.
- `recovered` reads `latent_demand_recovery/exp/demand/demand.parquet`, so
  **run a recovery with `MISSING_RATE=0` first** before any `recovered` forecast.

Examples:

```bash
./run_demand_forecasting.sh SSA                 # SSA on censored sales
./run_demand_forecasting.sh TFT recovered       # TFT train+predict on recovered demand
./run_demand_forecasting.sh DLinear censored    # DLinear via its own .sh wrapper
```

## Suggested order to test things

1. Fastest end-to-end smoke test (no GPU, no training):
   ```bash
   ./run_demand_forecasting.sh SSA
   ```
2. Produce recovered demand (DLinear is the quickest recovery model):
   ```bash
   ./run_latent_demand_recovery.sh DLinear
   ```
3. Forecast on the recovered demand:
   ```bash
   ./run_demand_forecasting.sh SSA recovered
   ```

## Metrics

Both tasks report **WAPE** (weighted absolute percentage error) and **WPE**
(weighted percentage error / bias), generally computed only on stocked-out rows
(the MNAR `valid_idx` mask in recovery, or `stock_hour6_22_cnt == 0` for the
clean subset in forecasting).
