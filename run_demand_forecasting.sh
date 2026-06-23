#!/usr/bin/env bash
set -euo pipefail

# Demand Forecasting launcher.
# Trains/evaluates a forecaster on censored sales (default) or on recovered
# demand. "recovered" requires a prior recovery run that produced
# latent_demand_recovery/exp/demand/demand.parquet
# (see ./run_latent_demand_recovery.sh).
#
# Usage:
#   ./run_demand_forecasting.sh [METHOD] [SOURCE]
#
#   METHOD  one of: SSA TFT DLinear   (default: SSA)
#   SOURCE  censored | recovered      (default: censored)
#
# Notes:
#   - SSA      is statistics-based (no training), fastest to smoke-test.
#   - TFT      runs train then predict.
#   - DLinear  delegates to its own train_predict[_on_recovered].sh wrapper.
#
# Examples:
#   ./run_demand_forecasting.sh SSA                # SSA on censored sales
#   ./run_demand_forecasting.sh TFT recovered      # TFT on recovered demand
#   ./run_demand_forecasting.sh DLinear censored

METHOD="${1:-SSA}"
SOURCE="${2:-censored}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the repo's virtualenv automatically (works in any shell, no `activate`
# needed). Skips this if you already have a conda/venv active, or set
# USE_VENV=0 to opt out. Also covers DLinear's nested train_predict*.sh, which
# call bare `python`, via the prepended PATH.
if [[ "${USE_VENV:-1}" == "1" && -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" \
      && -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
  export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
fi
echo ">> using python: $(command -v python)"

DEMAND_FLAG=""
case "$SOURCE" in
  recovered) DEMAND_FLAG="--demand" ;;
  censored)  DEMAND_FLAG="" ;;
  *) echo "SOURCE must be 'censored' or 'recovered' (got '$SOURCE')" >&2; exit 1 ;;
esac

case "$METHOD" in
  SSA)
    cd "$SCRIPT_DIR/demand_forecasting/SSA"
    echo ">> SSA | source=$SOURCE"
    python ssa_forecasting.py $DEMAND_FLAG
    ;;
  TFT)
    cd "$SCRIPT_DIR/demand_forecasting/TFT"
    echo ">> TFT train | source=$SOURCE"
    python3 trainTFT.py $DEMAND_FLAG
    echo ">> TFT predict | source=$SOURCE"
    python3 predictTFT.py $DEMAND_FLAG
    ;;
  DLinear)
    cd "$SCRIPT_DIR/demand_forecasting/DLinear"
    if [[ "$SOURCE" == "recovered" ]]; then
      echo ">> DLinear | source=recovered"
      sh train_predict_on_recovered.sh
    else
      echo ">> DLinear | source=censored"
      sh train_predict.sh
    fi
    ;;
  *)
    echo "METHOD must be one of: SSA TFT DLinear (got '$METHOD')" >&2
    exit 1
    ;;
esac
