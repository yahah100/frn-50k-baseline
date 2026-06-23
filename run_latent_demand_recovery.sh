#!/usr/bin/env bash
set -euo pipefail

# Latent Demand Recovery launcher.
# Reconstructs the true (latent) demand from censored sales, or evaluates the
# imputation quality under artificial MNAR masking.
#
# Usage:
#   ./run_latent_demand_recovery.sh [MODEL] [MISSING_RATE]
#
#   MODEL         one of: TimesNet ImputeFormer SAITS iTransformer GPVAE CSDI DLinear
#                 (default: TimesNet)
#   MISSING_RATE  0   -> real recovery; writes exp/demand/demand.parquet  (default)
#                 >0  -> artificial MNAR evaluation, prints WAPE/WPE, no parquet
#
# The parquet written with MISSING_RATE=0 is what the demand-forecasting
# scripts consume in their "recovered" mode.
#
# Examples:
#   ./run_latent_demand_recovery.sh                 # TimesNet, real recovery
#   ./run_latent_demand_recovery.sh DLinear         # fast model, real recovery
#   ./run_latent_demand_recovery.sh SAITS 0.3       # MNAR eval at 30% missing

MODEL="${1:-TimesNet}"
MISSING_RATE="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the repo's virtualenv automatically (works in any shell, no `activate`
# needed). Skips this if you already have a conda/venv active, or set
# USE_VENV=0 to opt out.
if [[ "${USE_VENV:-1}" == "1" && -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" \
      && -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
  export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
fi
echo ">> using python: $(command -v python)"

cd "$SCRIPT_DIR/latent_demand_recovery/exp"

echo ">> Latent Demand Recovery | model=$MODEL missing_rate=$MISSING_RATE"
python app.py --model "$MODEL" --missing_rate "$MISSING_RATE"
