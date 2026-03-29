#!/bin/bash
# Reduced 10q supplementary extension launcher.
#
# Keeps the completed 6q_5k results as the primary study and runs a
# deadline-safe 10q extension for both datasets.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/deadline_safe_10q_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  DEADLINE-SAFE 10Q EXTENSION RUN"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Log directory: $LOG_DIR"
echo ""
echo "Datasets to run:"
echo "  - IoT_Original_Distribution.csv"
echo "  - UNSW_2018_IoT_Botnet_Final_10_Best.csv"
echo ""
echo "Configuration to run:"
echo "  - 10 qubits x 2500 samples"
echo ""
echo "Phase 4 scope reduction:"
echo "  - Models: QSVC, QVE"
echo "  - Entanglement: full"
echo "  - Noise levels: 0.0, 0.002, 0.01, 0.05"
echo "  - Repeated runs: 10"
echo ""
echo "Study framing: supplementary 10q scalability extension"
echo "=========================================="
echo ""

if [ -f "$HOME/miniconda/bin/activate" ]; then
    source "$HOME/miniconda/bin/activate" qiskit || true
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting reduced 10q extension"
python3 -u circuit_depth_experiment.py \
    --datasets data/IoT_Original_Distribution.csv data/UNSW_2018_IoT_Botnet_Final_10_Best.csv \
    --config_specs 10:2500 \
    --n_runs 10 \
    --phase4_models QSVC QVE \
    --phase4_entanglements full \
    --phase4_noise_levels 0.0 0.002 0.01 0.05 \
    2>&1 | tee "${LOG_DIR}/deadline_safe_10q.log"

echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEADLINE-SAFE 10Q EXTENSION COMPLETE"
echo "=========================================="
echo "Results saved in: results/circuit_depth/"
echo "Logs saved in: $LOG_DIR"