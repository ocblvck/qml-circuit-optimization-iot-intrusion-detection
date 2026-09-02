#!/bin/bash
# Full circuit-depth experiment launcher.
#
# Runs the kernel-QML study focused on baseline vs Qiskit L3 circuits under
# ideal and noisy simulation with 30 repeated runs per configuration.
# Active model set: QSVC, QVE, QWE.

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/experiment_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  CIRCUIT DEPTH EXPERIMENT - FULL RUN"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Log directory: $LOG_DIR"
echo ""
echo "Configurations to run:"
echo "  1. 6 qubits × 5000 samples (primary batch)"
echo "  2. 10 qubits × 2500 samples (supplementary extension)"
echo ""
echo "Datasets to run:"
echo "  - IoT_Original_Distribution.csv"
echo "  - UNSW_2018_IoT_Botnet_Final_10_Best.csv"
echo "  - UNSW_NB15.csv"
echo ""
echo "Primary batch models: QSVC, QVE, QWE"
echo "Supplementary extension models: QSVC, QVE"
echo ""
echo "Optimization comparison: baseline (L0) vs Qiskit level 3 (L3)"
echo "Primary noise design: default 10 noise levels with 30 repeated runs"
echo "Supplementary noise design: 0.0, 0.002, 0.01, 0.05 with 10 repeated runs"
echo ""
echo "Estimated time: long multi-dataset run; monitor the combined log"
echo "=========================================="
echo ""

# Activate the original conda environment if it exists.
if [ -f "$HOME/miniconda/bin/activate" ]; then
    source "$HOME/miniconda/bin/activate" qiskit || true
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Run the final paper-aligned study sequence.
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 6q primary batch"
python3 -u circuit_depth_experiment.py \
    --datasets data/IoT_Original_Distribution.csv data/UNSW_2018_IoT_Botnet_Final_10_Best.csv data/UNSW_NB15.csv \
    --config_specs 6:5000 \
    --n_runs 30 \
    2>&1 | tee "${LOG_DIR}/primary_6q.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 10q supplementary extension"
python3 -u circuit_depth_experiment.py \
    --datasets data/IoT_Original_Distribution.csv data/UNSW_2018_IoT_Botnet_Final_10_Best.csv data/UNSW_NB15.csv \
    --config_specs 10:2500 \
    --n_runs 10 \
    --phase4_models QSVC QVE \
    --phase4_entanglements full \
    --phase4_noise_levels 0.0 0.002 0.01 0.05 \
    2>&1 | tee "${LOG_DIR}/supplementary_10q.log"

echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL CONFIGURATIONS COMPLETE!"
echo "=========================================="
echo "Results saved in: results/circuit_depth/"
echo "Logs saved in: $LOG_DIR"
echo ""
echo "Check results with:"
echo "  ls -la results/circuit_depth/"
echo "  tail -100 ${LOG_DIR}/primary_6q.log"
echo "  tail -100 ${LOG_DIR}/supplementary_10q.log"
