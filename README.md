# Circuit Optimization and Noise Robustness in Quantum Kernel Models for IoT Intrusion Detection

Archival code and results for the paper *"Circuit Optimization and Noise Robustness in
Quantum Kernel Models for IoT Intrusion Detection"* (IEEE Transactions on Quantum
Engineering).

The study asks an end-to-end question: when a quantum kernel model is trained on a real
intrusion-detection task under noise, does aggressive transpiler optimization buy
accuracy, what does it cost, and what else decides the outcome? The answer is a
two-level trade-off. Aggressive optimization (Qiskit L0 versus L3) compresses circuits
but is nearly decoupled from downstream accuracy, while the encoding and the
entanglement topology decide whether a kernel survives noise, through kernel
concentration.

This repository contains the experiment code, the three datasets, the archived result
files, the validation and mechanism studies, and the scripts that regenerate every table
and figure in the paper.

## Quick start

```bash
git clone https://github.com/ocblvck/qml-circuit-optimization-iot-intrusion-detection.git
cd qml-circuit-optimization-iot-intrusion-detection
git lfs install && git lfs pull          # datasets are stored with Git LFS

conda create -n qiskit python=3.11.13
conda activate qiskit
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
```

**Verify the install in about a minute, with no GPU and no dataset needed.** This
recomputes the paper's headline statistics from the archived per-run results:

```bash
python revision_statistics.py
```

Expected output includes `equivalent within +/-1.0 pp (TOST) : 169 (93.9%)` and
`significant (paired t, Holm) : 10`, matching the paper.

**Then try a short end-to-end experiment** (a few minutes on a GPU machine):

```bash
python circuit_depth_experiment.py --quick_test --disable_noise --config_specs 4:200
```

Results are written to `results/circuit_depth/`.

## Requirements

All packages are pinned in `requirements.txt` to the exact versions used for the
reported results: Python 3.11.13, Qiskit 1.4.4, Qiskit Aer (GPU) 0.15.1, Qiskit IBM
Runtime 0.42.0, cuML 26.2.0, scikit-learn 1.7.2. Reference hardware is 3x NVIDIA
RTX A6000 (48 GB) with CUDA 12.

On a **CPU-only machine**, replace `qiskit-aer-gpu` with `qiskit-aer==0.15.1` and remove
`cupy-cuda12x` and `cuml-cu12`. Ideal-simulation runs and the entire statistical
reanalysis still work; the noisy density-matrix sweeps become impractically slow (the
full study took roughly 3,800 GPU-hours).

The real-hardware scripts additionally need an IBM Quantum account with access to a
Heron-class device.

## What is included

**Main experiment**

- `circuit_depth_experiment.py` — the five-phase optimization and noise study (the
  primary 6-qubit batch and the supplementary 10-qubit extension)
- `run_full_experiment.sh` — the published study sequence across all three datasets
- `run_deadline_safe_10q_extension.sh` — the reduced 10-qubit extension alone

**Validation and mechanism studies**

- `device_noise_validation.py`, `device_noise_validation_v2.py` — calibrated IBM
  device-noise evaluation; v2 adds error-aware qubit-path selection (best and
  deliberately adverse layouts)
- `kernel_concentration_experiment.py` — the Gram-matrix concentration diagnostic
- `z_only_control_experiment.py` — the stand-alone Z-map control
- `feature_map_ablation.py` — the ensemble committee-size ablation
- `mitigation_rescue_experiment.py` — idealized readout-mitigation test
- `zne_rescue_experiment.py` — zero-noise extrapolation via unitary folding
- `run_hardware_kernel.py` — execution on IBM Heron hardware
- `shot_sensitivity_experiment.py` — shot-convergence check
- `classical_baselines.py`, `quantum_vs_classical_significance.py` — classical
  baselines and the paired quantum-versus-classical comparison
- `qsvc_ctuning_experiment.py` — regularization sweep
- `preprocessing_leakage_check.py` — preprocessing-order audit
- `backend_calibration_table.py` — backend calibration summary

**Analysis and figures**

- `revision_statistics.py` — equivalence (TOST) testing, factorial variance
  decomposition, normality diagnostics and Wilcoxon sensitivity, paired
  quantum-versus-classical tests, and exponential concentration fits
- `figures_src/` — the four scripts that regenerate all 11 paper figures

**Data, results, documentation**

- `data/` — the three datasets (Git LFS). **Read `data/README.md` first**: it records
  each dataset's source, licence, required citation, and interpretation caveats.
- `results/` — the archived result files. See `results/README.md`.
- `docs/` — the runbook and technical guide used during the study.
- `REPRODUCIBILITY.md` — environment setup, seed documentation, the raw-versus-derived
  file convention, and a table mapping **every paper table and figure to the script
  that produces it**.

## Repository layout

```text
.
├── circuit_depth_experiment.py          # main five-phase study
├── device_noise_validation.py           # calibrated device noise (v1 helpers)
├── device_noise_validation_v2.py        # + error-aware layout selection
├── kernel_concentration_experiment.py
├── z_only_control_experiment.py
├── feature_map_ablation.py
├── mitigation_rescue_experiment.py
├── zne_rescue_experiment.py
├── run_hardware_kernel.py
├── shot_sensitivity_experiment.py
├── classical_baselines.py
├── quantum_vs_classical_significance.py
├── qsvc_ctuning_experiment.py
├── preprocessing_leakage_check.py
├── backend_calibration_table.py
├── revision_statistics.py               # revision reanalyses
├── figures_src/                         # figure generators
├── run_full_experiment.sh
├── run_deadline_safe_10q_extension.sh
├── requirements.txt
├── REPRODUCIBILITY.md
├── data/                                # datasets (Git LFS) + licence notes
├── docs/
└── results/                             # archived results
```

## Datasets

- `data/IoT_Original_Distribution.csv` (IoTID20, after the wrangling in Sec. III-B)
- `data/UNSW_2018_IoT_Botnet_Final_10_Best.csv` (Bot-IoT, "Final 10 Best")
- `data/UNSW_NB15.csv` (UNSW-NB15)

Stored with Git LFS; run `git lfs pull` after cloning. Each dataset is the property of
its original authors and carries its own academic-use licence and required citation, all
recorded in `data/README.md`. The repository `LICENSE` (MIT) covers the code and derived
results only.

Scripts accept either `data/x.csv` or a bare `x.csv`; paths resolve relative to the
repository root, so the scripts work from any working directory.

## Results archive

`results/circuit_depth/` holds the **78-file core archive**: two configuration tiers
(6-qubit/5000-sample and 10-qubit/2500-sample) x three datasets x 13 files per batch.
It additionally holds the validation and mechanism outputs (device noise and selected
qubit paths, kernel concentration, regularization sweep, Z-only control and ablation,
classical baselines, significance tests) and one documented partial batch.

`results/hardware/` holds one JSON per IBM Quantum job — including the device
calibration in force at execution — plus the mitigation, ZNE, and shot-sensitivity
outputs. `results/leakage_checks/` holds the preprocessing-order audit.
`results/revision_stats/` holds the revision reanalyses.

See `results/README.md` for the raw-versus-derived convention.

## Reproducing the full study

```bash
bash run_full_experiment.sh                  # 6q primary + 10q extension, all datasets
bash run_deadline_safe_10q_extension.sh      # 10q extension only
```

Or invoke the experiment directly:

```bash
python circuit_depth_experiment.py \
  --datasets data/IoT_Original_Distribution.csv \
             data/UNSW_2018_IoT_Botnet_Final_10_Best.csv \
             data/UNSW_NB15.csv \
  --config_specs 6:5000 \
  --n_runs 30
```

The full sequence is a multi-day GPU run. `REPRODUCIBILITY.md` lists cheaper entry
points and the per-table script mapping.

## Citation

If you use this repository, cite the associated paper, and cite the original dataset
papers listed in `data/README.md`.

## License

Code and derived result files: see `LICENSE` (MIT). The datasets in `data/` are subject
to their original authors' terms, documented in `data/README.md`.
