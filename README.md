# Quantum Circuit Optimization for QML-Based IoT Intrusion Detection

This repository is the archival package for the experiment behind the paper "Noise Robustness and Circuit Optimization of Quantum Kernel Models for IoT Intrusion Detection." It contains the experiment code, the three intrusion-detection datasets used in the study, the archived result files from the completed 6-qubit and 10-qubit batches, and the code for the calibrated device-noise, real-hardware, and kernel-analysis studies reported in the paper.

The study asks a focused question: when kernel-based quantum machine learning models are applied to IoT intrusion detection, how much does circuit optimization change downstream classification performance and noise robustness, and what determines whether a kernel survives noise? Alongside the optimization sweep, the repository reproduces the calibrated device-noise validation, the runs on IBM Heron hardware, and the kernel-concentration analysis that explains the observed noise behavior.

## What is included

- `circuit_depth_experiment.py` is the main experiment entry point for the optimization and noise sweep.
- `run_full_experiment.sh` runs the final published study sequence: the 6-qubit primary batch followed by the reduced 10-qubit extension.
- `run_deadline_safe_10q_extension.sh` runs the reduced 10-qubit extension used in the final paper.
- `feature_map_ablation.py`, `kernel_concentration_experiment.py`, `qsvc_ctuning_experiment.py`, and `shot_sensitivity_experiment.py` reproduce the feature-map ablation, the kernel-concentration diagnostic, the regularization sweep, and the shot-convergence study.
- `run_hardware_kernel.py` runs the calibrated device-noise and real-hardware (IBM Heron) kernel evaluations; `mitigation_rescue_experiment.py` runs the error-mitigation study on the 10-qubit collapse.
- `data/` contains the three CSV datasets used in the study.
- `results/circuit_depth/` contains the 78 archived result artifacts from the completed 6-qubit and 10-qubit batches; `results/hardware/` contains the device-noise, on-device, and kernel-analysis outputs.
- `results/leakage_checks/` contains the preprocessing leakage check report used to audit the evaluation pipeline.
- `docs/` contains the runbook and technical guide used during the study.

## Repository layout

```text
.
├── circuit_depth_experiment.py
├── feature_map_ablation.py
├── kernel_concentration_experiment.py
├── qsvc_ctuning_experiment.py
├── shot_sensitivity_experiment.py
├── run_hardware_kernel.py
├── mitigation_rescue_experiment.py
├── preprocessing_leakage_check.py
├── run_full_experiment.sh
├── run_deadline_safe_10q_extension.sh
├── data/
├── docs/
└── results/
```

## Datasets

The repository includes the exact dataset files used in the reported runs:

- `data/IoT_Original_Distribution.csv`
- `data/UNSW_2018_IoT_Botnet_Final_10_Best.csv`
- `data/UNSW_NB15.csv`

These files are large and are tracked with Git LFS. Before redistributing them, confirm that redistribution is consistent with the original dataset terms.

## Results archive

The result archive reflects two completed experiment batches:

- `6q_5k` primary batch, with 30 repeated runs per condition on all three datasets
- `10q_2p5k` supplementary batch, with 10 repeated runs per condition on all three datasets

Together, these batches produced 78 archived result files in `results/circuit_depth/`. The main result families are circuit metrics, optimization comparisons, optimization-level sweeps, comprehensive noisy-study outputs, cross-validation outputs, and batch summaries. The `results/hardware/` directory holds the calibrated device-noise runs, the real-hardware (IBM Heron) kernel evaluations, and the kernel-concentration, regularization, shot-convergence, and error-mitigation outputs.

## Running the experiment

The experiment expects a Python environment with Qiskit, Qiskit Aer, scikit-learn, and the GPU-backed packages used in the original runs.

Published study sequence:

```bash
bash run_full_experiment.sh
```

Reduced 10-qubit extension:

```bash
bash run_deadline_safe_10q_extension.sh
```

You can also run the script directly:

```bash
python3 circuit_depth_experiment.py \
  --datasets data/IoT_Original_Distribution.csv data/UNSW_2018_IoT_Botnet_Final_10_Best.csv data/UNSW_NB15.csv \
  --n_runs 30
```

## Citation

If you use this repository, cite the associated paper and cite or link the repository artifact in any reproducibility note or data-availability statement.

## License

This repository uses the license in `LICENSE`. Dataset reuse may be subject to additional third-party terms.