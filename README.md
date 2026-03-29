# Quantum Circuit Optimization for QML-Based IoT Intrusion Detection

This repository is the archival package for the experiment behind the paper "Circuit Optimization and Noise Robustness of Quantum Kernel Models for IoT Intrusion Detection." It contains the experiment code, the two datasets used in the study, and the archived result files from the completed 6-qubit and 10-qubit batches.

The study asks a narrow question: when kernel-based quantum machine learning models are applied to IoT intrusion detection, how much does circuit optimization change downstream classification performance, noise robustness, and runtime?

## What is included

- `circuit_depth_experiment.py` is the main experiment entry point.
- `run_full_experiment.sh` runs the final published study sequence: the 6-qubit primary batch followed by the reduced 10-qubit extension.
- `run_deadline_safe_10q_extension.sh` runs the reduced 10-qubit extension used in the final paper.
- `data/` contains the two CSV datasets used in the study.
- `results/circuit_depth/` contains the 52 archived result artifacts from the completed experiment batches.
- `results/leakage_checks/` contains the preprocessing leakage check report used to audit the evaluation pipeline.
- `docs/` contains the runbook and technical guide used during the study.

## Repository layout

```text
.
├── circuit_depth_experiment.py
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

These files are large and are intended to be tracked with Git LFS. Before making the repository public, confirm that redistribution is consistent with the original dataset terms.

## Results archive

The result archive reflects two completed experiment batches:

- `6q_5k` primary batch, with 30 repeated runs per condition on both datasets
- `10q_2p5k` supplementary batch, with 10 repeated runs per condition on both datasets

Together, these batches produced 52 archived result files in `results/circuit_depth/`. The main result families are circuit metrics, optimization comparisons, optimization-level sweeps, comprehensive noisy-study outputs, cross-validation outputs, and batch summaries.

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
  --datasets data/IoT_Original_Distribution.csv data/UNSW_2018_IoT_Botnet_Final_10_Best.csv \
  --n_runs 30
```

## Citation

If you use this repository, cite the associated paper and cite or link the repository artifact in any reproducibility note or data-availability statement.

## License

This repository uses the license in `LICENSE`. Dataset reuse may be subject to additional third-party terms.