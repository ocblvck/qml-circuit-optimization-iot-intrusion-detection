# Experiment Runbook

This repository packages the experiment exactly as it was narrowed for the journal paper.

## Scope

The archive covers two completed batches:

- a primary 6-qubit, 5000-sample batch on both datasets with 30 repeated runs
- a supplementary 10-qubit, 2500-sample batch on both datasets with 10 repeated runs

The core model families are QSVC, QVE, and QWE. The 10-qubit extension keeps only QSVC and QVE, uses full entanglement only, and uses a reduced four-level noise grid.

## Main commands

Run the paper-aligned study sequence:

```bash
bash run_full_experiment.sh
```

Run only the supplementary 10-qubit extension:

```bash
bash run_deadline_safe_10q_extension.sh
```

Run the primary batch directly:

```bash
python3 circuit_depth_experiment.py \
    --datasets data/IoT_Original_Distribution.csv data/UNSW_2018_IoT_Botnet_Final_10_Best.csv \
    --config_specs 6:5000 \
    --n_runs 30
```

## Output locations

- `results/circuit_depth/` contains the archived experiment outputs
- `results/leakage_checks/` contains the preprocessing audit report

## Notes

The repository includes the exact datasets and result files used in the final paper package. The datasets are large and tracked with Git LFS.
