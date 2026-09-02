# Result archive

Archived outputs from the completed experiment runs reported in the paper. Nothing
here is regenerated at build time; these are the files the tables and figures read.

## Layout

| Directory | Contents |
|---|---|
| `circuit_depth/` | the main archive: the five-phase optimization study plus all validation and mechanism studies |
| `hardware/` | one JSON per IBM Quantum job, plus the mitigation, ZNE and shot-sensitivity outputs |
| `leakage_checks/` | preprocessing-order audit reports (JSON + Markdown) |
| `revision_stats/` | reanalyses added during revision: TOST equivalence, variance decomposition, Wilcoxon sensitivity, paired quantum-vs-classical |

## The core batch archive

The five-phase study contributes **78 files**: two configuration tiers (6-qubit /
5000-sample and 10-qubit / 2500-sample) x three datasets (IoTID20, UNSW-2018 Bot-IoT,
UNSW-NB15) x 13 files per dataset-specific batch. These comprise circuit metrics,
optimization comparisons, optimization-level sweeps, multi-method optimizer
comparisons, noisy-study aggregates, per-run noisy-study outputs, paired statistics,
cross-validation results, and batch summaries. This is the "78-file result archive"
referenced in the paper's Data Availability statement.

A seventh, **partial** batch is also present: `unsw_nb15_10q_5k_v2_noisy_20260420_124948`
(7 files). This is the abandoned attempt at a full 10-qubit / 5000-sample noisy study,
which proved computationally impractical under density-matrix simulation and was
replaced by the reduced 2500-sample tier. It is retained as evidence for the
feasibility statement in Section III-A and is deliberately excluded from the 78.

`circuit_depth/` additionally holds the validation and mechanism study outputs, which
are **not** part of that 78: calibrated device-noise runs (`device_noise_*`, including
the error-aware layout selections in `device_paths_*.json`), kernel-concentration
diagnostics (`concentration_*`), the regularization sweep (`ctune_*`), the Z-only
control and feature-map ablation (`ablation_feature_map_*`), classical baselines
(`classical_baseline_*`), and the quantum-vs-classical significance tests.

## Raw versus derived

- `*_runs_*.csv` are **raw** per-run records and carry a `seed` column.
- `*_summary_*.csv` are per-condition aggregates.
- `*_stats*.csv` / `*_statistics_*.csv` hold paired tests, effect sizes and
  Holm-adjusted p-values.
- `hardware/hw_*.json` are **raw** per-job records including `calibration_last_update`.

See `../REPRODUCIBILITY.md` for the mapping from each paper table and figure to the
script that produces it.
