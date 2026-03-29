# Technical Guide

## Research question

The experiment asks whether Qiskit circuit optimization changes the behavior of kernel-based quantum machine learning models for IoT intrusion detection under ideal and noisy simulation.

## Final study design

The archived paper package includes two evidence tiers.

The primary tier is a 6-qubit, 5000-sample study on two datasets with 30 repeated runs per condition. It evaluates QSVC, QVE, and QWE under L0 and L3 transpilation, full and linear entanglement, and a 10-level noise grid.

The supplementary tier is a reduced 10-qubit, 2500-sample extension on the same datasets with 10 repeated runs per condition. It evaluates QSVC and QVE only, keeps full entanglement, and uses a four-level noise grid.

## Result families

The archive contains the following result families for each completed dataset-and-configuration bundle:

- `circuit_metrics`
- `optimization_comparison`
- `optimization_levels`
- `optimization_statistics`
- `multi_method_model_results`
- `multi_method_optimization`
- `multi_method_statistics`
- `comprehensive_model_noise`
- `comprehensive_model_noise_runs`
- `comprehensive_model_noise_stats`
- `cv_results`
- `experiment_summary`
- `journal_summary`

## Reproducibility notes

The heavy workloads were run on a multi-GPU machine with Qiskit Aer GPU simulation and RAPIDS-backed classical learning. The result archive is small enough for normal Git, but the datasets require Git LFS because they exceed GitHub's standard file-size limit.
