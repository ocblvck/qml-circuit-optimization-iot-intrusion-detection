# Reproducibility guide

For *"Circuit Optimization and Noise Robustness in Quantum Kernel Models for IoT
Intrusion Detection"* (IEEE Transactions on Quantum Engineering).

## 1. Environment

```bash
conda create -n qiskit python=3.11.13
conda activate qiskit
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
```

Every package in `requirements.txt` is pinned with `==` to the exact version used to
produce the reported results. Reference hardware: 3× NVIDIA RTX A6000 (48 GB),
CUDA 12, Linux x86_64, Python 3.11.13.

`qiskit-aer-gpu`, `cupy-cuda12x` and `cuml-cu12` need CUDA 12 and NVIDIA GPUs. On a
CPU-only machine, swap `qiskit-aer-gpu` for `qiskit-aer==0.15.1` and drop cupy/cuml:
ideal-simulation and the full statistical reanalysis still run, but the noisy
density-matrix sweeps become impractical (the Phase-4 tiers took ≈3,400 GPU-hours).

Hardware experiments additionally need an IBM Quantum account with access to a
Heron-class device.

## 2. Data

The three datasets ship in `data/` via Git LFS. **Read `data/README.md` first** — it
records each dataset's source, verbatim licence, required citation, and the
prevalence caveat that governs how UNSW-2018 numbers may be interpreted.

```bash
git lfs install && git lfs pull
```

## 3. Random seeds

Seeds are fixed *and recorded*, which is what makes the paired analyses checkable:

- Primary 6-qubit study: **30 seeds**, beginning `42, 123, 456, 789, 1024, 2048,
  3072, 4096, 5120, 6144, …` (see `ALL_SEEDS` in `circuit_depth_experiment.py`).
  The supplementary 10-qubit tier uses the first **10** of the same list.
- A seed fixes the subsample draw, the train/test split, and every fitted
  preprocessing transform, so it identifies a complete experimental condition.
- **Every per-run result file carries a `seed` column**, so any single run can be
  located and re-executed.
- Transpilation uses a fixed `seed_transpiler=42`; Section III-E of the paper reports
  a ten-seed sweep quantifying sensitivity to that choice.
- Classical baselines reuse the identical seeds and splits — this is what licenses
  the *paired* quantum-versus-classical comparison in Section IV-G.

## 4. Raw versus derived results

| Pattern | Content | Kind |
|---|---|---|
| `*_runs_*.csv` | one row per run, carries `seed` | **raw** |
| `*_summary_*.csv` | per-condition aggregates (means, CIs) | derived |
| `*_stats*.csv`, `*_statistics_*.csv` | paired tests, effect sizes, Holm-adjusted p | derived |
| `results/hardware/hw_*.json` | one on-device job, with calibration timestamp | **raw** |
| `results/revision_stats/*.csv` | revision reanalyses (equivalence, ANOVA, paired) | derived |
| `results/leakage_checks/*` | preprocessing-order audit reports | derived |

Every derived file is regenerable from the raw files using the scripts below.

## 5. Regenerating each table and figure

| Paper artifact | Script |
|---|---|
| Table 2 — backend calibration | `backend_calibration_table.py` |
| Tables 3, 6 — circuit metrics, Holm contrasts | `circuit_depth_experiment.py` |
| Table 5 — TOST equivalence; variance decomposition; Wilcoxon; paired quantum-vs-classical; concentration fits | `revision_statistics.py` |
| Tables 7, 8 — concentration diagnostics | `kernel_concentration_experiment.py` |
| Table 9 — Z-only control | `z_only_control_experiment.py` |
| Tables 10, 11 — calibrated device noise, 6q and 10q, best/adverse layout | `device_noise_validation_v2.py` |
| Table 12 — idealized readout mitigation | `mitigation_rescue_experiment.py` |
| Table 13 — zero-noise extrapolation | `zne_rescue_experiment.py` |
| Table 14 — real-hardware runs | `run_hardware_kernel.py` |
| Table 15 — classical baselines | `classical_baselines.py` |
| Table 16 — paired quantum-vs-classical significance | `quantum_vs_classical_significance.py` |
| Table 17 — cross-validation | `circuit_depth_experiment.py` (Phase 5) |
| Table 18 — feature-map count ablation | `feature_map_ablation.py` |
| Table 20 — regularization sweep | `qsvc_ctuning_experiment.py` |
| Appendix C — preprocessing-order audit | `preprocessing_leakage_check.py` |
| Shot-convergence check (Sec. IV-F) | `shot_sensitivity_experiment.py` |
| Figure 2 — trade-off summary | `figures_src/generate_tradeoff_figure.py` |
| Figures 3, 8, 9 | `figures_src/generate_missing_figures.py` |
| Figures 4–7, 10, 11 | `figures_src/generate_enhancement_figures.py` |
| Revision figures | `figures_src/generate_revision_figures.py` |

### Fastest independent check

The statistical reanalysis needs **no GPU and no dataset download** — it reads only
the archived per-run CSVs:

```bash
python revision_statistics.py
```

It reproduces the headline revision numbers: 169 of 180 contrasts equivalent within
1 accuracy point, the variance decomposition (≈0.4% of MCC variance attributable to
optimization versus ≈53% to encoding and entanglement), the normality diagnostics
and Wilcoxon sensitivity analysis, the paired quantum-versus-classical tests, and the
exponential concentration fits.

Figures are regenerated with:

```bash
python figures_src/generate_tradeoff_figure.py     # writes figures_src/figures/
python figures_src/generate_missing_figures.py
python figures_src/generate_enhancement_figures.py
```

## 6. Hardware provenance

Each on-device job writes a JSON to `results/hardware/` recording the backend, UTC
execution timestamp, **`calibration_last_update`** (the device calibration in force
at execution), dataset, model, qubit count, entanglement, train/test sizes, shots,
optimization level, mitigation flag, circuit count, and wall time.

The six real-device runs span calibrations from 2026-06-22 to 2026-07-04. The
UNSW-NB15 QSVC run cleared the shared-device queue only after a multi-week
fair-share wait and therefore ran under the later 2026-07-04 calibration, which the
paper states explicitly; its archived JSON records that date.

## 7. Known limits on exact reproduction

- Kernel evaluation is GPU-accelerated, so floating-point non-determinism can change
  accuracies in the last decimal place across GPUs. Appendix A quantifies this with
  an independent replicate of an identical configuration.
- Real-hardware results depend on device calibration at execution time and will not
  reproduce exactly. The `Ideal` and `Heron-sim` rungs of each hardware row are the
  reproducible references.
- Fake-backend calibration snapshots ship inside `qiskit-ibm-runtime` and change
  between releases; the pinned version is the one that produced Table 2.
