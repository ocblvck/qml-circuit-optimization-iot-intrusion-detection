#!/usr/bin/env python3
"""
Device-Calibrated Noise Validation for Quantum Kernel IoT Intrusion Detection
=============================================================================

Research question
-----------------
The main study (circuit_depth_experiment.py) sweeps a single, hand-tuned,
spatially UNIFORM depolarizing + thermal-relaxation channel. A reviewer will
ask: do the robustness conclusions (QVE robust; full-entanglement QSVC / QWE
brittle) still hold under a REALISTIC, spatially HETEROGENEOUS, device-calibrated
noise model taken from an actual IBM Quantum backend?

This harness answers that by re-running the same three models under noise models
built with ``NoiseModel.from_backend(<IBM fake backend>)``. Each fake backend
ships the real device's calibration snapshot:

    * per-qubit T1 / T2 and gate times
    * per-qubit single-qubit gate error rates
    * per-pair two-qubit (CX/ECR) gate error rates
    * the device native basis gates and coupling map

Honest scope note
-----------------
The fidelity kernel used throughout this project (NoisyFidelityKernel) is
MEASUREMENT-FREE: it computes the Hilbert-Schmidt overlap Tr(rho(x) rho(z)) of
density matrices, with no terminal measurement. Consequently a backend's
*readout* error is NOT exercised by this kernel. What this experiment adds over
the uniform sweep is therefore device-calibrated, spatially heterogeneous GATE
and COHERENCE error plus realistic coupling-induced routing overhead -- not
readout error. The device readout error is reported only as contextual metadata.

To make the device noise actually attach, each feature map is transpiled to the
backend's native basis gates over a linear 6-qubit coupling map; otherwise the
noise-model gate labels would not match the circuit and noise would silently not
apply.

Design (kept tractable)
-----------------------
* Reuses the EXACT primitives from circuit_depth_experiment.py so numbers are
  comparable to the main study. Nothing in the main script is modified.
* Conditions: 'ideal' (statevector) + one or more device profiles.
* Models: QSVC (single ZZ kernel), QVE (Z+ZZ hard vote), QWE (ZZ+Pauli weighted).
* Entanglement: full and linear. Optimization: L0 and L3.
* Default 10 paired runs across all three datasets, multi-GPU.

Outputs (written to results/circuit_depth/)
-------------------------------------------
    device_noise_runs_<session>.csv      per-run metrics
    device_noise_summary_<session>.csv   mean/std/CI per (model, ent, opt, device)
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel
import qiskit_ibm_runtime.fake_provider as fake_provider

# --- Reuse the exact building blocks from the main study (no modification) ---
from circuit_depth_experiment import (
    RESULTS_DIR,
    DataProcessor,
    GPUFidelityKernel,
    NoisyFidelityKernel,
    calculate_all_metrics,
    create_feature_map,
    cuSVC,
    cuml_svc_predict_proba,
    get_gpu_manager,
    run_parallel_on_gpus,
    transpile_with_cache,
    _normalize_coupling_map_key,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("device_noise_validation")

# Seeds mirror circuit_depth_experiment.py for run-paired comparability.
ALL_SEEDS = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
    17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624,
]

# Standard model -> feature-map composition (matches the main study / 2-map ablation).
MODEL_MAPS: Dict[str, List[Tuple[str, int]]] = {
    'QSVC': [('ZZ', 2)],
    'QVE': [('Z', 1), ('ZZ', 2)],
    'QWE': [('ZZ', 2), ('Pauli', 1)],
}

# Allowed unitary basis gates we will keep when transpiling to a device basis.
_UNITARY_BASIS = {'id', 'rz', 'sx', 'x', 'cx', 'ecr', 'cz'}


# ----------------------------------------------------------------------------
# Device profile cache (built once per worker process, keyed by device name)
# ----------------------------------------------------------------------------
_DEVICE_CACHE: Dict[str, Tuple[NoiseModel, List[str], float]] = {}


def get_device_profile(device_name: str) -> Tuple[NoiseModel, List[str], float]:
    """Return (noise_model, basis_gates, mean_readout_error) for a fake backend."""
    if device_name in _DEVICE_CACHE:
        return _DEVICE_CACHE[device_name]
    if not hasattr(fake_provider, device_name):
        raise ValueError(f"Unknown fake backend: {device_name}")
    backend = getattr(fake_provider, device_name)()
    noise_model = NoiseModel.from_backend(backend)
    basis = [g for g in noise_model.basis_gates if g in _UNITARY_BASIS]
    # Contextual-only: mean single-qubit readout error from the calibration.
    mean_readout = _mean_readout_error(backend)
    _DEVICE_CACHE[device_name] = (noise_model, basis, mean_readout)
    return _DEVICE_CACHE[device_name]


def _mean_readout_error(backend) -> float:
    """Best-effort extraction of mean readout error (contextual metadata only)."""
    try:
        target = backend.target
        errs = []
        if 'measure' in target:
            for _, props in target['measure'].items():
                if props is not None and props.error is not None:
                    errs.append(float(props.error))
        if errs:
            return float(np.mean(errs))
    except Exception:  # noqa: BLE001
        pass
    return float('nan')


# ----------------------------------------------------------------------------
# Data loading (mirrors feature_map_ablation.load_dataset)
# ----------------------------------------------------------------------------
def _detect_csv_delimiter(dataset_path: str) -> str:
    import csv
    with open(dataset_path, 'r', encoding='utf-8', errors='ignore', newline='') as handle:
        sample = ''.join(handle.readline() for _ in range(5))
    if not sample:
        return ','
    try:
        return csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
    except csv.Error:
        return ';' if sample.count(';') > sample.count(',') else ','


def resolve_dataset_path(dataset_path: str) -> str:
    """Locate a dataset CSV whether it is given as ``data/x.csv`` or bare ``x.csv``.

    Datasets ship in ``data/`` in this repository, but earlier runs kept them in the
    repository root. Accept either form, and resolve relative to this file's
    directory so the scripts work from any working directory.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    name = os.path.basename(dataset_path)
    for cand in (dataset_path,
                 os.path.join(here, dataset_path),
                 os.path.join(here, 'data', name),
                 os.path.join(here, name)):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"Dataset not found: {dataset_path}. Run 'python data/get_datasets.py' for "
        f"status and download instructions. The two UNSW datasets are not "
        f"redistributed here and must be obtained from UNSW; the shipped dataset "
        f"needs 'git lfs pull'."
    )


def load_dataset(dataset_path: str, num_qubits: int, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    dataset_path = resolve_dataset_path(dataset_path)
    delimiter = _detect_csv_delimiter(dataset_path)
    df = pd.read_csv(dataset_path, sep=delimiter, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]
    processor = DataProcessor(num_qubits=num_qubits)
    X, y = processor.prepare_data(df, sample_size=sample_size)
    return X, y


# ----------------------------------------------------------------------------
# Per-run worker (executed by run_parallel_on_gpus; gpu_id appended by dispatcher)
# ----------------------------------------------------------------------------
def execute_device_run(
    task_id: int,
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    num_qubits: int,
    ent_type: str,
    opt_level: int,
    device: str,
    model_name: str,
    seed: int,
    run_idx: int,
    gpu_id: int,
) -> Dict[str, Any]:
    """One paired run for a single (dataset, model, ent, opt, device) cell.

    device == 'ideal' -> noiseless statevector kernel.
    device == <FakeBackend name> -> device-calibrated density-matrix kernel.
    """
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
        processor.fit(X_train_raw, y_train)
        X_train = processor.transform(X_train_raw)
        X_test = processor.transform(X_test_raw)

        is_ideal = (device == 'ideal')
        mean_readout = float('nan')
        if is_ideal:
            noise_model, basis = None, ['u', 'cx', 'rz', 'sx', 'x']
        else:
            noise_model, basis, mean_readout = get_device_profile(device)

        linear_coupling = CouplingMap.from_line(num_qubits)
        map_specs = MODEL_MAPS[model_name]

        def _build_kernel(feature_map):
            if is_ideal:
                return GPUFidelityKernel(feature_map, gpu_id=gpu_id, assume_pretranspiled=True)
            return NoisyFidelityKernel(feature_map, noise_model, gpu_id=gpu_id)

        # Build + transpile each component feature map to the device native basis.
        kernels = []
        agg_depth = 0
        agg_2q = 0
        for map_type, reps in map_specs:
            base_fm = create_feature_map(num_qubits, map_type, reps=reps, entanglement=ent_type)
            cache_key = (
                'device', device, map_type, reps, num_qubits, ent_type,
                opt_level, _normalize_coupling_map_key(linear_coupling),
            )
            fm = transpile_with_cache(
                base_fm, cache_key,
                optimization_level=opt_level, basis_gates=basis,
                coupling_map=linear_coupling, seed_transpiler=42,
            )
            agg_depth += fm.depth()
            agg_2q += sum(c for g, c in fm.count_ops().items() if g in {'cx', 'ecr', 'cz'})
            kernels.append(_build_kernel(fm))

        kernel_time = 0.0
        classifier_time = 0.0
        y_proba = None

        if model_name == 'QSVC':
            kern = kernels[0]
            k0 = time.perf_counter()
            Kt = np.nan_to_num(kern.evaluate(X_train, X_train), nan=0.0, posinf=1.0, neginf=0.0)
            Kte = np.nan_to_num(kern.evaluate(X_test, X_train), nan=0.0, posinf=1.0, neginf=0.0)
            kernel_time += time.perf_counter() - k0
            svc = cuSVC(kernel='precomputed', class_weight='balanced',
                        random_state=seed, cache_size=8192.0,
                        max_iter=10000, nochange_steps=100, output_type='numpy')
            c0 = time.perf_counter()
            svc.fit(Kt, y_train)
            y_pred = np.asarray(svc.predict(Kte))
            y_proba = cuml_svc_predict_proba(svc, Kte)
            classifier_time += time.perf_counter() - c0

        elif model_name == 'QVE':
            # Hard majority voting (ties broken by mean soft proba).
            preds, probas = [], []
            for kern in kernels:
                k0 = time.perf_counter()
                Kt = np.nan_to_num(kern.evaluate(X_train, X_train), nan=0.0, posinf=1.0, neginf=0.0)
                Kte = np.nan_to_num(kern.evaluate(X_test, X_train), nan=0.0, posinf=1.0, neginf=0.0)
                kernel_time += time.perf_counter() - k0
                svc = cuSVC(kernel='precomputed', class_weight='balanced',
                            random_state=seed, cache_size=8192.0,
                            max_iter=10000, nochange_steps=100, output_type='numpy')
                c0 = time.perf_counter()
                svc.fit(Kt, y_train)
                preds.append(np.asarray(svc.predict(Kte)))
                probas.append(cuml_svc_predict_proba(svc, Kte))
                classifier_time += time.perf_counter() - c0
            pred_matrix = np.array(preds).T
            avg_proba = np.mean(np.asarray(probas), axis=0)
            final = []
            for i, row in enumerate(pred_matrix):
                ranking = Counter(row).most_common()
                if len(ranking) > 1 and ranking[0][1] == ranking[1][1]:
                    final.append(int(np.argmax(avg_proba[i])))
                else:
                    final.append(int(ranking[0][0]))
            y_pred = np.array(final)
            y_proba = avg_proba

        elif model_name == 'QWE':
            # Validation-accuracy-weighted soft voting.
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
            )
            validation_scores, test_probas = [], []
            for kern in kernels:
                k0 = time.perf_counter()
                K_fit = np.nan_to_num(kern.evaluate(X_fit, X_fit), nan=0.0, posinf=1.0, neginf=0.0)
                K_val = np.nan_to_num(kern.evaluate(X_val, X_fit), nan=0.0, posinf=1.0, neginf=0.0)
                kernel_time += time.perf_counter() - k0
                svc_val = cuSVC(kernel='precomputed', class_weight='balanced',
                                random_state=seed, cache_size=8192.0,
                                max_iter=10000, nochange_steps=100, output_type='numpy')
                c0 = time.perf_counter()
                svc_val.fit(K_fit, y_fit)
                y_val_pred = svc_val.predict(K_val)
                classifier_time += time.perf_counter() - c0
                validation_scores.append(max(accuracy_score(y_val, y_val_pred), 1e-6))

                k0 = time.perf_counter()
                Kt = np.nan_to_num(kern.evaluate(X_train, X_train), nan=0.0, posinf=1.0, neginf=0.0)
                Kte = np.nan_to_num(kern.evaluate(X_test, X_train), nan=0.0, posinf=1.0, neginf=0.0)
                kernel_time += time.perf_counter() - k0
                svc = cuSVC(kernel='precomputed', class_weight='balanced',
                            random_state=seed, cache_size=8192.0,
                            max_iter=10000, nochange_steps=100, output_type='numpy')
                c0 = time.perf_counter()
                svc.fit(Kt, y_train)
                test_probas.append(cuml_svc_predict_proba(svc, Kte))
                classifier_time += time.perf_counter() - c0
            weights = np.asarray(validation_scores, dtype=np.float64)
            weights = weights / weights.sum()
            avg_proba = np.average(np.asarray(test_probas), axis=0, weights=weights)
            y_pred = np.argmax(avg_proba, axis=1)
            y_proba = avg_proba
        else:
            raise ValueError(f"Unknown model: {model_name}")

        train_time = kernel_time + classifier_time
        metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba=y_proba,
                                        train_time=train_time)
        return {
            'status': 'success',
            'dataset': dataset_name,
            'model': model_name,
            'device': device,
            'entanglement': ent_type,
            'optimization_level': opt_level,
            'num_qubits': num_qubits,
            'run_idx': run_idx,
            'seed': seed,
            'gpu_id': gpu_id,
            'aggregate_depth': agg_depth,
            'aggregate_two_qubit_gates': agg_2q,
            'mean_readout_error': mean_readout,
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'f1_score': metrics['f1_score'],
            'mcc': metrics['mcc'],
            'roc_auc': metrics['roc_auc'],
            'g_mean': metrics['g_mean'],
            'kernel_time': kernel_time,
            'classifier_time': classifier_time,
            'total_time': train_time,
        }
    except Exception as exc:  # noqa: BLE001 - record failure, keep batch alive
        logger.warning(f"Run failed (task {task_id}, {model_name}/{ent_type}/{device}, "
                       f"seed={seed}): {exc}")
        return {
            'status': 'failed', 'dataset': dataset_name, 'model': model_name,
            'device': device, 'entanglement': ent_type, 'optimization_level': opt_level,
            'run_idx': run_idx, 'seed': seed, 'error': str(exc),
        }
    finally:
        gc.collect()


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def _ci95(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(n))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ['dataset', 'model', 'entanglement', 'optimization_level', 'device', 'num_qubits']
    for key, g in df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row['n_runs'] = len(g)
        if 'mean_readout_error' in g:
            _ro = g['mean_readout_error'].to_numpy(dtype=float)
            row['mean_readout_error'] = float(np.nanmean(_ro)) if not np.all(np.isnan(_ro)) else float('nan')
        else:
            row['mean_readout_error'] = float('nan')
        row['aggregate_depth'] = float(np.mean(g['aggregate_depth'].to_numpy(dtype=float))) \
            if 'aggregate_depth' in g else float('nan')
        for metric in ['accuracy', 'balanced_accuracy', 'f1_score', 'mcc', 'roc_auc',
                       'g_mean', 'total_time']:
            vals = g[metric].to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(np.mean(vals))
            row[f'{metric}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{metric}_ci95'] = _ci95(vals)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def build_tasks(datasets, data_cache, num_qubits, entanglements, opt_levels,
                devices, models, seeds) -> List[Tuple]:
    tasks = []
    tid = 0
    for dataset_name, _ in datasets:
        X, y = data_cache[dataset_name]
        for model_name in models:
            for ent_type in entanglements:
                for opt_level in opt_levels:
                    for device in devices:
                        for run_idx, seed in enumerate(seeds):
                            tasks.append((
                                tid, dataset_name, X, y, num_qubits, ent_type,
                                opt_level, device, model_name, seed, run_idx,
                            ))
                            tid += 1
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Device-calibrated noise validation")
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset CSV paths')
    parser.add_argument('--num_qubits', type=int, default=6)
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--entanglements', nargs='+', default=['full', 'linear'])
    parser.add_argument('--opt_levels', nargs='+', type=int, default=[0, 3])
    parser.add_argument('--devices', nargs='+',
                        default=['ideal', 'FakeKolkataV2'],
                        help="'ideal' and/or IBM fake backend names (>= num_qubits qubits)")
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--models', nargs='+', default=['QSVC', 'QVE', 'QWE'])
    parser.add_argument('--max_gpus', type=int, default=3)
    parser.add_argument('--session_id', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny correctness test (sample_size=300, n_runs=1)')
    args = parser.parse_args()

    if args.smoke:
        args.sample_size = min(args.sample_size, 300)
        args.n_runs = 1
        args.entanglements = ['full', 'linear']
        args.opt_levels = [0]
        logger.info("SMOKE TEST mode: sample_size=300, n_runs=1, opt=[0]")

    session_id = args.session_id or f"devnoise_{time.strftime('%Y%m%d_%H%M%S')}"
    seeds = ALL_SEEDS[:args.n_runs]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = []
    for ds in args.datasets:
        if Path(ds).exists():
            datasets.append((Path(ds).stem, ds))
        else:
            logger.warning(f"Dataset not found, skipping: {ds}")
    if not datasets:
        logger.error("No valid datasets. Aborting.")
        return

    # Validate device qubit counts up front (fail fast with a clear message).
    for device in args.devices:
        if device == 'ideal':
            continue
        if not hasattr(fake_provider, device):
            logger.error(f"Unknown fake backend '{device}'. Aborting.")
            return
        nq = getattr(fake_provider, device)().num_qubits
        if nq < args.num_qubits:
            logger.error(f"Backend {device} has {nq} qubits < required {args.num_qubits}. Aborting.")
            return

    logger.info("=" * 70)
    logger.info("DEVICE-CALIBRATED NOISE VALIDATION")
    logger.info(f"  session       : {session_id}")
    logger.info(f"  datasets      : {[d[0] for d in datasets]}")
    logger.info(f"  qubits        : {args.num_qubits}  samples: {args.sample_size}")
    logger.info(f"  entanglements : {args.entanglements}  opt_levels: {args.opt_levels}")
    logger.info(f"  devices       : {args.devices}")
    logger.info(f"  models        : {args.models}  runs: {args.n_runs}")
    logger.info("=" * 70)

    # Report device profiles (gate basis + mean readout error context).
    for device in args.devices:
        if device == 'ideal':
            continue
        _, basis, mean_ro = get_device_profile(device)
        logger.info(f"  device {device}: basis={basis}  mean_readout_error={mean_ro:.4f} "
                    f"(contextual; not applied by measurement-free kernel)")

    data_cache = {}
    for dataset_name, dataset_path in datasets:
        X, y = load_dataset(dataset_path, args.num_qubits, args.sample_size)
        data_cache[dataset_name] = (X, y)
        logger.info(f"  loaded {dataset_name}: X={X.shape}, "
                    f"class balance={np.bincount(y).tolist()}")

    tasks = build_tasks(
        datasets, data_cache, args.num_qubits, args.entanglements,
        args.opt_levels, args.devices, args.models, seeds,
    )

    gpu_mgr = get_gpu_manager()
    num_workers = max(1, min(args.max_gpus, gpu_mgr.gpu_count or 1))
    logger.info(f"Dispatching {len(tasks)} runs across {num_workers} GPU worker(s)...")

    all_results = []
    completed = 0
    t_start = time.perf_counter()
    runs_csv = RESULTS_DIR / f'device_noise_runs_{session_id}.csv'

    for batch_start in range(0, len(tasks), num_workers):
        batch = tasks[batch_start:batch_start + num_workers]
        batch_results = run_parallel_on_gpus(
            batch, execute_device_run,
            num_workers=num_workers, timeout_per_task=7200,
        )
        for r in batch_results:
            if isinstance(r, dict):
                all_results.append(r)
        completed += len(batch)
        pd.DataFrame(all_results).to_csv(runs_csv, index=False)
        elapsed = time.perf_counter() - t_start
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = (len(tasks) - completed) / rate if rate > 0 else float('inf')
        logger.info(f"  progress {completed}/{len(tasks)} "
                    f"({elapsed/60:.1f} min elapsed, ETA {eta/60:.1f} min)")

    runs_df = pd.DataFrame(all_results)
    ok_df = runs_df[runs_df['status'] == 'success'].copy() if 'status' in runs_df else runs_df
    logger.info(f"Completed: {len(ok_df)}/{len(tasks)} successful runs")

    if ok_df.empty:
        logger.error("No successful runs; skipping aggregation.")
        return

    summary_df = summarize(ok_df)
    summary_csv = RESULTS_DIR / f'device_noise_summary_{session_id}.csv'
    summary_df.to_csv(summary_csv, index=False)

    logger.info("=" * 70)
    logger.info("DEVICE NOISE VALIDATION COMPLETE")
    logger.info(f"  per-run : {runs_csv}")
    logger.info(f"  summary : {summary_csv}")
    logger.info("=" * 70)

    # Console digest: ideal vs device accuracy/MCC per (dataset, model, ent).
    try:
        for (ds, model, ent), g in summary_df.groupby(['dataset', 'model', 'entanglement']):
            parts = []
            for _, row in g.sort_values(['device', 'optimization_level']).iterrows():
                parts.append(f"{row['device']}/L{int(row['optimization_level'])}="
                             f"{row['accuracy_mean']*100:.2f}(MCC{row['mcc_mean']:.2f})")
            logger.info(f"  {ds} | {model}/{ent}: " + ', '.join(parts))
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
