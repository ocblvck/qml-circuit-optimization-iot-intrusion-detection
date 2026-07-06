#!/usr/bin/env python3
"""
Feature-Map Count Ablation for QVE / QWE Quantum Ensembles
==========================================================

Research question
-----------------
The main study (circuit_depth_experiment.py) builds the two proposed ensembles
from exactly TWO heterogeneous quantum feature maps each:

    QVE = { ZZ(reps=2), Z(reps=1) }      (majority / voting ensemble)
    QWE = { ZZ(reps=2), Pauli(reps=1) }  (validation-weighted soft ensemble)

This harness asks whether adding MORE than two heterogeneous encodings improves
robustness/accuracy. It compares, for each model, three compositions:

    2map  -> the model's existing pair                (baseline = "what I have")
    3map  -> { Z(1), ZZ(2), Pauli(1) }
    4map  -> { Z(1), ZZ(2), Pauli(1), Custom(1) }

A useful side effect: with an ODD number of base learners (3map), QVE's
majority vote is well-defined with no ties — resolving the ill-posed 2-learner
majority vote used previously.

Design (kept deliberately tractable — days, not weeks)
------------------------------------------------------
* Reuses the EXACT primitives from circuit_depth_experiment.py (kernels,
  DataProcessor, cuSVC configuration, transpilation, multi-GPU dispatch) so the
  numbers are directly comparable to the main study. Nothing in the main script
  is modified.
* Default scope: 6 qubits, full entanglement, optimization level 0,
  noise grid {0.0, 0.002, 0.01, 0.05}, 10 paired runs, all three datasets.
* GPU acceleration is mandatory (inherited from the imported kernels).

Outputs (written to results/circuit_depth/)
-------------------------------------------
    ablation_feature_map_runs_<session>.csv     per-run metrics
    ablation_feature_map_summary_<session>.csv  mean/std/CI per composition
    ablation_feature_map_pairwise_<session>.csv paired t-tests (Holm-corrected)
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
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from qiskit.transpiler import CouplingMap

# --- Reuse the exact building blocks from the main study (no modification) ---
from circuit_depth_experiment import (
    NOISE_PARAMS,
    RESULTS_DIR,
    DataProcessor,
    GPUFidelityKernel,
    NoiseModelSimulator,
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
logger = logging.getLogger("feature_map_ablation")

BASIS_GATES = ['u', 'cx', 'rz', 'sx', 'x']

# Seeds mirror circuit_depth_experiment.py for run-paired comparability.
ALL_SEEDS = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
    17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624,
]

# Composition definitions: list of (map_type, reps). Order is irrelevant to
# voting. The 2map entry is the model's EXISTING pair ("what I have").
COMPOSITIONS: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    'QVE': {
        '2map': [('Z', 1), ('ZZ', 2)],
        '3map': [('Z', 1), ('ZZ', 2), ('Pauli', 1)],
        '4map': [('Z', 1), ('ZZ', 2), ('Pauli', 1), ('Custom', 1)],
    },
    'QWE': {
        '2map': [('ZZ', 2), ('Pauli', 1)],
        '3map': [('Z', 1), ('ZZ', 2), ('Pauli', 1)],
        '4map': [('Z', 1), ('ZZ', 2), ('Pauli', 1), ('Custom', 1)],
    },
}


# ----------------------------------------------------------------------------
# Data loading (mirrors circuit_depth_experiment._detect_csv_delimiter / prepare)
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


def load_dataset(dataset_path: str, num_qubits: int, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    delimiter = _detect_csv_delimiter(dataset_path)
    df = pd.read_csv(dataset_path, sep=delimiter, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]
    processor = DataProcessor(num_qubits=num_qubits)
    X, y = processor.prepare_data(df, sample_size=sample_size)
    return X, y


# ----------------------------------------------------------------------------
# Per-run worker (executed by run_parallel_on_gpus; gpu_id appended by dispatcher)
# ----------------------------------------------------------------------------
def execute_ablation_run(
    task_id: int,
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    num_qubits: int,
    ent_type: str,
    opt_level: int,
    noise_level: float,
    model_name: str,
    composition_name: str,
    map_specs: List[Tuple[str, int]],
    seed: int,
    run_idx: int,
    gpu_id: int,
) -> Dict[str, Any]:
    """One paired run for a single (dataset, model, composition, noise) cell."""
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
        processor.fit(X_train_raw, y_train)
        X_train = processor.transform(X_train_raw)
        X_test = processor.transform(X_test_raw)

        # Noise model (None for the ideal/statevector case)
        noise_model = None
        if noise_level > 0:
            noise_params = NOISE_PARAMS.copy()
            noise_params['single_qubit_error'] = noise_level
            noise_params['two_qubit_error'] = noise_level * 3
            noise_model = NoiseModelSimulator(params=noise_params).create_noise_model(num_qubits)

        linear_coupling = CouplingMap.from_line(num_qubits)

        def _build_kernel(feature_map):
            if noise_level == 0:
                return GPUFidelityKernel(feature_map, gpu_id=gpu_id, assume_pretranspiled=True)
            return NoisyFidelityKernel(feature_map, noise_model, gpu_id=gpu_id)

        # Build + transpile each component feature map, then its kernel.
        kernels = []
        for map_type, reps in map_specs:
            base_fm = create_feature_map(num_qubits, map_type, reps=reps, entanglement=ent_type)
            cache_key = (
                'ablation', map_type, reps, num_qubits, ent_type,
                opt_level, _normalize_coupling_map_key(linear_coupling),
            )
            fm = transpile_with_cache(
                base_fm, cache_key,
                optimization_level=opt_level, basis_gates=BASIS_GATES,
                coupling_map=linear_coupling, seed_transpiler=42,
            )
            kernels.append(_build_kernel(fm))

        kernel_time = 0.0
        classifier_time = 0.0

        if model_name == 'QVE':
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

            pred_matrix = np.array(preds).T            # (n_test, n_clf)
            avg_proba = np.mean(np.asarray(probas), axis=0)
            final = []
            for i, row in enumerate(pred_matrix):
                ranking = Counter(row).most_common()
                if len(ranking) > 1 and ranking[0][1] == ranking[1][1]:
                    final.append(int(np.argmax(avg_proba[i])))  # tie-break
                else:
                    final.append(int(ranking[0][0]))
            y_pred = np.array(final)

        elif model_name == 'QWE':
            # Validation-accuracy-weighted soft voting (generalized to N maps).
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
        else:
            raise ValueError(f"Unknown model: {model_name}")

        train_time = kernel_time + classifier_time
        metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
        return {
            'status': 'success',
            'dataset': dataset_name,
            'model': model_name,
            'composition': composition_name,
            'n_maps': len(map_specs),
            'entanglement': ent_type,
            'optimization_level': opt_level,
            'noise_level': noise_level,
            'num_qubits': num_qubits,
            'run_idx': run_idx,
            'seed': seed,
            'gpu_id': gpu_id,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'mcc': metrics['mcc'],
            'kernel_time': kernel_time,
            'classifier_time': classifier_time,
            'total_time': train_time,
        }
    except Exception as exc:  # noqa: BLE001 - record failure, keep batch alive
        logger.warning(f"Run failed (task {task_id}, {model_name}/{composition_name}, "
                       f"noise={noise_level}, seed={seed}): {exc}")
        return {
            'status': 'failed', 'dataset': dataset_name, 'model': model_name,
            'composition': composition_name, 'noise_level': noise_level,
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
    group_cols = ['dataset', 'model', 'composition', 'n_maps', 'noise_level', 'num_qubits']
    for key, g in df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row['n_runs'] = len(g)
        for metric in ['accuracy', 'f1_score', 'mcc', 'total_time']:
            vals = g[metric].to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(np.mean(vals))
            row[f'{metric}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{metric}_ci95'] = _ci95(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def _holm(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    n = len(p_values)
    adjusted = [1.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        corrected = min(1.0, (n - rank) * p_values[idx])
        running = max(running, corrected)
        adjusted[idx] = running
    return adjusted


def pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests between compositions, run-paired by seed, Holm-corrected."""
    rows = []
    p_for_holm = []
    cell_cols = ['dataset', 'model', 'noise_level', 'num_qubits']
    for cell_key, cell in df.groupby(cell_cols, sort=True):
        comps = sorted(cell['composition'].unique())
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a_name, b_name = comps[i], comps[j]
                a = cell[cell['composition'] == a_name][['run_idx', 'accuracy']]
                b = cell[cell['composition'] == b_name][['run_idx', 'accuracy']]
                merged = pd.merge(a, b, on='run_idx', suffixes=('_a', '_b')).sort_values('run_idx')
                if len(merged) < 2:
                    continue
                xa = merged['accuracy_a'].to_numpy(float)
                xb = merged['accuracy_b'].to_numpy(float)
                diff = xb - xa  # b minus a (e.g. 3map - 2map)
                if np.allclose(diff, 0.0):
                    t_stat, p_val = 0.0, 1.0
                else:
                    t_stat, p_val = stats.ttest_rel(xb, xa)
                pooled_sd = np.sqrt((np.var(xa, ddof=1) + np.var(xb, ddof=1)) / 2.0)
                cohens_d = float(np.mean(diff) / pooled_sd) if pooled_sd > 0 else 0.0
                row = dict(zip(cell_cols, cell_key))
                row.update({
                    'comparison': f'{b_name}_vs_{a_name}',
                    'mean_a': float(np.mean(xa)), 'mean_b': float(np.mean(xb)),
                    'mean_diff': float(np.mean(diff)),
                    't_stat': float(t_stat), 'p_value': float(p_val),
                    'cohens_d': cohens_d, 'paired_runs': len(merged),
                })
                rows.append(row)
                p_for_holm.append(float(p_val))
    result = pd.DataFrame(rows)
    if not result.empty:
        result['p_value_holm'] = _holm(p_for_holm)
        result['significant_holm'] = result['p_value_holm'] < 0.05
    return result


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def build_tasks(datasets, data_cache, num_qubits, ent_type, opt_level,
                noise_levels, models, compositions, seeds) -> List[Tuple]:
    tasks = []
    tid = 0
    for dataset_name, _ in datasets:
        X, y = data_cache[dataset_name]
        for model_name in models:
            for comp_name in compositions:
                map_specs = COMPOSITIONS[model_name][comp_name]
                for noise_level in noise_levels:
                    for run_idx, seed in enumerate(seeds):
                        tasks.append((
                            tid, dataset_name, X, y, num_qubits, ent_type,
                            opt_level, noise_level, model_name, comp_name,
                            map_specs, seed, run_idx,
                        ))
                        tid += 1
    return tasks


def main():
    parser = argparse.ArgumentParser(description="QVE/QWE feature-map count ablation")
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Dataset CSV paths')
    parser.add_argument('--num_qubits', type=int, default=6)
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--entanglement', default='full')
    parser.add_argument('--opt_level', type=int, default=0)
    parser.add_argument('--noise_levels', nargs='+', type=float,
                        default=[0.0, 0.002, 0.01, 0.05])
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--models', nargs='+', default=['QVE', 'QWE'])
    parser.add_argument('--compositions', nargs='+', default=['2map', '3map', '4map'])
    parser.add_argument('--max_gpus', type=int, default=3)
    parser.add_argument('--session_id', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny correctness/timing test (overrides scope)')
    args = parser.parse_args()

    if args.smoke:
        args.sample_size = min(args.sample_size, 300)
        args.n_runs = 1
        args.noise_levels = [0.0, 0.01]
        args.compositions = ['2map', '3map', '4map']
        logger.info("SMOKE TEST mode: sample_size=300, n_runs=1, noise=[0.0,0.01]")

    session_id = args.session_id or f"fmablation_{time.strftime('%Y%m%d_%H%M%S')}"
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

    logger.info("=" * 70)
    logger.info("FEATURE-MAP COUNT ABLATION")
    logger.info(f"  session      : {session_id}")
    logger.info(f"  datasets     : {[d[0] for d in datasets]}")
    logger.info(f"  qubits       : {args.num_qubits}  samples: {args.sample_size}")
    logger.info(f"  entanglement : {args.entanglement}  opt_level: {args.opt_level}")
    logger.info(f"  noise levels : {args.noise_levels}")
    logger.info(f"  models       : {args.models}  compositions: {args.compositions}")
    logger.info(f"  runs         : {args.n_runs}  seeds: {seeds}")
    logger.info("=" * 70)

    # Pre-load and prepare each dataset once (shared across threads).
    data_cache = {}
    for dataset_name, dataset_path in datasets:
        X, y = load_dataset(dataset_path, args.num_qubits, args.sample_size)
        data_cache[dataset_name] = (X, y)
        logger.info(f"  loaded {dataset_name}: X={X.shape}, "
                    f"class balance={np.bincount(y).tolist()}")

    tasks = build_tasks(
        datasets, data_cache, args.num_qubits, args.entanglement,
        args.opt_level, args.noise_levels, args.models, args.compositions, seeds,
    )

    gpu_mgr = get_gpu_manager()
    num_workers = max(1, min(args.max_gpus, gpu_mgr.gpu_count or 1))
    logger.info(f"Dispatching {len(tasks)} runs across {num_workers} GPU worker(s)...")

    all_results = []
    completed = 0
    t_start = time.perf_counter()
    runs_csv = RESULTS_DIR / f'ablation_feature_map_runs_{session_id}.csv'

    for batch_start in range(0, len(tasks), num_workers):
        batch = tasks[batch_start:batch_start + num_workers]
        batch_results = run_parallel_on_gpus(
            batch, execute_ablation_run,
            num_workers=num_workers, timeout_per_task=7200,
        )
        for r in batch_results:
            if isinstance(r, dict) and r.get('status') == 'success':
                all_results.append(r)
            elif isinstance(r, dict):
                all_results.append(r)
        completed += len(batch)
        # Incremental checkpoint so a crash never loses completed runs.
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
    pairwise_df = pairwise_tests(ok_df)

    summary_csv = RESULTS_DIR / f'ablation_feature_map_summary_{session_id}.csv'
    pairwise_csv = RESULTS_DIR / f'ablation_feature_map_pairwise_{session_id}.csv'
    summary_df.to_csv(summary_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)

    logger.info("=" * 70)
    logger.info("ABLATION COMPLETE")
    logger.info(f"  per-run   : {runs_csv}")
    logger.info(f"  summary   : {summary_csv}")
    logger.info(f"  pairwise  : {pairwise_csv}")
    logger.info("=" * 70)

    # Console digest: ideal-case accuracy by composition.
    try:
        digest = summary_df[summary_df['noise_level'] == 0.0]
        for (ds, model), g in digest.groupby(['dataset', 'model']):
            line = ', '.join(
                f"{row['composition']}={row['accuracy_mean']:.4f}±{row['accuracy_ci95']:.4f}"
                for _, row in g.sort_values('n_maps').iterrows()
            )
            logger.info(f"  [ideal] {ds} | {model}: {line}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
