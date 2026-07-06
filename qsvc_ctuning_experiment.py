#!/usr/bin/env python3
"""
E2 — QSVC regularisation (C) tuning fairness
===========================================

Reviewer-fairness study. The classical baselines in the manuscript are tuned,
but the quantum SVC head is fixed at C=1.0 (circuit_depth_experiment.py L5448).
A reviewer can object that the quantum-vs-classical-kernel comparison is unfair
because the quantum model was not given the same tuning freedom.

This harness closes that gap cheaply: it computes the ideal (statevector) QSVC
kernel ONCE per (dataset, seed) using the primary ZZ full-entanglement map, then
refits the cuML SVC across a grid of C values (the kernel is reused, so the
quantum work is done only once). It reports, per dataset, the C=1.0 accuracy,
the best-C accuracy, and their difference. A small gap demonstrates that the
fixed C=1.0 does not materially disadvantage the quantum kernel, so the reported
quantum-vs-classical comparison is fair.

Reuses circuit_depth_experiment.py primitives unchanged.

Outputs (results/circuit_depth/):
    ctune_runs_<session>.csv      per (dataset, seed, C) metrics
    ctune_summary_<session>.csv   per (dataset, C) mean/CI + best-C vs C=1.0
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit.transpiler import CouplingMap

from circuit_depth_experiment import (
    RESULTS_DIR,
    DataProcessor,
    GPUFidelityKernel,
    cuSVC,
    calculate_all_metrics,
    create_feature_map,
    transpile_with_cache,
    _normalize_coupling_map_key,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("ctune")

BASIS_GATES = ['u', 'cx', 'rz', 'sx', 'x']
ALL_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
             7168, 8192, 9216, 10240, 11264]
DEFAULT_C = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
DATASETS = {
    'IoT': 'IoT_Original_Distribution.csv',
    'UNSW-2018': 'UNSW_2018_IoT_Botnet_Final_10_Best.csv',
    'UNSW-NB15': 'UNSW_NB15.csv',
}


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


def run_seed(dataset_name: str, X: np.ndarray, y: np.ndarray, num_qubits: int,
             seed: int, c_grid: List[float], gpu_id: int) -> List[Dict[str, Any]]:
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
    processor.fit(X_tr_raw, y_tr)
    X_tr = processor.transform(X_tr_raw)
    X_te = processor.transform(X_te_raw)

    coupling = CouplingMap.from_line(num_qubits)
    base_fm = create_feature_map(num_qubits, 'ZZ', reps=2, entanglement='full')
    cache_key = ('ctune', 'ZZ', 2, num_qubits, 'full', 0,
                 _normalize_coupling_map_key(coupling))
    fm = transpile_with_cache(base_fm, cache_key, optimization_level=0,
                              basis_gates=BASIS_GATES, coupling_map=coupling,
                              seed_transpiler=42)
    kern = GPUFidelityKernel(fm, gpu_id=gpu_id, assume_pretranspiled=True)
    t0 = time.perf_counter()
    Ktr = np.nan_to_num(kern.evaluate(X_tr, X_tr), nan=0.0, posinf=1.0, neginf=0.0)
    Kte = np.nan_to_num(kern.evaluate(X_te, X_tr), nan=0.0, posinf=1.0, neginf=0.0)
    kt = time.perf_counter() - t0

    out = []
    for c in c_grid:
        svc = cuSVC(kernel='precomputed', class_weight='balanced', random_state=seed,
                    C=float(c), cache_size=8192.0, max_iter=10000,
                    nochange_steps=100, output_type='numpy')
        svc.fit(Ktr, y_tr)
        y_pred = np.asarray(svc.predict(Kte))
        m = calculate_all_metrics(y_te, y_pred)
        out.append({'dataset': dataset_name, 'num_qubits': num_qubits, 'seed': seed,
                    'C': float(c), 'accuracy': m['accuracy'], 'mcc': m['mcc'],
                    'f1_score': m['f1_score'], 'kernel_time': kt})
    del Ktr, Kte
    gc.collect()
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, c), g in df.groupby(['dataset', 'C'], sort=True):
        row = {'dataset': ds, 'C': c, 'n_runs': len(g)}
        for metric in ['accuracy', 'mcc', 'f1_score']:
            v = g[metric].to_numpy(float)
            row[f'{metric}_mean'] = float(np.mean(v))
            row[f'{metric}_ci95'] = float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        rows.append(row)
    sdf = pd.DataFrame(rows)
    # Annotate best-C vs C=1.0 per dataset.
    notes = []
    for ds, g in sdf.groupby('dataset'):
        base = g[np.isclose(g['C'], 1.0)]['accuracy_mean']
        base_acc = float(base.iloc[0]) if len(base) else float('nan')
        best_row = g.loc[g['accuracy_mean'].idxmax()]
        notes.append({'dataset': ds, 'acc_C1': base_acc,
                      'acc_bestC': float(best_row['accuracy_mean']),
                      'best_C': float(best_row['C']),
                      'delta_bestC_minus_C1': float(best_row['accuracy_mean']) - base_acc})
    return sdf, pd.DataFrame(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--num-qubits', type=int, default=6)
    ap.add_argument('--sample-size', type=int, default=3000)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--gpu-id', type=int, default=1)
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS.keys()))
    args = ap.parse_args()

    seeds = ALL_SEEDS[:args.seeds]
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for ds_name in args.datasets:
        path = str(Path(args.data_dir) / DATASETS[ds_name])
        logger.info(f"Loading {ds_name} from {path}")
        X, y = load_dataset(path, args.num_qubits, args.sample_size)
        for seed in seeds:
            records.extend(run_seed(ds_name, X, y, args.num_qubits, seed,
                                    DEFAULT_C, args.gpu_id))
        g = pd.DataFrame([r for r in records if r['dataset'] == ds_name])
        base = g[np.isclose(g['C'], 1.0)]['accuracy'].mean()
        best = g.groupby('C')['accuracy'].mean().max()
        logger.info(f"  {ds_name}: acc(C=1.0)={base*100:.2f}%  best-C acc={best*100:.2f}%  "
                    f"gain={100*(best-base):+.2f}pp")

    runs_df = pd.DataFrame(records)
    runs_path = out_dir / f'ctune_runs_{session}.csv'
    runs_df.to_csv(runs_path, index=False)
    sdf, notes = summarize(runs_df)
    sdf.to_csv(out_dir / f'ctune_summary_{session}.csv', index=False)
    notes.to_csv(out_dir / f'ctune_bestC_{session}.csv', index=False)
    logger.info(f"Saved runs -> {runs_path}")
    logger.info("Best-C vs C=1.0:\n" + notes.to_string(index=False))


if __name__ == '__main__':
    main()
