#!/usr/bin/env python3
"""
E1 — Kernel-concentration diagnostic vs noise
=============================================

Mechanism study for the manuscript. The primary study shows that under growing
noise the full-entanglement QSVC kernel collapses toward the majority-class rate
while QVE (which includes the shallow Z map) stays robust. The paper attributes
this to "kernel concentration" (Huang et al. exponential concentration) but never
MEASURES it. This harness quantifies concentration directly so the collapse is
explained rather than asserted.

For each (dataset, feature map, entanglement, noise level, seed) we build the
SAME deterministic density-matrix HS kernel used by the primary study
(NoisyFidelityKernel) and record two concentration diagnostics on the train Gram
matrix:

  * offdiag_var  : variance of the off-diagonal kernel entries. Exponential
                   concentration drives every off-diagonal entry to a common
                   value, so this variance -> 0 as the kernel concentrates.
  * kta          : kernel-target alignment <K, yy^T>_F / (||K||_F ||yy^T||_F)
                   with y in {-1,+1}; measures how much label-relevant geometry
                   survives. Drops toward 0 as the kernel loses discriminative
                   structure.

The scientific claim it supports: the depth-O(1) Z map (QVE's stabilising
component) barely concentrates, whereas the full-entanglement ZZ map (QSVC's
kernel) concentrates sharply once noise grows — exactly the accuracy-collapse
ordering seen in Phase 4.

Reuses the EXACT primitives from circuit_depth_experiment.py so numbers are
comparable to the main study. Nothing in the main script is modified.

Outputs (results/circuit_depth/):
    concentration_runs_<session>.csv      per-run diagnostics
    concentration_summary_<session>.csv   mean/CI per (dataset,map,ent,noise)
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
    NOISE_PARAMS,
    RESULTS_DIR,
    DataProcessor,
    GPUFidelityKernel,
    NoiseModelSimulator,
    NoisyFidelityKernel,
    create_feature_map,
    transpile_with_cache,
    _normalize_coupling_map_key,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("kernel_concentration")

BASIS_GATES = ['u', 'cx', 'rz', 'sx', 'x']

# Seeds mirror circuit_depth_experiment.py for run-paired comparability.
ALL_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# Noise grid mirrors the supplementary/ablation four-level grid plus two more
# intermediate points so the concentration onset is visible.
DEFAULT_NOISE = [0.0, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]

# Feature maps probed. The Z map is QVE's shallow stabiliser; ZZ-full is the
# QSVC kernel that collapses; ZZ-linear and Pauli-full bracket the depth axis.
DEFAULT_MAPS: List[Tuple[str, int, str]] = [
    ('Z', 1, 'linear'),      # depth O(1) — entanglement irrelevant for Z
    ('ZZ', 2, 'linear'),
    ('ZZ', 2, 'full'),       # QSVC primary kernel
    ('Pauli', 1, 'full'),
]

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


def _concentration_metrics(K: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Off-diagonal variance and kernel-target alignment on a train Gram matrix."""
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)
    off = K[iu]
    offdiag_var = float(np.var(off))
    offdiag_mean = float(np.mean(off))
    # Kernel-target alignment with y in {-1,+1}.
    yy = np.where(np.asarray(y) > 0, 1.0, -1.0)
    T = np.outer(yy, yy)
    num = float(np.sum(K * T))
    den = float(np.linalg.norm(K, 'fro') * np.linalg.norm(T, 'fro'))
    kta = num / den if den > 0 else 0.0
    return {'offdiag_var': offdiag_var, 'offdiag_mean': offdiag_mean, 'kta': kta}


def run_cell(dataset_name: str, X: np.ndarray, y: np.ndarray, num_qubits: int,
             map_type: str, reps: int, ent: str, noise_level: float,
             seed: int, gpu_id: int) -> Dict[str, Any]:
    X_tr_raw, _, y_tr, _ = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
    processor.fit(X_tr_raw, y_tr)
    X_tr = processor.transform(X_tr_raw)

    noise_model = None
    if noise_level > 0:
        np_params = NOISE_PARAMS.copy()
        np_params['single_qubit_error'] = noise_level
        np_params['two_qubit_error'] = noise_level * 3
        noise_model = NoiseModelSimulator(params=np_params).create_noise_model(num_qubits)

    coupling = CouplingMap.from_line(num_qubits)
    base_fm = create_feature_map(num_qubits, map_type, reps=reps, entanglement=ent)
    cache_key = ('concentration', map_type, reps, num_qubits, ent, 0,
                 _normalize_coupling_map_key(coupling))
    fm = transpile_with_cache(base_fm, cache_key, optimization_level=0,
                              basis_gates=BASIS_GATES, coupling_map=coupling,
                              seed_transpiler=42)
    if noise_level == 0:
        kern = GPUFidelityKernel(fm, gpu_id=gpu_id, assume_pretranspiled=True)
    else:
        kern = NoisyFidelityKernel(fm, noise_model, gpu_id=gpu_id)

    t0 = time.perf_counter()
    K = np.nan_to_num(kern.evaluate(X_tr, X_tr), nan=0.0, posinf=1.0, neginf=0.0)
    dt = time.perf_counter() - t0
    m = _concentration_metrics(K, y_tr)
    m.update({'dataset': dataset_name, 'map': f'{map_type}-{ent}', 'map_type': map_type,
              'entanglement': ent, 'reps': reps, 'noise_level': noise_level,
              'num_qubits': num_qubits, 'seed': seed, 'kernel_time': dt, 'n_train': len(X_tr)})
    del K
    gc.collect()
    return m


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gcols = ['dataset', 'map', 'map_type', 'entanglement', 'noise_level', 'num_qubits']
    for key, g in df.groupby(gcols, sort=True):
        row = dict(zip(gcols, key))
        row['n_runs'] = len(g)
        for metric in ['offdiag_var', 'offdiag_mean', 'kta']:
            v = g[metric].to_numpy(float)
            row[f'{metric}_mean'] = float(np.mean(v))
            row[f'{metric}_ci95'] = float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--num-qubits', type=int, default=6)
    ap.add_argument('--sample-size', type=int, default=600,
                    help='Subsample; concentration is a statistical property, '
                         'so a few hundred points suffices and keeps the full '
                         'noise-grid density-matrix workload tractable.')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--gpu-id', type=int, default=0)
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
        for (map_type, reps, ent) in DEFAULT_MAPS:
            for noise in DEFAULT_NOISE:
                for seed in seeds:
                    rec = run_cell(ds_name, X, y, args.num_qubits, map_type, reps,
                                   ent, noise, seed, args.gpu_id)
                    records.append(rec)
                agg = [r for r in records if r['dataset'] == ds_name
                       and r['map'] == f'{map_type}-{ent}' and r['noise_level'] == noise]
                logger.info(f"  {ds_name} {map_type}-{ent} noise={noise}: "
                            f"offdiag_var={np.mean([a['offdiag_var'] for a in agg]):.3e} "
                            f"kta={np.mean([a['kta'] for a in agg]):.4f}")

    runs_df = pd.DataFrame(records)
    runs_path = out_dir / f'concentration_runs_{session}.csv'
    runs_df.to_csv(runs_path, index=False)
    summary_df = summarize(runs_df)
    summary_path = out_dir / f'concentration_summary_{session}.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved runs -> {runs_path}")
    logger.info(f"Saved summary -> {summary_path}")


if __name__ == '__main__':
    main()
