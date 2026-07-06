#!/usr/bin/env python3
"""
E3 — Shot sensitivity of the compute-uncompute hardware estimator
================================================================

The real-hardware section estimates the fidelity kernel with the
compute-uncompute protocol at 4096 shots. Unlike the deterministic
density-matrix kernel used in the primary study, this estimator is stochastic,
so a reviewer will ask whether 4096 shots is enough. This harness quantifies the
convergence directly: under the Heron-generation FakeFez noise model it rebuilds
the compute-uncompute train Gram matrix at a range of shot counts and measures

  * rel_fro_err : ||K_shots - K_ref||_F / ||K_ref||_F, where K_ref is a
                  high-shot anchor of the SAME FakeFez noisy kernel. This
                  isolates sampling (shot) noise from the decoherence bias that
                  separates the noisy kernel from the ideal one; it should fall
                  ~1/sqrt(shots).
  * bias_vs_ideal : ||K_ref - K_ideal||_F / ||K_ideal||_F, the shot-free
                  decoherence bias of the noisy kernel relative to the exact
                  statevector Gram (constant in shots, for context).
  * accuracy / mcc of the downstream SVC at each shot count.

This shows the estimator is converged (or how far from converged) at the 4096
shots used on device, justifying the hardware protocol.

Reuses the exact hardware primitives from run_hardware_kernel.py.

Outputs (results/hardware/):
    shot_sensitivity_runs_<session>.csv
    shot_sensitivity_summary_<session>.csv
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from run_hardware_kernel import (
    make_feature_map,
    ideal_gram,
    ideal_rect,
    gram_matrix,
    rect_matrix,
    make_fake_backend_sampler,
    load_dataset,
    balanced_subsample,
)
from circuit_depth_experiment import DataProcessor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("shot_sensitivity")

DEFAULT_SHOTS = [256, 512, 1024, 2048, 4096, 8192]
ANCHOR_SHOTS = 32768  # high-shot proxy for the shot-free noisy-kernel expectation
SEEDS = [42, 123, 456]
DATASETS = {
    'IoT': 'IoT_Original_Distribution.csv',
    'UNSW-2018': 'UNSW_2018_IoT_Botnet_Final_10_Best.csv',
    'UNSW-NB15': 'UNSW_NB15.csv',
}
# QSVC = single ZZ full map (cleanest single-kernel convergence signal).
MAP = ('ZZ', 2, 'full')


def prep(dataset_path: str, nq: int, train_size: int, test_size: int, seed: int):
    X, y = load_dataset(dataset_path, nq, sample_size=5000)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    X_tr_raw, y_tr = balanced_subsample(X_tr_raw, y_tr, train_size, seed)
    X_te_raw, y_te = balanced_subsample(X_te_raw, y_te, test_size, seed + 1)
    pre = DataProcessor(num_qubits=nq, random_seed=seed)
    pre.fit(X_tr_raw, y_tr)
    return pre.transform(X_tr_raw), y_tr, pre.transform(X_te_raw), y_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--num-qubits', type=int, default=6)
    ap.add_argument('--train-size', type=int, default=40)
    ap.add_argument('--test-size', type=int, default=20)
    ap.add_argument('--opt-level', type=int, default=3)
    ap.add_argument('--backend', default='FakeFez')
    ap.add_argument('--shots', nargs='*', type=int, default=DEFAULT_SHOTS)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS.keys()))
    args = ap.parse_args()

    seeds = SEEDS[:args.seeds]
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('results/hardware')
    out_dir.mkdir(parents=True, exist_ok=True)
    nq = args.num_qubits
    map_type, reps, ent = MAP

    records: List[Dict[str, Any]] = []
    for ds_name in args.datasets:
        path = str(Path(args.data_dir) / DATASETS[ds_name])
        for seed in seeds:
            X_tr, y_tr, X_te, y_te = prep(path, nq, args.train_size, args.test_size, seed)
            fm = make_feature_map(nq, map_type, reps, ent)

            # Exact noiseless reference (statevector) for the decoherence-bias term.
            K_id = ideal_gram(fm, X_tr)
            fro_id = float(np.linalg.norm(K_id, 'fro'))

            # High-shot anchor of the SAME noisy kernel = shot-free expectation proxy.
            aargs = SimpleNamespace(backend=args.backend, shots=ANCHOR_SHOTS)
            abackend, asampler = make_fake_backend_sampler(aargs)
            apm = generate_preset_pass_manager(optimization_level=args.opt_level,
                                               backend=abackend)
            K_ref = gram_matrix(fm, X_tr, apm, asampler, 'fake', ANCHOR_SHOTS, nq)
            fro_ref = float(np.linalg.norm(K_ref, 'fro'))
            bias_vs_ideal = float(np.linalg.norm(K_ref - K_id, 'fro') / fro_id) if fro_id > 0 else float('nan')

            for shots in args.shots:
                sargs = SimpleNamespace(backend=args.backend, shots=shots)
                backend, sampler = make_fake_backend_sampler(sargs)
                pm = generate_preset_pass_manager(optimization_level=args.opt_level,
                                                  backend=backend)
                K_s = gram_matrix(fm, X_tr, pm, sampler, 'fake', shots, nq)
                R_s = rect_matrix(fm, X_te, X_tr, pm, sampler, 'fake', shots, nq)
                rel_fro = float(np.linalg.norm(K_s - K_ref, 'fro') / fro_ref) if fro_ref > 0 else float('nan')
                mae = float(np.mean(np.abs(K_s - K_ref)))
                svc = SVC(kernel='precomputed', class_weight='balanced', random_state=seed)
                svc.fit(K_s, y_tr)
                y_pred = svc.predict(R_s)
                rec = {'dataset': ds_name, 'map': f'{map_type}-{ent}', 'shots': shots,
                       'seed': seed, 'rel_fro_err': rel_fro, 'kernel_mae': mae,
                       'bias_vs_ideal': bias_vs_ideal,
                       'accuracy': float(accuracy_score(y_te, y_pred)),
                       'mcc': float(matthews_corrcoef(y_te, y_pred))}
                records.append(rec)
                logger.info(f"  {ds_name} shots={shots} seed={seed}: "
                            f"rel_fro_err={rel_fro:.4f} bias={bias_vs_ideal:.3f} "
                            f"acc={rec['accuracy']:.3f}")

    runs_df = pd.DataFrame(records)
    runs_path = out_dir / f'shot_sensitivity_runs_{session}.csv'
    runs_df.to_csv(runs_path, index=False)
    rows = []
    for (ds, shots), g in runs_df.groupby(['dataset', 'shots'], sort=True):
        row = {'dataset': ds, 'shots': shots, 'n_runs': len(g)}
        for m in ['rel_fro_err', 'kernel_mae', 'bias_vs_ideal', 'accuracy', 'mcc']:
            v = g[m].to_numpy(float)
            row[f'{m}_mean'] = float(np.mean(v))
            row[f'{m}_ci95'] = float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / f'shot_sensitivity_summary_{session}.csv', index=False)
    logger.info(f"Saved runs -> {runs_path}")
    logger.info("Summary:\n" + summary_df[['dataset', 'shots', 'rel_fro_err_mean',
                                            'accuracy_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
