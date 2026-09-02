#!/usr/bin/env python3
"""
E4 — Error-mitigation rescue (lean): does readout mitigation save the collapse?
==============================================================================

The 10-qubit device-noise study (paper Table 9) shows the full-entanglement
QSVC kernel collapsing to the majority-class rate under the high-error
FakeSherbrooke profile, while QVE survives. A reviewer will ask whether error
mitigation would rescue QSVC. E1 shows the collapse is exponential kernel
CONCENTRATION, which readout mitigation cannot reverse. This harness demonstrates
that directly and cheaply using an idealized UPPER BOUND on readout mitigation:
we run the compute-uncompute estimator under FakeSherbrooke twice ---

    raw          : full FakeSherbrooke noise (gate + readout errors)
    ro_mitigated : identical gate noise but readout error REMOVED
                   (NoiseModel.from_backend(..., readout_error=False)) --- i.e.
                   PERFECT readout mitigation, the best any readout-mitigation
                   scheme could achieve.

If QSVC still collapses with readout error entirely removed, then no
readout-mitigation method can rescue it: the failure is gate/depth-driven
concentration, and the effective lever is architectural (QVE), not mitigation.

Reuses run_hardware_kernel.py primitives (fit_eval_model, gram/rect, MODEL_MAPS).

Outputs (results/hardware/):
    mitigation_rescue_runs_<session>.csv
    mitigation_rescue_summary_<session>.csv
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
from sklearn.metrics import accuracy_score, matthews_corrcoef
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from run_hardware_kernel import (
    fidelity_circuit,
    gram_matrix,
    rect_matrix,
    fit_eval_model,
    load_dataset,
    balanced_subsample,
)
from circuit_depth_experiment import DataProcessor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("mitigation_rescue")

DATASETS = {
    'IoT': 'data/IoT_Original_Distribution.csv',
    'UNSW-2018': 'data/UNSW_2018_IoT_Botnet_Final_10_Best.csv',
    'UNSW-NB15': 'data/UNSW_NB15.csv',
}


def make_noise_sampler(fake_name: str, readout: bool, shots: int):
    """AerSampler under a fake backend's noise model, optionally with readout
    error removed (readout=False => idealized perfect readout mitigation)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider as fp
    fake = getattr(fp, fake_name)()
    nm = NoiseModel.from_backend(fake, readout_error=readout)
    sim = AerSimulator(noise_model=nm)
    sampler = AerSamplerV2.from_backend(sim, default_shots=shots)
    return fake, sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--num-qubits', type=int, default=10)
    ap.add_argument('--backend', default='FakeSherbrooke')
    ap.add_argument('--train-size', type=int, default=30)
    ap.add_argument('--test-size', type=int, default=16)
    ap.add_argument('--shots', type=int, default=2048)
    ap.add_argument('--opt-level', type=int, default=3)
    ap.add_argument('--models', nargs='*', default=['QSVC', 'QVE'])
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--datasets', nargs='*', default=['UNSW-NB15'])
    args = ap.parse_args()

    seeds = [42, 123, 456][:args.seeds]
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('results/hardware')
    out_dir.mkdir(parents=True, exist_ok=True)
    nq = args.num_qubits

    records: List[Dict[str, Any]] = []
    for ds_name in args.datasets:
        path = str(Path(args.data_dir) / DATASETS[ds_name])
        for seed in seeds:
            X, y = load_dataset(path, nq, sample_size=5000)
            X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y)
            X_tr_raw, y_tr = balanced_subsample(X_tr_raw, y_tr, args.train_size, seed)
            X_te_raw, y_te = balanced_subsample(X_te_raw, y_te, args.test_size, seed + 1)
            pre = DataProcessor(num_qubits=nq, random_seed=seed)
            pre.fit(X_tr_raw, y_tr)
            X_tr, X_te = pre.transform(X_tr_raw), pre.transform(X_te_raw)

            for model in args.models:
                for cond, readout in [('raw', True), ('ro_mitigated', False)]:
                    fake, sampler = make_noise_sampler(args.backend, readout, args.shots)
                    pm = generate_preset_pass_manager(optimization_level=args.opt_level,
                                                      backend=fake)
                    margs = SimpleNamespace(model=model, seed=seed, num_qubits=nq,
                                            entanglement='full')
                    depth_fn = lambda fm: pm.run(fidelity_circuit(fm, X_tr[0], X_tr[1])).depth()  # noqa: E731
                    eval_gram = lambda fm: gram_matrix(fm, X_tr, pm, sampler, 'fake', args.shots, nq)  # noqa: E731
                    eval_rect = lambda fm: rect_matrix(fm, X_te, X_tr, pm, sampler, 'fake', args.shots, nq)  # noqa: E731
                    y_pred, _, _, _ = fit_eval_model(
                        margs, X_tr, y_tr, X_te, y_te, eval_gram, eval_rect, depth_fn)
                    rec = {'dataset': ds_name, 'model': model, 'condition': cond,
                           'backend': args.backend, 'num_qubits': nq, 'seed': seed,
                           'accuracy': float(accuracy_score(y_te, y_pred)),
                           'mcc': float(matthews_corrcoef(y_te, y_pred))}
                    records.append(rec)
                    logger.info(f"  {ds_name} {model} {cond}: acc={rec['accuracy']:.3f} "
                                f"mcc={rec['mcc']:.3f}")

    runs_df = pd.DataFrame(records)
    runs_path = out_dir / f'mitigation_rescue_runs_{session}.csv'
    runs_df.to_csv(runs_path, index=False)
    rows = []
    for (ds, model, cond), g in runs_df.groupby(['dataset', 'model', 'condition'], sort=True):
        row = {'dataset': ds, 'model': model, 'condition': cond, 'n_runs': len(g)}
        for m in ['accuracy', 'mcc']:
            v = g[m].to_numpy(float)
            row[f'{m}_mean'] = float(np.mean(v))
            row[f'{m}_ci95'] = float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / f'mitigation_rescue_summary_{session}.csv', index=False)
    logger.info(f"Saved runs -> {runs_path}")
    logger.info("Summary:\n" + summary_df[['dataset', 'model', 'condition',
                                           'accuracy_mean', 'mcc_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
