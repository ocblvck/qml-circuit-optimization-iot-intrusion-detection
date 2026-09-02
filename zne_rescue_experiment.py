#!/usr/bin/env python3
"""
Zero-noise extrapolation (ZNE) test on the 10-qubit UNSW-NB15 kernel collapse.

The paper shows that removing readout error entirely does not rescue the
collapsed full-entanglement QSVC kernel under FakeSherbrooke, and argues
(citing Thanasilp et al.) that gate-error mitigation would need exponential
resources. This harness tests the most common gate-error mitigation directly:
global unitary folding ZNE on the compute-uncompute fidelity estimator.

For every kernel entry K_ij = P(all zeros) of U(x_j)^dag U(x_i)|0>, the
transpiled (ISA) circuit C is executed at noise-scale factors s in {1, 3, 5}
by running C (C^dag C)^((s-1)/2) at the physical level, and the entry is
extrapolated to s = 0 (linear and Richardson fits), clipped to [0, 1]. The
extrapolated Gram matrices are then used exactly like the raw ones (sklearn
SVC, precomputed kernel, balanced class weights), so the comparison is
raw (s=1) vs ZNE-linear vs ZNE-Richardson, for QSVC (ZZ) and QVE (Z + ZZ, hard
vote with soft tie-break), on the same balanced 30-train / 16-test subsample and
seeds as mitigation_rescue_experiment.py. We also report Gram-level diagnostics
(off-diagonal mean/variance, Frobenius distance to the exact statevector kernel)
to show whether ZNE moves the kernel back toward the noiseless one.

Same-transpilation note: the base circuit is transpiled once at optimization
level 3 to the backend (error-aware layout); the folded circuits are only
re-translated to the native basis (level 0, trivial layout, no routing), so all
scale factors share the same physical qubits and gate structure.

Outputs (results/hardware/):
    zne_rescue_runs_<session>.csv, zne_rescue_summary_<session>.csv
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from run_hardware_kernel import (
    make_feature_map, load_dataset, balanced_subsample, ideal_gram, ideal_rect,
)
from circuit_depth_experiment import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("zne_rescue")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(); _h.setFormatter(logging.Formatter("%(asctime)s [ZNE] %(message)s", "%H:%M:%S")); logger.addHandler(_h)
logger.propagate = False

DATASETS = {'IoT': 'data/IoT_Original_Distribution.csv',
            'UNSW-2018': 'data/UNSW_2018_IoT_Botnet_Final_10_Best.csv',
            'UNSW-NB15': 'data/UNSW_NB15.csv'}


def make_backend_and_sampler(fake_name: str, shots: int, seed: int):
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider as fp
    fake = getattr(fp, fake_name)()
    nm = NoiseModel.from_backend(fake)
    sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    sampler = AerSamplerV2.from_backend(sim, default_shots=shots)
    return fake, sampler


def logical_fidelity_circuit(fm, a, b) -> QuantumCircuit:
    ua = fm.assign_parameters(a)
    ub_inv = fm.assign_parameters(b).inverse()
    qc = ua.compose(ub_inv)
    qc.measure_all()
    return qc


def fold_isa(isa: QuantumCircuit, scale: int, pm0) -> QuantumCircuit:
    """Global folding at the physical level: C (C^dag C)^k, k=(scale-1)/2, measurements re-appended."""
    if scale == 1:
        return isa
    k = (scale - 1) // 2
    body = isa.remove_final_measurements(inplace=False)
    meas = [(inst.qubits, inst.clbits) for inst in isa.data if inst.operation.name == 'measure']
    folded = body.copy()
    inv = body.inverse()
    for _ in range(k):
        folded.compose(inv, inplace=True)
        folded.compose(body, inplace=True)
    out = QuantumCircuit(*isa.qregs, *isa.cregs)
    out.compose(folded, inplace=True)
    for qs, cs in meas:
        out.measure(qs, cs)
    return pm0.run(out)  # translate sxdg etc. into the native basis, no relayout


def all_zero_probs(sampler, circuits: List[QuantumCircuit], nq: int, shots: int, batch: int = 200) -> np.ndarray:
    out = []
    for i in range(0, len(circuits), batch):
        res = sampler.run(circuits[i:i + batch]).result()
        for r in res:
            counts = r.data.meas.get_counts()
            out.append(counts.get('0' * nq, 0) / shots)
    return np.asarray(out)


def estimate_kernels(fm, X_tr, X_te, pm3, pm0, sampler, nq, shots, scales) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {scale: (K_train, K_test)} of P(all zeros) estimates."""
    n = len(X_tr)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    tr_circs = [logical_fidelity_circuit(fm, X_tr[i], X_tr[j]) for i, j in pairs]
    te_circs = [logical_fidelity_circuit(fm, a, b) for a in X_te for b in X_tr]
    isa_tr = pm3.run(tr_circs)
    isa_te = pm3.run(te_circs)
    logger.info(f"    base ISA depth (train pair 0): {isa_tr[0].depth()}")
    out = {}
    for s in scales:
        f_tr = [fold_isa(c, s, pm0) for c in isa_tr]
        f_te = [fold_isa(c, s, pm0) for c in isa_te]
        logger.info(f"    scale {s}: depth {f_tr[0].depth()}, running {len(f_tr) + len(f_te)} circuits")
        v_tr = all_zero_probs(sampler, f_tr, nq, shots)
        v_te = all_zero_probs(sampler, f_te, nq, shots)
        K = np.eye(n)
        for (i, j), v in zip(pairs, v_tr):
            K[i, j] = K[j, i] = v
        out[s] = (K, v_te.reshape(len(X_te), n))
    return out


def extrapolate(vals_by_scale: Dict[int, np.ndarray], method: str) -> np.ndarray:
    scales = np.array(sorted(vals_by_scale), dtype=float)
    Y = np.stack([vals_by_scale[int(s)] for s in scales], axis=0)  # (S, ...)
    if method == 'linear':
        A = np.vstack([np.ones_like(scales), scales]).T
        coef, *_ = np.linalg.lstsq(A, Y.reshape(len(scales), -1), rcond=None)
        z = coef[0].reshape(Y.shape[1:])
    elif method == 'richardson':
        # exact polynomial of degree S-1 through the points, evaluated at 0
        z = np.zeros(Y.shape[1:])
        for i, si in enumerate(scales):
            w = np.prod([sj / (sj - si) for j, sj in enumerate(scales) if j != i])
            z = z + w * Y[i]
    else:
        raise ValueError(method)
    return np.clip(z, 0.0, 1.0)


def gram_diag(K: np.ndarray, K_ideal: np.ndarray) -> Dict[str, float]:
    iu = np.triu_indices(len(K), 1)
    off = K[iu]
    return {'offdiag_mean': float(off.mean()), 'offdiag_var': float(off.var()),
            'frob_rel_err_vs_ideal': float(np.linalg.norm(K - K_ideal) / np.linalg.norm(K_ideal))}


def fit_predict(Ktr, ytr, Kte, seed):
    svc = SVC(kernel='precomputed', class_weight='balanced', probability=True, random_state=seed)
    svc.fit(Ktr, ytr)
    return svc.predict(Kte), svc.predict_proba(Kte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-qubits', type=int, default=10)
    ap.add_argument('--backend', default='FakeSherbrooke')
    ap.add_argument('--train-size', type=int, default=30)
    ap.add_argument('--test-size', type=int, default=16)
    ap.add_argument('--shots', type=int, default=2048)
    ap.add_argument('--scales', nargs='+', type=int, default=[1, 3, 5])
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--datasets', nargs='*', default=['UNSW-NB15'])
    args = ap.parse_args()

    seeds = [42, 123, 456][:args.seeds]
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('results/hardware'); out_dir.mkdir(parents=True, exist_ok=True)
    nq = args.num_qubits
    records = []
    for ds_name in args.datasets:
        path = DATASETS[ds_name]
        for seed in seeds:
            X, y = load_dataset(path, nq, sample_size=5000)
            X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            X_tr_raw, y_tr = balanced_subsample(X_tr_raw, y_tr, args.train_size, seed)
            X_te_raw, y_te = balanced_subsample(X_te_raw, y_te, args.test_size, seed + 1)
            pre = DataProcessor(num_qubits=nq, random_seed=seed)
            pre.fit(X_tr_raw, y_tr)
            X_tr, X_te = pre.transform(X_tr_raw), pre.transform(X_te_raw)

            fake, sampler = make_backend_and_sampler(args.backend, args.shots, seed)
            pm3 = generate_preset_pass_manager(optimization_level=3, backend=fake, seed_transpiler=42)
            pm0 = generate_preset_pass_manager(optimization_level=0, backend=fake,
                                               layout_method='trivial', routing_method='none')
            kernels = {}
            for map_type, reps in [('Z', 1), ('ZZ', 2)]:
                fm = make_feature_map(nq, map_type, reps, 'full')
                logger.info(f"{ds_name} seed={seed} map={map_type}: estimating at scales {args.scales}")
                est = estimate_kernels(fm, X_tr, X_te, pm3, pm0, sampler, nq, args.shots, args.scales)
                K_id_tr, K_id_te = ideal_gram(fm, X_tr), ideal_rect(fm, X_te, X_tr)
                variants = {'raw': (est[1][0], est[1][1])}
                for m in ('linear', 'richardson'):
                    Ktr = extrapolate({s: est[s][0] for s in est}, m)
                    Kte = extrapolate({s: est[s][1] for s in est}, m)
                    np.fill_diagonal(Ktr, 1.0)
                    variants[f'zne_{m}'] = (Ktr, Kte)
                variants['ideal'] = (K_id_tr, K_id_te)
                kernels[map_type] = (variants, K_id_tr)

            for cond in ('ideal', 'raw', 'zne_linear', 'zne_richardson'):
                # QSVC = ZZ alone
                Ktr, Kte = kernels['ZZ'][0][cond]
                yp, _ = fit_predict(Ktr, y_tr, Kte, seed)
                d = gram_diag(Ktr, kernels['ZZ'][1])
                records.append({'dataset': ds_name, 'seed': seed, 'model': 'QSVC', 'condition': cond,
                                'accuracy': accuracy_score(y_te, yp), 'mcc': matthews_corrcoef(y_te, yp), **d})
                # QVE = Z + ZZ hard vote, ties by mean proba
                preds, probas = [], []
                for mt in ('Z', 'ZZ'):
                    Ktr, Kte = kernels[mt][0][cond]
                    p, pr = fit_predict(Ktr, y_tr, Kte, seed)
                    preds.append(p); probas.append(pr)
                P = np.array(preds); votes = P.mean(0); mp = np.mean(probas, 0)
                yv = (votes > 0.5).astype(int)
                tie = np.isclose(votes, 0.5); yv[tie] = (mp[tie] >= 0.5).astype(int)
                records.append({'dataset': ds_name, 'seed': seed, 'model': 'QVE', 'condition': cond,
                                'accuracy': accuracy_score(y_te, yv), 'mcc': matthews_corrcoef(y_te, yv),
                                **gram_diag(kernels['ZZ'][0][cond][0], kernels['ZZ'][1])})
                logger.info(f"  {ds_name} seed={seed} {cond}: QSVC acc={records[-2]['accuracy']:.3f} "
                            f"mcc={records[-2]['mcc']:.3f} | QVE acc={records[-1]['accuracy']:.3f} "
                            f"mcc={records[-1]['mcc']:.3f} | ZZ offdiag mean={d['offdiag_mean']:.3f} "
                            f"var={d['offdiag_var']:.2e} frob_err={d['frob_rel_err_vs_ideal']:.3f}")
            pd.DataFrame(records).to_csv(out_dir / f'zne_rescue_runs_{session}.csv', index=False)

    runs = pd.DataFrame(records)
    rows = []
    for (ds, model, cond), g in runs.groupby(['dataset', 'model', 'condition'], sort=True):
        row = {'dataset': ds, 'model': model, 'condition': cond, 'n_runs': len(g)}
        for m in ['accuracy', 'mcc', 'offdiag_mean', 'offdiag_var', 'frob_rel_err_vs_ideal']:
            v = g[m].to_numpy(float); row[f'{m}_mean'] = v.mean(); row[f'{m}_std'] = v.std(ddof=1) if len(v) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f'zne_rescue_summary_{session}.csv', index=False)
    pd.set_option('display.width', 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
