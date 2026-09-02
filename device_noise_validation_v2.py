#!/usr/bin/env python3
"""
Device-calibrated noise validation, v2 (layout-aware).

Why a v2
--------
device_noise_validation.py transpiles every feature map onto an ABSTRACT linear
chain over qubits 0..n-1 and then simulates it under NoiseModel.from_backend().
That noise model attaches errors to the backend's PHYSICAL qubit pairs, so:

  * on heavy-hex devices (FakeKolkataV2, FakeCairoV2) chain edges such as (3,4),
    (4,5), (5,6), (7,8) are not device edges and their CX gates ran NOISE-FREE
    (2 of 5 chain edges at 6 qubits, 4 of 9 at 10 qubits: the device noise was
    under-applied);
  * on FakeSherbrooke the calibration snapshot carries ECR error = 1.0 (dead
    gates) on edges (5,6), (6,7), (8,9), so the 10-qubit chain over physical
    qubits 0..9 was fully depolarized on three of nine edges (an artifact of the
    snapshot, not of circuit depth), while the 6-qubit chain 0..5 was healthy.

Fix
---
For each (backend, n) we select the connected n-qubit path of the device
coupling graph with the smallest summed two-qubit error over HEALTHY edges
(error < 0.5, i.e. excluding dead gates), which is what a layout pass scored by
calibration data would do. We then build an n-qubit noise model by relabelling
that path's calibrated per-qubit (sx/x/id/rz thermal + depolarizing) and
per-edge (cx/ecr/cz) errors onto chain qubits 0..n-1, using Aer's
basic_device_gate_errors on the backend target. The rest of the pipeline
(linear-chain transpilation to the backend basis, measurement-free
density-matrix kernel, models, seeds, metrics) is IDENTICAL to v1, so v2 numbers
are directly comparable to the paper's ideal reference values.

Extras over v1
--------------
  --zz_reps R   : depth of the ZZ map (default 2). QSVC=[ZZ(R)], QVE=[Z(1),ZZ(R)],
                  QWE=[ZZ(R),Pauli(1)]. Used for the depth-isolation runs (R=3,4).
  Chosen paths and their edge errors are logged and written to
  results/circuit_depth/device_paths_<session>.json.

Everything else (CLI, outputs device_noise_runs_/summary_<session>.csv) matches
device_noise_validation.py; the 'model' column is unchanged and the ZZ depth is
recorded in the 'zz_reps' column.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import qiskit_ibm_runtime.fake_provider as fake_provider
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.device import basic_device_gate_errors

import device_noise_validation as v1

logger = logging.getLogger("device_noise_validation_v2")

TWO_Q = ('cx', 'ecr', 'cz')
_PATH_CACHE: Dict[Tuple[str, int], List[int]] = {}
_PROFILE_CACHE: Dict[Tuple[str, int], Tuple[NoiseModel, List[str], float]] = {}
_CURRENT_NQ = {'n': 6}
_PROFILE_LOCK = threading.Lock()


def _edge_error_table(backend) -> Tuple[str, Dict[Tuple[int, int], float]]:
    t = backend.target
    gname = next(g for g in TWO_Q if g in t)
    errs: Dict[Tuple[int, int], float] = {}
    for qubits, props in t[gname].items():
        if props is None or props.error is None:
            continue
        a, b = qubits
        e = float(props.error)
        key = (min(a, b), max(a, b))
        # keep the smaller of the two directions if both are calibrated
        errs[key] = min(e, errs.get(key, e))
    return gname, errs


PATH_POLICY = {'policy': 'best'}


def select_best_path(backend, n: int, dead_threshold: float = 0.5,
                     policy: str = None) -> Tuple[List[int], float, str]:
    """Simple path of n qubits over HEALTHY edges (error < dead_threshold).

    policy='best'  : minimum summed two-qubit error (what an error-aware layout pass picks)
    policy='worst' : maximum summed two-qubit error among healthy paths (pessimistic bracket)
    """
    policy = policy or PATH_POLICY['policy']
    gname, errs = _edge_error_table(backend)
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for (a, b), e in errs.items():
        if e >= dead_threshold:
            continue
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))
    sign = 1.0 if policy == 'best' else -1.0
    best = (float('inf'), None)

    def dfs(path, cost):
        nonlocal best
        # branch-and-bound is only valid when minimising a monotone non-decreasing cost
        if policy == 'best' and cost >= best[0]:
            return
        if len(path) == n:
            if cost < best[0]:
                best = (cost, list(path))
            return
        for nb, e in adj.get(path[-1], []):
            if nb not in path:
                path.append(nb)
                dfs(path, cost + sign * e)
                path.pop()

    for start in sorted(adj):
        dfs([start], 0.0)
    if best[1] is None:
        raise RuntimeError(f"No healthy {n}-qubit path on {backend.name}")
    return best[1], sign * best[0], gname


def build_path_noise_model(backend, path: List[int]) -> Tuple[NoiseModel, List[str]]:
    """Relabel the backend's calibrated gate errors along `path` onto qubits 0..n-1."""
    target = backend.target
    remap = {p: i for i, p in enumerate(path)}
    edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
    edges |= {(b, a) for a, b in edges}
    nm = NoiseModel(basis_gates=[g for g in target.operation_names
                                 if g in {'id', 'rz', 'sx', 'x', 'cx', 'ecr', 'cz'}])
    added = 0
    all_errors = basic_device_gate_errors(target=target)
    calibrated_2q = {(name, tuple(q)) for name, q, _ in all_errors if len(q) == 2}
    for name, qubits, error in all_errors:
        qubits = tuple(qubits)
        if len(qubits) == 1:
            if qubits[0] in remap:
                nm.add_quantum_error(error, name, [remap[qubits[0]]])
                added += 1
        elif len(qubits) == 2 and qubits in edges:
            a, b = qubits
            nm.add_quantum_error(error, name, [remap[a], remap[b]])
            # if only one direction is calibrated, mirror it so both orientations are noisy
            if (name, (b, a)) not in calibrated_2q:
                nm.add_quantum_error(error, name, [remap[b], remap[a]])
            added += 1
    basis = [g for g in nm.basis_gates if g in v1._UNITARY_BASIS]
    logger.info(f"  path noise model: {added} calibrated errors relabelled onto {len(path)} qubits; basis={basis}")
    return nm, basis


def get_device_profile_v2(device_name: str):
    with _PROFILE_LOCK:  # backend Target objects are not safe for concurrent construction
        return _get_device_profile_v2_locked(device_name)


def _get_device_profile_v2_locked(device_name: str):
    n = _CURRENT_NQ['n']
    key = (device_name, n)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    if not hasattr(fake_provider, device_name):
        raise ValueError(f"Unknown fake backend: {device_name}")
    backend = getattr(fake_provider, device_name)()
    path, cost, gname = select_best_path(backend, n)
    _PATH_CACHE[key] = path
    _, errs = _edge_error_table(backend)
    edge_errs = [errs[(min(a, b), max(a, b))] for a, b in zip(path[:-1], path[1:])]
    logger.info(f"  {device_name} n={n}: path={path} ({gname}) edge errors="
                f"{np.round(edge_errs, 4).tolist()} sum={cost:.4f} "
                f"median={np.median(edge_errs):.4f} max={np.max(edge_errs):.4f}")
    nm, basis = build_path_noise_model(backend, path)
    mean_ro = float(np.mean([backend.target['measure'][(q,)].error for q in path]))
    _PROFILE_CACHE[key] = (nm, basis, mean_ro)
    return _PROFILE_CACHE[key]


def main():
    # Parse our extra flags, forward the rest to v1.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--zz_reps', type=int, default=2)
    ap.add_argument('--num_qubits', type=int, default=6)
    ap.add_argument('--session_id', default=None)
    ap.add_argument('--devices', nargs='+', default=['ideal', 'FakeKolkataV2'])
    ap.add_argument('--path_policy', choices=['best', 'worst'], default='best')
    known, _ = ap.parse_known_args()
    PATH_POLICY['policy'] = known.path_policy
    # strip our private flags before v1 parses argv
    for flag in ('--zz_reps', '--path_policy'):
        if flag in sys.argv:
            i = sys.argv.index(flag); del sys.argv[i:i + 2]

    _CURRENT_NQ['n'] = known.num_qubits
    r = known.zz_reps
    v1.MODEL_MAPS['QSVC'] = [('ZZ', r)]
    v1.MODEL_MAPS['QVE'] = [('Z', 1), ('ZZ', r)]
    v1.MODEL_MAPS['QWE'] = [('ZZ', r), ('Pauli', 1)]
    v1.get_device_profile = get_device_profile_v2  # execute_device_run resolves this global at call time

    session_id = known.session_id or f"devnoise_v2_{known.num_qubits}q_r{r}_{known.path_policy}_{time.strftime('%Y%m%d_%H%M%S')}"
    if '--session_id' not in sys.argv:
        sys.argv += ['--session_id', session_id]

    logger.info(f"v2 layout-aware device-noise validation: zz_reps={r}, num_qubits={known.num_qubits}")
    # Pre-compute and record the paths for the requested devices.
    v1.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for dev in known.devices:
        if dev == 'ideal':
            continue
        get_device_profile_v2(dev)
        paths[dev] = _PATH_CACHE[(dev, known.num_qubits)]
    (v1.RESULTS_DIR / f'device_paths_{session_id}.json').write_text(
        json.dumps({'num_qubits': known.num_qubits, 'zz_reps': r, 'path_policy': known.path_policy, 'paths': paths}, indent=2))

    # Run v1's driver; then append zz_reps to the outputs.
    v1.main()
    import pandas as pd
    for kind in ('runs', 'summary'):
        f = v1.RESULTS_DIR / f'device_noise_{kind}_{session_id}.csv'
        if f.exists():
            df = pd.read_csv(f)
            df['zz_reps'] = r
            df['layout'] = f'{known.path_policy}_healthy_path'
            df.to_csv(f, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')
    main()
