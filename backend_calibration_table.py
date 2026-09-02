#!/usr/bin/env python3
"""Summarize the calibration snapshots of the fake IBM backends used in the paper.

For each backend we report the median single-qubit gate error (sx), the median
two-qubit gate error (cx / ecr / cz, whichever the backend uses), the median
readout error, and the median T1 / T2, read straight from the backend target.
The paper currently characterises the backends by readout error only; this
table gives the two-qubit error that actually drives the kernel collapse.

Output: results/circuit_depth/backend_calibration.csv and a LaTeX snippet on stdout.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import qiskit_ibm_runtime.fake_provider as fp

BACKENDS = ['FakeKolkataV2', 'FakeCairoV2', 'FakeSherbrooke', 'FakeFez']  # Cairo: use cx entries (12 edges); its ecr entries include a dead gate
OUT = Path('results/circuit_depth/backend_calibration.csv')


def med(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float('nan')


def summarize(name: str) -> dict:
    b = getattr(fp, name)()
    t = b.target
    row = {'backend': name, 'num_qubits': b.num_qubits, 'processor': getattr(b, 'processor_type', {}) or ''}
    # single-qubit: sx
    sx = [p.error for _, p in t['sx'].items() if p is not None] if 'sx' in t else []
    row['sx_error_median'] = med(sx)
    # two-qubit
    twoq_name = next((g for g in ('cx', 'cz', 'ecr') if g in t), None)
    twoq = [p.error for _, p in t[twoq_name].items() if p is not None] if twoq_name else []
    row['two_qubit_gate'] = twoq_name
    row['two_qubit_error_median'] = med(twoq)
    row['two_qubit_error_max'] = float(np.max([v for v in twoq if v is not None])) if twoq else float('nan')
    ro = [p.error for _, p in t['measure'].items() if p is not None] if 'measure' in t else []
    row['readout_error_median'] = med(ro)
    row['readout_error_mean'] = float(np.mean(ro)) if ro else float('nan')
    t1 = [q.t1 for q in t.qubit_properties if q is not None and q.t1 is not None]
    t2 = [q.t2 for q in t.qubit_properties if q is not None and q.t2 is not None]
    row['t1_median_us'] = med(t1) * 1e6
    row['t2_median_us'] = med(t2) * 1e6
    return row


def main():
    rows = [summarize(n) for n in BACKENDS]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))
    print()
    print('% LaTeX rows (backend, qubits, 2q gate, median 1q %, median 2q %, median readout %, T1 us, T2 us)')
    for r in rows:
        print(f"\\texttt{{{r['backend']}}} & {r['num_qubits']} & \\texttt{{{r['two_qubit_gate']}}} & "
              f"{100*r['sx_error_median']:.3f} & {100*r['two_qubit_error_median']:.2f} & "
              f"{100*r['readout_error_median']:.2f} & {r['t1_median_us']:.0f} & {r['t2_median_us']:.0f} \\\\")


if __name__ == '__main__':
    main()
