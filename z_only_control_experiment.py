#!/usr/bin/env python3
"""
Z-map-only QSVC control for the kernel-concentration explanation of QVE.

The paper attributes the noise robustness of the voting ensemble (QVE = Z + ZZ)
to its shallow, non-concentrating Z encoding, and shows analytically that under
strong noise the QVE decision reduces to the Z-map classifier with a shifted
threshold. The direct control, a stand-alone Z-map QSVC swept over the same
noise grid, was not part of the factorial design. This harness runs it.

It reuses feature_map_ablation.py unchanged (same DataProcessor, kernels,
cuSVC settings, seeds, noise grid, multi-GPU dispatch) by registering two
single-member "compositions" for the QVE code path:

    1map_Z  : [Z(1)]    stand-alone Z-map QSVC        (the control)
    1map_ZZ : [ZZ(2)]   stand-alone ZZ-map QSVC       (paired same-harness reference)

With one member the QVE branch is exactly a single QSVC prediction, so the
'model' column is relabelled to QSVC_Z / QSVC_ZZ in the outputs. The 2-map QVE
baseline at identical seeds already exists in the ablation results
(fmablation_20260608_125602), so QSVC_Z, QSVC_ZZ and QVE(2map) are run-paired.

Outputs (results/circuit_depth/):
    ablation_feature_map_runs_<session>.csv / _summary_ / _pairwise_
"""
from __future__ import annotations
import sys
import time
import feature_map_ablation as fma

fma.COMPOSITIONS['QVE']['1map_Z'] = [('Z', 1)]
fma.COMPOSITIONS['QVE']['1map_ZZ'] = [('ZZ', 2)]

if __name__ == '__main__':
    session = f"zonly_{time.strftime('%Y%m%d_%H%M%S')}"
    if '--session_id' not in sys.argv:
        sys.argv += ['--session_id', session]
    if '--models' not in sys.argv:
        sys.argv += ['--models', 'QVE']
    if '--compositions' not in sys.argv:
        sys.argv += ['--compositions', '1map_Z', '1map_ZZ']
    fma.main()
    # Relabel model names in the outputs so the CSVs are self-describing.
    import pandas as pd
    sid = sys.argv[sys.argv.index('--session_id') + 1]
    for kind in ('runs', 'summary', 'pairwise'):
        f = fma.RESULTS_DIR / f'ablation_feature_map_{kind}_{sid}.csv'
        if f.exists() and f.stat().st_size > 1:
            df = pd.read_csv(f)
            if 'composition' in df.columns and 'model' in df.columns:
                df['model'] = df.apply(
                    lambda r: {'1map_Z': 'QSVC_Z', '1map_ZZ': 'QSVC_ZZ'}.get(str(r['composition']), r['model']), axis=1)
            df.to_csv(f, index=False)
