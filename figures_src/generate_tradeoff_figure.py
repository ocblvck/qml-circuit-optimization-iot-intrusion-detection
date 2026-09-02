#!/usr/bin/env python3
"""Two-level trade-off summary figure for the TQE manuscript.

Row 1 and 2: accuracy at the highest synthetic noise level (p1q = 0.05) versus
two-qubit gate count, one column per dataset, one row per entanglement pattern.
Each model appears twice (L0 hollow, L3 filled) joined by a connector, so the
transpiler effect is the length of the connector and the topology effect is the
difference between the rows.

Row 3: L3-minus-L0 accuracy delta versus L3-minus-L0 kernel-time delta for every
6-qubit condition (model x entanglement x 10 noise levels), Holm-significant
accuracy contrasts filled.

Reads only archived result files; no experiments are rerun.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results', 'circuit_depth')
OUT = os.path.join(HERE, 'figures', 'tradeoff_summary.png')

TAGS = {
    'IoT': 'iot_original_distribution_6q_5k_v2_noisy_20260310_105202',
    'UNSW-2018': 'unsw_2018_iot_botnet_final_10_best_6q_5k_v2_noisy_20260310_105202',
    'UNSW-NB15': 'unsw_nb15_6q_5k_v2_noisy_20260415_131112',
}
MODELS = ['QSVC', 'QVE', 'QWE']
COLORS = {'QSVC': '#1f77b4', 'QVE': '#2ca02c', 'QWE': '#d62728'}   # matches other figures
MARKERS = {'QSVC': 'o', 'QVE': '^', 'QWE': 's'}                     # redundant encoding
NOISE_HI = 0.05

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'legend.fontsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
FLOORS = {'IoT': 90.6, 'UNSW-2018': 9.53, 'UNSW-NB15': 63.9}

for col, (dname, tag) in enumerate(TAGS.items()):
    means = pd.read_csv(os.path.join(RES, f'comprehensive_model_noise_{tag}.csv'))
    stats = pd.read_csv(os.path.join(RES, f'comprehensive_model_noise_stats_{tag}.csv'))

    # ---- row 1: accuracy vs CX gates at p1q = 0.05, both topologies in one panel ----
    ax = axes[0, col]
    sub = means[np.isclose(means.noise_level, NOISE_HI)]
    for ent in ['linear', 'full']:
        for m in MODELS:
            s = sub[(sub.model == m) & (sub.entanglement == ent)].sort_values('optimization_level')
            if len(s) != 2:
                continue
            x = s.aggregate_two_qubit_gates.to_numpy(float)
            y = 100 * s.accuracy_mean.to_numpy(float)
            ax.plot(x, y, '-', color=COLORS[m], lw=1.2, alpha=0.8, zorder=2)
            ax.scatter(x[0], y[0], marker=MARKERS[m], s=55, facecolors='white',
                       edgecolors=COLORS[m], linewidths=1.6, zorder=3)
            ax.scatter(x[1], y[1], marker=MARKERS[m], s=55, facecolors=COLORS[m],
                       edgecolors=COLORS[m], linewidths=1.6, zorder=3)
            if ent == 'full':
                ax.annotate(m, (x[1], y[1]), xytext=(6, -3), textcoords='offset points',
                            fontsize=8, color='#333333')
    ax.axhline(FLOORS[dname], color='#999999', lw=0.8, ls=':', zorder=1)
    ax.text(0.01, FLOORS[dname], 'constant-class floor', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=7, color='#777777')
    ylo, yhi = ax.get_ylim()
    yhi2 = yhi + 0.14 * (yhi - ylo)
    ax.set_ylim(ylo, yhi2)
    ax.text(48, yhi2, 'linear entanglement', ha='center', va='top', fontsize=7.5, color='#555555')
    ax.text(183, yhi2, 'full entanglement', ha='center', va='top', fontsize=7.5, color='#555555')
    ax.set_xlim(0, 240)
    ax.set_title(f'{dname}, $p_{{1q}}=0.05$: accuracy vs. circuit size')
    ax.set_xlabel('Two-qubit (CX) gates after transpilation')
    if col == 0:
        ax.set_ylabel('Accuracy at highest noise (%)')

    # ---- row 2: dAcc vs dKernelTime for all conditions ----
    ax = axes[1, col]
    ax.axhline(0, color='#999999', lw=0.8, zorder=1)
    ax.axvline(0, color='#999999', lw=0.8, zorder=1)
    for m in MODELS:
        s = stats[stats.model == m]
        dacc = 100 * s.accuracy_delta_l3_minus_l0.to_numpy(float)
        dt = 100 * s.kernel_time_delta_l3_minus_l0.to_numpy(float) / s.kernel_time_mean_l0.to_numpy(float)
        sig = s.accuracy_significant_holm.astype(bool).to_numpy()
        ax.scatter(dt[~sig], dacc[~sig], marker=MARKERS[m], s=34, facecolors='white',
                   edgecolors=COLORS[m], linewidths=1.2, zorder=3, alpha=0.9)
        ax.scatter(dt[sig], dacc[sig], marker=MARKERS[m], s=46, facecolors=COLORS[m],
                   edgecolors='black', linewidths=0.8, zorder=4)
    ax.set_title(f'{dname}: L3 $-$ L0 over all 6-qubit conditions')
    ax.set_xlabel('Kernel-time change under L3 (%)')
    if col == 0:
        ax.set_ylabel('Accuracy change under L3 (pp)')
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_yticks([-1, 0, 1, 10, 50])
    ax.set_yticklabels(['-1', '0', '1', '10', '50'])

# legends
model_handles = [Line2D([0], [0], marker=MARKERS[m], color='none', markerfacecolor=COLORS[m],
                        markeredgecolor=COLORS[m], markersize=7, label=m) for m in MODELS]
level_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
                        markeredgecolor='#444444', markersize=7, label='L0'),
                 Line2D([0], [0], marker='o', color='none', markerfacecolor='#444444',
                        markeredgecolor='#444444', markersize=7, label='L3')]
sig_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#444444',
                      markeredgecolor='black', markersize=7, label='Holm-significant accuracy contrast'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
                      markeredgecolor='#444444', markersize=7, label='not significant')]
axes[0, 0].legend(handles=model_handles + level_handles, loc='center right', ncol=2, frameon=False)
axes[1, 0].legend(handles=sig_handles, loc='upper right', frameon=False)

fig.tight_layout(h_pad=1.6, w_pad=1.2)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print('saved', OUT)
