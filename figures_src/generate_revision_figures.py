#!/usr/bin/env python3
"""Additional figures for the TQE paper revision.

Generates three figures that were missing from the prior figure set:
  1. UNSW-NB15 6-qubit noise-response (dataset-resolved, previously only IoT/UNSW-2018).
  2. UNSW-NB15 6-qubit L3-L0 accuracy-delta heatmap.
  3. Calibrated device-noise validation (ideal vs FakeKolkataV2) bar chart.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import os
HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS = os.path.join(HERE, '..', 'results', 'circuit_depth')
OUTDIR = os.path.join(HERE, 'figures')
os.makedirs(OUTDIR, exist_ok=True)

NB15_MEANS = os.path.join(
    RESULTS, 'comprehensive_model_noise_unsw_nb15_6q_5k_v2_noisy_20260415_131112.csv')
NB15_STATS = os.path.join(
    RESULTS, 'comprehensive_model_noise_stats_unsw_nb15_6q_5k_v2_noisy_20260415_131112.csv')
DEVNOISE = os.path.join(RESULTS, 'device_noise_summary_devnoise_v2_6q_r2_best.csv')  # layout-aware v2

MODEL_COLORS = {'QSVC': '#1f77b4', 'QVE': '#2ca02c', 'QWE': '#d62728'}


# ============================================================
# FIGURE: UNSW-NB15 6-qubit noise response
# ============================================================
def generate_nb15_noise_response():
    d = pd.read_csv(NB15_MEANS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, ent in zip(axes, ['full', 'linear']):
        sub = d[d.entanglement == ent]
        for model in ['QSVC', 'QVE', 'QWE']:
            for opt, ls, lab in [(0, '--', 'L0'), (3, '-', 'L3')]:
                s = sub[(sub.model == model) & (sub.optimization_level == opt)]
                s = s.sort_values('noise_level')
                ax.plot(s.noise_level, s.accuracy_mean * 100,
                        ls, color=MODEL_COLORS[model], linewidth=1.8,
                        marker='o', markersize=3.5,
                        label=f'{model} {lab}')
        ax.set_xscale('symlog', linthresh=3e-4)
        ax.set_xlabel('Noise level ($p_{1q}$)', fontsize=11)
        ax.set_title(f'{ent.capitalize()} entanglement', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].legend(fontsize=8, ncol=2, loc='lower left', framealpha=0.9)
    fig.suptitle('UNSW-NB15 6-Qubit Noise Response', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'unsw_nb15_6q_noise_response.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated:', out)


# ============================================================
# FIGURE: UNSW-NB15 6-qubit L3-L0 accuracy-delta heatmap
# ============================================================
def generate_nb15_heatmap():
    rows = pd.read_csv(NB15_STATS)
    configs, noise_levels = [], []
    for _, r in rows.iterrows():
        lbl = f"{r['model']} / {r['entanglement']}"
        if lbl not in configs:
            configs.append(lbl)
        if r['noise_level'] not in noise_levels:
            noise_levels.append(r['noise_level'])
    noise_levels = sorted(noise_levels)
    matrix = np.full((len(configs), len(noise_levels)), np.nan)
    sig = np.full((len(configs), len(noise_levels)), False)
    sig_col = ('accuracy_significant_holm'
               if 'accuracy_significant_holm' in rows.columns else None)
    for _, r in rows.iterrows():
        ci = configs.index(f"{r['model']} / {r['entanglement']}")
        ni = noise_levels.index(r['noise_level'])
        matrix[ci, ni] = r['accuracy_delta_l3_minus_l0'] * 100
        if sig_col is not None:
            sig[ci, ni] = bool(r[sig_col])
        elif 'accuracy_p_value' in rows.columns and pd.notna(r['accuracy_p_value']):
            sig[ci, ni] = r['accuracy_p_value'] < 0.05

    fig, ax = plt.subplots(figsize=(9, 5))
    finite = matrix[np.isfinite(matrix)]
    non_outlier = finite[np.abs(finite) < 20]
    vmax = max(3, np.nanmax(np.abs(non_outlier))) if len(non_outlier) else 5
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap='RdBu', norm=norm, aspect='auto')
    for i in range(len(configs)):
        for j in range(len(noise_levels)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            txt = f'{val:.0f}' if abs(val) > 10 else (f'{val:.2f}' if abs(val) >= 0.005 else '0.00')
            if sig[i, j]:
                txt += '*'
            color = 'white' if abs(val) > vmax * 0.55 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=color,
                    fontweight='bold' if sig[i, j] else 'normal')
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f'{nl:.4f}'.rstrip('0').rstrip('.') for nl in noise_levels],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_xlabel('Noise level ($p_{1q}$)   [* = Holm-significant at $\\alpha=0.05$]', fontsize=10)
    ax.set_title('UNSW-NB15: Accuracy Delta (L3 $-$ L0) in pp', fontsize=12, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label('$\\Delta$ Accuracy (pp)', fontsize=9)
    plt.savefig(os.path.join(OUTDIR, 'unsw_nb15_heatmap_delta.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated: unsw_nb15_heatmap_delta.png')


# ============================================================
# FIGURE: Calibrated device-noise validation (ideal vs FakeKolkataV2)
# ============================================================
def generate_device_noise_figure():
    d = pd.read_csv(DEVNOISE)
    dsmap = {'IoT_Original_Distribution': 'IoT',
             'UNSW_2018_IoT_Botnet_Final_10_Best': 'UNSW-2018',
             'UNSW_NB15': 'UNSW-NB15'}
    d['ds'] = d.dataset.map(dsmap)
    g = (d.groupby(['ds', 'model', 'device'])
         .agg(acc=('accuracy_mean', 'mean'))
         .reset_index())
    datasets = ['IoT', 'UNSW-2018', 'UNSW-NB15']
    models = ['QSVC', 'QVE', 'QWE']
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=False)
    for ax, ds in zip(axes, datasets):
        sub = g[g.ds == ds]
        x = np.arange(len(models))
        w = 0.36
        ideal = [sub[(sub.model == m) & (sub.device == 'ideal')].acc.values[0] * 100 for m in models]
        fake = [sub[(sub.model == m) & (sub.device == 'FakeKolkataV2')].acc.values[0] * 100 for m in models]
        ax.bar(x - w / 2, ideal, w, label='Ideal', color='#4c72b0')
        ax.bar(x + w / 2, fake, w, label='FakeKolkataV2', color='#dd8452')
        for xi, (a, b) in enumerate(zip(ideal, fake)):
            ax.text(xi - w / 2, a + 0.1, f'{a:.1f}', ha='center', va='bottom', fontsize=7)
            ax.text(xi + w / 2, b + 0.1, f'{b:.1f}', ha='center', va='bottom', fontsize=7)
        lo = min(min(ideal), min(fake))
        ax.set_ylim(max(0, lo - 4), 100.5 if ds != 'UNSW-NB15' else max(max(ideal), max(fake)) + 4)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_title(ds, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].legend(fontsize=9, loc='lower right')
    fig.suptitle('Calibrated Device-Noise Validation: Ideal vs. FakeKolkataV2 (6 qubits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'device_noise_validation.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated: device_noise_validation.png')


if __name__ == '__main__':
    generate_nb15_noise_response()
    generate_nb15_heatmap()
    generate_device_noise_figure()
    print('All additional figures generated.')
