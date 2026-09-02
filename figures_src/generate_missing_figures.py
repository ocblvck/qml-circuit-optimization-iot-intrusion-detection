#!/usr/bin/env python3
"""Rebuild the three figures that had no generator script (March 2026 PNGs).

  iot_6q_noise_response.png    : IoT 6q noise response, same layout as the
  unsw_6q_noise_response.png   : UNSW-2018 counterpart          NB15 figure
  tenq_accuracy_runtime_tradeoff.png : 10q accuracy + runtime vs noise

All read only archived comprehensive_model_noise CSVs. Canvas sizes follow the
August 2026 layout pass (about 11 in wide for full-width figures so printed
fonts stay legible).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results', 'circuit_depth')
OUTDIR = os.path.join(HERE, 'figures')
MODEL_COLORS = {'QSVC': '#1f77b4', 'QVE': '#2ca02c', 'QWE': '#d62728'}

SIXQ = {
    'IoT': 'comprehensive_model_noise_iot_original_distribution_6q_5k_v2_noisy_20260310_105202.csv',
    'UNSW-2018': 'comprehensive_model_noise_unsw_2018_iot_botnet_final_10_best_6q_5k_v2_noisy_20260310_105202.csv',
}
TENQ = {
    'IoT': 'comprehensive_model_noise_iot_original_distribution_10q_2p5k_v2_noisy_20260323_114741.csv',
    'UNSW-2018': 'comprehensive_model_noise_unsw_2018_iot_botnet_final_10_best_10q_2p5k_v2_noisy_20260323_114741.csv',
    'UNSW-NB15': 'comprehensive_model_noise_unsw_nb15_10q_2p5k_v2_noisy_20260504_112058.csv',
}


def noise_response(dataset_label, csv_name, out_name):
    d = pd.read_csv(os.path.join(RESULTS, csv_name))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, ent in zip(axes, ['full', 'linear']):
        sub = d[d.entanglement == ent]
        for model in ['QSVC', 'QVE', 'QWE']:
            for opt, ls, lab in [(0, '--', 'L0'), (3, '-', 'L3')]:
                s = sub[(sub.model == model) & (sub.optimization_level == opt)]
                s = s.sort_values('noise_level')
                ax.plot(s.noise_level, s.accuracy_mean * 100,
                        ls, color=MODEL_COLORS[model], linewidth=1.8,
                        marker='o', markersize=3.5, label=f'{model} {lab}')
        ax.set_xscale('symlog', linthresh=3e-4)
        ax.set_xlabel('Noise level ($p_{1q}$)', fontsize=11)
        ax.set_title(f'{ent.capitalize()} entanglement', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].legend(fontsize=8, ncol=2, loc='lower left', framealpha=0.9)
    fig.suptitle(f'{dataset_label} 6-Qubit Noise Response', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTDIR, out_name)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated:', out)


def tenq_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    DS_STYLE = {'IoT': '-', 'UNSW-2018': '--', 'UNSW-NB15': ':'}
    for ds, csv_name in TENQ.items():
        d = pd.read_csv(os.path.join(RESULTS, csv_name))
        # 10q extension: full entanglement; average L0/L3 (their curves nearly coincide)
        sub = d[d.entanglement == 'full'].groupby(['model', 'noise_level'], as_index=False).agg(
            acc=('accuracy_mean', 'mean'), t=('avg_time', 'mean'))
        for model in ['QSVC', 'QVE']:
            s = sub[sub.model == model].sort_values('noise_level')
            axes[0].plot(s.noise_level, 100 * s.acc, DS_STYLE[ds], color=MODEL_COLORS[model],
                         marker='o', markersize=3.5, linewidth=1.8, label=f'{model}, {ds}')
            axes[1].plot(s.noise_level, s.t / 60.0, DS_STYLE[ds], color=MODEL_COLORS[model],
                         marker='o', markersize=3.5, linewidth=1.8, label=f'{model}, {ds}')
    for ax in axes:
        ax.set_xscale('symlog', linthresh=1e-3)
        ax.set_xlabel('Noise level ($p_{1q}$)', fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title('Accuracy under noise (10 qubits, full ent.)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean run time per configuration (min)', fontsize=11)
    axes[1].set_title('Runtime inflation under noise', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2, loc='lower left', framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'tenq_accuracy_runtime_tradeoff.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated:', out)


if __name__ == '__main__':
    noise_response('IoT', SIXQ['IoT'], 'iot_6q_noise_response.png')
    noise_response('UNSW-2018', SIXQ['UNSW-2018'], 'unsw_6q_noise_response.png')
    tenq_tradeoff()
