#!/usr/bin/env python3
"""Generate enhancement figures for the TQE journal paper."""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

import os
HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS = os.path.join(HERE, '..', 'results', 'circuit_depth')
OUTDIR = os.path.join(HERE, 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ── Helper: load noise stats CSV ──
def load_noise_stats(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

# ============================================================
# FIGURE 1: Heatmap of L3-L0 accuracy delta for IoT and UNSW
# ============================================================
def generate_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True,
                             gridspec_kw={'wspace': 0.35})

    for ax_idx, (dataset_label, dataset_tag) in enumerate([
        ('IoT', 'iot_original_distribution'),
        ('UNSW-2018', 'unsw_2018_iot_botnet_final_10_best')
    ]):
        fname = f'comprehensive_model_noise_stats_{dataset_tag}_6q_5k_v2_noisy_20260310_105202.csv'
        rows = load_noise_stats(os.path.join(RESULTS, fname))

        configs = []
        noise_levels = []
        for r in rows:
            lbl = f"{r['model']} / {r['entanglement']}"
            nl = float(r['noise_level'])
            if lbl not in configs:
                configs.append(lbl)
            if nl not in noise_levels:
                noise_levels.append(nl)

        noise_levels.sort()
        matrix = np.full((len(configs), len(noise_levels)), np.nan)
        sig_matrix = np.full((len(configs), len(noise_levels)), False)

        for r in rows:
            lbl = f"{r['model']} / {r['entanglement']}"
            ci = configs.index(lbl)
            ni = noise_levels.index(float(r['noise_level']))
            matrix[ci, ni] = float(r['accuracy_delta_l3_minus_l0']) * 100
            sig_matrix[ci, ni] = r['accuracy_significant_holm'] == 'True'

        # Per-panel color scale: use ±5pp unless the non-outlier range demands more
        finite = matrix[np.isfinite(matrix)]
        non_outlier = finite[np.abs(finite) < 20]
        vmax = max(3, np.nanmax(np.abs(non_outlier))) if len(non_outlier) > 0 else 5
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        ax = axes[ax_idx]
        im = ax.imshow(matrix, cmap='RdBu', norm=norm, aspect='auto')

        for i in range(len(configs)):
            for j in range(len(noise_levels)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                if abs(val) > 10:
                    txt = f'{val:.0f}'
                elif abs(val) < 0.005:
                    txt = '0.00'
                else:
                    txt = f'{val:.2f}'
                if sig_matrix[i, j]:
                    txt += '*'
                color = 'white' if abs(val) > vmax * 0.55 else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=color,
                        fontweight='bold' if sig_matrix[i, j] else 'normal')

        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f'{nl:.4f}'.rstrip('0').rstrip('.') for nl in noise_levels],
                           rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(configs)))
        if ax_idx == 0:
            ax.set_yticklabels(configs, fontsize=9)
        ax.set_xlabel('Noise level ($p_{1q}$)   [* = Holm-significant at $\\alpha=0.05$]', fontsize=10)
        ax.set_title(f'{dataset_label} Dataset', fontsize=12, fontweight='bold')
        # Individual colorbar per panel
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
        cb.set_label('$\\Delta$ Accuracy (pp)', fontsize=9)

    fig.suptitle('Accuracy Delta (L3 $-$ L0) in Percentage Points',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.savefig(os.path.join(OUTDIR, 'heatmap_accuracy_delta.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated: heatmap_accuracy_delta.png')

# ============================================================
# FIGURE 2: Cohen's d effect size dot plot
# ============================================================
def generate_cohens_d_plot():
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)

    for ax_idx, (dataset_label, dataset_tag) in enumerate([
        ('IoT', 'iot_original_distribution'),
        ('UNSW-2018', 'unsw_2018_iot_botnet_final_10_best')
    ]):
        fname = f'comprehensive_model_noise_stats_{dataset_tag}_6q_5k_v2_noisy_20260310_105202.csv'
        rows = load_noise_stats(os.path.join(RESULTS, fname))

        ax = axes[ax_idx]
        config_labels = []
        positions = []
        colors_list = []
        markers_list = []
        d_values = []
        sig_flags = []

        color_map = {
            ('QSVC', 'full'): '#1f77b4',
            ('QSVC', 'linear'): '#aec7e8',
            ('QVE', 'full'): '#2ca02c',
            ('QVE', 'linear'): '#98df8a',
            ('QWE', 'full'): '#d62728',
            ('QWE', 'linear'): '#ff9896',
        }

        noise_levels = sorted(set(float(r['noise_level']) for r in rows))
        y_pos = 0
        ytick_pos = []
        ytick_labels = []

        for nl in noise_levels:
            subset = [r for r in rows if float(r['noise_level']) == nl]
            for r in subset:
                d = float(r['accuracy_cohens_d_paired'])
                sig = r['accuracy_significant_holm'] == 'True'
                key = (r['model'], r['entanglement'])
                c = color_map.get(key, 'gray')

                d_values.append(d)
                positions.append(y_pos)
                colors_list.append(c)
                sig_flags.append(sig)
                y_pos += 1

            ytick_pos.append(y_pos - len(subset) / 2)
            ytick_labels.append(f'{nl:.4f}'.rstrip('0').rstrip('.'))

        # Plot dots
        for i in range(len(d_values)):
            marker = 's' if sig_flags[i] else 'o'
            edge = 'black' if sig_flags[i] else 'none'
            size = 50 if sig_flags[i] else 25
            ax.scatter(d_values[i], positions[i], c=colors_list[i], marker=marker,
                      s=size, edgecolors=edge, linewidths=1, zorder=3)

        # Reference lines
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        ax.axvline(0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(-0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axvline(-0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axvline(0.8, color='gray', linewidth=0.5, linestyle='-.', alpha=0.4)
        ax.axvline(-0.8, color='gray', linewidth=0.5, linestyle='-.', alpha=0.4)

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=8)
        ax.set_ylabel('Noise level ($p_{1q}$)', fontsize=10)
        ax.set_title(f'{dataset_label} Dataset — Paired Cohen\'s $d$ (L3 vs. L0)', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    axes[-1].set_xlabel("Paired Cohen's $d$ (positive = L3 better)", fontsize=10)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label='QSVC / full'),
        mpatches.Patch(color='#aec7e8', label='QSVC / linear'),
        mpatches.Patch(color='#2ca02c', label='QVE / full'),
        mpatches.Patch(color='#98df8a', label='QVE / linear'),
        mpatches.Patch(color='#d62728', label='QWE / full'),
        mpatches.Patch(color='#ff9896', label='QWE / linear'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=8, label='Holm-significant'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.04))

    # Effect size reference annotations
    axes[0].text(0.2, -0.5, 'small', fontsize=7, color='gray', ha='center', style='italic')
    axes[0].text(0.5, -0.5, 'medium', fontsize=7, color='gray', ha='center', style='italic')
    axes[0].text(0.8, -0.5, 'large', fontsize=7, color='gray', ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'cohens_d_effect_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated: cohens_d_effect_size.png')

# ============================================================
# FIGURE 3: Box plots of per-run accuracy at key noise levels
# ============================================================
def generate_boxplots():
    target_noise = [0.0, 0.005, 0.02, 0.05]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))

    for row_idx, (dataset_label, dataset_tag) in enumerate([
        ('IoT', 'iot_original_distribution'),
        ('UNSW-2018', 'unsw_2018_iot_botnet_final_10_best')
    ]):
        fname = f'comprehensive_model_noise_runs_{dataset_tag}_6q_5k_v2_noisy_20260310_105202.csv'
        with open(os.path.join(RESULTS, fname)) as f:
            all_runs = list(csv.DictReader(f))

        for col_idx, nl in enumerate(target_noise):
            ax = axes[row_idx][col_idx]
            subset = [r for r in all_runs if abs(float(r['noise_level']) - nl) < 1e-6]

            # Group by model/entanglement/optlevel
            groups = {}
            for r in subset:
                key = (r['model'], r['entanglement'], r['optimization_name'])
                if key not in groups:
                    groups[key] = []
                groups[key].append(float(r['accuracy']) * 100)

            # Order: QSVC full L0/L3, QSVC lin L0/L3, QVE full L0/L3, QVE lin L0/L3, QWE full L0/L3, QWE lin L0/L3
            order = []
            for model in ['QSVC', 'QVE', 'QWE']:
                for ent in ['full', 'linear']:
                    for opt in ['L0', 'L3']:
                        order.append((model, ent, opt))

            data = []
            labels = []
            colors = []
            color_l0 = '#4a90d9'
            color_l3 = '#e85d5d'

            for key in order:
                if key in groups:
                    data.append(groups[key])
                else:
                    data.append([])
                m, e, o = key
                labels.append(f'{m} {e[:3]} {o}')
                colors.append(color_l0 if o == 'L0' else color_l3)

            bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                          medianprops=dict(color='black', linewidth=1.5),
                          whiskerprops=dict(linewidth=0.8),
                          capprops=dict(linewidth=0.8),
                          flierprops=dict(markersize=3))

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
            ax.set_title(f'$p_{{1q}}={nl}$', fontsize=11)
            ax.grid(axis='y', alpha=0.3)

            # Keep the informative range visible: clip the axis above the
            # constant-class floor and mark collapsed (off-scale) groups with
            # a triangle at the bottom edge, so surviving distributions are
            # not compressed into a sliver (referee minor comment 14).
            all_vals = [v for g in data for v in g]
            if all_vals:
                hi = max(all_vals)
                surv = [v for v in all_vals if v > 30]
                if surv and min(all_vals) < 30:
                    lo = min(surv)
                    ax.set_ylim(max(30, lo - 8), hi + 1.5)
                    for xi, g in enumerate(data, start=1):
                        if g and max(g) < 30:
                            ax.plot(xi, max(30, lo - 8) + 0.6, marker='v', color='black',
                                    markersize=5, clip_on=False)
                            ax.annotate(f'{np.median(g):.1f}', (xi, max(30, lo - 8) + 1.0),
                                        ha='center', va='bottom', fontsize=6, rotation=90)

            if col_idx == 0:
                ax.set_ylabel(f'{dataset_label}\nAccuracy (%)', fontsize=11)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_l0, alpha=0.7, label='L0 (baseline)'),
        mpatches.Patch(facecolor=color_l3, alpha=0.7, label='L3 (optimized)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11,
              bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Per-Run Accuracy Distributions at Selected Noise Levels (6-Qubit Primary Study)',
                fontsize=13, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'boxplot_accuracy_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated: boxplot_accuracy_distributions.png')

# ============================================================
# FIGURE 4: Circuit metrics comparison (L0 vs L3) grouped bar
# ============================================================
def generate_circuit_metrics_figure():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4))

    # Data for 6q
    configs_6q = [
        ('ZZ r=2', 124, 124, 204, 196, 150, 142),
        ('ZZ r=3', 179, 181, 306, 323, 225, 200),
        ('Pauli r=2', 124, 124, 204, 196, 150, 142),
    ]
    # Data for 10q
    configs_10q = [
        ('ZZ r=2', 237, 195, 604, 592, 474, 372),
        ('ZZ r=3', 370, 289, 930, 807, 735, 612),
        ('Pauli r=2', 237, 195, 604, 592, 474, 372),
    ]

    all_configs = configs_6q + configs_10q
    labels = [f'6q\n{c[0]}' for c in configs_6q] + [f'10q\n{c[0]}' for c in configs_10q]

    x = np.arange(len(all_configs))
    width = 0.35

    metrics = [
        ('Circuit Depth', [c[1] for c in all_configs], [c[2] for c in all_configs]),
        ('Total Gates', [c[3] for c in all_configs], [c[4] for c in all_configs]),
        ('Two-Qubit (CX) Gates', [c[5] for c in all_configs], [c[6] for c in all_configs]),
    ]

    for ax_idx, (title, l0_vals, l3_vals) in enumerate(metrics):
        ax = axes[ax_idx]
        bars1 = ax.bar(x - width/2, l0_vals, width, label='L0 (baseline)', color='#4a90d9', alpha=0.8)
        bars2 = ax.bar(x + width/2, l3_vals, width, label='L3 (optimized)', color='#e85d5d', alpha=0.8)

        # Add reduction % labels
        for i in range(len(all_configs)):
            if l0_vals[i] > 0:
                pct = (l0_vals[i] - l3_vals[i]) / l0_vals[i] * 100
                if abs(pct) > 0.1:
                    y_pos = max(l0_vals[i], l3_vals[i]) + max(l0_vals) * 0.02
                    sign = '+' if pct < 0 else ''  # negative means L3 is bigger
                    ax.text(x[i], y_pos, f'{sign}{-pct:.1f}%' if pct < 0 else f'−{pct:.1f}%',
                           ha='center', fontsize=7, color='green' if pct > 0 else 'red')

        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Circuit Metrics: L0 (Baseline) vs. L3 (Optimized)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'circuit_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated: circuit_metrics_comparison.png')


if __name__ == '__main__':
    generate_heatmap()
    generate_cohens_d_plot()
    generate_boxplots()
    generate_circuit_metrics_figure()
    print('\nAll enhancement figures generated successfully.')
