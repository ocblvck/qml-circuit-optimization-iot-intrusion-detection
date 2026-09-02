#!/usr/bin/env python3
"""Quantum-kernel vs best-classical significance tests on the matched
quantum-feature (n-dim PCA) representation, 6 qubits, ideal setting.

For each dataset we compare the best quantum kernel model against the best
classical model evaluated on the identical n-dimensional PCA features, using
30 independent runs per side. We report Welch's t-test, Mann-Whitney U, and
Cohen's d, with Holm correction across the three datasets.
"""
import glob
import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj

RES = "results/circuit_depth"
DSMAP = {
    "IoT_Original_Distribution": "IoT",
    "UNSW_2018_IoT_Botnet_Final_10_Best": "UNSW-2018",
    "UNSW_NB15": "UNSW-NB15",
}
QFILES = {
    "IoT_Original_Distribution": f"{RES}/comprehensive_model_noise_runs_iot_original_distribution_6q_5k_v2_noisy_20260310_105202.csv",
    "UNSW_2018_IoT_Botnet_Final_10_Best": f"{RES}/comprehensive_model_noise_runs_unsw_2018_iot_botnet_final_10_best_6q_5k_v2_noisy_20260310_105202.csv",
    "UNSW_NB15": f"{RES}/comprehensive_model_noise_runs_unsw_nb15_6q_5k_v2_noisy_20260415_131112.csv",
}


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0


def quantum_best(ds):
    q = pd.read_csv(QFILES[ds])
    q = q[q.noise_level == q.noise_level.min()].copy()
    # per (model,ent): mean per seed across opt levels -> 30 values
    best, best_acc, best_vals = None, -1, None
    for (m, e), g in q.groupby(["model", "entanglement"]):
        vals = g.groupby("seed").accuracy.mean().values
        if vals.mean() > best_acc:
            best, best_acc, best_vals = f"{m}/{e}", vals.mean(), vals
    return best, best_vals


def classical_best(ds, regime, restrict=None):
    c = pd.read_csv(f"{RES}/classical_baseline_runs_classical_6q_5k.csv")
    c = c[(c.dataset == ds) & (c.feature_regime == regime)]
    if restrict is not None:
        c = c[c.model.isin(restrict)]
    best, best_acc, best_vals = None, -1, None
    for m, g in c.groupby("model"):
        vals = g.accuracy.values
        if vals.mean() > best_acc:
            best, best_acc, best_vals = m, vals.mean(), vals
    return best, best_vals


KERNEL_MODELS = ["RBF-SVM", "LinearSVM"]


def run_comparison(restrict, tag):
    rows = []
    for ds in DSMAP:
        qname, qv = quantum_best(ds)
        cname, cv = classical_best(ds, "quantum_features", restrict)
        n = min(len(qv), len(cv))
        qv2, cv2 = qv[:n], cv[:n]
        t, p_t = stats.ttest_ind(qv2, cv2, equal_var=False)
        u, p_u = stats.mannwhitneyu(qv2, cv2, alternative="two-sided")
        d = cohens_d(qv2, cv2)
        rows.append({
            "comparison": tag,
            "dataset": DSMAP[ds],
            "quantum": qname, "q_acc": qv2.mean() * 100, "q_sd": qv2.std(ddof=1) * 100,
            "classical": cname, "c_acc": cv2.mean() * 100, "c_sd": cv2.std(ddof=1) * 100,
            "delta_pp": (qv2.mean() - cv2.mean()) * 100,
            "welch_t": t, "p_welch": p_t, "mwu_p": p_u, "cohens_d": d, "n": n,
        })
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_welch"].values)
    return df


df_all = run_comparison(None, "vs_best_classical")
df_ker = run_comparison(KERNEL_MODELS, "vs_classical_kernel")
df = pd.concat([df_all, df_ker], ignore_index=True)
df.to_csv(f"{RES}/quantum_vs_classical_significance_6q.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", None)
print(df.round(4).to_string(index=False))
print("\nSaved:", f"{RES}/quantum_vs_classical_significance_6q.csv")
