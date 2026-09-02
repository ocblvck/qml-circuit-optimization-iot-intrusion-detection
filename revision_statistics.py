#!/usr/bin/env python3
"""Revision statistics for the TQE manuscript (advisor major comments 2, 3, 5, 6).

All analyses reuse archived per-run results; no new quantum experiments are run.

M2a  TOST equivalence tests on every L0-vs-L3 paired contrast, so that the
     Level-1 claim rests on a positive equivalence bound rather than on the
     absence of a significant difference.
M2b  Balanced factorial variance decomposition (partial eta^2) comparing the
     optimization effect against the model-by-noise and entanglement-by-noise
     effects, on MCC and on logit-transformed accuracy.
M6   Normality diagnostics of the within-seed paired differences and Wilcoxon
     signed-rank sensitivity analysis for every contrast.
M5   Paired inference (paired t and Wilcoxon) for the matched quantum-versus-
     classical comparison, replacing the unpaired Welch test.
M3   Exponential-decay fit of off-diagonal kernel variance against error rate.
"""
import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RES = "results/circuit_depth"
OUT = "results/revision_stats"
import os
os.makedirs(OUT, exist_ok=True)

DSNAME = {
    "IoT_Original_Distribution": "IoT",
    "UNSW_2018_IoT_Botnet_Final_10_Best": "UNSW-2018",
    "UNSW_NB15": "UNSW-NB15",
}
QFILES = {
    "IoT_Original_Distribution": f"{RES}/comprehensive_model_noise_runs_iot_original_distribution_6q_5k_v2_noisy_20260310_105202.csv",
    "UNSW_2018_IoT_Botnet_Final_10_Best": f"{RES}/comprehensive_model_noise_runs_unsw_2018_iot_botnet_final_10_best_6q_5k_v2_noisy_20260310_105202.csv",
    "UNSW_NB15": f"{RES}/comprehensive_model_noise_runs_unsw_nb15_6q_5k_v2_noisy_20260415_131112.csv",
}


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


def tost_paired(diffs, margin):
    """Two one-sided tests for practical equivalence of paired differences.

    Returns the TOST p-value: equivalence is declared when it falls below the
    chosen alpha, i.e. the mean difference is bounded inside +/- margin.
    """
    d = np.asarray(diffs, float)
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        # Degenerate case: every seed gives the identical difference.
        return (0.0 if abs(mean) < margin else 1.0), mean, 0.0
    se = sd / np.sqrt(n)
    df = n - 1
    t_lower = (mean + margin) / se       # H0: mean <= -margin
    t_upper = (mean - margin) / se       # H0: mean >= +margin
    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    return max(p_lower, p_upper), mean, se


# ----------------------------------------------------------------------
# M2a / M6 : per-contrast equivalence, normality, Wilcoxon
# ----------------------------------------------------------------------
rows = []
for ds, f in QFILES.items():
    q = pd.read_csv(f)
    for (m, e, nl), g in q.groupby(["model", "entanglement", "noise_level"]):
        piv = g.pivot_table(index="seed", columns="optimization_level", values="accuracy")
        if 0 not in piv.columns or 3 not in piv.columns:
            continue
        piv = piv.dropna()
        d = (piv[3] - piv[0]).values * 100.0  # percentage points
        n = len(d)
        # paired t (reproduces the archived analysis)
        if d.std(ddof=1) == 0:
            t_stat, p_t = (0.0, 1.0)
        else:
            t_stat, p_t = stats.ttest_rel(piv[3].values, piv[0].values)
        # normality of the paired differences
        if np.allclose(d, d[0]):
            sw_W, sw_p = (np.nan, np.nan)
        else:
            sw_W, sw_p = stats.shapiro(d)
        # Wilcoxon signed-rank sensitivity
        if np.allclose(d, 0):
            p_w = 1.0
        else:
            try:
                _, p_w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                p_w = 1.0
        p_tost1, mean_d, se_d = tost_paired(d, 1.0)
        p_tost05, _, _ = tost_paired(d, 0.5)
        ci_lo, ci_hi = (np.nan, np.nan)
        if se_d > 0:
            crit = stats.t.ppf(0.95, n - 1)   # 90% CI, the TOST-compatible interval
            ci_lo, ci_hi = mean_d - crit * se_d, mean_d + crit * se_d
        else:
            ci_lo = ci_hi = mean_d
        rows.append({
            "dataset": DSNAME[ds], "model": m, "entanglement": e, "noise_level": nl,
            "n": n, "mean_delta_pp": mean_d, "ci90_lo": ci_lo, "ci90_hi": ci_hi,
            "p_t": p_t, "p_wilcoxon": p_w, "shapiro_W": sw_W, "shapiro_p": sw_p,
            "p_tost_1pp": p_tost1, "p_tost_0p5pp": p_tost05,
        })

con = pd.DataFrame(rows)
# Holm within each dataset family of 60 contrasts, matching the archived analysis
for col, out in [("p_t", "p_t_holm"), ("p_wilcoxon", "p_wilcoxon_holm")]:
    con[out] = np.nan
    for ds, g in con.groupby("dataset"):
        con.loc[g.index, out] = holm(g[col].values)
con["sig_t_holm"] = con.p_t_holm < 0.05
con["sig_w_holm"] = con.p_wilcoxon_holm < 0.05
con["equiv_1pp"] = con.p_tost_1pp < 0.05
con["equiv_0p5pp"] = con.p_tost_0p5pp < 0.05
con.to_csv(f"{OUT}/contrast_level_revision_stats_6q.csv", index=False)

print("=" * 78)
print("M2a  TOST EQUIVALENCE  (180 six-qubit L0-vs-L3 accuracy contrasts)")
print("=" * 78)
N = len(con)
print(f"total contrasts                     : {N}")
print(f"significant (paired t, Holm)        : {con.sig_t_holm.sum()}")
print(f"equivalent within +/-1.0 pp (TOST)  : {con.equiv_1pp.sum()}  ({100*con.equiv_1pp.mean():.1f}%)")
print(f"equivalent within +/-0.5 pp (TOST)  : {con.equiv_0p5pp.sum()}  ({100*con.equiv_0p5pp.mean():.1f}%)")
neither = (~con.equiv_1pp) & (~con.sig_t_holm)
print(f"neither equivalent nor significant  : {neither.sum()}")
print(f"|mean delta| <= 1 pp                : {(con.mean_delta_pp.abs()<=1).sum()}")
print("\nby dataset (equivalent at 1 pp / significant / total):")
for ds, g in con.groupby("dataset"):
    print(f"  {ds:<11} {g.equiv_1pp.sum():>3} / {g.sig_t_holm.sum():>2} / {len(g)}")
print("\ncontrasts that are NEITHER equivalent (1pp) NOR significant:")
cols = ["dataset", "model", "entanglement", "noise_level", "mean_delta_pp", "ci90_lo", "ci90_hi", "p_t_holm"]
print(con[neither][cols].round(3).to_string(index=False))

print()
print("=" * 78)
print("M6  NORMALITY OF PAIRED DIFFERENCES + WILCOXON SENSITIVITY")
print("=" * 78)
testable = con.shapiro_p.notna()
viol = con[testable & (con.shapiro_p < 0.05)]
print(f"contrasts with testable variation   : {testable.sum()} of {N}")
print(f"Shapiro-Wilk rejects normality (5%) : {len(viol)}  ({100*len(viol)/testable.sum():.1f}% of testable)")
print(f"significant contrasts (paired t)    : {con.sig_t_holm.sum()}")
print(f"significant contrasts (Wilcoxon)    : {con.sig_w_holm.sum()}")
agree = (con.sig_t_holm == con.sig_w_holm).sum()
print(f"agreement between the two tests     : {agree} of {N}")
disagree = con[con.sig_t_holm != con.sig_w_holm]
if len(disagree):
    print("\ncontrasts where the two tests disagree:")
    print(disagree[["dataset", "model", "entanglement", "noise_level", "mean_delta_pp",
                    "p_t_holm", "p_wilcoxon_holm", "shapiro_p"]].round(4).to_string(index=False))
print("\nnormality of the paired differences in the significant contrasts:")
print(con[con.sig_t_holm][["dataset", "model", "entanglement", "noise_level", "mean_delta_pp",
                           "shapiro_W", "shapiro_p", "p_t_holm", "p_wilcoxon_holm"]].round(4).to_string(index=False))


# ----------------------------------------------------------------------
# M2b : balanced factorial variance decomposition
# ----------------------------------------------------------------------
def ss(group_means, counts, grand):
    return float(np.sum(counts * (group_means - grand) ** 2))


def anova_partial_eta(df, response):
    """Exact sums of squares for a balanced factorial design.

    Factors: optimization, model, entanglement, noise, plus seed as a block.
    Reports the main effects and the interactions that the two-level thesis
    contrasts against one another.
    """
    y = df[response].values
    grand = y.mean()
    sst = float(np.sum((y - grand) ** 2))
    facs = {"opt": "optimization_level", "model": "model",
            "ent": "entanglement", "noise": "noise_level", "seed": "seed"}
    terms = {}
    # main effects
    for name, col in facs.items():
        g = df.groupby(col)[response]
        terms[name] = ss(g.mean().values, g.size().values, grand)
    # two-way interactions of interest
    def inter(a, b):
        ca, cb = facs[a], facs[b]
        g = df.groupby([ca, cb])[response]
        cell, cnt = g.mean(), g.size()
        ma = df.groupby(ca)[response].mean()
        mb = df.groupby(cb)[response].mean()
        tot = 0.0
        for (la, lb), v in cell.items():
            eff = v - ma[la] - mb[lb] + grand
            tot += cnt[(la, lb)] * eff ** 2
        return float(tot)
    for a, b in [("opt", "noise"), ("model", "noise"), ("ent", "noise"),
                 ("model", "ent"), ("opt", "model"), ("opt", "ent")]:
        terms[f"{a}x{b}"] = inter(a, b)
    sse = sst - sum(terms.values())
    out = []
    for k, v in terms.items():
        out.append({"term": k, "SS": v, "eta2": v / sst,
                    "partial_eta2": v / (v + sse) if (v + sse) > 0 else 0.0})
    return pd.DataFrame(out).sort_values("eta2", ascending=False), sst, sse


print()
print("=" * 78)
print("M2b  VARIANCE DECOMPOSITION  (balanced factorial, per dataset)")
print("=" * 78)
dec_rows = []
for ds, f in QFILES.items():
    q = pd.read_csv(f)
    q = q.dropna(subset=["accuracy", "mcc"])
    # logit accuracy, clipped away from the bounds
    a = np.clip(q["accuracy"].values, 1e-3, 1 - 1e-3)
    q["logit_acc"] = np.log(a / (1 - a))
    for resp in ["mcc", "logit_acc"]:
        tab, sst, sse = anova_partial_eta(q, resp)
        tab.insert(0, "response", resp)
        tab.insert(0, "dataset", DSNAME[ds])
        dec_rows.append(tab)
        print(f"\n{DSNAME[ds]}  response={resp}   (SS_total={sst:.3f})")
        print(tab.round(4).to_string(index=False))
dec = pd.concat(dec_rows, ignore_index=True)
dec.to_csv(f"{OUT}/variance_decomposition_6q.csv", index=False)

print("\nSUMMARY  eta^2 (share of total variance), averaged over the three datasets:")
piv = dec.pivot_table(index="term", columns="response", values="eta2", aggfunc="mean")
print((piv * 100).round(3).sort_values("mcc", ascending=False).to_string())


# ----------------------------------------------------------------------
# M5 : paired quantum-versus-classical inference
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("M5  PAIRED QUANTUM-VERSUS-CLASSICAL COMPARISON")
print("=" * 78)
KERNEL_MODELS = ["RBF-SVM", "LinearSVM"]
cl = pd.read_csv(f"{RES}/classical_baseline_runs_classical_6q_5k.csv")


def quantum_best_series(ds):
    q = pd.read_csv(QFILES[ds])
    q = q[q.noise_level == q.noise_level.min()]
    best, best_acc, best_s = None, -1, None
    for (m, e), g in q.groupby(["model", "entanglement"]):
        s = g.groupby("seed").accuracy.mean()
        if s.mean() > best_acc:
            best, best_acc, best_s = f"{m}/{e}", s.mean(), s
    return best, best_s


def classical_best_series(ds, restrict=None):
    c = cl[(cl.dataset == ds) & (cl.feature_regime == "quantum_features")]
    if restrict is not None:
        c = c[c.model.isin(restrict)]
    best, best_acc, best_s = None, -1, None
    for m, g in c.groupby("model"):
        s = g.set_index("seed").accuracy
        if s.mean() > best_acc:
            best, best_acc, best_s = m, s.mean(), s
    return best, best_s


pair_rows = []
for tag, restrict in [("vs_best_classical", None), ("vs_classical_kernel", KERNEL_MODELS)]:
    for ds in QFILES:
        qname, qs = quantum_best_series(ds)
        cname, cs = classical_best_series(ds, restrict)
        common = sorted(set(qs.index) & set(cs.index))
        qv = qs.loc[common].values * 100
        cv = cs.loc[common].values * 100
        d = qv - cv
        t_stat, p_t = stats.ttest_rel(qv, cv)
        try:
            _, p_w = stats.wilcoxon(d, alternative="two-sided")
        except ValueError:
            p_w = 1.0
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
        sw = stats.shapiro(d) if not np.allclose(d, d[0]) else (np.nan, np.nan)
        pair_rows.append({
            "comparison": tag, "dataset": DSNAME[ds], "quantum": qname, "classical": cname,
            "n_pairs": len(common), "q_acc": qv.mean(), "c_acc": cv.mean(),
            "delta_pp": d.mean(), "sd_diff": d.std(ddof=1),
            "t_paired": t_stat, "p_paired": p_t, "p_wilcoxon": p_w,
            "cohens_dz": dz, "shapiro_p": sw[1],
        })
pairdf = pd.DataFrame(pair_rows)
for tag, g in pairdf.groupby("comparison"):
    pairdf.loc[g.index, "p_holm"] = holm(g.p_paired.values)
    pairdf.loc[g.index, "p_wilcoxon_holm"] = holm(g.p_wilcoxon.values)
pairdf.to_csv(f"{OUT}/quantum_vs_classical_paired_6q.csv", index=False)
print(pairdf.round(4).to_string(index=False))

print("\ncomparison with the archived unpaired (Welch) analysis:")
old = pd.read_csv(f"{RES}/quantum_vs_classical_significance_6q.csv")
for _, r in pairdf.iterrows():
    o = old[(old.comparison == r.comparison) & (old.dataset == r.dataset)]
    if len(o):
        o = o.iloc[0]
        print(f"  {r.comparison:<20} {r.dataset:<11} delta {r.delta_pp:+7.3f} pp | "
              f"Welch d={o.cohens_d:+6.2f} p_holm={o.p_holm:.2e} -> "
              f"paired dz={r.cohens_dz:+6.2f} p_holm={r.p_holm:.2e}")


# ----------------------------------------------------------------------
# M3 : exponential decay of off-diagonal variance in the error rate
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("M3  EXPONENTIAL-DECAY FIT OF OFF-DIAGONAL KERNEL VARIANCE")
print("=" * 78)
fit_rows = []
for f in sorted(glob.glob(f"{RES}/concentration_summary_*.csv")):
    c = pd.read_csv(f)
    for (ds, mp, nq), g in c.groupby(["dataset", "map", "num_qubits"]):
        g = g.sort_values("noise_level")
        if len(g) < 4:
            continue
        p = g.noise_level.values
        v = g.offdiag_var_mean.values
        if np.any(v <= 0):
            continue
        ln = np.log(v)
        sl, ic, r, pv, se = stats.linregress(p, ln)
        fit_rows.append({
            "file": f.split("/")[-1], "dataset": ds, "map": mp, "qubits": nq,
            "n_levels": len(g), "decay_rate_per_unit_p": -sl, "r2": r ** 2,
            "p_value": pv, "var_at_0": v[0], "var_at_max": v[-1],
            "orders_of_magnitude": np.log10(v[0] / v[-1]),
        })
fits = pd.DataFrame(fit_rows).sort_values(["qubits", "dataset", "map"])
fits.to_csv(f"{OUT}/concentration_exponential_fits.csv", index=False)
show = ["dataset", "map", "qubits", "n_levels", "decay_rate_per_unit_p", "r2",
        "orders_of_magnitude", "var_at_0", "var_at_max"]
print(fits[show].round(4).to_string(index=False))
print("\nfull-entanglement ZZ (the collapsing kernel), by width:")
zz = fits[fits["map"].str.contains("ZZ-full", case=False, na=False)]
print(zz[show].round(4).to_string(index=False))

print(f"\nAll outputs written to {OUT}/")
