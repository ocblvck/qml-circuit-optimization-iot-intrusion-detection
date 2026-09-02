#!/usr/bin/env python
"""Real-hardware (and fake-backend dry-run) fidelity quantum-kernel evaluation.

This is the hardware counterpart to the simulated kernels in
``circuit_depth_experiment.py`` / ``device_noise_validation.py``. On real QPUs we
cannot read statevectors or density matrices, so each kernel entry is estimated
with the compute-uncompute circuit  U(x_j)^dagger U(x_i)  measured in the
computational basis:  K_ij ~= P(000...0).

Design constraints (read before running on real hardware):
  * The Gram matrix needs N_train*(N_train-1)/2 circuits plus N_test*N_train,
    each at `--shots`. KEEP THE SUBSAMPLE SMALL (e.g. 40 train / 20 test).
  * Always validate on a fake backend first (default) -- it costs zero QPU time.
  * Only pass --real once the dry run looks sane.

Feature maps mirror the paper exactly:
    QSVC = ZZ(reps=2)
    QVE  = Z(reps=1)  + ZZ(reps=2)   (majority vote)
    QWE  = ZZ(reps=2) + Pauli(reps=1) (equal-weight soft vote here; the paper's
           validation-accuracy weighting can be added once it is on hardware)

Run (dry run, no QPU time):
    python run_hardware_kernel.py \
        --dataset UNSW_2018_IoT_Botnet_Final_10_Best.csv --model QSVC \
        --num-qubits 6 --train-size 40 --test-size 20 --shots 4096

Run on real hardware (after saving your IBM account once):
    ... --real --backend ibm_sherbrooke --mitigation
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Feature-map recipe per model (matches device_noise_validation.MODEL_MAPS and
# feature_map_ablation.COMPOSITIONS for the committee-size variants).
MODEL_MAPS = {
    "QSVC": [("ZZ", 2)],
    "QVE": [("Z", 1), ("ZZ", 2)],
    "QWE": [("ZZ", 2), ("Pauli", 1)],
    # Feature-map count ablation variants of QVE (hard majority voting).
    "QVE3": [("Z", 1), ("ZZ", 2), ("Pauli", 1)],
    "QVE4": [("Z", 1), ("ZZ", 2), ("Pauli", 1), ("Custom", 1)],
}
# Models whose ensemble uses hard majority voting (soft-probability tie-break).
MAJORITY_VOTE_MODELS = {"QVE", "QVE3", "QVE4"}
RESULTS_DIR = Path(__file__).parent / "results" / "hardware"


# --------------------------------------------------------------------------- #
# Feature maps (same configuration as create_feature_map in the paper code)
# --------------------------------------------------------------------------- #
def make_feature_map(num_qubits: int, map_type: str, reps: int, entanglement: str):
    if map_type == "Z":
        return ZFeatureMap(num_qubits, reps=reps)
    if map_type == "ZZ":
        return ZZFeatureMap(num_qubits, reps=reps, entanglement=entanglement)
    if map_type == "Pauli":
        return PauliFeatureMap(
            num_qubits, reps=reps, paulis=["Z", "ZZ"], entanglement=entanglement
        )
    if map_type == "Custom":
        # Enhanced-expressibility map (matches create_feature_map 'Custom').
        fm = QuantumCircuit(num_qubits)
        params = ParameterVector("x", num_qubits)
        for _ in range(reps):
            for i in range(num_qubits):
                fm.h(i)
            for i in range(num_qubits):
                fm.rz(params[i], i)
                fm.ry(params[i], i)
            if entanglement == "linear":
                for i in range(num_qubits - 1):
                    fm.cx(i, i + 1)
            else:  # full
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        fm.cx(i, j)
        return fm
    raise ValueError(f"Unknown map_type {map_type!r}")


# --------------------------------------------------------------------------- #
# Preprocessing & loading reuse the paper code for byte-faithful parity with the
# simulated runs (CSV delimiter + label handling + SelectKBest/PCA/angle encode).
# Requires the conda `qiskit` env (has cuML), as in the rest of the project.
# --------------------------------------------------------------------------- #
from device_noise_validation import load_dataset  # noqa: E402
from circuit_depth_experiment import DataProcessor  # noqa: E402


def balanced_subsample(X, y, n, seed):
    """Pick ~n samples with balanced classes."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = max(1, n // len(classes))
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, size=min(per, len(ci)), replace=False))
    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


# --------------------------------------------------------------------------- #
# Compute-uncompute circuit builder
# --------------------------------------------------------------------------- #
def fidelity_circuit(fm: QuantumCircuit, a: np.ndarray, b: np.ndarray) -> QuantumCircuit:
    """U(b)^dagger U(a) followed by a full measurement; K ~= P(all-zeros)."""
    ua = fm.assign_parameters(a)
    ub_inv = fm.assign_parameters(b).inverse()
    qc = ua.compose(ub_inv)
    qc.measure_all()
    return qc


def all_zero_prob(counts: dict, num_qubits: int, shots: int) -> float:
    zero = "0" * num_qubits
    return counts.get(zero, 0) / shots


# --------------------------------------------------------------------------- #
# Ideal exact kernel (noiseless statevector inner products; no sampling shots).
# This is the cheapest, exact reference rung of the comparison ladder.
# --------------------------------------------------------------------------- #
def ideal_gram(fm, X):
    from qiskit.quantum_info import Statevector
    svs = [Statevector(fm.assign_parameters(x)) for x in X]
    n = len(svs)
    K = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = abs(svs[i].inner(svs[j])) ** 2
    return K


def ideal_rect(fm, X_test, X_train):
    from qiskit.quantum_info import Statevector
    sa = [Statevector(fm.assign_parameters(x)) for x in X_test]
    sb = [Statevector(fm.assign_parameters(x)) for x in X_train]
    return np.array([[abs(s1.inner(s2)) ** 2 for s2 in sb] for s1 in sa])


# --------------------------------------------------------------------------- #
# Backend / sampler wiring (ideal exact | fake-backend dry run | real hardware)
# --------------------------------------------------------------------------- #
def get_real_backend(args):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    return (
        service.backend(args.backend)
        if args.backend
        else service.least_busy(operational=True, simulator=False,
                                min_num_qubits=args.num_qubits)
    )


def make_real_sampler(session_or_backend, args):
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=session_or_backend)
    sampler.options.default_shots = args.shots
    if args.mitigation:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True  # TREX readout mitigation
    return sampler


def make_fake_backend_sampler(args):
    """Local noisy simulation of a fake backend (no QPU time)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    from qiskit_ibm_runtime import fake_provider as fp
    fake_name = args.backend or "FakeFez"
    fake = getattr(fp, fake_name)()
    backend = AerSimulator.from_backend(fake)
    sampler = AerSamplerV2.from_backend(backend, default_shots=args.shots)
    return fake, sampler


def _result_with_retry(job, retries=8, base_delay=30):
    """Fetch a job's result, retrying transient server/network errors.

    Hardware jobs execute and their results persist on the IBM servers, so a
    transient outage during retrieval (e.g. Cloudflare "525 SSL handshake
    failed", 502/503/504, or a dropped connection) must not discard days of
    already-completed queue work. We retry the fetch with exponential backoff
    rather than crashing the run.
    """
    for attempt in range(1, retries + 1):
        try:
            return job.result()
        except Exception as exc:  # noqa: BLE001 - intentional broad retry
            msg = str(exc)
            transient = any(s in msg for s in
                            ("525", "502", "503", "504", "handshake",
                             "Connection", "connection", "timeout", "Timeout",
                             "Max retries", "Server Error"))
            if attempt == retries or not transient:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"    [retry {attempt}/{retries}] transient error fetching "
                  f"result; retrying in {delay}s: {msg[:120]}", flush=True)
            time.sleep(delay)


def run_circuits(circuits, pm, sampler, mode, shots, num_qubits, batch=300):
    """Transpile to ISA, submit ALL chunks first, then collect results.

    Submitting every chunk up front (instead of submit->block->submit) keeps a
    Batch session busy back-to-back rather than idling across multi-hour queue
    waits, which previously tripped the session TTL ("Session has been closed").
    """
    isa = pm.run(circuits)
    jobs = []
    for i in range(0, len(isa), batch):
        jobs.append(sampler.run(isa[i : i + batch]))
    print(f"    submitted {len(jobs)} job(s) covering {len(isa)} circuits "
          f"[{mode}]", flush=True)
    probs = []
    for k, job in enumerate(jobs, 1):
        res = _result_with_retry(job)
        for r in res:
            counts = r.data.meas.get_counts()
            probs.append(all_zero_prob(counts, num_qubits, shots))
        print(f"    collected {min(k * batch, len(isa))}/{len(isa)} circuits "
              f"[{mode}]", flush=True)
    return np.array(probs)


def gram_matrix(fm, X, pm, sampler, mode, shots, nq):
    """Symmetric train Gram via unique upper-triangle compute-uncompute circuits."""
    n = len(X)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    circs = [fidelity_circuit(fm, X[i], X[j]) for i, j in pairs]
    vals = run_circuits(circs, pm, sampler, mode, shots, nq)
    K = np.eye(n)
    for (i, j), v in zip(pairs, vals):
        K[i, j] = K[j, i] = v
    return K


def rect_matrix(fm, X_test, X_train, pm, sampler, mode, shots, nq):
    circs = [fidelity_circuit(fm, a, b) for a in X_test for b in X_train]
    vals = run_circuits(circs, pm, sampler, mode, shots, nq)
    return vals.reshape(len(X_test), len(X_train))


def kernels_for_model(model):
    return MODEL_MAPS[model]


def fit_eval_model(args, X_tr, y_tr, X_te, y_te, eval_gram, eval_rect, depth_fn):
    """Build per-component kernels, train SVCs, return ensemble preds + metadata."""
    preds, probas, depths = [], [], []
    n_circ = 0
    for map_type, reps in kernels_for_model(args.model):
        fm = make_feature_map(args.num_qubits, map_type, reps, args.entanglement)
        d = depth_fn(fm)
        depths.append(d)
        print(f"  [{map_type} reps={reps}] depth={d}", flush=True)
        Ktr = eval_gram(fm)
        Kte = eval_rect(fm)
        n_circ += len(X_tr) * (len(X_tr) - 1) // 2 + len(X_te) * len(X_tr)
        svc = SVC(kernel="precomputed", class_weight="balanced", probability=True,
                  random_state=args.seed)
        svc.fit(Ktr, y_tr)
        preds.append(svc.predict(Kte))
        probas.append(svc.predict_proba(Kte)[:, 1] if len(np.unique(y_tr)) == 2
                      else svc.predict_proba(Kte).max(1))
    P = np.array(preds)
    y_proba = np.mean(probas, axis=0)
    if P.shape[0] > 1 and args.model in MAJORITY_VOTE_MODELS:
        # Exact hard majority vote; ties (possible for even committees) broken
        # by the mean soft probability, matching the feature-map ablation rule.
        votes = P.mean(0)
        y_pred = (votes > 0.5).astype(int)
        tie = np.isclose(votes, 0.5)
        y_pred[tie] = (y_proba[tie] >= 0.5).astype(int)
    elif P.shape[0] > 1:
        y_pred = (P.mean(0) >= 0.5).astype(int)
    else:
        y_pred = P[0]
    return y_pred, y_proba, depths, n_circ


def make_latex_table():
    """Aggregate all results/hardware/*.json into a LaTeX comparison table."""
    import glob
    rows = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        d = json.loads(Path(f).read_text())
        rung = d["mode"]
        if d["mode"] == "real":
            rung = "real+mit" if d.get("mitigation") else "real"
        rows.append((d["dataset"].split(".")[0], d["model"], d["num_qubits"],
                     d["entanglement"], rung, d["backend"],
                     d["metrics"]["accuracy"] * 100, d["metrics"]["mcc"]))
    if not rows:
        print("No result JSONs found in", RESULTS_DIR)
        return
    print("% Auto-generated from results/hardware/*.json")
    print("\\begin{tabular}{|l|l|c|l|l|c|c|}")
    print("\\hline")
    print("Dataset & Model & $n_q$ & Ent. & Condition & Acc.\\ (\\%) & MCC \\\\")
    print("\\hline")
    for r in rows:
        print(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & "
              f"{r[6]:.2f} & {r[7]:.3f} \\\\")
    print("\\hline")
    print("\\end{tabular}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="CSV (label col handled by paper loader)")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--model", default="QSVC", choices=list(MODEL_MAPS))
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--entanglement", default="full", choices=["full", "linear"])
    ap.add_argument("--train-size", type=int, default=40)
    ap.add_argument("--test-size", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=5000,
                    help="Rows loaded before subsampling (matches paper loader)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ideal", action="store_true",
                    help="Exact noiseless statevector kernel (no backend, no shots)")
    ap.add_argument("--real", action="store_true", help="Use a real IBM QPU (Batch)")
    ap.add_argument("--backend", default=None, help="Backend name (real or Fake*)")
    ap.add_argument("--mitigation", action="store_true",
                    help="Enable dynamical decoupling + TREX (real backend only)")
    ap.add_argument("--make-table", action="store_true",
                    help="Print a LaTeX table from saved results and exit")
    args = ap.parse_args()

    if args.make_table:
        make_latex_table()
        return
    if not args.dataset:
        ap.error("--dataset is required unless --make-table is given")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # --- load + subsample + preprocess (paper-faithful) ------------------
    path = Path(args.data_dir) / args.dataset
    X, y = load_dataset(str(path), args.num_qubits, sample_size=args.pool_size)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    X_tr_raw, y_tr = balanced_subsample(X_tr_raw, y_tr, args.train_size, args.seed)
    X_te_raw, y_te = balanced_subsample(X_te_raw, y_te, args.test_size, args.seed + 1)
    pre = DataProcessor(num_qubits=args.num_qubits, random_seed=args.seed)
    pre.fit(X_tr_raw, y_tr)
    X_tr, X_te = pre.transform(X_tr_raw), pre.transform(X_te_raw)
    print(f"Subsample: {len(X_tr)} train / {len(X_te)} test, {args.num_qubits} qubits")

    # --- dispatch on mode -------------------------------------------------
    backend = None
    if args.ideal:
        mode, bname = "ideal", "statevector"
        print("Mode: ideal (exact statevector)  no shots")
        eval_gram = lambda fm: ideal_gram(fm, X_tr)        # noqa: E731
        eval_rect = lambda fm: ideal_rect(fm, X_te, X_tr)  # noqa: E731
        depth_fn = lambda fm: fm.decompose().depth()       # noqa: E731
        y_pred, y_proba, depths, n_circ = fit_eval_model(
            args, X_tr, y_tr, X_te, y_te, eval_gram, eval_rect, depth_fn)
    elif args.real:
        from qiskit_ibm_runtime import Batch
        mode = "real"
        backend = get_real_backend(args)
        bname = backend.name
        pm = generate_preset_pass_manager(optimization_level=args.opt_level,
                                          backend=backend)
        print(f"Backend: {bname}  mode=real  shots={args.shots}  "
              f"mitigation={args.mitigation}  queue="
              f"{backend.status().pending_jobs}")
        depth_fn = lambda fm: pm.run(fidelity_circuit(fm, X_tr[0], X_tr[1])).depth()  # noqa: E731
        with Batch(backend=backend) as session:
            sampler = make_real_sampler(session, args)
            eval_gram = lambda fm: gram_matrix(                 # noqa: E731
                fm, X_tr, pm, sampler, mode, args.shots, args.num_qubits)
            eval_rect = lambda fm: rect_matrix(                 # noqa: E731
                fm, X_te, X_tr, pm, sampler, mode, args.shots, args.num_qubits)
            y_pred, y_proba, depths, n_circ = fit_eval_model(
                args, X_tr, y_tr, X_te, y_te, eval_gram, eval_rect, depth_fn)
    else:
        mode = "fake"
        backend, sampler = make_fake_backend_sampler(args)
        bname = backend.name if not callable(backend.name) else backend.name()
        pm = generate_preset_pass_manager(optimization_level=args.opt_level,
                                          backend=backend)
        print(f"Backend: {bname}  mode=fake  shots={args.shots}")
        depth_fn = lambda fm: pm.run(fidelity_circuit(fm, X_tr[0], X_tr[1])).depth()  # noqa: E731
        eval_gram = lambda fm: gram_matrix(                    # noqa: E731
            fm, X_tr, pm, sampler, mode, args.shots, args.num_qubits)
        eval_rect = lambda fm: rect_matrix(                    # noqa: E731
            fm, X_te, X_tr, pm, sampler, mode, args.shots, args.num_qubits)
        y_pred, y_proba, depths, n_circ = fit_eval_model(
            args, X_tr, y_tr, X_te, y_te, eval_gram, eval_rect, depth_fn)

    # --- metrics + persist ------------------------------------------------
    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "mcc": float(matthews_corrcoef(y_te, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_te, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "timestamp_utc": stamp,
        "mode": mode,
        "backend": bname,
        "dataset": args.dataset,
        "model": args.model,
        "num_qubits": args.num_qubits,
        "entanglement": args.entanglement,
        "train_size": int(len(X_tr)),
        "test_size": int(len(X_te)),
        "shots": args.shots if mode != "ideal" else 0,
        "opt_level": args.opt_level,
        "mitigation": bool(args.mitigation and mode == "real"),
        "depths": depths,
        "total_circuits": int(n_circ),
        "metrics": metrics,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    try:
        if mode == "real" and backend is not None:
            props = backend.properties()
            out["calibration_last_update"] = str(getattr(props, "last_update_date", ""))
    except Exception:
        pass

    fname = RESULTS_DIR / f"hw_{args.model}_{args.num_qubits}q_{mode}_{stamp}.json"
    fname.write_text(json.dumps(out, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved -> {fname}")


if __name__ == "__main__":
    main()
