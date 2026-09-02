#!/usr/bin/env python3
"""
Classical Baselines for Quantum Kernel IoT Intrusion Detection
==============================================================

Research question
-----------------
The main study (circuit_depth_experiment.py) and the feature-map ablation
report quantum-kernel performance (QSVC / QVE / QWE) but contain NO classical
comparison. A reviewer will immediately ask: are the quantum kernels actually
better than a Random Forest or an RBF-SVM on the very same features?

This harness answers that by running strong classical baselines under the
EXACT same conditions as the quantum experiments:

    * identical stratified 70/30 train/test splits (same seeds)
    * identical leakage-safe preprocessing (DataProcessor:
      SelectKBest -> StandardScaler -> PCA(n) -> MinMaxScaler[0, pi])
    * identical metric suite (calculate_all_metrics)

Two feature regimes are evaluated so the comparison is honest in both
directions:

    quantum_features : the n-qubit PCA features the quantum kernels actually
                       consume (apples-to-apples; "can classical match quantum
                       on the same compressed encoding?")
    full_features    : all cleaned numeric features with standardization only
                       (the classical ceiling; "what does classical achieve
                       when not bottlenecked to n qubits?")

Models
------
    RBF-SVM        sklearn.svm.SVC(kernel='rbf', probability=True)
    LinearSVM      sklearn.svm.SVC(kernel='linear', probability=True)
    RandomForest   sklearn.ensemble.RandomForestClassifier
    GradientBoost  XGBoost if available, else HistGradientBoostingClassifier
    MLP            sklearn.neural_network.MLPClassifier

These are CPU models (fast on 5000 samples x n features); no GPU is required.

Outputs (written to results/circuit_depth/)
-------------------------------------------
    classical_baseline_runs_<session>.csv     per-run metrics
    classical_baseline_summary_<session>.csv  mean/std/CI per (model, regime)
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --- Reuse the exact building blocks from the main study (no modification) ---
from circuit_depth_experiment import (
    RESULTS_DIR,
    DataProcessor,
    calculate_all_metrics,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # noqa: BLE001 - optional dependency
    _HAS_XGB = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("classical_baselines")

# Seeds mirror circuit_depth_experiment.py / feature_map_ablation.py exactly so
# every classical run is paired with the corresponding quantum run by seed.
ALL_SEEDS = [
    42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
    7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
    17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624,
]


# ----------------------------------------------------------------------------
# Data loading (mirrors feature_map_ablation.load_dataset)
# ----------------------------------------------------------------------------
def _detect_csv_delimiter(dataset_path: str) -> str:
    import csv
    with open(dataset_path, 'r', encoding='utf-8', errors='ignore', newline='') as handle:
        sample = ''.join(handle.readline() for _ in range(5))
    if not sample:
        return ','
    try:
        return csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
    except csv.Error:
        return ';' if sample.count(';') > sample.count(',') else ','


def load_dataset(dataset_path: str, num_qubits: int, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    delimiter = _detect_csv_delimiter(dataset_path)
    df = pd.read_csv(dataset_path, sep=delimiter, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]
    processor = DataProcessor(num_qubits=num_qubits)
    X, y = processor.prepare_data(df, sample_size=sample_size)
    return X, y


# ----------------------------------------------------------------------------
# Classifier factory (fresh, independent estimator per run/seed)
# ----------------------------------------------------------------------------
def build_classifier(model_name: str, seed: int):
    if model_name == 'RBF-SVM':
        return SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                   probability=True, random_state=seed)
    if model_name == 'LinearSVM':
        return SVC(kernel='linear', C=1.0, class_weight='balanced',
                   probability=True, random_state=seed)
    if model_name == 'RandomForest':
        return RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                      n_jobs=-1, random_state=seed)
    if model_name == 'GradientBoost':
        if _HAS_XGB:
            return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric='logloss', tree_method='hist',
                                 n_jobs=-1, random_state=seed)
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              random_state=seed)
    if model_name == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                             alpha=1e-4, max_iter=500, early_stopping=True,
                             random_state=seed)
    raise ValueError(f"Unknown model: {model_name}")


def gradient_boost_label() -> str:
    return 'XGBoost' if _HAS_XGB else 'HistGradientBoosting'


# ----------------------------------------------------------------------------
# Feature preparation per regime (leakage-safe: fit on train only)
# ----------------------------------------------------------------------------
def prepare_features(regime: str, num_qubits: int, seed: int,
                     X_train_raw: np.ndarray, X_test_raw: np.ndarray,
                     y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_train, X_test) for the requested feature regime.

    quantum_features : identical pipeline to the quantum kernels
                       (SelectKBest -> StandardScaler -> PCA(n) -> MinMaxScaler[0, pi]).
    full_features    : all cleaned numeric features, standardized only.
    """
    if regime == 'quantum_features':
        processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
        processor.fit(X_train_raw, y_train)
        return processor.transform(X_train_raw), processor.transform(X_test_raw)
    if regime == 'full_features':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return X_train, X_test
    raise ValueError(f"Unknown feature regime: {regime}")


# ----------------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------------
def execute_baseline_run(dataset_name: str, X: np.ndarray, y: np.ndarray,
                         num_qubits: int, regime: str, model_name: str,
                         seed: int, run_idx: int) -> Dict[str, Any]:
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        X_train, X_test = prepare_features(
            regime, num_qubits, seed, X_train_raw, X_test_raw, y_train
        )

        clf = build_classifier(model_name, seed)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        train_time = time.perf_counter() - t0

        y_proba = None
        try:
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(X_test)
        except Exception:  # noqa: BLE001
            y_proba = None

        metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba=y_proba,
                                        train_time=train_time)
        return {
            'status': 'success',
            'dataset': dataset_name,
            'model': model_name,
            'feature_regime': regime,
            'num_qubits': num_qubits,
            'run_idx': run_idx,
            'seed': seed,
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'f1_score': metrics['f1_score'],
            'mcc': metrics['mcc'],
            'roc_auc': metrics['roc_auc'],
            'g_mean': metrics['g_mean'],
            'training_time': metrics['training_time'],
        }
    except Exception as exc:  # noqa: BLE001 - record failure, keep batch alive
        logger.warning(f"Run failed ({dataset_name}/{model_name}/{regime}, "
                       f"seed={seed}): {exc}")
        return {
            'status': 'failed', 'dataset': dataset_name, 'model': model_name,
            'feature_regime': regime, 'num_qubits': num_qubits,
            'run_idx': run_idx, 'seed': seed, 'error': str(exc),
        }


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def _ci95(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(n))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ['dataset', 'model', 'feature_regime', 'num_qubits']
    for key, g in df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row['n_runs'] = len(g)
        for metric in ['accuracy', 'balanced_accuracy', 'f1_score', 'mcc',
                       'roc_auc', 'g_mean', 'training_time']:
            vals = g[metric].to_numpy(dtype=float)
            row[f'{metric}_mean'] = float(np.mean(vals))
            row[f'{metric}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{metric}_ci95'] = _ci95(vals)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classical baselines for quantum kernel IoT NID")
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset CSV paths')
    parser.add_argument('--num_qubits', type=int, default=6,
                        help='PCA target dim for quantum_features regime (matches qubit count)')
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--n_runs', type=int, default=30)
    parser.add_argument('--models', nargs='+',
                        default=['RBF-SVM', 'LinearSVM', 'RandomForest', 'GradientBoost', 'MLP'])
    parser.add_argument('--regimes', nargs='+',
                        default=['quantum_features', 'full_features'])
    parser.add_argument('--session_id', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny correctness test (sample_size=300, n_runs=2)')
    args = parser.parse_args()

    if args.smoke:
        args.sample_size = min(args.sample_size, 300)
        args.n_runs = 2
        logger.info("SMOKE TEST mode: sample_size=300, n_runs=2")

    session_id = args.session_id or f"classical_{time.strftime('%Y%m%d_%H%M%S')}"
    seeds = ALL_SEEDS[:args.n_runs]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = []
    for ds in args.datasets:
        if Path(ds).exists():
            datasets.append((Path(ds).stem, ds))
        else:
            logger.warning(f"Dataset not found, skipping: {ds}")
    if not datasets:
        logger.error("No valid datasets. Aborting.")
        return

    logger.info("=" * 70)
    logger.info("CLASSICAL BASELINES")
    logger.info(f"  session       : {session_id}")
    logger.info(f"  datasets      : {[d[0] for d in datasets]}")
    logger.info(f"  num_qubits/PCA: {args.num_qubits}  samples: {args.sample_size}")
    logger.info(f"  models        : {args.models} (GradientBoost -> {gradient_boost_label()})")
    logger.info(f"  regimes       : {args.regimes}")
    logger.info(f"  runs          : {args.n_runs}  seeds: {seeds}")
    logger.info("=" * 70)

    # Pre-load and prepare each dataset once.
    data_cache = {}
    for dataset_name, dataset_path in datasets:
        X, y = load_dataset(dataset_path, args.num_qubits, args.sample_size)
        data_cache[dataset_name] = (X, y)
        logger.info(f"  loaded {dataset_name}: X={X.shape}, "
                    f"class balance={np.bincount(y).tolist()}")

    # Build the full task list.
    tasks = []
    for dataset_name, _ in datasets:
        for regime in args.regimes:
            for model_name in args.models:
                for run_idx, seed in enumerate(seeds):
                    tasks.append((dataset_name, regime, model_name, run_idx, seed))

    logger.info(f"Executing {len(tasks)} classical runs (CPU)...")

    all_results = []
    completed = 0
    t_start = time.perf_counter()
    runs_csv = RESULTS_DIR / f'classical_baseline_runs_{session_id}.csv'

    for dataset_name, regime, model_name, run_idx, seed in tasks:
        X, y = data_cache[dataset_name]
        result = execute_baseline_run(
            dataset_name, X, y, args.num_qubits, regime, model_name, seed, run_idx
        )
        all_results.append(result)
        completed += 1
        if completed % 10 == 0 or completed == len(tasks):
            pd.DataFrame(all_results).to_csv(runs_csv, index=False)
            elapsed = time.perf_counter() - t_start
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (len(tasks) - completed) / rate if rate > 0 else float('inf')
            logger.info(f"  progress {completed}/{len(tasks)} "
                        f"({elapsed/60:.1f} min elapsed, ETA {eta/60:.1f} min)")

    runs_df = pd.DataFrame(all_results)
    ok_df = runs_df[runs_df['status'] == 'success'].copy() if 'status' in runs_df else runs_df
    logger.info(f"Completed: {len(ok_df)}/{len(tasks)} successful runs")

    if ok_df.empty:
        logger.error("No successful runs; skipping aggregation.")
        return

    summary_df = summarize(ok_df)
    summary_csv = RESULTS_DIR / f'classical_baseline_summary_{session_id}.csv'
    summary_df.to_csv(summary_csv, index=False)

    logger.info("=" * 70)
    logger.info("CLASSICAL BASELINES COMPLETE")
    logger.info(f"  per-run : {runs_csv}")
    logger.info(f"  summary : {summary_csv}")
    logger.info("=" * 70)

    # Console digest: quantum-feature regime accuracy/MCC by model.
    try:
        digest = summary_df[summary_df['feature_regime'] == 'quantum_features']
        for ds, g in digest.groupby('dataset'):
            logger.info(f"  [quantum_features] {ds}:")
            for _, row in g.sort_values('accuracy_mean', ascending=False).iterrows():
                logger.info(f"      {row['model']:<14} "
                            f"acc={row['accuracy_mean']*100:.2f}±{row['accuracy_ci95']*100:.2f}  "
                            f"MCC={row['mcc_mean']:.3f}  AUC={row['roc_auc_mean']:.3f}")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
