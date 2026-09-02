#!/usr/bin/env python3

"""Compare preprocessing orders for leakage risk on IoTID20.

This script reproduces the same preprocessing stages used in iot_multigpu.py:
1. SelectKBest with mutual information to keep 2n features.
2. MinMaxScaler to [0, pi].
3. PCA to n components.

It evaluates two protocols on the same sampled dataset:
- leaky_full_fit: fit preprocessing on the full dataset, then split.
- clean_split_first: split first, then fit preprocessing on train only.

The goal is not to rerun the full quantum experiment, but to quantify whether
the preprocessing order materially changes downstream metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Match the paper's macro one-vs-rest specificity for multiclass tasks."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    specificities = []
    total = cm.sum()
    for class_index in range(len(unique_classes)):
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = total - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specificities))


def sanitize_features(data: np.ndarray) -> np.ndarray:
    """Keep downstream preprocessing stable in the presence of non-finite values."""
    return np.asarray(np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


def pad_or_reduce(X_train: np.ndarray, X_test: np.ndarray, num_qubits: int, seed: int):
    """Apply PCA reduction when needed, or zero-pad to reach num_qubits features."""
    pca = None
    variance_retained = None

    if X_train.shape[1] > num_qubits:
        n_components = min(num_qubits, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        X_train = sanitize_features(pca.fit_transform(X_train))
        X_test = sanitize_features(pca.transform(X_test))
        variance_retained = float(np.sum(pca.explained_variance_ratio_))

    if X_train.shape[1] < num_qubits:
        padding = num_qubits - X_train.shape[1]
        X_train = np.pad(X_train, ((0, 0), (0, padding)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, padding)), mode="constant")

    return X_train, X_test, variance_retained


@dataclass
class ProtocolArtifacts:
    selected_features: int
    selected_feature_names: list[str]
    pca_variance_retained: float | None
    pca_component_summaries: list[dict[str, object]]
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]


def summarize_pca_components(pca: PCA | None, feature_names: list[str], top_k: int = 5) -> list[dict[str, object]]:
    """Summarize the strongest original-feature contributors for each PCA component."""
    if pca is None:
        return []

    component_summaries: list[dict[str, object]] = []
    for component_index, component in enumerate(pca.components_, start=1):
        ranked_indices = np.argsort(np.abs(component))[::-1][:top_k]
        top_features = []
        for index in ranked_indices:
            top_features.append(
                {
                    "feature": feature_names[int(index)],
                    "loading": float(component[int(index)]),
                    "abs_loading": float(abs(component[int(index)])),
                }
            )
        explained_ratio = None
        if hasattr(pca, "explained_variance_ratio_"):
            explained_ratio = float(pca.explained_variance_ratio_[component_index - 1])
        component_summaries.append(
            {
                "component": component_index,
                "explained_variance_ratio": explained_ratio,
                "top_features": top_features,
            }
        )

    return component_summaries


def load_dataset(dataset_path: str, sample_size: int | None, seed: int):
    df = pd.read_csv(dataset_path)

    if "Label" in df.columns:
        X = df.drop(columns=["Label"]).copy()
        y = df["Label"].copy()
    elif "label" in df.columns:
        X = df.drop(columns=["label"]).copy()
        y = df["label"].copy()
    else:
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()

    feature_names = X.columns.astype(str).tolist()

    for col in X.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        X[col] = pd.Series(encoder.fit_transform(X[col].astype(str)), index=X.index)

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.to_numpy()

    X = sanitize_features(X.to_numpy(dtype=np.float32))
    y = np.asarray(y)

    if sample_size is not None and sample_size < len(X):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y, feature_names


def preprocess_full_fit_then_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    num_qubits: int,
    test_size: float,
    seed: int,
):
    selected_features = X.shape[1]
    selected_feature_names = list(feature_names)

    if X.shape[1] > num_qubits * 2:
        selector = SelectKBest(mutual_info_classif, k=min(num_qubits * 2, X.shape[1]))
        X = sanitize_features(selector.fit_transform(X, y))
        selected_features = X.shape[1]
        selected_feature_names = [name for name, keep in zip(feature_names, selector.get_support()) if keep]

    scaler = MinMaxScaler(feature_range=cast(Any, (0.0, np.pi)))
    X = sanitize_features(scaler.fit_transform(X))

    if X.shape[1] > num_qubits:
        n_components = min(num_qubits, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        X = sanitize_features(pca.fit_transform(X))
        variance_retained = float(np.sum(pca.explained_variance_ratio_))
    else:
        pca = None
        variance_retained = None

    if X.shape[1] < num_qubits:
        padding = num_qubits - X.shape[1]
        X = np.pad(X, ((0, 0), (0, padding)), mode="constant")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    artifacts = ProtocolArtifacts(
        selected_features=selected_features,
        selected_feature_names=selected_feature_names,
        pca_variance_retained=variance_retained,
        pca_component_summaries=summarize_pca_components(pca, selected_feature_names),
        train_shape=X_train.shape,
        test_shape=X_test.shape,
    )
    return X_train, X_test, y_train, y_test, artifacts


def preprocess_split_first_train_only(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    num_qubits: int,
    test_size: float,
    seed: int,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    selected_features = X_train.shape[1]
    selected_feature_names = list(feature_names)
    if X_train.shape[1] > num_qubits * 2:
        selector = SelectKBest(mutual_info_classif, k=min(num_qubits * 2, X_train.shape[1]))
        X_train = sanitize_features(selector.fit_transform(X_train, y_train))
        X_test = sanitize_features(np.asarray(selector.transform(X_test)))
        selected_features = X_train.shape[1]
        selected_feature_names = [name for name, keep in zip(feature_names, selector.get_support()) if keep]

    scaler = MinMaxScaler(feature_range=cast(Any, (0.0, np.pi)))
    X_train = sanitize_features(scaler.fit_transform(X_train))
    X_test = sanitize_features(np.asarray(scaler.transform(X_test)))

    pca = None
    variance_retained = None
    if X_train.shape[1] > num_qubits:
        n_components = min(num_qubits, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        X_train = sanitize_features(pca.fit_transform(X_train))
        X_test = sanitize_features(pca.transform(X_test))
        variance_retained = float(np.sum(pca.explained_variance_ratio_))

    if X_train.shape[1] < num_qubits:
        padding = num_qubits - X_train.shape[1]
        X_train = np.pad(X_train, ((0, 0), (0, padding)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, padding)), mode="constant")

    artifacts = ProtocolArtifacts(
        selected_features=selected_features,
        selected_feature_names=selected_feature_names,
        pca_variance_retained=variance_retained,
        pca_component_summaries=summarize_pca_components(pca, selected_feature_names),
        train_shape=X_train.shape,
        test_shape=X_test.shape,
    )
    return X_train, X_test, y_train, y_test, artifacts


def build_models(seed: int):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
    }


def evaluate_models(X_train, X_test, y_train, y_test, seed: int):
    results = {}
    for name, model in build_models(seed).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "specificity": float(compute_specificity(y_test, y_pred)),
        }
    return results


def compare_results(leaky_results: dict, clean_results: dict):
    comparison = {}
    for model_name in leaky_results:
        comparison[model_name] = {}
        for metric_name, leaky_value in leaky_results[model_name].items():
            clean_value = clean_results[model_name][metric_name]
            comparison[model_name][metric_name] = {
                "full_fit_then_split": leaky_value,
                "split_first_fit_train_only": clean_value,
                "delta_full_minus_clean": float(leaky_value - clean_value),
            }
    return comparison


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "not applied"
    return f"{value:.6f}"


def build_markdown_report(payload: dict) -> str:
    lines = []
    lines.append("# Preprocessing Leakage Check Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report compares two preprocessing orders on the same IoTID20 sample to assess whether fitting preprocessing before the train/test split can change downstream evaluation metrics."
    )
    lines.append("")
    lines.append("## Protocols Compared")
    lines.append("")
    lines.append("1. `full_fit_then_split`: reproduces the current experiment order, where `SelectKBest`, `MinMaxScaler`, and `PCA` are fit on the full sampled dataset before the train/test split.")
    lines.append("2. `split_first_fit_train_only`: applies the recommended order, where the train/test split happens first and all preprocessing is fit only on the training partition before transforming the test partition.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Dataset: `{payload['dataset']}`")
    lines.append(f"- Sample size used: `{payload['sample_size']}`")
    lines.append(f"- Target qubits / PCA output dimension: `{payload['num_qubits']}`")
    lines.append(f"- Test split fraction: `{payload['test_size']}`")
    lines.append(f"- Random seed: `{payload['seed']}`")
    lines.append("")
    lines.append("## Preprocessing Summary")
    lines.append("")
    lines.append("| Protocol | Selected Features | PCA Variance Retained | Train Shape | Test Shape |")
    lines.append("|---|---:|---:|---|---|")
    for protocol_name, protocol_data in payload["protocols"].items():
        lines.append(
            f"| `{protocol_name}` | {protocol_data['selected_features']} | {_format_optional_float(protocol_data['pca_variance_retained'])} | {tuple(protocol_data['train_shape'])} | {tuple(protocol_data['test_shape'])} |"
        )
    lines.append("")
    lines.append("## Selected Feature Names")
    lines.append("")
    for protocol_name, protocol_data in payload["protocols"].items():
        lines.append(f"### {protocol_name}")
        lines.append("")
        for index, feature_name in enumerate(protocol_data["selected_feature_names"], start=1):
            lines.append(f"{index}. `{feature_name}`")
        lines.append("")
    lines.append("## PCA Component Summaries")
    lines.append("")
    lines.append("These lists show the strongest original-feature contributors to each retained principal component after `SelectKBest`. PCA does not directly select original columns, so this is the interpretable comparison for the PCA stage.")
    lines.append("")
    for protocol_name, protocol_data in payload["protocols"].items():
        lines.append(f"### {protocol_name}")
        lines.append("")
        if not protocol_data["pca_component_summaries"]:
            lines.append("PCA was not applied for this protocol.")
            lines.append("")
            continue
        for component_summary in protocol_data["pca_component_summaries"]:
            lines.append(
                f"Component {component_summary['component']} (explained variance ratio: {_format_optional_float(component_summary['explained_variance_ratio'])})"
            )
            for feature_info in component_summary["top_features"]:
                lines.append(
                    f"- `{feature_info['feature']}` with loading {feature_info['loading']:+.6f}"
                )
            lines.append("")
    lines.append("## Metric Comparison")
    lines.append("")
    lines.append("Positive delta means the `full_fit_then_split` order scored higher than the clean split-first protocol.")
    lines.append("")
    for model_name, metrics in payload["results"].items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |")
        lines.append("|---|---:|---:|---:|")
        for metric_name, metric_values in metrics.items():
            lines.append(
                f"| `{metric_name}` | {metric_values['full_fit_then_split']:.6f} | {metric_values['split_first_fit_train_only']:.6f} | {metric_values['delta_full_minus_clean']:+.6f} |"
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "If the two protocols produce different test metrics, then the preprocessing order is affecting evaluation. That is evidence that fitting preprocessing before the split changes the measured outcome and should be treated as a leakage risk, especially because `SelectKBest(mutual_info_classif)` is supervised."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check preprocessing leakage from split order.")
    parser.add_argument(
        "--dataset",
        default="data/IoT_Original_Distribution.csv",
        help="Path to the dataset CSV file.",
    )
    parser.add_argument("--num-qubits", type=int, default=10, help="Target feature dimension after PCA.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Optional sample size for a quick check.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full comparison as JSON.",
    )
    parser.add_argument(
        "--output-report",
        default="results/leakage_checks/preprocessing_leakage_report.md",
        help="Path to save a human-readable Markdown report.",
    )
    args = parser.parse_args()

    X, y, feature_names = load_dataset(args.dataset, args.sample_size, args.seed)

    leaky = preprocess_full_fit_then_split(X, y, feature_names, args.num_qubits, args.test_size, args.seed)
    clean = preprocess_split_first_train_only(X, y, feature_names, args.num_qubits, args.test_size, args.seed)

    X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky, leaky_artifacts = leaky
    X_train_clean, X_test_clean, y_train_clean, y_test_clean, clean_artifacts = clean

    leaky_results = evaluate_models(X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky, args.seed)
    clean_results = evaluate_models(X_train_clean, X_test_clean, y_train_clean, y_test_clean, args.seed)
    comparison = compare_results(leaky_results, clean_results)

    payload = {
        "dataset": args.dataset,
        "sample_size": int(len(X)),
        "num_qubits": args.num_qubits,
        "test_size": args.test_size,
        "seed": args.seed,
        "protocols": {
            "full_fit_then_split": {
                "selected_features": leaky_artifacts.selected_features,
                "selected_feature_names": leaky_artifacts.selected_feature_names,
                "pca_variance_retained": leaky_artifacts.pca_variance_retained,
                "pca_component_summaries": leaky_artifacts.pca_component_summaries,
                "train_shape": list(leaky_artifacts.train_shape),
                "test_shape": list(leaky_artifacts.test_shape),
            },
            "split_first_fit_train_only": {
                "selected_features": clean_artifacts.selected_features,
                "selected_feature_names": clean_artifacts.selected_feature_names,
                "pca_variance_retained": clean_artifacts.pca_variance_retained,
                "pca_component_summaries": clean_artifacts.pca_component_summaries,
                "train_shape": list(clean_artifacts.train_shape),
                "test_shape": list(clean_artifacts.test_shape),
            },
        },
        "results": comparison,
    }

    report_text = build_markdown_report(payload)

    print("Dataset:", args.dataset)
    print("Sample size:", len(X))
    print("Target qubits:", args.num_qubits)
    print()
    print("Protocol summary:")
    print("  full_fit_then_split:", payload["protocols"]["full_fit_then_split"])
    print("  split_first_fit_train_only:", payload["protocols"]["split_first_fit_train_only"])
    print()
    print("Metric comparison:")
    for model_name, model_metrics in comparison.items():
        print(f"  {model_name}:")
        for metric_name, metric_values in model_metrics.items():
            print(
                "    "
                f"{metric_name}: "
                f"full_fit_then_split={metric_values['full_fit_then_split']:.6f}, "
                f"split_first_fit_train_only={metric_values['split_first_fit_train_only']:.6f}, "
                f"delta={metric_values['delta_full_minus_clean']:+.6f}"
            )

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print()
    print(f"Saved Markdown report to {report_path}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print()
        print(f"Saved JSON report to {json_path}")


if __name__ == "__main__":
    main()