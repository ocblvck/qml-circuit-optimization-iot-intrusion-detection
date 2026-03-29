# Preprocessing Leakage Check Report

## Purpose

This report compares two preprocessing orders on the same IoTID20 sample to assess whether fitting preprocessing before the train/test split can change downstream evaluation metrics.

## Protocols Compared

1. `full_fit_then_split`: reproduces the current experiment order, where `SelectKBest`, `MinMaxScaler`, and `PCA` are fit on the full sampled dataset before the train/test split.
2. `split_first_fit_train_only`: applies the recommended order, where the train/test split happens first and all preprocessing is fit only on the training partition before transforming the test partition.

## Run Configuration

- Dataset: `IoT_Original_Distribution.csv`
- Sample size used: `5000`
- Target qubits / PCA output dimension: `10`
- Test split fraction: `0.3`
- Random seed: `42`

## Preprocessing Summary

| Protocol | Selected Features | PCA Variance Retained | Train Shape | Test Shape |
|---|---:|---:|---|---|
| `full_fit_then_split` | 20 | 0.987618 | (3500, 10) | (1500, 10) |
| `split_first_fit_train_only` | 20 | 0.985590 | (3500, 10) | (1500, 10) |

## Metric Comparison

Positive delta means the `full_fit_then_split` order scored higher than the clean split-first protocol.

### logistic_regression

| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |
|---|---:|---:|---:|
| `accuracy` | 0.968000 | 0.970000 | -0.002000 |
| `f1_weighted` | 0.967240 | 0.969233 | -0.001993 |
| `mcc` | 0.803386 | 0.815365 | -0.011980 |
| `specificity` | 0.987500 | 0.988971 | -0.001471 |

### random_forest

| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |
|---|---:|---:|---:|
| `accuracy` | 0.991333 | 0.991333 | +0.000000 |
| `f1_weighted` | 0.991143 | 0.991143 | +0.000000 |
| `mcc` | 0.947921 | 0.947921 | +0.000000 |
| `specificity` | 1.000000 | 1.000000 | +0.000000 |

## Interpretation

If the two protocols produce different test metrics, then the preprocessing order is affecting evaluation. That is evidence that fitting preprocessing before the split changes the measured outcome and should be treated as a leakage risk, especially because `SelectKBest(mutual_info_classif)` is supervised.
