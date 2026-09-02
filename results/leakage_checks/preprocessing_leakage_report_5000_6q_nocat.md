# Preprocessing Leakage Check Report

## Purpose

This report compares two preprocessing orders on the same IoTID20 sample to assess whether fitting preprocessing before the train/test split can change downstream evaluation metrics.

## Protocols Compared

1. `full_fit_then_split`: reproduces the current experiment order, where `SelectKBest`, `MinMaxScaler`, and `PCA` are fit on the full sampled dataset before the train/test split.
2. `split_first_fit_train_only`: applies the recommended order, where the train/test split happens first and all preprocessing is fit only on the training partition before transforming the test partition.

## Run Configuration

- Dataset: `IoT_Original_Distribution.csv`
- Sample size used: `5000`
- Target qubits / PCA output dimension: `6`
- Test split fraction: `0.3`
- Random seed: `42`

## Preprocessing Summary

| Protocol | Selected Features | PCA Variance Retained | Train Shape | Test Shape |
|---|---:|---:|---|---|
| `full_fit_then_split` | 12 | 0.977928 | (3500, 6) | (1500, 6) |
| `split_first_fit_train_only` | 12 | 0.976606 | (3500, 6) | (1500, 6) |

## Selected Feature Names

### full_fit_then_split

1. `Src_Port`
2. `Dst_Port`
3. `Flow_Duration`
4. `TotLen_Bwd_Pkts`
5. `Flow_Pkts/s`
6. `Flow_IAT_Mean`
7. `Flow_IAT_Max`
8. `Pkt_Size_Avg`
9. `Subflow_Bwd_Byts`
10. `Init_Bwd_Win_Byts`
11. `Idle_Max`
12. `Pkt_Rate`

### split_first_fit_train_only

1. `Src_Port`
2. `Dst_Port`
3. `Flow_Duration`
4. `TotLen_Bwd_Pkts`
5. `Flow_Pkts/s`
6. `Flow_IAT_Mean`
7. `Flow_IAT_Max`
8. `Bwd_Pkts/s`
9. `Subflow_Bwd_Byts`
10. `Init_Bwd_Win_Byts`
11. `Idle_Max`
12. `Pkt_Rate`

## PCA Component Summaries

These lists show the strongest original-feature contributors to each retained principal component after `SelectKBest`. PCA does not directly select original columns, so this is the interpretable comparison for the PCA stage.

### full_fit_then_split

Component 1 (explained variance ratio: 0.512778)
- `Flow_IAT_Mean` with loading +0.428068
- `Flow_IAT_Max` with loading +0.427611
- `Flow_Duration` with loading +0.416551
- `Idle_Max` with loading +0.413410
- `Src_Port` with loading -0.360553

Component 2 (explained variance ratio: 0.244013)
- `Pkt_Size_Avg` with loading +0.641377
- `Subflow_Bwd_Byts` with loading +0.493689
- `TotLen_Bwd_Pkts` with loading +0.493689
- `Flow_Pkts/s` with loading -0.247969
- `Src_Port` with loading -0.168954

Component 3 (explained variance ratio: 0.105585)
- `Dst_Port` with loading +0.823099
- `Src_Port` with loading -0.487203
- `Subflow_Bwd_Byts` with loading -0.146184
- `TotLen_Bwd_Pkts` with loading -0.146184
- `Idle_Max` with loading -0.126558

Component 4 (explained variance ratio: 0.067588)
- `Init_Bwd_Win_Byts` with loading +0.947247
- `Flow_Pkts/s` with loading -0.186660
- `Flow_Duration` with loading -0.159313
- `Flow_IAT_Max` with loading -0.132826
- `Idle_Max` with loading -0.094713

Component 5 (explained variance ratio: 0.028226)
- `Src_Port` with loading +0.748326
- `Dst_Port` with loading +0.488328
- `Flow_Pkts/s` with loading -0.404536
- `Idle_Max` with loading +0.114401
- `Flow_IAT_Max` with loading +0.105240

Component 6 (explained variance ratio: 0.019738)
- `Flow_Pkts/s` with loading +0.749569
- `Init_Bwd_Win_Byts` with loading +0.265459
- `Dst_Port` with loading +0.262947
- `Idle_Max` with loading +0.251923
- `TotLen_Bwd_Pkts` with loading +0.220594

### split_first_fit_train_only

Component 1 (explained variance ratio: 0.566266)
- `Flow_IAT_Mean` with loading +0.409061
- `Flow_IAT_Max` with loading +0.407305
- `Flow_Duration` with loading +0.395800
- `Idle_Max` with loading +0.393677
- `Src_Port` with loading -0.354774

Component 2 (explained variance ratio: 0.154350)
- `Subflow_Bwd_Byts` with loading +0.649118
- `TotLen_Bwd_Pkts` with loading +0.649118
- `Flow_Pkts/s` with loading -0.280052
- `Bwd_Pkts/s` with loading -0.145354
- `Src_Port` with loading -0.138083

Component 3 (explained variance ratio: 0.113171)
- `Dst_Port` with loading +0.822678
- `Src_Port` with loading -0.481226
- `Idle_Max` with loading -0.141344
- `Subflow_Bwd_Byts` with loading -0.128919
- `TotLen_Bwd_Pkts` with loading -0.128919

Component 4 (explained variance ratio: 0.072676)
- `Init_Bwd_Win_Byts` with loading +0.923927
- `Flow_Pkts/s` with loading -0.199638
- `Flow_Duration` with loading -0.174926
- `Bwd_Pkts/s` with loading -0.151133
- `Flow_IAT_Max` with loading -0.150156

Component 5 (explained variance ratio: 0.044731)
- `Bwd_Pkts/s` with loading +0.703597
- `Src_Port` with loading -0.455501
- `Flow_Pkts/s` with loading +0.345603
- `Init_Bwd_Win_Byts` with loading +0.265786
- `Subflow_Bwd_Byts` with loading +0.155748

Component 6 (explained variance ratio: 0.025411)
- `Src_Port` with loading +0.636811
- `Dst_Port` with loading +0.548633
- `Idle_Max` with loading +0.246372
- `Bwd_Pkts/s` with loading +0.223381
- `Flow_IAT_Max` with loading +0.220989

## Metric Comparison

Positive delta means the `full_fit_then_split` order scored higher than the clean split-first protocol.

### logistic_regression

| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |
|---|---:|---:|---:|
| `accuracy` | 0.946667 | 0.965333 | -0.018667 |
| `f1_weighted` | 0.940256 | 0.963442 | -0.023186 |
| `mcc` | 0.638611 | 0.780083 | -0.141472 |
| `specificity` | 0.991912 | 0.991912 | +0.000000 |

### random_forest

| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |
|---|---:|---:|---:|
| `accuracy` | 0.979333 | 0.981333 | -0.002000 |
| `f1_weighted` | 0.978730 | 0.980684 | -0.001954 |
| `mcc` | 0.873047 | 0.885297 | -0.012250 |
| `specificity` | 0.994853 | 0.997059 | -0.002206 |

## Interpretation

If the two protocols produce different test metrics, then the preprocessing order is affecting evaluation. That is evidence that fitting preprocessing before the split changes the measured outcome and should be treated as a leakage risk, especially because `SelectKBest(mutual_info_classif)` is supervised.
