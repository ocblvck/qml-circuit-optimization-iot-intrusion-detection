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
| `full_fit_then_split` | 20 | 0.989886 | (3500, 10) | (1500, 10) |
| `split_first_fit_train_only` | 20 | 0.989417 | (3500, 10) | (1500, 10) |

## Selected Feature Names

### full_fit_then_split

1. `Src_Port`
2. `Dst_Port`
3. `Flow_Duration`
4. `TotLen_Bwd_Pkts`
5. `Bwd_Pkt_Len_Mean`
6. `Flow_Pkts/s`
7. `Flow_IAT_Mean`
8. `Flow_IAT_Max`
9. `Bwd_Header_Len`
10. `Bwd_Pkts/s`
11. `Pkt_Len_Max`
12. `Pkt_Len_Mean`
13. `Pkt_Size_Avg`
14. `Bwd_Seg_Size_Avg`
15. `Subflow_Bwd_Byts`
16. `Init_Bwd_Win_Byts`
17. `Idle_Mean`
18. `Idle_Max`
19. `Fwd_Bwd_Len_Ratio`
20. `Pkt_Rate`

### split_first_fit_train_only

1. `Src_Port`
2. `Dst_Port`
3. `Flow_Duration`
4. `TotLen_Bwd_Pkts`
5. `Bwd_Pkt_Len_Mean`
6. `Flow_Pkts/s`
7. `Flow_IAT_Mean`
8. `Flow_IAT_Max`
9. `Bwd_IAT_Tot`
10. `Bwd_Header_Len`
11. `Bwd_Pkts/s`
12. `Pkt_Len_Mean`
13. `Pkt_Size_Avg`
14. `Bwd_Seg_Size_Avg`
15. `Subflow_Bwd_Byts`
16. `Init_Bwd_Win_Byts`
17. `Idle_Mean`
18. `Idle_Max`
19. `Fwd_Bwd_Len_Ratio`
20. `Pkt_Rate`

## PCA Component Summaries

These lists show the strongest original-feature contributors to each retained principal component after `SelectKBest`. PCA does not directly select original columns, so this is the interpretable comparison for the PCA stage.

### full_fit_then_split

Component 1 (explained variance ratio: 0.457473)
- `Pkt_Len_Max` with loading +0.403246
- `Bwd_Seg_Size_Avg` with loading +0.384990
- `Bwd_Pkt_Len_Mean` with loading +0.384990
- `Pkt_Len_Mean` with loading +0.370877
- `Pkt_Size_Avg` with loading +0.347869

Component 2 (explained variance ratio: 0.349042)
- `Flow_Pkts/s` with loading +0.350949
- `Flow_IAT_Mean` with loading -0.340121
- `Idle_Mean` with loading -0.338273
- `Flow_IAT_Max` with loading -0.330921
- `Src_Port` with loading +0.328787

Component 3 (explained variance ratio: 0.061150)
- `Dst_Port` with loading +0.795028
- `Src_Port` with loading -0.471591
- `TotLen_Bwd_Pkts` with loading -0.198755
- `Subflow_Bwd_Byts` with loading -0.198755
- `Bwd_Header_Len` with loading -0.135212

Component 4 (explained variance ratio: 0.044125)
- `Init_Bwd_Win_Byts` with loading +0.807224
- `Bwd_Header_Len` with loading +0.455245
- `Flow_Duration` with loading -0.154460
- `Flow_Pkts/s` with loading -0.151514
- `Flow_IAT_Max` with loading -0.140516

Component 5 (explained variance ratio: 0.028789)
- `Bwd_Pkts/s` with loading +0.652449
- `Bwd_Header_Len` with loading +0.418132
- `Src_Port` with loading -0.336933
- `Flow_Pkts/s` with loading +0.325468
- `Init_Bwd_Win_Byts` with loading -0.205383

Component 6 (explained variance ratio: 0.018367)
- `Bwd_Header_Len` with loading +0.484011
- `Init_Bwd_Win_Byts` with loading -0.413178
- `Src_Port` with loading +0.373650
- `Dst_Port` with loading +0.336970
- `Flow_Pkts/s` with loading -0.313612

Component 7 (explained variance ratio: 0.014513)
- `Src_Port` with loading +0.602301
- `Dst_Port` with loading +0.472905
- `Idle_Mean` with loading +0.287243
- `Init_Bwd_Win_Byts` with loading +0.286818
- `Idle_Max` with loading +0.273565

Component 8 (explained variance ratio: 0.006339)
- `Flow_Duration` with loading +0.431694
- `Pkt_Len_Max` with loading -0.410934
- `Flow_Pkts/s` with loading +0.407125
- `Idle_Mean` with loading -0.358714
- `Bwd_Pkts/s` with loading -0.298976

Component 9 (explained variance ratio: 0.005169)
- `Pkt_Len_Max` with loading +0.626926
- `Flow_Pkts/s` with loading +0.449200
- `Bwd_Pkts/s` with loading -0.279722
- `Bwd_Seg_Size_Avg` with loading -0.272966
- `Bwd_Pkt_Len_Mean` with loading -0.272966

Component 10 (explained variance ratio: 0.004920)
- `Idle_Max` with loading +0.453780
- `Idle_Mean` with loading +0.429227
- `Flow_IAT_Mean` with loading -0.356691
- `Flow_Pkts/s` with loading +0.307295
- `Bwd_Pkts/s` with loading -0.299004

### split_first_fit_train_only

Component 1 (explained variance ratio: 0.443778)
- `Bwd_IAT_Tot` with loading +0.340354
- `Flow_IAT_Max` with loading +0.322547
- `Flow_IAT_Mean` with loading +0.318100
- `Flow_Duration` with loading +0.314794
- `Idle_Max` with loading +0.309184

Component 2 (explained variance ratio: 0.346704)
- `Bwd_Pkt_Len_Mean` with loading +0.389864
- `Bwd_Seg_Size_Avg` with loading +0.389864
- `Pkt_Len_Mean` with loading +0.374279
- `Pkt_Size_Avg` with loading +0.350760
- `Flow_Pkts/s` with loading -0.259829

Component 3 (explained variance ratio: 0.066386)
- `Dst_Port` with loading +0.759571
- `Src_Port` with loading -0.475426
- `Bwd_IAT_Tot` with loading -0.208716
- `Subflow_Bwd_Byts` with loading -0.207309
- `TotLen_Bwd_Pkts` with loading -0.207309

Component 4 (explained variance ratio: 0.046022)
- `Init_Bwd_Win_Byts` with loading +0.795893
- `Bwd_Header_Len` with loading +0.467972
- `Flow_Duration` with loading -0.155488
- `Flow_IAT_Max` with loading -0.145829
- `Flow_Pkts/s` with loading -0.137986

Component 5 (explained variance ratio: 0.036486)
- `Bwd_Pkts/s` with loading +0.574795
- `Bwd_IAT_Tot` with loading +0.418364
- `Bwd_Header_Len` with loading +0.341133
- `Flow_Pkts/s` with loading +0.334429
- `Src_Port` with loading -0.280821

Component 6 (explained variance ratio: 0.019011)
- `Bwd_Header_Len` with loading +0.419053
- `Init_Bwd_Win_Byts` with loading -0.407936
- `Src_Port` with loading +0.395072
- `Bwd_Pkts/s` with loading -0.350073
- `Flow_Pkts/s` with loading -0.338057

Component 7 (explained variance ratio: 0.014463)
- `Src_Port` with loading +0.608449
- `Dst_Port` with loading +0.496475
- `Idle_Mean` with loading +0.291054
- `Idle_Max` with loading +0.280361
- `Init_Bwd_Win_Byts` with loading +0.255586

Component 8 (explained variance ratio: 0.008061)
- `Idle_Mean` with loading +0.495102
- `Bwd_IAT_Tot` with loading -0.449372
- `Flow_Duration` with loading -0.374021
- `Bwd_Pkts/s` with loading +0.294077
- `Idle_Max` with loading +0.288339

Component 9 (explained variance ratio: 0.005424)
- `Flow_Pkts/s` with loading +0.610867
- `Bwd_Pkts/s` with loading -0.450301
- `Idle_Max` with loading +0.394676
- `Flow_IAT_Mean` with loading -0.335966
- `Bwd_Header_Len` with loading +0.171416

Component 10 (explained variance ratio: 0.003081)
- `TotLen_Bwd_Pkts` with loading +0.440665
- `Subflow_Bwd_Byts` with loading +0.440664
- `Pkt_Size_Avg` with loading -0.402608
- `Pkt_Len_Mean` with loading -0.386898
- `Bwd_Header_Len` with loading -0.280525

## Metric Comparison

Positive delta means the `full_fit_then_split` order scored higher than the clean split-first protocol.

### logistic_regression

| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |
|---|---:|---:|---:|
| `accuracy` | 0.971333 | 0.967333 | +0.004000 |
| `f1_weighted` | 0.969710 | 0.965751 | +0.003960 |
| `mcc` | 0.819509 | 0.794109 | +0.025400 |
| `specificity` | 0.995588 | 0.991912 | +0.003676 |

### random_forest

| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |
|---|---:|---:|---:|
| `accuracy` | 0.976000 | 0.977333 | -0.001333 |
| `f1_weighted` | 0.975073 | 0.976629 | -0.001556 |
| `mcc` | 0.851148 | 0.860314 | -0.009167 |
| `specificity` | 0.994853 | 0.994118 | +0.000735 |

## Interpretation

If the two protocols produce different test metrics, then the preprocessing order is affecting evaluation. That is evidence that fitting preprocessing before the split changes the measured outcome and should be treated as a leakage risk, especially because `SelectKBest(mutual_info_classif)` is supervised.
