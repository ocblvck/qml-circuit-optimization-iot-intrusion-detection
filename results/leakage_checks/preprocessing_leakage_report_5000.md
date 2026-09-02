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

## Selected Feature Names

### full_fit_then_split

1. `Src_Port`
2. `Dst_Port`
3. `Flow_Duration`
4. `TotLen_Bwd_Pkts`
5. `Flow_Pkts/s`
6. `Flow_IAT_Mean`
7. `Flow_IAT_Max`
8. `Bwd_Header_Len`
9. `Bwd_Pkts/s`
10. `Pkt_Len_Mean`
11. `Pkt_Size_Avg`
12. `Bwd_Seg_Size_Avg`
13. `Subflow_Bwd_Byts`
14. `Init_Bwd_Win_Byts`
15. `Idle_Mean`
16. `Idle_Max`
17. `Fwd_Bwd_Len_Ratio`
18. `Pkt_Rate`
19. `Cat`
20. `Sub_Cat`

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
12. `Pkt_Size_Avg`
13. `Subflow_Bwd_Byts`
14. `Init_Bwd_Win_Byts`
15. `Idle_Mean`
16. `Idle_Max`
17. `Fwd_Bwd_Len_Ratio`
18. `Pkt_Rate`
19. `Cat`
20. `Sub_Cat`

## PCA Component Summaries

These lists show the strongest original-feature contributors to each retained principal component after `SelectKBest`. PCA does not directly select original columns, so this is the interpretable comparison for the PCA stage.

### full_fit_then_split

Component 1 (explained variance ratio: 0.447629)
- `Flow_IAT_Max` with loading +0.357003
- `Flow_IAT_Mean` with loading +0.355576
- `Flow_Duration` with loading +0.348554
- `Idle_Max` with loading +0.345338
- `Idle_Mean` with loading +0.341464

Component 2 (explained variance ratio: 0.308801)
- `Bwd_Seg_Size_Avg` with loading +0.479826
- `Pkt_Len_Mean` with loading +0.462638
- `Pkt_Size_Avg` with loading +0.433988
- `TotLen_Bwd_Pkts` with loading +0.321370
- `Subflow_Bwd_Byts` with loading +0.321370

Component 3 (explained variance ratio: 0.076397)
- `Dst_Port` with loading +0.717201
- `Src_Port` with loading -0.419949
- `Cat` with loading +0.303110
- `Sub_Cat` with loading +0.300621
- `Init_Bwd_Win_Byts` with loading +0.164955

Component 4 (explained variance ratio: 0.053514)
- `Init_Bwd_Win_Byts` with loading +0.739564
- `Bwd_Header_Len` with loading +0.416503
- `Dst_Port` with loading -0.259410
- `Cat` with loading +0.235357
- `Sub_Cat` with loading +0.190223

Component 5 (explained variance ratio: 0.033642)
- `Bwd_Pkts/s` with loading +0.601103
- `Src_Port` with loading -0.457890
- `Bwd_Header_Len` with loading +0.416462
- `Flow_Pkts/s` with loading +0.294666
- `Cat` with loading -0.215429

Component 6 (explained variance ratio: 0.024839)
- `Sub_Cat` with loading +0.449560
- `Init_Bwd_Win_Byts` with loading -0.396077
- `Src_Port` with loading +0.318099
- `Bwd_Header_Len` with loading +0.290914
- `Cat` with loading +0.289676

Component 7 (explained variance ratio: 0.018219)
- `Bwd_Header_Len` with loading +0.478051
- `Init_Bwd_Win_Byts` with loading -0.374469
- `Flow_Pkts/s` with loading -0.361257
- `Bwd_Pkts/s` with loading -0.345229
- `Idle_Mean` with loading -0.291129

Component 8 (explained variance ratio: 0.012988)
- `Src_Port` with loading +0.598232
- `Dst_Port` with loading +0.587662
- `Sub_Cat` with loading -0.318525
- `Init_Bwd_Win_Byts` with loading +0.287509
- `Cat` with loading -0.284872

Component 9 (explained variance ratio: 0.006643)
- `Flow_Pkts/s` with loading +0.581791
- `Flow_Duration` with loading +0.456243
- `Bwd_Pkts/s` with loading -0.423574
- `Idle_Mean` with loading -0.334390
- `Flow_IAT_Max` with loading +0.196692

Component 10 (explained variance ratio: 0.004945)
- `Idle_Max` with loading +0.445114
- `Idle_Mean` with loading +0.433590
- `Flow_IAT_Mean` with loading -0.374719
- `Flow_IAT_Max` with loading -0.326436
- `Flow_Pkts/s` with loading +0.314511

### split_first_fit_train_only

Component 1 (explained variance ratio: 0.497627)
- `Bwd_IAT_Tot` with loading +0.357597
- `Flow_IAT_Mean` with loading +0.341753
- `Flow_IAT_Max` with loading +0.341353
- `Flow_Duration` with loading +0.333637
- `Idle_Max` with loading +0.329031

Component 2 (explained variance ratio: 0.241196)
- `Bwd_Pkt_Len_Mean` with loading +0.564857
- `Pkt_Size_Avg` with loading +0.510079
- `TotLen_Bwd_Pkts` with loading +0.381638
- `Subflow_Bwd_Byts` with loading +0.381638
- `Flow_Pkts/s` with loading -0.211515

Component 3 (explained variance ratio: 0.080644)
- `Dst_Port` with loading +0.697138
- `Src_Port` with loading -0.428443
- `Cat` with loading +0.281986
- `Sub_Cat` with loading +0.277161
- `Bwd_Pkts/s` with loading -0.191343

Component 4 (explained variance ratio: 0.055057)
- `Init_Bwd_Win_Byts` with loading +0.743298
- `Bwd_Header_Len` with loading +0.425556
- `Cat` with loading +0.244940
- `Dst_Port` with loading -0.211577
- `Sub_Cat` with loading +0.211319

Component 5 (explained variance ratio: 0.040177)
- `Bwd_Pkts/s` with loading +0.573639
- `Bwd_IAT_Tot` with loading +0.397463
- `Bwd_Header_Len` with loading +0.393402
- `Src_Port` with loading -0.331322
- `Flow_Pkts/s` with loading +0.319003

Component 6 (explained variance ratio: 0.025993)
- `Sub_Cat` with loading +0.507644
- `Src_Port` with loading +0.428576
- `Init_Bwd_Win_Byts` with loading -0.426086
- `Cat` with loading +0.354627
- `Idle_Max` with loading +0.224056

Component 7 (explained variance ratio: 0.018269)
- `Bwd_Header_Len` with loading +0.417483
- `Bwd_Pkts/s` with loading -0.400119
- `Flow_Pkts/s` with loading -0.360540
- `Idle_Mean` with loading -0.333372
- `Init_Bwd_Win_Byts` with loading -0.324059

Component 8 (explained variance ratio: 0.012995)
- `Src_Port` with loading +0.602927
- `Dst_Port` with loading +0.586884
- `Init_Bwd_Win_Byts` with loading +0.300796
- `Sub_Cat` with loading -0.287947
- `Cat` with loading -0.264431

Component 9 (explained variance ratio: 0.008039)
- `Idle_Mean` with loading +0.432502
- `Bwd_IAT_Tot` with loading -0.404862
- `Flow_Pkts/s` with loading -0.391863
- `Flow_Duration` with loading -0.383659
- `Bwd_Pkts/s` with loading +0.373315

Component 10 (explained variance ratio: 0.005592)
- `Flow_Pkts/s` with loading +0.530628
- `Idle_Max` with loading +0.420551
- `Bwd_Pkts/s` with loading -0.393938
- `Flow_IAT_Mean` with loading -0.335727
- `Bwd_Header_Len` with loading +0.251647

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
