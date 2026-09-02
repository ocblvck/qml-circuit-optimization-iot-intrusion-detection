# Preprocessing Leakage Check Report

## Purpose

This report compares two preprocessing orders on the same IoTID20 sample to assess whether fitting preprocessing before the train/test split can change downstream evaluation metrics.

## Protocols Compared

1. `full_fit_then_split`: reproduces the current experiment order, where `SelectKBest`, `MinMaxScaler`, and `PCA` are fit on the full sampled dataset before the train/test split.
2. `split_first_fit_train_only`: applies the recommended order, where the train/test split happens first and all preprocessing is fit only on the training partition before transforming the test partition.

## Run Configuration

- Dataset: `IoT_Original_Distribution.csv`
- Sample size used: `10000`
- Target qubits / PCA output dimension: `10`
- Test split fraction: `0.3`
- Random seed: `42`

## Preprocessing Summary

| Protocol | Selected Features | PCA Variance Retained | Train Shape | Test Shape |
|---|---:|---:|---|---|
| `full_fit_then_split` | 20 | 0.986553 | (7000, 10) | (3000, 10) |
| `split_first_fit_train_only` | 20 | 0.985131 | (7000, 10) | (3000, 10) |

## Selected Feature Names

### full_fit_then_split

1. `Src_Port`
2. `Dst_Port`
3. `Flow_Duration`
4. `TotLen_Bwd_Pkts`
5. `Flow_Pkts/s`
6. `Flow_IAT_Mean`
7. `Flow_IAT_Max`
8. `Bwd_IAT_Tot`
9. `Bwd_Header_Len`
10. `Bwd_Pkts/s`
11. `Pkt_Len_Mean`
12. `Pkt_Size_Avg`
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
5. `Bwd_Pkt_Len_Min`
6. `Flow_Pkts/s`
7. `Flow_IAT_Mean`
8. `Flow_IAT_Max`
9. `Bwd_Header_Len`
10. `Bwd_Pkts/s`
11. `Pkt_Len_Mean`
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

Component 1 (explained variance ratio: 0.501262)
- `Bwd_IAT_Tot` with loading +0.358849
- `Flow_IAT_Mean` with loading +0.342398
- `Flow_IAT_Max` with loading +0.342045
- `Flow_Duration` with loading +0.334500
- `Idle_Max` with loading +0.328590

Component 2 (explained variance ratio: 0.237893)
- `Pkt_Len_Mean` with loading +0.550684
- `Pkt_Size_Avg` with loading +0.517739
- `TotLen_Bwd_Pkts` with loading +0.383654
- `Subflow_Bwd_Byts` with loading +0.383654
- `Flow_Pkts/s` with loading -0.217828

Component 3 (explained variance ratio: 0.079151)
- `Dst_Port` with loading +0.705577
- `Src_Port` with loading -0.427071
- `Cat` with loading +0.277578
- `Sub_Cat` with loading +0.271295
- `Bwd_Pkts/s` with loading -0.191355

Component 4 (explained variance ratio: 0.054874)
- `Init_Bwd_Win_Byts` with loading +0.754611
- `Bwd_Header_Len` with loading +0.408274
- `Cat` with loading +0.251782
- `Dst_Port` with loading -0.231747
- `Sub_Cat` with loading +0.211894

Component 5 (explained variance ratio: 0.039689)
- `Bwd_Pkts/s` with loading +0.591138
- `Bwd_IAT_Tot` with loading +0.393182
- `Bwd_Header_Len` with loading +0.390877
- `Flow_Pkts/s` with loading +0.328628
- `Src_Port` with loading -0.326307

Component 6 (explained variance ratio: 0.027538)
- `Sub_Cat` with loading +0.504623
- `Src_Port` with loading +0.455537
- `Init_Bwd_Win_Byts` with loading -0.393602
- `Cat` with loading +0.347059
- `Idle_Max` with loading +0.237875

Component 7 (explained variance ratio: 0.017795)
- `Bwd_Header_Len` with loading +0.453165
- `Bwd_Pkts/s` with loading -0.372277
- `Flow_Pkts/s` with loading -0.363170
- `Idle_Mean` with loading -0.347736
- `Init_Bwd_Win_Byts` with loading -0.336195

Component 8 (explained variance ratio: 0.014151)
- `Src_Port` with loading +0.598430
- `Dst_Port` with loading +0.572898
- `Sub_Cat` with loading -0.308891
- `Init_Bwd_Win_Byts` with loading +0.307516
- `Cat` with loading -0.283714

Component 9 (explained variance ratio: 0.008198)
- `Idle_Mean` with loading +0.454183
- `Flow_Duration` with loading -0.389756
- `Bwd_IAT_Tot` with loading -0.388348
- `Flow_Pkts/s` with loading -0.367422
- `Bwd_Pkts/s` with loading +0.338970

Component 10 (explained variance ratio: 0.006003)
- `Flow_Pkts/s` with loading +0.521389
- `Bwd_Pkts/s` with loading -0.423176
- `Idle_Max` with loading +0.403996
- `Flow_IAT_Mean` with loading -0.371760
- `Bwd_Header_Len` with loading +0.238448

### split_first_fit_train_only

Component 1 (explained variance ratio: 0.443090)
- `Flow_IAT_Max` with loading +0.355779
- `Flow_IAT_Mean` with loading +0.354265
- `Flow_Duration` with loading +0.347388
- `Idle_Max` with loading +0.342588
- `Idle_Mean` with loading +0.338572

Component 2 (explained variance ratio: 0.304012)
- `Bwd_Pkt_Len_Min` with loading +0.466840
- `Pkt_Len_Mean` with loading +0.462849
- `Pkt_Size_Avg` with loading +0.435183
- `Subflow_Bwd_Byts` with loading +0.316749
- `TotLen_Bwd_Pkts` with loading +0.316749

Component 3 (explained variance ratio: 0.076846)
- `Dst_Port` with loading +0.727334
- `Src_Port` with loading -0.422185
- `Cat` with loading +0.294341
- `Sub_Cat` with loading +0.293774
- `Subflow_Bwd_Byts` with loading -0.164165

Component 4 (explained variance ratio: 0.055447)
- `Init_Bwd_Win_Byts` with loading +0.718292
- `Bwd_Header_Len` with loading +0.429107
- `Cat` with loading +0.259620
- `Dst_Port` with loading -0.236328
- `Sub_Cat` with loading +0.212937

Component 5 (explained variance ratio: 0.033817)
- `Bwd_Pkts/s` with loading +0.599181
- `Src_Port` with loading -0.451307
- `Bwd_Header_Len` with loading +0.421207
- `Flow_Pkts/s` with loading +0.284633
- `Cat` with loading -0.219660

Component 6 (explained variance ratio: 0.026384)
- `Init_Bwd_Win_Byts` with loading +0.465505
- `Sub_Cat` with loading -0.410121
- `Src_Port` with loading -0.330145
- `Bwd_Header_Len` with loading -0.298366
- `Cat` with loading -0.265988

Component 7 (explained variance ratio: 0.019030)
- `Bwd_Header_Len` with loading +0.408696
- `Flow_Pkts/s` with loading -0.387259
- `Bwd_Pkts/s` with loading -0.386561
- `Init_Bwd_Win_Byts` with loading -0.361568
- `Idle_Mean` with loading -0.313631

Component 8 (explained variance ratio: 0.013776)
- `Src_Port` with loading +0.595110
- `Dst_Port` with loading +0.583055
- `Sub_Cat` with loading -0.339990
- `Cat` with loading -0.304246
- `Init_Bwd_Win_Byts` with loading +0.273745

Component 9 (explained variance ratio: 0.007138)
- `Flow_Pkts/s` with loading +0.541729
- `Flow_Duration` with loading +0.473919
- `Bwd_Pkts/s` with loading -0.393954
- `Idle_Mean` with loading -0.358034
- `Flow_IAT_Max` with loading +0.206411

Component 10 (explained variance ratio: 0.005592)
- `Idle_Max` with loading +0.424377
- `Idle_Mean` with loading +0.390071
- `Flow_IAT_Mean` with loading -0.376651
- `Flow_Pkts/s` with loading +0.356133
- `Bwd_Pkts/s` with loading -0.339180

## Metric Comparison

Positive delta means the `full_fit_then_split` order scored higher than the clean split-first protocol.

### logistic_regression

| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |
|---|---:|---:|---:|
| `accuracy` | 0.972333 | 0.970000 | +0.002333 |
| `f1_weighted` | 0.972028 | 0.969428 | +0.002599 |
| `mcc` | 0.828824 | 0.812054 | +0.016769 |
| `specificity` | 0.987170 | 0.987537 | -0.000367 |

### random_forest

| Metric | Full Fit Then Split | Split First Fit Train Only | Delta (Full - Clean) |
|---|---:|---:|---:|
| `accuracy` | 0.995333 | 0.996000 | -0.000667 |
| `f1_weighted` | 0.995294 | 0.995959 | -0.000665 |
| `mcc` | 0.971428 | 0.975549 | -0.004121 |
| `specificity` | 0.999267 | 1.000000 | -0.000733 |

## Interpretation

If the two protocols produce different test metrics, then the preprocessing order is affecting evaluation. That is evidence that fitting preprocessing before the split changes the measured outcome and should be treated as a leakage risk, especially because `SelectKBest(mutual_info_classif)` is supervised.
