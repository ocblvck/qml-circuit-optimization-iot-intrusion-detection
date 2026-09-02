# Data: provenance, licences, and how to obtain each dataset

This study uses three intrusion-detection datasets. **Only one is redistributed here.**

| File | Shipped? | Why |
|---|---|---|
| `IoT_Original_Distribution.csv` | **Yes** (Git LFS) | A *derived* artifact of this project that cannot be reconstructed without our pipeline |
| `UNSW_2018_IoT_Botnet_Final_10_Best.csv` | No | Redistribution not granted by its licence; download from UNSW |
| `UNSW_NB15.csv` | No | Redistribution not granted by its licence; download from UNSW |

Run this to see what you have, get download instructions, and verify every file
against the exact copies used for the published results:

```bash
python data/get_datasets.py            # status + verification
python data/get_datasets.py --build    # also rebuild UNSW_NB15.csv from its parts
```

The repository `LICENSE` (MIT) covers **the code and the derived result files only**. It
confers no rights over third-party data. Each dataset's authors have asserted their own
copyright, and each dataset must be cited as shown below.

---

## 1. IoTID20 — `IoT_Original_Distribution.csv` (shipped)

- **Original source:** <https://sites.google.com/view/iot-network-intrusion-dataset/home>
- **Licence, verbatim from the authors:** "Free use of the IoT Network Intrusion
  Dataset for academic research purposes is hereby granted in perpetuity."
- **Required citation:**

  > I. Ullah and Q. H. Mahmoud, "A Scheme for Generating a Dataset for Anomalous
  > Activity Detection in IoT Networks," in *Advances in Artificial Intelligence*
  > (Canadian AI 2020), Lecture Notes in Computer Science, vol. 12109, Springer,
  > Cham, 2020, pp. 508–520.

**Why this one is included.** It is not the IoTID20 release. It is the result of the
wrangling pipeline described in Section III-B of the paper: removal of `Flow_ID`,
`Src_IP`, `Dst_IP` and `Timestamp`; median/mode imputation; infinite-value handling;
exact duplicate removal (625,783 → 293,696 rows); IQR outlier capping; three engineered
ratio features; and label encoding. The result is 293,696 rows × 85 columns, 90.6%
anomaly. Nobody can reconstruct it from the raw download without our exact pipeline, so
omitting it would break reproducibility of every IoT result in the paper. IoTID20's
licence is also the least restrictive of the three: academic use in perpetuity, no
commercial clause, and no assertion of copyright.

If the IoTID20 authors would prefer this derived file not be hosted here, open an issue
and it will be removed and replaced with a regeneration script.

## 2. Bot-IoT — `UNSW_2018_IoT_Botnet_Final_10_Best.csv` (not shipped)

- **Original source:** <https://research.unsw.edu.au/projects/bot-iot-dataset>
- **Licence, verbatim from the authors:** "Free use of the Bot-IoT dataset for
  academic research purposes is hereby granted in perpetuity. Use for commercial
  purposes should be agreed by the authors. The authors have asserted their rights
  under the Copyright."
- **Required citation:**

  > N. Koroniotis, N. Moustafa, E. Sitnikova, and B. Turnbull, "Towards the
  > development of realistic botnet dataset in the Internet of Things for network
  > forensic analytics: Bot-IoT dataset," *Future Generation Computer Systems*,
  > vol. 100, pp. 779–796, 2019.

Download the pre-selected **"Final 10 Best"** feature release and place it at
`data/UNSW_2018_IoT_Botnet_Final_10_Best.csv`. Expected form: 3,668,522 data rows,
semicolon-delimited, with a leading index column; the header begins
`;pkSeqID;proto;saddr;`. Verify with `python data/get_datasets.py`.

**Prevalence warning.** This release contains only 477 normal flows among 3,668,522
records (0.013%). Proportional stratified sampling at N=5000 would retain fewer than one
normal flow, so the experiment sampler enforces a minority-class floor raising the normal
class to 9.5–10%. Every UNSW-2018 metric in the paper is computed at that altered
prevalence and must **not** be read as deployment performance at the native prevalence.
See Section V-C of the paper.

## 3. UNSW-NB15 — `UNSW_NB15.csv` (not shipped)

- **Original source:** <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- **Licence, verbatim from the authors:** "Free use of the UNSW-NB15 dataset for
  academic research purposes is hereby granted in perpetuity. Use for commercial
  purposes should be agreed by the authors."
- **Required citation:**

  > N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network
  > intrusion detection systems (UNSW-NB15 network data set)," in *Proc. Military
  > Communications and Information Systems Conference (MilCIS)*, 2015, pp. 1–6.

The experiments use the training and testing partitions **combined**. Download
`UNSW_NB15_training-set.csv` (175,341 rows) and `UNSW_NB15_testing-set.csv` (82,332
rows) into `data/`, then run:

```bash
python data/get_datasets.py --build
```

This concatenates the training partition followed by the testing partition and writes the
result without an index column, giving 257,673 rows × 45 columns. That recipe was verified
to reproduce the file used for the published results **byte-for-byte**, so the script's
checksum check will pass.

---

## Integrity

`data/get_datasets.py` verifies every file against the SHA-256 of the exact copy used for
the published results:

| File | SHA-256 |
|---|---|
| `IoT_Original_Distribution.csv` | `85c1de873d38d669adb50ab8155aff2c389da155f3f5ee61dd4cdcdc20f41bd5` |
| `UNSW_2018_IoT_Botnet_Final_10_Best.csv` | `cf412171c67832fd783811541dc7dc09ed4625a8f3b8a544f644e572bc50bd7b` |
| `UNSW_NB15.csv` | `98679f73d06851d76da6b982e95f06a1f41bcb3223684e7b9fd5cf992b0eb5e8` |

A mismatch means you have a different copy from the one behind the reported numbers, and
results computed from it may legitimately differ.

## Note on repository history

The two UNSW datasets were present in earlier commits of this repository before being
removed. Git history therefore still references them. If either dataset's authors would
prefer that history be purged as well, open an issue and the history will be rewritten.
