# Data: provenance, licences, and required attribution

This directory holds the three intrusion-detection datasets used in the reported
experiments, stored via Git LFS so the repository is a self-contained artifact:

- `IoT_Original_Distribution.csv` — derived from IoTID20
- `UNSW_2018_IoT_Botnet_Final_10_Best.csv` — Bot-IoT, "Final 10 Best" release
- `UNSW_NB15.csv` — UNSW-NB15

**These datasets are the property of their original authors, not of this project.**
The repository `LICENSE` (MIT) covers the *code and the derived result files only*.
It confers no rights over the data below, whose authors have asserted copyright.

Each dataset is redistributed here solely to make the reported experiments
reproducible for academic research, which is the use each licence grants. Anyone
using these files must comply with the original terms and must cite the original
papers, reproduced below. If you intend any non-academic or commercial use, obtain
permission from the dataset authors first; two of the three licences require it
explicitly.

---

## 1. IoTID20 — `IoT_Original_Distribution.csv`

- **Original source:** <https://sites.google.com/view/iot-network-intrusion-dataset/home>
- **Licence, verbatim from the authors:** "Free use of the IoT Network Intrusion
  Dataset for academic research purposes is hereby granted in perpetuity."
- **Required citation:**

  > I. Ullah and Q. H. Mahmoud, "A Scheme for Generating a Dataset for Anomalous
  > Activity Detection in IoT Networks," in *Advances in Artificial Intelligence*
  > (Canadian AI 2020), Lecture Notes in Computer Science, vol. 12109, Springer,
  > Cham, 2020, pp. 508–520.

- **Note:** this file is *derived*, not the raw release. It is IoTID20 after the
  wrangling pipeline in Section III-B of the paper: removal of `Flow_ID`, `Src_IP`,
  `Dst_IP`, `Timestamp`; median/mode imputation; infinite-value handling; exact
  duplicate removal (625,783 → 293,696 rows); IQR outlier capping; three engineered
  ratio features; label encoding. Result: 293,696 rows × 85 columns, 90.6% anomaly.

## 2. Bot-IoT — `UNSW_2018_IoT_Botnet_Final_10_Best.csv`

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

- **Prevalence warning.** This release contains only 477 normal flows among
  3,668,522 records (0.013%). Proportional stratified sampling at N=5000 would
  retain fewer than one normal flow, so the experiment sampler enforces a
  minority-class floor raising the normal class to 9.5–10%. Every UNSW-2018 metric
  in the paper is computed at that altered prevalence and must **not** be read as
  deployment performance at the native prevalence. See Section V-C of the paper.

## 3. UNSW-NB15 — `UNSW_NB15.csv`

- **Original source:** <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- **Licence, verbatim from the authors:** "Free use of the UNSW-NB15 dataset for
  academic research purposes is hereby granted in perpetuity. Use for commercial
  purposes should be agreed by the authors."
- **Required citation:**

  > N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network
  > intrusion detection systems (UNSW-NB15 network data set)," in *Proc. Military
  > Communications and Information Systems Conference (MilCIS)*, 2015, pp. 1–6.

---

## Integrity

To confirm you have the same inputs as the reported runs:

```bash
sha256sum *.csv > data_checksums.sha256
```

## Removal requests

If any dataset author objects to redistribution here, open an issue or contact the
repository owner and the file will be removed and replaced with a download script.
