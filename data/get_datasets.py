#!/usr/bin/env python3
"""Acquire and verify the datasets this study uses.

Two of the three datasets are NOT redistributed in this repository, because their
licences grant academic use but do not explicitly grant redistribution (see
README.md in this directory). Download them from their official sources, then run
this script to assemble and verify them.

    python data/get_datasets.py            # check what is present and verify it
    python data/get_datasets.py --build    # also rebuild UNSW_NB15.csv from its parts

The script never downloads anything itself: both UNSW datasets are distributed by
UNSW Canberra under terms you should read and accept yourself. It tells you where
to get each file, reconstructs the one combined file the experiments expect, and
verifies every file byte-for-byte against the checksums of the exact copies used
for the published results.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# SHA-256 of the exact files used to produce every result in the paper.
CHECKSUMS = {
    "IoT_Original_Distribution.csv":
        "85c1de873d38d669adb50ab8155aff2c389da155f3f5ee61dd4cdcdc20f41bd5",
    "UNSW_2018_IoT_Botnet_Final_10_Best.csv":
        "cf412171c67832fd783811541dc7dc09ed4625a8f3b8a544f644e572bc50bd7b",
    "UNSW_NB15.csv":
        "98679f73d06851d76da6b982e95f06a1f41bcb3223684e7b9fd5cf992b0eb5e8",
}

SHIPPED = {"IoT_Original_Distribution.csv"}

SOURCES = {
    "UNSW_2018_IoT_Botnet_Final_10_Best.csv": (
        "Bot-IoT, the pre-selected 'Final 10 Best' feature release.\n"
        "      Source:  https://research.unsw.edu.au/projects/bot-iot-dataset\n"
        "      Licence: free use for academic research in perpetuity; commercial\n"
        "               use by agreement with the authors.\n"
        "      Expected: 3,668,522 data rows, semicolon-delimited, leading index\n"
        "               column, header begins ';pkSeqID;proto;saddr;...'\n"
        "      Place it at data/UNSW_2018_IoT_Botnet_Final_10_Best.csv"
    ),
    "UNSW_NB15.csv": (
        "UNSW-NB15, the training and testing partitions combined.\n"
        "      Source:  https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
        "      Licence: free use for academic research in perpetuity; commercial\n"
        "               use by agreement with the authors.\n"
        "      Download 'UNSW_NB15_training-set.csv' (175,341 rows) and\n"
        "      'UNSW_NB15_testing-set.csv' (82,332 rows) into data/, then run\n"
        "      this script with --build to produce the combined 257,673-row file."
    ),
}

NB15_PARTS = ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_nb15() -> bool:
    """Reconstruct UNSW_NB15.csv exactly as used in the study.

    The combined file is the training partition followed by the testing partition,
    concatenated with pandas and written without an index column. This recipe was
    verified to reproduce the published file byte-for-byte.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  ! pandas is required for --build (pip install -r requirements.txt)")
        return False

    parts = [os.path.join(HERE, p) for p in NB15_PARTS]
    missing = [p for p in parts if not os.path.isfile(p)]
    if missing:
        print("  ! cannot build UNSW_NB15.csv; missing:")
        for m in missing:
            print(f"      {os.path.relpath(m)}")
        return False

    print("  building UNSW_NB15.csv from the training and testing partitions ...")
    train = pd.read_csv(parts[0], low_memory=False)
    test = pd.read_csv(parts[1], low_memory=False)
    combined = pd.concat([train, test], ignore_index=True)
    out = os.path.join(HERE, "UNSW_NB15.csv")
    combined.to_csv(out, index=False)
    print(f"  wrote {os.path.relpath(out)} ({len(combined):,} rows)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="rebuild UNSW_NB15.csv from its training/testing partitions")
    args = ap.parse_args()

    if args.build:
        build_nb15()
        print()

    ok = True
    print("Dataset status")
    print("=" * 72)
    for name, expected in CHECKSUMS.items():
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            ok = False
            tag = "shipped with this repository" if name in SHIPPED else "not redistributed"
            print(f"\n  [MISSING] {name}   ({tag})")
            if name in SHIPPED:
                print("      This file should be present. Run 'git lfs pull'.")
            else:
                print(f"      {SOURCES[name]}")
            continue

        digest = sha256(path)
        size_mb = os.path.getsize(path) / (1 << 20)
        if digest == expected:
            print(f"  [OK]      {name}  ({size_mb:,.0f} MB, checksum verified)")
        else:
            ok = False
            print(f"  [MISMATCH] {name}  ({size_mb:,.0f} MB)")
            print(f"      expected {expected}")
            print(f"      found    {digest}")
            print("      This is not the copy used for the published results. Results")
            print("      computed from it may differ from those reported in the paper.")

    print("=" * 72)
    if ok:
        print("All three datasets present and verified. You can run the experiments.")
    else:
        print("Some datasets are missing or do not match. See the notes above and")
        print("data/README.md for licence terms and required citations.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
