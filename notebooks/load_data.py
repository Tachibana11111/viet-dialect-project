import pandas as pd
import os

ROOT = "/content/drive/MyDrive/viet-dialect-project"

LABEL2ID = {"Thanh-Nghe-Tinh": 0, "Binh-Tri-Thien": 1, "Nam-Trung-Bo": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASS_WEIGHTS = [1.3156, 1.3174, 0.6753]
MAX_LENGTH = 64

def load_vidia2std():
    train = pd.read_csv(f"{ROOT}/data/splits/vidia_train.csv")
    dev   = pd.read_csv(f"{ROOT}/data/splits/vidia_dev.csv")
    test  = pd.read_csv(f"{ROOT}/data/splits/vidia_test.csv")
    print(f"ViDia2Std — train:{len(train):,} | dev:{len(dev):,} | test:{len(test):,}")
    return train, dev, test

def load_classifier_data():
    train = pd.read_csv(f"{ROOT}/data/splits/clf_train.csv")
    val   = pd.read_csv(f"{ROOT}/data/splits/clf_val.csv")
    test  = pd.read_csv(f"{ROOT}/data/splits/clf_test.csv")
    print(f"Classifier — train:{len(train):,} | val:{len(val):,} | test:{len(test):,}")
    return train, val, test

if __name__ == "__main__":
    print("=== Load ViDia2Std (normalization) ===")
    vidia_train, vidia_dev, vidia_test = load_vidia2std()

    print("\n=== Load Classifier data ===")
    clf_train, clf_val, clf_test = load_classifier_data()

    print("\n=== Config ===")
    print(f"  LABEL2ID     : {LABEL2ID}")
    print(f"  CLASS_WEIGHTS: {CLASS_WEIGHTS}")
    print(f"  MAX_LENGTH   : {MAX_LENGTH}")

    print("\n✓ Phase 1 data pipeline OK — sẵn sàng cho Phase 2")
