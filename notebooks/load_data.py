import pandas as pd
import os

ROOT = "/content/drive/MyDrive/viet-dialect-project"

def load_vidia2std():
    train = pd.read_csv("hf://datasets/Biu3010/ViDia2Std/train.csv").dropna()
    dev   = pd.read_csv("hf://datasets/Biu3010/ViDia2Std/dev.csv").dropna()
    test  = pd.read_csv("hf://datasets/Biu3010/ViDia2Std/test.csv").dropna()

    train = train[train["region"] == "central"].reset_index(drop=True)
    dev   = dev[dev["region"] == "central"].reset_index(drop=True)
    test  = test[test["region"] == "central"].reset_index(drop=True)

    print(f"ViDia2Std Central — train: {len(train):,} | dev: {len(dev):,} | test: {len(test):,}")
    return train, dev, test

def load_vimd_central():
    cache = f"{ROOT}/data/raw/vimd_central_distribution.csv"
    assert os.path.exists(cache), "Cache not found — chạy stream script trước"

    CENTRAL_CODES = {
        "36","37","38",   # Thanh-Nghe-Tinh
        "73","74","75",   # Binh-Tri-Thien
        "43","92","76",   # Nam-Trung-Bo
        "77","78","79"
    }
    GROUP_MAP = {
        "36":"Thanh-Nghe-Tinh", "37":"Thanh-Nghe-Tinh", "38":"Thanh-Nghe-Tinh",
        "73":"Binh-Tri-Thien",  "74":"Binh-Tri-Thien",  "75":"Binh-Tri-Thien",
        "43":"Nam-Trung-Bo",    "92":"Nam-Trung-Bo",     "76":"Nam-Trung-Bo",
        "77":"Nam-Trung-Bo",    "78":"Nam-Trung-Bo",     "79":"Nam-Trung-Bo"
    }

    df = pd.read_csv(cache)
    total = df["count"].sum()
    print(f"ViMD Central — {len(df)} tỉnh | {total:,} samples")
    print(df.groupby("group")["count"].sum().to_string())
    return df, CENTRAL_CODES, GROUP_MAP

if __name__ == "__main__":
    print("=== Load ViDia2Std ===")
    train, dev, test = load_vidia2std()

    print("\n=== Load ViMD Central ===")
    df_vimd, codes, groups = load_vimd_central()

    print("\n✓ Data pipeline OK — sẵn sàng cho Phase 1")
