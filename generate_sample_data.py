"""sample_data.csv (temiz) və sample_data_dirty.csv (bilərəkdən çirkli) yaradır.
İstifadə:  python generate_sample_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

PRODUCTS = {  # product: (category, qiymət)
    "Laptop": ("Computers", 1500), "Monitor": ("Computers", 300), "Keyboard": ("Computers", 60),
    "Phone": ("Mobile", 800), "Tablet": ("Mobile", 500), "Headphones": ("Audio", 100),
}
REGIONS = ["Baku", "Ganja", "Sumgait", "Lankaran"]


def make_clean(n=500) -> pd.DataFrame:
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    prod = rng.choice(list(PRODUCTS), n, p=[.15, .15, .2, .2, .15, .15])
    qty = rng.integers(1, 6, n)
    price = np.array([PRODUCTS[p][1] for p in prod])
    revenue = np.round(qty * price * rng.uniform(0.9, 1.1, n), 2)
    df = pd.DataFrame({
        "Date": dates, "Product": prod, "Category": [PRODUCTS[p][0] for p in prod],
        "Quantity": qty, "Revenue": revenue, "Region": rng.choice(REGIONS, n),
    }).sort_values("Date").reset_index(drop=True)
    return df


def make_dirty(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    n = len(df)
    # tarix formatları qarışıq
    fmts = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%b %d, %Y"]
    df["Date"] = [d.strftime(fmts[i % len(fmts)]) for i, d in enumerate(df["Date"])]
    # mətn problemləri
    idx = rng.choice(n, 40, replace=False)
    for k, i in enumerate(idx):
        p = df.at[i, "Product"]
        df.at[i, "Product"] = [f" {p} ", p.lower(), p.upper()][k % 3]
    for i in rng.choice(n, 20, replace=False):
        df.at[i, "Region"] = f"  {df.at[i, 'Region']}"
    # boş xanalar
    for col, cnt in [("Revenue", 15), ("Region", 12), ("Quantity", 8)]:
        df.loc[rng.choice(n, cnt, replace=False), col] = np.nan
    # şübhəli rəqəmlər
    df.loc[3, "Revenue"] = -500
    df.loc[77, "Revenue"] = 9_999_999
    df.loc[150, "Quantity"] = 5000
    # dublikatlar
    df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


if __name__ == "__main__":
    clean = make_clean()
    clean.to_csv("sample_data.csv", index=False)
    make_dirty(clean).to_csv("sample_data_dirty.csv", index=False)
    print("Yaradıldı: sample_data.csv, sample_data_dirty.csv")
