import numpy as np
import pandas as pd

CITIES = ["St. John's", "Mount Pearl", "Corner Brook", "Gander", "Grand Falls-Windsor"]
PRODUCTS = {"Milk (2L)": 6.5, "White bread": 4.0, "Eggs (dozen)": 5.2,
            "Gasoline (per L)": 1.75, "Furnace oil (per L)": 1.55}
ERROR_TYPES = ["missing", "wrong_type", "outlier", "negative",
               "inconsistent", "invalid_date", "duplicate"]


def make_clean_data(months=36, seed=42):
    """Clean monthly prices for 5 NL cities and 5 products."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=months, freq="MS")
    rows = []
    for city in CITIES:
        city_factor = rng.uniform(0.97, 1.05)
        for product, base in PRODUCTS.items():
            for i, d in enumerate(dates):
                price = base * city_factor * (1 + 0.004 * i) * rng.normal(1, 0.02)
                rows.append({"date": d.strftime("%Y-%m-%d"), "city": city,
                             "product": product, "price": round(price, 2),
                             "units_sold": int(rng.integers(80, 400))})
    return pd.DataFrame(rows)


def inject_errors(df, rate=0.02, seed=7):
    """Add known errors. Returns (dirty_data, ground_truth)."""
    rng = np.random.default_rng(seed)
    dirty = df.copy()
    dirty["price"] = dirty["price"].astype(object)
    dirty["units_sold"] = dirty["units_sold"].astype(float)

    n = max(1, int(len(df) * rate))
    pool = rng.permutation(dirty.index.to_numpy())[: n * len(ERROR_TYPES)]
    picks = dict(zip(ERROR_TYPES, np.array_split(pool, len(ERROR_TYPES))))
    truth = []

    for idx in picks["missing"]:
        col = rng.choice(["price", "units_sold", "city"])
        dirty.at[idx, col] = np.nan
        truth.append((idx, col, "missing"))

    for idx in picks["wrong_type"]:
        dirty.at[idx, "price"] = f"${dirty.at[idx, 'price']}"
        truth.append((idx, "price", "wrong_type"))

    for idx in picks["outlier"]:
        dirty.at[idx, "price"] = round(float(dirty.at[idx, "price"]) * 20, 2)
        truth.append((idx, "price", "outlier"))

    for idx in picks["negative"]:
        dirty.at[idx, "units_sold"] = -abs(dirty.at[idx, "units_sold"])
        truth.append((idx, "units_sold", "negative"))

    for idx in picks["inconsistent"]:
        city = dirty.at[idx, "city"]
        dirty.at[idx, "city"] = rng.choice([city.lower(), city.upper(), city + " "])
        truth.append((idx, "city", "inconsistent"))

    for idx in picks["invalid_date"]:
        dirty.at[idx, "date"] = rng.choice(["2024-02-31", "2023-13-01", "not a date"])
        truth.append((idx, "date", "invalid_date"))

    dups = dirty.loc[picks["duplicate"]].copy()
    start = dirty.index.max() + 1
    dups.index = range(start, start + len(dups))
    dirty = pd.concat([dirty, dups])
    truth += [(i, None, "duplicate") for i in dups.index]

    return dirty, pd.DataFrame(truth, columns=["row", "column", "type"])