import pandas as pd


def profile(df: pd.DataFrame) -> dict:
    """Summarize a dataset: size, missing values, duplicates, column stats."""
    columns = []
    for col in df.columns:
        s = df[col]
        info = {
            "column": col,
            "type": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_%": round(s.isna().mean() * 100, 2),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            info["min"] = s.min()
            info["max"] = s.max()
            info["mean"] = round(s.mean(), 2)
        columns.append(info)

    return {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "column_profile": pd.DataFrame(columns),
    }