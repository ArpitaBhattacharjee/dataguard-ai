import pandas as pd


def _rows(issues, issue_type, column):
    mask = issues["type"] == issue_type
    if column and column != "(whole row)":
        mask &= issues["column"] == column
    return issues.loc[mask, "row"].tolist()


def apply_fixes(df: pd.DataFrame, issues: pd.DataFrame, fixes: pd.DataFrame):
    """Apply approved fixes. Returns (clean_df, audit_log)."""
    out = df.copy().astype(object)
    audit = []
    rows_to_drop = set()

    for _, f in fixes.iterrows():
        col = f["column"] if f["column"] != "(whole row)" else None
        rows = [r for r in _rows(issues, f["issue_type"], f["column"]) if r in out.index]
        action = f["action"]

        for r in rows:
            before = out.at[r, col] if col else "(row)"
            after = before

            if action in ("drop_duplicate_rows", "drop_rows"):
                rows_to_drop.add(r)
                after = "(row removed)"
            elif action == "convert_to_number" and col:
                cleaned = str(before).replace("$", "").replace(",", "").strip()
                after = pd.to_numeric(cleaned, errors="coerce")
            elif action == "standardize_category" and col:
                key = str(before).strip().lower()
                values = out[col].dropna().astype(str)
                same = values[values.str.strip().str.lower() == key]
                after = same.value_counts().index[0] if not same.empty else before
            elif action == "set_to_missing" and col:
                after = None
            elif action == "make_positive" and col:
                after = abs(pd.to_numeric(before, errors="coerce"))
            elif action in ("fill_with_median", "fill_with_most_common") and col:
                continue  # done once per column below
            elif action == "flag_only":
                continue

            if col and after is not before:
                out.at[r, col] = after
            audit.append({"row": r, "column": col or "(whole row)",
                          "action": action, "before": str(before), "after": str(after)})

        if action == "fill_with_median" and col:
            numbers = pd.to_numeric(out[col], errors="coerce")
            median = round(float(numbers.median()), 2)
            for r in rows:
                if pd.isna(out.at[r, col]):
                    out.at[r, col] = median
                    audit.append({"row": r, "column": col, "action": action,
                                  "before": "missing", "after": str(median)})
        if action == "fill_with_most_common" and col:
            common = out[col].mode().iloc[0]
            for r in rows:
                if pd.isna(out.at[r, col]):
                    out.at[r, col] = common
                    audit.append({"row": r, "column": col, "action": action,
                                  "before": "missing", "after": str(common)})

    out = out.drop(index=list(rows_to_drop))
    for col in out.columns:  # turn number-like text columns back into numbers
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() == out[col].notna().sum():
                out[col] = converted
    return out, pd.DataFrame(audit)


def quality_score(df: pd.DataFrame, issues: pd.DataFrame) -> float:
    """0–100: share of cells without a detected problem."""
    total_cells = df.size
    if total_cells == 0:
        return 100.0
    bad = issues[issues["type"] != "anomaly"]
    bad_cells = len(bad[bad["column"].notna()]) + len(bad[bad["column"].isna()]) * df.shape[1]
    return round(max(0.0, 100 * (1 - bad_cells / total_cells)), 1)