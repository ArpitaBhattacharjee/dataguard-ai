import pandas as pd
from sklearn.ensemble import IsolationForest


def _is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _text_columns(df):
    return [c for c in df.columns if _is_text(df[c])]


def _issue(row, column, kind, severity, detail, value=None):
    return {"row": row, "column": column, "type": kind,
            "severity": severity, "detail": detail, "value": value}


def _numeric_columns(df):
    """Numeric columns, including text columns that are mostly numbers."""
    out = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s
        elif _is_text(s) and s.notna().sum() > 0:
            parsed = pd.to_numeric(s, errors="coerce")
            if parsed.notna().sum() / s.notna().sum() >= 0.8:
                out[col] = parsed
    return out


def check_missing(df):
    issues = []
    for col in df.columns:
        for idx in df.index[df[col].isna()]:
            issues.append(_issue(idx, col, "missing", "medium", "Value is missing"))
    return issues


def check_duplicates(df):
    return [_issue(i, None, "duplicate", "high", "Exact duplicate row")
            for i in df.index[df.duplicated()]]


def check_wrong_type(df):
    issues = []
    for col in _text_columns(df):
        s = df[col].dropna()
        if s.empty:
            continue
        parsed = pd.to_numeric(s, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            for idx in s.index[parsed.isna()]:
                issues.append(_issue(idx, col, "wrong_type", "high",
                                     "Expected a number", s[idx]))
    return issues


def check_outliers(df, num_cols):
    issues = []
    for col, s in num_cols.items():
        s = s.dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        for idx in s.index[(s < lo) | (s > hi)]:
            issues.append(_issue(idx, col, "outlier", "medium",
                                 f"Outside expected range {lo:.2f} to {hi:.2f}", s[idx]))
    return issues


def check_negatives(df, num_cols):
    issues = []
    for col, s in num_cols.items():
        s = s.dropna()
        if len(s) < 10:
            continue
        neg = s < 0
        if 0 < neg.mean() <= 0.05:  # column is almost always positive
            for idx in s.index[neg]:
                issues.append(_issue(idx, col, "negative", "high",
                                     "Negative value in a positive column", s[idx]))
    return issues


def check_categories(df, num_cols):
    issues = []
    for col in _text_columns(df):
        if col in num_cols:
            continue
        s = df[col].dropna().astype(str)
        if s.empty or s.nunique() > 50:  # skip free text
            continue
        key = s.str.strip().str.lower()
        for _, group in s.groupby(key):
            forms = group.value_counts()
            if len(forms) > 1:
                main = forms.index[0]
                for idx in group.index[group != main]:
                    issues.append(_issue(idx, col, "inconsistent", "low",
                                         f"Should probably be '{main}'", s[idx]))
    return issues


def check_dates(df):
    issues = []
    for col in df.columns:
        if "date" not in col.lower():
            continue
        s = df[col].dropna()
        parsed = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
        for idx in s.index[parsed.isna()]:
            issues.append(_issue(idx, col, "invalid_date", "high",
                                 "Not a valid date", s[idx]))
    return issues


def check_anomalies(df, num_cols, contamination=0.01):
    """Machine learning: Isolation Forest flags rows unlike the others."""
    num = pd.DataFrame(num_cols)
    if num.empty or len(num) < 50:
        return []
    X = num.fillna(num.median())
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X)
    scores = -model.score_samples(X)
    return [_issue(idx, None, "anomaly", "low", "Row looks unusual (ML)",
                   round(float(scores[i]), 3))
            for i, idx in enumerate(X.index) if labels[i] == -1]


def run_all_checks(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = _numeric_columns(df)
    issues = (check_missing(df) + check_duplicates(df) + check_wrong_type(df)
              + check_outliers(df, num_cols) + check_negatives(df, num_cols)
              + check_categories(df, num_cols) + check_dates(df)
              + check_anomalies(df, num_cols))
    cols = ["row", "column", "type", "severity", "detail", "value"]
    return pd.DataFrame(issues, columns=cols)