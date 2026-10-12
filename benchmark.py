import pandas as pd


def _keys(df, error_type):
    """Duplicates match by row; other errors match by (row, column)."""
    if error_type == "duplicate":
        return set(df["row"])
    return set(zip(df["row"], df["column"]))


def evaluate(issues: pd.DataFrame, truth: pd.DataFrame):
    """Compare detected issues with the known injected errors."""
    results = []
    total_tp = total_det = total_truth = 0

    for t in truth["type"].unique():
        truth_keys = _keys(truth[truth["type"] == t], t)
        det_keys = _keys(issues[issues["type"] == t], t)
        tp = len(truth_keys & det_keys)
        precision = tp / len(det_keys) if det_keys else 0.0
        recall = tp / len(truth_keys) if truth_keys else 0.0
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        results.append({"error_type": t, "injected": len(truth_keys),
                        "detected": len(det_keys), "correct": tp,
                        "precision": round(precision, 3),
                        "recall": round(recall, 3), "f1": round(f1, 3)})
        total_tp += tp
        total_det += len(det_keys)
        total_truth += len(truth_keys)

    # Machine learning check: how many flagged rows really had an error?
    ml = issues[issues["type"] == "anomaly"]
    bad_rows = set(truth["row"])
    ml_hits = int(ml["row"].isin(bad_rows).sum())

    summary = {
        "precision": round(total_tp / total_det, 3) if total_det else 0.0,
        "recall": round(total_tp / total_truth, 3) if total_truth else 0.0,
        "ml_flagged_rows": len(ml),
        "ml_rows_with_real_errors": ml_hits,
    }
    return pd.DataFrame(results), summary